"""
llm_stage_judge.py
==================

Token-logprob-based DMIS stage classifier.

The model is asked to emit ONE digit (1-6) corresponding to the DMIS
stage that best describes a given turn / speaker history. Top-token
logprobs at that response position are extracted, restricted to the
six digits, then softmaxed into a 6-element probability distribution
over stages. This gives us:

  - argmax_idx     : the predicted stage as an int 0..5
  - argmax_stage   : the stage name
  - distribution   : 6-element list, sums to ~1
  - raw_logprobs   : per-stage logprob (for debugging)

Mapping:
  1 -> Denial
  2 -> Defense
  3 -> Minimization
  4 -> Acceptance
  5 -> Adaptation
  6 -> Integration

This drop-in replaces the old DMIS Likert simulator
(measure_dmis_for_speaker / measure_per_speaker_distributions) while
keeping the same return shape so all downstream metrics (trajectory
AUC, signed Wasserstein-1, ambiguous filter, etc.) work unchanged.

Compatibility with existing OpenRouter pipeline
-----------------------------------------------
Uses the OpenAI Python SDK with `logprobs=True, top_logprobs=20`. The
provider must support logprobs (OpenAI, Llama-via-OpenRouter, Mistral,
Gemini are OK; Anthropic models do NOT currently expose logprobs and
will raise when this code is run against them).
"""
from __future__ import annotations

import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI


STAGE_ORDER = ["Denial", "Defense", "Minimization",
               "Acceptance", "Adaptation", "Integration"]
STAGE_TO_DIGIT = {s: str(i + 1) for i, s in enumerate(STAGE_ORDER)}
DIGIT_TO_STAGE = {v: k for k, v in STAGE_TO_DIGIT.items()}
DIGIT_TO_IDX = {str(i + 1): i for i in range(6)}


STAGE_DEFINITIONS = {
    "Denial": (
        "Speakers fail to register cultural difference as a meaningful "
        "category and treat the disagreement as if no cultural dimension "
        "were in play. Markers: 'live and let live', 'people figure it "
        "out', 'doesn't matter to me', subject changes, oblivious "
        "reframings; speaker is disengaged from the cultural question."
    ),
    "Defense": (
        "Polarized us-versus-them frame with simplistic, evaluative "
        "stereotypes. Markers: 'actually...', 'in our way', flat "
        "contradictions, 'naive', 'backward', occasional ALL-CAPS, "
        "sharp evaluative language about the other group."
    ),
    "Minimization": (
        "Subsumes cultural difference under universal sameness; treats "
        "own cultural patterns as the neutral baseline. Markers: 'deep "
        "down we are all the same', 'people are pretty much motivated "
        "by the same things', 'small world after all', 'at the end of "
        "the day we both know'; warm tone but engaged."
    ),
    "Acceptance": (
        "Accepts cultural difference as real, important, and valid "
        "without yet acting on it. Curious questions about the other "
        "side's frame, explicit acknowledgement of differing frames as "
        "legitimate."
    ),
    "Adaptation": (
        "Shifts frame of reference to engage with the other culture's "
        "perspective. Articulates the other side's view from inside "
        "that frame and adjusts behavior accordingly."
    ),
    "Integration": (
        "Fluidly draws on multiple cultural frames as part of identity. "
        "Negotiates difference without anchoring in a single frame."
    ),
}


JUDGE_SYSTEM = """You are an expert in Bennett's Developmental Model of Intercultural Sensitivity (DMIS). For a given conversation history, classify which DMIS stage the target speaker exhibits at the marked turn.

DMIS stages:
  1 = Denial        - speaker fails to register cultural difference; "live and let live", "doesn't matter to me", subject changes, oblivious; disengaged from the cultural question.
  2 = Defense       - polarized us-versus-them frame; "actually...", "in our way", flat contradictions, "naive", "backward", evaluative language about the other group.
  3 = Minimization  - subsumes difference under universal sameness; "deep down we are all the same", "people are pretty much motivated by the same things", warm but engaged.
  4 = Acceptance    - accepts cultural difference as real and valid; curious questions, explicit acknowledgement of differing frames as legitimate.
  5 = Adaptation    - shifts frame of reference; articulates the other side's view from inside that frame; adjusts behavior accordingly.
  6 = Integration   - fluidly draws on multiple cultural frames; negotiates difference without anchoring in any single frame.

Respond with ONLY a single digit (1, 2, 3, 4, 5, or 6). No prose, no JSON, no explanation. Just one digit.
"""


def _build_user(history_text: str, marked_speaker: str,
                marked_turn_text: Optional[str] = None) -> str:
    blocks = [
        "Conversation history (the marked speaker's perspective):",
        history_text.strip(),
    ]
    if marked_turn_text:
        blocks += [
            "",
            f"The MARKED TURN is by {marked_speaker}:",
            f'"""{marked_turn_text.strip()}"""',
        ]
    blocks += [
        "",
        f"Classify {marked_speaker}'s DMIS stage at this point. "
        "Respond with ONLY one digit (1-6).",
    ]
    return "\n".join(blocks)


# Lazy-init OpenAI client — caller can override via env vars.
_client: Optional[OpenAI] = None


def _get_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def _logprob_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_retries: int = 3,
) -> tuple[Optional[list], Optional[str]]:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                logprobs=True,
                top_logprobs=20,
                max_tokens=2,
                temperature=0.0,
                extra_headers={"HTTP-Referer": "https://localhost/cc",
                               "X-Title": "CC Stage Judge"},
            )
            content = resp.choices[0].logprobs.content
            if content and len(content) > 0:
                return content[0].top_logprobs, None
            last_err = "empty logprob content"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 * (attempt + 1))
    return None, last_err


def _logprobs_to_distribution(top_logprobs) -> tuple[list[float], dict[str, float]]:
    """Restrict to digits 1-6, softmax to a 6-vector. Missing digits get
    a very small floor logprob so the distribution stays well-defined."""
    FLOOR = -50.0
    per_digit: dict[str, float] = {str(i): FLOOR for i in range(1, 7)}
    raw_per_digit: dict[str, float] = {}
    for tok in top_logprobs:
        # Trim whitespace; some providers prepend a space or newline.
        t = (tok.token if hasattr(tok, "token") else tok["token"]).strip()
        lp = tok.logprob if hasattr(tok, "logprob") else tok["logprob"]
        if t in per_digit and lp > per_digit[t]:
            per_digit[t] = float(lp)
            raw_per_digit[t] = float(lp)
    log_max = max(per_digit.values())
    exps = [math.exp(per_digit[str(i + 1)] - log_max) for i in range(6)]
    z = sum(exps)
    dist = [e / z for e in exps]
    return dist, raw_per_digit


def measure_stage_via_logprob(
    sim_model: str,
    history_text: str,
    api_key: str,
    marked_speaker: str = "the speaker",
    marked_turn_text: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict:
    """Single-call DMIS stage classifier with calibrated distribution.

    Returns
    -------
    {
        "argmax_idx":   int 0..5 or None on failure,
        "argmax_stage": str or None,
        "distribution": list[float] length 6 or None,
        "max_prob":     float or None,
        "entropy":      float or None,
        "raw_logprobs": dict[str, float],
        "error":        str or None,
    }
    """
    client = _get_client(api_key, base_url)
    user = _build_user(history_text, marked_speaker, marked_turn_text)
    top_logprobs, err = _logprob_call(client, sim_model, JUDGE_SYSTEM, user)
    if err or top_logprobs is None:
        return {
            "argmax_idx": None, "argmax_stage": None, "distribution": None,
            "max_prob": None, "entropy": None, "raw_logprobs": None,
            "error": err or "no logprobs returned",
        }
    dist, raw = _logprobs_to_distribution(top_logprobs)
    idx = max(range(6), key=lambda i: dist[i])
    max_p = float(dist[idx])
    entropy = -sum(p * math.log(p) for p in dist if p > 0)
    return {
        "argmax_idx": idx,
        "argmax_stage": STAGE_ORDER[idx],
        "distribution": dist,
        "max_prob": max_p,
        "entropy": entropy,
        "raw_logprobs": raw,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Drop-in replacement for measure_per_speaker_distributions
# ---------------------------------------------------------------------------

def measure_per_speaker_logprob(
    histories: dict[str, list[dict]],
    agents_meta: dict[str, dict],
    sim_model: str,
    api_key: str,
    max_workers: int = 4,
    base_url: str = "https://openrouter.ai/api/v1",
) -> Optional[dict[str, dict]]:
    """Same input/output contract as
    evaluate_mediation_effectiveness.measure_per_speaker_distributions
    but uses the logprob-based LLM judge instead of the 43-item Likert
    DMIS simulator. Returns None if any speaker measurement fails."""

    # Local import to avoid circular import at module load time.
    from evaluate_mediation_effectiveness import histories_to_plaintext

    profiles: dict[str, dict] = {}

    def _go(aid: str):
        history_text = histories_to_plaintext(histories[aid])
        speaker_name = (agents_meta.get(aid) or {}).get("name") or aid
        res = measure_stage_via_logprob(
            sim_model, history_text, api_key,
            marked_speaker=speaker_name,
            base_url=base_url,
        )
        return aid, res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_go, aid) for aid in histories]
        for fut in as_completed(futs):
            aid, res = fut.result()
            profiles[aid] = {
                # NEW logprob-method fields
                "distribution": res.get("distribution"),
                "argmax_idx": res.get("argmax_idx"),
                "argmax_stage": res.get("argmax_stage"),
                "max_prob": res.get("max_prob"),
                "entropy": res.get("entropy"),
                "raw_logprobs": res.get("raw_logprobs"),
                # Legacy-compat fields (set to None - downstream W1 uses
                # `distribution` only; stage_means/responses are no longer
                # produced and any code reading them must tolerate None).
                "stage_means": None,
                "responses": None,
                "raw_response": None,
                "error": res.get("error"),
            }

    if any(p["distribution"] is None for p in profiles.values()):
        return None
    return profiles
