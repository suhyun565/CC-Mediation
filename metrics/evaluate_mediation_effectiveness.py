"""
evaluate_mediation_effectiveness.py
===================================

Evaluate the effect of LLM-generated mediator utterances on a dialogue's
DMIS stage trajectory.

For each dialogue and each (mediator model x intervention path), we:
  1. Predict the intervention turn (model picks; or use GT).
  2. Optionally provide the stage-specific mediation definition.
  3. Generate a mediator utterance, INJECT it into both speakers'
     histories as a user-role message, and then continue the dialogue
     from intervention_turn + 1 up to n_turns. Each speaker sees the
     mediator's utterance and responds to it as a regular conversational
     contribution.
  4. Measure each speaker's DMIS Likert distribution at:
       - Pre        : conflict_turn      (right after the conflict agent
                                          has surfaced the goal_2
                                          disagreement and enacted the
                                          ethnocentric pattern; the
                                          mediator has not yet spoken)
       - Post       : conflict_turn + 3  (after the mediator has spoken and
                                          each speaker has produced one new
                                          turn)
       - Trajectory : conflict_turn + 4, +5, ... up to n_turns
                      (one measurement per remaining turn)
  5. Score the path with both-success diagnostics:
       - stage_shift_per_agent      : post_argmax_idx - pre_argmax_idx
       - both_acceptance_at_post    : both agents' post argmax >= 3
       - first_acceptance_turn      : earliest trajectory turn where both
                                      agents simultaneously have argmax >= 3
       - turns_to_acceptance        : first_acceptance_turn - intervention_turn
       - regressed_at_some_point    : did either agent's argmax drop below
                                      its post value at any later turn?
       - max_regression             : largest single-agent step-down across
                                      trajectory
  6. Compute the Paige-weighted signed Wasserstein-1 distance and the
     expected_position_shift on each agent's distributions (effect size).
  7. Send (mediation_definition, mediator_utterance) to the judge model
     for a 5-point definition adherence rating.

The four paths run per model:

    predicted_path     : predicted turn, no definition
    gt_path            : GT turn,        no definition
    predicted_def_path : predicted turn, with definition
    gt_def_path        : GT turn,        with definition

Original-dialogue baseline (mediator-free)
------------------------------------------

In addition to the four mediator paths, an "original_dialogue" row uses
the original (mediator-free) continuation in the source JSON file for the
same Pre/Post/trajectory measurements. Because this row is independent of
the mediator model, it is attached ONLY to the openai/gpt-4o-mini summary
block; for every other mediator model the field is "-" in the table so it
is not double-counted.

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python evaluate_mediation_effectiveness.py
    python evaluate_mediation_effectiveness.py --limit 5
    python evaluate_mediation_effectiveness.py --models openai/gpt-4o-mini
    python evaluate_mediation_effectiveness.py --no-judge
"""

from __future__ import annotations


# --- release bootstrap: make shared/ and metrics/ importable ---
import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parent.parent
for _sub in ("shared", "metrics"):
    _p = str(_RELEASE_ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---



import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from dmis_items import STAGE_ORDER  # canonical 6-stage list
from dmis_distribution import (
    combine_distributions,
    compare_distributions,
    expected_position,
    STAGE_POSITIONS,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from pathlib import Path as _P
_REPO = _P(__file__).resolve().parent.parent
DATA_DIR = str(_REPO / "data" / "CC_dialogues")
DEFAULT_UTILS_PATH = str(_REPO / "shared" / "utils.py")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]

DEFAULT_SIM_MODEL = "openai/gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

CONTINUATION_TEMPERATURE = 0.8
CONTINUATION_MAX_TOKENS = 300

# The mediator-free baseline (original dialogue from the source JSON) is
# evaluated only once per dialogue; we attach the result to this single
# model row in the summary so it does not get repeated four times.
ORIGINAL_BASELINE_MODEL = "openai/gpt-4o-mini"

# Acceptance threshold on the stage index grid:
#   0 Denial, 1 Defense, 2 Minimization,
#   3 Acceptance, 4 Adaptation, 5 Integration
# both-success acceptance = both agents' argmax >= 3.
ACCEPTANCE_IDX_THRESHOLD = 3


# ---------- Stage 1: turn prediction ----------------------------------------

# ---------- Streaming conflict-turn detection ------------------------------

# Conflict-turn detection is now run turn-by-turn: at each turn we ask
# the model whether the marked turn has surfaced a cultural-value conflict
# with an ethnocentric DMIS pattern. The first 'yes' is taken as the
# predicted conflict_turn; the mediator inserts immediately after that
# turn (no separate intervention_turn).

CONFLICT_DETECT_SYSTEM = """You are detecting when a cultural-value conflict has surfaced in a two-speaker dialogue.

You will be shown the dialogue prefix up to and including a MARKED TURN. Decide: at the marked turn, has the speaker surfaced the cultural-value disagreement (goal_2) AND enacted one of the three ethnocentric DMIS patterns?

DMIS ethnocentric patterns to look for:
  - Denial: speaker disengages from cultural difference; markers like "live and let live", "people figure it out", "doesn't matter to me", topic changes, oblivious reframings.
  - Defense: polarized us-vs-them frame; markers like "actually...", "in our way", flat contradictions ("No, that's not how it works"), "naive", "backward", evaluative language about the other group.
  - Minimization: subsumes cultural difference under universal sameness; markers like "deep down we are all the same", "people are pretty much motivated by the same things", "small world after all".

Respond with ONLY one character:
  Y  - yes, the marked turn IS the conflict turn (mediator should intervene right after this turn)
  N  - no, the marked turn has not yet surfaced the cultural conflict, or does not exhibit an ethnocentric pattern

No prose, no JSON, no explanation. Just one character: Y or N.
"""


CONFLICT_DETECT_USER_TEMPLATE = """Dialogue prefix (up to and including the MARKED TURN):

{dialogue}

The MARKED TURN is Turn {target_turn} by {target_speaker}:
\"\"\"{target_message}\"\"\"

Has the conflict turn occurred? Respond with only Y or N."""


# ---------- Stage 2: mediation generation -----------------------------------

MEDIATION_SYSTEM_PROMPT = """You are a skilled third-party mediator joining a conversation between two people from different cultural backgrounds.

You will be shown the dialogue up to the moment where you decide to step in. Your task is to produce ONE mediator utterance that you would speak next, addressed to both participants.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{"mediation": "<your mediator utterance, 1-4 sentences>"}
"""

MEDIATION_USER_PROMPT_TEMPLATE = """Dialogue so far (you are about to step in right after Turn {turn}):

{dialogue}

Write your mediator utterance now. Respond with only:
{{"mediation": "<your utterance>"}}"""


MEDIATION_DEF_SYSTEM_PROMPT = """You are a skilled third-party mediator joining a conversation between two people from different cultural backgrounds.

You will be shown:
  - the dialogue up to the moment where you decide to step in
  - a description of the cultural-difficulty pattern the speakers are currently exhibiting
  - the corresponding mediation move you should make to help them out of that pattern

Your task is to produce ONE mediator utterance addressed to both participants that actually performs the indicated mediation move. Be concrete: include specific questions, framings, or invitations rather than generic acknowledgement.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{"mediation": "<your mediator utterance, 1-4 sentences>"}
"""

MEDIATION_DEF_USER_PROMPT_TEMPLATE = """Dialogue so far (you are about to step in right after Turn {turn}):

{dialogue}

Stage: {stage}

Pattern in the dialogue (description):
"{description}"

Mediation move you should perform:
"{mediation}"

Write your mediator utterance now. Respond with only:
{{"mediation": "<your utterance>"}}"""


# ---------- DMIS Likert administration --------------------------------------

# NOTE: The 43-item DMIS Likert simulator and its prompts have been
# removed. Stage diagnosis now goes through the logprob-based LLM judge
# in llm_stage_judge.py (see measure_per_speaker_distributions below).

# ---------- LLM judge for definition adherence ------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of intercultural mediation.

You will be given:
  1. A description of a cultural-difficulty pattern that two speakers are exhibiting in a conversation.
  2. A mediation move definition specifying what a third-party mediator should do to help them out of that pattern.
  3. A single mediator utterance produced by a model.

Your task is to rate, on a 5-point scale, how well the mediator utterance is semantically consistent with the mediation move definition (does it actually do what the definition asks?). Focus on whether the utterance performs the right kind of move, not on stylistic polish or cultural-context detail.

Scoring rubric (5-point Likert):
  5 = excellent: the utterance clearly and concretely performs the mediation move; it would visibly help the speakers exit the described pattern.
  4 = good: the utterance performs the mediation move, with minor weaknesses in concreteness or framing.
  3 = partial: the utterance is in the right direction but is generic, hedged, or only partially performs the move.
  2 = poor: the utterance addresses the conversation but does not perform the indicated mediation move; it may even reinforce the described pattern.
  1 = wrong: the utterance does the opposite of the move, or is irrelevant.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{"score": <integer 1-5>, "reasoning": "<one or two sentences>"}
"""

JUDGE_USER_PROMPT_TEMPLATE = """Stage: {stage}

Pattern description:
"{description}"

Mediation move definition:
"{mediation}"

Mediator utterance to evaluate:
"{utterance}"

Rate the semantic consistency between the utterance and the mediation move definition.
Respond with only:
{{"score": <int 1-5>, "reasoning": "<short>"}}"""


# ---------------------------------------------------------------------------
# CONFLICT_TABLE loader (3-stage flat structure)
# ---------------------------------------------------------------------------

def load_conflict_table(utils_path: str) -> dict:
    p = Path(utils_path)
    if not p.exists():
        raise FileNotFoundError(f"utils.py not found at: {utils_path}")
    spec = importlib.util.spec_from_file_location("cc_utils", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build module spec for {utils_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "CONFLICT_TABLE"):
        raise AttributeError(f"{utils_path} has no CONFLICT_TABLE")
    return mod.CONFLICT_TABLE


def lookup_mediation_components(
    stage: str, table: Optional[dict],
) -> Optional[dict]:
    """Look up the (description, mediation) pair for a stage in the new
    3-stage flat CONFLICT_TABLE."""
    if table is None or not stage:
        return None
    entry = table.get(stage)
    if not isinstance(entry, dict):
        return None
    description = (entry.get("description") or "").strip()
    mediation = (entry.get("mediation") or "").strip()
    if not description or not mediation:
        return None
    return {"description": description, "mediation": mediation, "stage": stage}


# ---------------------------------------------------------------------------
# Speaker-history serialization (mirrors generate_dialogue.py)
# ---------------------------------------------------------------------------

def build_speaker_system_prompt(agent: dict) -> str:
    """Char-for-char copy of generate_dialogue.py:build_system_prompt so
    measurement and continuation see exactly the prompt the original
    speaker saw."""
    goals = agent.get("goals", {}) or {}
    rules = agent.get("rules", []) or []

    lines = [
        f"TASK: {agent.get('task', '')}",
        f"SCENARIO: {agent.get('scenario', '')}",
        f"YOUR BACKGROUND: {agent.get('background', '')}",
        f"YOUR CULTURAL VALUE (your stable underlying stance \u2014 this NEVER changes "
        f"during this conversation): {agent.get('cultural_value', '')}",
        "",
        "GOALS:",
        f"  goal_1: {goals.get('goal_1', '')}",
        f"  goal_2: {goals.get('goal_2', '')}",
        "",
        "STANCE STABILITY \u2014 APPLIES THROUGHOUT THE WHOLE CONVERSATION:",
        "  \u2022 Your cultural value is your stable belief. It does NOT shift mid-conversation.",
        "  \u2022 You may change HOW you engage (push back, retreat, withdraw, deflect) \u2014 "
        "but you do NOT change your underlying mind.",
        "  \u2022 Capitulation phrases are FORBIDDEN. Never produce: \"You're right, I was "
        "wrong\", \"I see your point now\", \"I didn't realize that\", \"You've "
        "convinced me\", \"I now think...\", \"Good point \u2014 I'll change my view.\"",
        "  \u2022 If pressure builds and you must yield, yield from the EXCHANGE (drop the "
        "topic, change subject, go quiet) \u2014 never from your position.",
        "",
        "OUTPUT \u2014 DIALOGUE ONLY:",
        "  \u2022 Output ONLY what your character SAYS OUT LOUD. Speech only.",
        "  \u2022 NO stage directions, NO action descriptions, NO narration.",
        "  \u2022 NO scene-setting prose.",
        "",
        "RULES:",
    ]
    for r in rules:
        lines.append(f"  - {r}")
    return "\n".join(lines)


def _agent_for_original_turn(turn: int) -> str:
    """generate_dialogue.py uses turn parity (odd -> agent_1, even -> agent_2)."""
    return "agent_1" if turn % 2 == 1 else "agent_2"


def identify_conflict_agent(
    data: dict,
    conflict_turn: int,
) -> Optional[str]:
    """Return the agent_id of the conflict agent for this dialogue.

    Resolution order:
      1. Top-level "conflict_agent" field in the source JSON
         (set by build_agent_prompts.py for asymmetric setups).
      2. The agent_prompts entry whose ``is_conflict_agent`` flag is True.
      3. Fallback: the agent who speaks at conflict_turn (parity rule),
         which matches build_agent_prompts.py's initiator logic.
    """
    explicit = data.get("conflict_agent")
    if isinstance(explicit, str) and explicit:
        return explicit
    agent_prompts = data.get("agent_prompts") or {}
    for aid, info in agent_prompts.items():
        if info.get("is_conflict_agent"):
            return aid
    return _agent_for_original_turn(conflict_turn)


def build_initial_histories(
    dialogue_entries: list[dict],
    agent_prompts: dict,
    up_to_turn: int,
) -> dict[str, list[dict]]:
    """Replay the original dialogue into per-agent histories up to and
    including ``up_to_turn``. Each agent's own utterances are tagged
    'assistant' in their own history and 'user' in the partner's history.

    The conflict_instruction is INTENTIONALLY NOT inserted into history.
    In the asymmetric one-shot setup, conflict_instruction is delivered
    to the conflict agent as an ephemeral user prompt at conflict_turn
    only and is not stored in the persistent history. To make
    measurement and continuation consistent with generation, history
    reconstruction here mirrors that and contains base prompt + dialogue
    turns only.
    """
    agent_ids = list(agent_prompts.keys())
    histories: dict[str, list[dict]] = {
        aid: [{"role": "system",
               "content": build_speaker_system_prompt(agent_prompts[aid])}]
        for aid in agent_ids
    }

    for entry in dialogue_entries:
        turn = entry["turn"]
        if turn > up_to_turn:
            break
        speaker = entry["agent"]
        if speaker not in histories:
            histories[speaker] = [{
                "role": "system",
                "content": build_speaker_system_prompt(
                    agent_prompts.get(speaker, {})
                ),
            }]
        message = entry["message"]
        for aid in histories:
            role = "assistant" if aid == speaker else "user"
            histories[aid].append({"role": role, "content": message})
    return histories


def histories_to_plaintext(history: list[dict]) -> str:
    """Serialize a single speaker's chat history (system/user/assistant
    messages) into the plain-text role-tagged form that the DMIS rater
    sees. Mirrors evaluate_dataset_quality.py."""
    blocks: list[str] = []
    for msg in history:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_dialogue(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _agent_name_from_background(background: str, fallback: str) -> str:
    bg = (background or "").strip()
    if ":" in bg:
        return bg.split(":", 1)[0].strip() or fallback
    return fallback


def agent_metadata(data: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for aid, info in data.get("agent_prompts", {}).items():
        bg = info.get("background", "") or ""
        out[aid] = {
            "name": _agent_name_from_background(bg, aid),
            "background": bg.strip(),
            "cultural_value": (info.get("cultural_value") or "").strip(),
        }
    return out


def format_dialogue_for_turn_picker(
    dialogue_entries: list[dict],
    agent_prompts: dict,
) -> str:
    """Compact representation of the original dialogue for the turn-pick
    and mediator-generation prompts. Mediator-LLM-facing only; not used
    for DMIS measurement."""
    lines: list[str] = []
    for entry in dialogue_entries:
        bg = (
            agent_prompts.get(entry["agent"], {}).get("background", "")
            if isinstance(agent_prompts, dict) else ""
        )
        speaker = _agent_name_from_background(bg, entry["agent"])
        msg = (entry.get("message") or "").strip()
        lines.append(f"Turn {entry['turn']} - {speaker}:\n{msg}")
    return "\n\n".join(lines)


def first_intervention_turn(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, bool):
                continue
            if isinstance(x, (int, float)):
                return int(x)
            if isinstance(x, str):
                m = re.search(r"\d+", x)
                if m:
                    return int(m.group(0))
        return None
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _normalize_smart_punct(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))


def parse_json_object(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    cleaned = _strip_fences(text)
    for candidate in (cleaned, _normalize_smart_punct(cleaned)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    return None


def parse_turn(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    obj = parse_json_object(text)
    if obj and "intervention_turn" in obj:
        v = obj["intervention_turn"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            if m:
                return int(m.group(0))
    cleaned = _strip_fences(text)
    m = re.search(r'"intervention_turn"\s*:\s*(\d+)', cleaned)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,3})\b", cleaned)
    if m:
        return int(m.group(1))
    return None


def detect_conflict_turn_streaming(
    sim_model: str,
    dialogue_entries: list[dict],
    agent_prompts: dict,
    api_key: str,
    n_turns: int,
    max_workers: int = 1,
) -> tuple[Optional[int], list[dict]]:
    """Walk turn-by-turn through the dialogue prefix and ask the model
    whether the marked turn has surfaced a cultural-value conflict
    (i.e. exhibits an ethnocentric DMIS pattern: Denial / Defense /
    Minimization).

    Stops at the FIRST turn that gets a 'Y' answer and returns it as the
    predicted conflict turn. Returns ``(None, per_turn_log)`` if no turn
    is flagged. ``per_turn_log`` is always returned for debugging.

    Note: this is intrinsically sequential (we stop on first 'yes'), so
    ``max_workers`` is unused. It is kept for signature compatibility.
    """
    log: list[dict] = []
    by_turn = {t["turn"]: t for t in dialogue_entries}
    for tn in sorted(by_turn.keys()):
        if tn < 1 or tn > n_turns:
            continue
        turn = by_turn[tn]
        speaker = turn.get("speaker") or turn.get("agent", "")
        if not speaker and turn.get("agent"):
            bg = (agent_prompts.get(turn["agent"]) or {}).get(
                "background", ""
            ).strip()
            speaker = bg.split(":", 1)[0].strip() if ":" in bg else (
                bg.split()[0] if bg else turn["agent"]
            )
        prefix = [e for e in dialogue_entries if e["turn"] <= tn]
        dialog_text = format_dialogue_for_turn_picker(prefix, agent_prompts)
        user = CONFLICT_DETECT_USER_TEMPLATE.format(
            dialogue=dialog_text,
            target_turn=tn,
            target_speaker=speaker,
            target_message=(turn.get("message") or "").strip(),
        )
        raw, err = call_simple(
            sim_model, CONFLICT_DETECT_SYSTEM, user, api_key,
            max_tokens=5, temperature=0.0,
        )
        decision = None
        if raw:
            t = raw.strip().upper()
            if t.startswith("Y"):
                decision = True
            elif t.startswith("N"):
                decision = False
        log.append({"turn": tn, "decision": decision, "raw": raw, "error": err})
        if decision is True:
            return tn, log
    return None, log


def parse_mediation(text: Optional[str]) -> Optional[str]:
    obj = parse_json_object(text)
    if obj and isinstance(obj.get("mediation"), str):
        v = obj["mediation"].strip()
        if v:
            return v
    return None


def parse_judge_score(
    text: Optional[str],
) -> tuple[Optional[int], Optional[str]]:
    """Returns (score, reasoning)."""
    obj = parse_json_object(text)
    if not obj:
        return None, None
    score = obj.get("score")
    reasoning = obj.get("reasoning")
    if isinstance(score, bool):
        score = None
    elif isinstance(score, (int, float)):
        score = int(score)
        if not 1 <= score <= 5:
            score = None
    elif isinstance(score, str):
        m = re.search(r"\d", score)
        score = int(m.group(0)) if m else None
        if score is not None and not 1 <= score <= 5:
            score = None
    else:
        score = None
    if not isinstance(reasoning, str):
        reasoning = None
    return score, reasoning


def _coerce_int_in_range(v, lo: int, hi: int) -> Optional[int]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            n = int(round(float(v)))
            return n if lo <= n <= hi else None
        if isinstance(v, str):
            m = re.search(r"-?\d+", v)
            if not m:
                return None
            n = int(m.group(0))
            return n if lo <= n <= hi else None
    except (ValueError, TypeError):
        return None
    return None


def parse_likert_responses(
    text: Optional[str], n_items: int,
) -> Optional[list[Optional[int]]]:
    obj = parse_json_object(text)
    if not obj:
        return None
    out: list[Optional[int]] = []
    for i in range(1, n_items + 1):
        out.append(_coerce_int_in_range(obj.get(f"item_{i}"), *LIKERT_RANGE))
    return out


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")


def call_ollama(
    model: str,
    messages: list[dict],
    max_tokens: int = 200,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> tuple[Optional[str], Optional[str]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        },
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"], None
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, last_err


def call_openrouter(
    model: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> tuple[Optional[str], Optional[str]]:
    if model.startswith("ollama/"):
        return call_ollama(model[len("ollama/"):], messages,
                           max_tokens=max_tokens, temperature=temperature,
                           max_retries=max_retries)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/cc-eval",
        "X-Title": "CC Mediation Effectiveness",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                last_err = "429 rate limit"
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"], None
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, last_err


def call_simple(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> tuple[Optional[str], Optional[str]]:
    return call_openrouter(
        model,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        api_key, max_tokens=max_tokens, temperature=temperature,
    )


# ---------------------------------------------------------------------------
# DMIS scoring helpers
# ---------------------------------------------------------------------------

def _argmax_idx(dist: list[float]) -> int:
    return max(range(len(dist)), key=lambda i: dist[i])


def measure_per_speaker_distributions(
    histories: dict[str, list[dict]],
    agents_meta: dict[str, dict],
    sim_model: str,
    api_key: str,
    max_workers: int,
) -> Optional[dict[str, dict]]:
    """Per-speaker DMIS-stage profile via the logprob-based LLM judge.

    The legacy 43-item Likert simulator (measure_dmis_for_speaker) had
    very low Denial accuracy (~18%) and produced flat distributions on
    62% of continuation turns. This implementation replaces it with a
    single logprob call that returns a calibrated 6-stage distribution
    over the digit tokens 1..6 (one per stage). Same return shape as
    before so all downstream metrics (trajectory_auc, signed_w1, ...)
    keep working.
    """
    # Local import keeps the logprob module optional and avoids a
    # circular import: llm_stage_judge imports histories_to_plaintext
    # from this module.
    from llm_stage_judge import measure_stage_via_logprob

    profiles: dict[str, dict] = {}

    def _go(aid: str):
        history_text = histories_to_plaintext(histories[aid])
        speaker = (agents_meta.get(aid) or {}).get("name") or aid
        res = measure_stage_via_logprob(
            sim_model, history_text, api_key,
            marked_speaker=speaker,
        )
        return aid, res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_go, aid) for aid in histories]
        for fut in as_completed(futs):
            aid, res = fut.result()
            profiles[aid] = {
                "distribution": res.get("distribution"),
                "argmax_idx": res.get("argmax_idx"),
                "argmax_stage": res.get("argmax_stage"),
                "max_prob": res.get("max_prob"),
                "entropy": res.get("entropy"),
                "raw_logprobs": res.get("raw_logprobs"),
                # Legacy-compat keys retained as None — the 43-item
                # Likert simulator no longer runs, so these are unused.
                "stage_means": None,
                "responses": None,
                "raw_response": None,
                "error": res.get("error"),
            }

    if any(p["distribution"] is None for p in profiles.values()):
        return None
    return profiles


# ---------------------------------------------------------------------------
# Continuation generation (mediator IS in the speakers' histories)
# ---------------------------------------------------------------------------

def generate_continuation_with_mediator(
    sim_model: str,
    intervention_turn: int,
    n_turns: int,
    dialogue_entries: list[dict],
    agent_prompts: dict,
    mediator_utterance: str,
    api_key: str,
) -> tuple[dict[str, list[dict]], list[dict], Optional[str]]:
    """Build per-speaker histories up to intervention_turn, inject the
    mediator utterance as a 'user'-role message in BOTH histories, then
    generate continuation turns intervention_turn+1 .. n_turns. Each
    speaker sees the mediator and responds.

    Returns (histories, new_turns, error). histories[aid] is the full
    in-memory message list for that speaker after generation. new_turns
    is the list of generated turn records.
    """
    histories = build_initial_histories(
        dialogue_entries, agent_prompts, intervention_turn,
    )

    # Inject mediator utterance into BOTH speakers' histories as a user
    # message (mediator visible to both, persists in history).
    mediator_msg = {"role": "user", "content": f"[mediator]: {mediator_utterance}"}
    for aid in histories:
        histories[aid].append(dict(mediator_msg))

    # Schedule of any conflict_instruction injections still to come on
    # turns > intervention_turn. In the asymmetric setup with
    # INTERVENTION_DELAY=1 the conflict agent's injection happens at
    # conflict_turn, which is <= intervention_turn, so this map is
    # usually empty. We still honor it for robustness in case a future
    # configuration places injection past the intervention.
    injection_at_turn: dict[int, str] = {}
    for entry in dialogue_entries:
        t = entry["turn"]
        if t <= intervention_turn:
            continue
        if entry.get("prompt_injected"):
            injection_at_turn[t] = entry["agent"]

    new_turns: list[dict] = []
    for t in range(intervention_turn + 1, n_turns + 1):
        speaker = _agent_for_original_turn(t)
        if speaker not in histories:
            return histories, new_turns, f"unknown speaker for turn {t}"

        # If this turn is a scheduled injection slot for this speaker,
        # deliver conflict_instruction as a one-shot user prompt for
        # THIS call only (mirrors generate.py). Do NOT append it to
        # the persistent history.
        ci_call_history = histories[speaker]
        if injection_at_turn.get(t) == speaker:
            ci = (agent_prompts.get(speaker, {})
                  .get("conflict_instruction", ""))
            if ci:
                ci_call_history = histories[speaker] + [
                    {"role": "user", "content": ci}
                ]

        raw, err = call_openrouter(
            sim_model, ci_call_history, api_key,
            max_tokens=CONTINUATION_MAX_TOKENS,
            temperature=CONTINUATION_TEMPERATURE,
        )
        if not raw:
            return histories, new_turns, err or f"empty response at turn {t}"
        message = raw.strip()
        new_turns.append({"turn": t, "agent": speaker, "message": message})
        for aid in histories:
            role = "assistant" if aid == speaker else "user"
            histories[aid].append({"role": role, "content": message})

    return histories, new_turns, None


def truncate_histories_to_turn(
    full_histories: dict[str, list[dict]],
    dialogue_entries: list[dict],
    intervention_turn: int,
    new_turns: list[dict],
    target_turn: int,
    mediator_utterance: Optional[str],
    agent_prompts: dict,
) -> dict[str, list[dict]]:
    """Rebuild each speaker's history as it WOULD have been right after
    'target_turn' was uttered, given the original dialogue, the optional
    mediator utterance, and the continuation new_turns. Used to take a
    DMIS measurement at trajectory turns."""
    if intervention_turn >= target_turn:
        # Replay original dialogue up to target_turn; mediator hasn't
        # spoken yet at this point, so do not include it.
        return build_initial_histories(
            dialogue_entries, agent_prompts, target_turn,
        )

    # Replay original dialogue up to intervention_turn, then add mediator,
    # then add continuation turns up to target_turn.
    histories = build_initial_histories(
        dialogue_entries, agent_prompts, intervention_turn,
    )
    if mediator_utterance is not None:
        msg = {"role": "user",
               "content": f"[mediator]: {mediator_utterance}"}
        for aid in histories:
            histories[aid].append(dict(msg))
    for nt in new_turns:
        if nt["turn"] > target_turn:
            break
        speaker = nt["agent"]
        message = nt["message"]
        for aid in histories:
            role = "assistant" if aid == speaker else "user"
            histories[aid].append({"role": role, "content": message})
    return histories


def truncate_histories_original(
    dialogue_entries: list[dict],
    agent_prompts: dict,
    target_turn: int,
) -> dict[str, list[dict]]:
    """Histories using ONLY the original dialogue file (no mediator,
    no synthetic continuation). Used for the original-dialogue baseline
    Pre/Post/trajectory measurements."""
    return build_initial_histories(
        dialogue_entries, agent_prompts, target_turn,
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers (both-success)
# ---------------------------------------------------------------------------

def stage_shift_per_agent(
    pre_profiles: dict[str, dict],
    post_profiles: dict[str, dict],
) -> dict[str, int]:
    """post_argmax_idx - pre_argmax_idx for each agent (positive = forward
    along the DMIS scale)."""
    out: dict[str, int] = {}
    for aid in pre_profiles:
        pre = pre_profiles.get(aid, {}).get("argmax_idx")
        post = post_profiles.get(aid, {}).get("argmax_idx")
        if pre is None or post is None:
            out[aid] = None
        else:
            out[aid] = post - pre
    return out


def both_acceptance(profiles: dict[str, dict]) -> bool:
    """Both speakers' argmax >= ACCEPTANCE_IDX_THRESHOLD."""
    for aid, p in profiles.items():
        idx = p.get("argmax_idx")
        if idx is None or idx < ACCEPTANCE_IDX_THRESHOLD:
            return False
    return True


def find_first_acceptance_turn(
    trajectory: list[dict],
) -> Optional[int]:
    """trajectory: ordered list of {turn, profiles}. Returns the smallest
    turn at which both agents have argmax >= 3, or None."""
    for tp in trajectory:
        if both_acceptance(tp["profiles"]):
            return tp["turn"]
    return None


def regression_summary(
    post_profiles: dict[str, dict],
    trajectory: list[dict],
) -> dict:
    """Did any agent's argmax_idx drop below its post value at any
    later trajectory turn? max_regression = largest single-agent
    step-down across the trajectory.

    A trajectory entry whose argmax is None is skipped.
    """
    regressed = False
    max_reg = 0
    per_agent_min: dict[str, int] = {}
    for aid, p in post_profiles.items():
        post_idx = p.get("argmax_idx")
        if post_idx is None:
            continue
        per_agent_min[aid] = post_idx
        for tp in trajectory:
            traj_p = tp["profiles"].get(aid, {})
            t_idx = traj_p.get("argmax_idx")
            if t_idx is None:
                continue
            if t_idx < post_idx:
                regressed = True
                drop = post_idx - t_idx
                if drop > max_reg:
                    max_reg = drop
            if t_idx < per_agent_min[aid]:
                per_agent_min[aid] = t_idx
    return {
        "regressed_at_some_point": regressed,
        "max_regression": max_reg,
        "per_agent_min_idx": per_agent_min,
    }


def find_first_next_stage_turn_per_agent(
    pre_profiles: dict[str, dict],
    post_profiles: dict[str, dict],
    trajectory: list[dict],
    intervention_turn: int,
    post_measurement_turn: int,
) -> dict[str, dict]:
    """For each agent, find the first turn (across post + trajectory) at
    which the agent's argmax_idx >= pre_idx + 1, i.e. they advanced at
    least one DMIS stage from their Pre baseline. Returns:
        {agent_id: {"reached": bool, "turn": int|None,
                    "turns_after_intervention": int|None}}
    The post-measurement profile counts as the first measurement point;
    if post is already at or above pre_idx+1, post_measurement_turn is
    used as the reach turn."""
    out: dict[str, dict] = {}
    for aid, pre in pre_profiles.items():
        pre_idx = pre.get("argmax_idx")
        if pre_idx is None:
            out[aid] = {"reached": False, "turn": None,
                        "turns_after_intervention": None}
            continue
        target_idx = pre_idx + 1
        reach_turn = None
        # Check post first (it is chronologically first after the
        # intervention).
        post_idx = (post_profiles.get(aid) or {}).get("argmax_idx")
        if post_idx is not None and post_idx >= target_idx:
            reach_turn = post_measurement_turn
        else:
            for tp in trajectory:
                t_idx = (tp["profiles"].get(aid) or {}).get("argmax_idx")
                if t_idx is None:
                    continue
                if t_idx >= target_idx:
                    reach_turn = tp["turn"]
                    break
        out[aid] = {
            "reached": reach_turn is not None,
            "turn": reach_turn,
            "turns_after_intervention":
                (reach_turn - intervention_turn)
                if reach_turn is not None else None,
        }
    return out


def find_first_both_acceptance_turn(
    post_profiles: dict[str, dict],
    trajectory: list[dict],
    post_measurement_turn: int,
) -> Optional[int]:
    """First chronologically-ordered measurement turn (post or any
    trajectory turn) at which BOTH agents' argmax_idx >= 3."""
    if both_acceptance(post_profiles):
        return post_measurement_turn
    for tp in trajectory:
        if both_acceptance(tp["profiles"]):
            return tp["turn"]
    return None


def effect_size_per_agent(
    pre_profiles: dict[str, dict],
    post_profiles: dict[str, dict],
) -> dict[str, dict]:
    """Per-agent signed Wasserstein-1 (Paige weighted) and
    expected_position_shift, computed from the dataset_quality
    compare_distributions helper."""
    out: dict[str, dict] = {}
    for aid in pre_profiles:
        pre_d = pre_profiles[aid].get("distribution")
        post_d = post_profiles.get(aid, {}).get("distribution")
        if pre_d is None or post_d is None:
            out[aid] = None
            continue
        cmp_ = compare_distributions(pre_d, post_d)
        out[aid] = {
            "signed_wasserstein_1": cmp_["signed_wasserstein_1"],
            "expected_position_pre": cmp_["expected_position_pre"],
            "expected_position_post": cmp_["expected_position_post"],
            "expected_position_shift": cmp_["expected_position_shift"],
        }
    return out


def pair_effect_size(
    pre_profiles: dict[str, dict],
    post_profiles: dict[str, dict],
) -> Optional[dict]:
    """Combine the two agents' distributions into a single pair-level
    distribution (mean) and compute the signed Wasserstein-1 between
    pair_pre and pair_post. This treats the two speakers as one
    collective state and measures how that joint state shifted from Pre
    to Post.
    """
    pre_dists = []
    post_dists = []
    for aid in pre_profiles:
        d_pre = pre_profiles[aid].get("distribution")
        d_post = (post_profiles.get(aid) or {}).get("distribution")
        if d_pre is None or d_post is None:
            return None
        pre_dists.append(d_pre)
        post_dists.append(d_post)
    if not pre_dists or not post_dists:
        return None
    pair_pre = combine_distributions(pre_dists, method="mean")
    pair_post = combine_distributions(post_dists, method="mean")
    cmp_ = compare_distributions(pair_pre, pair_post)
    return {
        "pair_pre_distribution": pair_pre,
        "pair_post_distribution": pair_post,
        "signed_wasserstein_1": cmp_["signed_wasserstein_1"],
        "expected_position_pre": cmp_["expected_position_pre"],
        "expected_position_post": cmp_["expected_position_post"],
        "expected_position_shift": cmp_["expected_position_shift"],
    }


def trajectory_auc(
    pre_profiles: dict[str, dict],
    post_profiles: dict[str, dict],
    trajectory: list[dict],
    post_measurement_turn: int,
    intervention_turn: int,
    n_turns: int,
    agent_subset: Optional[list[str]] = None,
) -> Optional[float]:
    """Time-normalized area under the stage-shift trajectory, integrated
    over the post-intervention window mapped to [0, 1].

    By default the metric averages over all agents in pre_profiles. To
    focus on a single agent (e.g. the conflict agent in an asymmetric
    setup), pass agent_subset=[agent_id]. The averaging then runs over
    only the agents listed in agent_subset.

    For each measurement timepoint t we have

        progress(t) = mean over selected agents of (argmax_idx_t - argmax_idx_pre)

    To make trajectories of different lengths comparable, every
    measurement turn t is mapped to a normalized time

        tau = (t - intervention_turn) / (n_turns - intervention_turn)

    so that tau in (0, 1] for every measured turn. The metric is the
    area under the progress(tau) curve from tau = 0 to tau = 1, which
    equals the time-weighted mean progress over the post-intervention
    window:

        trajectory_auc = integral from 0 to 1 of progress(tau) dtau

    Edge handling
    -------------
    * (0, tau_first) is filled with progress(tau_first), i.e. the Post
      value is extended backward to the moment of intervention. Reaching
      a level by Post counts as having reached it from the moment the
      mediator spoke.
    * (tau_last, 1) is filled with progress(tau_last) when the last
      timepoint does not coincide with n_turns (in this setup it always
      does).
    * If only one timepoint is available (Post is the only measurement),
      trajectory_auc = progress(post).
    * If n_turns == intervention_turn the window is undefined; returns
      None.

    Interpretation
    --------------
    * +3.0 : pair reaches Acceptance at Post and holds it through
             n_turns (best case, given pre = Denial).
    * 0.0  : no average movement.
    * negative : regression below Pre dominates the window.

    The metric is independent of trajectory length: a Post-only dialogue
    that scores +1 receives the same window-mean as a longer dialogue
    that holds +1 across all measured turns.
    """
    # Restrict to agent_subset if provided. Otherwise average over all
    # agents that have a Pre measurement.
    pre_idx: dict[str, Optional[int]] = {
        aid: (p or {}).get("argmax_idx") for aid, p in pre_profiles.items()
        if (agent_subset is None or aid in agent_subset)
    }
    if any(v is None for v in pre_idx.values()):
        return None
    if not pre_idx:
        return None

    duration = n_turns - intervention_turn
    if duration <= 0:
        return None

    def _progress_at(profs: dict[str, dict]) -> Optional[float]:
        deltas: list[int] = []
        for aid, pre_v in pre_idx.items():
            t_idx = (profs.get(aid) or {}).get("argmax_idx")
            if t_idx is None:
                continue
            deltas.append(t_idx - pre_v)
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    # Build (tau, progress) point list, sorted by tau ascending.
    raw: list[dict] = []
    if post_profiles is not None:
        raw.append({
            "turn": post_measurement_turn,
            "profiles": post_profiles,
        })
    raw.extend(trajectory or [])

    points: list[tuple[float, float]] = []
    for tp in raw:
        prog = _progress_at(tp.get("profiles") or {})
        if prog is None:
            continue
        tau = (tp["turn"] - intervention_turn) / duration
        # Clamp tau into [0, 1] just in case (post_measurement_turn could
        # have been clipped to n_turns).
        if tau < 0:
            tau = 0.0
        elif tau > 1:
            tau = 1.0
        points.append((tau, prog))
    points.sort(key=lambda x: x[0])

    if not points:
        return None
    if len(points) == 1:
        # Only one measurement -> the entire [0, 1] window takes that value.
        return points[0][1]

    auc = 0.0

    # Backward fill: [0, tau_first] held at progress(tau_first).
    tau_first, prog_first = points[0]
    if tau_first > 0:
        auc += prog_first * tau_first

    # Trapezoidal integration between consecutive measured points.
    for i in range(len(points) - 1):
        tau_a, prog_a = points[i]
        tau_b, prog_b = points[i + 1]
        auc += (prog_a + prog_b) / 2.0 * (tau_b - tau_a)

    # Forward fill: (tau_last, 1] held at progress(tau_last). In our
    # setup tau_last is normally 1.0, so this contributes 0.
    tau_last, prog_last = points[-1]
    if tau_last < 1.0:
        auc += prog_last * (1.0 - tau_last)

    return auc


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def run_judge(
    judge_model: str,
    components: dict,
    utterance: str,
    api_key: str,
) -> dict:
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        stage=components.get("stage", ""),
        description=components["description"],
        mediation=components["mediation"],
        utterance=utterance,
    )
    raw, err = call_simple(
        judge_model, JUDGE_SYSTEM_PROMPT, user_prompt, api_key,
        max_tokens=200,
    )
    score, reasoning = parse_judge_score(raw)
    return {
        "judge_model": judge_model,
        "score": score,
        "reasoning": reasoning,
        "raw_response": raw,
        "error": err,
    }


# ---------------------------------------------------------------------------
# One mediation path
# ---------------------------------------------------------------------------

def run_mediation_path(
    *,
    label: str,
    intervention_turn: int,
    conflict_turn: int,
    n_turns: int,
    pre_profiles: dict[str, dict],
    mediator_model: str,
    sim_model: str,
    judge_model: Optional[str],
    dialogue_entries: list[dict],
    agent_prompts: dict,
    agents_meta: dict[str, dict],
    api_key: str,
    max_workers: int,
    components: Optional[dict] = None,
    use_definition: bool = False,
    gt_turn_for_skip: Optional[int] = None,
    conflict_agent_id: Optional[str] = None,
) -> dict:
    """One mediation evaluation path.

    Pre measurement is shared across paths and passed in as ``pre_profiles``
    (computed once at conflict_turn by the caller). This function:

      1. Generates the mediator utterance (with or without the
         description+mediation definitions, depending on ``components``).
      2. Continues the dialogue turn-by-turn, with the mediator visible
         in both speakers' histories.
      3. Measures Post profiles at conflict_turn + 3 and trajectory
         profiles at every turn from conflict_turn + 4 .. n_turns.
      4. Computes both-success diagnostics and effect sizes.
      5. Optionally calls the judge (only when ``components`` is provided
         AND a judge_model was passed; otherwise judge stays None).
    """
    base = {
        "label": label,
        "intervention_turn": intervention_turn,
        "definition_provided": use_definition,
        "components": components,
        "mediator_model": mediator_model,
    }

    pre_text = format_dialogue_for_turn_picker(
        [e for e in dialogue_entries if e["turn"] <= intervention_turn],
        agent_prompts,
    )

    # ---- mediator utterance ----
    if use_definition and components:
        med_user = MEDIATION_DEF_USER_PROMPT_TEMPLATE.format(
            dialogue=pre_text,
            turn=intervention_turn,
            stage=components.get("stage", ""),
            description=components["description"],
            mediation=components["mediation"],
        )
        med_system = MEDIATION_DEF_SYSTEM_PROMPT
    else:
        med_user = MEDIATION_USER_PROMPT_TEMPLATE.format(
            dialogue=pre_text, turn=intervention_turn,
        )
        med_system = MEDIATION_SYSTEM_PROMPT

    med_raw, med_err = call_simple(
        mediator_model, med_system, med_user, api_key, max_tokens=600,
    )
    mediation_utterance = parse_mediation(med_raw)

    if not mediation_utterance:
        return {
            **base,
            "mediation_utterance": None,
            "mediation_error": med_err or "could not parse mediation",
            "continuation_turns": None,
            "post_profiles": None,
            "trajectory": None,
            "diagnostics": None,
            "effect_size": None,
            "judge": None,
        }

    # ---- continuation (mediator visible) ----
    if intervention_turn >= n_turns:
        # No continuation possible; we cannot reach Post = conflict_turn + 3.
        return {
            **base,
            "mediation_utterance": mediation_utterance,
            "continuation_turns": [],
            "post_profiles": None,
            "trajectory": None,
            "diagnostics": None,
            "effect_size": None,
            "judge": None,
            "skip_reason": "intervention_turn >= n_turns",
        }

    full_histories, new_turns, cont_err = generate_continuation_with_mediator(
        sim_model, intervention_turn, n_turns,
        dialogue_entries, agent_prompts, mediation_utterance, api_key,
    )
    if cont_err is not None:
        return {
            **base,
            "mediation_utterance": mediation_utterance,
            "continuation_turns": new_turns,
            "post_profiles": None,
            "trajectory": None,
            "diagnostics": None,
            "effect_size": None,
            "judge": None,
            "continuation_error": cont_err,
        }

    # ---- Post measurement at conflict_turn + 3 ----
    post_target = conflict_turn + 3
    post_target = min(post_target, n_turns)
    post_histories = truncate_histories_to_turn(
        full_histories, dialogue_entries, intervention_turn, new_turns,
        post_target, mediation_utterance, agent_prompts,
    )
    post_profiles = measure_per_speaker_distributions(
        post_histories, agents_meta, sim_model, api_key, max_workers,
    )
    if post_profiles is None:
        return {
            **base,
            "mediation_utterance": mediation_utterance,
            "continuation_turns": new_turns,
            "post_profiles": None,
            "trajectory": None,
            "diagnostics": None,
            "effect_size": None,
            "judge": None,
            "skip_reason": "post measurement failed",
        }

    # ---- Trajectory measurement: turns conflict_turn+4 .. n_turns ----
    # Skip if SKIP_TRAJECTORY_MEASUREMENT env var is set (uses post-only AUC, ~15-25% faster).
    trajectory: list[dict] = []
    if os.environ.get("SKIP_TRAJECTORY_MEASUREMENT") != "1":
        for t in range(post_target + 1, n_turns + 1):
            traj_histories = truncate_histories_to_turn(
                full_histories, dialogue_entries, intervention_turn, new_turns,
                t, mediation_utterance, agent_prompts,
            )
            traj_profiles = measure_per_speaker_distributions(
                traj_histories, agents_meta, sim_model, api_key, max_workers,
            )
            trajectory.append({
                "turn": t,
                "profiles": traj_profiles or {},
            })

    # ---- Diagnostics ----
    shifts = stage_shift_per_agent(pre_profiles, post_profiles)
    diag = {
        "stage_shift_per_agent": shifts,
        "pre_argmax_per_agent": {
            aid: pre_profiles[aid].get("argmax_idx") for aid in pre_profiles
        },
        "post_argmax_per_agent": {
            aid: post_profiles[aid].get("argmax_idx") for aid in post_profiles
        },
        "pre_argmax_stage_per_agent": {
            aid: pre_profiles[aid].get("argmax_stage") for aid in pre_profiles
        },
        "post_argmax_stage_per_agent": {
            aid: post_profiles[aid].get("argmax_stage")
            for aid in post_profiles
        },
        "both_acceptance_at_post": both_acceptance(post_profiles),
        "post_measurement_turn": post_target,
    }
    # Per-agent next-stage reach (pre_idx + 1), counted from the
    # intervention turn. If the agent reached next stage already at Post,
    # post_target is used as the reach turn.
    next_stage = find_first_next_stage_turn_per_agent(
        pre_profiles, post_profiles, trajectory,
        intervention_turn, post_target,
    )
    diag["next_stage_per_agent"] = next_stage
    # Both-success acceptance turn: across post + trajectory, the first
    # turn where BOTH agents' argmax_idx >= 3.
    both_acc_turn = find_first_both_acceptance_turn(
        post_profiles, trajectory, post_target,
    )
    diag["both_acceptance_turn"] = both_acc_turn
    diag["turns_to_both_acceptance_after_intervention"] = (
        both_acc_turn - intervention_turn
        if both_acc_turn is not None else None
    )
    diag.update(regression_summary(post_profiles, trajectory))

    # Combined two-agent movement metric: time-normalized AUC of pair
    # ---- Pair-level metrics (averaged over BOTH agents) ----
    # Time-normalized AUC of pair mean stage shift over the
    # post-intervention window. Trajectory-length invariant.
    diag["trajectory_auc"] = trajectory_auc(
        pre_profiles, post_profiles, trajectory, post_target,
        intervention_turn=intervention_turn, n_turns=n_turns,
    )

    effect = effect_size_per_agent(pre_profiles, post_profiles)
    pair_w1 = pair_effect_size(pre_profiles, post_profiles)

    # ---- Conflict-agent-only metrics ----
    # In an asymmetric setup the mediator's main target is the conflict
    # agent. Pair-level metrics dilute this effect by averaging in the
    # non-conflict agent (whose stance is driven only by the base
    # prompt). Reporting conflict-agent-only versions of the same
    # metrics isolates the mediator's effect on the agent who is
    # actually exhibiting the cultural difficulty.
    if conflict_agent_id and conflict_agent_id in pre_profiles:
        conflict_auc = trajectory_auc(
            pre_profiles, post_profiles, trajectory, post_target,
            intervention_turn=intervention_turn, n_turns=n_turns,
            agent_subset=[conflict_agent_id],
        )
        diag["conflict_trajectory_auc"] = conflict_auc
        # Conflict agent's per-agent signed W1 (already in `effect`).
        ca_effect = (effect or {}).get(conflict_agent_id) or {}
        conflict_w1_value = ca_effect.get("signed_wasserstein_1")
    else:
        diag["conflict_trajectory_auc"] = None
        conflict_w1_value = None
    conflict_effect = {
        "agent_id": conflict_agent_id,
        "signed_wasserstein_1": conflict_w1_value,
    }

    # ---- Judge: every path; skip only when (predicted-turn path AND
    # predicted_turn != gt_turn AND the conflict agent's Pre-measurement
    # argmax stage does not match the target stage). Reusing the
    # pre_profiles avoids an extra measurement; semantically Pre is the
    # conflict_turn state right after the conflict agent surfaced the
    # disagreement and enacted the ethnocentric pattern, so it is the
    # right reference for "did the dialogue actually express the target
    # pattern at the moment the mediator stepped in?". ----
    judge_result = None
    judge_skip_reason = None
    if judge_model and components:
        skip = False
        if label.startswith("predicted") and (
            intervention_turn != gt_turn_for_skip
        ):
            target_stage = (components or {}).get("stage")
            stages_at_pre = [
                p.get("argmax_stage") for p in pre_profiles.values()
            ]
            if target_stage:
                neither_matches = all(
                    s != target_stage for s in stages_at_pre
                )
                if neither_matches:
                    skip = True
                    judge_skip_reason = (
                        f"predicted_turn={intervention_turn} != gt_turn="
                        f"{gt_turn_for_skip}: pre argmax stages "
                        f"{stages_at_pre} both != target {target_stage}"
                    )
        if not skip:
            judge_result = run_judge(
                judge_model, components, mediation_utterance, api_key,
            )

    return {
        **base,
        "mediation_utterance": mediation_utterance,
        "continuation_turns": new_turns,
        "post_profiles": post_profiles,
        "trajectory": trajectory,
        "diagnostics": diag,
        "effect_size": effect,
        "pair_effect": pair_w1,
        "conflict_effect": conflict_effect,
        "judge": judge_result,
        "judge_skip_reason": judge_skip_reason,
    }


# ---------------------------------------------------------------------------
# Original-dialogue baseline (mediator-free)
# ---------------------------------------------------------------------------

def run_original_baseline(
    *,
    conflict_turn: int,
    n_turns: int,
    pre_profiles: dict[str, dict],
    sim_model: str,
    dialogue_entries: list[dict],
    agent_prompts: dict,
    agents_meta: dict[str, dict],
    api_key: str,
    max_workers: int,
    conflict_agent_id: Optional[str] = None,
) -> dict:
    """Take Post and trajectory measurements on the ORIGINAL dialogue file,
    using the same Pre profiles. There is no mediator and no synthetic
    continuation; each measurement just rebuilds the original speaker
    histories up to the corresponding turn."""
    post_target = min(conflict_turn + 3, n_turns)
    post_histories = truncate_histories_original(
        dialogue_entries, agent_prompts, post_target,
    )
    post_profiles = measure_per_speaker_distributions(
        post_histories, agents_meta, sim_model, api_key, max_workers,
    )
    if post_profiles is None:
        return {
            "label": "original_dialogue",
            "post_profiles": None,
            "trajectory": None,
            "diagnostics": None,
            "effect_size": None,
            "skip_reason": "post measurement failed",
        }

    trajectory: list[dict] = []
    if os.environ.get("SKIP_TRAJECTORY_MEASUREMENT") != "1":
        for t in range(post_target + 1, n_turns + 1):
            h = truncate_histories_original(dialogue_entries, agent_prompts, t)
            p = measure_per_speaker_distributions(
                h, agents_meta, sim_model, api_key, max_workers,
            )
            trajectory.append({"turn": t, "profiles": p or {}})

    shifts = stage_shift_per_agent(pre_profiles, post_profiles)
    diag = {
        "stage_shift_per_agent": shifts,
        "pre_argmax_per_agent": {
            aid: pre_profiles[aid].get("argmax_idx") for aid in pre_profiles
        },
        "post_argmax_per_agent": {
            aid: post_profiles[aid].get("argmax_idx") for aid in post_profiles
        },
        "pre_argmax_stage_per_agent": {
            aid: pre_profiles[aid].get("argmax_stage") for aid in pre_profiles
        },
        "post_argmax_stage_per_agent": {
            aid: post_profiles[aid].get("argmax_stage")
            for aid in post_profiles
        },
        "both_acceptance_at_post": both_acceptance(post_profiles),
        "post_measurement_turn": post_target,
    }
    first_acc = find_first_acceptance_turn(trajectory)
    diag["first_acceptance_turn"] = first_acc
    diag["turns_to_acceptance_after_post"] = (
        first_acc - post_target if first_acc is not None else None
    )
    diag.update(regression_summary(post_profiles, trajectory))

    # Time-normalized AUC for the original (mediator-free) continuation.
    # We use conflict_turn + 1 as the reference "intervention point" so
    # the time window matches the mediator paths' Pre-to-n_turns window
    # exactly.
    diag["trajectory_auc"] = trajectory_auc(
        pre_profiles, post_profiles, trajectory, post_target,
        intervention_turn=conflict_turn + 1, n_turns=n_turns,
    )

    effect = effect_size_per_agent(pre_profiles, post_profiles)

    # Conflict-agent-only metrics (asymmetric setup).
    if conflict_agent_id and conflict_agent_id in pre_profiles:
        diag["conflict_trajectory_auc"] = trajectory_auc(
            pre_profiles, post_profiles, trajectory, post_target,
            intervention_turn=conflict_turn + 1, n_turns=n_turns,
            agent_subset=[conflict_agent_id],
        )
        ca_effect = (effect or {}).get(conflict_agent_id) or {}
        conflict_w1_value = ca_effect.get("signed_wasserstein_1")
    else:
        diag["conflict_trajectory_auc"] = None
        conflict_w1_value = None
    conflict_effect = {
        "agent_id": conflict_agent_id,
        "signed_wasserstein_1": conflict_w1_value,
    }

    return {
        "label": "original_dialogue",
        "post_profiles": post_profiles,
        "trajectory": trajectory,
        "diagnostics": diag,
        "effect_size": effect,
        "pair_effect": pair_effect_size(pre_profiles, post_profiles),
        "conflict_effect": conflict_effect,
    }


# ---------------------------------------------------------------------------
# Per-dialogue evaluation
# ---------------------------------------------------------------------------

def evaluate_dialogue(
    filepath: Path,
    models: list[str],
    sim_model: str,
    judge_model: Optional[str],
    api_key: str,
    *,
    do_predicted_path: bool = True,
    do_predicted_def_path: bool = True,
    do_gt_def_path: bool = True,
    do_gt_path: bool = True,
    do_random_path: bool = False,
    do_random_def_path: bool = False,
    do_original_baseline: bool = True,
    conflict_table: Optional[dict] = None,
    max_workers: int = 4,
) -> Optional[dict]:
    data = load_dialogue(filepath)
    dialogue_entries = data.get("dialogue", []) or []
    agent_prompts = data.get("agent_prompts", {}) or {}
    agents_meta = agent_metadata(data)
    if len(agents_meta) < 2:
        return None
    n_turns = len(dialogue_entries)
    if n_turns < 1:
        return None

    conflict_turn = data.get("conflict_turn")
    if not isinstance(conflict_turn, int):
        return None

    # Pre measurement turn = conflict_turn (right after the conflict
    # agent has surfaced the goal_2 disagreement and enacted the
    # ethnocentric pattern; the mediator has not yet spoken).
    pre_target = conflict_turn
    if pre_target < 1 or pre_target > n_turns:
        return None

    # GT intervention point = conflict_turn itself. The mediator inserts
    # immediately after the conflict turn surfaces (no separate
    # intervention_turn). Any legacy "intervention_turn" field in the
    # source JSON is ignored.
    gt_turn = conflict_turn

    # Stage of this dialogue (from build_agent_prompts.py:dmis)
    stage = ((data.get("dmis") or {}).get("stage") or "").strip()
    components = lookup_mediation_components(stage, conflict_table)

    # Asymmetric setup: only the conflict agent received a
    # conflict_instruction. Conflict-agent-only metrics isolate the
    # mediator's effect on the agent who is actually exhibiting the
    # cultural difficulty; pair-level metrics still average over both.
    conflict_agent_id = identify_conflict_agent(data, conflict_turn)

    # ---------- Pre measurement (shared across paths) ----------
    pre_histories = build_initial_histories(
        dialogue_entries, agent_prompts, pre_target,
    )
    pre_profiles = measure_per_speaker_distributions(
        pre_histories, agents_meta, sim_model, api_key, max_workers,
    )
    if pre_profiles is None:
        return None

    # ---------- Stage 1: streaming conflict-turn detection ----------
    # Each model walks turn-by-turn over the dialogue prefix; the first
    # turn that earns a 'Y' is the predicted conflict_turn. The mediator
    # inserts immediately after that turn.
    needs_prediction = do_predicted_path or do_predicted_def_path

    turn_predictions: dict[str, dict] = {}
    if needs_prediction:
        random_pred_after_gt = os.environ.get("RANDOM_PRED_AFTER_GT") == "1"

        def _pred(model: str):
            if random_pred_after_gt and gt_turn is not None:
                import random as _r
                lo = gt_turn + 1
                hi = max(lo, n_turns - 1)
                predicted = _r.randint(lo, hi) if lo <= hi else None
                log = [{"random_pred_after_gt": True,
                        "lo": lo, "hi": hi, "picked": predicted}]
                return model, predicted, log
            predicted, log = detect_conflict_turn_streaming(
                model, dialogue_entries, agent_prompts, api_key, n_turns,
            )
            return model, predicted, log

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_pred, m) for m in models]
            for fut in as_completed(futs):
                model, predicted, log = fut.result()
                turn_predictions[model] = {
                    "predicted_turn": predicted,
                    "detection_log": log,
                    "error": None,
                    "correct": (
                        predicted == gt_turn if predicted is not None else False
                    ),
                }
    else:
        for model in models:
            turn_predictions[model] = {
                "predicted_turn": None,
                "detection_log": None,
                "error": None,
                "correct": False,
            }

    # ---------- Per-model: 4 paths ----------
    by_model: dict[str, dict] = {}
    for model in models:
        tp = turn_predictions[model]
        pt = tp["predicted_turn"]
        pt_valid = pt is not None and 1 <= pt <= n_turns

        predicted_path = None
        if do_predicted_path and pt_valid:
            predicted_path = run_mediation_path(
                label="predicted",
                intervention_turn=pt,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=False,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        predicted_def_path = None
        if do_predicted_def_path and pt_valid and components:
            predicted_def_path = run_mediation_path(
                label="predicted_def",
                intervention_turn=pt,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=True,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        gt_path = None
        if do_gt_path:
            gt_path = run_mediation_path(
                label="gt",
                intervention_turn=gt_turn,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=False,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        # Compute a random intervention turn AFTER gt_turn (shared by both random paths so they use the same insertion point per sid/model).
        rand_turn = None
        if (do_random_path or do_random_def_path) and gt_turn is not None:
            import random as _rmod
            _lo = gt_turn + 1
            _hi = max(_lo, n_turns - 1)
            if _lo <= _hi:
                rand_turn = _rmod.randint(_lo, _hi)
        rt_valid = rand_turn is not None and 1 <= rand_turn <= n_turns

        random_path = None
        if do_random_path and rt_valid:
            random_path = run_mediation_path(
                label="random",
                intervention_turn=rand_turn,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=False,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        random_def_path = None
        if do_random_def_path and rt_valid and components:
            random_def_path = run_mediation_path(
                label="random_def",
                intervention_turn=rand_turn,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=True,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        gt_def_path = None
        if do_gt_def_path and components:
            gt_def_path = run_mediation_path(
                label="gt_def",
                intervention_turn=gt_turn,
                conflict_turn=conflict_turn,
                n_turns=n_turns,
                pre_profiles=pre_profiles,
                mediator_model=model,
                sim_model=sim_model,
                judge_model=judge_model,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                agents_meta=agents_meta,
                api_key=api_key,
                max_workers=max_workers,
                components=components,
                use_definition=True,
                gt_turn_for_skip=gt_turn,
                conflict_agent_id=conflict_agent_id,
            )

        by_model[model] = {
            "turn_prediction": tp,
            "random_turn": rand_turn,
            "predicted_path": predicted_path,
            "predicted_def_path": predicted_def_path,
            "gt_path": gt_path,
            "gt_def_path": gt_def_path,
            "random_path": random_path,
            "random_def_path": random_def_path,
        }

    # ---------- Original-dialogue baseline (one only) ----------
    original_baseline = None
    if do_original_baseline:
        original_baseline = run_original_baseline(
            conflict_turn=conflict_turn,
            n_turns=n_turns,
            pre_profiles=pre_profiles,
            sim_model=sim_model,
            dialogue_entries=dialogue_entries,
            agent_prompts=agent_prompts,
            agents_meta=agents_meta,
            api_key=api_key,
            max_workers=max_workers,
            conflict_agent_id=conflict_agent_id,
        )

    return {
        "scenario_id": filepath.stem,
        "ground_truth_turn": gt_turn,
        "n_turns": n_turns,
        "conflict_turn": conflict_turn,
        "conflict_agent_id": conflict_agent_id,
        "pre_measurement_turn": pre_target,
        "post_measurement_turn": min(conflict_turn + 3, n_turns),
        "stage": stage,
        "stage_components_available": components is not None,
        "components": components,
        "agents": agents_meta,
        "agent_prompts": agent_prompts,
        "source_dialogue": dialogue_entries,
        "pre_profiles": pre_profiles,
        "by_model": by_model,
        "original_baseline": original_baseline,
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def _path_stats(path_results: list[dict]) -> dict:
    """Aggregate one path's results across dialogues. Each list element is
    a path dict (from run_mediation_path) or None.

    Aggregated metrics:
      - trajectory_auc : time-normalized AUC of pair mean stage shift
      - pair_signed_w1 : signed Wasserstein-1 between pair_pre and pair_post
                         (the two agents' distributions averaged into one)
      - judge          : mean LLM-judge score, with skip-tracking
    """
    n_total = 0
    n_skipped = 0
    auc_values: list[float] = []
    pair_w1_values: list[float] = []
    conflict_auc_values: list[float] = []
    conflict_w1_values: list[float] = []
    judge_scores: list[int] = []
    n_judge_skipped = 0
    for path in path_results:
        if path is None:
            continue
        if not path.get("diagnostics"):
            n_skipped += 1
            continue
        n_total += 1
        diag = path["diagnostics"]

        ta = diag.get("trajectory_auc")
        if isinstance(ta, (int, float)):
            auc_values.append(float(ta))

        pw = (path.get("pair_effect") or {}).get("signed_wasserstein_1")
        if isinstance(pw, (int, float)):
            pair_w1_values.append(float(pw))

        # Conflict-agent-only metrics (asymmetric setup).
        cta = diag.get("conflict_trajectory_auc")
        if isinstance(cta, (int, float)):
            conflict_auc_values.append(float(cta))

        cw = (path.get("conflict_effect") or {}).get("signed_wasserstein_1")
        if isinstance(cw, (int, float)):
            conflict_w1_values.append(float(cw))

        j = path.get("judge")
        if j is None:
            if path.get("judge_skip_reason"):
                n_judge_skipped += 1
        else:
            s = j.get("score")
            if isinstance(s, int):
                judge_scores.append(s)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n_total,
        "n_skipped": n_skipped,
        "mean_trajectory_auc": _mean(auc_values),
        "n_trajectory_auc": len(auc_values),
        "mean_pair_signed_w1": _mean(pair_w1_values),
        "n_pair_w1": len(pair_w1_values),
        "mean_conflict_trajectory_auc": _mean(conflict_auc_values),
        "n_conflict_trajectory_auc": len(conflict_auc_values),
        "mean_conflict_signed_w1": _mean(conflict_w1_values),
        "n_conflict_w1": len(conflict_w1_values),
        "mean_judge_score": _mean(judge_scores),
        "n_judged": len(judge_scores),
        "n_judge_skipped": n_judge_skipped,
    }


def aggregate_stats(all_results: dict, models: list[str]) -> dict:
    """Aggregate across dialogues. Per model: per-path stats + turn
    accuracy. Plus original_baseline aggregated separately."""
    by_model: dict[str, dict] = {}
    for model in models:
        per_path = {
            "predicted_path": [],
            "predicted_def_path": [],
            "gt_path": [],
            "gt_def_path": [],
            "random_path": [],
            "random_def_path": [],
        }
        n = 0
        n_correct = 0
        n_parsed = 0
        abs_errs: list[int] = []
        for fres in all_results.values():
            block = fres.get("by_model", {}).get(model)
            if block is None:
                continue
            n += 1
            tp = block.get("turn_prediction") or {}
            pt = tp.get("predicted_turn")
            if pt is not None:
                n_parsed += 1
                gt = fres.get("ground_truth_turn")
                if isinstance(gt, int):
                    abs_errs.append(abs(pt - gt))
            if tp.get("correct"):
                n_correct += 1
            for k in per_path:
                per_path[k].append(block.get(k))
        by_model[model] = {
            "n_total": n,
            "n_turn_parsed": n_parsed,
            "n_turn_correct": n_correct,
            "turn_accuracy": n_correct / n if n else None,
            "mean_abs_turn_error": (
                sum(abs_errs) / len(abs_errs) if abs_errs else None
            ),
            "predicted_path": _path_stats(per_path["predicted_path"]),
            "predicted_def_path": _path_stats(per_path["predicted_def_path"]),
            "gt_path": _path_stats(per_path["gt_path"]),
            "gt_def_path": _path_stats(per_path["gt_def_path"]),
            "random_path": _path_stats(per_path["random_path"]),
            "random_def_path": _path_stats(per_path["random_def_path"]),
        }

    # Original baseline (one row, attached to ORIGINAL_BASELINE_MODEL)
    original_list: list[dict] = []
    for fres in all_results.values():
        ob = fres.get("original_baseline")
        if ob:
            original_list.append(ob)
    original_stats = _path_stats(original_list)

    return {
        "by_model": by_model,
        "original_baseline": original_stats,
        "original_baseline_model": ORIGINAL_BASELINE_MODEL,
    }


def print_summary(summary: dict, models: list[str]) -> None:
    print()
    print("=" * 130)
    print("RESULTS - mediation effectiveness across four paths")
    print("Pre = conflict_turn   Post = conflict_turn + 3   "
          "Trajectory = conflict_turn + 4 ... n_turns")
    print("=" * 130)

    by_model = summary["by_model"]
    orig_stats = summary["original_baseline"]
    orig_model = summary["original_baseline_model"]

    def _f(v, nd=2):
        return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "    --"

    def _u(v, nd=2):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "  --"

    def _pct(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "  --"

    # ------ Section 1: turn prediction accuracy ------
    print()
    print("[ 1 ] Turn prediction accuracy")
    print("-" * 130)
    print(
        f"{'Model':<32} {'TurnAcc':>10} {'|err|':>7} {'parsed':>10} {'N':>5}"
    )
    print("-" * 130)
    for model in models:
        s = by_model.get(model)
        if s is None:
            continue
        ta = _pct(s["turn_accuracy"])
        err = _u(s["mean_abs_turn_error"], 1)
        parsed = f"{s['n_turn_parsed']}/{s['n_total']}"
        print(
            f"{model:<32} {ta:>10} {err:>7} {parsed:>10} "
            f"{s['n_total']:>5}"
        )

    # ------ Section 2: per-path mediation effectiveness ------
    path_order = [
        ("predicted_path",     "PATH 1 - Predicted turn, NO definition"),
        ("gt_path",            "PATH 2 - GT turn,        NO definition"),
        ("predicted_def_path", "PATH 3 - Predicted turn, WITH definition"),
        ("gt_def_path",        "PATH 4 - GT turn,        WITH definition"),
        ("random_path",        "PATH 5 - Random turn (after GT), NO definition"),
        ("random_def_path",    "PATH 6 - Random turn (after GT), WITH definition"),
    ]

    for path_key, title in path_order:
        print()
        print(f"[ {title} ]")
        print("-" * 130)
        header = (
            f"{'Model':<32} "
            f"{'AUC':>8} "
            f"{'W1':>8} "
            f"{'judge(j/N)':>15} "
            f"{'N':>5}"
        )
        print(header)
        print("-" * 130)
        for model in models:
            s = by_model.get(model)
            if s is None:
                continue
            ps = s.get(path_key) or {}
            n = ps.get("n", 0)
            judge = ps.get("mean_judge_score")
            n_j = ps.get("n_judged", 0)
            judge_str = (
                f"{judge:.2f} ({n_j}/{n})" if judge is not None
                else f"-- ({n_j}/{n})"
            )
            print(
                f"{model:<32} "
                f"{_f(ps.get('mean_conflict_trajectory_auc')):>8} "
                f"{_f(ps.get('mean_conflict_signed_w1')):>8} "
                f"{judge_str:>15} "
                f"{n:>5}"
            )

    # ------ Section 3: original-dialogue baseline ------
    print()
    print(f"[ ORIGINAL DIALOGUE BASELINE - mediator-free, {orig_model} only ]")
    print("-" * 130)
    print(
        f"{'Source':<32} "
        f"{'AUC':>8} "
        f"{'W1':>8} "
        f"{'':>15} "
        f"{'N':>5}"
    )
    print("-" * 130)
    n = orig_stats.get("n", 0)
    print(
        f"{'(original dialogue continuation)':<32} "
        f"{_f(orig_stats.get('mean_conflict_trajectory_auc')):>8} "
        f"{_f(orig_stats.get('mean_conflict_signed_w1')):>8} "
        f"{'':>15} "
        f"{n:>5}"
    )

    # ------ Legend ------
    print()
    print("=" * 130)
    print("Legend:")
    print("  TurnAcc   = % dialogues where streaming-detected conflict_turn matches GT conflict_turn")
    print("  |err|     = mean absolute turn-number error of the prediction")
    print("  parsed    = number of dialogues for which the detector produced a valid turn (out of N)")
    print("  AUC       = time-normalized AUC of the CONFLICT AGENT's stage-shift trajectory")
    print("              (post-intervention window mapped to [0,1]; +3.0 = reaches Acceptance and")
    print("              holds; 0.0 = no movement; negative = regression dominates).")
    print("  W1        = signed Wasserstein-1 of the CONFLICT AGENT's distribution shift")
    print("              (Pre -> Post; positive = forward along the DMIS scale).")
    print("  judge(j/N)= mean LLM-judge score (1-5); j = dialogues actually judged")
    print("              (skipped when detected turn != GT and pre stages != target stage)")
    print("=" * 130)


# ---------------------------------------------------------------------------
# IO: per-scenario per-model per-path JSON layout
# ---------------------------------------------------------------------------

def find_files(data_dir: str, ids: Optional[list[str]]) -> list[Path]:
    base = Path(data_dir)
    if ids:
        files = [base / f"{i}.json" for i in ids]
        return [f for f in files if f.exists()]
    files = list(base.glob("*.json"))

    def _key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    return sorted(files, key=_key)


def _model_dirname(model: str) -> str:
    return model.replace("/", "_")


PATH_FILENAMES = {
    "predicted_path":     "turn_predicted_wo_df",
    "predicted_def_path": "turn_predicted_w_df",
    "gt_path":            "turn_gt_wo_df",
    "gt_def_path":        "turn_gt_w_df",
    "random_path":        "turn_random_wo_df",
    "random_def_path":    "turn_random_w_df",
}


def _build_dialogue_entries(
    source_dialogue: list[dict],
    agents_meta: dict[str, dict],
    intervention_turn: int,
    mediator_utterance: Optional[str],
    continuation_turns: list[dict],
    pre_profiles: dict[str, dict],
    post_profiles: Optional[dict[str, dict]],
    post_measurement_turn: int,
    trajectory: Optional[list[dict]],
    target_stage: str,
) -> list[dict]:
    """Build the simplified dialogue list for the path JSON output.

    Each turn record has: turn, agent, speaker, message, prompt_injected,
    argmax_stage, argmax_idx. Original-dialogue turns up to and including
    intervention_turn carry no argmax label (argmax_stage=None) UNLESS the
    turn carries prompt_injected=True (the conflict agent's conflict_turn
    in the asymmetric setup). For continuation turns we look up the
    argmax from the matching trajectory or post measurement.

    A pseudo-entry with kind='mediator' is inserted between
    intervention_turn and intervention_turn+1.
    """

    # Index profile measurements by turn for quick lookup.
    # post_profiles correspond to post_measurement_turn (each agent's
    # argmax_idx at that turn). trajectory has more turns.
    profiles_by_turn: dict[int, dict[str, dict]] = {}
    if post_profiles is not None:
        profiles_by_turn[post_measurement_turn] = post_profiles
    for tp in (trajectory or []):
        profiles_by_turn[tp["turn"]] = tp.get("profiles") or {}

    out: list[dict] = []

    for entry in source_dialogue:
        turn = entry["turn"]
        if turn > intervention_turn:
            break
        speaker = (agents_meta.get(entry["agent"]) or {}).get(
            "name", entry["agent"]
        )
        prompt_injected = bool(entry.get("prompt_injected", False))
        # Annotate the prompt-injected turns with the target stage label
        # (this is the dataset's ground-truth stage for those turns).
        argmax_stage = target_stage if prompt_injected else None
        argmax_idx = (
            STAGE_ORDER.index(target_stage)
            if argmax_stage and target_stage in STAGE_ORDER
            else None
        )
        out.append({
            "turn": turn,
            "agent": entry["agent"],
            "speaker": speaker,
            "message": (entry.get("message") or "").strip(),
            "prompt_injected": prompt_injected,
            "argmax_stage": argmax_stage,
            "argmax_idx": argmax_idx,
            "source": "original",
        })

    # Mediator pseudo-entry.
    if mediator_utterance is not None:
        out.append({
            "turn": None,
            "agent": "mediator",
            "speaker": "mediator",
            "message": mediator_utterance.strip(),
            "after_turn": intervention_turn,
            "source": "mediator",
        })

    # Continuation turns; attach argmax measurements from
    # profiles_by_turn when available.
    for nt in continuation_turns or []:
        turn = nt["turn"]
        speaker = (agents_meta.get(nt["agent"]) or {}).get(
            "name", nt["agent"]
        )
        # Find this agent's argmax at this turn (if measured).
        prof = (profiles_by_turn.get(turn) or {}).get(nt["agent"]) or {}
        out.append({
            "turn": turn,
            "agent": nt["agent"],
            "speaker": speaker,
            "message": (nt.get("message") or "").strip(),
            "prompt_injected": False,
            "argmax_stage": prof.get("argmax_stage"),
            "argmax_idx": prof.get("argmax_idx"),
            "source": "continuation",
        })

    return out


def save_dialogue_results(
    output_dir: Path, scenario_id: str, dialogue_result: dict,
) -> None:
    """Save one folder per scenario, one subfolder per model, one JSON per
    path. The per-path JSON contains a simplified dialogue list (with
    per-turn argmax labels) plus minimal metadata (intervention_turn,
    target_stage, mediator_utterance, judge_score, judge_reasoning)."""
    scenario_dir = output_dir / scenario_id
    by_model = dialogue_result.get("by_model") or {}
    source_dialogue = dialogue_result.get("source_dialogue") or []
    agents_meta = dialogue_result.get("agents") or {}
    target_stage = dialogue_result.get("stage") or ""
    pre_profiles = dialogue_result.get("pre_profiles") or {}
    post_meas_turn = dialogue_result.get("post_measurement_turn")

    for model in by_model:
        model_dir = scenario_dir / _model_dirname(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        for path_key, path_label in PATH_FILENAMES.items():
            path_block = by_model[model].get(path_key)
            if path_block is None:
                continue

            mediator_utt = path_block.get("mediation_utterance")
            cont = path_block.get("continuation_turns") or []
            traj = path_block.get("trajectory") or []
            post_profiles = path_block.get("post_profiles")
            intervention_turn = path_block.get("intervention_turn")

            judge = path_block.get("judge") or {}
            judge_score = judge.get("score") if judge else None
            judge_reasoning = judge.get("reasoning") if judge else None
            judge_skip_reason = path_block.get("judge_skip_reason")

            dialogue_view = _build_dialogue_entries(
                source_dialogue=source_dialogue,
                agents_meta=agents_meta,
                intervention_turn=intervention_turn,
                mediator_utterance=mediator_utt,
                continuation_turns=cont,
                pre_profiles=pre_profiles,
                post_profiles=post_profiles,
                post_measurement_turn=post_meas_turn,
                trajectory=traj,
                target_stage=target_stage,
            )

            pe = path_block.get("pair_effect") or {}
            ce = path_block.get("conflict_effect") or {}
            payload = {
                "scenario_id": scenario_id,
                "model": model,
                "path_label": path_label,
                "intervention_turn": intervention_turn,
                "target_stage": target_stage,
                "mediator_utterance": mediator_utt,
                "judge_score": judge_score,
                "judge_reasoning": judge_reasoning,
                "judge_skip_reason": judge_skip_reason,
                "pair_w1": pe.get("signed_wasserstein_1"),
                "conflict_w1": ce.get("signed_wasserstein_1"),
                "dialogue": dialogue_view,
            }
            with open(model_dir / f"{path_label}.json",
                      "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

    # Save original-dialogue baseline at the scenario root (one file).
    ob = dialogue_result.get("original_baseline")
    if ob:
        scenario_dir.mkdir(parents=True, exist_ok=True)
        # For the baseline we have no mediator and no synthetic
        # continuation; the "continuation" is the original dialogue's
        # post-conflict turns.
        post_profiles = ob.get("post_profiles")
        traj = ob.get("trajectory") or []
        # Build dialogue view using the original turns past
        # intervention_turn = pre_measurement_turn (no mediator).
        intervention_turn = dialogue_result.get("pre_measurement_turn")
        # The "continuation_turns" for the baseline is just the original
        # turns past intervention_turn, drawn from source_dialogue.
        baseline_cont = [
            {"turn": e["turn"], "agent": e["agent"], "message": e["message"]}
            for e in source_dialogue if e["turn"] > intervention_turn
        ]
        dialogue_view = _build_dialogue_entries(
            source_dialogue=source_dialogue,
            agents_meta=agents_meta,
            intervention_turn=intervention_turn,
            mediator_utterance=None,
            continuation_turns=baseline_cont,
            pre_profiles=pre_profiles,
            post_profiles=post_profiles,
            post_measurement_turn=post_meas_turn,
            trajectory=traj,
            target_stage=target_stage,
        )
        with open(scenario_dir / "original_dialogue.json",
                  "w", encoding="utf-8") as f:
            json.dump({
                "scenario_id": scenario_id,
                "target_stage": target_stage,
                "intervention_turn": None,
                "mediator_utterance": None,
                "dialogue": dialogue_view,
            }, f, indent=2, ensure_ascii=False)


def save_summary(
    output_dir: Path, args, all_results: dict, summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "models": args.models,
                "sim_model": args.sim_model,
                "judge_model": args.judge_model if args.judge else None,
                "data_dir": args.data_dir,
                "utils_path": args.utils_path,
                "stage_judge_method": "llm_logprob (6-digit token logprob softmax)",
                "stages": STAGE_ORDER,
                "continuation_temperature": CONTINUATION_TEMPERATURE,
                "mediator_in_speaker_history": True,
                "pre_measurement_rule": "conflict_turn",
                "post_measurement_rule": "conflict_turn + 3 (clipped to n_turns)",
                "trajectory_measurement_rule":
                    "every turn from conflict_turn + 4 to n_turns",
                "acceptance_threshold_idx": ACCEPTANCE_IDX_THRESHOLD,
                "original_baseline_model": ORIGINAL_BASELINE_MODEL,
            },
            "scenario_ids": sorted(all_results.keys()),
            "results": all_results,
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--utils-path", default=DEFAULT_UTILS_PATH)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sim-model", default=DEFAULT_SIM_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--no-judge", dest="judge", action="store_false",
        help="Skip the LLM-judge definition-adherence rating.",
    )
    parser.set_defaults(judge=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument(
        "--start", type=int, default=None, metavar="ID",
        help="Start from this scenario number (filename stem must be "
             "an integer; e.g. --start 41 begins at 41.json and skips "
             "1-40). Combined with --limit to bound the upper end. "
             "Ignored if --ids is given.",
    )
    parser.add_argument(
        "--human-eval-only", action="store_true",
        help="Restrict evaluation to the 160 scenario IDs that humans "
             "rated in data/eval_pool.json. Overrides --ids / --start / "
             "--limit when supplied.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip scenarios whose summary entry already exists under "
             "<output-dir>/<scenario_id>/ (i.e. at least one model "
             "subfolder is present). Useful for resuming a partially-"
             "completed evaluation run without re-paying API costs.",
    )
    parser.add_argument(
        "--output-dir", default="mediation_effectiveness_results",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-predicted-path", dest="predicted_path", action="store_false",
    )
    parser.add_argument(
        "--no-predicted-def-path", dest="predicted_def_path",
        action="store_false",
    )
    parser.add_argument(
        "--no-gt-path", dest="gt_path", action="store_false",
    )
    parser.add_argument(
        "--no-gt-def-path", dest="gt_def_path", action="store_false",
    )
    parser.add_argument(
        "--random-path", dest="random_path", action="store_true",
        help="Run mediation with RANDOM turn after GT, no definition.",
    )
    parser.add_argument(
        "--random-def-path", dest="random_def_path", action="store_true",
        help="Run mediation with RANDOM turn after GT, with definition.",
    )
    parser.add_argument(
        "--no-original-baseline", dest="original_baseline",
        action="store_false",
    )
    parser.add_argument(
        "--random-pred-after-gt", action="store_true", default=False,
        help="Instead of streaming-detecting the predicted conflict turn, "
             "pick a RANDOM turn strictly after the GT conflict_turn for the "
             "predicted path. Useful for ablations / preference-data generation.",
    )
    parser.set_defaults(
        predicted_path=True,
        predicted_def_path=True,
        gt_path=True,
        gt_def_path=True,
        random_path=False,
        random_def_path=False,
        original_baseline=True,
    )
    args = parser.parse_args()

    if args.random_pred_after_gt:
        os.environ["RANDOM_PRED_AFTER_GT"] = "1"
        print("[random-pred-after-gt] enabled: predicted_turn will be "
              "a RANDOM int in (gt_turn, n_turns-1].")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY environment variable")

    conflict_table = None
    needs_def = (
        args.predicted_def_path or args.gt_def_path or args.judge
    )
    if needs_def:
        try:
            conflict_table = load_conflict_table(args.utils_path)
            print(f"Loaded CONFLICT_TABLE from {args.utils_path}")
        except Exception as e:
            sys.exit(
                f"ERROR loading CONFLICT_TABLE: {e}\n"
                f"Set --utils-path correctly, or run with "
                f"--no-predicted-def-path --no-gt-def-path --no-judge."
            )

    # --human-eval-only takes precedence: load scenario_ids from the
    # human-evaluation pool and use them as the explicit --ids list.
    if args.human_eval_only:
        pool_path = Path(args.data_dir).parent / "eval_pool.json"
        if not pool_path.exists():
            sys.exit(f"ERROR: --human-eval-only set but {pool_path} missing")
        pool = json.load(open(pool_path, encoding="utf-8"))
        args.ids = [it["scenario_id"] for it in pool["items"]]
        print(
            f"--human-eval-only: restricting to {len(args.ids)} scenarios "
            f"from {pool_path}"
        )

    files = find_files(args.data_dir, args.ids)

    # --start filters by scenario id (>= start). Ignored if --ids is
    # given, since --ids already specifies an explicit selection.
    if args.start is not None and not args.ids:
        def _scenario_id_int(p: Path) -> int:
            digits = re.sub(r"\D", "", p.stem)
            return int(digits) if digits else 10**9
        before = len(files)
        files = [f for f in files if _scenario_id_int(f) >= args.start]
        print(f"--start {args.start}: {len(files)}/{before} files remain.")

    if args.limit:
        files = files[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --skip-existing: drop scenarios whose output folder already
    # contains at least one model subfolder. We treat the presence of
    # any model subfolder as evidence that this scenario was processed
    # in a previous run.
    if args.skip_existing:
        before = len(files)
        kept: list[Path] = []
        for fp in files:
            scen_dir = output_dir / fp.stem
            already_done = (
                scen_dir.exists()
                and any(p.is_dir() for p in scen_dir.iterdir())
            )
            if not already_done:
                kept.append(fp)
        skipped = before - len(kept)
        files = kept
        if skipped:
            print(
                f"--skip-existing: skipped {skipped} already-evaluated "
                f"scenario(s); {len(files)} remain."
            )

    if not files:
        sys.exit(f"No dialogue files to evaluate (after filtering).")

    print(f"Evaluating {len(files)} dialogues x {len(args.models)} models")
    print(f"Sim model:   {args.sim_model}")
    print(f"Judge model: {args.judge_model if args.judge else 'OFF'}")
    print(f"Pre = conflict_turn   Post = conflict_turn + 3   "
          f"Trajectory through n_turns")
    print(f"Acceptance threshold = idx >= {ACCEPTANCE_IDX_THRESHOLD} "
          f"(both agents must exceed)")
    print(f"Original baseline reported on: {ORIGINAL_BASELINE_MODEL}")
    print(f"Saving to: {output_dir}")
    print()

    judge_model = args.judge_model if args.judge else None

    all_results: dict[str, dict] = {}
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}", flush=True)
        try:
            res = evaluate_dialogue(
                fp, args.models, args.sim_model, judge_model, api_key,
                do_predicted_path=args.predicted_path,
                do_predicted_def_path=args.predicted_def_path,
                do_gt_def_path=args.gt_def_path,
                do_gt_path=args.gt_path,
                do_random_path=args.random_path,
                do_random_def_path=args.random_def_path,
                do_original_baseline=args.original_baseline,
                conflict_table=conflict_table,
                max_workers=args.workers,
            )
        except Exception as e:
            print(f"   FAILED: {e}", file=sys.stderr)
            continue
        if res is None:
            print(f"   skipped (missing data / pre measurement infeasible)")
            continue
        all_results[fp.stem] = res

        gt = res["ground_truth_turn"]
        ob = res.get("original_baseline")
        # Pre stage labels (same across paths)
        pre_a1 = (res["pre_profiles"].get("agent_1") or {}).get("argmax_stage")
        pre_a2 = (res["pre_profiles"].get("agent_2") or {}).get("argmax_stage")
        pre_a1_idx = (res["pre_profiles"].get("agent_1") or {}).get("argmax_idx")
        pre_a2_idx = (res["pre_profiles"].get("agent_2") or {}).get("argmax_idx")
        target_stage = res.get("stage", "?")
        print(
            f"   stage={target_stage:<13} pre: a1={pre_a1}({pre_a1_idx}) "
            f"a2={pre_a2}({pre_a2_idx})"
        )

        def _path_line(label: str, p: Optional[dict],
                       baseline: bool = False) -> str:
            """Render one mediation path's diagnostics as a compact one-liner.

            Shows: per-agent post argmax stages (idx -> idx), combined
            progress score, group W1 distribution shift, regression flag,
            judge score (or skip reason marker)."""
            if p is None:
                return f"     {label:<14} (skipped)"
            diag = p.get("diagnostics") or {}
            if not diag:
                reason = p.get("skip_reason") or p.get("continuation_error")
                return f"     {label:<14} (no diagnostic) {reason or ''}"
            post_a1 = (diag.get("post_argmax_stage_per_agent") or {}).get("agent_1")
            post_a2 = (diag.get("post_argmax_stage_per_agent") or {}).get("agent_2")
            post_a1_idx = (diag.get("post_argmax_per_agent") or {}).get("agent_1")
            post_a2_idx = (diag.get("post_argmax_per_agent") or {}).get("agent_2")
            t_auc = diag.get("trajectory_auc")
            c_auc = diag.get("conflict_trajectory_auc")
            pe = p.get("pair_effect") or {}
            pair_w1 = pe.get("signed_wasserstein_1")
            ce = p.get("conflict_effect") or {}
            conflict_w1 = ce.get("signed_wasserstein_1")
            jr = (p.get("judge") or {}).get("score") if not baseline else None
            jskip = p.get("judge_skip_reason") if not baseline else None

            def _ff(v):
                return f"{v:+.2f}" if isinstance(v, (int, float)) else "  ?"

            if jr is not None:
                j_str = f"  judge={jr}"
            elif jskip:
                j_str = "  judge=-"
            else:
                j_str = ""

            return (
                f"     {label:<14} "
                f"a1:{pre_a1_idx}->{post_a1_idx}({post_a1[:3] if post_a1 else '?'})  "
                f"a2:{pre_a2_idx}->{post_a2_idx}({post_a2[:3] if post_a2 else '?'})  "
                f"cAUC={_ff(c_auc)} pAUC={_ff(t_auc)}  "
                f"cW1={_ff(conflict_w1)} pW1={_ff(pair_w1)}"
                f"{j_str}"
            )

        for model in args.models:
            r = res["by_model"].get(model)
            if r is None:
                continue
            tp = r.get("turn_prediction") or {}
            pt = tp.get("predicted_turn")
            t_str = (
                "FAIL" if pt is None
                else (f"{pt}o" if tp.get("correct") else f"{pt}x")
            )
            print(
                f"   {model:<32} turn_pred={t_str:>5} (gt={gt})"
            )
            print(_path_line("[predicted]",     r.get("predicted_path")))
            print(_path_line("[predicted+d]",   r.get("predicted_def_path")))
            print(_path_line("[gt]",            r.get("gt_path")))
            print(_path_line("[gt+def]",        r.get("gt_def_path")))
            print(_path_line("[random]",        r.get("random_path")))
            print(_path_line("[random+def]",    r.get("random_def_path")))

        # Original-dialogue baseline (one row, only if present)
        if ob:
            print(_path_line("[ORIGINAL]", ob, baseline=True))

        save_dialogue_results(output_dir, fp.stem, res)
        summary = aggregate_stats(all_results, args.models)
        save_summary(output_dir, args, all_results, summary)

    elapsed = time.time() - t0
    summary = aggregate_stats(all_results, args.models)
    print_summary(summary, args.models)
    print(f"\nElapsed: {elapsed:.1f}s   Saved: {output_dir}/")


if __name__ == "__main__":
    main()