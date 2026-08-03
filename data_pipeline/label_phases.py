"""
label_phases.py
===============

Per-turn phase / stage labeling for generated dialogues, designed to
test build_agent_prompts.py + generate.py output.

Labeling rule
-------------
For each turn in dialogue:
  * turn < conflict_turn          -> LLM judge call:
                                       'unrelated_topic'  (purely goal_1)
                                       'transition'       (goal_2 engaged)
  * turn == conflict_turn         -> the SCENARIO's target DMIS stage
                                     (one of Denial / Defense / Minimization)
                                     taken directly from agent_prompts.dmis.stage
  * turn  > conflict_turn         -> per-turn DMIS measurement
                                     argmax over the 6 DMIS stages
                                     (Denial, Defense, Minimization,
                                      Acceptance, Adaptation, Integration)

Reuses helpers from evaluate_mediation_effectiveness.py:
  - build_initial_histories
  - histories_to_plaintext
  - measure_dmis_for_speaker
  - call_simple

Inputs
------
  <repo>/data/CC_dialogues/*.json

Outputs
-------
  <repo>/data/CC_labeled/{sid}.json
  <repo>/data/CC_labeled/_summary.json

Usage
-----
  python label_phases.py
  python label_phases.py -n 30 --judge-model openai/gpt-4o-mini
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
import json
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reach evaluate_mediation_effectiveness.py at repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared"))
sys.path.insert(0, str(_REPO / "metrics"))

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

from evaluate_mediation_effectiveness import (  # noqa: E402
    build_initial_histories,
    histories_to_plaintext,
    call_simple,
    _argmax_idx,
)
from llm_stage_judge import (  # noqa: E402
    measure_stage_via_logprob,
    STAGE_ORDER,
)
import math  # noqa: E402


DIALOGUES_DIR = _REPO / "data" / "CC_dialogues"
OUTPUT_DIR = _REPO / "data" / "CC_labeled"

# Distribution-flatness thresholds for the "ambiguous" label.
# A DMIS distribution over 6 stages has max entropy log(6) ~= 1.792.
# When the rater cannot decide a stage, the distribution is nearly
# uniform; we flag those cases as ambiguous so they don't pollute the
# argmax-based downstream stats.
AMBIGUOUS_MAX_PROB = 0.30   # if argmax probability < this -> ambiguous
AMBIGUOUS_ENTROPY = 1.65    # if entropy >= this (>= 92% of max) -> ambiguous


# ---------------------------------------------------------------------------
# Phase-label LLM judge for pre-conflict turns
# ---------------------------------------------------------------------------

PHASE_SYSTEM_PROMPT = """You are an annotator that labels each turn of a two-speaker dialogue with a phase tag.

There are exactly TWO possible labels for a pre-conflict turn:

  unrelated_topic
    The turn stays entirely on goal_1 (the cordial / logistical opener,
    typically planning a gift, party, meeting time, or similar non-
    cultural topic). The cultural-value disagreement (goal_2) is NOT
    raised, hinted at, or engaged with.

  transition
    The turn engages with goal_2 in any way - the speaker introduces
    the cultural-value topic, asks about it, hints at it, or otherwise
    moves the conversation toward the cultural disagreement. Even a
    single sentence that names or alludes to goal_2 makes the turn a
    transition.

Respond with ONLY a single JSON object, no prose, no markdown fences:
{"label": "unrelated_topic" | "transition", "reason": "<one short sentence>"}
"""

PHASE_USER_PROMPT_TEMPLATE = """Goal_1 (cordial topic):
"{goal_1}"

Goal_2 (cultural-value topic that triggers the conflict):
"{goal_2}"

Dialogue so far (for context):
{prefix}

The turn to label is Turn {turn_n} ({speaker}):
"{message}"

Classify this turn. Respond with only:
{{"label": "unrelated_topic" | "transition", "reason": "<short>"}}"""


def _strip_fences(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def parse_phase_response(raw: str) -> dict:
    if not raw:
        return {"label": None, "reason": None, "raw": raw}
    s = _strip_fences(raw)
    try:
        obj = json.loads(s)
    except Exception:
        # Try to grab the first JSON object substring.
        m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
        if not m:
            return {"label": None, "reason": None, "raw": raw}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {"label": None, "reason": None, "raw": raw}
    label = (obj.get("label") or "").strip().lower()
    if label not in ("unrelated_topic", "transition"):
        label = None
    return {
        "label": label,
        "reason": (obj.get("reason") or "").strip(),
        "raw": raw,
    }


def _format_prefix(dialogue_entries: list[dict], up_to_turn_exclusive: int) -> str:
    lines = []
    for e in dialogue_entries:
        if e["turn"] >= up_to_turn_exclusive:
            break
        a = e.get("speaker") or e.get("agent")
        msg = (e.get("message") or "").strip().replace("\n", " ")
        lines.append(f"  Turn {e['turn']} ({a}): {msg}")
    return "\n".join(lines) if lines else "  (no prior turns)"


def label_phase_turn(
    judge_model: str,
    api_key: str,
    *,
    goal_1: str,
    goal_2: str,
    dialogue_entries: list[dict],
    turn_n: int,
    speaker: str,
    message: str,
) -> dict:
    user = PHASE_USER_PROMPT_TEMPLATE.format(
        goal_1=goal_1.strip(),
        goal_2=goal_2.strip(),
        prefix=_format_prefix(dialogue_entries, turn_n),
        turn_n=turn_n,
        speaker=speaker,
        message=(message or "").strip(),
    )
    raw, err = call_simple(
        judge_model, PHASE_SYSTEM_PROMPT, user, api_key, max_tokens=120,
    )
    parsed = parse_phase_response(raw or "")
    parsed["error"] = err
    return parsed


# ---------------------------------------------------------------------------
# Per-turn DMIS for continuation turns
# ---------------------------------------------------------------------------

def measure_turn_dmis(
    sim_model: str,
    api_key: str,
    *,
    speaker_id: str,
    speaker_name: str,
    dialogue_entries: list[dict],
    agent_prompts: dict,
    turn_n: int,
) -> dict:
    """Classify the marked turn's DMIS stage with the logprob LLM judge.

    Builds per-agent histories up to and including ``turn_n`` (so the
    speaker's just-uttered message is in their assistant slot), then
    invokes ``measure_stage_via_logprob`` to obtain a calibrated 6-stage
    distribution. The ``ambiguous`` flag is recomputed from the new
    distribution's max_prob / entropy.
    """
    histories = build_initial_histories(
        dialogue_entries, agent_prompts, up_to_turn=turn_n,
    )
    history_text = histories_to_plaintext(histories[speaker_id])
    target_msg = next(
        (t.get("message") for t in dialogue_entries if t["turn"] == turn_n),
        None,
    )
    res = measure_stage_via_logprob(
        sim_model, history_text, api_key,
        marked_speaker=speaker_name,
        marked_turn_text=target_msg,
    )
    distribution = res.get("distribution")
    if distribution is None:
        return {
            "argmax_idx": None, "argmax_stage": None, "label": None,
            "ambiguous": None, "max_prob": None, "entropy": None,
            "distribution": None, "raw_logprobs": None,
            "error": res.get("error"),
        }
    idx = res["argmax_idx"]
    max_prob = res["max_prob"]
    entropy = res["entropy"]
    ambiguous = (max_prob < AMBIGUOUS_MAX_PROB) or (entropy >= AMBIGUOUS_ENTROPY)
    label_value = "ambiguous" if ambiguous else STAGE_ORDER[idx]
    return {
        "argmax_idx": idx,
        "argmax_stage": STAGE_ORDER[idx],
        "label": label_value,
        "ambiguous": ambiguous,
        "max_prob": max_prob,
        "entropy": entropy,
        "distribution": distribution,
        "raw_logprobs": res.get("raw_logprobs"),
        "error": res.get("error"),
    }


# ---------------------------------------------------------------------------
# Main labeler
# ---------------------------------------------------------------------------

def speaker_name_from_prompt(agent_prompt: dict, fallback: str) -> str:
    bg = (agent_prompt.get("background") or "").strip()
    if bg:
        head = bg.split(":", 1)[0].split(" is ", 1)[0].strip()
        if head:
            return head
        return bg.split()[0]
    return fallback


def label_dialogue(
    data: dict,
    judge_model: str,
    sim_model: str,
    api_key: str,
    max_workers: int = 8,
) -> dict:
    """Return labeled dialogue dict with per-turn 'phase_label' field."""
    conflict_turn = int(data["conflict_turn"])
    target_stage = (data.get("dmis") or {}).get("stage")
    agent_prompts = data["agent_prompts"]
    dialogue = data["dialogue"]

    # Goals come from agent_prompts (each agent has goal_1, goal_2).
    goals_by_agent = {
        aid: (info.get("goals") or {}) for aid, info in agent_prompts.items()
    }

    speaker_names = {
        aid: speaker_name_from_prompt(info, aid)
        for aid, info in agent_prompts.items()
    }

    # Per-turn worker.
    def _label_one(entry):
        turn = entry["turn"]
        aid = entry["agent"]
        msg = entry["message"]
        sp_name = speaker_names.get(aid, aid)
        out = {
            "turn": turn,
            "agent": aid,
            "speaker": sp_name,
            "message": msg,
            "prompt_injected": entry.get("prompt_injected", False),
        }
        if turn < conflict_turn:
            g = goals_by_agent.get(aid) or {}
            res = label_phase_turn(
                judge_model, api_key,
                goal_1=g.get("goal_1", ""),
                goal_2=g.get("goal_2", ""),
                dialogue_entries=dialogue,
                turn_n=turn,
                speaker=sp_name,
                message=msg,
            )
            out["phase"] = "pre_conflict"
            out["label"] = res["label"]
            out["label_source"] = "llm_judge"
            out["label_reason"] = res.get("reason")
            out["label_error"] = res.get("error")
        elif turn == conflict_turn:
            out["phase"] = "conflict"
            out["label"] = target_stage
            out["label_source"] = "target_stage"
        else:
            out["phase"] = "continuation"
            res = measure_turn_dmis(
                sim_model, api_key,
                speaker_id=aid,
                speaker_name=sp_name,
                dialogue_entries=dialogue,
                agent_prompts=agent_prompts,
                turn_n=turn,
            )
            # ``label`` is the human-facing tag; if the DMIS distribution
            # is too flat to decide, it becomes "ambiguous". The raw
            # argmax stage is preserved separately for analysis.
            out["label"] = res.get("label")
            out["label_source"] = "llm_judge_logprob"
            out["argmax_stage"] = res.get("argmax_stage")
            out["ambiguous"] = res.get("ambiguous")
            out["max_prob"] = res.get("max_prob")
            out["entropy"] = res.get("entropy")
            out["distribution"] = res.get("distribution")
            out["raw_logprobs"] = res.get("raw_logprobs")
            out["label_error"] = res.get("error")
        return out

    # Parallel within a single dialogue: each turn is independent.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_label_one, e): e for e in dialogue}
        labeled = [None] * len(dialogue)
        for fut in as_completed(futures):
            entry = futures[fut]
            idx = next(i for i, e in enumerate(dialogue) if e is entry)
            try:
                labeled[idx] = fut.result()
            except Exception as e:
                labeled[idx] = {
                    "turn": entry["turn"], "agent": entry["agent"],
                    "message": entry["message"],
                    "phase": "error", "label": None,
                    "label_error": str(e),
                }

    return {
        "scenario_id": data.get("scenario_id") or None,
        "conflict_turn": conflict_turn,
        "intervention_turn": data.get("intervention_turn"),
        "conflict_agent": data.get("conflict_agent"),
        "target_stage": target_stage,
        "agent_models": data.get("agent_models"),
        "labeled_dialogue": labeled,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-n", "--count", type=int, default=None)
    p.add_argument("--input-dir", default=str(DIALOGUES_DIR))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--sim-model", default="openai/gpt-4o-mini",
                   help="Model used as the DMIS rater for continuation turns.")
    p.add_argument("--max-turn-workers", type=int, default=8,
                   help="Parallel turns within a single dialogue.")
    p.add_argument("--max-dialogue-workers", type=int, default=4,
                   help="Parallel dialogues at the top level.")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _scenario_id_from_path(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0]
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits else 10**9


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; put it in .env")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"), key=lambda p: _scenario_id_from_path(str(p)))
    if args.count is not None:
        files = files[: args.count]

    if args.skip_existing:
        before = len(files)
        files = [f for f in files if not (out_dir / f.name).exists()]
        if before != len(files):
            print(f"  skipped {before-len(files)} already-labeled.")

    print(f"Labeling {len(files)} dialogue(s) -> {out_dir}")

    summary_rows = []

    def _process(fp: Path):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        sid = fp.stem
        data.setdefault("scenario_id", sid)
        labeled = label_dialogue(
            data,
            judge_model=args.judge_model,
            sim_model=args.sim_model,
            api_key=api_key,
            max_workers=args.max_turn_workers,
        )
        labeled["scenario_id"] = sid
        with open(out_dir / fp.name, "w", encoding="utf-8") as f:
            json.dump(labeled, f, ensure_ascii=False, indent=2)
        return labeled

    with ThreadPoolExecutor(max_workers=args.max_dialogue_workers) as ex:
        futs = {ex.submit(_process, fp): fp for fp in files}
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                labeled = fut.result()
                summary_rows.append({
                    "scenario_id": labeled["scenario_id"],
                    "target_stage": labeled["target_stage"],
                    "conflict_turn": labeled["conflict_turn"],
                    "labels": [
                        {"turn": t["turn"], "phase": t["phase"], "label": t.get("label")}
                        for t in labeled["labeled_dialogue"]
                    ],
                })
                print(f"  ✓ {fp.name}")
            except Exception as e:
                print(f"  ✗ {fp.name}: {e}")
                summary_rows.append({
                    "scenario_id": fp.stem, "error": str(e),
                })

    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"n": len(summary_rows), "rows": summary_rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nWrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
