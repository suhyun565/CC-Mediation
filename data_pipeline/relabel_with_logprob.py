"""
relabel_with_logprob.py
=======================

Walk every CC_labeled/*.json, drop the obsolete DMIS-Likert fields, and
write a NEW continuation-turn label using the logprob-based LLM judge
in llm_stage_judge.py.

After running:
  - Each continuation turn carries:
      label                : LLM judge argmax stage (one of the 6 stages)
      label_source         : "llm_judge_logprob"
      distribution         : 6-vector probabilities
      max_prob, entropy    : confidence diagnostics
      argmax_stage         : same as label (kept for backward compat)
      ambiguous            : recomputed from new max_prob/entropy
      raw_logprobs         : per-digit logprob for debugging
  - Pre-conflict turn rows are left untouched (LLM judge already labels
    them as unrelated_topic / transition).
  - Conflict turn rows keep label = target_stage as before.
  - All DMIS-Likert leftovers are removed:
      dmis_distribution, dmis_stage_means, llm_judge_label,
      llm_judge_reason, responses, raw_response.

Default model: openai/gpt-4o-mini (logprobs supported).
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
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
from dotenv import load_dotenv
load_dotenv(f"{REPO}/.env")

from llm_stage_judge import (
    measure_stage_via_logprob, STAGE_ORDER,
)

LABELED_DIR = Path(f"{REPO}/data/CC_labeled")
DIALOGUES_DIR = Path(f"{REPO}/data/CC_dialogues")

AMBIGUOUS_MAX_PROB = 0.30
AMBIGUOUS_ENTROPY = 1.65

# Fields that should not appear on continuation rows after re-labeling.
DROP_FIELDS = (
    "dmis_distribution", "dmis_stage_means", "llm_judge_label",
    "llm_judge_reason", "responses", "raw_response",
)


def speaker_name(agent_prompts: dict, agent_id: str) -> str:
    info = agent_prompts.get(agent_id) or {}
    bg = (info.get("background") or "").strip()
    if bg:
        head = bg.split(":", 1)[0].split(" is ", 1)[0].strip()
        if head:
            return head
        return bg.split()[0]
    return agent_id


def build_history_for_turn(dialogue: list[dict], turn_n: int,
                           agent_prompts: dict, conflict_agent: str) -> str:
    rows = []
    for t in dialogue:
        if t["turn"] > turn_n:
            break
        sp = speaker_name(agent_prompts, t["agent"])
        role = ("conflict speaker"
                if t["agent"] == conflict_agent else "non-conflict speaker")
        rows.append(f"Turn {t['turn']} ({sp}, {role}): {t['message'].strip()}")
    return "\n".join(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--judge-model", default="openai/gpt-4o-mini")
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rejudge", action="store_true",
                   help="Re-run even if continuation already has llm_judge_logprob source.")
    p.add_argument("--labeled-dir", default=str(LABELED_DIR),
                   help="Directory with labeled JSON files (read+write).")
    p.add_argument("--dialogues-dir", default=str(DIALOGUES_DIR),
                   help="Directory with source dialogue JSON files.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")

    labeled_dir = Path(args.labeled_dir)
    dialogues_dir = Path(args.dialogues_dir)
    sids = sorted([p.stem for p in labeled_dir.glob("*.json")
                   if not p.name.startswith("_")],
                  key=lambda s: int(s) if s.isdigit() else 1e9)
    if args.limit:
        sids = sids[: args.limit]

    work: list[tuple] = []
    cache_lab: dict[str, dict] = {}
    cache_dlg: dict[str, dict] = {}

    for sid in sids:
        lab_p = labeled_dir / f"{sid}.json"
        dlg_p = dialogues_dir / f"{sid}.json"
        if not (lab_p.exists() and dlg_p.exists()):
            continue
        lab = json.load(open(lab_p, encoding="utf-8"))
        dlg = json.load(open(dlg_p, encoding="utf-8"))
        cache_lab[sid] = lab
        cache_dlg[sid] = dlg
        cag = lab.get("conflict_agent") or dlg.get("conflict_agent")
        for r in lab["labeled_dialogue"]:
            if r["phase"] != "continuation":
                continue
            if (not args.rejudge
                    and r.get("label_source") == "llm_judge_logprob"
                    and r.get("distribution")):
                continue
            work.append((sid, r["turn"], dlg, cag))

    print(f"Continuation turns to (re-)judge: {len(work)} (model={args.judge_model})")

    def _go(item):
        sid, tn, dlg, cag = item
        agent_prompts = dlg.get("agent_prompts") or {}
        target_row = next((t for t in dlg["dialogue"] if t["turn"] == tn), None)
        if target_row is None:
            return sid, tn, None, "turn_missing"
        history = build_history_for_turn(dlg["dialogue"], tn,
                                         agent_prompts, cag)
        sp = speaker_name(agent_prompts, target_row["agent"])
        res = measure_stage_via_logprob(
            args.judge_model, history, api_key,
            marked_speaker=sp,
            marked_turn_text=target_row["message"],
        )
        return sid, tn, res, res.get("error")

    results: dict[tuple[str, int], dict] = {}
    errors: list[tuple] = []
    if work:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_go, w): w for w in work}
            done = 0
            for f in as_completed(futs):
                sid, tn, res, err = f.result()
                done += 1
                if err:
                    errors.append((sid, tn, err))
                else:
                    results[(sid, tn)] = res
                if done % 100 == 0:
                    print(f"  ... {done}/{len(work)}")

    # Apply results: rewrite each continuation row, drop DMIS-leftover fields.
    n_files_written = 0
    for sid, lab in cache_lab.items():
        changed = False
        for r in lab["labeled_dialogue"]:
            if r["phase"] != "continuation":
                continue
            for f in DROP_FIELDS:
                if f in r:
                    r.pop(f, None)
                    changed = True
            key = (sid, r["turn"])
            res = results.get(key)
            if not res or res.get("argmax_idx") is None:
                # Leave existing label_source if the call failed.
                continue
            mp = float(res["max_prob"])
            ent = float(res["entropy"])
            ambiguous = (mp < AMBIGUOUS_MAX_PROB) or (ent >= AMBIGUOUS_ENTROPY)
            r["label"] = "ambiguous" if ambiguous else res["argmax_stage"]
            r["label_source"] = "llm_judge_logprob"
            r["argmax_stage"] = res["argmax_stage"]
            r["distribution"] = res["distribution"]
            r["max_prob"] = mp
            r["entropy"] = ent
            r["ambiguous"] = ambiguous
            r["raw_logprobs"] = res["raw_logprobs"]
            r.pop("label_error", None)
            changed = True
        if changed:
            with open(labeled_dir / f"{sid}.json", "w", encoding="utf-8") as f:
                json.dump(lab, f, ensure_ascii=False, indent=2)
            n_files_written += 1

    print(f"\nDone. judged={len(results)} turns, "
          f"updated={n_files_written} files, errors={len(errors)}.")
    if errors:
        print(f"First 5 errors: {errors[:5]}")


if __name__ == "__main__":
    main()
