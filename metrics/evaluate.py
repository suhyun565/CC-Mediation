"""evaluate.py — score a completed dialogue (mediator utterance already
injected) with the paper's mediation metrics.

Difference from `eval_base_auc_w1.py` (Part 2):

* `eval_base_auc_w1.py` takes `(sid, predicted_turn, utterance)` predictions,
  **re-simulates** the continuation with an LLM sim actor, then measures.
* This script takes the **completed dialogue** — the continuation is
  already there, with the mediator utterance already inserted at the
  intervention turn — and only runs the DMIS labeler / judge.

Use this when you generated the continuation yourself (any sim, any decoding)
and just want the paper's `AUC / signed W1 / Judge` under identical formulas.

------------------------------------------------------------------------
INPUT — a JSON list (or object with `records` field). Each record needs:

    {
      "sid":                str,
      "conflict_turn":      int,     # required (PRE reference point)
      "conflict_agent":     str,     # agent id whose stage we track
      "target_stage":       str,     # one of Denial/Defense/Minimization
      "mediator_turn":      int,     # the turn AT WHICH the mediator spoke
      "mediator_utterance": str,     # the mediator's utterance itself
      "n_turns":            int,     # last turn number in the continuation
      "agent_prompts":      dict,    # same schema as data/CC_dialogues/*.json
      "dialogue":           list,    # original 1..n_turns turns, before
                                     # mediator injection (used for PRE)
      "continuation":       list     # continuation turns (turns
                                     # mediator_turn+1 .. n_turns) that
                                     # ARE the model-generated response
                                     # to the mediator. Each entry:
                                     #   {"turn": int, "agent": str, "message": str}
    }

If your data has the continuation already merged into `dialogue`, split it:
  * `dialogue` = turns 1..mediator_turn  (original conflict conversation)
  * `continuation` = turns mediator_turn+1..n_turns  (post-mediator generated)

------------------------------------------------------------------------
OUTPUT — same schema as `eval_base_auc_w1.py`:
    {"summary": {...}, "per_sid": [...]}
"""
from __future__ import annotations

# --- release bootstrap ---
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
import statistics as st
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from evaluate_mediation_effectiveness import (
    build_initial_histories,
    truncate_histories_to_turn,
    measure_per_speaker_distributions,
    agent_metadata,
    run_judge,
)
from dmis_distribution import compare_distributions

# Paige et al. (2003) developmental positions — identical to eval_base_auc_w1
STAGE_POSITIONS = [-3, -2, -1, +1, +2, +3]


def trajectory_auc(pre_idx: int,
                   trajectory: list[tuple[int, int]],
                   pred_turn: int,
                   n_turns: int) -> Optional[float]:
    duration = n_turns - pred_turn
    if duration <= 0 or not trajectory:
        return None
    points = []
    for tt, ax in trajectory:
        tau = max(0.0, min(1.0, (tt - pred_turn) / duration))
        delta = STAGE_POSITIONS[ax] - STAGE_POSITIONS[pre_idx]
        points.append((tau, delta))
    points.sort(key=lambda x: x[0])
    if len(points) == 1:
        return float(points[0][1])
    auc = 0.0
    tf, pf = points[0]
    if tf > 0:
        auc += pf * tf
    for i in range(len(points) - 1):
        ta, pa = points[i]
        tb, pb = points[i + 1]
        auc += (pa + pb) / 2.0 * (tb - ta)
    tl, pl = points[-1]
    if tl < 1.0:
        auc += pl * (1.0 - tl)
    return float(auc)


def _stats(xs):
    if not xs:
        return {"n": 0}
    return {
        "n":      len(xs),
        "mean":   st.mean(xs),
        "median": st.median(xs),
        "min":    min(xs),
        "max":    max(xs),
        "std":    st.stdev(xs) if len(xs) > 1 else 0.0,
    }


# CONFLICT_TABLE lives in shared/utils.py; imported lazily to avoid a
# hard dependency at module load time (the judge is optional).
def _load_conflict_table():
    from utils import CONFLICT_TABLE
    return CONFLICT_TABLE


def score_one(rec: dict, sim_model: str, judge_model: str,
              api_key: str, skip_judge: bool = False) -> dict:
    """Compute AUC / W1 / Judge for a single completed-dialogue record."""
    sid              = str(rec.get("sid"))
    conflict_turn    = int(rec["conflict_turn"])
    conflict_agent   = rec["conflict_agent"]
    target_stage     = rec.get("target_stage", "")
    mediator_turn    = int(rec["mediator_turn"])
    mediator_utt     = rec.get("mediator_utterance") or ""
    n_turns          = int(rec["n_turns"])
    agent_prompts    = rec["agent_prompts"]
    dialogue_entries = rec["dialogue"]
    continuation     = rec.get("continuation", [])

    out = {"sid": sid, "mediator_turn": mediator_turn,
           "conflict_turn": conflict_turn}

    # Build the temporary dict shape our helpers expect (agent_prompts +
    # agents_meta come from evaluate_mediation_effectiveness.agent_metadata).
    fake_data = {"agent_prompts": agent_prompts}
    agents_meta = agent_metadata(fake_data)

    # --- PRE at conflict_turn (original dialogue, no mediator) ---
    pre_hist = build_initial_histories(
        dialogue_entries, agent_prompts, conflict_turn,
    )
    pre = measure_per_speaker_distributions(
        histories={conflict_agent: pre_hist[conflict_agent]},
        agents_meta=agents_meta,
        sim_model=sim_model, api_key=api_key, max_workers=1,
    )
    pre_idx  = ((pre or {}).get(conflict_agent) or {}).get("argmax_idx")
    pre_dist = ((pre or {}).get(conflict_agent) or {}).get("distribution")
    if pre_idx is None:
        out["skip"] = "pre_none"
        return out
    out["pre_idx"] = pre_idx

    # --- POST at min(conflict_turn + 3, n_turns), using continuation ---
    post_target = min(conflict_turn + 3, n_turns)
    post_hist = truncate_histories_to_turn(
        full_histories=None,     # rebuilt from scratch inside helper
        dialogue_entries=dialogue_entries,
        intervention_turn=mediator_turn,
        new_turns=continuation,
        target_turn=post_target,
        mediator_utterance=mediator_utt,
        agent_prompts=agent_prompts,
    )
    post = measure_per_speaker_distributions(
        histories={conflict_agent: post_hist[conflict_agent]},
        agents_meta=agents_meta,
        sim_model=sim_model, api_key=api_key, max_workers=1,
    )
    post_idx  = ((post or {}).get(conflict_agent) or {}).get("argmax_idx")
    post_dist = ((post or {}).get(conflict_agent) or {}).get("distribution")
    out["post_idx"] = post_idx

    # --- Trajectory at mediator_turn+1 .. n_turns ---
    traj = []
    for tt in range(mediator_turn + 1, n_turns + 1):
        hist_tt = truncate_histories_to_turn(
            full_histories=None,
            dialogue_entries=dialogue_entries,
            intervention_turn=mediator_turn,
            new_turns=continuation,
            target_turn=tt,
            mediator_utterance=mediator_utt,
            agent_prompts=agent_prompts,
        )
        tp = measure_per_speaker_distributions(
            histories={conflict_agent: hist_tt[conflict_agent]},
            agents_meta=agents_meta,
            sim_model=sim_model, api_key=api_key, max_workers=1,
        )
        ax = ((tp or {}).get(conflict_agent) or {}).get("argmax_idx")
        if ax is not None:
            traj.append((tt, ax))
    out["trajectory"] = traj

    # --- AUC (Paige-weighted trapezoidal) ---
    out["auc"] = trajectory_auc(pre_idx, traj, mediator_turn, n_turns)

    # --- Signed W1 ---
    w1 = None
    if pre_dist is not None and post_dist is not None:
        try:
            w1 = compare_distributions(pre_dist, post_dist).get(
                "signed_wasserstein_1"
            )
        except Exception:
            w1 = None
    out["w1"] = w1

    # --- Judge (rubric-based, independent of DMIS labeler) ---
    if not skip_judge and mediator_utt and target_stage:
        try:
            CT = _load_conflict_table()
            comp = CT.get(target_stage, {})
            jres = run_judge(
                judge_model=judge_model,
                components={"stage": target_stage,
                            "description": comp.get("description", ""),
                            "mediation":   comp.get("mediation", "")},
                utterance=mediator_utt,
                api_key=api_key,
            )
            out["judge"] = (jres or {}).get("score")
        except Exception as e:
            out["judge"] = None
            out["judge_error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="JSON list (or object with 'records' field) of "
                         "completed-dialogue evaluation records.")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--sim-model",   default="openai/gpt-4o-mini",
                    help="Model used by the DMIS logprob labeler.")
    ap.add_argument("--judge-model", default="openai/gpt-4o-mini",
                    help="Model used by the rubric-based 1-5 judge.")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip Judge scoring (halves API cost).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_dotenv(_RELEASE_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY missing. Copy .env.example → .env and set it."
        )

    raw = json.load(open(args.records))
    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON list or {'records': [...]}.")
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[evaluate] {len(records)} records to score")

    aucs, w1s, judges = [], [], []
    per_sid = []
    for i, rec in enumerate(records, 1):
        try:
            res = score_one(rec, args.sim_model, args.judge_model, api_key,
                            skip_judge=args.skip_judge)
        except Exception as e:
            res = {"sid": str(rec.get("sid")), "skip": f"exc:{e!r}"}
        per_sid.append(res)
        if res.get("auc")   is not None: aucs.append(res["auc"])
        if res.get("w1")    is not None: w1s.append(res["w1"])
        if res.get("judge") is not None: judges.append(res["judge"])
        if i % 10 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] last sid={res.get('sid')} "
                  f"auc={res.get('auc')} w1={res.get('w1')} judge={res.get('judge')}")

    summary = {
        "n_scenarios": len(records),
        "n_evaluated": len(aucs),
        "auc":         _stats(aucs),
        "w1":          _stats(w1s),
        "judge":       _stats(judges),
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({"summary": summary, "per_sid": per_sid}, f, indent=2,
                  ensure_ascii=False)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
