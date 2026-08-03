"""Trajectory AUC / W1 / judge for the BASE (prompt-only) mediator.

Reuses the (predicted_turn, utterance) pairs produced by eval_base_hf.py and
runs the same continuation + DMIS distribution measurement that
eval_sft_auc_w1.py uses for SFT/DPO. The only difference is the source of
the mediator utterance: instead of running a fine-tuned model, we read it
from eval_<key>_base_hf.json.

Usage:
    python3 eval_base_auc_w1.py --key llama \\
        --base-hf training/results/eval_llama_base_hf.json \\
        --out     training/results/eval_llama_base_auc_w1.json
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
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

load_dotenv(REPO / ".env")
# HF_TOKEN is expected to be provided via .env (see .env.example).
# Only required if you use the local HF logprob labeler with a gated model.

from evaluate_mediation_effectiveness import (
    generate_continuation_with_mediator,
    measure_per_speaker_distributions,
    truncate_histories_to_turn,
    build_initial_histories,
    run_judge,
    agent_metadata,
)
from dmis_distribution import compare_distributions

STAGE_POSITIONS = [-3, -2, -1, +1, +2, +3]


def _stats(xs):
    if not xs: return {"n": 0}
    return {
        "n": len(xs),
        "mean": st.mean(xs),
        "median": st.median(xs),
        "min": min(xs), "max": max(xs),
        "std": st.stdev(xs) if len(xs) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--base-hf", required=True,
                    help="eval_<key>_base_hf.json with per-sid predicted_turn+utterance")
    ap.add_argument("--eval-file", default="training/data/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sim-model",  default="openai/gpt-4o-mini")
    ap.add_argument("--judge-model",default="openai/gpt-4o-mini")
    args = ap.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]

    base_hf = json.load(open(args.base_hf))
    per_sid_in = {str(r["sid"]): r for r in base_hf.get("per_sid", [])}
    print(f"[base-auc] key={args.key}  loaded {len(per_sid_in)} predictions")

    # Load eval records (for ground-truth turn order + agent metadata source)
    eval_recs: List[Dict] = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line: eval_recs.append(json.loads(line))
    if args.limit > 0:
        eval_recs = eval_recs[: args.limit]

    aucs, w1s, judges = [], [], []
    per_sid = []

    for i, rec in enumerate(eval_recs, 1):
        sid = str(rec["scenario_id"])
        gt_turn = rec.get("gt_intervention_turn") or rec.get("intervention_turn")
        pred_rec = per_sid_in.get(sid, {})
        pred_turn = pred_rec.get("predicted_turn")
        utterance = (pred_rec.get("utterance") or "").strip()
        if pred_turn is None or not utterance:
            per_sid.append({"sid": sid, "skip": "no_speak"})
            continue

        try:
            dlg_p = REPO / "data/CC_dialogues" / f"{sid}.json"
            lab_p = REPO / "data/CC_labeled/training" / f"{sid}.json"
            if not lab_p.exists():
                lab_p = REPO / "data/CC_labeled/evaluate" / f"{sid}.json"
            if not (dlg_p.exists() and lab_p.exists()):
                per_sid.append({"sid": sid, "skip": "no_source"})
                continue
            dlg = json.load(open(dlg_p))
            lab = json.load(open(lab_p))
            dialogue_entries = dlg.get("dialogue", [])
            agent_prompts = dlg.get("agent_prompts", {})
            conflict_agent = lab.get("conflict_agent")
            conflict_turn  = lab.get("conflict_turn")
            target_stage   = lab.get("target_stage") or rec.get("target_stage", "")
            if not (conflict_agent and conflict_turn):
                per_sid.append({"sid": sid, "skip": "no_meta"})
                continue
            n_turns = max(t["turn"] for t in dialogue_entries)
            agents_meta = agent_metadata(dlg)

            # PRE measurement at conflict_turn
            pre_hist = build_initial_histories(
                dialogue_entries, agent_prompts, conflict_turn,
            )
            pre = measure_per_speaker_distributions(
                histories={conflict_agent: pre_hist[conflict_agent]},
                agents_meta=agents_meta,
                sim_model=args.sim_model, api_key=api_key, max_workers=1,
            )
            pre_idx = ((pre or {}).get(conflict_agent) or {}).get("argmax_idx")
            pre_dist = ((pre or {}).get(conflict_agent) or {}).get("distribution")
            if pre_idx is None:
                per_sid.append({"sid": sid, "skip": "pre_none"})
                continue

            # Continuation with base utterance
            histories, new_turns, cont_err = generate_continuation_with_mediator(
                sim_model=args.sim_model,
                intervention_turn=pred_turn,
                n_turns=n_turns,
                dialogue_entries=dialogue_entries,
                agent_prompts=agent_prompts,
                mediator_utterance=utterance,
                api_key=api_key,
            )
            if cont_err:
                per_sid.append({"sid": sid, "skip": f"cont_err:{cont_err}"})
                continue

            # POST at conflict_turn + 3 (capped)
            post_target = min(conflict_turn + 3, n_turns)
            post_hist = truncate_histories_to_turn(
                full_histories=histories,
                dialogue_entries=dialogue_entries,
                intervention_turn=pred_turn,
                new_turns=new_turns,
                target_turn=post_target,
                mediator_utterance=utterance,
                agent_prompts=agent_prompts,
            )
            post = measure_per_speaker_distributions(
                histories={conflict_agent: post_hist[conflict_agent]},
                agents_meta=agents_meta,
                sim_model=args.sim_model, api_key=api_key, max_workers=1,
            )
            post_idx = ((post or {}).get(conflict_agent) or {}).get("argmax_idx")
            post_dist = ((post or {}).get(conflict_agent) or {}).get("distribution")
            if post_idx is None:
                per_sid.append({"sid": sid, "skip": "post_none"})
                continue

            # Trajectory pred_turn+1..n_turns
            traj = []
            for tt in range(pred_turn + 1, n_turns + 1):
                hist_tt = truncate_histories_to_turn(
                    full_histories=histories,
                    dialogue_entries=dialogue_entries,
                    intervention_turn=pred_turn,
                    new_turns=new_turns,
                    target_turn=tt,
                    mediator_utterance=utterance,
                    agent_prompts=agent_prompts,
                )
                tp = measure_per_speaker_distributions(
                    histories={conflict_agent: hist_tt[conflict_agent]},
                    agents_meta=agents_meta,
                    sim_model=args.sim_model, api_key=api_key, max_workers=1,
                )
                ax = ((tp or {}).get(conflict_agent) or {}).get("argmax_idx")
                if ax is not None: traj.append((tt, ax))

            duration = n_turns - pred_turn
            if duration <= 0 or not traj:
                per_sid.append({"sid": sid, "skip": "no_traj"})
                continue

            # Trapezoidal AUC, time-normalised (Paige-weighted positions)
            points = []
            for tt, ax in traj:
                tau = max(0.0, min(1.0, (tt - pred_turn) / duration))
                delta = STAGE_POSITIONS[ax] - STAGE_POSITIONS[pre_idx]
                points.append((tau, delta))
            points.sort(key=lambda x: x[0])
            if len(points) == 1:
                auc = float(points[0][1])
            else:
                auc = 0.0
                tf, pf = points[0]
                if tf > 0: auc += pf * tf
                for j in range(len(points) - 1):
                    ta, pa = points[j]; tb, pb = points[j+1]
                    auc += (pa + pb) / 2.0 * (tb - ta)
                tl, pl = points[-1]
                if tl < 1.0: auc += pl * (1.0 - tl)

            # Signed W1
            w1 = None
            if pre_dist is not None and post_dist is not None:
                try:
                    cmp_ = compare_distributions(pre_dist, post_dist)
                    w1 = cmp_.get("signed_wasserstein_1")
                except Exception:
                    pass

            # Judge
            judge_score = None
            try:
                jres = run_judge(
                    judge_model=args.judge_model,
                    components={"stage": target_stage,
                                "description": (lab.get("dmis") or {}).get("description", ""),
                                "mediation":   (lab.get("dmis") or {}).get("mediation", "")},
                    utterance=utterance,
                    api_key=api_key,
                )
                judge_score = (jres or {}).get("score")
            except Exception:
                pass

            aucs.append(auc)
            if w1 is not None: w1s.append(w1)
            if judge_score is not None: judges.append(judge_score)
            per_sid.append({
                "sid": sid, "pred_turn": pred_turn, "gt_turn": gt_turn,
                "auc": auc, "w1": w1, "judge": judge_score,
                "pre_idx": pre_idx, "post_idx": post_idx,
            })
        except Exception as e:
            per_sid.append({"sid": sid, "skip": f"exc:{e!r}"})

        if i % 10 == 0 or i == len(eval_recs):
            last = per_sid[-1] if per_sid else {}
            print(f"  [{i}/{len(eval_recs)}] {last}")

    summary = {
        "n_scenarios":   len(eval_recs),
        "n_evaluated":   len(aucs),
        "auc":           _stats(aucs),
        "w1":            _stats(w1s),
        "judge":         _stats(judges),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sid": per_sid}, f,
                  indent=2, ensure_ascii=False)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
