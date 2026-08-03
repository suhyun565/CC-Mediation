"""
generate_training_data.py

Run the full CC pipeline (build_agent_prompts → generate → label_phases →
relabel_with_logprob) over UNUSED SocialCC_JSON scenarios for one "pass",
then save the final labeled dialogues into:
    data/CC_labeled/training/{sid}_p{pass}.json

Run this script multiple times with different --pass-id (and --seed) to
build up to 10K dialogues, since each pass produces one dialogue per
scenario.

Excludes scenarios that appear in either eval_pool.json or the main
data/CC_labeled/ folder (so we don't pollute the train set with eval
material).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO / "data/SocialCC_JSON"
EVAL_POOL = REPO / "data/eval_pool.json"
MAIN_LABELED = REPO / "data/CC_labeled"
TRAINING_OUT = REPO / "data/CC_labeled/training"


def collect_available_scenarios() -> list[str]:
    """All SocialCC scenarios not in eval_pool.json and not already in main CC_labeled/."""
    all_sids = sorted(p.stem for p in SCENARIO_DIR.glob("*.json") if p.stem.isdigit())
    eval_pool = json.load(open(EVAL_POOL))
    eval_sids = {str(it["scenario_id"]) for it in eval_pool["items"]}
    main_labeled = {p.stem for p in MAIN_LABELED.glob("*.json") if p.stem.isdigit()}
    used = eval_sids | main_labeled
    return [s for s in all_sids if s not in used]


def run(cmd: list[str], desc: str):
    print(f"\n[{desc}] $ {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        raise RuntimeError(f"FAILED: {desc} (returncode={res.returncode})")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pass-id", type=int, required=True,
                   help="Pass number (used as seed and as filename suffix).")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of scenarios this pass (default: all unused).")
    p.add_argument("--workers-gen", type=int, default=8,
                   help="generate.py concurrency.")
    p.add_argument("--workers-label-turn", type=int, default=6,
                   help="label_phases.py per-turn workers.")
    p.add_argument("--workers-label-dlg", type=int, default=3,
                   help="label_phases.py per-dialogue workers.")
    p.add_argument("--workers-relabel", type=int, default=8,
                   help="relabel_with_logprob.py workers.")
    p.add_argument("--gen-model", default="openai/gpt-4o-mini",
                   help="Dialogue-generation model (matches existing pipeline).")
    p.add_argument("--judge-model", default="openai/gpt-4o-mini",
                   help="Stage / phase LLM judge model.")
    p.add_argument("--skip-prompts", action="store_true",
                   help="Skip prompt-build step (use existing /tmp dir).")
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--skip-label", action="store_true")
    p.add_argument("--skip-relabel", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pass_id = args.pass_id
    work = Path(f"/tmp/training_pass{pass_id}")
    prompts_d = work / "CC_prompts"
    dialogues_d = work / "CC_dialogues"
    labeled_d = work / "CC_labeled"
    for d in [prompts_d, dialogues_d, labeled_d]:
        d.mkdir(parents=True, exist_ok=True)
    TRAINING_OUT.mkdir(parents=True, exist_ok=True)

    avail = collect_available_scenarios()
    if args.limit:
        avail = avail[: args.limit]
    print(f"Pass {pass_id}: {len(avail)} scenarios available "
          f"(excluding eval_pool + main CC_labeled).")

    # 1) build_agent_prompts — always build for ALL 3060 scenarios so that
    #    every available sid has a prompt; generate.py then filters via --ids.
    if not args.skip_prompts:
        run([
            "python3", "build_agent_prompts.py",
            "--scenario-dir", str(SCENARIO_DIR),
            "--output-dir", str(prompts_d),
            "--seed", str(pass_id * 1000 + 7),
        ], desc=f"pass{pass_id}: build_agent_prompts")

    # 2) generate.py — write only for the available scenario ids.
    if not args.skip_gen:
        # Generate one batch with --ids to filter to available scenarios.
        # --skip-existing avoids re-runs on partial outputs.
        run([
            "python3", "generate.py",
            "--input-dir", str(prompts_d),
            "--output-dir", str(dialogues_d),
            "--ids", *avail,
            "--model", args.gen_model,
            "--skip-existing",
        ], desc=f"pass{pass_id}: generate.py")

    # 3) label_phases.py
    if not args.skip_label:
        run([
            "python3", "label_phases.py",
            "--input-dir", str(dialogues_d),
            "--output-dir", str(labeled_d),
            "--judge-model", args.judge_model,
            "--sim-model", args.judge_model,
            "--max-turn-workers", str(args.workers_label_turn),
            "--max-dialogue-workers", str(args.workers_label_dlg),
            "--skip-existing",
        ], desc=f"pass{pass_id}: label_phases")

    # 4) relabel_with_logprob.py
    if not args.skip_relabel:
        run([
            "python3", "relabel_with_logprob.py",
            "--labeled-dir", str(labeled_d),
            "--dialogues-dir", str(dialogues_d),
            "--judge-model", args.judge_model,
            "--max-workers", str(args.workers_relabel),
        ], desc=f"pass{pass_id}: relabel_with_logprob")

    # 5) Consolidate into data/CC_labeled/training/{sid}_p{pass}.json
    n_copied = 0
    for src in labeled_d.glob("*.json"):
        if src.name.startswith("_"):
            continue
        sid = src.stem
        dst = TRAINING_OUT / f"{sid}_p{pass_id}.json"
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"\nPass {pass_id} done. Copied {n_copied} labeled dialogues into "
          f"{TRAINING_OUT}/{{sid}}_p{pass_id}.json")


if __name__ == "__main__":
    main()
