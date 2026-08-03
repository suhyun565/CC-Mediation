#!/usr/bin/env bash
# =============================================================================
# scripts/build_dataset.sh
#
# End-to-end data-generation pipeline.
# Turns raw SocialCC scenarios into (dialogues, labelled turns, preference
# pairs). Runs one "pass": each pass produces one dialogue per unused
# scenario. Rerun with different --pass-id / --seed to accumulate more data.
#
# Prereqs:
#   * OPENROUTER_API_KEY set (via .env or shell env)
#   * Raw scenarios present at data/SocialCC_JSON/*.json
#
# Usage:
#   bash scripts/build_dataset.sh 0                    # single pass, id=0
#   bash scripts/build_dataset.sh 0 --limit 50         # small dry-run
#   for i in 0 1 2 3 4; do bash scripts/build_dataset.sh $i; done   # 5 passes
#
# What runs (see data_pipeline/generate_training_data.py):
#   ① build_agent_prompts.py       — scenario → per-agent prompts
#   ② generate.py                  — prompts → 10-turn dialogue
#   ③ label_phases.py              — per-turn phase labels
#   ④ relabel_with_logprob.py      — continuation turns → DMIS argmax
# (⑤/⑥ = build_cc_mediation_unified.py + remeasure_pre_auc.py — run separately
#  after you have enough labelled dialogues; see README §1 for the invocation.)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PASS_ID="${1:-0}"
shift || true

python data_pipeline/generate_training_data.py \
    --pass-id "$PASS_ID" \
    --seed "$((42 + PASS_ID))" \
    "$@"

echo ""
echo "== pass $PASS_ID complete =="
echo "Next steps (once labelled dialogues have accumulated):"
echo "  python data_pipeline/build_cc_mediation_unified.py   # ⑤ pair (chosen/rejected/original)"
echo "  python data_pipeline/remeasure_pre_auc.py            # ⑥ recompute pre_profile at conflict_turn"
