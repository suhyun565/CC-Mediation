#!/usr/bin/env bash
# =============================================================================
# scripts/simulate_and_score.sh
#
# Full simulator + evaluation: given a JSON of (sid, predicted_turn,
# utterance) predictions from some mediator, re-simulate the continuation
# under an LLM sim actor and score with the paper's trajectory metrics:
#
#   * PRE distribution at conflict_turn (via DMIS logprob labeler)
#   * Continuation from predicted_turn with the mediator utterance injected
#   * POST distribution at conflict_turn + 3
#   * Trajectory argmax at every turn from predicted_turn+1 to n_turns
#   * Trapezoidal, Paige-weighted AUC on the trajectory
#   * signed_wasserstein_1(pre_dist, post_dist)
#   * LLM Judge score (1–5) against the target stage's mediation rubric
#
# All continuation/labeler/judge calls hit OpenRouter, so this costs API
# time. See --limit for smoke-runs.
#
# Prereqs:
#   * OPENROUTER_API_KEY set (via .env or shell env)
#   * data/CC_dialogues/*.json and data/CC_labeled/{training,evaluate}/*.json
#     present (from the data pipeline)
#   * An eval-set JSONL (one row per scenario) with fields
#       scenario_id, turns, gt_intervention_turn, target_stage, n_turns
#     — same schema the paper uses at training/data/eval.jsonl
#
# Usage:
#   bash scripts/simulate_and_score.sh \
#       my_model                                   # KEY (used only in filenames)
#       predictions/mymodel_base_hf.json           # BASE_HF (per-sid predictions)
#       eval_data/eval.jsonl                       # EVAL_FILE
#       results/mymodel_base_auc_w1.json           # OUT
#       --limit 20                                 # (optional) smoke run
#
# The BASE_HF file must contain a per_sid list where each row has at least
#   {"sid": "42", "predicted_turn": 6, "utterance": "It sounds like ..."}
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${1:?usage: bash scripts/simulate_and_score.sh KEY BASE_HF EVAL_FILE OUT [extra args]}"
BASE_HF="${2:?BASE_HF required (JSON with per-sid predictions)}"
EVAL_FILE="${3:?EVAL_FILE required (eval.jsonl)}"
OUT="${4:?OUT required (output JSON path)}"
shift 4 || true

python metrics/eval_base_auc_w1.py \
    --key       "$KEY" \
    --base-hf   "$BASE_HF" \
    --eval-file "$EVAL_FILE" \
    --out       "$OUT" \
    "$@"

echo ""
echo "== simulate+score complete → $OUT =="
