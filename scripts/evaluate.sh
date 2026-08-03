#!/usr/bin/env bash
# =============================================================================
# scripts/evaluate.sh
#
# Score completed dialogues (mediator utterance already injected) with the
# paper's mediation metrics. No re-simulation — the continuation is expected
# to be already generated. Only the DMIS logprob labeler (for PRE / trajectory
# / POST distributions) and the rubric-based judge are called.
#
# Difference from scripts/simulate_and_score.sh:
#   * simulate_and_score.sh RE-SIMULATES the continuation via an LLM sim
#     actor given (predicted_turn, utterance).
#   * evaluate.sh takes the ALREADY-GENERATED continuation as input and
#     just measures it.
#
# Prereqs:
#   * OPENROUTER_API_KEY set (via .env or shell env) — for the DMIS logprob
#     labeler and the 1-5 rubric judge.
#   * Input records JSON (see README §3 for full schema); minimum fields:
#       sid, conflict_turn, conflict_agent, target_stage, mediator_turn,
#       mediator_utterance, n_turns, agent_prompts, dialogue, continuation
#
# Usage:
#   bash scripts/evaluate.sh RECORDS_JSON OUT_JSON [extra args]
#
# Extra args are forwarded to metrics/evaluate.py, e.g.:
#   --limit 20              # smoke run
#   --skip-judge            # skip Judge (halves API cost)
#   --sim-model X           # override DMIS labeler
#   --judge-model X         # override rubric judge
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RECORDS="${1:?usage: bash scripts/evaluate.sh RECORDS_JSON OUT_JSON [extra args]}"
OUT="${2:?OUT_JSON required}"
shift 2 || true

python metrics/evaluate.py --records "$RECORDS" --out "$OUT" "$@"

echo ""
echo "== evaluate complete → $OUT =="
