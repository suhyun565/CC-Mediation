# CC-Mediation

Code for the **CC-Mediation** benchmark: a DMIS-grounded evaluation of LLM
mediators in cross-cultural dialogue.

The repo has three independently-useful entry points, each with a matching
bash wrapper under `scripts/`:

| Part | What it does | Entry point | Bash wrapper |
|---|---|---|---|
| **1. Data pipeline** | SocialCC scenarios → dialogues → labelled turns → preference pairs | `data_pipeline/generate_training_data.py` | `scripts/build_dataset.sh` |
| **2. Simulator + eval** | Given a mediator's `(sid, predicted_turn, utterance)` predictions, **re-simulate** continuation via OpenRouter and score trajectory AUC / signed W₁ / Judge | `metrics/eval_base_auc_w1.py` | `scripts/simulate_and_score.sh` |
| **3. Eval only** | Given **already-generated dialogues** (mediator utterance + continuation already present), only run the DMIS labeler + judge and compute AUC / W₁ / Judge | `metrics/evaluate.py` | `scripts/evaluate.sh` |

Training and evaluation of the mediator model itself, human-annotation apps,
and the paper-internal analyses live in separate repos and are not included
here.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env with your API keys
├── .gitignore
│
├── shared/                       # shared building blocks
│   ├── utils.py                  # CONFLICT_TABLE: DMIS stage definitions + mediation moves
│   ├── dmis_items.py             # DMIS Likert items + IDI weights (Bennett 1986/1993/2011)
│   └── dmis_distribution.py      # Paige-weighted signed W₁, JS, TVD, first-order dominance
│
├── data_pipeline/                # Part 1: SocialCC → CC-Mediation records
│   ├── build_agent_prompts.py    # ① scenario → per-agent prompts with sampled ethnocentric stage
│   ├── generate.py               # ② prompts → 10-turn agent-agent dialogue
│   ├── label_phases.py           # ③ per-turn phase labels (pre_conflict / conflict / continuation)
│   ├── relabel_with_logprob.py   # ④ continuation turns → argmax DMIS stage via logprob labeler
│   ├── build_cc_mediation_unified.py   # ⑤ (shared_prefix, chosen, rejected, original) per sid
│   ├── remeasure_pre_auc.py      # ⑥ recompute pre_profile at conflict_turn and update summary.json
│   └── generate_training_data.py # end-to-end orchestrator (① → ④) for a single "pass"
│
├── metrics/                      # Parts 2 & 3: trajectory AUC / W₁ / judge
│   ├── llm_stage_judge.py        # OpenRouter logprob DMIS classifier (6-stage softmax on digits)
│   ├── llm_stage_judge_local.py  # same, but forward-pass on a local HF model
│   ├── evaluate_mediation_effectiveness.py  # dialogue simulator + pre/post/trajectory
│   ├── eval_base_auc_w1.py       # Part 2 entry: sim + score from (predicted_turn, utterance)
│   └── evaluate.py               # Part 3 entry: score from already-generated dialogues
│
└── scripts/
    ├── build_dataset.sh          # Part 1 wrapper
    ├── simulate_and_score.sh     # Part 2 wrapper
    └── evaluate.sh               # Part 3 wrapper
```

---

## Install

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
cp .env.example .env      # paste OPENROUTER_API_KEY (and HF_TOKEN if using local labeler)
```

Python 3.10+ recommended.

---

## Part 1 — Data-generation pipeline

**What it produces.** For each raw SocialCC scenario, one pass yields:

* `data/CC_prompts/<sid>.json` — per-agent prompts with a sampled
  ethnocentric DMIS stage (Denial / Defense / Minimization) as the
  conflict pattern.
* `data/CC_dialogues/<sid>.json` — a 10-turn agent-agent dialogue.
* `data/CC_labeled/training/<sid>.json` — the same dialogue with per-turn
  phase labels and, on continuation turns, an argmax DMIS stage from a
  logprob-based labeler (`metrics/llm_stage_judge.py`).

Later steps (`build_cc_mediation_unified.py`, `remeasure_pre_auc.py`)
combine dialogues into `(chosen, rejected)` preference pairs under
`data/CC_mediation/{training,evaluate}/<sid>/` and re-measure the PRE
reference distribution at `conflict_turn`.

**Admission gate** applied by the pair-construction step (§3.4):

$$\Delta_\text{AUC} > 0 \;\;\lor\;\; \bigl(\Delta_\text{AUC} = 0 \;\land\; \Delta_{W_1} > 0\bigr)$$

where $\Delta_X = X_\text{chosen} - X_\text{rejected}$ and both metrics are
measured with the same logprob DMIS labeler.

**Run it.**

```bash
# Requires OPENROUTER_API_KEY and raw scenarios at data/SocialCC_JSON/*.json
bash scripts/build_dataset.sh 0                       # one pass, id=0
bash scripts/build_dataset.sh 0 --limit 50            # small dry-run
for i in 0 1 2 3 4; do bash scripts/build_dataset.sh $i; done   # 5 passes
```

Each pass produces one dialogue per unused scenario. Repeat with
increasing `--pass-id` and different seeds (the wrapper uses
`42 + pass_id`) to accumulate multiple dialogues per scenario without
polluting the eval split.

Once labelled dialogues have accumulated, build preference pairs and
recompute the PRE reference:

```bash
python data_pipeline/build_cc_mediation_unified.py   # ⑤
python data_pipeline/remeasure_pre_auc.py            # ⑥
```

---

## Part 2 — Simulator + evaluation

**When to use.** You have a mediator (any model, any decoding strategy)
that emits, for each held-out scenario, an intervention turn and an
utterance. You want the paper's trajectory metrics against the same
sim / labeler / judge that Tables 4–5 use.

**What it computes**, per scenario:

1. **PRE distribution** at `conflict_turn`, from the logprob DMIS labeler
   on the original dialogue history.
2. **Continuation** from `predicted_turn`, with the mediator utterance
   injected into both agents' histories as a user-role message.
3. **POST distribution** at `min(conflict_turn + 3, n_turns)`.
4. **Trajectory** argmax stage at every turn from `predicted_turn + 1` to
   `n_turns`.
5. **Trajectory AUC** — trapezoidal integral over normalised time
   $\tau \in [0,1]$ of
   $\bigl(\text{Paige}[\text{argmax}_t] - \text{Paige}[\text{argmax}_{t_\text{pre}}]\bigr)$
   using $\text{Paige}=(-3,-2,-1,+1,+2,+3)$.
6. **Signed W₁** — `signed_wasserstein_1(pre_dist, post_dist)` from
   `shared/dmis_distribution.py`.
7. **LLM Judge** — 1–5 score against the target stage's mediation move.

**Input schema** (`--base-hf`). One JSON with a `per_sid` list, each row
containing at minimum:

```json
{"sid": "42", "predicted_turn": 6, "utterance": "It sounds like ..."}
```

**Run it.**

```bash
bash scripts/simulate_and_score.sh \
    my_model                                     \
    predictions/mymodel_base_hf.json             \
    eval_data/eval.jsonl                         \
    results/mymodel_base_auc_w1.json             \
    --limit 20
```

Positional args: `KEY BASE_HF EVAL_FILE OUT`. Extra flags after `OUT` are
forwarded to `eval_base_auc_w1.py` (e.g., `--limit`, `--sim-model`,
`--judge-model`).

**Note.** This calls OpenRouter for continuation, labeler, and judge, so
API cost / time is proportional to `n_scenarios × (trajectory_length + 3)`.
Use `--limit` for smoke runs. `--sim-model` and `--judge-model` default
to `openai/gpt-4o-mini`; override for ablation.

**Output** (`--out`) matches the paper's
`eval_<key>_{base,sft,dpo}_auc_w1.json`:

```json
{
  "summary": {
    "n_scenarios": 159,
    "n_evaluated": 157,
    "auc":   {"n": 157, "mean": ..., "median": ..., "std": ...},
    "w1":    {"n": 157, "mean": ..., "median": ..., "std": ...},
    "judge": {"n": 157, "mean": ..., "median": ..., "std": ...}
  },
  "per_sid": [
    {"sid": "2", "pred_turn": 4, "gt_turn": 4,
     "auc": 1.5, "w1": 1.66, "judge": 5,
     "pre_idx": 2, "post_idx": 3,
     "trajectory": [[5, 2], [6, 3], ...]}
  ]
}
```

---

## Part 3 — Evaluation only (dialogue already generated)

**When to use.** You already generated the continuation yourself (any
sim actor, any decoding strategy) and produced the completed dialogue
with the mediator utterance already injected at the intervention turn.
You just want the paper's `AUC / signed W₁ / Judge` under identical
formulas.

**Difference from Part 2.** Part 2 re-simulates the continuation via an
LLM sim actor given `(predicted_turn, utterance)`. Part 3 takes the
already-generated continuation as input and only calls the DMIS logprob
labeler (for PRE / trajectory / POST distributions) and the rubric
Judge. No sim-actor cost.

**Input schema** (`--records`). A JSON list (or object with a
top-level `records`), one entry per scenario:

```json
{
  "sid":                "42",
  "conflict_turn":      8,
  "conflict_agent":     "agent_2",
  "target_stage":       "Minimization",
  "mediator_turn":      6,
  "mediator_utterance": "It sounds like ...",
  "n_turns":            10,
  "agent_prompts":      { "agent_1": { ... }, "agent_2": { ... } },
  "dialogue":           [
    {"turn": 1, "agent": "agent_1", "message": "..."},
    ...
    {"turn": 6, "agent": "agent_2", "message": "..."}
  ],
  "continuation":       [
    {"turn": 7,  "agent": "agent_1", "message": "..."},
    {"turn": 8,  "agent": "agent_2", "message": "..."},
    ...
    {"turn": 10, "agent": "agent_2", "message": "..."}
  ]
}
```

* `dialogue` = original conversation up to and including `mediator_turn`.
* `continuation` = turns `mediator_turn + 1 .. n_turns`, generated by
  your sim actor in response to the mediator.
* `agent_prompts` has the same schema as `data/CC_dialogues/*.json`
  (`background`, `cultural_value`, `goals`, `rules`, …) — required so the
  DMIS labeler and the plaintext-history reconstruction have the correct
  system prompts.

**Run it.**

```bash
# API cost: (n_turns − mediator_turn + 2) labeler calls + 1 judge call per record.
bash scripts/evaluate.sh path/to/records.json path/to/out.json

# Halve API cost by skipping the Judge:
bash scripts/evaluate.sh path/to/records.json path/to/out.json --skip-judge

# Smoke run:
bash scripts/evaluate.sh path/to/records.json path/to/out.json --limit 5
```

Output schema matches Part 2's `{"summary": ..., "per_sid": ...}`.

---

## Key implementation details worth reading

- **Paige weighting.** `shared/dmis_distribution.STAGE_POSITIONS = (-3, -2, -1, +1, +2, +3)`.
  The **two-unit gap between Minimization (−1) and Acceptance (+1)** encodes
  Bennett's paradigm shift between ethnocentric and ethnorelative
  orientations — a Minimization → Acceptance transition costs twice any
  other adjacent-stage transition under W₁.

- **Signed W₁.** Sign comes from the change in expected position under
  Paige weights (`E_post − E_pre`); magnitude is the (unsigned)
  Paige-weighted Wasserstein-1 distance. Range roughly [−6, +6];
  0 means no movement.

- **Logprob DMIS labeler.** The model is prompted to emit one digit `1–6`
  for the DMIS stage. Top-token logprobs at that position are restricted
  to those six digits and softmaxed into a 6-element probability vector.
  This gives a proper distribution (not just an argmax), so W₁ and AUC
  use the same underlying measurement channel.

- **PRE is at `conflict_turn`**, not at `predicted_turn`. If the mediator
  fires several turns before the conflict, the trajectory therefore
  measures a counterfactual continuation against the original conflict
  distribution — an intentional design choice (the mediator's job is to
  move the trajectory away from the conflicted state).

- **Judge is independent of the DMIS labeler.** The judge is a 1–5 rubric
  scorer against the target stage's mediation move; it is not derived
  from the same logprob channel that the admission gate uses.

---

## Citation

If you use this code or data, please cite the CC-Mediation paper (link TBA)
and the referenced works:

- Bennett, M. J. (1986, 1993, 2011). *A Developmental Model of
  Intercultural Sensitivity*. Intercultural Development Research Institute.
- Hammer, M. R., Bennett, M. J., & Wiseman, R. (2003). *Measuring
  intercultural sensitivity: The Intercultural Development Inventory*.
  Int. J. Intercultural Relations.
- Paige, R. M. et al. (2003). *Assessing intercultural sensitivity: An
  empirical analysis of the Hammer and Bennett Intercultural Development
  Inventory*. Int. J. Intercultural Relations.
- Villani, C. (2008). *Optimal Transport: Old and New*.

## License

Add your preferred license file (`LICENSE`) at the repo root.
