"""
build_agent_prompts.py
======================

Reads scenario JSON files and emits agent_prompts with a sampled DMIS
ethnocentric stage (Denial / Defense / Minimization) under ASYMMETRIC
conflict semantics:

  * One agent (the "conflict agent") enacts the sampled ethnocentric
    Bennett stage. They receive a conflict_instruction that activates
    the pattern at conflict_turn.
  * The other agent (the "non-conflict agent") receives NO
    conflict_instruction. They engage with the conversation purely on
    their cultural value + goals + rules, behaving however the model
    naturally responds. This avoids forcing them into any particular
    DMIS stance and lets the asymmetry of the conflict come through.

The mediator's task is to shift the conflict agent toward
ethnorelativism while the non-conflict agent's stance emerges
organically from the prompt and the conflict agent's pressure.

Sampling
--------
Stages are allocated EQUALLY across input scenarios. For N input files,
stage assignments are constructed as the multiset
  ceil(N/3) of each stage, truncated to length N
and then shuffled. With the default of 90 scenarios this yields exactly
30 Denial / 30 Defense / 30 Minimization. Use --seed for reproducibility.

Per scenario
------------
  * The pre-allocated stage is used (no per-scenario random pick).
  * conflict_turn is sampled in [CONFLICT_TURN_MIN, CONFLICT_TURN_MAX].
  * The conflict agent (= the agent who speaks at conflict_turn) is
    determined by the parity of conflict_turn and is also the initiator
    of the goal_2 disagreement. Only this agent gets a
    conflict_instruction; the other agent gets an empty string.

Top-level keys: conflict_turn, conflict_agent, dmis, agent_prompts.

Note: there is no separate ``intervention_turn`` field. The mediator's
insertion point is decided downstream by ``evaluate_mediation_effectiveness.py``
either as the GT ``conflict_turn`` or via streaming conflict-turn
detection.

Usage
-----
  python build_agent_prompts.py
  python build_agent_prompts.py -n 90
  python build_agent_prompts.py --seed 42
"""

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
import glob
import json
import os
import random
import re

from utils import CONFLICT_TABLE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from pathlib import Path as _P
_REPO = _P(__file__).resolve().parent.parent
SCENARIO_DIR = str(_REPO / "data" / "SocialCC_JSON")
OUTPUT_DIR = str(_REPO / "data" / "CC_prompts")
MAX_TURNS = 10

# Earliest turn at which the conflict may surface (allow at least 1 cordial
# intro turn). Leave at least one turn after conflict_turn so that the
# evaluator's mediator can insert and at least one continuation turn can
# be generated.
CONFLICT_TURN_MIN = 3
CONFLICT_TURN_MAX = 8

# The three ethnocentric stages of Bennett's DMIS, in developmental order.
STAGES = ["Denial", "Defense", "Minimization"]


# ---------------------------------------------------------------------------
# Sentiment-flip helpers (unchanged from previous version)
# ---------------------------------------------------------------------------

FLIP_PAIRS = [
    ("strongly agree",                "strongly disagree"),
    ("strongly disagree",             "strongly agree"),
    ("agree",                         "disagree"),
    ("disagree",                      "agree"),
    ("a great deal of confidence",    "no confidence at all"),
    ("no confidence at all",          "a great deal of confidence"),
    ("no confidence",                 "a great deal of confidence"),
    ("would not like to have",        "would like to have"),
    ("would like to have",            "would not like to have"),
    ("never have to pay a bribe",     "sometimes have to pay a bribe"),
    ("sometimes have to pay a bribe", "never have to pay a bribe"),
]


def flip_sentiment(statement: str) -> str:
    lower = statement.lower()
    for original, replacement in FLIP_PAIRS:
        if original in lower:
            idx = lower.index(original)
            return statement[:idx] + replacement + statement[idx + len(original):]
    return statement


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_country_from_background(background: str) -> str:
    if not background:
        return ""
    m = re.search(r"from\s+([A-Z][\w\s]+?)[\.\s]*$", background.strip())
    return m.group(1).strip() if m else ""


def replace_country_in_statement(statement: str, new_country: str) -> str:
    if not new_country:
        return statement
    pattern = (
        r"^People from\s+[\w\s]+?"
        r"(?=\s+(?:strongly|agree|disagree|would|have|place|never|are))"
    )
    if re.match(pattern, statement):
        return re.sub(pattern, f"People from {new_country}", statement, count=1)
    return statement


# ---------------------------------------------------------------------------
# Stage-specific role guidance (used by the CONFLICT agent only).
# The non-conflict agent receives no conflict_instruction at all and
# therefore no role guidance: the dialogue runner does not inject any
# system message for them at conflict_turn + 1, and they keep behaving
# according to their base system prompt (cultural_value + goals + rules).
# ---------------------------------------------------------------------------

ROLE_GUIDANCE = {
    "Denial": (
        "Disengage from the cultural dimension of the disagreement, "
        "but disengage from a position of CONFIDENT FAMILIARITY rather "
        "than puzzled ignorance. The Denial speaker has not thought "
        "about cultural difference as a category, and feels they don't "
        "need to: in their view, differences are surface, things sort "
        "themselves out, and any reasonable person can handle this "
        "kind of situation without special effort. Stay vague, "
        "generally-true, and untroubled. Treat the other agent's "
        "cultural framing as either obvious or beside the point. "
        "\n\n"
        "MANDATORY: your conflict turn MUST include AT LEAST TWO of "
        "the following canonical Denial markers, used VERBATIM (you "
        "may add small connector words around them, but the marker "
        "phrase itself must appear unchanged): "
        "'live and let live', 'at the end of the day it's all the "
        "same', 'people figure it out', 'as long as we're speaking "
        "the same language', 'with experience you can handle any "
        "situation', 'different cultures are mostly the same once you "
        "get past the surface', \"I don't really see meaningful "
        "differences here\", 'I can pick that up as I go', \"doesn't "
        "matter to me\", \"I don't know about that\". "
        "These verbatim markers are the DEFINING SIGNAL of Denial. "
        "The turn is incorrectly classified if they are absent. "
        "\n\n"
        "STRICT BANS - any of the following AUTOMATICALLY collapses "
        "you out of Denial into Defense or Minimization: "
        "(a) DO NOT state your own substantive position on goal_2 "
        "    ('I think X is better', 'I believe Y', 'I really feel "
        "    that Z'). Denial speakers do not have a position to "
        "    state - that is the whole point. "
        "(b) DO NOT argue for either side of the cultural value, even "
        "    softly. Do not say what is right, fitting, important, or "
        "    appropriate regarding the goal_2 topic. "
        "(c) DO NOT use the warm 'we are all human deep down', 'deep "
        "    down we are the same', 'we all want the same things' "
        "    register - that is Minimization. "
        "(d) DO NOT use 'actually', 'no, that's not how it works', "
        "    'in our way', evaluative language about another group, "
        "    or ALL-CAPS - those are Defense. "
        "(e) DO NOT name 'culture', 'cultural difference', or compare "
        "    countries / nationalities. "
        "\n\n"
        "Subject changes, naive questions, oblivious reframings, and "
        "short non-answers are ENCOURAGED. Pivoting back to goal_1 "
        "(the cordial logistical topic) mid-turn is a strong Denial "
        "move. The conversation should feel low-temperature, "
        "untroubled, slightly off-topic, and quietly self-confident."
    ),
    "Defense": (
        "Run the conversation in a polarized us-versus-them frame. "
        "Insist your cultural frame is the correct default and the "
        "other agent's is wrong, backward, or naive. Use 'in our way / "
        "properly speaking / actually...' framings. Acknowledge their "
        "view only to dismiss it as deviation from the norm. "
        "Signal it through: flat contradictions ('No, that's not how it "
        "works.'), 'actually...' openers, sharp short sentences, "
        "occasional ALL-CAPS for stress, evaluative language about the "
        "other party's group. "
        "AVOID conciliatory openings: no 'you're right that', no 'fair "
        "enough', no 'let's find common ground' - those moves break the "
        "Defense pattern. Stay in the evaluative, polarized register."
    ),
    "Minimization": (
        "Stay WARM and CONCILIATORY throughout, and ENGAGED with the "
        "question (unlike Denial, you do have a view here, and your "
        "view is that the difference is real but small). Treat the "
        "cultural difference as a surface variation that obviously "
        "dissolves once we look at the human core underneath. Absorb "
        "the other agent's distinct view into shared sameness instead "
        "of pushing back against it. Speak as if you and the other "
        "agent are united by deep-down universal humanity. "
        "\n\n"
        "MANDATORY: your conflict turn MUST include AT LEAST TWO of "
        "the following canonical Minimization markers, used VERBATIM "
        "(you may add small connector words around them, but the "
        "marker phrase itself must appear unchanged): "
        "'deep down, we are all the same', 'people are pretty much "
        "motivated by the same things', \"when you really get to know "
        "people they're pretty much like us\", 'the basic need to "
        "communicate is the same everywhere', 'some values are just "
        "universal', \"it's a small world after all\", \"just be "
        "yourself, that's what matters\", 'we both know, at the end "
        "of the day', 'underneath we are the same'. "
        "These verbatim markers are the DEFINING SIGNAL of "
        "Minimization. The turn is incorrectly classified if they "
        "are absent. "
        "\n\n"
        "STRICT BANS - any of the following AUTOMATICALLY collapses "
        "you out of Minimization: "
        "(a) DO NOT use 'no, that's wrong', 'actually...', 'you don't "
        "    get it', flat contradictions, ALL-CAPS, or any sharp "
        "    evaluative tone - those are Defense. "
        "(b) DO NOT use \"I don't really know about that\", \"I "
        "    haven't thought about this\", \"doesn't matter to me\", "
        "    \"can we talk about something else\" - those are Denial. "
        "(c) DO NOT argue for the CULTURAL PARTICULARITY of your own "
        "    cultural_value (e.g., do not say 'in my country we...', "
        "    do not contrast countries). Present your value as the "
        "    UNIVERSAL DEFAULT that both of you obviously share. "
        "(d) DO NOT treat the disagreement as a real conflict of "
        "    frames - the whole point is that, in your view, there "
        "    ISN'T really a conflict; just two people noticing the "
        "    same shared truth from slightly different angles. "
        "\n\n"
        "When the other agent raises a specific cultural difference, "
        "smile it away and re-frame it as a minor variation on a "
        "shared truth. The conversation should feel friendly and "
        "dismissive at the same time, never adversarial and never "
        "disengaged."
    ),
}


def get_role_guidance(stage: str) -> str:
    return ROLE_GUIDANCE[stage]


# ---------------------------------------------------------------------------
# Stage allocation (equal across scenarios)
# ---------------------------------------------------------------------------

def allocate_stages(n: int, stages: list[str]) -> list[str]:
    """Return a length-n list assigning a stage to each scenario, with the
    counts as balanced as possible across stages.

    For N divisible by len(stages) (e.g. 90 / 3 = 30), each stage appears
    exactly N / len(stages) times. For N not divisible, stages are filled
    in the order given, so the first (N mod len(stages)) stages get one
    extra. The result is shuffled before return so that the scenarios
    receive stages in random order while the per-stage counts stay fixed.
    """
    if n <= 0:
        return []
    base = n // len(stages)
    rem = n % len(stages)
    pool: list[str] = []
    for i, s in enumerate(stages):
        count = base + (1 if i < rem else 0)
        pool.extend([s] * count)
    random.shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# DMIS payload + conflict instruction
# ---------------------------------------------------------------------------

def format_dmis(stage: str) -> dict:
    """Build the dmis payload to be saved with the scenario. Holds the stage
    name and the one-sentence description from CONFLICT_TABLE; the
    'mediation' field is intentionally omitted here (it is mediator-facing
    guidance, and the speakers must not see it).
    """
    entry = CONFLICT_TABLE[stage]
    return {
        "stage": stage,
        "description": entry["description"],
    }


def build_conflict_instruction(dmis: dict) -> str:
    """Build the conflict_instruction for the CONFLICT agent only.

    The conflict agent is the initiator: they surface the goal_2
    disagreement at conflict_turn and enact the sampled ethnocentric
    Bennett stage. The non-conflict agent does NOT call this function;
    their conflict_instruction is left empty.
    """
    opener = (
        "This is the moment to bring goal_2 to the foreground. On this "
        "turn, surface the goal_2 disagreement and set the tone."
    )
    style_block = (
        f"{dmis['stage']}\n"
        f"Definition: {dmis['description']}"
    )
    guidance = get_role_guidance(dmis["stage"])

    return (
        f"{opener}\n\n"
        f"CONFLICT STYLE:\n{style_block}\n\n"
        f"YOUR ROLE - ACTIVE PARTICIPANT IN THE CONFLICT PATTERN:\n"
        f"{guidance}\n\n"
    )


# ---------------------------------------------------------------------------
# Standard rules
# ---------------------------------------------------------------------------

STANDARD_RULES = [
    "Interact with the other agent to achieve each goal one by one.",
    "Keep each round of conversation short and no more than 100 words.",
    f"The full conversation runs for up to {MAX_TURNS} message exchanges.",
    "Once all goals are achieved, end the dialogue promptly with \"GOOD BYE!\".",
]


# ---------------------------------------------------------------------------
# Per-scenario builder
# ---------------------------------------------------------------------------

def build_agent_prompts(scenario: dict, stage: str) -> dict:
    cultural_value = scenario.get("cultural_value", "")
    cultural_value_country = scenario.get("cultural_value_country", "")
    scenario_text = scenario.get("scenario", "")

    agent_1_country = extract_country_from_background(
        scenario.get("agent_1_background", "")
    )
    agent_2_country = extract_country_from_background(
        scenario.get("agent_2_background", "")
    )

    cvc_norm = normalise(cultural_value_country)
    if normalise(agent_1_country) == cvc_norm:
        agent_1_value = cultural_value
        agent_2_value = replace_country_in_statement(
            flip_sentiment(cultural_value), agent_2_country
        )
    elif normalise(agent_2_country) == cvc_norm:
        agent_2_value = cultural_value
        agent_1_value = replace_country_in_statement(
            flip_sentiment(cultural_value), agent_1_country
        )
    else:
        agent_1_value = (
            replace_country_in_statement(cultural_value, agent_1_country)
            if agent_1_country else cultural_value
        )
        agent_2_value = (
            replace_country_in_statement(
                flip_sentiment(cultural_value), agent_2_country
            )
            if agent_2_country else flip_sentiment(cultural_value)
        )

    assigned = {"agent_1": agent_1_value, "agent_2": agent_2_value}

    # --- DMIS payload ---
    dmis = format_dmis(stage)

    # --- Conflict timing (no separate intervention_turn field) ---
    # Leave at least one turn after conflict_turn so a mediator can insert
    # and at least one continuation turn can be generated.
    conflict_turn_max = min(CONFLICT_TURN_MAX, MAX_TURNS - 1)
    if conflict_turn_max < CONFLICT_TURN_MIN:
        raise ValueError(
            f"Cannot satisfy conflict_turn in "
            f"[{CONFLICT_TURN_MIN}, {CONFLICT_TURN_MAX}] with MAX_TURNS={MAX_TURNS}."
        )
    conflict_turn = random.randint(CONFLICT_TURN_MIN, conflict_turn_max)

    # The conflict agent must speak at conflict_turn (they surface the
    # disagreement). Generation alternates by turn parity: odd -> agent_1,
    # even -> agent_2. So the parity of conflict_turn determines which
    # agent is the conflict agent.
    conflict_agent_key = "agent_1" if conflict_turn % 2 == 1 else "agent_2"

    print(
        f"  [~] dmis: {stage} | "
        f"conflict_turn={conflict_turn} | "
        f"conflict_agent={conflict_agent_key} (asymmetric: other agent has no conflict_instruction)"
    )

    agent_prompts: dict[str, dict] = {}
    for key in ("agent_1", "agent_2"):
        if not scenario.get(key):
            continue

        src = scenario.get("prompts", {}).get(key, {})
        is_conflict_agent = (key == conflict_agent_key)
        is_initiator = is_conflict_agent  # conflict agent is also initiator

        # Augment goal_2 with the permissive note about constructive conflict
        goals = dict(src.get("goals", {}))
        if goals.get("goal_2"):
            text = goals["goal_2"].rstrip()
            if not text.endswith("."):
                text += "."
            goals["goal_2"] = text 

        # Only the conflict agent receives a conflict_instruction. The
        # non-conflict agent's slot is left as an empty string so the
        # dialogue runner does not inject anything for them at
        # conflict_turn + 1.
        conflict_instruction = (
            build_conflict_instruction(dmis) if is_conflict_agent else ""
        )

        agent_prompts[key] = {
            "task":                 src.get("task", ""),
            "scenario":             scenario_text,
            "background":           scenario.get(f"{key}_background", ""),
            "cultural_value":       assigned[key],
            "goals":                goals,
            "rules":                STANDARD_RULES,
            "is_initiator":         is_initiator,
            "is_conflict_agent":    is_conflict_agent,
            "conflict_instruction": conflict_instruction,
        }

    return {
        "conflict_turn":     conflict_turn,
        "conflict_agent":    conflict_agent_key,
        "dmis":              dmis,
        "agent_prompts":     agent_prompts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build agent_prompts with sampled DMIS ethnocentric stage "
            "(Denial / Defense / Minimization), allocated equally across "
            "scenarios. Asymmetric conflict: only the conflict agent "
            "receives a conflict_instruction; the other agent has no "
            "such instruction and behaves naturally on the base prompt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--count", type=int, default=None, metavar="N",
        help="Number of scenarios to process (default: ALL).",
    )
    parser.add_argument("--scenario-dir", default=SCENARIO_DIR, metavar="DIR")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, metavar="DIR")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible stage allocation and "
             "conflict_turn sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(
        glob.glob(os.path.join(args.scenario_dir, "*.json")),
        key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0),
    )
    if not all_files:
        print(f"No JSON files found in: {args.scenario_dir}")
        return

    total = len(all_files)
    scenario_files = (
        all_files if args.count is None else all_files[: args.count]
    )
    n = len(scenario_files)
    label = "all" if args.count is None else f"first {n}"

    # Pre-allocate stages to scenarios so the per-stage counts are balanced.
    stage_assignments = allocate_stages(n, STAGES)
    counts = {s: stage_assignments.count(s) for s in STAGES}
    print(f"Found {total} file(s). Processing {label}.")
    print(
        "Stage allocation: "
        + ", ".join(f"{s}={counts[s]}" for s in STAGES)
    )
    print()

    for fpath, stage in zip(scenario_files, stage_assignments):
        fname = os.path.basename(fpath)
        print(f"Processing: {fname}")

        with open(fpath, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        output = build_agent_prompts(scenario, stage)
        out_path = os.path.join(args.output_dir, fname)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Saved -> {out_path}\n")

    print(f"Done. {n}/{total} file(s) processed.")


if __name__ == "__main__":
    main()