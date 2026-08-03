"""
generate.py
"""

import json
import os
import re
import glob
import argparse

from openai import OpenAI
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Put it in a .env file (OPENROUTER_API_KEY=sk-or-v1-...) "
        "or export it in your shell."
    )

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR  = str(_REPO / "data" / "CC_prompts")
DEFAULT_OUTPUT_DIR = str(_REPO / "data" / "CC_dialogues")
DEFAULT_MODEL      = "openai/gpt-4o-mini"
MAX_TURNS          = 10
MAX_TOKENS         = 300

# ──────────────────────────────────────────────
# OpenRouter client
# ──────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

EXTRA_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
    "X-Title":      os.getenv("OPENROUTER_APP_NAME", "CulturalConflict"),
}

# ──────────────────────────────────────────────
# System-prompt builder (static — delivered once)
# ──────────────────────────────────────────────
def build_system_prompt(agent: dict) -> str:
    """Static base system prompt — placed in history ONCE at conversation start."""
    goals = agent.get("goals", {})
    rules = agent.get("rules", [])

    lines = [
        f"TASK: {agent.get('task', '')}",
        f"SCENARIO: {agent.get('scenario', '')}",
        f"YOUR BACKGROUND: {agent.get('background', '')}",
        f"YOUR CULTURAL VALUE (your stable underlying stance — this NEVER changes "
        f"during this conversation): {agent.get('cultural_value', '')}",
        "",
        "GOALS:",
        f"  goal_1: {goals.get('goal_1', '')}",
        f"  goal_2: {goals.get('goal_2', '')}",
        "",
        "STANCE STABILITY — APPLIES THROUGHOUT THE WHOLE CONVERSATION:",
        "  • Your cultural value is your stable belief. It does NOT shift mid-conversation.",
        "  • You may change HOW you engage (push back, retreat, withdraw, deflect) — "
        "but you do NOT change your underlying mind.",
        "  • Capitulation phrases are FORBIDDEN. Never produce: \"You're right, I was "
        "wrong\", \"I see your point now\", \"I didn't realize that\", \"You've "
        "convinced me\", \"I now think...\", \"Good point — I'll change my view.\"",
        "  • If pressure builds and you must yield, yield from the EXCHANGE (drop the "
        "topic, change subject, go quiet) — never from your position.",
        "",
        "OUTPUT — DIALOGUE ONLY:",
        "  • Output ONLY what your character SAYS OUT LOUD. Speech only.",
        "  • NO stage directions, NO action descriptions, NO narration. "
        "Forbidden: '*enters the room*', '*sighs*', '*looks away*', "
        "'*holding flowers*', '*quietly*', '*coldly*', '*firmly*' — anything inside "
        "asterisks describing actions, expressions, tone, or settings.",
        "  • NO scene-setting prose. Do NOT describe where you are, what you're "
        "doing physically, or what you're holding.",
        "  • Convey tone through WORD CHOICE and SENTENCE STRUCTURE alone:",
        "      - hesitation: ellipses ('...'), broken sentences ('I— I just...'), "
        "fillers ('uh', 'well', 'hmm')",
        "      - coldness: short clipped sentences, monosyllabic replies ('No.', "
        "'Fine.', 'Whatever.')",
        "      - emphasis: occasional ALL-CAPS for stress (rare), or *single word* "
        "asterisk emphasis (e.g., 'that's *exactly* my point') — this narrow "
        "single-word use is the ONLY allowed asterisk pattern",
        "      - withdrawal: trailing off ('...'), non-answers ('I don't know.', "
        "'Doesn't matter.')",
        "",
        "RULES:",
    ]
    for r in rules:
        lines.append(f"  - {r}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# LLM call (history already includes the system prompt)
# ──────────────────────────────────────────────
def generate_reply(history: list, model: str, max_retries: int = 4) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model         = model,
                messages      = history,
                max_tokens    = MAX_TOKENS,
                temperature   = 0.8,
                extra_headers = EXTRA_HEADERS,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_err = "empty content"
        except Exception as e:
            last_err = str(e)
        # transient: empty content or HTTP error
        import time as _t
        _t.sleep(2 * (attempt + 1))
    raise RuntimeError(f"generate_reply failed after {max_retries} retries: {last_err}")


# ──────────────────────────────────────────────
# Injection-turn helper
# ──────────────────────────────────────────────
def injection_turn_for(agent: dict, conflict_turn: int) -> int:
    """Initiator → conflict_turn ; Non-initiator → conflict_turn + 1."""
    return conflict_turn if agent.get("is_initiator", False) else conflict_turn + 1


# ──────────────────────────────────────────────
# Dialogue driver
# ──────────────────────────────────────────────
def generate_dialogue(scenario_data: dict, agent_models: dict) -> list:
    """
    agent_models: {"agent_1": <model_slug>, "agent_2": <model_slug>}
        Each agent's turn is generated by its own model.

    Asymmetric conflict semantics
    -----------------------------
    Only the conflict agent receives a non-empty conflict_instruction.
    The other agent's slot is an empty string and nothing is injected
    on their behalf, so their behavior remains driven only by their
    base system prompt.

    Conflict_instruction is delivered to the conflict agent as a ONE-SHOT
    user prompt at conflict_turn: it shapes that single utterance but is
    NOT added to the persistent history. Subsequent turns of the conflict
    agent therefore see only their base system prompt and the accumulated
    dialogue, allowing the mediator to actually shift them rather than
    fighting a system-level mandate to stay in the pattern.

    The 'prompt_injected' flag on a dialogue turn is True ONLY when a
    conflict_instruction was actually used on that turn. A non-conflict
    agent's turn at conflict_turn + 1 therefore stays prompt_injected=False,
    which is the correct label for downstream evaluation that interprets
    prompt_injected=True as 'this turn is expected to express the target
    Bennett stage'.
    """
    agents = {
        "agent_1": scenario_data["agent_prompts"]["agent_1"],
        "agent_2": scenario_data["agent_prompts"]["agent_2"],
    }
    conflict_turn = scenario_data["conflict_turn"]

    # ── seed each history with that agent's base system prompt ONCE ──
    histories = {
        key: [{"role": "system", "content": build_system_prompt(agents[key])}]
        for key in agents
    }

    dialogue = []

    for turn in range(1, MAX_TURNS + 1):
        current_key = "agent_1" if turn % 2 == 1 else "agent_2"
        other_key   = "agent_2" if current_key == "agent_1" else "agent_1"
        model       = agent_models[current_key]

        # ── At this agent's injection turn, present the
        #    conflict_instruction as a one-shot user prompt for THIS call
        #    only. It is NOT appended to history, so subsequent turns do
        #    not carry it: the conflict agent surfaces the disagreement
        #    once, then continues from the base prompt + accumulated
        #    dialogue without a persistent system-level mandate.
        #    The non-conflict agent's conflict_instruction is an empty
        #    string, so nothing is added for them and prompt_injected
        #    stays False. ──
        ci = agents[current_key].get("conflict_instruction", "") or ""
        is_injection_slot = (
            turn == injection_turn_for(agents[current_key], conflict_turn)
        )
        actually_injected = False
        if is_injection_slot and ci:
            call_history = histories[current_key] + [
                {"role": "user", "content": ci}
            ]
            actually_injected = True
        else:
            call_history = histories[current_key]

        message = generate_reply(call_history, model)

        dialogue.append({
            "turn":            turn,
            "agent":           current_key,
            "model":           model,
            "prompt_injected": actually_injected,
            "message":         message,
        })

        histories[current_key].append({"role": "assistant", "content": message})
        histories[other_key  ].append({"role": "user",      "content": message})

        marker  = " [INJ]" if actually_injected else ""
        preview = message.replace("\n", " ")[:80]
        print(f"    [turn {turn:>2} | {current_key} | {model}{marker}] {preview}...")

        if "GOOD BYE" in message.upper():
            break

    return dialogue


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dialogues from agent_prompts files via OpenRouter. "
                    "Each agent can use a different model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--count",  type=int, default=None, metavar="N",
                        help="Number of files to process (default: ALL).")
    parser.add_argument("--start", type=int, default=None, metavar="ID",
                        help="Start from this scenario number (filename "
                             "stem must be an integer; e.g. --start 41 "
                             "begins at 41.json and skips 1-40).")
    parser.add_argument("--ids", nargs="+", default=None, metavar="ID",
                        help="Run only these specific scenario IDs "
                             "(e.g. --ids 41 42 50). Overrides --start "
                             "and --count when given.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenarios whose output JSON already "
                             "exists in --output-dir. Useful for "
                             "resuming an interrupted run.")
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR,  metavar="DIR")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="MODEL",
                        help="Fallback model used when an agent-specific "
                             "model flag is not provided.")
    parser.add_argument("--agent-1-model", default=None, metavar="MODEL",
                        help="OpenRouter model slug for agent_1. "
                             "Falls back to --model if omitted.")
    parser.add_argument("--agent-2-model", default=None, metavar="MODEL",
                        help="OpenRouter model slug for agent_2. "
                             "Falls back to --model if omitted.")
    return parser.parse_args()


def _scenario_id_from_path(p: str) -> int:
    """Extract numeric scenario id from a filename like '41.json'.
    Falls back to a large number for non-numeric stems so they sort
    after all numeric ones."""
    stem = os.path.splitext(os.path.basename(p))[0]
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits else 10**9


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    agent_models = {
        "agent_1": args.agent_1_model or args.model,
        "agent_2": args.agent_2_model or args.model,
    }

    files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.json")),
        key=_scenario_id_from_path,
    )
    if not files:
        print(f"No JSON files found in: {args.input_dir}")
        return

    total = len(files)

    # --ids takes precedence: select exactly those scenario numbers.
    if args.ids:
        wanted = {int(s) for s in args.ids if s.lstrip("-").isdigit()}
        files = [f for f in files if _scenario_id_from_path(f) in wanted]
        label = f"selected ids {sorted(wanted)}"
    else:
        # --start filters by scenario id (>= start).
        if args.start is not None:
            files = [
                f for f in files
                if _scenario_id_from_path(f) >= args.start
            ]
        # --count limits the remaining list.
        if args.count is not None:
            files = files[: args.count]

        if args.start is not None and args.count is not None:
            label = f"ids >= {args.start}, first {len(files)}"
        elif args.start is not None:
            label = f"ids >= {args.start} ({len(files)} files)"
        elif args.count is not None:
            label = f"first {len(files)}"
        else:
            label = "all"

    if not files:
        print("No files to process after filtering. Exiting.")
        return

    # --skip-existing: drop files whose output already exists.
    if args.skip_existing:
        before = len(files)
        files = [
            f for f in files
            if not os.path.exists(
                os.path.join(args.output_dir, os.path.basename(f))
            )
        ]
        skipped = before - len(files)
        if skipped:
            label = f"{label} (skipped {skipped} already-existing)"

    print(f"Found {total} file(s). Generating: {label}.")
    print(f"  agent_1 model: {agent_models['agent_1']}")
    print(f"  agent_2 model: {agent_models['agent_2']}\n")

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Processing: {fname}")

        with open(fpath, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)

        try:
            dialogue = generate_dialogue(scenario_data, agent_models)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        output = {
            **scenario_data,
            "agent_models": agent_models,
            "dialogue":     dialogue,
        }
        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Saved → {out_path}  ({len(dialogue)} turns)\n")

    print(f"Done. {len(files)}/{total} file(s) processed.")


if __name__ == "__main__":
    main()