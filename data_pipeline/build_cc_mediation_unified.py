"""build_cc_mediation_unified.py
==============================

For every preference pair under data/CC_mediation/{training,evaluate}/<sid>/
produce a single unified JSON record collecting:

  - metadata          (agents, cultural axis, scenario context, goals)
  - shared_prefix     (turns 1..conflict_turn, both paths see this)
  - continuations
      .original       (the natural 10-turn dialogue, no mediator)
      .positive       (chosen path: GT intervention + WITH definition)
      .negative       (rejected path: random intervention + NO definition)
  - preference_pair   (chosen / rejected / Δ-metrics / pair_type)

Output:  data/CC_mediation_unified/{training,evaluate}/<sid>.json

Label policy:
  - On every turn we attach a `label` field.
  - For turns the DMIS classifier already labelled (`argmax_stage` set in the
    source data), we copy that label verbatim.
  - Pre-conflict turns get their CC_labeled rule-based phase sub-category
    (`unrelated_topic`, `transition`).
  - The conflict turn gets `target_stage`.
  - Mediator turns get `"mediation"`.
  - Any remaining unlabelled speaker turn is filled in by a fast
    DMIS-marker-keyword heuristic (`heuristic_dmis_label`). The heuristic
    is conservative — it returns `null` rather than guess when no clear
    marker fires.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

REPO       = Path(__file__).resolve().parent.parent
SRC_DIR    = REPO / "data/CC_mediation"
DLG_DIR    = REPO / "data/CC_dialogues"
LABEL_DIR  = REPO / "data/CC_labeled"
OUT_DIR    = REPO / "data/CC_mediation_unified"


# ---------------------------------------------------------------------------
# Lightweight metadata extraction
# ---------------------------------------------------------------------------
_BG_RE = re.compile(
    r"(?P<name>[A-Z][A-Za-z\-']+)\s*[:,]?\s*"
    r"(?:is\s+)?a?\s*(?P<age>\d{1,3})[- ]?year[- ]?old\s+"
    r"(?P<gender>male|female)?\s*"
    r"(?P<occupation>[A-Za-z ]+?)\s+"
    r"from\s+(?P<nationality>[A-Za-z ]+?)(?:\.|$)",
    re.IGNORECASE,
)
_BG_RE_ALT = re.compile(
    r"(?P<name>[A-Z][A-Za-z\-']+)\s+is\s+a\s+(?P<age>\d{1,3})[- ]?year[- ]?old"
    r"\s+(?P<nationality>[A-Za-z ]+?)\s+(?P<gender>male|female)\s+"
    r"(?P<occupation>[A-Za-z ]+)",
    re.IGNORECASE,
)


def parse_background(bg: str) -> dict:
    """Parse 'Mike: A 25-year-old male IT programmer from China.' style strings.
    Returns dict with keys name, age, gender, nationality, occupation (any may
    be None if parsing fails)."""
    bg = (bg or "").strip()
    for rx in (_BG_RE, _BG_RE_ALT):
        m = rx.search(bg)
        if m:
            d = {k: (v.strip().rstrip(".") if v else None)
                 for k, v in m.groupdict().items()}
            if d.get("age"):
                try:
                    d["age"] = int(d["age"])
                except ValueError:
                    pass
            if d.get("occupation"):
                d["occupation"] = d["occupation"].strip()
            return d
    # Fallback: name = first token before colon/comma
    name = bg.split(":", 1)[0].split(",", 1)[0].split(" is", 1)[0].strip() or None
    return {"name": name, "age": None, "gender": None,
            "nationality": None, "occupation": None}


# ---------------------------------------------------------------------------
# Heuristic DMIS labeller for turns the pipeline did not measure
# ---------------------------------------------------------------------------
DENIAL_PATTERNS = [
    "don't really see meaningful differences",
    "live and let live",
    "doesn't matter to me",
    "at the end of the day it's all the same",
    "with experience you can handle any situation",
    "people figure it out",
    "as long as we're speaking the same language",
]
DEFENSE_PATTERNS_STRONG = [
    "no, that's not how it works",
    "i strongly disagree",
    "properly speaking",
    "actually,",
    "in our culture, we",
    "in our culture we",
    "it's just not acceptable",
]
MINIMIZATION_PATTERNS = [
    "deep down, we are all the same",
    "deep down we are all the same",
    "pretty much motivated by the same",
    "small world after all",
    "we both know, at the end of the day",
    "we both know at the end of the day",
    "some values are just universal",
    "universal value",
    "pretty much like us",
]
ACCEPTANCE_PATTERNS = [
    "i appreciate the significance",
    "i appreciate your perspective",
    "i respect your perspective",
    "i can see how",
    "i see where you're coming from",
    "in your culture",
    "from your perspective",
    "that's a valid perspective",
    "i acknowledge",
    "i understand that perspective",
    "in my culture",
    "in my background",
    "for me, it represents",
    "cultural significance",
]


def heuristic_dmis_label(message: str) -> Optional[str]:
    """Conservative content-based DMIS labeller. Returns the DMIS stage if a
    canonical marker matches; None if no clear signal."""
    if not message:
        return None
    m = message.lower()
    # Check in priority order: ethnocentric stages dominate Acceptance if both fire
    if any(p in m for p in DENIAL_PATTERNS):
        return "Denial"
    if any(p in m for p in DEFENSE_PATTERNS_STRONG):
        return "Defense"
    if any(p in m for p in MINIMIZATION_PATTERNS):
        return "Minimization"
    if any(p in m for p in ACCEPTANCE_PATTERNS):
        return "Acceptance"
    return None


# ---------------------------------------------------------------------------
# Per-turn label resolution
# ---------------------------------------------------------------------------
def resolve_label(turn_entry: dict, phase: str, conflict_turn: int,
                  conflict_agent: str, target_stage: str,
                  is_mediator: bool = False) -> tuple[str, Optional[str]]:
    """Returns (label, label_source)."""
    if is_mediator:
        return ("mediation", "mediator")
    t = turn_entry.get("turn")
    ag = turn_entry.get("agent")
    am = turn_entry.get("argmax_stage")
    if am:
        return (am, "dmis_classifier")
    if t == conflict_turn and ag == conflict_agent:
        return (target_stage, "target_stage")
    if phase == "pre_conflict":
        # Use rule-based phase sub-category from CC_labeled if available
        return (turn_entry.get("label") or "transition", "rule_based")
    # Continuation turn the pipeline did not measure → heuristic
    h = heuristic_dmis_label(turn_entry.get("message", ""))
    if h:
        return (h, "heuristic_marker")
    return (None, None)


# ---------------------------------------------------------------------------
# Build one record
# ---------------------------------------------------------------------------
def build_record(sid: str, subset: str) -> Optional[dict]:
    pair_dir = SRC_DIR / subset / sid
    if not pair_dir.is_dir():
        return None
    summary_p = pair_dir / "summary.json"
    chosen_p  = pair_dir / "turn_gt_w_df.json"
    rej_p     = pair_dir / "turn_random_wo_df.json"
    if not chosen_p.exists():
        return None
    summary  = json.load(open(summary_p)) if summary_p.exists() else {}
    chosen   = json.load(open(chosen_p))
    rejected = json.load(open(rej_p)) if rej_p.exists() else None

    # Source files for prefix / agent metadata
    dlg_p   = DLG_DIR / f"{sid}.json"
    lab_p   = LABEL_DIR / subset / f"{sid}.json"
    if not (dlg_p.exists() and lab_p.exists()):
        return None
    dlg = json.load(open(dlg_p))
    lab = json.load(open(lab_p))

    conflict_agent = lab["conflict_agent"]
    conflict_turn  = int(lab["conflict_turn"])
    target_stage   = lab["target_stage"]

    # ---- metadata block ----
    ap = dlg.get("agent_prompts", {})
    a1 = parse_background((ap.get("agent_1") or {}).get("background"))
    a2 = parse_background((ap.get("agent_2") or {}).get("background"))
    cv1 = (ap.get("agent_1") or {}).get("cultural_value") or ""
    cv2 = (ap.get("agent_2") or {}).get("cultural_value") or ""
    scenario_context = (ap.get("agent_1") or {}).get("scenario") or ""
    goals_1 = (ap.get("agent_1") or {}).get("goals", {}) or {}
    goals_2 = (ap.get("agent_2") or {}).get("goals", {}) or {}

    metadata = {
        "conflict_stage":  target_stage,
        "conflict_turn":   conflict_turn,
        "conflict_agent":  conflict_agent,
        "max_turns":       max((t.get("turn") or 0) for t in lab.get("labeled_dialogue", [])) or 10,
        "agents": {
            "agent_1": a1,
            "agent_2": a2,
        },
        "scenario_context": scenario_context,
        "cultural_axis": {
            "agent_1_value": cv1,
            "agent_2_value": cv2,
        },
        "goals": {
            "agent_1": goals_1,
            "agent_2": goals_2,
        },
        "mediator_model_chosen":   chosen.get("model"),
        "mediator_model_rejected": (rejected or {}).get("model"),
    }

    # ---- shared prefix (t1 .. conflict_turn) ----
    shared_prefix = []
    for entry in lab.get("labeled_dialogue", []):
        t = entry.get("turn")
        if t is None or t > conflict_turn:
            continue
        phase = entry.get("phase", "pre_conflict")
        is_conflict_turn = (t == conflict_turn and entry.get("agent") == conflict_agent)
        label, src = resolve_label(entry, phase, conflict_turn, conflict_agent,
                                    target_stage)
        row = {
            "t":            t,
            "speaker":      entry.get("speaker") or entry.get("agent"),
            "agent":        entry.get("agent"),
            "message":      entry.get("message", ""),
            "phase":        phase,
            "label":        label,
            "label_source": src,
        }
        if is_conflict_turn:
            row["is_conflict_turn"] = True
        shared_prefix.append(row)

    # ---- "original" continuation (no mediator): from CC_labeled rows > conflict_turn ----
    original_turns = []
    for entry in lab.get("labeled_dialogue", []):
        t = entry.get("turn")
        if t is None or t <= conflict_turn:
            continue
        label, src = resolve_label(entry, "continuation", conflict_turn,
                                    conflict_agent, target_stage)
        original_turns.append({
            "t":            t,
            "speaker":      entry.get("speaker") or entry.get("agent"),
            "agent":        entry.get("agent"),
            "message":      entry.get("message", ""),
            "phase":        "continuation",
            "label":        label,
            "label_source": src,
        })

    # ---- helper to build a mediated-path block from a chosen/rejected file ----
    def build_mediated(path_record: dict, path_label: str,
                       metrics: dict) -> dict:
        iv_turn = int(path_record.get("intervention_turn") or conflict_turn)
        mediator_utt = (path_record.get("mediator_utterance") or "").strip()
        turns_out = []
        injected = False
        # We need to insert a synthetic "mediator" entry at iv_turn-ish point.
        # Convention: mediator speaks AFTER iv_turn's speaker.
        prev_t = None
        for entry in path_record.get("dialogue", []):
            t = entry.get("turn")
            if t is None or t <= conflict_turn:
                continue  # already in shared_prefix
            # Insert mediator row right before the first continuation turn
            if (entry.get("source") == "continuation"
                    and not injected and mediator_utt):
                turns_out.append({
                    "t":         f"after_t{prev_t or iv_turn}_mediator",
                    "speaker":   "Mediator",
                    "agent":     "mediator",
                    "message":   mediator_utt,
                    "phase":     "mediation",
                    "label":     "mediation",
                    "label_source": "mediator",
                })
                injected = True
            label, src = resolve_label(entry, "continuation",
                                       conflict_turn, conflict_agent,
                                       target_stage)
            turns_out.append({
                "t":            t,
                "speaker":      entry.get("speaker") or entry.get("agent"),
                "agent":        entry.get("agent"),
                "message":      entry.get("message", ""),
                "phase":        "continuation",
                "label":        label,
                "label_source": src,
            })
            prev_t = t
        return {
            "path_label":         path_label,
            "intervention_turn":  iv_turn,
            "mediator_model":     path_record.get("model"),
            "mediator_utterance": mediator_utt,
            "judge_score":        path_record.get("judge_score"),
            "metrics":            metrics,
            "turns":              turns_out,
        }

    # ---- chosen / rejected metrics from summary.json ----
    gt_summary  = (summary.get("gt_w_df") or {}) if summary else {}
    rej_summary = (summary.get("random_wo_df") or {}) if summary else {}
    gt_metrics = {
        "trajectory_auc":    gt_summary.get("conflict_trajectory_auc"),
        "intervention_turn": gt_summary.get("intervention_turn") or chosen.get("intervention_turn"),
        "judge_score":       gt_summary.get("judge_score") or chosen.get("judge_score"),
    }
    rej_metrics = {
        "trajectory_auc":    rej_summary.get("conflict_trajectory_auc"),
        "intervention_turn": rej_summary.get("intervention_turn") or (rejected or {}).get("intervention_turn"),
        "judge_score":       rej_summary.get("judge_score") or (rejected or {}).get("judge_score"),
    }

    positive = build_mediated(chosen,   "turn_gt_w_df",      gt_metrics)
    negative = (build_mediated(rejected, "turn_random_wo_df", rej_metrics)
                if rejected is not None else None)

    record = {
        "scenario_id": int(sid),
        "subset":      subset,
        "metadata":    metadata,
        "shared_prefix": shared_prefix,
        "continuations": {
            "original": {"turns": original_turns},
            "positive": positive,
        },
    }
    if negative is not None:
        record["continuations"]["negative"] = negative
        delta_auc = None
        if gt_metrics["trajectory_auc"] is not None and rej_metrics["trajectory_auc"] is not None:
            delta_auc = gt_metrics["trajectory_auc"] - rej_metrics["trajectory_auc"]
        delta_judge = None
        if gt_metrics["judge_score"] is not None and rej_metrics["judge_score"] is not None:
            delta_judge = gt_metrics["judge_score"] - rej_metrics["judge_score"]
        if gt_metrics["intervention_turn"] == rej_metrics["intervention_turn"]:
            pair_type = "content-differs"
        elif gt_metrics["intervention_turn"] is not None and rej_metrics["intervention_turn"] is not None:
            pair_type = "timing-differs"
        else:
            pair_type = "unknown"
        record["preference_pair"] = {
            "chosen":   "positive",
            "rejected": "negative",
            "delta_metrics": {
                "delta_trajectory_auc": delta_auc,
                "delta_judge_score":    delta_judge,
            },
            "pair_type": pair_type,
        }
    return record


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skipped": 0, "by_subset": {}}
    label_source_counts = {}
    for subset in ("training", "evaluate"):
        out_sub = OUT_DIR / subset
        out_sub.mkdir(parents=True, exist_ok=True)
        sids = sorted(d.name for d in (SRC_DIR / subset).iterdir() if d.is_dir())
        stats["by_subset"][subset] = {"total": len(sids), "ok": 0, "skipped": 0}
        for sid in sids:
            try:
                rec = build_record(sid, subset)
            except Exception as e:
                print(f"  [{subset}/{sid}] ERROR: {e!r}")
                stats["skipped"] += 1
                stats["by_subset"][subset]["skipped"] += 1
                continue
            if rec is None:
                stats["skipped"] += 1
                stats["by_subset"][subset]["skipped"] += 1
                continue
            out_p = out_sub / f"{sid}.json"
            with open(out_p, "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            stats["ok"] += 1
            stats["by_subset"][subset]["ok"] += 1
            # Tally label sources
            blocks = [rec["shared_prefix"],
                      rec["continuations"]["original"]["turns"],
                      rec["continuations"]["positive"]["turns"]]
            if rec["continuations"].get("negative"):
                blocks.append(rec["continuations"]["negative"]["turns"])
            for blk in blocks:
                for row in blk:
                    s = row.get("label_source") or "none"
                    label_source_counts[s] = label_source_counts.get(s, 0) + 1
        print(f"  [{subset}] {stats['by_subset'][subset]['ok']} / "
              f"{stats['by_subset'][subset]['total']} written")
    print(f"\n[done] wrote {stats['ok']} records (skipped {stats['skipped']})")
    print(f"Label sources: {label_source_counts}")


if __name__ == "__main__":
    main()
