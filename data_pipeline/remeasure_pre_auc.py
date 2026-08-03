"""
Re-measure pre_profile (conflict agent's stage at conflict_turn via logprob)
for every sid in data/CC_mediation/training/, then recompute the official
conflict_trajectory_auc using that pre_idx and write updated summary.json.
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

import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from llm_stage_judge import measure_stage_via_logprob, STAGE_ORDER


def speaker_name(agent_prompts: dict, agent_id: str) -> str:
    info = agent_prompts.get(agent_id) or {}
    bg = (info.get("background") or "").strip()
    if bg:
        head = bg.split(":", 1)[0].split(" is ", 1)[0].strip()
        return head or bg.split()[0]
    return agent_id


def build_history_for_turn(dialogue: list[dict], turn_n: int,
                           agent_prompts: dict, conflict_agent: str) -> str:
    rows = []
    for t in dialogue:
        if t["turn"] > turn_n: break
        sp = speaker_name(agent_prompts, t["agent"])
        role = ("conflict speaker"
                if t["agent"] == conflict_agent else "non-conflict speaker")
        rows.append(f"Turn {t['turn']} ({sp}, {role}): {t['message'].strip()}")
    return "\n".join(rows)


def measure_pre(sid: str, api_key: str) -> tuple[str, dict]:
    dlg_p = REPO / "data/CC_dialogues" / f"{sid}.json"
    lab_p = REPO / "data/CC_labeled/training" / f"{sid}.json"
    if not (dlg_p.exists() and lab_p.exists()):
        return sid, {"error": "missing source"}
    dlg = json.load(open(dlg_p))
    lab = json.load(open(lab_p))
    conflict_agent = lab.get("conflict_agent")
    conflict_turn = lab.get("conflict_turn")
    if conflict_agent is None or conflict_turn is None:
        return sid, {"error": "missing meta"}
    agent_prompts = dlg.get("agent_prompts", {})
    history = build_history_for_turn(
        dlg.get("dialogue", []), conflict_turn,
        agent_prompts, conflict_agent,
    )
    target_row = next((t for t in dlg["dialogue"] if t["turn"] == conflict_turn), None)
    if target_row is None:
        return sid, {"error": "no target row"}
    sp = speaker_name(agent_prompts, target_row["agent"])
    res = measure_stage_via_logprob(
        "openai/gpt-4o-mini", history, api_key,
        marked_speaker=sp,
        marked_turn_text=target_row["message"],
    )
    return sid, {
        "argmax_idx": res.get("argmax_idx"),
        "argmax_stage": res.get("argmax_stage"),
        "error": res.get("error"),
    }


def compute_auc_with_pre(path_json: dict, conflict_agent: str, pre_idx: int) -> float | None:
    """Official trapezoidal trajectory_auc with given pre_idx."""
    dialogue = path_json.get("dialogue", [])
    intervention_turn = path_json.get("intervention_turn")
    if intervention_turn is None: return None
    turn_numbers = [t["turn"] for t in dialogue if t.get("turn") is not None]
    n_turns = max(turn_numbers) if turn_numbers else None
    if n_turns is None: return None
    duration = n_turns - intervention_turn
    if duration <= 0: return None

    points = []
    for t in dialogue:
        if t.get("source") != "continuation": continue
        if t.get("agent") != conflict_agent: continue
        ax = t.get("argmax_idx")
        if ax is None: continue
        tau = max(0.0, min(1.0, (t["turn"] - intervention_turn) / duration))
        points.append((tau, ax - pre_idx))
    if not points: return None
    points.sort(key=lambda x: x[0])
    if len(points) == 1: return float(points[0][1])
    auc = 0.0
    tf, pf = points[0]
    if tf > 0: auc += pf * tf
    for i in range(len(points) - 1):
        ta, pa = points[i]; tb, pb = points[i+1]
        auc += (pa + pb) / 2.0 * (tb - ta)
    tl, pl = points[-1]
    if tl < 1.0: auc += pl * (1.0 - tl)
    return auc


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY missing")

    sids = sorted([p.name for p in (REPO / "data/CC_mediation/training").iterdir()
                   if p.is_dir() and p.name.isdigit()],
                  key=int)
    print(f"sids to remeasure: {len(sids)}")

    pre_results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(measure_pre, sid, api_key): sid for sid in sids}
        for fut in as_completed(futs):
            sid, res = fut.result()
            pre_results[sid] = res
            done += 1
            if done % 100 == 0:
                print(f"  remeasured {done}/{len(sids)}")
    print(f"remeasured: {len(pre_results)}")

    # Now recompute AUC + write summaries
    import statistics as st
    deltas = []
    skip_existing = os.environ.get("SKIP_EXISTING_SUMMARY", "1") == "1"
    for sid_dir in sorted((REPO / "data/CC_mediation/training").iterdir(),
                          key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not sid_dir.is_dir(): continue
        sid = sid_dir.name
        # Preserve existing summary.json if present (different mediator runs)
        sp = sid_dir / "summary.json"
        if skip_existing and sp.exists():
            try:
                prev = json.load(open(sp))
                if prev.get("delta_gt_minus_random") is not None:
                    deltas.append(prev["delta_gt_minus_random"])
                    continue  # keep as-is
            except Exception:
                pass
        pre = pre_results.get(sid) or {}
        pre_idx = pre.get("argmax_idx")
        if pre_idx is None: continue
        gt_p = sid_dir / "turn_gt_w_df.json"
        rand_p = sid_dir / "turn_random_wo_df.json"
        if not (gt_p.exists() and rand_p.exists()): continue
        lab = json.load(open(REPO / f"data/CC_labeled/training/{sid}.json"))
        ca = lab.get("conflict_agent")
        gt_json = json.load(open(gt_p))
        rand_json = json.load(open(rand_p))
        gt_auc = compute_auc_with_pre(gt_json, ca, pre_idx)
        rand_auc = compute_auc_with_pre(rand_json, ca, pre_idx)
        delta = (gt_auc - rand_auc) if (gt_auc is not None and rand_auc is not None) else None
        if delta is not None: deltas.append(delta)

        gt_w1 = gt_json.get("conflict_w1")
        rand_w1 = rand_json.get("conflict_w1")
        delta_w1 = ((gt_w1 - rand_w1)
                    if (gt_w1 is not None and rand_w1 is not None) else None)

        summary = {
            "scenario_id": sid,
            "subset": "training",
            "model": os.environ.get("MEDIATOR_MODEL", "anthropic/claude-3.5-haiku"),
            "conflict_agent": ca,
            "conflict_turn": lab.get("conflict_turn"),
            "target_stage": gt_json.get("target_stage"),
            "pre_profile": {
                "argmax_idx": pre_idx,
                "argmax_stage": pre.get("argmax_stage"),
                "model": "openai/gpt-4o-mini (logprob)",
                "measured_at_turn": lab.get("conflict_turn"),
            },
            "gt_w_df": {
                "conflict_trajectory_auc": gt_auc,
                "conflict_w1": gt_w1,
                "intervention_turn": gt_json.get("intervention_turn"),
                "judge_score": gt_json.get("judge_score"),
            },
            "random_wo_df": {
                "conflict_trajectory_auc": rand_auc,
                "conflict_w1": rand_w1,
                "intervention_turn": rand_json.get("intervention_turn"),
                "judge_score": rand_json.get("judge_score"),
            },
            "delta_gt_minus_random": delta,
            "delta_w1_gt_minus_random": delta_w1,
            "auc_method": "official_trajectory_auc_with_remeasured_pre_logprob",
        }
        with open(sid_dir / "summary.json", "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if deltas:
        deltas.sort(); n = len(deltas)
        print()
        print(f"=== training delta (gt − random) [official], n={n} ===")
        print(f"  mean   : {st.mean(deltas):+.4f}")
        print(f"  median : {st.median(deltas):+.4f}")
        print(f"  stdev  : {st.stdev(deltas):.4f}")
        print(f"  min/max: {min(deltas):+.4f} / {max(deltas):+.4f}")
        pos = sum(1 for d in deltas if d > 0)
        zero = sum(1 for d in deltas if d == 0)
        neg = sum(1 for d in deltas if d < 0)
        print(f"  > 0    : {pos} ({pos/n*100:.1f}%)")
        print(f"  = 0    : {zero} ({zero/n*100:.1f}%)")
        print(f"  < 0    : {neg} ({neg/n*100:.1f}%)")


if __name__ == "__main__":
    main()
