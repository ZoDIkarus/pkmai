"""Deterministic scheduling and durable quality tracking for PKMAI."""
import json
import os
from statistics import median

STAGE_ORDER = ("intro_complete", "stairs_down", "left_house", "starter")
STAGE_ROLE = {
    "intro_complete": "intro", "stairs_down": "stairs",
    "left_house": "exit", "starter": "starter",
}
STAGE_LABEL = {
    "intro_complete": "Intro abschließen",
    "stairs_down": "Treppe erreichen",
    "left_house": "Haus verlassen",
    "starter": "Starter erhalten",
}
MIN_EVALUATION_EPISODES = 10
MIN_SUCCESS_RATE = 0.60
RECENT_WINDOW = 20


def local_frontier_roles(agent_count, milestones):
    """Allocate a small local fleet by frontier while retaining every skill."""
    n = max(1, int(agent_count))
    milestones = set(milestones or ())
    if "intro_complete" not in milestones:
        plan = [("intro", 8), ("stairs", 2)]
    elif "stairs_down" not in milestones:
        plan = [("intro", 1), ("stairs", 8), ("exit", 1)]
    elif "left_house" not in milestones:
        plan = [("intro", 1), ("stairs", 1), ("exit", 8)]
    elif "starter" not in milestones:
        plan = [("intro", 1), ("stairs", 1), ("exit", 1), ("starter", 7)]
    else:
        plan = [
            ("intro", 1), ("stairs", 1), ("exit", 1), ("starter", 1),
            ("battle", 2), ("progress", 4),
        ]
    roles = [role for role, count in plan for _ in range(count)]
    return tuple((roles + [roles[-1]] * n)[:n])


def stage_speed_bonus(elapsed_steps, target_steps, max_bonus):
    elapsed_steps = max(0, int(elapsed_steps)); target_steps = max(1, int(target_steps))
    return round(max(0.0, float(max_bonus)) * max(0.0, 1.0 - elapsed_steps / target_steps), 4)


def adaptive_stage_limit(best_steps, fallback_limit):
    """Allow at most twice the best real success, capped by a safe fallback."""
    fallback_limit = max(1, int(fallback_limit))
    if best_steps is None:
        return fallback_limit
    try:
        best_steps = max(1, int(best_steps))
    except (TypeError, ValueError):
        return fallback_limit
    return min(fallback_limit, best_steps * 2)


def stage_is_confirmed(record):
    recent = list((record or {}).get("recent", ()))
    return len(recent) >= MIN_EVALUATION_EPISODES and sum(bool(x) for x in recent) / len(recent) >= MIN_SUCCESS_RATE


def confirmed_stages(milestones, status):
    milestones, status = set(milestones or ()), status or {}
    # Existing earlier savestates are usable regression states. The newest stage
    # still needs its rolling proof before it opens the following frontier.
    result = set()
    for stage in STAGE_ORDER:
        if stage not in milestones:
            break
        result.add(stage)
        if not stage_is_confirmed(status.get(stage)) and stage == "starter":
            break
    return result


def curriculum_roles(agent_count, milestones, status=None, watcher_validation=None):
    """Frontier majority plus one maintenance worker for every prior state."""
    n, milestones, status = max(1, int(agent_count)), set(milestones or ()), status or {}
    # The visual from-start watcher is an independent acceptance gate for the
    # first playable transition; stale trainer states cannot bypass it.
    if watcher_validation is not None and not bool(
        (watcher_validation.get("intro_complete") or {}).get("passed", False)
    ):
        return tuple((["intro"] * 8 + ["stairs"] * 2)[:n])
    # A state is a usable start point, not proof that the current policy is
    # reliable. Revalidate the full early chain after policy/reward changes.
    if "intro_complete" not in milestones or not stage_is_confirmed(status.get("intro_complete")):
        plan = [("intro", 8), ("stairs", 2)]
    elif "stairs_down" not in milestones or not stage_is_confirmed(status.get("stairs_down")):
        plan = [("intro", 1), ("stairs", 8), ("exit", 1)]
    elif "left_house" not in milestones or not stage_is_confirmed(status.get("left_house")):
        plan = [("intro", 1), ("stairs", 1), ("exit", 8)]
    elif "starter" not in milestones or not stage_is_confirmed(status.get("starter")):
        plan = [("intro", 1), ("stairs", 1), ("exit", 1), ("starter", 7)]
    else:
        plan = [("intro", 1), ("stairs", 1), ("exit", 1), ("starter", 1), ("battle", 2), ("progress", 4)]
    roles = [role for role, count in plan for _ in range(count)]
    return tuple((roles + [roles[-1]] * n)[:n])


def watcher_completion_stage(milestones, status=None):
    """Highest demonstrated stage the from-start watcher must reproduce."""
    milestones = set(milestones or ())
    for stage in ("starter", "left_house", "stairs_down", "intro_complete"):
        if stage in milestones:
            return stage
    return None


def watcher_start_state(milestones, status=None):
    milestones, status = set(milestones or ()), status or {}
    if "starter" in milestones and stage_is_confirmed(status.get("starter")):
        return "starter"
    for stage in ("left_house", "stairs_down", "intro_complete"):
        if stage in milestones:
            return stage
    return None


def load_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError, TypeError): return {"stages": {}}


def quality_snapshot(stages):
    """Comparable rolling policy quality: stage depth, reliability, then speed."""
    stages = stages or {}
    confirmed = sum(1 for record in stages.values() if stage_is_confirmed(record))
    recent = [bool(value) for record in stages.values() for value in record.get("recent", ())]
    successes = sum(recent)
    medians = [int(record["median_success_steps"]) for record in stages.values() if record.get("median_success_steps") is not None]
    # Tuples are deliberately ordered by the product priorities.
    return {
        "confirmed_stages": confirmed,
        "successes": successes,
        "attempts": len(recent),
        "speed_cost": sum(medians),
    }


def curriculum_progress_report(stages):
    """Format the rolling curriculum quality as a compact trainer log report."""
    stages = stages or {}
    lines = ["📚 Curriculum (letzte 20 Versuche je Stage):"]
    for stage in STAGE_ORDER:
        record = stages.get(stage, {})
        recent = list(record.get("recent", ()))
        label = STAGE_LABEL[stage]
        if not recent:
            lines.append(f"  ⏳ {label}: noch keine Messwerte")
            continue
        successes = sum(bool(value) for value in recent)
        attempts = len(recent)
        rate = round(successes / attempts * 100)
        median_steps = record.get("median_success_steps")
        speed = (
            f"Median {int(median_steps):,} Schritte ab Stage-Start"
            if median_steps is not None
            else "noch keine erfolgreiche Messung"
        )
        status = "✅ bestätigt" if stage_is_confirmed(record) else "🔄 wird geprüft"
        lines.append(
            f"  {status} · {label}: {successes}/{attempts} = {rate}% "
            f"| {speed} | insgesamt {int(record.get('successes', 0))}/"
            f"{int(record.get('attempts', 0))}"
        )
    return "\n".join(lines)


def quality_is_better(candidate, baseline):
    """Strictly compare quality without allowing faster failures to win."""
    candidate, baseline = candidate or {}, baseline or {}
    c_attempts, b_attempts = max(1, int(candidate.get("attempts", 0))), max(1, int(baseline.get("attempts", 0)))
    c_key = (int(candidate.get("confirmed_stages", 0)), int(candidate.get("successes", 0)) / c_attempts, -int(candidate.get("speed_cost", 0)))
    b_key = (int(baseline.get("confirmed_stages", 0)), int(baseline.get("successes", 0)) / b_attempts, -int(baseline.get("speed_cost", 0)))
    return c_key > b_key


def record_stage_result(path, stage, success, steps):
    data = load_status(path); stages = data.setdefault("stages", {}); record = stages.setdefault(stage, {"recent": [], "success_steps": []})
    record["recent"] = (list(record.get("recent", [])) + [bool(success)])[-RECENT_WINDOW:]
    if success: record["success_steps"] = (list(record.get("success_steps", [])) + [int(steps)])[-RECENT_WINDOW:]
    record["attempts"] = int(record.get("attempts", 0)) + 1; record["successes"] = int(record.get("successes", 0)) + int(bool(success))
    record["success_rate"] = round(sum(record["recent"]) / len(record["recent"]), 3)
    record["median_success_steps"] = int(median(record["success_steps"])) if record.get("success_steps") else None
    record["confirmed"] = stage_is_confirmed(record)
    tmp = path + ".tmp"; os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, sort_keys=True)
    os.replace(tmp, path); return data
