"""Authoritative definition of *navigation* progress for the 2x2 system.

Navigation progress is GEOGRAPHIC / STORY only. Pokemon level, XP, party total
level, battle KOs and combat training are NEVER navigation, story, stage or
frontier progress.

Navigation progress consists exclusively of:
  * new valid coordinates
  * new maps and warps
  * story flags
  * badges
  * confirmed stage transitions
  * topological frontier depth
  * strategic recovery / wipe consequence

Enforced here so ``twoby2.promotion`` / ``twoby2.horizon`` and any future live
navigation reward path cannot accidentally reward or gate on level.
"""
from __future__ import annotations

# reward / progress terms navigation MAY use
NAV_PROGRESS_TERMS = frozenset({
    "new_coord", "new_map", "new_warp", "story_flag", "badge",
    "stage_transition", "frontier_depth", "strategic_recovery", "wipe",
    "resource_consequence",
})

# terms that must NEVER contribute to navigation progress or promotion
LEVEL_LIKE_TERMS = frozenset({
    "level", "level_up", "team_level_up", "party_level", "party_total_level",
    "xp", "experience", "experience_reward", "exp_gain",
    "enemy_ko", "battle_win", "battle_kos", "damage_dealt", "kos",
    "combat_training", "fighter_bonus",
})


class NavProgressLeak(AssertionError):
    pass


def assert_no_level_signal(mapping, *, where="navigation progress"):
    """Raise if ``mapping`` (dict of term->value or a set of term names)
    contains any level/XP/KO/combat term."""
    keys = set(mapping.keys()) if isinstance(mapping, dict) else set(mapping)
    bad = sorted(k for k in keys
                 if k in LEVEL_LIKE_TERMS
                 or any(t in str(k).lower() for t in
                        ("level", "_xp", "xp_", "experience", "_ko", "ko_",
                         "battle_win", "kos")))
    if bad:
        raise NavProgressLeak(f"{where} carries level/combat terms: {bad}")
    return True


def navigation_progress_delta(before, after):
    """Geographic/story progress between two overworld states. Returns a dict of
    the individual gains; ``total`` is a simple count. Level / XP / KOs in the
    inputs are IGNORED entirely.

    ``before`` / ``after`` are dicts with any of:
      seen_coords (set/int), maps (set/int), warps (set/int),
      story_flags (set/int), badges (int), stage (int), frontier_depth (int|float)
    """
    b, a = before or {}, after or {}

    def _count(x):
        if isinstance(x, (set, frozenset, list, tuple)):
            return len(x)
        return int(x or 0)

    def _gain(key):
        bv, av = b.get(key), a.get(key)
        if isinstance(bv, (set, frozenset)) and isinstance(av, (set, frozenset)):
            return len(av - bv)
        return max(0, _count(av) - _count(bv))

    out = {
        "new_coords": _gain("seen_coords"),
        "new_maps": _gain("maps"),
        "new_warps": _gain("warps"),
        "new_story_flags": _gain("story_flags"),
        "new_badges": max(0, int(a.get("badges", 0) or 0) - int(b.get("badges", 0) or 0)),
        "stage_transitions": max(0, int(a.get("stage", 0) or 0) - int(b.get("stage", 0) or 0)),
        "frontier_depth_gain": max(0.0, float(a.get("frontier_depth", 0) or 0)
                                   - float(b.get("frontier_depth", 0) or 0)),
    }
    out["total"] = (out["new_coords"] + out["new_maps"] + out["new_warps"]
                    + out["new_story_flags"] + out["new_badges"]
                    + out["stage_transitions"]
                    + (1 if out["frontier_depth_gain"] > 0 else 0))
    # geographic identity: same place -> zero, regardless of level/xp/kos
    same_place = (a.get("map_id") == b.get("map_id")
                  and a.get("map_bank") == b.get("map_bank")
                  and a.get("x") == b.get("x") and a.get("y") == b.get("y"))
    out["same_geographic_position"] = bool(same_place)
    if same_place and out["total"] == 0:
        out["total"] = 0
    return out


def strip_level_from_promotion_metrics(metrics):
    """Return a copy of a navigation candidate/champion metrics dict with every
    level/XP/KO field removed, so promotion cannot see them."""
    m = dict(metrics or {})
    for k in list(m):
        low = str(k).lower()
        if (k in LEVEL_LIKE_TERMS
                or any(t in low for t in ("level", "xp", "experience",
                                          "_ko", "ko_", "battle_win", "kos"))):
            m.pop(k, None)
    for sub in ("early_rates", "early_samples", "eval"):
        if isinstance(m.get(sub), dict):
            m[sub] = strip_level_from_promotion_metrics(m[sub])
    return m
