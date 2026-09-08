"""Reset baseline + Navigation/Battle reward-channel separation (2×2).

Two jobs:

1. **Reset baseline snapshot.** When the canonical post-parcel master is loaded,
   everything already present (starter, Pokédex, delivered parcel, initial
   party, initial level/XP, initial story flags, initial inventory, initial
   map/position) is worth exactly **0** reward. Only *changes after* the
   baseline snapshot are reward-eligible.

2. **Channel separation.** ``navigation_reward`` may only contain strategic /
   navigation terms; ``battle_reward`` may only contain combat terms. Neither
   double-counts a battle: the battle brain gets damage/KO/win, navigation gets
   only the strategic end state + wipe / resource loss + a genuinely
   newly-set story event.

The existing live reward code (``pokemon_env`` / ``reward_state`` /
``trainer_rewards``) is NOT rewritten here. This module implements the clean
split for the new 2×2 path; the switch-over is a later migration / feature-gate
step.
"""
from __future__ import annotations

NAV_REWARD_SCHEMA = "nav_reward_v1"
BATTLE_REWARD_SCHEMA = "battle_reward_v1"

# Reward terms navigation is ALLOWED to carry.
NAV_ALLOWED_TERMS = frozenset({
    "new_coord",            # a RAM coordinate first seen this episode
    "new_map", "new_warp",  # map / warp discovery
    "story_flag", "badge", "transition",   # story / gym / stage transition
    "strategic_recovery",   # healed / regrouped after a setback
    "stuck_penalty", "time_penalty",
    "wipe", "resource_consequence",        # coarse aftermath of a lost fight
    "battle_triggered_story_change",       # a story flag a battle actually set
})

# Reward terms navigation must NEVER carry.
NAV_FORBIDDEN_TERMS = frozenset({
    "damage_per_turn", "enemy_ko", "battle_win", "move_used", "menu_move",
    "cursor_move", "battle_advantage", "logit", "fighter_bonus",
    "per_turn_reward", "switch_reward",
})

# Reward terms battle is ALLOWED to carry.
BATTLE_ALLOWED_TERMS = frozenset({
    "damage_dealt", "enemy_ko", "battle_win", "own_faint", "wipe",
    "invalid_action", "wasted_turn", "switch_loop", "illegal_flee",
    "flee_wild", "turn_cost", "resource_use",
})

# Reward terms battle must NEVER carry.
BATTLE_FORBIDDEN_TERMS = frozenset({
    "new_coord", "new_map", "new_warp", "tile", "story_flag", "badge",
    "transition", "world_stage", "pokecenter", "overworld_reward",
    "navigation_transition",
})


class RewardChannelError(AssertionError):
    pass


# --------------------------------------------------------------------------
# reset baseline
# --------------------------------------------------------------------------
BASELINE_FIELDS = (
    "map_group", "map_id", "x", "y",
    "party_species", "party_levels", "party_total_xp",
    "story_flags", "badges", "pokedex_owned", "parcel_delivered",
    "inventory_item_ids", "money",
)


class ResetBaseline:
    """An immutable snapshot of the state that exists the instant the master is
    loaded. ``reward_for(now)`` returns only what changed."""

    SCHEMA = "reset_baseline_v1"

    def __init__(self, snapshot):
        self._b = {k: _freeze(snapshot.get(k)) for k in BASELINE_FIELDS}

    @classmethod
    def from_master_load(cls, snapshot):
        return cls(snapshot or {})

    def get(self, key):
        return self._b.get(key)

    def is_initial(self, key, value):
        """True when ``value`` for ``key`` is exactly what the baseline had —
        i.e. it must NOT be rewarded."""
        return _freeze(value) == self._b.get(key)

    def newly_owned_items(self, current_ids):
        base = set(self._b.get("inventory_item_ids") or ())
        return sorted(set(current_ids or ()) - base)

    def newly_set_story_flags(self, current_flags):
        base = set(self._b.get("story_flags") or ())
        return sorted(set(current_flags or ()) - base)

    def gained_badges(self, current_badges):
        return max(0, int(current_badges or 0) - int(self._b.get("badges") or 0))

    def moved_from_start(self, snapshot):
        s = snapshot or {}
        return (s.get("map_group"), s.get("map_id"), s.get("x"), s.get("y")) != (
            self._b.get("map_group"), self._b.get("map_id"),
            self._b.get("x"), self._b.get("y"))

    def to_dict(self):
        return {"schema": self.SCHEMA, "baseline": dict(self._b)}


def _freeze(v):
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, set):
        return tuple(sorted(_freeze(x) for x in v))
    return v


# --------------------------------------------------------------------------
# channel guards
# --------------------------------------------------------------------------
def assert_nav_reward_clean(terms):
    """``terms`` = dict {term_name: value}. Raise if any battle-micro term is
    present or any term is unknown."""
    bad = [t for t in terms if t in NAV_FORBIDDEN_TERMS]
    if bad:
        raise RewardChannelError(f"navigation reward carries battle terms: {bad}")
    unknown = [t for t in terms if t not in NAV_ALLOWED_TERMS]
    if unknown:
        raise RewardChannelError(f"navigation reward has unknown terms: {unknown}")
    return True


def assert_battle_reward_clean(terms):
    bad = [t for t in terms if t in BATTLE_FORBIDDEN_TERMS]
    if bad:
        raise RewardChannelError(f"battle reward carries navigation terms: {bad}")
    unknown = [t for t in terms if t not in BATTLE_ALLOWED_TERMS]
    if unknown:
        raise RewardChannelError(f"battle reward has unknown terms: {unknown}")
    return True


def navigation_reward(terms):
    assert_nav_reward_clean(terms)
    return round(sum(float(v) for v in terms.values()), 6)


def battle_reward(terms):
    assert_battle_reward_clean(terms)
    return round(sum(float(v) for v in terms.values()), 6)


def no_double_count(nav_terms, battle_terms):
    """A battle win must not appear in BOTH channels. The only battle-derived
    navigation term allowed is ``battle_triggered_story_change`` (a story flag
    the fight genuinely set) — and that is a story term, not a combat term."""
    overlap = (set(nav_terms) & set(battle_terms)) - {"wipe"}
    if overlap:
        raise RewardChannelError(f"term double-counted across channels: {sorted(overlap)}")
    if "battle_win" in nav_terms:
        raise RewardChannelError("navigation channel must not contain 'battle_win'")
    return True
