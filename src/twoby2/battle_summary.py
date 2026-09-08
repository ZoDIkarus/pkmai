"""The ONLY thing navigation learns from a battle: a strategic summary.

Navigation never sees a battle observation, a battle menu, a raw battle button
or a per-turn battle reward. When a battle sub-episode ends the navigation-only
wrapper hands the policy one summary dict as the *consequence* of the overworld
decision that walked into the encounter.

The summary is validated **recursively** — a forbidden key nested anywhere
inside ``resulting_world_state`` is a leak and is rejected. Every number is
normalised fail-closed so a NaN / Inf / string / weird type can never crash the
navigation reward path.
"""
from __future__ import annotations

import math

OUTCOMES = ("win", "loss", "wipe", "fled", "caught")

# spec §15: the shared outcome + reward-component vocabulary lives in ONE place
# so the battle env, the summary, promotion and the web all speak the same
# language and nobody reconstructs a component list from log text.
BATTLE_OUTCOMES = ("win", "loss", "wipe", "fled", "caught", "timeout",
                   "terminal_unknown", "menu_stall")
CATCH_OUTCOMES = ("caught", "broke_free", "battle_continues", "target_fainted",
                  "no_balls", "party_full", "sent_to_pc", "illegal_trainer_catch",
                  "bag_unreadable", "menu_stall", "executor_abort", "unreadable")
REWARD_COMPONENT_NAMES = (
    "turn_cost", "wasted_turn", "damage", "enemy_ko", "battle_win", "own_faint",
    "wipe", "flee_illegal", "fled", "switch_loop", "invalid_action",
    "unrequested_catch", "illegal_trainer_catch", "ball_cost",
    "premature_catch_attempt", "catch_success", "failed_catch", "catch_target_ko",
)

# spec §15: the FULL battle-episode summary fields (telemetry + promotion +
# FULL-wrapper internal use). This is NOT the navigation-safe projection — it
# deliberately carries reward totals, which navigation must never see.
BATTLE_EPISODE_FIELDS = (
    "objective_mode", "catch_requested", "catch_reason", "target_species_id",
    "outcome", "caught_species_id", "catch_success", "catch_attempts",
    "failed_catch_attempts", "balls_used", "target_ko", "unrequested_catch",
    "trainer_catch_attempt", "battle_turns", "hp_lost", "pp_used",
    "battle_reward_total", "navigation_strategic_reward", "reward_components",
)

# Exactly the keys navigation is allowed to receive at the top level.
ALLOWED_SUMMARY_KEYS = frozenset({
    "outcome",                 # one of OUTCOMES
    "party_hp_lost",           # total HP points lost across the party
    "party_hp_fraction_lost",  # 0..1
    "party_hp_total_after",    # coarse resource state
    "pp_remaining_total",      # coarse resource state
    "pp_spent",                # total PP consumed
    "items_used",              # count of items consumed
    "own_faints",              # own Pokémon that fainted
    "enemy_faints",            # enemies defeated
    "turns",                   # turn count
    "fled",                    # bool
    "blackout",                # bool — party wiped, respawned at a Pokémon Center
    "resulting_world_state",   # strictly-schema'd strategic overworld state
    "battle_kind",             # "wild" | "trainer" (routing context, not a reward)
    "duration_steps",          # aggregated emulator steps the battle consumed
    # --- Catch-v2 (spec §3): strategic, non-micro. Navigation uses these ONLY
    # to decide the small one-off species reward; none is a per-turn / damage /
    # menu signal. ``catch_requested`` is the gate for the +0.5.
    "objective_mode",          # "combat" | "catch"
    "catch_requested",         # bool — the strategic planner asked for a catch
    "catch_success",           # bool — a catch was RAM-confirmed
    "caught_species_id",       # species id actually caught (0 if none)
    "target_species_id",       # the species the objective wanted
    "is_new_species_this_run", # bool — first of this species this training run
})

# resulting_world_state: a strict, shallow, typed schema. NOTHING else is kept.
WORLD_STATE_SCHEMA = {
    "map_group": int, "map_id": int, "x": int, "y": int,
    "story_flag_count": int, "badges": int, "party_hp_total": int,
    "party_alive": int, "respawn_map_group": int, "respawn_map_id": int,
}

# Substrings that, appearing in ANY key at ANY depth, prove micro-combat data
# leaked. Deliberately precise so legitimate names ("party_hp_fraction_lost")
# don't trip.
FORBIDDEN_SUBSTRINGS = (
    "reward", "macro", "move_", "q_value", "qvalue", "logit", "advantage",
    "rollout", "observation", "menu", "cursor", "button", "frame", "per_turn",
    "perturn", "gradient", "logprob", "log_prob", "pixel", "screen",
    "value_target", "damage", "crit_", "effectiveness", "turn_log",
    "move_pick", "selection", "picked", "chosen", "history", "stab_",
)

# Self-check: neither allow-list may itself contain a forbidden fragment.
_bad = [k for k in list(ALLOWED_SUMMARY_KEYS) + list(WORLD_STATE_SCHEMA)
        for s in FORBIDDEN_SUBSTRINGS if s in k.lower()]
assert not _bad, f"allow-list contains micro-combat-looking keys: {_bad}"
del _bad


class BattleSummaryLeak(AssertionError):
    pass


def _num(v, *, lo=None, hi=None, default=0.0, integer=False):
    """Fail-closed number: strings/None/NaN/Inf/junk -> ``default``, then clamp."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return int(default) if integer else default
    if not math.isfinite(f):
        return int(default) if integer else default
    if lo is not None:
        f = max(lo, f)
    if hi is not None:
        f = min(hi, f)
    return int(f) if integer else f


def _clean_world_state(raw):
    """Project ``raw`` onto WORLD_STATE_SCHEMA. Unknown keys dropped; every
    value coerced fail-closed. Nested dicts/lists are NOT allowed here."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key, typ in WORLD_STATE_SCHEMA.items():
        if key not in raw:
            continue
        val = raw[key]
        if isinstance(val, (dict, list, tuple, set)):
            raise BattleSummaryLeak(
                f"resulting_world_state.{key} is a container — only scalars allowed")
        out[key] = _num(val, lo=0, integer=(typ is int))
    return out


def summarize_battle(raw):
    """Project an internal battle-result dict down to the allowed keys.

    Unknown/forbidden top-level keys are dropped. ``resulting_world_state`` is
    re-built from scratch against WORLD_STATE_SCHEMA. Call
    :func:`assert_navigation_safe` on the result to *prove* it is clean.
    """
    raw = raw or {}
    out = {}

    outcome = raw.get("outcome")
    if outcome not in OUTCOMES:
        if raw.get("catch_success"):
            outcome = "caught"
        elif raw.get("fled"):
            outcome = "fled"
        elif _num(raw.get("own_alive", raw.get("party_alive", 1)), integer=True) <= 0:
            outcome = "wipe"
        else:
            outcome = "loss"
    out["outcome"] = outcome
    out["fled"] = bool(raw.get("fled", outcome == "fled"))
    out["blackout"] = bool(raw.get("blackout", outcome == "wipe"))

    # --- Catch-v2 strategic projection (spec §3) -----------------------
    _mode = raw.get("objective_mode")
    out["objective_mode"] = _mode if _mode in ("combat", "catch") else "combat"
    out["catch_requested"] = bool(raw.get("catch_requested"))
    out["catch_success"] = bool(raw.get("catch_success")) or outcome == "caught"
    out["caught_species_id"] = _num(raw.get("caught_species_id"), lo=0, integer=True)
    out["target_species_id"] = _num(raw.get("target_species_id"), lo=0, integer=True)
    out["is_new_species_this_run"] = bool(raw.get("is_new_species_this_run"))

    for k in ("party_hp_lost", "party_hp_total_after", "pp_remaining_total",
              "pp_spent", "items_used", "own_faints", "enemy_faints", "turns",
              "duration_steps"):
        out[k] = _num(raw.get(k), lo=0, integer=True)
    out["party_hp_fraction_lost"] = _num(raw.get("party_hp_fraction_lost"),
                                         lo=0.0, hi=1.0)
    out["battle_kind"] = raw.get("battle_kind") if raw.get("battle_kind") in (
        "wild", "trainer") else "unknown"
    out["resulting_world_state"] = _clean_world_state(raw.get("resulting_world_state"))
    return out


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            yield here, k
            yield from _walk_keys(v, here)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def assert_navigation_safe(summary):
    """Raise :class:`BattleSummaryLeak` if ``summary`` carries anything
    navigation must not see — checked **recursively**."""
    if not isinstance(summary, dict):
        raise BattleSummaryLeak(f"summary is {type(summary).__name__}, not dict")

    extra = set(summary) - ALLOWED_SUMMARY_KEYS
    if extra:
        raise BattleSummaryLeak(f"disallowed top-level keys: {sorted(extra)}")

    # recursive forbidden-substring scan over every key at every depth
    for full_path, key in _walk_keys(summary):
        low = str(key).lower()
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in low:
                raise BattleSummaryLeak(
                    f"micro-combat key leaked to navigation at {full_path!r}")

    # resulting_world_state must match the strict schema exactly
    ws = summary.get("resulting_world_state")
    if ws is not None:
        if not isinstance(ws, dict):
            raise BattleSummaryLeak("resulting_world_state must be a dict")
        ws_extra = set(ws) - set(WORLD_STATE_SCHEMA)
        if ws_extra:
            raise BattleSummaryLeak(
                f"resulting_world_state has non-schema keys: {sorted(ws_extra)}")
        for k, v in ws.items():
            if isinstance(v, (dict, list, tuple, set)):
                raise BattleSummaryLeak(
                    f"resulting_world_state.{k} is a container")
            if not isinstance(v, (int, float)) or (
                    isinstance(v, float) and not math.isfinite(v)):
                raise BattleSummaryLeak(
                    f"resulting_world_state.{k} is not a finite number: {v!r}")
    return True


# wild-win decay ladder (plan I): a FULL agent gets a small strategic bonus for
# the first couple of wild wins on a map per training run, then 0 - so grinding
# wild battles instead of navigating is never a net income. The whole repeatable
# battle reward on a map stays far below the next geographic checkpoint reward.
WILD_WIN_LADDER = (0.5, 0.2)

# spec §3 (accepted): first strategically-requested new species per run +0.5,
# level bonus 0, duplicate -0.2. Kept in sync with
# pokemon_env.PokemonFireRedEnv.SPECIES_CAUGHT_FIRST_REWARD /
# SPECIES_CAUGHT_DUPLICATE_PENALTY.
CATCH_NEW_SPECIES_REWARD = 0.5
CATCH_DUPLICATE_PENALTY = -0.2


def navigation_battle_reward(summary, *, wild_win_index=None):
    """Navigation's strategic signal from a finished battle.

    Deliberately coarse: it rewards the *route decision*, not the fight. There
    is no per-turn term and no move/KO micro-reward — those live only in the
    battle system. Fail-closed on every number.

    ``wild_win_index`` (0 = first wild win on this map this run, 1 = second,
    2+ = further) drives the decay ladder; ``None`` disables it.
    """
    assert_navigation_safe(summary)
    outcome = summary.get("outcome")
    base = {
        "win": 0.0,      # winning a fight you walked into is not nav progress
        "caught": 0.0,   # a requested catch is a fine route decision, no more
        "loss": -0.5,
        "wipe": -3.0,
        "fled": -0.5,    # an unnecessary flee is a wrong route decision
    }.get(outcome, -0.5)
    frac = _num(summary.get("party_hp_fraction_lost"), lo=0.0, hi=1.0)
    r = base - 0.5 * frac
    if (outcome == "win" and summary.get("battle_kind") != "trainer"
            and wild_win_index is not None and int(wild_win_index) >= 0):
        i = int(wild_win_index)
        r += WILD_WIN_LADDER[i] if i < len(WILD_WIN_LADDER) else 0.0
    # Catch-v2 (spec §3): the small one-off species term. Only a strategically
    # REQUESTED, RAM-confirmed catch of the target species pays; a raw / random
    # catch pays nothing here (and pokemon_env's species_caught_first block is
    # gated the same way). The persistent per-run dedup lives in NavAgentState.
    if (bool(summary.get("catch_requested")) and bool(summary.get("catch_success"))
            and int(summary.get("caught_species_id", 0) or 0)
            == int(summary.get("target_species_id", 0) or 0) != 0):
        r += CATCH_NEW_SPECIES_REWARD if summary.get("is_new_species_this_run") \
            else CATCH_DUPLICATE_PENALTY
    return round(r, 4)


def battle_episode_summary(raw, *, reward_components=None,
                           battle_reward_total=None,
                           navigation_strategic_reward=None):
    """The FULL battle-episode summary (spec §15) — telemetry + promotion +
    FULL-wrapper internal use. **NOT** navigation-safe: it deliberately carries
    ``reward_components`` / ``battle_reward_total``, which navigation must never
    receive. Every field fail-closed.

    ``reward_components`` is a list of ``(name, amount)`` captured where
    ``BattleEnv`` computed the reward — never reconstructed from log text.
    """
    raw = raw or {}
    comps = list(reward_components or raw.get("reward_components") or [])
    comp_sum = round(sum(float(a) for _n, a in comps), 4) if comps else None
    total = (float(battle_reward_total) if battle_reward_total is not None
             else (comp_sum if comp_sum is not None else _num(raw.get("battle_reward_total"))))
    outcome = raw.get("outcome")
    if outcome not in BATTLE_OUTCOMES:
        outcome = "caught" if raw.get("catch_success") else (
            "fled" if raw.get("fled") else "loss")
    mode = raw.get("objective_mode")
    return {
        "objective_mode": mode if mode in ("combat", "catch") else "combat",
        "catch_requested": bool(raw.get("catch_requested")),
        "catch_reason": str(raw.get("catch_reason") or ""),
        "target_species_id": _num(raw.get("target_species_id"), lo=0, integer=True),
        "outcome": outcome,
        "caught_species_id": _num(raw.get("caught_species_id"), lo=0, integer=True),
        "catch_success": bool(raw.get("catch_success")) or outcome == "caught",
        "catch_attempts": _num(raw.get("catch_attempts"), lo=0, integer=True),
        "failed_catch_attempts": _num(raw.get("failed_catch_attempts"), lo=0, integer=True),
        "balls_used": _num(raw.get("balls_used"), lo=0, integer=True),
        "target_ko": bool(raw.get("target_ko")),
        "unrequested_catch": bool(raw.get("unrequested_catch")),
        "trainer_catch_attempt": bool(raw.get("trainer_catch_attempt")
                                      or raw.get("illegal_trainer_catch")),
        "battle_turns": _num(raw.get("battle_turns", raw.get("turns")), lo=0, integer=True),
        "hp_lost": _num(raw.get("hp_lost", raw.get("party_hp_lost")), lo=0, integer=True),
        "pp_used": _num(raw.get("pp_used", raw.get("pp_spent")), lo=0, integer=True),
        "battle_reward_total": round(float(total), 4),
        "navigation_strategic_reward": (None if navigation_strategic_reward is None
                                        else round(float(navigation_strategic_reward), 4)),
        "reward_components": [[str(n), round(float(a), 4)] for n, a in comps],
        "reward_sum_ok": (comp_sum is None
                          or abs(comp_sum - float(total)) <= 1e-3),
    }
