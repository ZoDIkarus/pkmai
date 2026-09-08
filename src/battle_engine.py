"""Generation III (FireRed / BPRD) damage / KO / speed engine.

Pure functions over already-decoded battle structs. No RAM, no emulator.

Scope: the **integer** damage path that pret/pokefirered's ``CalculateBaseDamage``
+ ``Cmd_typecalc`` + ``ApplyRandomDmgMultiplier`` implement for a single battle
with **no abilities, held items, weather, screens, or dynamic-power effects**.
Those are *not modelled*; when any of them could matter the result is flagged
``mechanics_complete=False`` / ``damage_is_estimate=True`` and lists the reason
in ``unknown_reasons`` (a stable enum). The engine may still return a damage
estimate - it just never claims it is exact or a true guarantee.

Integer order (matches pokefirered, no crit path shown):

    A  = APPLY_STAT_MOD(base_off_stat, stage)          # int:  x*num//den
    D  = APPLY_STAT_MOD(base_def_stat, stage)
    d  = A * power
    d  = d * (2 * level // 5 + 2)
    d  = d // D
    d  = d // 50
    if physical burn and not Guts:  d = d // 2
    d  = d + 2
    if crit:                        d = d * 2
    if STAB:                        d = d * 15 // 10
    for each unique defender type:  d = d * mult10 // 10   (mult10 in {0,5,10,20};
                                    if d == 0 and mult10 != 0: d = 1)
    for roll in 85..100:            v = d * roll // 100;  if v == 0 and d != 0: v = 1
"""
from __future__ import annotations

from battle_types import (
    type_pair_multiplier, is_physical_type, is_valid_type, TYPE_MYSTERY,
)

# FireRed party-struct status1 bitfield (the u32 at mon+0x50).
STATUS_SLEEP_MASK = 0x07
STATUS_POISON = 0x08
STATUS_BURN = 0x10
STATUS_FREEZE = 0x20
STATUS_PARALYSIS = 0x40
STATUS_TOXIC = 0x80

# Gen III stage-0 critical-hit rate (for non-high-crit moves, no items/abilities).
BASE_CRIT_RATE = 1.0 / 16.0

# Gen III: APPLY_STAT_MOD ratio numerator/denominator per stat stage (-6..+6).
# stat = base * num // den   (pokefirered gStatStageRatios for offensive/def stats)
_STAGE_RATIO = {
    -6: (10, 40), -5: (10, 35), -4: (10, 30), -3: (10, 25), -2: (10, 20),
    -1: (10, 15), 0: (10, 10), 1: (15, 10), 2: (20, 10), 3: (25, 10),
    4: (30, 10), 5: (35, 10), 6: (40, 10),
}

# Stable enum of reasons a damage number is only an estimate.
UNKNOWN_ABILITY = "ability_not_authoritative"
UNKNOWN_IN_BATTLE_TYPES = "in_battle_types_not_authoritative"
UNKNOWN_STAT_STAGES = "stat_stages_unknown"
UNKNOWN_WEATHER = "weather_unknown"
UNKNOWN_SCREENS = "screens_unknown"
UNKNOWN_ITEM = "item_effects_unknown"
UNKNOWN_DYNAMIC_POWER = "dynamic_power_or_special_effect_not_modelled"

ALL_UNKNOWN_REASONS = (
    UNKNOWN_ABILITY, UNKNOWN_IN_BATTLE_TYPES, UNKNOWN_STAT_STAGES,
    UNKNOWN_WEATHER, UNKNOWN_SCREENS, UNKNOWN_ITEM, UNKNOWN_DYNAMIC_POWER,
)

# Gen III move effect ids whose real power / behaviour is not a constant we can
# read from the move table alone. Conservative superset for the FR early game;
# extend as needed. Any of these -> UNKNOWN_DYNAMIC_POWER.
_DYNAMIC_POWER_EFFECTS = frozenset({
    28,   # EFFECT_LEVEL_DAMAGE (Seismic Toss / Night Shade)
    39,   # EFFECT_OHKO
    40,   # EFFECT_RAZOR_WIND (charge)
    41,   # EFFECT_SUPER_FANG (half current HP)
    42,   # EFFECT_DRAGON_RAGE (fixed 40)
    43,   # EFFECT_TRAP (Bind etc.)
    44,   # EFFECT_HIGH_CRITICAL
    45,   # EFFECT_DOUBLE_HIT / twineedle-ish
    47,   # EFFECT_SONICBOOM (fixed 20)
    49,   # EFFECT_MULTI_HIT
    66,   # EFFECT_PSYWAVE
    68,   # EFFECT_COUNTER
    69,   # EFFECT_ENCORE (n/a)
    83,   # EFFECT_HIDDEN_POWER
    84,   # EFFECT_RAIN_DANCE (n/a here)
    88,   # EFFECT_RETURN / FRUSTRATION (happiness based)
    89,   # EFFECT_PRESENT
    90,   # EFFECT_FRUSTRATION
    99,   # EFFECT_MAGNITUDE
    103,  # EFFECT_QUICK_ATTACK is fine (priority); kept out below - handled by priority
    111,  # EFFECT_FURY_CUTTER (ramps)
    118,  # EFFECT_ROLLOUT (ramps)
    121,  # EFFECT_FLAIL / REVERSAL (HP based)
    128,  # EFFECT_FACADE
    136,  # EFFECT_ERUPTION / WATER_SPOUT (HP based)
    151,  # EFFECT_WEATHER_BALL
    152,  # EFFECT_LOW_KICK (weight)
})
_DYNAMIC_POWER_EFFECTS = _DYNAMIC_POWER_EFFECTS - {103}


def stage_ratio(stage):
    try:
        s = int(stage)
    except (TypeError, ValueError):
        return (10, 10)
    s = max(-6, min(6, s))
    return _STAGE_RATIO[s]


def apply_stage_int(base_stat, stage):
    """Integer APPLY_STAT_MOD. None passthrough."""
    if base_stat is None:
        return None
    try:
        b = int(base_stat)
    except (TypeError, ValueError):
        return None
    if b <= 0:
        return None
    num, den = stage_ratio(stage)
    return (b * num) // den


def has_status(status_value, mask):
    try:
        v = int(status_value)
    except (TypeError, ValueError):
        return False
    if mask == STATUS_SLEEP_MASK:
        return (v & STATUS_SLEEP_MASK) != 0
    return (v & mask) != 0


def _mult10(move_type, defender_type):
    """Gen-III effectiveness for one type pair as an integer x10 (0/5/10/20)."""
    f = type_pair_multiplier(move_type, defender_type)
    return int(round(f * 10))


def _unique_types(types):
    seen = []
    for t in types or []:
        if is_valid_type(t) and t not in seen:
            seen.append(t)
    return seen


def classify_move(move):
    """'status' | 'damaging' | 'unknown'.

    * 'status'   - explicitly a status move, or power known and == 0.
    * 'damaging' - a valid attacking type AND a known power > 0.
    * 'unknown'  - not enough is known to tell (missing/invalid type or power).
      An unknown move is NEVER treated as a status move.
    """
    if not isinstance(move, dict):
        return "unknown"
    if move.get("mechanics_known") is False:
        return "unknown"
    p = move.get("power")
    try:
        p = int(p)
        power_known = True
    except (TypeError, ValueError):
        power_known = False
    if move.get("is_status") is True and power_known and p == 0:
        return "status"
    if move.get("is_status") is True and not power_known:
        return "status"
    if power_known and p == 0:
        return "status"
    t = move.get("type")
    if (is_valid_type(t) or t == TYPE_MYSTERY) and power_known and p > 0:
        return "damaging"
    return "unknown"


def is_damaging_move(move):
    return classify_move(move) == "damaging"


def _collect_unknown_reasons(attacker, defender, move, *, ability_known,
                             in_battle_types_authoritative, stat_stages_known,
                             weather_known, screens_known, item_known):
    reasons = []
    if not ability_known:
        reasons.append(UNKNOWN_ABILITY)
    if not in_battle_types_authoritative:
        reasons.append(UNKNOWN_IN_BATTLE_TYPES)
    if not stat_stages_known:
        reasons.append(UNKNOWN_STAT_STAGES)
    if not weather_known:
        reasons.append(UNKNOWN_WEATHER)
    if not screens_known:
        reasons.append(UNKNOWN_SCREENS)
    if not item_known:
        reasons.append(UNKNOWN_ITEM)
    eff = move.get("effect") if isinstance(move, dict) else None
    try:
        if int(eff) in _DYNAMIC_POWER_EFFECTS:
            reasons.append(UNKNOWN_DYNAMIC_POWER)
    except (TypeError, ValueError):
        pass
    return reasons


def calc_damage(attacker, defender, move, *, is_crit=False,
                ability_known=False, in_battle_types_authoritative=False,
                stat_stages_known=False, weather_known=False,
                screens_known=False, item_known=False):
    """Full Gen-III integer damage result for ONE move at ONE crit state.

    Returns a dict. Keys:
      unknown            - True when no damage number could be produced at all
      is_status          - the move is a (recognised) status move
      is_damaging        - a damage number (0..N) was produced
      rolls              - list of the 16 integer damage values (roll 85..100),
                           or [0]*16 for an immunity, or None when unknown
      min_damage/max_damage/expected  - from ``rolls``
      effectiveness      - float product (0, .25, .5, 1, 2, 4)
      stab               - 1.0 or 1.5
      physical           - Gen-III physical/special (by move type)
      mechanics_complete - True ONLY when every relevant mechanic is authoritative
      damage_is_estimate - True when a number was produced but not complete
      unknown_reasons    - stable enum list (see ALL_UNKNOWN_REASONS)
    """
    out = {
        "unknown": True, "is_status": False, "is_damaging": False,
        "rolls": None, "min_damage": None, "max_damage": None, "expected": None,
        "effectiveness": None, "stab": None, "physical": None,
        "mechanics_complete": False, "damage_is_estimate": False,
        "unknown_reasons": [],
    }
    if not (isinstance(attacker, dict) and isinstance(defender, dict)):
        return out

    cls = classify_move(move)
    if cls == "status":
        out["unknown"] = False
        out["is_status"] = True
        return out
    if cls == "unknown":
        return out

    mtype = int(move["type"])
    power = int(move["power"])
    physical = is_physical_type(mtype)   # MYSTERY is in PHYSICAL_TYPES
    out["physical"] = physical

    a_types = _unique_types(attacker.get("types"))
    d_types = _unique_types(defender.get("types"))
    if not d_types:
        return out   # cannot resolve effectiveness

    try:
        level = int(attacker.get("level"))
        if not (1 <= level <= 100):
            return out
    except (TypeError, ValueError):
        return out

    a_stats = attacker.get("stats") or {}
    d_stats = defender.get("stats") or {}
    a_stages = attacker.get("stat_stages") or {}
    d_stages = defender.get("stat_stages") or {}
    if physical:
        a_raw, d_raw = a_stats.get("attack"), d_stats.get("defense")
        a_key, d_key = "attack", "defense"
    else:
        a_raw, d_raw = a_stats.get("sp_attack"), d_stats.get("sp_defense")
        a_key, d_key = "sp_attack", "sp_defense"

    # Gen III crit: ignore the attacker's *lowered* offensive stage and the
    # defender's *raised* defensive stage.
    a_stage = a_stages.get(a_key, 0) or 0
    d_stage = d_stages.get(d_key, 0) or 0
    if is_crit and a_stage < 0:
        a_stage = 0
    if is_crit and d_stage > 0:
        d_stage = 0
    A = apply_stage_int(a_raw, a_stage)
    D = apply_stage_int(d_raw, d_stage)
    if A is None or D is None or D <= 0:
        return out

    # --- integer base ---
    d = A * power
    d = d * (2 * level // 5 + 2)
    d = d // D
    d = d // 50
    # burn (physical only). Guts ability would negate it -> unknown reason below.
    if physical and has_status(attacker.get("status"), STATUS_BURN):
        d = d // 2
    d = d + 2

    if is_crit:
        d = d * 2

    # STAB (Struggle / MYSTERY type never matches a real Pokémon type -> 1.0)
    stab = 1.5 if (mtype in a_types) else 1.0
    out["stab"] = stab
    if stab == 1.5:
        d = d * 15 // 10

    # type effectiveness, one modulation per unique defender type
    eff = 1.0
    for dt in d_types:
        m10 = _mult10(mtype, dt)
        eff *= m10 / 10.0
        if m10 == 0:
            d = 0
            break
        d = d * m10 // 10
        if d == 0:      # ModulateDmgByType min-1 for a non-immune roll-to-zero
            d = 1
    out["effectiveness"] = eff

    if d <= 0:
        rolls = [0] * 16
    else:
        rolls = []
        for roll in range(85, 101):
            v = d * roll // 100
            if v == 0:
                v = 1
            rolls.append(v)

    reasons = _collect_unknown_reasons(
        attacker, defender, move, ability_known=ability_known,
        in_battle_types_authoritative=in_battle_types_authoritative,
        stat_stages_known=stat_stages_known, weather_known=weather_known,
        screens_known=screens_known, item_known=item_known)
    complete = not reasons

    out.update(
        unknown=False, is_damaging=True, rolls=rolls,
        min_damage=min(rolls), max_damage=max(rolls),
        expected=round(sum(rolls) / len(rolls), 2),
        mechanics_complete=complete,
        damage_is_estimate=not complete,
        unknown_reasons=reasons,
    )
    return out


def damage_range(attacker, defender, move, **kw):
    """Compatibility view over ``calc_damage`` plus a crit-blended expected.

    Adds ``crit_expected`` (crit-rate-weighted expected). All the exactness /
    confidence keys from ``calc_damage`` are passed through unchanged.
    """
    out = calc_damage(attacker, defender, move, is_crit=False, **kw)
    if out.get("is_damaging") and out.get("expected") is not None:
        crit = calc_damage(attacker, defender, move, is_crit=True, **kw)
        if crit.get("is_damaging") and crit.get("expected") is not None:
            out["crit_expected"] = round(
                (1 - BASE_CRIT_RATE) * out["expected"]
                + BASE_CRIT_RATE * crit["expected"], 2)
        else:
            out["crit_expected"] = out["expected"]
    else:
        out["crit_expected"] = None
    return out


def ko_assessment(dmg, defender_cur_hp, *, hit_probability=None):
    """KO analysis for one ``calc_damage``/``damage_range`` result.

    Returns:
      ko_on_hit_guaranteed - every one of the 16 rolls KOs (assumes it lands)
      ko_on_hit_possible   - at least one roll KOs
      ko_rolls_fraction    - fraction of the 16 rolls that KO (0..1)
      hit_probability      - echoed back (None if unknown)
      ko_probability       - hit_probability * ko_rolls_fraction, or None
      true_guaranteed      - True ONLY if ko_on_hit_guaranteed AND
                             hit_probability == 1.0 AND dmg is
                             ``mechanics_complete``. Never True on any unknown.
      fraction_expected    - expected damage / hp (capped 1.0), or None
    """
    out = {
        "ko_on_hit_guaranteed": False, "ko_on_hit_possible": False,
        "ko_rolls_fraction": None, "hit_probability": hit_probability,
        "ko_probability": None, "true_guaranteed": False,
        "fraction_expected": None,
    }
    if not isinstance(dmg, dict) or dmg.get("unknown") or not dmg.get("is_damaging"):
        return out
    try:
        hp = int(defender_cur_hp)
    except (TypeError, ValueError):
        return out
    rolls = dmg.get("rolls") or []
    if not rolls:
        return out
    if hp <= 0:
        out.update(ko_on_hit_guaranteed=True, ko_on_hit_possible=True,
                   ko_rolls_fraction=1.0, fraction_expected=1.0)
        # even here, only "true" with full mechanics + certain hit
        if hit_probability == 1.0 and dmg.get("mechanics_complete"):
            out["true_guaranteed"] = True
        if hit_probability is not None:
            out["ko_probability"] = round(hit_probability, 4)
        return out

    ko_count = sum(1 for v in rolls if v >= hp)
    frac = ko_count / len(rolls)
    out["ko_on_hit_guaranteed"] = ko_count == len(rolls)
    out["ko_on_hit_possible"] = ko_count > 0
    out["ko_rolls_fraction"] = round(frac, 4)
    exp = dmg.get("expected")
    if exp is not None:
        out["fraction_expected"] = round(min(1.0, exp / hp), 4)
    if hit_probability is not None:
        out["ko_probability"] = round(hit_probability * frac, 4)
    out["true_guaranteed"] = bool(
        out["ko_on_hit_guaranteed"]
        and hit_probability == 1.0
        and dmg.get("mechanics_complete") is True
    )
    return out


# Backwards-compatible thin shim.
#   "guaranteed" here means "every damage roll KOs *assuming the move lands*" -
#   i.e. ``ko_on_hit_guaranteed``. It is NOT a true certainty: accuracy,
#   abilities (Levitate / Wonder Guard / Sturdy), Focus Sash etc. are not
#   modelled. Use ``ko_assessment(...)['true_guaranteed']`` for the strict flag.
def ko_estimate(dmg, defender_cur_hp):
    a = ko_assessment(dmg, defender_cur_hp)
    return {
        "guaranteed": a["ko_on_hit_guaranteed"],
        "possible": a["ko_on_hit_possible"],
        "fraction_expected": a["fraction_expected"],
        "true_guaranteed": a["true_guaranteed"],
    }


def move_hits_probability(move):
    """Accuracy as a 0..1 float. Unknown/None -> None (caller stays cautious).
    Gen III accuracy 0 means 'bypasses the accuracy check' -> 1.0."""
    if not isinstance(move, dict):
        return None
    if move.get("mechanics_known") is False:
        return None
    acc = move.get("accuracy")
    if acc is None:
        return None
    try:
        acc = int(acc)
    except (TypeError, ValueError):
        return None
    if acc <= 0:
        return 1.0
    return min(1.0, acc / 100.0)


def effective_speed(mon):
    """Integer Speed after stage + paralysis (x1/4 in Gen III). None if unknown."""
    if not isinstance(mon, dict):
        return None
    base = (mon.get("stats") or {}).get("speed")
    stages = mon.get("stat_stages") or {}
    spd = apply_stage_int(base, stages.get("speed", 0))
    if spd is None:
        return None
    if has_status(mon.get("status"), STATUS_PARALYSIS):
        spd = spd // 4
    return spd


def order_of_action(a_mon, a_move, b_mon, b_move):
    """"a" | "b" | "tie" | "unknown". Priority first, then effective Speed."""
    def prio(m):
        if not isinstance(m, dict):
            return 0
        try:
            return int(m.get("priority"))
        except (TypeError, ValueError):
            return 0
    pa, pb = prio(a_move), prio(b_move)
    if pa != pb:
        return "a" if pa > pb else "b"
    sa, sb = effective_speed(a_mon), effective_speed(b_mon)
    if sa is None or sb is None:
        return "unknown"
    if sa > sb:
        return "a"
    if sb > sa:
        return "b"
    return "tie"
