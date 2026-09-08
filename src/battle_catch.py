"""Gen-III (FireRed / BPRD) catch mechanics + ball data + outcome classifier.

Pure, deterministic, no emulator. The catch *probability estimate* uses the
real Gen-III formula and is only produced when every input (species catch rate,
HP, status, ball type) is reliably known — an unknown input yields ``None``,
never a guess (spec §5).

The ``classify_catch_outcome`` helper turns a verified RAM before/after diff
into one of the required outcomes (spec §7/§8); it never infers ``caught`` from
"battle ended" or "party grew" alone.
"""
from __future__ import annotations

import math

# --- ball types --------------------------------------------------------------
# id -> (name, static multiplier, dynamic(fn) | None). FireRed item ids.
POKE_BALL, GREAT_BALL, ULTRA_BALL, MASTER_BALL = 4, 3, 2, 1
SAFARI_BALL, NET_BALL, DIVE_BALL, NEST_BALL = 5, 6, 7, 8
REPEAT_BALL, TIMER_BALL, LUXURY_BALL, PREMIER_BALL = 9, 10, 11, 12

_BALL_NAME = {
    POKE_BALL: "poke", GREAT_BALL: "great", ULTRA_BALL: "ultra",
    MASTER_BALL: "master", SAFARI_BALL: "safari", NET_BALL: "net",
    DIVE_BALL: "dive", NEST_BALL: "nest", REPEAT_BALL: "repeat",
    TIMER_BALL: "timer", LUXURY_BALL: "luxury", PREMIER_BALL: "premier",
}
# the pockets a live bag reader must be able to enumerate; kept minimal + real.
CATCH_BALL_IDS = tuple(_BALL_NAME)


def ball_name(ball_id):
    return _BALL_NAME.get(int(ball_id), f"ball_{int(ball_id)}")


def is_ball(item_id):
    return int(item_id) in _BALL_NAME


def ball_bonus(ball_id, *, target_types=(), target_level=None, turn=None,
               in_water_or_dive=False, already_owned=False):
    """The Gen-III capture multiplier for one ball. Returns ``None`` when the
    ball's bonus depends on data we were not given (e.g. Nest/Timer without
    level/turn) so the caller can fall back to ``objective_mode`` decisions
    without an invented number."""
    b = int(ball_id)
    if b == MASTER_BALL:
        return math.inf
    if b in (POKE_BALL, PREMIER_BALL, LUXURY_BALL):
        return 1.0
    if b in (GREAT_BALL, SAFARI_BALL):
        return 1.5
    if b == ULTRA_BALL:
        return 2.0
    if b == NET_BALL:
        t = {str(x).lower() for x in target_types}
        return 3.0 if (t & {"water", "bug", "11", "6"}) else 1.0
    if b == DIVE_BALL:
        return 3.5 if in_water_or_dive else 1.0
    if b == NEST_BALL:
        if target_level is None:
            return None
        return max(1.0, (40.0 - int(target_level)) / 10.0)
    if b == REPEAT_BALL:
        return 3.0 if already_owned else 1.0
    if b == TIMER_BALL:
        if turn is None:
            return None
        return min(4.0, 1.0 + int(turn) * 1229.0 / 4096.0)
    return 1.0


_STATUS_BONUS = {   # normalised status name -> multiplier
    "sleep": 2.0, "asleep": 2.0, "freeze": 2.0, "frozen": 2.0,
    "paralysis": 1.5, "paralyzed": 1.5, "burn": 1.5, "burned": 1.5,
    "poison": 1.5, "poisoned": 1.5, "badly_poisoned": 1.5, "toxic": 1.5,
    "none": 1.0, "": 1.0,
}


def status_bonus(status):
    return _STATUS_BONUS.get(str(status or "none").lower(), 1.0)


def catch_probability(*, catch_rate, cur_hp, max_hp, ball_bonus,
                      status_bonus=1.0):
    """Gen-III single-ball catch probability in [0, 1], or ``None`` if any
    input is missing / non-positive. Uses the real shake-check maths."""
    try:
        cr = float(catch_rate); ch = float(cur_hp)
        mh = float(max_hp); bb = float(ball_bonus); sb = float(status_bonus)
    except (TypeError, ValueError):
        return None
    if cr <= 0 or mh <= 0 or ch < 0 or ch > mh or bb <= 0:
        return None
    if math.isinf(bb):
        return 1.0
    a = ((3.0 * mh - 2.0 * ch) * cr * bb) / (3.0 * mh) * sb
    a = max(1.0, min(255.0, a))
    if a >= 255.0:
        return 1.0
    b = 1048560.0 / math.sqrt(math.sqrt(16711680.0 / a))
    p_shake = min(1.0, b / 65536.0)
    return p_shake ** 4


def safe_catch_window(*, catch_rate, max_hp, ball_bonus, status_bonus=1.0,
                      target_prob=0.60):
    """The HP (absolute) at or below which a single ball reaches
    ``target_prob``. Used for catch-mode damage shaping (spec §9B): positive
    damage reward only while the target is ABOVE this window. ``None`` if
    inputs are missing."""
    try:
        cr = float(catch_rate); mh = float(max_hp)
        bb = float(ball_bonus); sb = float(status_bonus)
    except (TypeError, ValueError):
        return None
    if cr <= 0 or mh <= 0 or bb <= 0:
        return None
    if math.isinf(bb):
        return mh
    # invert the formula for p == target_prob
    p_shake = target_prob ** 0.25
    if p_shake >= 1.0:
        return mh
    b = p_shake * 65536.0
    a = 16711680.0 / ((1048560.0 / b) ** 4)
    # a = ((3*mh - 2*ch) * cr * bb) / (3*mh) * sb  ->  solve ch
    lhs = a * 3.0 * mh / (cr * bb * sb)
    ch = (3.0 * mh - lhs) / 2.0
    return max(0.0, min(mh, ch))


# --- outcome classification (spec §7 / §8) ----------------------------------
CATCH_OUTCOMES = (
    "caught", "broke_free", "battle_continues", "target_fainted",
    "no_balls", "party_full", "sent_to_pc", "illegal_trainer_catch",
    "bag_unreadable", "menu_stall", "executor_abort", "unreadable",
)


def classify_catch_outcome(before, after, *, is_trainer=False,
                            ball_thrown=True, pc_capture_supported=False):
    """``before`` / ``after`` are verified RAM snapshots (dicts). Never returns
    ``caught`` without stable positive evidence: a ball was consumed AND
    (a new party member with the target's identity appeared) OR (a verified
    pokedex 'owned' bit flipped for the target) OR (a verified PC transfer).
    """
    b, a = before or {}, after or {}
    if is_trainer:
        return "illegal_trainer_catch"

    for k in ("ball_counts", "battle_active", "enemy_hp"):
        if k not in a:
            return "unreadable"

    if not ball_thrown:
        # precheck rejected before the first button
        if int(b.get("usable_balls", 0)) <= 0:
            return "no_balls"
        return "battle_continues"

    balls_before = int((b.get("ball_counts") or {}).get("total", 0))
    balls_after = int((a.get("ball_counts") or {}).get("total", 0))
    ball_consumed = balls_after == balls_before - 1

    tgt = b.get("target_identity")
    new_member = a.get("new_party_member")           # {"species_id","identity"}
    party_grew = int(a.get("party_size", 0)) > int(b.get("party_size", 0))
    dex_owned_flip = bool(a.get("target_dex_owned_now")
                          and not b.get("target_dex_owned_before"))
    pc_transfer = bool(a.get("pc_transfer_confirmed"))

    matched_new_member = bool(
        new_member and tgt is not None
        and new_member.get("identity") == tgt)

    if ball_consumed and (matched_new_member or dex_owned_flip
                          or (pc_transfer and pc_capture_supported)):
        if pc_transfer and not party_grew:
            return "sent_to_pc"
        if not party_grew and not dex_owned_flip:
            return "unreadable"
        return "caught"

    if ball_consumed and a.get("battle_active") is True and int(a.get("enemy_hp", 1)) > 0:
        return "broke_free"

    if a.get("battle_active") is True and int(a.get("enemy_hp", 0)) <= 0:
        return "target_fainted"

    if not ball_consumed and party_grew:
        # party grew but no ball spent -> not a catch we performed
        return "unreadable"

    return "battle_continues"
