"""Rule-based Generation III battle controller.

**Status: OFFLINE BASELINE / REFERENCE CONTROLLER ONLY.**
This module is *not* an "immediately usable live champion". It is:
  * an offline baseline the Battle-PPO is evaluated against,
  * a safety fallback the executor may consult once RAM + menu-cursor
    readiness is confirmed (``battle_ram.battle_snapshot_ready_for_execution``).
Until that readiness function returns True, ``decide()`` output must be treated
as advisory telemetry: it is computed from a heuristic active-slot guess and
from move mechanics that are type/power-accurate but not in-battle-authoritative
(no stat stages, abilities, items, weather, screens). Every result carries
``executable`` (True only when the snapshot is execution-ready) so a caller
cannot mistake an advisory decision for a live-safe one.

Input: a ``battle_ram.battle_snapshot`` dict (or the equivalent assembled by a
test). Output: one tactical **macro action** plus a human-readable reason and a
decision detail block.

Macro action space (identical to the future Battle-PPO):
    MOVE_1 .. MOVE_4      use the Nth move of the active Pokémon
    SWITCH_1 .. SWITCH_6  switch to the Nth party slot
    RUN                   flee (only ever proposed for an escapable wild battle)

Design rule: **on uncertain data, choose a conservative safe action, never a
guess.** Unknown enemy types collapse effectiveness to 1.0; unknown move
mechanics make a move a last resort; unknown everything -> MOVE_1. No claim of
"guaranteed" is made - the strongest phrasing is "estimated KO on hit".
"""
from __future__ import annotations

from battle_engine import (
    damage_range, ko_assessment, classify_move, move_hits_probability,
)
import battle_types as bt  # noqa: F401  (kept for downstream/test symmetry)

MOVE_ACTIONS = ("MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4")
SWITCH_ACTIONS = ("SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4",
                  "SWITCH_5", "SWITCH_6")
RUN_ACTION = "RUN"

# A switch target counts as "safe" only if its worst plausible incoming hit
# leaves this much current-HP headroom (covers unmodelled boosts within reason).
_SAFE_INCOMING_FRACTION = 0.5


def _pp(move):
    try:
        return int((move or {}).get("pp", 0))
    except (TypeError, ValueError):
        return 0


def _usable_moves(active):
    """[(index, move_dict)] for moves with PP > 0. Index is 0-based."""
    out = []
    for i, mv in enumerate((active or {}).get("moves") or []):
        if i >= 4:
            break
        if _pp(mv) > 0:
            out.append((i, mv))
    return out


def _move_is_status(move):
    """True only for a *recognised* status move. Unknown -> False (never
    silently treated as a harmless status move)."""
    return classify_move(move) == "status"


def _score_move(active, enemy, move):
    """Return a dict describing one candidate move."""
    dmg = damage_range(active, enemy, move)
    hit = move_hits_probability(move)
    ko = ko_assessment(dmg, (enemy or {}).get("cur_hp"), hit_probability=hit)
    cls = classify_move(move)
    exp = dmg.get("expected")
    eff = dmg.get("effectiveness")
    # expected damage weighted by hit chance (unknown accuracy -> assume 0.8,
    # conservative, so a shaky move is not blindly preferred).
    hit_w = hit if hit is not None else 0.8
    weighted = (exp * hit_w) if exp is not None else None
    return {
        "classification": cls,                 # 'status' | 'damaging' | 'unknown'
        "is_status": cls == "status",
        "mechanics_known": cls != "unknown" and not dmg.get("unknown"),
        "damage_is_estimate": bool(dmg.get("damage_is_estimate")),
        "expected": exp,
        "weighted_expected": weighted,
        "effectiveness": eff,
        "stab": dmg.get("stab"),
        "hit_prob": hit,
        "ko_on_hit_guaranteed": ko["ko_on_hit_guaranteed"],
        "ko_on_hit_possible": ko["ko_on_hit_possible"],
        "ko_fraction": ko["fraction_expected"],
        "physical": dmg.get("physical"),
    }


def _best_enemy_threat(active, enemy):
    """Worst-case expected damage the enemy can do to us this turn.
    Uses only enemy moves we actually know; unknown -> None (treated as risky
    but not fabricated). Returns (worst_expected, threat_fully_known)."""
    best = None
    all_known = True
    saw_move = False
    for mv in (enemy or {}).get("moves") or []:
        saw_move = True
        d = damage_range(enemy, active, mv)
        if d.get("unknown"):
            all_known = False
            continue
        if not d.get("is_damaging"):
            continue
        e = d.get("expected")
        if e is not None and (best is None or e > best):
            best = e
    return best, (all_known and saw_move)


def _switch_candidates(snapshot):
    """Party slots (0-based) that are alive and not the current active mon."""
    party = snapshot.get("player_party") or []
    active = snapshot.get("player_active") or {}
    active_slot = active.get("slot")
    out = []
    for mon in party:
        if not mon.get("checksum_ok", True):
            continue
        try:
            if int(mon.get("cur_hp", 0)) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        if active_slot is not None and mon.get("slot") == active_slot:
            continue
        out.append(mon)
    return out


def _defensive_value(candidate, enemy):
    """(worst_incoming_fraction, safe).

    ``worst_incoming_fraction`` = the enemy's largest **max-roll** damage as a
    fraction of the candidate's **current** HP (not max HP), over the enemy
    moves whose type/power we can read.

    ``safe`` is True only if ALL of:
      * the candidate has current HP > 0 and a valid checksum,
      * the enemy has at least one readable move and NO unreadable move
        (an unknown enemy move could be anything -> never "safe"),
      * the worst readable incoming hit leaves >= 50% current-HP headroom.
    Incomplete mechanics / unknown moves are never interpreted as safety.
    """
    try:
        hp = int(candidate.get("cur_hp", 0))
    except (TypeError, ValueError):
        return None, False
    if hp <= 0 or not candidate.get("checksum_ok", True):
        return None, False

    moves = (enemy or {}).get("moves") or []
    if not moves:
        return None, False                      # no known threat -> cannot prove safe

    worst = None
    all_readable = True
    for mv in moves:
        d = damage_range(enemy, candidate, mv)
        if d.get("unknown"):
            all_readable = False
            continue
        if not d.get("is_damaging"):
            continue
        md = d.get("max_damage")
        if md is None:
            all_readable = False
            continue
        frac = md / hp
        if worst is None or frac > worst:
            worst = frac

    safe = bool(all_readable and worst is not None
                and worst <= _SAFE_INCOMING_FRACTION)
    return worst, safe


def _execution_ready(snapshot):
    """Mirror of battle_ram.battle_snapshot_ready_for_execution without the
    import cycle: a decision is only live-executable with an authoritative
    active slot AND confirmed menu/cursor detection."""
    s = snapshot or {}
    return bool(s.get("active_slot_authoritative")
                and s.get("menu_cursor_known"))


def decide(snapshot):
    """Main entry. Returns {"action", "reason", "detail", "executable"}.

    ``executable`` is False whenever the snapshot is not execution-ready
    (Phase 1: always False). The action is still computed for offline
    evaluation / telemetry.
    """
    detail = {"candidates": [], "notes": []}
    snapshot = snapshot or {}
    active = snapshot.get("player_active")
    enemy = snapshot.get("enemy_active")
    executable = _execution_ready(snapshot)
    if not snapshot.get("active_slot_authoritative", False):
        detail["notes"].append(
            "active Pokémon is a heuristic guess (no gBattlerPartyIndexes); "
            "decision is advisory only, executable=False")

    def result(action, reason):
        return {"action": action, "reason": reason, "detail": detail,
                "executable": executable}

    if not isinstance(active, dict) or not active.get("moves"):
        return result("MOVE_1", "no readable active Pokémon/moves; safe default")

    usable = _usable_moves(active)
    if not usable:
        # every move is out of PP -> Struggle happens automatically; pressing
        # a move is still the safe menu path.
        return result("MOVE_1", "no PP on any move (Struggle); safe default")

    if not isinstance(enemy, dict) or not enemy.get("cur_hp"):
        # In battle but enemy not readable -> attack with the highest-PP
        # non-status move we can read; fall back to the first usable slot.
        ranked = sorted(
            (t for t in usable if not _move_is_status(t[1])),
            key=lambda t: -_pp(t[1]))
        idx = ranked[0][0] if ranked else usable[0][0]
        return result(MOVE_ACTIONS[idx],
                      "enemy not readable; highest-PP readable attack")

    scored = []
    for i, mv in usable:
        s = _score_move(active, enemy, mv)
        s["index"] = i
        s["move_id"] = mv.get("id")
        scored.append(s)
    detail["candidates"] = scored

    damaging = [s for s in scored if s["classification"] == "damaging"]
    enemy_threat, threat_known = _best_enemy_threat(active, enemy)
    try:
        our_hp = int(active.get("cur_hp", 0))
    except (TypeError, ValueError):
        our_hp = 0
    lethal_next_turn = (threat_known and enemy_threat is not None
                        and enemy_threat >= our_hp)
    detail["notes"].append(
        f"our_hp={our_hp} enemy_expected_hit={enemy_threat} "
        f"lethal_next_turn={lethal_next_turn} threat_known={threat_known}")

    # 1) a move whose every damage roll KOs the enemy *if it lands*. Not a true
    #    guarantee (accuracy / abilities / items unmodelled) -> phrased as an
    #    estimate.
    ko_moves = [s for s in damaging if s["ko_on_hit_guaranteed"]]
    if ko_moves:
        ko_moves.sort(key=lambda s: (-(s["hit_prob"] or 0.0),
                                     -(s["weighted_expected"] or 0)))
        s = ko_moves[0]
        return result(MOVE_ACTIONS[s["index"]],
                      f"estimated KO on hit (eff x{s['effectiveness']}, "
                      f"acc {s['hit_prob']}); mechanics incomplete")

    # 2) if we would be KO'd next turn, consider switching to something safer.
    if lethal_next_turn:
        ranked = []
        for c in _switch_candidates(snapshot):
            worst, safe = _defensive_value(c, enemy)
            if safe:
                ranked.append((worst, c))
        if ranked:
            ranked.sort(key=lambda t: t[0])
            slot = ranked[0][1].get("slot")
            if isinstance(slot, int) and 0 <= slot < 6:
                return result(
                    SWITCH_ACTIONS[slot],
                    f"about to be KO'd; switch to a slot that survives the "
                    f"readable hits (incoming <=~{ranked[0][0]:.0%} current HP)")
        # no provably-safe switch. try to trade - best possible-KO move.
        poss = [s for s in damaging if s["ko_on_hit_possible"]]
        if poss:
            poss.sort(key=lambda s: -(s["weighted_expected"] or 0))
            s = poss[0]
            return result(MOVE_ACTIONS[s["index"]],
                          "no provably-safe switch; possible-KO trade attempt")
        # cannot even possibly KO the enemy this turn, we die next turn, no safe
        # switch: flee if this is an escapable wild battle.
        if snapshot.get("can_escape") and not snapshot.get("is_trainer"):
            return result(RUN_ACTION,
                          "about to be KO'd, unwinnable this turn, escapable "
                          "wild; flee")
        if damaging:
            s = sorted(damaging, key=lambda s: -(s["weighted_expected"] or 0))[0]
            return result(MOVE_ACTIONS[s["index"]],
                          "no safe switch, cannot flee; best remaining attack")

    # 3) otherwise: highest weighted expected damage among damaging moves.
    if damaging:
        damaging.sort(key=lambda s: (-(s["weighted_expected"] or -1),
                                     -(s["effectiveness"] or 0)))
        s = damaging[0]
        # 3b) escape only for a plainly unwinnable escapable WILD battle:
        #     our best move is barely denting AND we are threatened.
        if (snapshot.get("can_escape") and not snapshot.get("is_trainer")
                and s["ko_fraction"] is not None and s["ko_fraction"] < 0.15
                and lethal_next_turn):
            return result(RUN_ACTION,
                          "escapable wild battle we cannot meaningfully win; flee")
        return result(MOVE_ACTIONS[s["index"]],
                      f"best expected damage (eff x{s['effectiveness']}, "
                      f"~{s['ko_fraction']} of enemy HP)")

    # 4) only status moves available (or all mechanics unknown).
    known_status = [s for s in scored if s["classification"] == "status"]
    if known_status:
        known_status.sort(key=lambda s: -_pp(
            (active.get("moves") or [{}])[s["index"]]))
        s = known_status[0]
        return result(MOVE_ACTIONS[s["index"]],
                      "no damaging option; highest-PP status move")

    return result(MOVE_ACTIONS[usable[0][0]], "fallback: first usable move")
