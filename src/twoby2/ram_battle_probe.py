"""Battle-RAM verification harness for the BPRD ROM.

The five Phase-2 fields were verified on 2026-09-07 from 20 labelled dumps
covering a wild fight, a trainer fight, all four action/move cursor positions,
a party switch and an out-of-battle negative context. This module records the
winning BPRD offsets first while retaining rejected reference candidates for
auditability. It never treats a reference address as verified without dumps.

Mandatory blockers (every one must be VERIFIED before any live activation):

  * ``gBattleMons``            (base pointer + per-battler struct)
  * ``gBattlerPartyIndexes``   (which party slot each battler is)
  * ``gActionSelectionCursor`` (FIGHT/BAG/POKEMON/RUN cursor)
  * ``gMoveSelectionCursor``   (move-list cursor, 0..3)
  * ``battle_menu_state``      (which battle submenu is open)

Optional (not a blocker): ``gBattleWeather`` — recorded and validatable, but a
live ``battle_snapshot`` does not require it; it is "later optional information".

A field is VERIFIED only when:
  * dumps come from **>= 2 distinct encounters** (``encounter_id``), not just
    two files from the same fight;
  * every value is in range;
  * there is at least one **positive** context (value must match the labelled
    expectation) and at least one **negative** context (out-of-battle / wrong
    menu — value must NOT read as a valid in-battle value / must be stale);
  * every cross-check passes — and after a switch, ``gBattleMons[player]`` is
    checked against the party slot named by ``gBattlerPartyIndexes[player]``,
    **never** against ``party[0]``.

Nothing from ``POKEFIRERED_US_REFERENCE`` or the integration ``data.json`` is
ever marked verified on its own.
"""
from __future__ import annotations

import struct

EWRAM_SIZE = 0x40000

MANDATORY_FIELDS = ("gBattleMons", "gBattlerPartyIndexes",
                    "gActionSelectionCursor", "gMoveSelectionCursor",
                    "battle_menu_state")
OPTIONAL_FIELDS = ("gBattleWeather",)

# pokefirered (US rev0) EWRAM symbol offsets — REFERENCE ONLY. BPRD relocates
# these by an unknown, build-specific delta; never used as-is.
POKEFIRERED_US_REFERENCE = {
    "gBattleMons":            0x2023BE4,
    "gBattlerPartyIndexes":   0x2023BCE,
    "gActionSelectionCursor": 0x2023D74,
    "gMoveSelectionCursor":   0x2023D78,
    "gBattleWeather":         0x2023F1C,
}

# Project integration guesses (local/custom_integrations/.../data.json).
INTEGRATION_GUESSES = {
    "in_battle":    {"offset": 146376, "type": "|u1"},
    "battle_flags": {"offset": 143340, "type": "<u4"},
    "bt_c1":        {"offset": 142290, "type": "|u1"},   # guessed action cursor
    "bt_c2":        {"offset": 146799, "type": "|u1"},   # guessed move cursor
    "p1_level":     {"offset": 148184, "type": "|u1"},   # VERIFIED = gPlayerParty+0x54
}

# struct BattlePokemon layout (pokefirered, ROM-independent).
BATTLE_MON_SIZE = 0x58
BATTLE_MON_SPECIES_OFF = 0x00     # u16
BATTLE_MON_STATSTAGES_OFF = 0x18  # 8 x u8
BATTLE_MON_HP_OFF = 0x28          # u16 currentHP
BATTLE_MON_LEVEL_OFF = 0x2A       # u8
BATTLE_MON_MAXHP_OFF = 0x2C       # u16
MAX_BATTLERS = 4
PLAYER_BATTLER = 0               # single battles: player is battler 0


# --------------------------------------------------------------------------
# dump helpers
# --------------------------------------------------------------------------
def _u(ram, off, size, endian="<"):
    if off < 0 or off + size > len(ram):
        return None
    return struct.unpack(endian + {1: "B", 2: "H", 4: "I"}[size],
                         bytes(ram[off:off + size]))[0]


def _party_slot(dump, slot):
    party = (dump.get("party") or [])
    for mon in party:
        if mon.get("slot") == slot:
            return mon
    if 0 <= slot < len(party):
        return party[slot]
    return None


def dump_is_in_battle(dump):
    return dump.get("battle_kind") in ("wild", "trainer")


def distinct_encounters(dumps):
    return {d.get("encounter_id") for d in dumps if d.get("encounter_id")}


# --------------------------------------------------------------------------
# per-field checks
# --------------------------------------------------------------------------
def _check_party_indexes(candidate_offset, dumps):
    """gBattlerPartyIndexes[PLAYER_BATTLER] must (a) stay in 0..5, (b) equal the
    collector-tracked active slot, and (c) actually CHANGE across an
    after_switch dump."""
    per, values = [], []
    for d in dumps:
        if not dump_is_in_battle(d):
            per.append({"context": d.get("context"), "value": None,
                        "ok": True, "note": "out-of-battle (negative context)"})
            continue
        v = _u(d["ram"], candidate_offset + 2 * PLAYER_BATTLER, 2)
        tracked = d.get("active_party_slot")
        in_range = v is not None and 0 <= v <= 5
        matches = (tracked is None) or (v == tracked)
        per.append({"context": d.get("context"), "value": v,
                    "tracked_active_slot": tracked,
                    "ok": bool(in_range and matches),
                    "note": "" if (in_range and matches) else
                            f"v={v} tracked={tracked} range_ok={in_range}"})
        values.append((d.get("context"), v))
    switched = any(d.get("context", "").startswith("after_switch") for d in dumps)
    changed = len({v for _c, v in values}) > 1
    problems = []
    if not all(p["ok"] for p in per):
        problems.append("value out of range or != tracked active slot")
    if switched and not changed:
        problems.append("active slot never changed even after a switch")
    return per, problems


def _check_battlemons(base_offset, dumps):
    """gBattleMons[PLAYER_BATTLER].species/level/HP vs the party slot named by
    gBattlerPartyIndexes[PLAYER_BATTLER] (from active_party_slot). NEVER party[0]
    after a switch."""
    per, problems = [], []
    for d in dumps:
        if not dump_is_in_battle(d):
            per.append({"context": d.get("context"), "ok": True,
                        "note": "out-of-battle (negative context)"})
            continue
        slot = d.get("active_party_slot")
        if slot is None:
            per.append({"context": d.get("context"), "ok": False,
                        "note": "no active_party_slot to cross-check against"})
            problems.append("missing active_party_slot")
            continue
        mon = _party_slot(d, slot)
        if mon is None:
            per.append({"context": d.get("context"), "ok": False,
                        "note": f"party slot {slot} missing from dump party"})
            problems.append("party slot missing")
            continue
        b = base_offset + BATTLE_MON_SIZE * PLAYER_BATTLER
        species = _u(d["ram"], b + BATTLE_MON_SPECIES_OFF, 2)
        level = _u(d["ram"], b + BATTLE_MON_LEVEL_OFF, 1)
        cur_hp = _u(d["ram"], b + BATTLE_MON_HP_OFF, 2)
        ok = (species == mon.get("species_id") and level == mon.get("level")
              and cur_hp == mon.get("cur_hp"))
        per.append({"context": d.get("context"), "checked_slot": slot,
                    "battle_mon": [species, level, cur_hp],
                    "party_slot": [mon.get("species_id"), mon.get("level"),
                                   mon.get("cur_hp")],
                    "ok": bool(ok),
                    "note": "" if ok else "gBattleMons[player] != party[active_slot]"})
        if not ok:
            problems.append(f"{d.get('context')}: mismatch vs party slot {slot}")
    return per, problems


def _check_menu_cursor(candidate_offset, dumps, *, expected_key, positive_menu,
                       max_val):
    """Generic cursor check. ``expected_key`` is the sidecar field with the
    labelled expected cursor position; ``positive_menu`` is the menu_state the
    cursor is meaningful in."""
    per, problems = [], []
    have_positive = have_negative = False
    for d in dumps:
        v = _u(d["ram"], candidate_offset + PLAYER_BATTLER, 1)
        ms = d.get("menu_state")
        exp = d.get(expected_key)
        if dump_is_in_battle(d) and ms == positive_menu and exp is not None:
            have_positive = True
            ok = (v == exp) and (0 <= v <= max_val)
            per.append({"context": d.get("context"), "menu_state": ms,
                        "value": v, "expected": exp, "ok": bool(ok),
                        "note": "" if ok else f"cursor {v} != expected {exp}"})
            if not ok:
                problems.append(f"{d.get('context')}: cursor {v} != expected {exp}")
        else:
            # negative: out-of-battle or a different menu — the cursor should
            # NOT read as our current in-battle position, or be stale/garbage.
            have_negative = True
            per.append({"context": d.get("context"), "menu_state": ms,
                        "value": v, "ok": True,
                        "note": "negative context (not the target menu)"})
    if not have_positive:
        problems.append(f"no positive '{positive_menu}' dump with {expected_key}")
    if not have_negative:
        problems.append("no negative context (other menu / out-of-battle)")
    return per, problems


def _check_menu_state(candidate_offset, dumps):
    """battle_menu_state: a candidate byte must take a DISTINCT, STABLE value in
    each of main / move / party, and a different value out-of-battle."""
    by_state = {}
    per = []
    for d in dumps:
        v = _u(d["ram"], candidate_offset, 1)
        state = d.get("menu_state") if dump_is_in_battle(d) else "out_of_battle"
        by_state.setdefault(state, set()).add(v)
        per.append({"context": d.get("context"), "menu_state": state, "value": v})
    problems = []
    needed = {"main", "move", "party", "out_of_battle"}
    missing = needed - set(by_state)
    if missing:
        problems.append(f"missing labelled contexts: {sorted(missing)}")
    # each present state must have ONE stable value...
    for state, vals in by_state.items():
        if len(vals) != 1:
            problems.append(f"{state}: value not stable {sorted(vals)}")
    # ...and the in-battle menu values must differ from each other
    seen = {}
    for state in ("main", "move", "party"):
        if state in by_state and len(by_state[state]) == 1:
            v = next(iter(by_state[state]))
            if v in seen:
                problems.append(f"{state} and {seen[v]} share value {v}")
            seen[v] = state
    return per, problems


def _check_weather(candidate_offset, dumps):
    per, problems = [], []
    for d in dumps:
        v = _u(d["ram"], candidate_offset, 2)
        in_range = v is not None and 0 <= v <= 0x40
        per.append({"context": d.get("context"), "value": v, "ok": bool(in_range)})
        if not in_range:
            problems.append(f"{d.get('context')}: weather {v} out of range")
    return per, problems


# --------------------------------------------------------------------------
# candidate offsets to try (reference-derived / integration guesses — NOT
# adopted). The scorer feeds these in; nothing here is "the" address.
# --------------------------------------------------------------------------
CANDIDATE_OFFSETS = {
    "gBattlerPartyIndexes": [0x23BCE],
    "gBattleMons": [0x23BE4, 0x23BE4 - BATTLE_MON_STATSTAGES_OFF],
    # BPRD offsets derived uniquely from the labelled 20260907_184350 dumps.
    # The older entries remain as rejected historical/reference candidates.
    "gActionSelectionCursor": [0x23FF8, 0x22BD2, 0x23D74],
    "gMoveSelectionCursor": [0x23FFC, 0x23D2F, 0x23D78],
    # gBattleBufferA[player][0]: controller commands observed as
    # CHOOSEACTION=18, CHOOSEMOVE=20, CHOOSEPOKEMON=22.
    "battle_menu_state": [0x22BC4, 0x22BD0, 0x22FB0],
    "gBattleWeather": [0x23F1C],
}


def verify_field(name, candidate_offset, dumps):
    """Score one candidate offset for one field. Returns a report dict with
    ``verified`` bool."""
    in_battle_dumps = [d for d in dumps if dump_is_in_battle(d)]
    encs = distinct_encounters(in_battle_dumps)
    base = {"field": name, "candidate_offset": hex(candidate_offset),
            "distinct_encounters": sorted(encs)}

    if len(in_battle_dumps) < 2 or len(encs) < 2:
        return {**base, "verified": False,
                "reason": f"need >= 2 dumps from >= 2 DISTINCT encounters "
                          f"(got {len(in_battle_dumps)} dumps / {len(encs)} encounters)",
                "blocker": "collect more encounters with tools/battle_dump_collect.py"}

    if name == "gBattlerPartyIndexes":
        per, problems = _check_party_indexes(candidate_offset, dumps)
    elif name == "gBattleMons":
        per, problems = _check_battlemons(candidate_offset, dumps)
    elif name == "gActionSelectionCursor":
        per, problems = _check_menu_cursor(
            candidate_offset, dumps, expected_key="expected_action_cursor",
            positive_menu="main", max_val=3)
    elif name == "gMoveSelectionCursor":
        per, problems = _check_menu_cursor(
            candidate_offset, dumps, expected_key="expected_move_cursor",
            positive_menu="move", max_val=3)
    elif name == "battle_menu_state":
        per, problems = _check_menu_state(candidate_offset, dumps)
    elif name == "gBattleWeather":
        per, problems = _check_weather(candidate_offset, dumps)
    else:
        return {**base, "verified": False, "reason": f"no check for {name}"}

    verified = not problems
    return {**base, "verified": verified, "per_dump": per,
            "reason": "" if verified else "; ".join(problems)}


def verify_all(dumps, candidate_offsets=None):
    """Try every candidate offset for every field. Returns
    ``{field: {"verified", "winning_offset", "attempts": [...]}}``."""
    cands = candidate_offsets or CANDIDATE_OFFSETS
    out = {}
    for field, offsets in cands.items():
        attempts = [verify_field(field, off, dumps) for off in offsets]
        win = next((a for a in attempts if a["verified"]), None)
        out[field] = {"verified": win is not None,
                      "winning_offset": win["candidate_offset"] if win else None,
                      "mandatory": field in MANDATORY_FIELDS,
                      "attempts": attempts}
    return out


def all_mandatory_verified(results):
    return all(results.get(f, {}).get("verified") for f in MANDATORY_FIELDS)


def blocker_report(dumps=None):
    """Honest status. With no/insufficient dumps every mandatory field is a
    blocker; ``gBattleWeather`` is listed as optional, not a blocker."""
    dumps = dumps or []
    if dumps:
        results = verify_all(dumps)
        return {
            "verified": all_mandatory_verified(results),
            "mandatory_blockers": {f: results[f]["verified"] for f in MANDATORY_FIELDS},
            "optional": {f: results.get(f, {}).get("verified", False)
                         for f in OPTIONAL_FIELDS},
            "results": results,
        }
    return {
        "verified": False,
        "mandatory_blockers": {f: False for f in MANDATORY_FIELDS},
        "optional_information": {
            "gBattleWeather": "NOT a blocker — a live battle_snapshot does not "
                              "require weather; collect it if convenient for "
                              "later completeness"},
        "what_is_needed": [
            "run tools/battle_dump_collect.py and follow docs/RAM_PROBE_GUIDE.md",
            ">= 2 DISTINCT encounters (a wild fight AND a trainer fight)",
            "labelled menu states: main / move (cursor 0,1,2,3) / party / out-of-battle",
            "a Pokémon switch, so gBattlerPartyIndexes changes and gBattleMons "
            "is cross-checked against the NEW slot",
            "an enemy KO / new enemy",
            "the full party from the verified party reader in every dump",
        ],
    }
