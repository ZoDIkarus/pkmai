"""Live in-battle RAM readers for BPRD — the VERIFIED Phase-2 addresses.

Verified 2026-09-07 from the 20-dump dataset ``runtime/ram_probe/20260907_184350``
(wild + trainer fight, every cursor position, a switch, an out-of-battle
negative context). ``tools/battle_dump_score.py`` reports ALL MANDATORY FIELDS
VERIFIED. See :mod:`twoby2.ram_battle_probe`.

Stable-Retro RAM-view offsets (GBA address = 0x02000000 + offset):

  gBattlerPartyIndexes   0x23BCE   u16[4]   which party slot each battler is
  gBattleMons            0x23BE4   4 x 0x58 struct BattlePokemon
  gActionSelectionCursor 0x23FF8   u8[4]    FIGHT/BAG/POKEMON/RUN = 0/1/2/3
  gMoveSelectionCursor   0x23FFC   u8[4]    move list 0..3
  battle_menu_state      0x22BC4   u8       18=CHOOSEACTION 20=CHOOSEMOVE 22=CHOOSEPOKEMON
  gBattleWeather         0x23F1C   u16      (optional)

Single battle: battler 0 = player, battler 1 = enemy. Doubles are structurally
supported (4 battlers) but the live driver keeps doubles fail-closed until the
action routing for two active battlers is separately verified.

``gMain.inBattle`` (via :class:`battle_state.MainBattleReader`) is the only
trusted battle-active latch; the integration byte 0x23BC8 is a pulsing helper
signal and must not be used as the persistent state.
"""
from __future__ import annotations

import struct

# --- verified offsets -------------------------------------------------------
OFF_PARTY_INDEXES = 0x23BCE
OFF_BATTLE_MONS = 0x23BE4
OFF_ACTION_CURSOR = 0x23FF8
OFF_MOVE_CURSOR = 0x23FFC
OFF_MENU_STATE = 0x22BC4
OFF_WEATHER = 0x23F1C
OFF_INTEGRATION_PULSE = 0x23BC8   # helper only — NEVER the persistent state

MAX_BATTLERS = 4
PLAYER_BATTLER = 0
ENEMY_BATTLER = 1
BATTLE_MON_SIZE = 0x58

# battle_menu_state values
MENU_CHOOSEACTION = 18   # main menu
MENU_CHOOSEMOVE = 20     # move list
MENU_CHOOSEPOKEMON = 22  # party list
MENU_VALUES = {MENU_CHOOSEACTION: "main", MENU_CHOOSEMOVE: "move",
               MENU_CHOOSEPOKEMON: "party"}

# struct BattlePokemon field offsets (verified against the dump dataset)
_F = {
    "species": (0x00, 2), "attack": (0x02, 2), "defense": (0x04, 2),
    "speed": (0x06, 2), "sp_attack": (0x08, 2), "sp_defense": (0x0A, 2),
    "move1": (0x0C, 2), "move2": (0x0E, 2), "move3": (0x10, 2), "move4": (0x12, 2),
    "ability": (0x20, 1), "type1": (0x21, 1), "type2": (0x22, 1),
    "pp1": (0x24, 1), "pp2": (0x25, 1), "pp3": (0x26, 1), "pp4": (0x27, 1),
    "hp": (0x28, 2), "level": (0x2A, 1), "max_hp": (0x2C, 2),
    "held_item": (0x2E, 2), "status1": (0x4C, 4), "status2": (0x50, 4),
}
_STAT_STAGES_OFF = 0x18   # 8 x u8, index 0 unused (HP), 1..7 real stages
_STAT_STAGE_KEYS = ("_hp", "attack", "defense", "speed", "sp_attack",
                    "sp_defense", "accuracy", "evasion")


class LiveReadError(RuntimeError):
    pass


def _u(ram, off, size):
    if ram is None or off < 0 or off + size > len(ram):
        return None
    return struct.unpack("<" + {1: "B", 2: "H", 4: "I"}[size],
                         bytes(ram[off:off + size]))[0]


# --- primitive reads -------------------------------------------------------
def battler_party_index(ram, battler):
    return _u(ram, OFF_PARTY_INDEXES + 2 * battler, 2)


def action_cursor(ram, battler=PLAYER_BATTLER):
    return _u(ram, OFF_ACTION_CURSOR + battler, 1)


def move_cursor(ram, battler=PLAYER_BATTLER):
    return _u(ram, OFF_MOVE_CURSOR + battler, 1)


def menu_state_raw(ram):
    return _u(ram, OFF_MENU_STATE, 1)


def menu_state(ram):
    """'main' | 'move' | 'party' | None (unknown / other -> fail-closed)."""
    return MENU_VALUES.get(menu_state_raw(ram))


def weather(ram):
    return _u(ram, OFF_WEATHER, 2)


def read_battle_mon(ram, battler):
    """Full struct BattlePokemon for one battler, or None if unreadable."""
    base = OFF_BATTLE_MONS + BATTLE_MON_SIZE * battler
    if ram is None or base + BATTLE_MON_SIZE > len(ram):
        return None
    out = {}
    for name, (o, sz) in _F.items():
        out[name] = _u(ram, base + o, sz)
    species = out.get("species")
    if not species or not (1 <= species <= 1000):
        return None
    stages = {}
    for i, key in enumerate(_STAT_STAGE_KEYS):
        if key.startswith("_"):
            continue
        v = _u(ram, base + _STAT_STAGES_OFF + i, 1)
        # Gen III stat stages are stored as 0..12 with 6 == neutral
        stages[key] = (v - 6) if (v is not None and 0 <= v <= 12) else 0
    moves = []
    for mi in range(1, 5):
        mid = out.get(f"move{mi}")
        pp = out.get(f"pp{mi}")
        if mid:
            moves.append({"id": int(mid), "pp": int(pp or 0)})
    types = [out["type1"]]
    if out["type2"] is not None and out["type2"] != out["type1"]:
        types.append(out["type2"])
    return {
        "battler": battler,
        "species_id": int(species),
        "level": int(out["level"] or 0),
        "cur_hp": int(out["hp"] or 0),
        "max_hp": int(out["max_hp"] or 0),
        "status": int(out["status1"] or 0),
        "status2": int(out["status2"] or 0),
        "ability": int(out["ability"] or 0),
        "types": [int(t) for t in types if t is not None],
        "held_item": int(out["held_item"] or 0),
        "stats": {"attack": int(out["attack"] or 0),
                  "defense": int(out["defense"] or 0),
                  "speed": int(out["speed"] or 0),
                  "sp_attack": int(out["sp_attack"] or 0),
                  "sp_defense": int(out["sp_defense"] or 0)},
        "stat_stages": stages,
        "moves": moves,
    }


def read_all(ram):
    """Everything the driver / observation needs from the verified addresses.

    Returns a dict; any field that is not authoritatively readable is ``None``
    (fail-closed). Does not decide whether a battle is active — pass
    ``in_battle`` from :class:`battle_state.MainBattleReader`.
    """
    if ram is None:
        raise LiveReadError("no RAM")
    return {
        "party_index_player": battler_party_index(ram, PLAYER_BATTLER),
        "party_index_enemy": battler_party_index(ram, ENEMY_BATTLER),
        "action_cursor": action_cursor(ram, PLAYER_BATTLER),
        "move_cursor": move_cursor(ram, PLAYER_BATTLER),
        "menu_state_raw": menu_state_raw(ram),
        "menu_state": menu_state(ram),
        "weather": weather(ram),
        "battle_mon_player": read_battle_mon(ram, PLAYER_BATTLER),
        "battle_mon_enemy": read_battle_mon(ram, ENEMY_BATTLER),
    }


# --------------------------------------------------------------------------
# Catch-v2 (spec §8): the bag / ball-pocket / catch-result RAM is NOT verified
# for the BPRD ROM. Everything here is FAIL-CLOSED — it returns ``None`` /
# ``(False, [...])`` until ``tools/catch_ram_probe.py`` confirms real offsets
# over multiple independent attempts. NO address is invented. The registry
# below is what a live catch would need and does not yet have.
# --------------------------------------------------------------------------
UNVERIFIED_CATCH_ADDRESSES = {
    "gBagPockets[BALLS] items": "the in-battle ball pocket (item id + count, cursor order)",
    "battle bag cursor": "which pocket / which row the in-battle bag is on",
    "gPokedexOwned bitfield": "the 'owned' dex bit for a species (catch confirmation)",
    "catch result / ball shake counter": "the shake / caught / broke-free signal",
    "sent-to-PC flag": "whether a caught mon went to the PC (full party)",
}

# item-id range sanity for a Poke-Ball pocket entry (FireRed ball ids 1..12).
_BALL_ID_LO, _BALL_ID_HI = 1, 12


def ball_pocket(ram):
    """The in-battle BALL pocket as ``[(item_id, count), ...]`` in cursor order,
    or ``None``. FAIL-CLOSED: the pocket address is unverified for BPRD, so this
    always returns ``None`` until the probe verifies it. A future verified
    implementation must reject a non-ball id (1..12) in the pocket as a read
    fault (return ``None``)."""
    return None


def party_slot_count(ram):
    """Number of occupied party slots, or ``None``. FAIL-CLOSED here — the live
    driver already has the VERIFIED ``firered_ram.read_player_party`` count and
    should use that; this stub exists only so the probe has a single place to
    wire a battle-RAM-view party counter if one is ever needed."""
    return None


def pokedex_owned(ram, species_id):
    """``True`` / ``False`` if the 'owned' dex bit for ``species_id`` can be
    read, else ``None``. FAIL-CLOSED: the dex bitfield offset is unverified for
    BPRD."""
    return None


def catch_result_signal(ram):
    """The catch outcome signal (shake count / caught / broke-free), or
    ``None``. FAIL-CLOSED: unverified for BPRD."""
    return None


def catch_ram_ready(ram=None):
    """``(ready: bool, missing: list[str])`` — may a LIVE CATCH macro run?

    In this build this ALWAYS returns ``(False, [...])`` because none of
    :data:`UNVERIFIED_CATCH_ADDRESSES` is verified for the BPRD ROM. The live
    catch path (``EmulatorBattleDriver`` + ``MacroExecutor._do_catch``) must
    stay masked while ``ready`` is ``False``. ``tools/catch_ram_probe.py``
    produces the dumps; ``twoby2.ram_battle_probe`` (extended) scores them; only
    a passing score flips this.
    """
    missing = list(UNVERIFIED_CATCH_ADDRESSES.values())
    return (False, missing)


def crosscheck_against_party(live, party):
    """``gBattleMons[player]`` must match ``party[gBattlerPartyIndexes[player]]``
    — never party[0] after a switch. Returns (ok, detail)."""
    slot = live.get("party_index_player")
    mon = live.get("battle_mon_player")
    if slot is None or mon is None:
        return False, "player battle-mon or party index unreadable"
    if not (0 <= slot < len(party or [])):
        return False, f"party index {slot} outside party of {len(party or [])}"
    p = party[slot]
    ok = (p.get("species_id") == mon["species_id"]
          and p.get("level") == mon["level"]
          and p.get("cur_hp") == mon["cur_hp"])
    return ok, (f"battle_mon {mon['species_id']}/L{mon['level']}/{mon['cur_hp']} "
                f"vs party[{slot}] {p.get('species_id')}/L{p.get('level')}/"
                f"{p.get('cur_hp')}")
