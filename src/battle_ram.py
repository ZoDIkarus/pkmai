"""Battle-state RAM reads for the isolated Battle system (Phase 1).

**Isolation:** nothing here is imported by ``pokemon_env`` / ``train`` /
``watch``. It is used only by the (not-yet-wired) battle controller, battle
environment and battle-PPO observation builder.

**Fail-closed contract:** any value that cannot be decoded from a *verified*
RAM location is returned as ``None`` and the containing dict carries an
explicit ``*_known`` flag = ``False``. Nothing is guessed.

Verified sources (reused from ``firered_ram``, confirmed earlier in this
project against pret/pokefirered + BPRD ROM inspection):
  * player party  @ EWRAM+0x24284   (Gen-III box decrypt + checksum)
  * enemy party   @ EWRAM+0x2402C
  * gBattleTypeFlags u32 @ EWRAM+0x22B4C
  * trainer id / battle outcome (fail-closed pair)
  * gMain.inBattle bit  (located by ROM-callback signature, version-independent)

Held item is decoded here directly from the already-decrypted Growth substruct
(offset +2) so ``firered_ram`` stays byte-for-byte unchanged.

UNVERIFIED for BPRD (German FireRed) - deliberately NOT read, returned as
unknown until confirmed against this exact ROM:
  * gBattleMons  (the live in-battle Pokémon copies: authoritative HP, stat
    stages, in-battle types, ability, in-battle item)
  * gBattlerPartyIndexes (which party slot is the active battler)
  * gBattleStruct / gActionSelectionCursor / gMoveSelectionCursor
    (battle-menu + cursor state - needed by the Phase-2 macro executor)
  * gBattleWeather
See ``UNVERIFIED_BATTLE_ADDRESSES`` and docs/BATTLE_ARCHITECTURE.md.
"""
from __future__ import annotations

from firered_ram import (
    read_player_party, read_enemy_party, read_battle_type_flags,
    read_trainer_battle, read_player_location,
    _decrypt_box_data, _u16_bytes, _u32,
    PLAYER_PARTY_OFFSET, ENEMY_PARTY_OFFSET, BATTLE_TYPE_FLAGS_OFFSET,
    POKEMON_STRUCT_SIZE, MAX_PARTY_SIZE,
)
from battle_state import MainBattleReader
import pokedb

# --- gBattleTypeFlags bits (Gen III, confirmed) -------------------------
BATTLE_TYPE_DOUBLE = 0x0001
BATTLE_TYPE_LINK = 0x0002
BATTLE_TYPE_WILD = 0x0004         # NOTE: not reliably set for every wild fight
BATTLE_TYPE_TRAINER = 0x0008
BATTLE_TYPE_FIRST_BATTLE = 0x0010
BATTLE_TYPE_SAFARI = 0x0080
BATTLE_TYPE_OLD_MAN_TUTORIAL = 0x0100
BATTLE_TYPE_ROAMER = 0x8000
BATTLE_TYPE_LEGENDARY = 0x2000000

# Party-struct status1 bitfield (u32 at mon+0x50) - Gen III.
STATUS_SLEEP_MASK = 0x07
STATUS_POISON = 0x08
STATUS_BURN = 0x10
STATUS_FREEZE = 0x20
STATUS_PARALYSIS = 0x40
STATUS_TOXIC = 0x80

# Registry of the things we know we would need but have NOT verified for BPRD.
UNVERIFIED_BATTLE_ADDRESSES = {
    "gBattleMons": "live in-battle stat stages / HP / in-battle types / ability",
    "gBattlerPartyIndexes": "active party slot per battler",
    "gActionSelectionCursor": "top-level battle menu cursor (FIGHT/BAG/POKEMON/RUN)",
    "gMoveSelectionCursor": "move-list cursor",
    "gBattleWeather": "battle weather",
    "gBattleStruct.runTries / escape flags": "escape-attempt bookkeeping",
}

_status_bits = {
    "sleep": lambda v: (v & STATUS_SLEEP_MASK) != 0,
    "poison": lambda v: (v & STATUS_POISON) != 0,
    "burn": lambda v: (v & STATUS_BURN) != 0,
    "freeze": lambda v: (v & STATUS_FREEZE) != 0,
    "paralysis": lambda v: (v & STATUS_PARALYSIS) != 0,
    "toxic": lambda v: (v & STATUS_TOXIC) != 0,
}


def status_flags(status_value):
    try:
        v = int(status_value)
    except (TypeError, ValueError):
        return {k: False for k in _status_bits}
    return {k: fn(v) for k, fn in _status_bits.items()}


# ---------------------------------------------------------------------------
# held item - decoded from the already-decrypted Growth substruct (+2 u16)
# ---------------------------------------------------------------------------
def _held_item_for_slot(ram, party_offset, slot):
    base = party_offset + slot * POKEMON_STRUCT_SIZE
    if ram is None or base < 0 or base + POKEMON_STRUCT_SIZE > len(ram):
        return None
    try:
        personality, _ot, chunks = _decrypt_box_data(ram, base)
    except Exception:
        return None
    growth = chunks.get("G")
    if growth is None or len(growth) < 4:
        return None
    return _u16_bytes(growth, 2)


def _annotate_mon(mon, ram, party_offset):
    """Add DB-derived + safe-decode fields to a firered_ram mon dict.
    Everything unknown stays None with an explicit flag."""
    if not isinstance(mon, dict):
        return mon
    sid = mon.get("species_id")
    types = pokedb.species_types(sid)
    m = dict(mon)
    m["types"] = types                       # None if species unknown
    m["types_known"] = types is not None
    m["held_item"] = _held_item_for_slot(ram, party_offset, mon.get("slot", -1))
    m["status_flags"] = status_flags(mon.get("status"))
    # in-battle stat stages are UNVERIFIED for this ROM -> neutral + flag.
    m["stat_stages"] = {k: 0 for k in
                        ("attack", "defense", "speed", "sp_attack", "sp_defense",
                         "accuracy", "evasion")}
    m["stat_stages_known"] = False
    # enrich moves with DB mechanics
    enriched = []
    for mv in m.get("moves") or []:
        mech = pokedb.move_mechanics(mv.get("id"))
        e = dict(mv)
        if mech:
            e.update({
                "type": mech["type"], "power": mech["power"],
                "accuracy": mech["accuracy"], "priority": mech["priority"],
                "is_status": mech["is_status"], "max_pp": mech["pp"],
                "mechanics_known": True,
            })
        else:
            e["mechanics_known"] = False
        enriched.append(e)
    m["moves"] = enriched
    return m


def _first_active(party):
    """Heuristic active battler = first non-fainted, valid-checksum mon.

    Singles only. Correct in the overwhelming majority of cases but NOT
    authoritative (a mid-battle switch to a later slot is not tracked without
    gBattlerPartyIndexes). The returned dict carries ``active_confidence``.
    """
    for mon in party or []:
        if mon.get("checksum_ok") and int(mon.get("cur_hp", 0)) > 0:
            return mon
    return None


def read_battle_type_flags_checked(env):
    """``(flags:int | None, known:bool)``.

    ``known`` is True ONLY when a RAM buffer long enough to actually contain
    ``gBattleTypeFlags`` was read. A short / absent buffer -> ``(None, False)``.

    This exists because :func:`firered_ram.read_battle_type_flags` collapses
    "buffer too short" and "flags are legitimately 0" to the same ``0``. A wild
    battle in Gen III really does have ``flags == 0``, so that ambiguity must be
    resolved before ``can_escape`` may be trusted. On a valid read of a real
    wild battle this returns ``(0, True)``.
    """
    try:
        ram = env.get_ram()
    except Exception:
        ram = None
    if ram is None or len(ram) < BATTLE_TYPE_FLAGS_OFFSET + 4:
        return None, False
    return _u32(ram, BATTLE_TYPE_FLAGS_OFFSET), True


def can_escape(flags):
    """Flag-level check: *nothing in these flags forbids fleeing*.

    Fail-closed: unknown / non-integer flags -> False. A trainer / link /
    safari / tutorial flag -> False. ``0`` (an ordinary wild battle) -> True.

    This is NOT sufficient on its own to actually press RUN - use
    :func:`escape_allowed`, which also requires a *verified* flag read and a
    confirmed active battle. Trapping moves / abilities are still not modelled;
    the executor must verify the real menu response.
    """
    try:
        f = int(flags)
    except (TypeError, ValueError):
        return False
    if f == 0:
        return True
    blocking = (BATTLE_TYPE_TRAINER | BATTLE_TYPE_LINK | BATTLE_TYPE_SAFARI
                | BATTLE_TYPE_OLD_MAN_TUTORIAL)
    return (f & blocking) == 0


def escape_allowed(flags, flags_known, in_battle_confirmed):
    """Fail-closed "may we propose RUN?" decision.

    True only if ALL of:
      * ``flags_known`` is True (the gBattleTypeFlags read was provably valid),
      * ``in_battle_confirmed`` is exactly True (a battle is confirmed active),
      * the flags classify as an escapable wild battle (:func:`can_escape`),
      * no hard blocker bit is set (trainer / link / safari / tutorial /
        roamer / legendary).
    Anything unknown or unverified -> False.
    """
    if flags_known is not True or in_battle_confirmed is not True:
        return False
    if not can_escape(flags):
        return False
    try:
        f = int(flags)
    except (TypeError, ValueError):
        return False
    hard_block = (BATTLE_TYPE_TRAINER | BATTLE_TYPE_LINK | BATTLE_TYPE_SAFARI
                  | BATTLE_TYPE_OLD_MAN_TUTORIAL | BATTLE_TYPE_ROAMER
                  | BATTLE_TYPE_LEGENDARY)
    return (f & hard_block) == 0


def is_trainer_battle(flags):
    try:
        return bool(int(flags) & BATTLE_TYPE_TRAINER)
    except (TypeError, ValueError):
        return False


def in_battle(env, battle_reader=None):
    """(bool_or_None, source). Uses the version-independent gMain.inBattle
    locator; falls back to None (caller uses the conservative BattleState)."""
    try:
        ram = env.get_ram()
    except Exception:
        ram = None
    if ram is None:
        return None, "no_ram"
    reader = battle_reader if battle_reader is not None else MainBattleReader()
    val = reader.read(ram)
    if val is None:
        return None, "gMain_not_located"
    return bool(val), "gMain.inBattle"


def battle_snapshot(env, battle_reader=None):
    """Assemble everything the rule controller / battle observation needs.

    Returns a dict with an explicit confidence surface. Never raises.
    """
    snap = {
        "schema": "battle_snapshot_v2",
        "ram_ok": False,
        "in_battle": None, "in_battle_source": "unknown",
        "battle_type_flags": None,
        "battle_type_flags_known": False,     # 0 is a real value; distinguish it
        "is_trainer": None, "is_double": None, "is_wild_flagged": None,
        "can_escape": False,
        "trainer_id": 0, "battle_outcome": None,
        "player_active": None, "enemy_active": None,
        "player_active_source": "heuristic_first_alive",
        "player_party": [], "enemy_party": [],
        "active_slot_authoritative": False,   # would need gBattlerPartyIndexes
        "stat_stages_known": False,           # would need gBattleMons
        "menu_cursor_known": False,           # would need gActionSelectionCursor
        "weather_known": False,
        "db_available": pokedb.is_available(),
        "unverified": dict(UNVERIFIED_BATTLE_ADDRESSES),
    }
    try:
        ram = env.get_ram()
    except Exception:
        ram = None
    if ram is None:
        return snap
    snap["ram_ok"] = True

    ib, src = in_battle(env, battle_reader)
    snap["in_battle"] = ib
    snap["in_battle_source"] = src

    flags, flags_known = read_battle_type_flags_checked(env)
    snap["battle_type_flags"] = int(flags) if flags_known else None
    snap["battle_type_flags_known"] = flags_known
    if flags_known:
        snap["is_trainer"] = is_trainer_battle(flags)
        snap["is_double"] = bool(flags & BATTLE_TYPE_DOUBLE)
        snap["is_wild_flagged"] = bool(flags & BATTLE_TYPE_WILD)
    snap["can_escape"] = escape_allowed(flags, flags_known, snap["in_battle"])

    tid, outcome = read_trainer_battle(env)
    snap["trainer_id"] = int(tid)
    snap["battle_outcome"] = outcome

    p_party = [_annotate_mon(m, ram, PLAYER_PARTY_OFFSET)
               for m in read_player_party(env)]
    e_party = [_annotate_mon(m, ram, ENEMY_PARTY_OFFSET)
               for m in read_enemy_party(env)]
    snap["player_party"] = p_party
    snap["enemy_party"] = e_party
    snap["player_active"] = _first_active(p_party)
    snap["enemy_active"] = _first_active(e_party)
    # The active mon is a heuristic guess until gBattlerPartyIndexes is verified.
    snap["player_active_source"] = "heuristic_first_alive"
    return snap


# Fields that must be AUTHORITATIVE before a live executing battle controller
# (Phase 2 Router / Executor / Battle-PPO acting on the real game) may run.
_EXECUTION_REQUIREMENTS = (
    ("in_battle", lambda s: s.get("in_battle") is True,
     "confirmed active battle (gMain.inBattle == True)"),
    ("battle_type_flags_known", lambda s: bool(s.get("battle_type_flags_known")),
     "verified gBattleTypeFlags read"),
    ("stat_stages_known", lambda s: bool(s.get("stat_stages_known")),
     "gBattleMons (live HP / stat stages / in-battle types / ability)"),
    ("active_slot_authoritative", lambda s: bool(s.get("active_slot_authoritative")),
     "gBattlerPartyIndexes (authoritative active party slot)"),
    ("menu_cursor_known", lambda s: bool(s.get("menu_cursor_known")),
     "gActionSelectionCursor / gMoveSelectionCursor (battle-menu state)"),
    ("player_active", lambda s: isinstance(s.get("player_active"), dict),
     "readable active player Pokémon"),
    ("enemy_active", lambda s: isinstance(s.get("enemy_active"), dict),
     "readable active enemy Pokémon"),
)


def battle_snapshot_ready_for_execution(snapshot):
    """``(ready: bool, missing: list[str])`` — the FULL requirement set (every
    action). See :func:`battle_snapshot_ready_for_action` for the per-action
    subset.

    In Phase 1 this ALWAYS returns ``(False, [...])`` because ``gBattleMons``,
    ``gBattlerPartyIndexes`` and the battle-menu cursors are not verified for
    the BPRD ROM. Phase 2 / Router / Executor / Battle-PPO MUST NOT be
    live-activated while this returns ``ready == False``.
    """
    s = snapshot or {}
    missing = [why for _key, ok, why in _EXECUTION_REQUIREMENTS if not ok(s)]
    return (not missing, missing)


# Which authoritative fields each macro *kind* actually needs. Anything not
# listed stays masked but does not block that action.
_ACTION_REQUIREMENTS = {
    "MOVE": ("in_battle", "player_active", "enemy_active", "menu_cursor_known",
             "stat_stages_known"),
    "SWITCH": ("in_battle", "player_active", "active_slot_authoritative",
               "menu_cursor_known"),
    "RUN": ("in_battle", "battle_type_flags_known", "menu_cursor_known"),
}
_REQ_BY_KEY = {k: (ok, why) for k, ok, why in _EXECUTION_REQUIREMENTS}


def battle_snapshot_ready_for_action(snapshot, action):
    """``(ready, missing)`` for one macro action.

    * ``MOVE_n``  needs active battlers, the menu/cursor state and live move/PP
      mechanics (``gBattleMons``);
    * ``SWITCH_n`` needs the authoritative active slot, party states and the
      switch menu — never the ``_first_active`` heuristic;
    * ``RUN``     needs the battle-type flags, escape permission and the menu.

    Unknown fields stay masked. In Phase 1 every branch still returns
    ``ready == False`` (menu cursors unverified).
    """
    s = snapshot or {}
    kind = "MOVE" if str(action).startswith("MOVE") else \
           "SWITCH" if str(action).startswith("SWITCH") else \
           "RUN" if str(action) == "RUN" else None
    if kind is None:
        return False, [f"unknown action {action!r}"]
    missing = []
    for key in _ACTION_REQUIREMENTS[kind]:
        ok, why = _REQ_BY_KEY[key]
        if not ok(s):
            missing.append(why)
    if kind == "SWITCH" and s.get("player_active_source") == "heuristic_first_alive":
        missing.append("SWITCH must not use the first-alive heuristic slot")
    if kind == "RUN" and s.get("can_escape") is not True:
        missing.append("escape not permitted / not confirmed")
    return (not missing, missing)
