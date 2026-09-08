"""Shiny detection RAM layer for BPRD — FAIL-CLOSED (spec ZIEL C).

Gen-III shininess needs three values read from RAM:
  * the active WILD enemy's 32-bit personality value (PID),
  * the player's Trainer ID (TID, low 16 of ``playerTrainerId``),
  * the player's Secret ID (SID, high 16 of ``playerTrainerId``).

The PID comes from the enemy party struct (already decoded by
``firered_ram.read_enemy_party``). The player's ``playerTrainerId`` lives in
SaveBlock2 and is **not verified for the BPRD ROM** — so this module keeps
``SHINY_RAM_VERIFIED = False`` and :func:`shiny_status` returns ``unknown``
with the reason ``shiny_ram_unverified`` until ``tools/shiny_ram_probe.py``
confirms all three values AND their stability across a whole encounter.

Nothing here is guessed. No RAM is written. A trainer battle is never a
shiny-catchable encounter.
"""
from __future__ import annotations

import shiny as _shiny

# Flip to True ONLY after tools/shiny_ram_probe.py verifies:
#   * the SaveBlock2 playerTrainerId offset for BPRD (TID + SID),
#   * the active wild enemy PID is stable for the whole encounter,
#   * >= 2 distinct encounters agree.
SHINY_RAM_VERIFIED = False

UNVERIFIED_SHINY_ADDRESSES = {
    "gSaveBlock2Ptr->playerTrainerId": "player Trainer ID (low16) + Secret ID (high16)",
    "active wild enemy PID stability": "personality value must not drift during the encounter",
}

# statuses shiny_status() can return
STATUS_UNKNOWN = "unknown"                 # cannot decide -> never counted / rewarded
STATUS_NOT_APPLICABLE = "not_applicable"   # trainer / not a catchable wild
STATUS_VERIFIED_SHINY = "verified_shiny"
STATUS_VERIFIED_NORMAL = "verified_normal"


def _blank(status, reason, **extra):
    d = {"status": status, "reason": reason, "shiny_value": None,
         "pid": None, "tid": None, "sid": None, "ram_verified": SHINY_RAM_VERIFIED}
    d.update(extra)
    return d


def read_player_trainer_id(ram):
    """``(tid, sid)`` or ``None``. FAIL-CLOSED: the SaveBlock2 offset is
    unverified for BPRD."""
    return None


def read_active_wild_pid(enemy_party):
    """The active (first non-fainted, checksum-ok) wild enemy PID, or ``None``.
    Reads only the already-decoded ``personality`` from a verified enemy party
    struct — never invents an address."""
    for mon in enemy_party or []:
        if not isinstance(mon, dict):
            continue
        if mon.get("checksum_ok") and int(mon.get("cur_hp", 0) or 0) > 0:
            pid = mon.get("personality")
            try:
                return int(pid) & 0xFFFFFFFF if pid is not None else None
            except (TypeError, ValueError):
                return None
    return None


def shiny_status(*, is_trainer, enemy_party=None, ram=None, pid=None,
                 trainer_id=None, secret_id=None):
    """Decide the shiny status of the current wild encounter, fail-closed.

    A trainer battle is ``not_applicable``. Anything unverified / unreadable is
    ``unknown`` with a diagnostic reason — never ``False``, never a reward, never
    a catch decision.
    """
    if is_trainer:
        return _blank(STATUS_NOT_APPLICABLE, "trainer_battle")
    if not SHINY_RAM_VERIFIED:
        return _blank(STATUS_UNKNOWN, "shiny_ram_unverified")

    p = pid if pid is not None else read_active_wild_pid(enemy_party)
    tid, sid = (trainer_id, secret_id)
    if tid is None or sid is None:
        got = read_player_trainer_id(ram)
        if got is not None:
            tid, sid = got
    if p is None:
        return _blank(STATUS_UNKNOWN, "wild_pid_unreadable")
    if tid is None or sid is None:
        return _blank(STATUS_UNKNOWN, "player_trainer_id_unreadable")

    sv = _shiny.shiny_value(tid, sid, p)
    if sv is None:
        return _blank(STATUS_UNKNOWN, "shiny_value_uncomputable")
    return _blank(
        STATUS_VERIFIED_SHINY if sv < _shiny.SHINY_THRESHOLD else STATUS_VERIFIED_NORMAL,
        "computed",
        shiny_value=int(sv), pid=int(p), tid=int(tid), sid=int(sid))


def is_verified_shiny(status_dict):
    return (status_dict or {}).get("status") == STATUS_VERIFIED_SHINY
