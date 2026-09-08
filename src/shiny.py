"""Gen-III (FireRed / BPRD) shininess — pure maths, no emulator, no RAM.

A Pokemon is shiny when

    shiny_value = (trainer_id XOR secret_id XOR pid_high16 XOR pid_low16) < 8

All four inputs are 16-bit. ``trainer_id`` / ``secret_id`` are the player's
(low / high halves of the 32-bit ``playerTrainerId``); ``pid`` is the wild
Pokemon's 32-bit personality value.

Every function returns ``None`` when an input is missing or out of range — a
shiny check is NEVER guessed. The RAM side (:mod:`twoby2.shiny_ram`) stays
fail-closed until the PID / TID / SID addresses are verified for the BPRD ROM.
"""
from __future__ import annotations

SHINY_THRESHOLD = 8


def _u16(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v & 0xFFFF if 0 <= v <= 0xFFFFFFFF else None


def shiny_value(trainer_id, secret_id, pid):
    """The Gen-III shiny value (0..65535), or ``None`` if any input is
    missing / invalid. ``< SHINY_THRESHOLD`` means shiny."""
    tid = _u16(trainer_id)
    sid = _u16(secret_id)
    try:
        p = int(pid)
    except (TypeError, ValueError):
        return None
    if tid is None or sid is None or not (0 <= p <= 0xFFFFFFFF):
        return None
    pid_low = p & 0xFFFF
    pid_high = (p >> 16) & 0xFFFF
    return tid ^ sid ^ pid_high ^ pid_low


def is_shiny(trainer_id, secret_id, pid):
    """``True`` / ``False`` / ``None`` (inputs missing). Never guesses."""
    sv = shiny_value(trainer_id, secret_id, pid)
    if sv is None:
        return None
    return sv < SHINY_THRESHOLD


def split_trainer_id(player_trainer_id_u32):
    """``(trainer_id, secret_id)`` from the packed 32-bit player id, or
    ``(None, None)``. TID = low 16 bits, SID = high 16 bits."""
    try:
        v = int(player_trainer_id_u32)
    except (TypeError, ValueError):
        return None, None
    if not (0 <= v <= 0xFFFFFFFF):
        return None, None
    return v & 0xFFFF, (v >> 16) & 0xFFFF
