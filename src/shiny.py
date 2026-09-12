"""Fail-closed pure Gen-III FireRed shininess calculations."""

SHINY_THRESHOLD = 8


def _u16(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value & 0xFFFF if 0 <= value <= 0xFFFFFFFF else None


def shiny_value(trainer_id, secret_id, pid):
    """Return the Gen-III shiny value, or None for invalid input."""
    tid = _u16(trainer_id)
    sid = _u16(secret_id)
    try:
        personality = int(pid)
    except (TypeError, ValueError):
        return None
    if tid is None or sid is None or not 0 <= personality <= 0xFFFFFFFF:
        return None
    return tid ^ sid ^ (personality & 0xFFFF) ^ ((personality >> 16) & 0xFFFF)


def is_shiny(trainer_id, secret_id, pid):
    """Return True/False, or None when the source values are unavailable."""
    value = shiny_value(trainer_id, secret_id, pid)
    return None if value is None else value < SHINY_THRESHOLD


def split_trainer_id(player_trainer_id_u32):
    """Split packed FireRed player id into (trainer id, secret id)."""
    try:
        value = int(player_trainer_id_u32)
    except (TypeError, ValueError):
        return None, None
    if not 0 <= value <= 0xFFFFFFFF:
        return None, None
    return value & 0xFFFF, (value >> 16) & 0xFFFF
