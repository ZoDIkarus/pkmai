"""Loader for the Generation III battle-mechanics database.

The JSON tables under ``src/pokedb/`` are produced once by
``tools/extract_gen3_data.py`` directly from the local FireRed ROM (BPRD rev 0,
md5 6648a0484a56097ca75d6af87ebce225). No ROM bytes are stored, only derived
mechanics numbers, and the source offset of every table is recorded in each
file's ``_meta`` block.

Every lookup is **fail-closed**: an unknown species / move id returns ``None``,
never a guessed value. Callers must treat ``None`` conservatively.
"""
from __future__ import annotations

import json
import os
import threading

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pokedb")
_SPECIES_PATH = os.path.join(_DIR, "species_gen3.json")
_MOVES_PATH = os.path.join(_DIR, "moves_gen3.json")

_lock = threading.Lock()
_species = None
_moves = None
_meta = None


def _load():
    global _species, _moves, _meta
    if _species is not None and _moves is not None:
        return
    with _lock:
        if _species is not None and _moves is not None:
            return
        try:
            with open(_SPECIES_PATH) as f:
                sd = json.load(f)
            with open(_MOVES_PATH) as f:
                md = json.load(f)
        except (OSError, ValueError):
            _species, _moves, _meta = {}, {}, {"available": False}
            return
        _species = {int(k): v for k, v in (sd.get("species") or {}).items()}
        _moves = {int(k): v for k, v in (md.get("moves") or {}).items()}
        _meta = {
            "available": bool(_species) and bool(_moves),
            "species_meta": sd.get("_meta", {}),
            "moves_meta": md.get("_meta", {}),
            "species_count": len(_species),
            "moves_count": len(_moves),
        }


def is_available():
    _load()
    return bool(_meta and _meta.get("available"))


def db_meta():
    _load()
    return dict(_meta or {})


def species_info(species_id):
    """Full base-stats + types record, or None."""
    _load()
    try:
        return _species.get(int(species_id))
    except (TypeError, ValueError):
        return None


def species_types(species_id):
    """List of 1 or 2 Gen-III type ids, or None if the species is unknown."""
    info = species_info(species_id)
    if not info:
        return None
    t = info.get("types")
    return list(t) if t else None


def base_stats(species_id):
    """dict with base_hp/base_attack/... or None."""
    info = species_info(species_id)
    if not info:
        return None
    return {
        "hp": info.get("base_hp"),
        "attack": info.get("base_attack"),
        "defense": info.get("base_defense"),
        "speed": info.get("base_speed"),
        "sp_attack": info.get("base_sp_attack"),
        "sp_defense": info.get("base_sp_defense"),
    }


def move_info(move_id):
    """Full move record (type/power/accuracy/pp/priority/is_status/...), or None."""
    _load()
    try:
        return _moves.get(int(move_id))
    except (TypeError, ValueError):
        return None


def move_mechanics(move_id):
    """Compact mechanics view used by the engine/controller. None if unknown."""
    mi = move_info(move_id)
    if not mi:
        return None
    return {
        "id": int(move_id),
        "type": mi.get("type"),
        "power": mi.get("power"),
        "accuracy": mi.get("accuracy"),
        "pp": mi.get("pp"),
        "priority": mi.get("priority"),
        "is_status": bool(mi.get("is_status")),
        "makes_contact": bool(mi.get("makes_contact")),
        "secondary_chance": mi.get("secondary_chance"),
    }
