"""
Fast and trusted Pokemon FireRed RAM resolver for Stable-Retro.

FireRed relocates SaveBlock1 at runtime.
SaveBlock1:
  +0x00 x
  +0x02 y
  +0x04 mapGroup
  +0x05 mapNum

The resolver only accepts a real SaveBlock pointer signature.
It deliberately does NOT return arbitrary EWRAM coordinate scans as valid,
because those can stay static and break live mapping.
"""

EWRAM_BUS_BASE = 0x02000000
IWRAM_BUS_BASE = 0x03000000
EWRAM_SIZE = 0x40000

GSAVEBLOCK1PTR_BUS = 0x03005008

# Common Stable-Retro memory layouts.
PTR_SLOT_COMBINED = EWRAM_SIZE + (GSAVEBLOCK1PTR_BUS - IWRAM_BUS_BASE)  # 0x45008
PTR_SLOT_IWRAM_ONLY = GSAVEBLOCK1PTR_BUS - IWRAM_BUS_BASE              # 0x5008

_cached_ptr_slot = None


def _u8(ram, off):
    if off < 0 or off >= len(ram):
        return 0
    return int(ram[off]) & 0xFF


def _u16(ram, off):
    return _u8(ram, off) | (_u8(ram, off + 1) << 8)


def _u32(ram, off):
    return (
        _u8(ram, off)
        | (_u8(ram, off + 1) << 8)
        | (_u8(ram, off + 2) << 16)
        | (_u8(ram, off + 3) << 24)
    )


def _ewram_to_offset(ptr):
    if EWRAM_BUS_BASE <= ptr < EWRAM_BUS_BASE + EWRAM_SIZE:
        return ptr - EWRAM_BUS_BASE
    return None


def _valid_location(ram, base):
    if base is None or base < 0 or base + 6 > len(ram):
        return False

    x = _u16(ram, base + 0)
    y = _u16(ram, base + 2)
    bank = _u8(ram, base + 4)
    map_id = _u8(ram, base + 5)

    # Reject blank/uninitialised SaveBlock.
    if x == 0 and y == 0 and bank == 0 and map_id == 0:
        return False

    return x < 512 and y < 512 and bank < 128 and map_id < 128


def _valid_pointer_slot(ram, slot):
    if slot < 0 or slot + 12 > len(ram):
        return False

    # FireRed keeps 3 consecutive global save pointers.
    p1 = _u32(ram, slot)
    p2 = _u32(ram, slot + 4)
    p3 = _u32(ram, slot + 8)

    if len({p1, p2, p3}) != 3:
        return False

    b1 = _ewram_to_offset(p1)
    b2 = _ewram_to_offset(p2)
    b3 = _ewram_to_offset(p3)

    if b1 is None or b2 is None or b3 is None:
        return False

    return _valid_location(ram, b1)


def _find_pointer_slot(ram):
    # Fast known locations first.
    for slot in (PTR_SLOT_COMBINED, PTR_SLOT_IWRAM_ONLY):
        if _valid_pointer_slot(ram, slot):
            return slot

    # One-time aligned scan for the 3-pointer signature.
    # Much safer than scanning for arbitrary plausible x/y bytes.
    for slot in range(0, max(0, len(ram) - 12), 4):
        if _valid_pointer_slot(ram, slot):
            return slot

    return None


def read_player_location(env, allow_scan=True):
    global _cached_ptr_slot

    try:
        ram = env.get_ram()
    except Exception:
        ram = None

    if ram is None:
        return {
            "valid": False, "trusted": False, "source": "no_ram",
            "ram_size": 0, "ptr_slot": None,
            "saveblock_ptr": 0, "saveblock_array": None,
            "map_bank": 0, "map_id": 0, "x_pos": 0, "y_pos": 0,
        }

    # Revalidate cached slot; FireRed may relocate SaveBlock itself,
    # but the global pointer variable stays at the same slot.
    if _cached_ptr_slot is not None and not _valid_pointer_slot(ram, _cached_ptr_slot):
        _cached_ptr_slot = None

    if _cached_ptr_slot is None:
        # Fast known pointer slots first.
        for slot in (PTR_SLOT_COMBINED, PTR_SLOT_IWRAM_ONLY):
            if _valid_pointer_slot(ram, slot):
                _cached_ptr_slot = slot
                break

        # Expensive full-RAM discovery is allowed for the Watcher, but
        # deliberately disabled for the 30 training environments.
        if _cached_ptr_slot is None and allow_scan:
            _cached_ptr_slot = _find_pointer_slot(ram)

    if _cached_ptr_slot is None:
        return {
            "valid": False, "trusted": False, "source": "pointer_not_found",
            "ram_size": int(len(ram)), "ptr_slot": None,
            "saveblock_ptr": 0, "saveblock_array": None,
            "map_bank": 0, "map_id": 0, "x_pos": 0, "y_pos": 0,
        }

    ptr = _u32(ram, _cached_ptr_slot)
    base = _ewram_to_offset(ptr)

    if not _valid_location(ram, base):
        return {
            "valid": False, "trusted": False, "source": "pointer_invalid",
            "ram_size": int(len(ram)), "ptr_slot": int(_cached_ptr_slot),
            "saveblock_ptr": int(ptr), "saveblock_array": None,
            "map_bank": 0, "map_id": 0, "x_pos": 0, "y_pos": 0,
        }

    if _cached_ptr_slot == PTR_SLOT_COMBINED:
        source = "gSaveBlock1Ptr@combined"
    elif _cached_ptr_slot == PTR_SLOT_IWRAM_ONLY:
        source = "gSaveBlock1Ptr@iwram"
    else:
        source = "gSaveBlock1Ptr@discovered"

    return {
        "valid": True,
        "trusted": True,
        "source": source,
        "ram_size": int(len(ram)),
        "ptr_slot": int(_cached_ptr_slot),
        "saveblock_ptr": int(ptr),
        "saveblock_array": int(base),
        "map_bank": int(_u8(ram, base + 4)),
        "map_id": int(_u8(ram, base + 5)),
        "x_pos": int(_u16(ram, base + 0)),
        "y_pos": int(_u16(ram, base + 2)),
    }
