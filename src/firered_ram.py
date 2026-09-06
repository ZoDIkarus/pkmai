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
SAVEBLOCK1_VARS_OFFSET = 0x1000
VARS_START = 0x4000

VAR_PALLET_OAKS_LAB = 0x4055
VAR_VIRIDIAN_OLD_MAN = 0x4051
VAR_VIRIDIAN_MART = 0x4057

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


def _save_var(ram, base, var_id):
    """Read a FireRed u16 SaveBlock1 variable by its 0x4xxx id."""
    idx = int(var_id) - VARS_START
    if base is None or idx < 0:
        return 0
    return _u16(ram, base + SAVEBLOCK1_VARS_OFFSET + idx * 2)


def _scene_var(ram, base, var_id):
    """Reject implausible values if a language build/layout ever differs."""
    value = _save_var(ram, base, var_id)
    return value if 0 <= value <= 16 else 0


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

    # Group 0 contains only the five link rooms, never map 57.
    if bank == 0 and map_id >= 5:
        return False
    return x < 512 and y < 512 and bank < 43 and map_id < 128


def _valid_pointer_slot(ram, slot, require_location=True):
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

    return not require_location or _valid_location(ram, b1)


def _find_pointer_slot(ram):
    # Fast known locations first.
    for slot in ((PTR_SLOT_COMBINED,) if len(ram) >= EWRAM_SIZE + 0x8000
                 else (PTR_SLOT_IWRAM_ONLY,)):
        if _valid_pointer_slot(ram, slot):
            return slot

    # Global save-pointer variables live in IWRAM. Scanning EWRAM also
    # matches ordinary data and stale pointer copies after a restore.
    start = EWRAM_SIZE if len(ram) >= EWRAM_SIZE + 0x8000 else 0
    end = min(len(ram), start + 0x8000)
    for slot in range(start, max(start, end - 12), 4):
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
    if _cached_ptr_slot is not None and not _valid_pointer_slot(ram, _cached_ptr_slot, require_location=False):
        _cached_ptr_slot = None

    if _cached_ptr_slot is None:
        # Fast known pointer slots first.
        for slot in ((PTR_SLOT_COMBINED,) if len(ram) >= EWRAM_SIZE + 0x8000
                 else (PTR_SLOT_IWRAM_ONLY,)):
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
        # Source-backed early-story state. These distinguish real progress
        # from arbitrary indoor warps.
        "viridian_mart_scene": int(_scene_var(ram, base, VAR_VIRIDIAN_MART)),
        "viridian_old_man_scene": int(_scene_var(ram, base, VAR_VIRIDIAN_OLD_MAN)),
        "pallet_oaks_lab_scene": int(_scene_var(ram, base, VAR_PALLET_OAKS_LAB)),
    }


# ---------------------------------------------------------------------------
# FireRed party Pokemon decoder (Gen III, US FireRed-compatible RAM layout)
# ---------------------------------------------------------------------------
# Existing Stable-Retro integration confirms:
#   gPlayerParty      EWRAM + 0x24284
#   party[0].level    +0x54 == EWRAM + 0x242D8
#
# struct Pokemon size = 100 bytes.
PLAYER_PARTY_OFFSET = 0x24284
ENEMY_PARTY_OFFSET = 0x2402C
BATTLE_TYPE_FLAGS_OFFSET = 0x22FEC
POKEMON_STRUCT_SIZE = 100
MAX_PARTY_SIZE = 6

# Stored 48-byte encrypted substruct permutation by personality % 24.
_SUBSTRUCT_ORDERS = (
    "GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA",
    "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
    "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG",
    "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG",
)

# Kanto species names; unknown Gen III species fall back to "Species #N".
_KANTO_SPECIES = (
    None,
    "Bulbasaur","Ivysaur","Venusaur","Charmander","Charmeleon","Charizard",
    "Squirtle","Wartortle","Blastoise","Caterpie","Metapod","Butterfree",
    "Weedle","Kakuna","Beedrill","Pidgey","Pidgeotto","Pidgeot","Rattata",
    "Raticate","Spearow","Fearow","Ekans","Arbok","Pikachu","Raichu",
    "Sandshrew","Sandslash","Nidoran♀","Nidorina","Nidoqueen","Nidoran♂",
    "Nidorino","Nidoking","Clefairy","Clefable","Vulpix","Ninetales",
    "Jigglypuff","Wigglytuff","Zubat","Golbat","Oddish","Gloom","Vileplume",
    "Paras","Parasect","Venonat","Venomoth","Diglett","Dugtrio","Meowth",
    "Persian","Psyduck","Golduck","Mankey","Primeape","Growlithe","Arcanine",
    "Poliwag","Poliwhirl","Poliwrath","Abra","Kadabra","Alakazam","Machop",
    "Machoke","Machamp","Bellsprout","Weepinbell","Victreebel","Tentacool",
    "Tentacruel","Geodude","Graveler","Golem","Ponyta","Rapidash","Slowpoke",
    "Slowbro","Magnemite","Magneton","Farfetch'd","Doduo","Dodrio","Seel",
    "Dewgong","Grimer","Muk","Shellder","Cloyster","Gastly","Haunter","Gengar",
    "Onix","Drowzee","Hypno","Krabby","Kingler","Voltorb","Electrode",
    "Exeggcute","Exeggutor","Cubone","Marowak","Hitmonlee","Hitmonchan",
    "Lickitung","Koffing","Weezing","Rhyhorn","Rhydon","Chansey","Tangela",
    "Kangaskhan","Horsea","Seadra","Goldeen","Seaking","Staryu","Starmie",
    "Mr. Mime","Scyther","Jynx","Electabuzz","Magmar","Pinsir","Tauros",
    "Magikarp","Gyarados","Lapras","Ditto","Eevee","Vaporeon","Jolteon",
    "Flareon","Porygon","Omanyte","Omastar","Kabuto","Kabutops","Aerodactyl",
    "Snorlax","Articuno","Zapdos","Moltres","Dratini","Dragonair","Dragonite",
    "Mewtwo","Mew",
)

# Common early-game / Squirtle-family moves. Unknown moves remain readable by ID.
_MOVE_NAMES = {
    10:"Scratch", 22:"Vine Whip", 33:"Tackle", 36:"Take Down",
    39:"Tail Whip", 43:"Leer", 44:"Bite", 45:"Growl",
    52:"Ember", 55:"Water Gun", 56:"Hydro Pump", 73:"Leech Seed",
    99:"Rage", 110:"Withdraw", 130:"Skull Bash", 145:"Bubble",
    182:"Protect", 205:"Rollout", 229:"Rapid Spin", 240:"Rain Dance",
}

def _u16_bytes(buf, off):
    return int(buf[off]) | (int(buf[off + 1]) << 8)

def _u32_bytes(buf, off):
    return (
        int(buf[off])
        | (int(buf[off + 1]) << 8)
        | (int(buf[off + 2]) << 16)
        | (int(buf[off + 3]) << 24)
    )

def _species_name(species_id):
    if 0 < species_id < len(_KANTO_SPECIES):
        return _KANTO_SPECIES[species_id]
    return f"Species #{species_id}"

def _move_name(move_id):
    if move_id <= 0:
        return None
    return _MOVE_NAMES.get(move_id, f"Move #{move_id}")

def _decrypt_box_data(ram, base):
    personality = _u32(ram, base + 0)
    ot_id = _u32(ram, base + 4)
    key = personality ^ ot_id

    encrypted = bytearray(48)
    for i in range(0, 48, 4):
        word = _u32(ram, base + 32 + i) ^ key
        encrypted[i + 0] = word & 0xFF
        encrypted[i + 1] = (word >> 8) & 0xFF
        encrypted[i + 2] = (word >> 16) & 0xFF
        encrypted[i + 3] = (word >> 24) & 0xFF

    order = _SUBSTRUCT_ORDERS[personality % 24]
    chunks = {}
    for stored_idx, label in enumerate(order):
        start = stored_idx * 12
        chunks[label] = encrypted[start:start + 12]

    return personality, ot_id, chunks

def _valid_pokemon_checksum(ram, base, chunks):
    expected = _u16(ram, base + 28)
    total = 0
    # Checksum is sum of all 24 decrypted uint16s.
    raw = b"".join(chunks[label] for label in "GAEM")
    for i in range(0, 48, 2):
        total = (total + _u16_bytes(raw, i)) & 0xFFFF
    return total == expected

def _decode_party_mon(ram, base, slot):
    if base < 0 or base + POKEMON_STRUCT_SIZE > len(ram):
        return None

    level = _u8(ram, base + 84)
    cur_hp = _u16(ram, base + 86)
    max_hp = _u16(ram, base + 88)

    # Empty party slots are effectively zeroed / invalid.
    if not (1 <= level <= 100):
        return None

    try:
        personality, ot_id, chunks = _decrypt_box_data(ram, base)
    except Exception:
        return None

    growth = chunks.get("G")
    attacks = chunks.get("A")
    if growth is None or attacks is None:
        return None

    species_id = _u16_bytes(growth, 0)
    if not (1 <= species_id <= 440):
        return None

    # Gen III Growth substruct:
    # +0 species, +2 held item, +4..7 total experience.
    experience = _u32_bytes(growth, 4)

    move_ids = [
        _u16_bytes(attacks, 0),
        _u16_bytes(attacks, 2),
        _u16_bytes(attacks, 4),
        _u16_bytes(attacks, 6),
    ]
    pp = [
        int(attacks[8]),
        int(attacks[9]),
        int(attacks[10]),
        int(attacks[11]),
    ]

    moves = []
    for move_id, current_pp in zip(move_ids, pp):
        if move_id <= 0:
            continue
        moves.append({
            "id": int(move_id),
            "name": _move_name(int(move_id)),
            "pp": int(current_pp),
        })

    return {
        "slot": int(slot),
        "id": int(species_id),
        "species_id": int(species_id),
        "name": _species_name(int(species_id)),
        "level": int(level),
        "experience": int(experience),
        "cur_hp": int(cur_hp),
        "max_hp": int(max_hp),
        "hp_ratio": round(cur_hp / max_hp, 4) if max_hp else 0.0,
        "fainted": bool(cur_hp == 0 and max_hp > 0),
        "status": int(_u32(ram, base + 80)),
        "stats": {
            "attack": int(_u16(ram, base + 90)),
            "defense": int(_u16(ram, base + 92)),
            "speed": int(_u16(ram, base + 94)),
            "sp_attack": int(_u16(ram, base + 96)),
            "sp_defense": int(_u16(ram, base + 98)),
        },
        "moves": moves,
        "checksum_ok": bool(_valid_pokemon_checksum(ram, base, chunks)),
        "personality": int(personality),
        "ot_id": int(ot_id),
    }

def read_player_party(env):
    """
    Read the six FireRed party structs directly from EWRAM.

    This is telemetry only. It does not alter reward or PPO observations.
    Invalid/empty slots are skipped rather than guessed.
    """
    try:
        ram = env.get_ram()
    except Exception:
        return []

    if ram is None or len(ram) < PLAYER_PARTY_OFFSET + POKEMON_STRUCT_SIZE:
        return []

    party = []
    for slot in range(MAX_PARTY_SIZE):
        base = PLAYER_PARTY_OFFSET + slot * POKEMON_STRUCT_SIZE
        mon = _decode_party_mon(ram, base, slot)
        if mon is None or not mon.get("checksum_ok", False):
            continue
        cur_hp = int(mon.get("cur_hp", 0))
        max_hp = int(mon.get("max_hp", 0))
        level = int(mon.get("level", 0))
        # Uninitialisierte/verschobene RAM-Daten koennen zufaellig wie eine
        # bekannte Species aussehen (haeufig #1/Bisasam). Nur strukturell
        # plausible, per Gen-III-Checksum bestaetigte Eintraege anzeigen.
        if 1 <= level <= 100 and 0 < max_hp <= 999 and 0 <= cur_hp <= max_hp:
            party.append(mon)
    return party


def read_enemy_party(env):
    """Read FireRed opponent/wild party from gEnemyParty."""
    try:
        ram = env.get_ram()
    except Exception:
        return []

    if ram is None or len(ram) < ENEMY_PARTY_OFFSET + POKEMON_STRUCT_SIZE:
        return []

    party = []
    for slot in range(MAX_PARTY_SIZE):
        base = ENEMY_PARTY_OFFSET + slot * POKEMON_STRUCT_SIZE
        mon = _decode_party_mon(ram, base, slot)
        if mon is None or not mon.get("checksum_ok", False):
            continue
        cur_hp = int(mon.get("cur_hp", 0))
        max_hp = int(mon.get("max_hp", 0))
        if 0 < max_hp <= 999 and 0 <= cur_hp <= max_hp:
            party.append(mon)
    return party


def read_battle_type_flags(env):
    """Read the confirmed FireRed ``gBattleTypeFlags`` uint32 directly."""
    try:
        ram = env.get_ram()
    except Exception:
        return 0
    if ram is None or len(ram) < BATTLE_TYPE_FLAGS_OFFSET + 4:
        return 0
    return _u32(ram, BATTLE_TYPE_FLAGS_OFFSET)
