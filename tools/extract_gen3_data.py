#!/usr/bin/env python3
"""Reproducible extraction of the Generation III battle-mechanics database
from the local Pokémon FireRed ROM.

The ROM is NOT distributed. This script only reads it and writes plain JSON
data tables under ``src/pokedb/`` so the battle system does not depend on the
ROM at runtime and the source of every number is auditable.

It self-verifies: the base-stats and moves tables are located by searching for
a well-known signature (Bulbasaur's base stats / Pound's move data) rather than
by a hard-coded offset, and the result is cross-checked against a handful of
famous values. If a check fails the script aborts and writes nothing.

Usage (read-only, run once):
    python tools/extract_gen3_data.py --rom local/custom_integrations/PokemonFireRed-Gba/rom.gba

Nothing here is imported by the trainer or the env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

# BPRD (German FireRed) rev 0 - the ROM this project ships its integration for.
EXPECTED_MD5 = "6648a0484a56097ca75d6af87ebce225"
GAMECODE = b"BPRD"

BASE_STATS_STRUCT = 28
MOVE_STRUCT = 12
NUM_SPECIES_GEN3 = 412      # 1..411 valid national+hoenn internal in Gen III FR
NUM_MOVES_GEN3 = 355        # 0..354

TYPE_NAMES = [
    "Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
    "Ghost", "Steel", "???", "Fire", "Water", "Grass", "Electric", "Psychic",
    "Ice", "Dragon", "Dark",
]

# --- signatures -------------------------------------------------------------
# Bulbasaur base stats: HP45 Atk49 Def49 Spd45 SpA65 SpD65 type1=Grass(12) type2=Poison(3)
BULBASAUR_SIG = bytes([45, 49, 49, 45, 65, 65, 12, 3])
# Species index 1 = Bulbasaur; index 0 is a dummy entry -> table start = hit-28.

# Pound (move 1): effect 0, power 40, type Normal(0), accuracy 100, pp 35
POUND_SIG = bytes([0, 40, 0, 100, 35])


def _find_all(hay, needle, align=1):
    i = hay.find(needle)
    while i != -1:
        if i % align == 0:
            yield i
        i = hay.find(needle, i + 1)


def locate_base_stats(rom):
    for hit in _find_all(rom, BULBASAUR_SIG):
        start = hit - BASE_STATS_STRUCT      # back up over the dummy entry
        if start < 0:
            continue
        # sanity: Ivysaur (index 2) HP60 Atk62 Def63 Spd60 SpA80 SpD80
        ivy = rom[start + 2 * BASE_STATS_STRUCT: start + 2 * BASE_STATS_STRUCT + 6]
        if list(ivy) == [60, 62, 63, 60, 80, 80]:
            return start
    return None


def locate_moves(rom):
    for hit in _find_all(rom, POUND_SIG):
        start = hit - MOVE_STRUCT             # back up over move 0 (None)
        if start < 0:
            continue
        # Karate Chop (move 2): power 50, type Fighting(1), acc 100, pp 25
        kc = rom[start + 2 * MOVE_STRUCT + 1: start + 2 * MOVE_STRUCT + 5]
        if list(kc) == [50, 1, 100, 25]:
            return start
    return None


def parse_base_stats(rom, table):
    out = {}
    for sid in range(1, NUM_SPECIES_GEN3):
        off = table + sid * BASE_STATS_STRUCT
        if off + BASE_STATS_STRUCT > len(rom):
            break
        b = rom[off:off + BASE_STATS_STRUCT]
        hp, atk, dfn, spd, spa, spd_ = b[0], b[1], b[2], b[3], b[4], b[5]
        t1, t2 = b[6], b[7]
        catch, expyield = b[8], b[9]
        item1 = struct.unpack_from("<H", b, 12)[0]
        item2 = struct.unpack_from("<H", b, 14)[0]
        ability1, ability2 = b[22], b[23]
        if not (1 <= hp <= 255) or t1 > 17 or t2 > 17:
            continue
        types = [t1] if t1 == t2 else [t1, t2]
        out[str(sid)] = {
            "base_hp": hp, "base_attack": atk, "base_defense": dfn,
            "base_speed": spd, "base_sp_attack": spa, "base_sp_defense": spd_,
            "types": types,
            "type_names": [TYPE_NAMES[t] for t in types],
            "catch_rate": catch, "base_exp_yield": expyield,
            "held_item1": item1, "held_item2": item2,
            "ability1": ability1, "ability2": ability2,
        }
    return out


def parse_moves(rom, table):
    out = {}
    for mid in range(0, NUM_MOVES_GEN3):
        off = table + mid * MOVE_STRUCT
        if off + MOVE_STRUCT > len(rom):
            break
        b = rom[off:off + MOVE_STRUCT]
        effect, power, mtype, accuracy, pp = b[0], b[1], b[2], b[3], b[4]
        secondary, target = b[5], b[6]
        priority = struct.unpack_from("<b", b, 7)[0]
        flags = b[8]
        if mtype > 17:
            continue
        is_status = power == 0
        out[str(mid)] = {
            "effect": effect, "power": power, "type": mtype,
            "type_name": TYPE_NAMES[mtype],
            "accuracy": accuracy, "pp": pp,
            "secondary_chance": secondary, "target": target,
            "priority": priority, "flags": flags,
            "is_status": is_status,
            "makes_contact": bool(flags & 0x1),
        }
    return out


def cross_check(species, moves):
    problems = []
    def s(i): return species.get(str(i), {})
    def m(i): return moves.get(str(i), {})
    # Gen-III (FR) canon - deliberately NOT the modern re-balanced numbers.
    checks = [
        (s(6).get("types") == [10, 2], "Charizard = Fire/Flying"),
        (s(9).get("types") == [11], "Blastoise = Water (mono)"),
        (s(4).get("types") == [10], "Charmander = Fire (mono)"),
        (s(25).get("base_speed") == 90, "Pikachu base speed 90"),
        (s(150).get("base_sp_attack") == 154, "Mewtwo base SpAtk 154"),
        (m(55).get("type") == 11 and m(55).get("power") == 40, "Water Gun 40 BP Water"),
        (m(56).get("type") == 11 and m(56).get("power") == 120
         and m(56).get("accuracy") == 80, "Hydro Pump 120 BP / 80% (Gen III)"),
        (m(57).get("type") == 11 and m(57).get("power") == 95, "Surf 95 BP Water"),
        (m(53).get("type") == 10 and m(53).get("power") == 95, "Flamethrower 95 BP Fire"),
        (m(98).get("priority") == 1, "Quick Attack priority +1"),
        (m(45).get("is_status") is True, "Growl is a status move"),
        (m(33).get("power") == 35, "Tackle 35 BP (Gen III)"),
    ]
    for ok, label in checks:
        if not ok:
            problems.append(label)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="local/custom_integrations/PokemonFireRed-Gba/rom.gba")
    ap.add_argument("--out", default="src/pokedb")
    ap.add_argument("--force-md5", action="store_true",
                    help="proceed even if the ROM md5 does not match BPRD rev0")
    args = ap.parse_args()

    if not os.path.exists(args.rom):
        sys.exit(f"ROM not found: {args.rom}")
    rom = open(args.rom, "rb").read()
    md5 = hashlib.md5(rom).hexdigest()
    code = rom[0xAC:0xB0]
    print(f"ROM: {len(rom)} bytes  gamecode={code!r}  md5={md5}")
    if code != GAMECODE:
        sys.exit(f"unexpected gamecode {code!r} (need {GAMECODE!r})")
    if md5 != EXPECTED_MD5 and not args.force_md5:
        sys.exit(f"unexpected md5 {md5} (need {EXPECTED_MD5}); use --force-md5 to override")

    bs = locate_base_stats(rom)
    mv = locate_moves(rom)
    if bs is None:
        sys.exit("could not locate the base-stats table by signature")
    if mv is None:
        sys.exit("could not locate the moves table by signature")
    print(f"base-stats table @ 0x{bs:06X}   moves table @ 0x{mv:06X}")

    species = parse_base_stats(rom, bs)
    moves = parse_moves(rom, mv)
    problems = cross_check(species, moves)
    if problems:
        sys.exit("cross-check FAILED:\n  - " + "\n  - ".join(problems))
    print(f"cross-check OK  ({len(species)} species, {len(moves)} moves)")

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "schema": "gen3_battle_db_v1",
        "generation": 3,
        "source_rom_gamecode": "BPRD",
        "source_rom_md5": md5,
        "base_stats_rom_offset": f"0x{bs:06X}",
        "moves_rom_offset": f"0x{mv:06X}",
        "type_names": TYPE_NAMES,
        "note": "Reproducible extraction via tools/extract_gen3_data.py; "
                "no ROM bytes are stored, only derived mechanics tables.",
    }
    _write(os.path.join(args.out, "species_gen3.json"),
           {"_meta": meta, "species": species})
    _write(os.path.join(args.out, "moves_gen3.json"),
           {"_meta": meta, "moves": moves})
    print("wrote", args.out + "/species_gen3.json", "and moves_gen3.json")


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
