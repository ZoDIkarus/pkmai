#!/usr/bin/env python3
"""Interactively capture ONE battle-start scenario and APPEND it to the pool.

The battle champion gate needs several wild AND trainer scenarios at varied
level / HP / opponents (a single Route-1 wild can never certify a general
champion). This tool loads a NON-protected starting savestate in a visible
window, you walk into a wild encounter or a trainer, and once the verified
battle main menu is stable it snapshots that state and appends it to
``runtime/battle/scenarios/index.json`` (deduped by id).

Nothing protected is ever written. The master save and the Route-1 probe seed
are refused as a ``--from`` source.

    PYTHONPATH=src python tools/capture_battle_scenario.py \
        --from runtime/curriculum/checkpoints/stage_3.state.gz \
        --id route22_wild_1 --area route22
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import firered_ram as fr
from battle_state import MainBattleReader
from twoby2 import battle_ram_live as L
from twoby2 import protected_assets as pa
from twoby2.emulator_battle_driver import build_live_snapshot


class _Env:
    def __init__(self, env):
        self._env = env

    def get_ram(self):
        return self._env.get_ram()

SCEN_DIR = os.path.join(ROOT, "runtime", "battle", "scenarios")
INDEX = os.path.join(SCEN_DIR, "index.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_state_bytes(path):
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_index():
    try:
        with open(INDEX) as f:
            d = json.load(f) or {}
        return list(d.get("scenarios") or [])
    except (OSError, ValueError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="non-protected starting savestate (.state or .state.gz)")
    ap.add_argument("--id", required=True, help="unique scenario id")
    ap.add_argument("--area", required=True, help="area label, e.g. route22")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite an existing scenario with the same id")
    ap.add_argument("--grass-anchor", action="store_true",
                    help="capture a GRASS OVERWORLD anchor (spec ZIEL A): stop "
                         "on a grass tile OUT of battle; the isolated fighter's "
                         "WildEncounterHarvester then walks a ±3-tile corridor "
                         "from here to fresh wild encounters")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    if not os.path.isfile(src):
        raise SystemExit(f"--from not found: {src}")
    # never start from (or risk writing) a protected asset
    pa.assert_not_protected(src, op="read")
    src_sha = sha256(src)

    existing = _read_index()
    if any(s.get("id") == args.id for s in existing) and not args.replace:
        raise SystemExit(f"scenario id '{args.id}' already exists (use --replace)")

    import cv2
    import numpy as np
    import stable_retro as retro

    retro.data.Integrations.add_custom_path(
        os.path.join(ROOT, "local", "custom_integrations"))
    env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=None)
    win = f"PKMai - Kampfszenario aufnehmen [{args.id}]"
    try:
        env.reset()
        env.em.set_state(_load_state_bytes(src))
        for _ in range(8):
            env.em.step()

        print("\nSichtbare Einmal-Aufnahme:")
        print("1. w/a/s/d laufen, j = A, k = B.")
        print("2. In einen wilden Kampf ODER zu einem Trainer laufen.")
        print("3. Sobald KAMPF / BEUTEL / POKEMON / FLUCHT stabil steht:")
        print("   NICHTS mehr druecken - das Tool speichert und beendet sich.")
        print("q bricht ab. Nichts Geschuetztes wird angefasst.\n")

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 720, 540)
        neutral = np.zeros(len(env.buttons), dtype=np.uint8)

        def mask_for(name):
            m = neutral.copy()
            if name in env.buttons:
                m[env.buttons.index(name)] = 1
            return m

        keys = {ord("w"): mask_for("UP"), ord("s"): mask_for("DOWN"),
                ord("a"): mask_for("LEFT"), ord("d"): mask_for("RIGHT"),
                ord("j"): mask_for("A"), ord("k"): mask_for("B"),
                ord("n"): mask_for("START")}
        mbr = MainBattleReader()
        action = neutral
        hold = stable = held_key = 0
        want = ("stabiles Kampfmenue" if not args.grass_anchor
                else "OUT of battle auf Gras, dann q")
        while True:
            env.em.set_button_mask(action if hold > 0 else neutral)
            env.em.step()
            hold = max(0, hold - 1)
            ram = env.get_ram()
            live = mbr.read(ram, frames=1)
            at_menu = bool(live and L.menu_state(ram) == "main")
            stable = stable + 1 if at_menu else 0

            frame = cv2.cvtColor(env.get_screen(), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (720, 540), interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, f"w/a/s/d  j=A  k=B | warte auf {want} | q=Abbruch/Fertig",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 118), 1)
            cv2.imshow(win, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                if args.grass_anchor and not live:
                    break                    # grass anchor: q FINISHES the capture
                raise KeyboardInterrupt
            if key in keys:
                action = keys[key]
                hold = 16 if key in (ord("w"), ord("a"), ord("s"), ord("d")) else 6
            if not args.grass_anchor and stable >= 40:
                break

        env.em.set_button_mask(neutral)

        if args.grass_anchor:
            for _ in range(8):
                env.em.step()
            if mbr.read(env.get_ram(), frames=1):
                raise RuntimeError("grass anchor must be captured OUT of battle")
            loc = fr.read_player_location(_Env(env))
            if not loc.get("valid"):
                raise RuntimeError("player location not readable at the anchor")
            party = fr.read_player_party(_Env(env))
            os.makedirs(SCEN_DIR, exist_ok=True)
            state_path = os.path.join(SCEN_DIR, f"{args.id}.state.gz")
            pa.assert_write_target_ok(state_path, op="write")
            tmp = state_path + ".tmp"
            with gzip.open(tmp, "wb") as f:
                f.write(env.em.get_state())
            os.replace(tmp, state_path)
            scenario = {
                "id": args.id, "area": args.area, "battle_kind": "wild",
                "is_trainer": False, "can_escape": True,
                "harvest": True,
                "savestate_path": state_path,
                "anchor": {"map_bank": int(loc["map_bank"]), "map_id": int(loc["map_id"]),
                           "x_pos": int(loc["x_pos"]), "y_pos": int(loc["y_pos"])},
                "player_levels": [m.get("level") for m in party],
                "own_species": [m.get("species_id") for m in party],
                "objective_mode": "combat",
                "rng_seed_id": abs(hash(args.id)) % 100000,
                "source_state_sha256": src_sha,
                "scenario_sha256": sha256(state_path),
            }
            merged = [s for s in existing if s.get("id") != args.id] + [scenario]
            _atomic_json(INDEX, {"schema": "live_battle_scenarios_v1",
                                 "scenarios": merged})
            print(f"\nOK: GRASS ANCHOR '{args.id}' @ {args.area} "
                  f"({scenario['anchor']}) gespeichert. Der Harvester laeuft "
                  f"von hier +-3 Kacheln.")
            print(state_path)
            print(f"Pool: {len(merged)} Szenarien.")
            return

        snap = build_live_snapshot(env, main_battle_reader=mbr)
        if snap.get("in_battle") is not True or snap.get("menu_state") != "main":
            raise RuntimeError("capture lost the verified battle main menu")
        is_trainer = bool(snap.get("is_trainer"))
        can_escape = bool(snap.get("can_escape"))
        if is_trainer and can_escape:
            raise RuntimeError("inconsistent snapshot: trainer battle marked escapable")

        os.makedirs(SCEN_DIR, exist_ok=True)
        state_path = os.path.join(SCEN_DIR, f"{args.id}.state.gz")
        pa.assert_write_target_ok(state_path, op="write")
        tmp = state_path + ".tmp"
        with gzip.open(tmp, "wb") as f:
            f.write(env.em.get_state())
        os.replace(tmp, state_path)

        enemy = snap.get("enemy_active") or {}
        party = fr.read_player_party(_Env(env))
        from twoby2.scenario_pool import scenario_bucket
        scenario = {
            "id": args.id,
            "area": args.area,
            "battle_kind": "trainer" if is_trainer else "wild",
            "is_trainer": is_trainer,
            "can_escape": can_escape,
            "savestate_path": state_path,
            "source_state_sha256": src_sha,
            "scenario_sha256": sha256(state_path),
            # spec ZIEL B metadata
            "enemy_species": enemy.get("species_id"), "enemy_level": enemy.get("level"),
            "enemy_hp": enemy.get("cur_hp"), "enemy_max_hp": enemy.get("max_hp"),
            "player_levels": [m.get("level") for m in party],
            "own_species": [m.get("species_id") for m in party],
            "objective_mode": "combat",
            "bucket": scenario_bucket(
                area=args.area, enemy_species=[enemy.get("species_id")],
                enemy_level=enemy.get("level"),
                player_levels=[m.get("level") for m in party],
                own_species=[m.get("species_id") for m in party]),
            "trained_count": 0,
        }
        merged = [s for s in existing if s.get("id") != args.id] + [scenario]
        _atomic_json(INDEX, {"schema": "live_battle_scenarios_v1",
                             "scenarios": merged})
        if sha256(src) != src_sha:
            raise RuntimeError("starting savestate changed during capture")
        print(f"\nOK: {scenario['battle_kind']} scenario '{args.id}' @ {args.area} "
              f"(can_escape={can_escape}) gespeichert.")
        print(state_path)
        print(f"Pool: {len(merged)} Szenarien.")
    except KeyboardInterrupt:
        print("\nAufnahme abgebrochen; nichts geschrieben.")
        raise SystemExit(130)
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
