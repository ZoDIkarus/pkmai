#!/usr/bin/env python3
"""Interactively capture one unprotected Route-1 battle-start scenario.

The protected Route-1 probe seed is only read and hash-checked. The user walks
to a wild encounter in a visible emulator window. Once the verified battle
main menu is stable, this tool writes a separate training savestate and exits.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_state import MainBattleReader
from twoby2 import battle_ram_live as L
from twoby2 import protected_assets as pa
from twoby2.emulator_battle_driver import build_live_snapshot


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    import cv2
    import numpy as np
    import stable_retro as retro

    src = os.path.join(ROOT, pa.RAM_PROBE_ROUTE1_SEED["state"])
    expected = pa.RAM_PROBE_ROUTE1_SEED["state_sha256"]
    before = sha256(src)
    if before != expected:
        raise RuntimeError("protected Route-1 probe seed hash mismatch")

    retro.data.Integrations.add_custom_path(
        os.path.join(ROOT, "local", "custom_integrations"))
    env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY,
                     render_mode=None)
    win = "PKMai - Route 1 Battle Scenario"
    try:
        env.reset()
        with gzip.open(src, "rb") as f:
            env.em.set_state(f.read())
        for _ in range(8):
            env.em.step()

        print("\nSichtbare Einmal-Aufnahme:")
        print("1. Mit w/a/s/d ins hohe Gras laufen.")
        print("2. Kampftexte mit j (A) bestaetigen.")
        print("3. Sobald KAMPF / BEUTEL / POKEMON / FLUCHT sichtbar ist:")
        print("   NICHTS mehr druecken. Das Tool speichert automatisch und beendet sich.")
        print("q bricht ab; der geschuetzte Save wird niemals beschrieben.\n")

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 720, 540)
        neutral = np.zeros(len(env.buttons), dtype=np.uint8)

        def mask_for(name):
            mask = neutral.copy()
            if name in env.buttons:
                mask[env.buttons.index(name)] = 1
            return mask

        keys = {ord("w"): mask_for("UP"), ord("s"): mask_for("DOWN"),
                ord("a"): mask_for("LEFT"), ord("d"): mask_for("RIGHT"),
                ord("j"): mask_for("A"), ord("k"): mask_for("B"),
                ord("n"): mask_for("START")}
        mbr = MainBattleReader()
        action = neutral
        hold = 0
        stable = 0
        while True:
            env.em.set_button_mask(action if hold > 0 else neutral)
            env.em.step()
            hold = max(0, hold - 1)
            ram = env.get_ram()
            live = mbr.read(ram, frames=1)
            flags = int.from_bytes(bytes(ram[0x22B4C:0x22B50]), "little")
            wild_main = bool(live and not (flags & 0x0008)
                             and L.menu_state(ram) == "main")
            stable = stable + 1 if wild_main else 0

            frame = cv2.cvtColor(env.get_screen(), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (720, 540), interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, "w/a/s/d  j=A  k=B | warte auf wildes Kampfmenue | q=Abbruch",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 118), 1)
            cv2.imshow(win, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key in keys:
                action = keys[key]
                hold = 16 if key in (ord("w"), ord("a"), ord("s"), ord("d")) else 6
            if stable >= 40:
                break

        env.em.set_button_mask(neutral)
        snap = build_live_snapshot(env, main_battle_reader=mbr)
        if (snap.get("in_battle") is not True or snap.get("is_trainer")
                or snap.get("menu_state") != "main"):
            raise RuntimeError("capture lost the verified wild battle main menu")

        outdir = os.path.join(ROOT, "runtime", "battle", "scenarios")
        os.makedirs(outdir, exist_ok=True)
        state_path = os.path.join(outdir, "route1_wild_probe.state.gz")
        pa.assert_write_target_ok(state_path, op="write")
        tmp = state_path + ".tmp"
        with gzip.open(tmp, "wb") as f:
            f.write(env.em.get_state())
        os.replace(tmp, state_path)
        scenario = {
            "id": "route1_wild_probe",
            "area": "route1",
            "battle_kind": "wild",
            "is_trainer": False,
            "can_escape": True,
            "savestate_path": state_path,
            "source_seed_sha256": before,
            "scenario_sha256": sha256(state_path),
        }
        index_path = os.path.join(outdir, "index.json")
        _atomic_json(index_path, {"schema": "live_battle_scenarios_v1",
                                  "scenarios": [scenario]})
        if sha256(src) != before:
            raise RuntimeError("protected source seed changed")
        print("\nOK: Route-1-Kampfszenario gespeichert:")
        print(state_path)
        print("Der geschuetzte Ausgangs-Save ist byte-identisch geblieben.")
    except KeyboardInterrupt:
        print("\nAufnahme abgebrochen; nichts aktiviert.")
        raise SystemExit(130)
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
