#!/usr/bin/env python3
"""Interactive REAL battle canary for the 2x2 EmulatorBattleDriver.

Isolated emulator, read-only working copy of the registered Route-1 probe seed.
YOU walk into grass / into a trainer. The tool detects the battle itself via the
verified stable in-battle latch and then hands control to the REAL
``EmulatorBattleDriver`` — there is NO fake driver in this test.

    PYTHONPATH=src python tools/live_battle_canary.py [--phase wild|trainer] [--resume]

Report: runtime/live_battle_canary/<timestamp>/canary_report.json
Resumable: partial results are saved after each phase in the same directory.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2 import protected_assets as pa                 # noqa: E402
from twoby2 import battle_ram_live as L                   # noqa: E402
from twoby2 import ram_battle_probe as rp                 # noqa: E402
from twoby2.emulator_battle_driver import (               # noqa: E402
    EmulatorBattleDriver, build_live_snapshot)
from battle_state import MainBattleReader                 # noqa: E402
from battle_executor import ALL_MACROS                    # noqa: E402

SEED_REL = pa.RAM_PROBE_ROUTE1_SEED["state"]
SEED_SHA = pa.RAM_PROBE_ROUTE1_SEED["state_sha256"]
ROM_PATH = os.path.join(ROOT, "local", "custom_integrations",
                        "PokemonFireRed-Gba", "rom.gba")
CANARY_ROOT = os.path.join(ROOT, "runtime", "live_battle_canary")

_STABLE_FRAMES = 40          # consecutive frames the latch must agree
_TURN_SETTLE_FRAMES = 480


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _make_env():
    import stable_retro as retro
    retro.data.Integrations.add_custom_path(os.path.join(ROOT, "local", "custom_integrations"))
    env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
    env.reset()
    return env


def _load_working_copy(env, workdir):
    src = os.path.join(ROOT, SEED_REL)
    before = _sha256(src)
    work = os.path.join(workdir, "working_seed.state.gz")
    with open(src, "rb") as fi, open(work, "wb") as fo:
        fo.write(fi.read())
    with gzip.open(work, "rb") as f:
        raw = f.read()
    env.em.set_state(raw)
    for _ in range(8):
        env.em.step()
    after = _sha256(src)
    return {"seed_rel": SEED_REL, "seed_sha_before": before,
            "seed_sha_after": after,
            "seed_unchanged": before == after == SEED_SHA,
            "working_copy": os.path.relpath(work, ROOT)}


class _Emu:
    """Small env-shaped adapter EmulatorBattleDriver / build_live_snapshot use."""
    def __init__(self, env):
        self.env = env
        self.em = env.em
        self.buttons = list(env.buttons)

    def get_ram(self):
        return self.env.get_ram()


def _wait_for_battle(env, mbr, *, prompt, expected_phase, cv2=None, np=None):
    """Block while the user plays; return once the in-battle latch is stably
    True. Draws the emulator + a HUD if cv2 is available."""
    print(prompt)
    stable = 0
    win = "live_battle_canary"
    if cv2 is not None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 720, 540)
    n = len(env.buttons)
    keymap = {}
    if cv2 is not None:
        def mk(*names):
            m = [0] * n
            for x in names:
                if x in env.buttons:
                    m[env.buttons.index(x)] = 1
            return np.array(m, dtype=np.uint8)
        keymap = {ord("w"): mk("UP"), ord("s"): mk("DOWN"), ord("a"): mk("LEFT"),
                  ord("d"): mk("RIGHT"), ord("j"): mk("A"), ord("k"): mk("B"),
                  ord("n"): mk("START")}
        neutral = np.zeros(n, dtype=np.uint8)
    hold, action = 0, None
    wrong_kind_announced = False
    while True:
        if cv2 is not None:
            env.em.set_button_mask(action if (hold > 0 and action is not None) else neutral)
        env.em.step()
        hold = max(0, hold - 1)
        ram = env.get_ram()
        # This loop advances exactly one emulator frame per sample.  Passing
        # MainBattleReader's historical default (14) prevents gMain discovery.
        latch = mbr.read(ram, frames=1)
        flags = int.from_bytes(bytes(ram[0x22B4C:0x22B50]), "little")
        is_trainer = bool(flags & 0x0008)
        expected_kind = is_trainer if expected_phase == "trainer" else not is_trainer
        menu_ready = L.menu_state(ram) == "main"
        if latch and not expected_kind:
            stable = 0
            if not wrong_kind_announced:
                print("\n--- Falsche Kampfart fuer diese Phase: bitte manuell "
                      "beenden/fliehen und danach weiterlaufen. ---\n")
                wrong_kind_announced = True
        elif latch and menu_ready:
            stable += 1
            if stable >= _STABLE_FRAMES:
                print("\n*** Kampf erkannt - jetzt KEINE Tasten mehr druecken. ***\n")
                if cv2 is not None:
                    for _ in range(30):
                        env.em.set_button_mask(neutral); env.em.step()
                return True
        else:
            stable = 0
            if not latch:
                wrong_kind_announced = False
        if cv2 is not None:
            frame = cv2.cvtColor(env.get_screen(), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (720, 540), interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, "w/a/s/d walk  j=A k=B  |  warte auf Kampf...  q=abbruch",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 118), 1)
            cv2.imshow(win, frame)
            k = cv2.waitKey(16) & 0xFF
            if k == ord("q"):
                return False
            if k in keymap:
                action = keymap[k]
                hold = 16 if k in (ord("w"), ord("s"), ord("a"), ord("d")) else 6


def _settle(env, mbr, frames=_TURN_SETTLE_FRAMES):
    import numpy as np
    neutral = np.zeros(len(env.buttons), dtype=np.uint8)
    for _ in range(frames):
        env.em.set_button_mask(neutral)
        env.em.step()
        if not mbr.read(env.get_ram(), frames=1):
            return False
    return True


def run_wild(env, mbr, io):
    steps = []
    driver = EmulatorBattleDriver(io, main_battle_reader=mbr)
    snap = build_live_snapshot(io, main_battle_reader=mbr)
    reader_ok = (snap["player_active"] is not None and snap["enemy_active"] is not None
                 and snap["menu_cursor_known"] and snap["active_slot_authoritative"])
    steps.append({"check": "reader_plausible", "ok": reader_ok,
                  "menu_state": snap["menu_state"],
                  "player": _mon_brief(snap["player_active"]),
                  "enemy": _mon_brief(snap["enemy_active"]),
                  "action_mask": snap.get("action_cursor")})

    mask = driver.action_mask(snap)
    mask_ok = (mask[0] == 1 and mask[ALL_MACROS.index("RUN")] == 1
               and snap["is_trainer"] is False)
    steps.append({"check": "action_mask_plausible", "ok": mask_ok,
                  "mask": mask, "is_trainer": snap["is_trainer"]})

    # --- real MOVE_1 macro ---
    hp0 = snap["enemy_active"]["cur_hp"]
    php0 = snap["player_active"]["cur_hp"]
    pp0 = [m.get("pp") for m in snap["player_active"]["moves"]]
    s1, ev = driver.apply_macro("MOVE_1", dry_run=True)
    _settle(env, mbr)
    s1 = build_live_snapshot(io, main_battle_reader=mbr)
    changed = _state_changed(snap, s1, hp0, php0, pp0)
    steps.append({"check": "MOVE_1_real_macro_changes_state", "ok": changed["any"],
                  "before": {"enemy_hp": hp0, "player_hp": php0, "pp": pp0},
                  "after": {"enemy_hp": _hp(s1, "enemy_active"),
                            "player_hp": _hp(s1, "player_active"),
                            "pp": [m.get("pp") for m in (s1["player_active"] or {}).get("moves", [])]},
                  "executor": {"ok": ev.get("aborted") is not True,
                               "reason": ev.get("reason")},
                  "detail": changed})

    # --- SWITCH stays fail-closed without a verified party-list cursor ---
    switch_legal = any(mask[ALL_MACROS.index(m)] for m in
                       ("SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4",
                        "SWITCH_5", "SWITCH_6"))
    if not switch_legal:
        steps.append({"check": "switch_actions_fail_closed", "ok": True,
                      "note": "party-list cursor not verified; all live switches masked"})
    elif not driver.in_battle():
        steps.append({"check": "switch_slot_1", "ok": False,
                      "note": "battle ended before switch could be tested"})
    else:
        s_before = build_live_snapshot(io, main_battle_reader=mbr)
        _s, ev2 = driver.apply_macro("SWITCH_2", dry_run=True)
        _settle(env, mbr)
        s2 = build_live_snapshot(io, main_battle_reader=mbr)
        ram = env.get_ram()
        idx = L.battler_party_index(ram, 0)
        bmon = L.read_battle_mon(ram, 0)
        party = s2.get("player_party") or []
        slot1_ok = (idx == 1 and bmon is not None and len(party) > 1
                    and bmon["species_id"] == party[1].get("species_id"))
        steps.append({"check": "switch_to_slot_1", "ok": bool(slot1_ok),
                      "gBattlerPartyIndexes_player": idx,
                      "gBattleMons0": _mon_brief(bmon),
                      "party_slot_1": _mon_brief(party[1] if len(party) > 1 else None),
                      "executor_reason": ev2.get("reason")})
        # --- switch back to slot 0 ---
        if driver.in_battle():
            driver.apply_macro("SWITCH_1", dry_run=True)
            _settle(env, mbr)
            ram = env.get_ram()
            steps.append({"check": "switch_back_to_slot_0",
                          "ok": L.battler_party_index(ram, 0) == 0,
                          "gBattlerPartyIndexes_player": L.battler_party_index(ram, 0)})

    # --- RUN (wild only) ---
    if driver.in_battle():
        snr = build_live_snapshot(io, main_battle_reader=mbr)
        run_allowed = snr["is_trainer"] is False
        _s, evr = driver.apply_macro("RUN", dry_run=True) if run_allowed else (None, {"skipped": True})
        _settle(env, mbr, frames=240)
        oob = not driver.in_battle()
        steps.append({"check": "RUN_wild_and_out_of_battle", "ok": bool(oob and run_allowed),
                      "run_allowed": run_allowed, "out_of_battle_after": oob,
                      "executor_reason": (evr or {}).get("reason")})
    else:
        steps.append({"check": "RUN_wild_and_out_of_battle", "ok": True,
                      "note": "battle already ended (KO/flee via prior actions)"})

    # --- stable OUT_OF_BATTLE ---
    import numpy as np
    neutral = np.zeros(len(env.buttons), dtype=np.uint8)
    oob_stable = 0
    for _ in range(300):
        env.em.set_button_mask(neutral); env.em.step()
        if not mbr.read(env.get_ram(), frames=1):
            oob_stable += 1
        else:
            oob_stable = 0
    steps.append({"check": "OUT_OF_BATTLE_stable", "ok": oob_stable >= 60,
                  "consecutive_oob_frames": oob_stable})

    ok = all(s.get("ok") for s in steps)
    return {"phase": "wild", "ok": ok, "steps": steps, "diagnostics": driver.diagnostics}


def run_trainer(env, mbr, io):
    steps = []
    driver = EmulatorBattleDriver(io, main_battle_reader=mbr)
    snap = build_live_snapshot(io, main_battle_reader=mbr)
    is_trainer = snap["is_trainer"] is True
    steps.append({"check": "battle_kind_is_trainer", "ok": is_trainer,
                  "is_trainer": snap["is_trainer"], "trainer_id": snap["trainer_id"],
                  "battle_type_flags": snap["battle_type_flags"]})

    mask = driver.action_mask(snap)
    run_masked = mask[ALL_MACROS.index("RUN")] == 0
    steps.append({"check": "RUN_is_masked_in_trainer_battle", "ok": run_masked,
                  "mask": mask})

    # the canary must NEVER send a RUN sequence here
    _s, ev = driver.apply_macro("RUN", dry_run=True)
    steps.append({"check": "driver_refuses_RUN_no_button_sent",
                  "ok": ev.get("invalid") is True and ev.get("aborted") is not True,
                  "reason": ev.get("reason"),
                  "presses": [d.get("presses") for d in driver.diagnostics]})

    # one allowed MOVE
    if driver.in_battle():
        hp0 = _hp(snap, "enemy_active")
        _s, ev2 = driver.apply_macro("MOVE_1", dry_run=True)
        _settle(env, mbr)
        s1 = build_live_snapshot(io, main_battle_reader=mbr)
        steps.append({"check": "one_allowed_MOVE_executed",
                      "ok": ev2.get("aborted") is not True,
                      "enemy_hp_before": hp0, "enemy_hp_after": _hp(s1, "enemy_active"),
                      "reason": ev2.get("reason")})
    else:
        steps.append({"check": "one_allowed_MOVE_executed", "ok": False,
                      "note": "battle ended before a MOVE could run"})

    ok = all(s.get("ok") for s in steps)
    return {"phase": "trainer", "ok": ok, "steps": steps,
            "diagnostics": driver.diagnostics,
            "note": "trainer not fully defeated (canary stops early by design)"}


# -- helpers -------------------------------------------------------------
def _mon_brief(m):
    if not m:
        return None
    return {"species_id": m.get("species_id"), "level": m.get("level"),
            "hp": f"{m.get('cur_hp')}/{m.get('max_hp')}", "types": m.get("types"),
            "moves": [{"id": x.get("id"), "pp": x.get("pp")} for x in m.get("moves", [])]}


def _hp(snap, key):
    return (snap.get(key) or {}).get("cur_hp")


def _state_changed(before, after, ehp0, php0, pp0):
    e1 = _hp(after, "enemy_active")
    p1 = _hp(after, "player_active")
    pp1 = [m.get("pp") for m in (after.get("player_active") or {}).get("moves", [])]
    d = {
        "enemy_hp_changed": e1 is not None and ehp0 is not None and e1 != ehp0,
        "player_hp_changed": p1 is not None and php0 is not None and p1 != php0,
        "pp_changed": pp1 != pp0,
        "turn_progressed": (after.get("in_battle") is not True)
        or (after.get("menu_state") == "main"),
    }
    d["any"] = bool(d["enemy_hp_changed"] or d["player_hp_changed"]
                    or d["pp_changed"] or (after.get("in_battle") is not True))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["wild", "trainer", "both"], default="both")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        import cv2
        import numpy as np
    except Exception:
        cv2, np = None, None
        print("(cv2/numpy not available — running without a window; you must "
              "trigger the battle another way)")

    os.makedirs(CANARY_ROOT, exist_ok=True)
    if args.resume:
        dirs = sorted(d for d in os.listdir(CANARY_ROOT)
                      if os.path.isdir(os.path.join(CANARY_ROOT, d)))
        cdir = os.path.join(CANARY_ROOT, dirs[-1]) if dirs else None
    else:
        cdir = None
    if cdir is None:
        cdir = os.path.join(CANARY_ROOT,
                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(cdir, exist_ok=True)
    state_path = os.path.join(cdir, "canary_state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)

    report = {
        "schema": "live_battle_canary_v1",
        "timestamp": datetime.datetime.now().isoformat(),
        "rom_sha256": _sha256(ROM_PATH),
        "rom_expected": pa.ROM_SHA256,
        "verified_ram_addresses": {
            "gBattlerPartyIndexes": hex(L.OFF_PARTY_INDEXES),
            "gBattleMons": hex(L.OFF_BATTLE_MONS),
            "gActionSelectionCursor": hex(L.OFF_ACTION_CURSOR),
            "gMoveSelectionCursor": hex(L.OFF_MOVE_CURSOR),
            "battle_menu_state": hex(L.OFF_MENU_STATE),
            "gBattleWeather": hex(L.OFF_WEATHER),
        },
        "mandatory_fields": list(rp.MANDATORY_FIELDS),
        "phases": dict(state.get("phases", {})),
        "canary_dir": os.path.relpath(cdir, ROOT),
    }
    report["rom_ok"] = report["rom_sha256"] == report["rom_expected"]

    phases = (["wild", "trainer"] if args.phase == "both" else [args.phase])
    # stable_retro permits only one emulator instance for the lifetime of a
    # process. Reuse it across both phases and restore the working seed before
    # each phase instead of closing and constructing a second instance.
    env = _make_env()
    io = _Emu(env)
    try:
        for phase in phases:
            if phase in report["phases"] and report["phases"][phase].get("ok"):
                print(f"[{phase}] already PASSED (resume) — skipping")
                continue
            mbr = MainBattleReader()
            seedinfo = _load_working_copy(env, cdir)
            report["seed"] = seedinfo
            if not seedinfo["seed_unchanged"]:
                report["phases"][phase] = {
                    "ok": False,
                    "error": "seed hash changed / mismatch",
                    "seed": seedinfo,
                }
                break
            prompt = (f"\n=== Phase: {phase.upper()} ===\n"
                      + ("Lauf ins hohe Gras, bis ein WILDKAMPF startet.\n"
                         if phase == "wild" else
                         "Lauf zu einem noch nicht besiegten TRAINER und starte den Kampf.\n")
                  + ("Falls zuerst ein Wildkampf startet: selbst FLUCHT waehlen und "
                     "danach weiter zum Trainer laufen.\n" if phase == "trainer" else "")
                  + "Texte mit j bestaetigen. Erst bei 'Kampf erkannt' die Haende weg.")
            try:
                got = _wait_for_battle(env, mbr, prompt=prompt,
                                       expected_phase=phase, cv2=cv2, np=np)
                if not got:
                    report["phases"][phase] = {"ok": False, "error": "aborted / no battle"}
                else:
                    res = run_wild(env, mbr, io) if phase == "wild" else run_trainer(env, mbr, io)
                    report["phases"][phase] = res
            except Exception as exc:
                import traceback
                report["phases"][phase] = {"ok": False, "error": str(exc),
                                           "trace": traceback.format_exc()}
            finally:
                try:
                    if cv2 is not None:
                        cv2.destroyAllWindows()
                except Exception:
                    pass
            state = {"phases": report["phases"]}
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
    finally:
        env.close()

    wild_ok = report["phases"].get("wild", {}).get("ok") is True
    trn_ok = report["phases"].get("trainer", {}).get("ok") is True
    report["wild_result"] = "PASS" if wild_ok else "FAIL/absent"
    report["trainer_result"] = "PASS" if trn_ok else "FAIL/absent"
    report["overall"] = "PASS" if (wild_ok and trn_ok and report["rom_ok"]) else "FAIL"
    report["seed_unchanged_final"] = _sha256(os.path.join(ROOT, SEED_REL)) == SEED_SHA

    out = os.path.join(cdir, "canary_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== CANARY {report['overall']} ===")
    print(f"report: {os.path.relpath(out, ROOT)}")
    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
