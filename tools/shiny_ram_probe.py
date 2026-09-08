#!/usr/bin/env python3
"""Shiny RAM probe for the BPRD ROM (spec ZIEL C.2).

Verifies the three values Gen-III shininess needs, over a real encounter, in
its OWN isolated emulator (running trainer / watcher / web untouched):

  * the player's ``playerTrainerId`` (TID low16 + SID high16) — SaveBlock2
    offset is candidate-scanned and cross-checked against the value shown on
    the in-game trainer card / a party mon's ``otId``;
  * the active WILD enemy's 32-bit PID (personality) from the verified enemy
    party struct;
  * that the PID is STABLE for the whole encounter (no drift between the
    "appeared!" frame and the action menu).

Only after >= 2 distinct encounters agree does it print
``SHINY_RAM_VERIFIED: ok`` — then a human flips the flag in
``src/twoby2/shiny_ram.py``. This tool writes nothing to that module and never
changes a savestate.

    PYTHONPATH=src python tools/shiny_ram_probe.py
    -> runtime/shiny_probe/<timestamp>/shiny_probe_report.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_state import MainBattleReader                    # noqa: E402
import shiny as shiny_math                                    # noqa: E402

DEFAULT_START_STATE = os.path.join(
    ROOT, "brain_backups", "ram_probe_route1_seed", "stage_frontier_2.state.gz")

# SaveBlock2 playerTrainerId candidate offsets (EWRAM-relative). pokefirered US
# has gSaveBlock2Ptr -> +0x0A playerTrainerId[4]; BPRD relocates the block by an
# unknown delta, so we scan a small set of plausible EWRAM windows and confirm
# by cross-check, never by assumption.
_TID_CANDIDATE_WINDOWS = ((0x1000, 0x4000), (0xF000, 0x12000), (0x25000, 0x28000))


# --------------------------------------------------------------------------
# pure helpers (unit-testable)
# --------------------------------------------------------------------------
def u32(buf, off):
    if buf is None or off < 0 or off + 4 > len(buf):
        return None
    return struct.unpack_from("<I", buf, off)[0]


def tid_sid_candidates(ram, otid_hint, *, windows=_TID_CANDIDATE_WINDOWS):
    """Offsets whose u32 equals ``otid_hint`` (a party mon's otId, which for the
    player's own mons IS the packed playerTrainerId). Most-plausible first."""
    if otid_hint is None:
        return []
    hits = []
    for lo, hi in windows:
        hi = min(hi, len(ram) - 4) if ram else lo
        for off in range(lo, hi, 4):
            if u32(ram, off) == int(otid_hint):
                hits.append(off)
    return hits


def score_probe(attempts):
    """``attempts``: list of per-encounter dicts. Verified only with >= 2
    distinct encounters agreeing on one TID offset and a stable PID each."""
    enc = {a.get("encounter_id") for a in attempts if a.get("encounter_id")}
    tid_sets = [set(a.get("tid_offsets") or []) for a in attempts if a.get("tid_offsets")]
    common = set.intersection(*tid_sets) if tid_sets else set()
    pid_stable = all(a.get("pid_stable") for a in attempts) and bool(attempts)
    verified = len(enc) >= 2 and len(common) == 1 and pid_stable
    return {
        "n_attempts": len(attempts), "distinct_encounters": sorted(enc),
        "tid_offset": (sorted(common)[0] if common else None),
        "tid_offset_verified": len(enc) >= 2 and len(common) == 1,
        "pid_stable_all": pid_stable,
        "SHINY_RAM_VERIFIED": verified,
        "notes": ([] if verified else
                  ["need >= 2 distinct encounters, one common TID offset, "
                   "PID stable in every encounter"]),
    }


# --------------------------------------------------------------------------
# interactive probe
# --------------------------------------------------------------------------
class _E:
    def __init__(self, env):
        self._env = env

    def get_ram(self):
        return self._env.get_ram()


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_START_STATE)
    args = ap.parse_args(argv)
    start_state = _resolve(args.state)
    if not os.path.isfile(start_state):
        print(f"start state not found: {start_state}")
        return 2
    try:
        import cv2
        import numpy as np
        import stable_retro as retro
        import gzip
    except Exception as exc:                       # pragma: no cover
        print(f"needs cv2 + numpy + stable_retro: {exc}")
        return 1

    import firered_ram as fr
    retro.data.Integrations.add_custom_path(
        os.path.join(ROOT, "local", "custom_integrations"))
    env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
    env.reset()
    with gzip.open(start_state, "rb") as f:
        env.em.set_state(f.read())
    env.em.step()

    outdir = os.path.join(ROOT, "runtime", "shiny_probe",
                          datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(__doc__)
    print("w/a/s/d walk · j=A · k=B · c=capture this encounter · r=rescore · q=quit\n")

    btns = list(env.buttons)

    def mask(*names):
        m = [0] * len(btns)
        for n in names:
            if n in btns:
                m[btns.index(n)] = 1
        return np.array(m, dtype=np.uint8)

    WALK = {ord("w"): mask("UP"), ord("s"): mask("DOWN"),
            ord("a"): mask("LEFT"), ord("d"): mask("RIGHT")}
    BTN = {ord("j"): mask("A"), ord("k"): mask("B")}
    NOOP = np.zeros(len(btns), dtype=np.uint8)
    mbr = MainBattleReader()
    attempts = []
    enc = 0
    was_in_battle = False
    pid_samples = []

    cv2.namedWindow("shiny_ram_probe", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("shiny_ram_probe", 720, 540)
    action, hold = NOOP, 0

    def capture():
        nonlocal enc
        ram = bytes(env.get_ram())
        party = fr.read_player_party(_E(env))
        try:
            enemy_party = fr.read_enemy_party(_E(env))
        except Exception:
            enemy_party = []
        own_otid = next((int(m["ot_id"]) for m in party
                         if m and m.get("checksum_ok") and "ot_id" in m), None)
        emon = next((m for m in enemy_party
                     if m and m.get("checksum_ok") and int(m.get("cur_hp", 0)) > 0), None)
        pid = int(emon["personality"]) if emon and "personality" in emon else None
        tid_offs = tid_sid_candidates(ram, own_otid)
        pid_stable = bool(pid is not None and pid_samples
                          and all(s == pid for s in pid_samples))
        tid = sid = sv = None
        if own_otid is not None:
            tid, sid = shiny_math.split_trainer_id(own_otid)
            if pid is not None:
                sv = shiny_math.shiny_value(tid, sid, pid)
        rec = {
            "encounter_id": f"enc{enc:02d}",
            "own_otid": own_otid, "tid": tid, "sid": sid,
            "enemy_species": (emon or {}).get("species_id"),
            "enemy_level": (emon or {}).get("level"),
            "wild_pid": pid, "pid_stable": pid_stable,
            "pid_samples": list(pid_samples),
            "tid_offsets": tid_offs,
            "shiny_value_from_otid": sv,
            "would_be_shiny": (sv is not None and sv < shiny_math.SHINY_THRESHOLD),
            "in_battle": bool(mbr.read(ram)),
        }
        base = os.path.join(outdir, f"enc{enc:02d}.json")
        with open(base, "w") as f:
            json.dump(rec, f, indent=2)
        attempts.append(rec)
        print(f"  captured enc{enc:02d}: pid={pid} stable={pid_stable} "
              f"otid={own_otid} tid_offsets={tid_offs[:6]} "
              f"shiny_value={sv} would_be_shiny={rec['would_be_shiny']}")

    def rescore():
        rep = score_probe(attempts)
        with open(os.path.join(outdir, "shiny_probe_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))

    while True:
        env.em.set_button_mask(action if hold > 0 else NOOP)
        env.em.step()
        hold = max(0, hold - 1)
        ram = bytes(env.get_ram())
        in_b = bool(mbr.read(ram))
        if in_b:
            try:
                ep = fr.read_enemy_party(_E(env))
                em = next((m for m in ep if m and m.get("checksum_ok")
                           and int(m.get("cur_hp", 0)) > 0), None)
                if em and "personality" in em:
                    pid_samples.append(int(em["personality"]))
                    pid_samples[:] = pid_samples[-40:]
            except Exception:
                pass
        if in_b and not was_in_battle:
            enc += 1
            pid_samples[:] = []
            print(f"  -- encounter enc{enc:02d} --")
        was_in_battle = in_b

        frame = env.em.get_screen() if hasattr(env.em, "get_screen") else None
        if frame is not None:
            cv2.imshow("shiny_ram_probe", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if k == 0xFF:
            continue
        if k == ord("q"):
            break
        if k in WALK:
            action, hold = WALK[k], 8
        elif k in BTN:
            action, hold = BTN[k], 6
        elif k == ord("c"):
            capture()
        elif k == ord("r"):
            rescore()

    rescore()
    cv2.destroyAllWindows()
    env.close()
    print(f"\ndone -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
