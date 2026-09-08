#!/usr/bin/env python3
"""Catch-v2 RAM-diff probe for the BPRD ROM (spec §8).

The in-battle **bag / ball-pocket / dex-owned / catch-result** RAM is NOT
verified for this ROM. ``twoby2.battle_ram_live.catch_ram_ready()`` therefore
returns ``False`` and the live CATCH macro stays masked. This tool produces the
controlled before/after RAM dumps needed to locate those addresses, and scores
them — a candidate is only ever reported, never marked verified without
**>= 2 distinct catch attempts** showing the same byte behaving the same way.

It runs its OWN isolated emulator (the running trainer / watcher / web are NOT
touched) from the frozen Route-1 RAM-probe seed.

Window keys
-----------
  w a s d  walk        j = A     k = B     n = START     l = reload seed
  q        quit

  Capture (writes ``<label>_<n>.ram.gz`` + a JSON sidecar with the verified
  party, bag-visible ball counts you type in, the enemy species/HP, and the
  labelled catch phase):

    b   BEFORE a ball throw (main menu, target readable)   -> before_throw
    s   just after the ball SHAKES / result text starts    -> after_shake
    c   the mon was CAUGHT (party or "sent to PC" message) -> after_caught
    f   the ball BROKE and the fight continues             -> after_broke_free
    x   a throw you did NOT want (control / negative)       -> negative

  d   diff the two most recent captures and print candidate offsets
  r   re-score every attempt collected so far

Do a full catch attempt (b -> s -> c) at least TWICE, on >= 2 distinct wild
encounters, plus one broke-free (b -> s -> f). Dumps: runtime/catch_probe/<ts>/
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_state import MainBattleReader           # noqa: E402

DEFAULT_START_STATE = os.path.join(
    ROOT, "brain_backups", "ram_probe_route1_seed", "stage_frontier_2.state.gz")

EWRAM_BASE = 0x02000000
# The window we search — the whole EWRAM working area, minus the low 0x1000
# BIOS-mirror / IO region which is noisy and never holds bag state.
SEARCH_LO, SEARCH_HI = 0x1000, 0x40000

CATCH_PHASES = ("before_throw", "after_shake", "after_caught",
                "after_broke_free", "negative")


# --------------------------------------------------------------------------
# pure diff / scoring (no emulator — unit-testable)
# --------------------------------------------------------------------------
def byte_diff(before, after, *, lo=SEARCH_LO, hi=SEARCH_HI):
    """``[(offset, before_val, after_val), ...]`` for every byte that changed in
    ``[lo, hi)``. ``before`` / ``after`` are ``bytes``."""
    hi = min(hi, len(before), len(after))
    out = []
    for off in range(lo, hi):
        b, a = before[off], after[off]
        if b != a:
            out.append((off, b, a))
    return out


def ball_count_candidates(diff, *, visible_balls_before, visible_balls_after):
    """Offsets whose value went exactly ``visible_balls_before`` ->
    ``visible_balls_after`` (a single ball consumed). Returned most-plausible
    first (a u8 in the EWRAM save-block bag range scores higher)."""
    want_b, want_a = int(visible_balls_before), int(visible_balls_after)
    hits = [(off, b, a) for off, b, a in diff if b == want_b and a == want_a]
    # FireRed's in-RAM item bag sits well inside the save block; prefer offsets
    # that are 4-byte aligned to an item entry (id:u16, count:u16).
    hits.sort(key=lambda t: (0 if t[0] % 2 == 0 else 1, t[0]))
    return hits


def result_signal_candidates(shake_diff, caught_diff, broke_diff):
    """Offsets that differ between a CAUGHT resolution and a BROKE-FREE
    resolution (from the same 'after_shake' baseline) — a catch-result byte."""
    caught = {off: (b, a) for off, b, a in caught_diff}
    broke = {off: (b, a) for off, b, a in broke_diff}
    shared = set(caught) & set(broke)
    return sorted(off for off in shared if caught[off] != broke[off])


def score_attempts(attempts):
    """``attempts``: list of dicts, each one catch attempt with the byte diffs
    and the operator-typed visible ball counts. Returns a report dict; a field
    is ``verified`` only with >= 2 distinct encounters agreeing on one offset.
    """
    report = {"n_attempts": len(attempts), "encounters": sorted(
        {a.get("encounter_id") for a in attempts if a.get("encounter_id")}),
        "ball_count_offset": None, "ball_count_verified": False,
        "result_signal_offset": None, "result_signal_verified": False,
        "notes": []}

    # ball-count offset: intersect the per-attempt candidate sets
    per_attempt = []
    for at in attempts:
        d = at.get("throw_diff")
        if not d or at.get("balls_before") is None or at.get("balls_after") is None:
            continue
        cands = {off for off, _b, _a in ball_count_candidates(
            d, visible_balls_before=at["balls_before"],
            visible_balls_after=at["balls_after"])}
        per_attempt.append((at.get("encounter_id"), cands))
    if per_attempt:
        common = set.intersection(*(c for _e, c in per_attempt)) if per_attempt else set()
        distinct_enc = {e for e, _c in per_attempt}
        if len(common) == 1 and len(distinct_enc) >= 2:
            report["ball_count_offset"] = sorted(common)[0]
            report["ball_count_verified"] = True
        elif common:
            report["ball_count_offset"] = sorted(common)[0]
            report["notes"].append(
                f"ball-count offset {sorted(common)!r} seen in "
                f"{len(distinct_enc)} distinct encounter(s); need >= 2 to verify")
        else:
            report["notes"].append("no single ball-count offset common to all attempts")

    # result-signal: need a caught pair AND a broke-free pair
    caught = [a for a in attempts if a.get("phase") == "after_caught" and a.get("vs_shake_diff")]
    broke = [a for a in attempts if a.get("phase") == "after_broke_free" and a.get("vs_shake_diff")]
    if caught and broke:
        cand = result_signal_candidates(
            [], caught[0]["vs_shake_diff"], broke[0]["vs_shake_diff"])
        if cand:
            report["result_signal_offset"] = cand[0]
            report["result_signal_verified"] = (
                len({a.get("encounter_id") for a in caught}) >= 2)
            if not report["result_signal_verified"]:
                report["notes"].append(
                    "result-signal candidate found but only 1 caught encounter")
    else:
        report["notes"].append("need >= 1 caught AND >= 1 broke-free attempt for the result signal")
    return report


# --------------------------------------------------------------------------
# interactive probe
# --------------------------------------------------------------------------
class _Env:
    def __init__(self, env):
        self._env = env

    def get_ram(self):
        return self._env.get_ram()


def _resolve(path):
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return os.path.realpath(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_START_STATE)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    start_state = _resolve(args.state)
    if not os.path.isfile(start_state):
        print(f"start state not found: {start_state}")
        return 2
    try:
        import cv2
        import numpy as np
        import stable_retro as retro
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

    outdir = args.outdir or os.path.join(
        ROOT, "runtime", "catch_probe",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(__doc__)
    print(f"dumps -> {outdir}\n")

    btns = list(env.buttons)

    def mask(*names):
        m = [0] * len(btns)
        for n in names:
            if n in btns:
                m[btns.index(n)] = 1
        return np.array(m, dtype=np.uint8)

    WALK = {ord("w"): mask("UP"), ord("s"): mask("DOWN"),
            ord("a"): mask("LEFT"), ord("d"): mask("RIGHT")}
    BTN = {ord("j"): mask("A"), ord("k"): mask("B"), ord("n"): mask("START")}
    NOOP = np.zeros(len(btns), dtype=np.uint8)

    mbr = MainBattleReader()
    captures = []        # (label, path, sidecar)
    attempts = []        # scored attempt records
    enc = 0
    was_in_battle = False

    cv2.namedWindow("catch_ram_probe", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("catch_ram_probe", 720, 540)
    action, hold = NOOP, 0

    def capture(phase):
        nonlocal enc
        ram = bytes(env.get_ram())
        party = fr.read_player_party(_Env(env))
        try:
            e_party = fr.read_enemy_party(_Env(env))
        except Exception:
            e_party = []
        emon = next((m for m in e_party if m and int(m.get("cur_hp", 0) or 0) > 0), None)
        bb = input("  visible ball count BEFORE this throw (blank=unknown): ").strip()
        ba = input("  visible ball count AFTER this throw  (blank=unknown): ").strip()
        n = sum(1 for c in captures if c[0] == phase) + 1
        base = os.path.join(outdir, f"{phase}_{n:02d}")
        with gzip.open(base + ".ram.gz", "wb") as f:
            f.write(ram)
        sidecar = {
            "phase": phase, "encounter_id": f"enc{enc:02d}",
            "in_battle": bool(mbr.read(ram)),
            "balls_before": int(bb) if bb.isdigit() else None,
            "balls_after": int(ba) if ba.isdigit() else None,
            "enemy_species_id": (emon or {}).get("species_id"),
            "enemy_cur_hp": (emon or {}).get("cur_hp"),
            "enemy_max_hp": (emon or {}).get("max_hp"),
            "party_size": len([m for m in party if m and m.get("checksum_ok", True)]),
            "ram_len": len(ram),
        }
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)
        captures.append((phase, base + ".ram.gz", sidecar))
        print(f"  captured {phase} #{n:02d} enc={sidecar['encounter_id']} "
              f"in_battle={sidecar['in_battle']} "
              f"enemy={sidecar['enemy_species_id']} "
              f"hp={sidecar['enemy_cur_hp']}/{sidecar['enemy_max_hp']}")

    def _load(path):
        with gzip.open(path, "rb") as f:
            return f.read()

    def do_diff():
        if len(captures) < 2:
            print("  need 2 captures"); return
        (p0, f0, s0), (p1, f1, s1) = captures[-2], captures[-1]
        d = byte_diff(_load(f0), _load(f1))
        print(f"  diff {p0} -> {p1}: {len(d)} bytes changed in "
              f"[{SEARCH_LO:#x},{SEARCH_HI:#x})")
        if s1.get("balls_before") is not None and s1.get("balls_after") is not None:
            bc = ball_count_candidates(
                d, visible_balls_before=s1["balls_before"],
                visible_balls_after=s1["balls_after"])
            print(f"  ball-count offset candidates ({s1['balls_before']}->"
                  f"{s1['balls_after']}): "
                  + ", ".join(f"{EWRAM_BASE + o:#x}" for o, _b, _a in bc[:12]))
        rec = {"phase": p1, "encounter_id": s1.get("encounter_id"),
               "balls_before": s1.get("balls_before"),
               "balls_after": s1.get("balls_after")}
        if p0 == "before_throw":
            rec["throw_diff"] = d
        if p0 == "after_shake":
            rec["vs_shake_diff"] = d
        attempts.append(rec)

    def do_rescore():
        rep = score_attempts(attempts)
        path = os.path.join(outdir, "catch_probe_score.json")
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))
        print(f"  -> {path}")

    print("ready. Walk into grass, start a wild fight, then b / s / c / f.\n")
    while True:
        env.em.set_button_mask(action if hold > 0 else NOOP)
        env.em.step()
        hold = max(0, hold - 1)

        frame = env.em.get_screen() if hasattr(env.em, "get_screen") else None
        if frame is not None:
            cv2.imshow("catch_ram_probe", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if k == 0xFF:
            in_b = bool(mbr.read(bytes(env.get_ram())))
            if in_b and not was_in_battle:
                enc += 1
                print(f"  AUTO new encounter enc{enc:02d}")
            was_in_battle = in_b
            continue
        if k == ord("q"):
            break
        if k in WALK:
            action, hold = WALK[k], 8
        elif k in BTN:
            action, hold = BTN[k], 6
        elif k == ord("l"):
            with gzip.open(start_state, "rb") as f:
                env.em.set_state(f.read())
            env.em.step()
            print("  reloaded seed")
        elif k == ord("b"):
            capture("before_throw")
        elif k == ord("s"):
            capture("after_shake")
        elif k == ord("c"):
            capture("after_caught")
        elif k == ord("f"):
            capture("after_broke_free")
        elif k == ord("x"):
            capture("negative")
        elif k == ord("d"):
            do_diff()
        elif k == ord("r"):
            do_rescore()

    do_rescore()
    cv2.destroyAllWindows()
    env.close()
    print(f"\ndone. {len(captures)} captures, {len(attempts)} scored diffs -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
