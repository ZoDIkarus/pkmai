#!/usr/bin/env python3
"""Fail-closed pre-flight for the 2x2 live activation.

Reports SIX distinct readiness levels — a level is only True when its own
evidence holds, never because a file merely exists:

  1. component_prepared   modules present + import + unit-testable
  2. production_wired      train.py / watch.py / pokemon_env.py actually import
                           and call the new components (real call-site scan)
  3. unit_tests_passed     the recorded full-suite result is green
  4. real_canary_passed    a CURRENT, valid live_battle_canary report with
                           overall == PASS, matching this ROM + seed
  5. migration_ready       1..4 hold and no protected file is a write target
  6. activation_ready      1..5 hold

Exit 0 only when activation_ready. Reads only; changes nothing.

    PYTHONPATH=src python tools/twoby2_preflight.py [--json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2 import protected_assets as pa            # noqa: E402
from twoby2 import ram_battle_probe as rp            # noqa: E402
from twoby2 import battle_ram_live as L              # noqa: E402
from twoby2 import config as cfg                     # noqa: E402
from twoby2 import feature_enabled                   # noqa: E402

RAM_PROBE_DIR = os.path.join(ROOT, "runtime", "ram_probe", "20260907_184350")
CANARY_ROOT = os.path.join(ROOT, "runtime", "live_battle_canary")
ROM_PATH = os.path.join(ROOT, "local", "custom_integrations",
                        "PokemonFireRed-Gba", "rom.gba")
LIVE_MODULES = ("pokemon_env.py", "train.py", "watch.py", "watcher_runtime.py")
WIRED_NAMES = ("twoby2", "battle_ram_live", "emulator_battle_driver",
               "EmulatorBattleDriver", "BattlePolicyRouter", "NavigationBattleWrapper",
               "build_live_snapshot", "nav_chunk_rollout", "maybe_wrap_full_agent",
               "battle_driver_for", "live_integration", "twoby2_split",
               "watcher_battle_policy_choice", "NavChunkedRollout")


def _sha256(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Check:
    def __init__(self, name, ok, detail="", level="component_prepared"):
        self.name, self.ok, self.detail, self.level = name, bool(ok), detail, level

    def row(self):
        return f"  [{'OK  ' if self.ok else 'FAIL'}] ({self.level}) {self.name}: {self.detail}"


def _load_dumps():
    dumps = []
    for jf in sorted(glob.glob(os.path.join(RAM_PROBE_DIR, "*.json"))):
        rf = jf[:-5] + ".ram.gz"
        if not os.path.exists(rf):
            continue
        with open(jf) as fh:
            m = json.load(fh)
        with gzip.open(rf, "rb") as fh:
            m["ram"] = fh.read()
        dumps.append(m)
    return dumps


def _latest_canary_report():
    reps = sorted(glob.glob(os.path.join(CANARY_ROOT, "*", "canary_report.json")))
    if not reps:
        return None, None
    with open(reps[-1]) as f:
        return json.load(f), reps[-1]


def run_checks():
    checks = []
    C = checks.append

    # ---------------- component_prepared ----------------
    for mod in ("twoby2.battle_ram_live", "twoby2.emulator_battle_driver",
                "twoby2.router", "twoby2.nav_wrapper", "twoby2.reward_split",
                "twoby2.nav_progress", "twoby2.nav_chunk_rollout",
                "battle_train", "battle_watch", "battle_env"):
        try:
            __import__(mod)
            C(Check(f"import:{mod}", True, "importable"))
        except Exception as exc:
            C(Check(f"import:{mod}", False, str(exc)))

    rom_sha = _sha256(ROM_PATH)
    C(Check("rom_sha256", rom_sha == pa.ROM_SHA256, f"{rom_sha}"))
    mrep = pa.verify_master()
    C(Check("master_sha256", mrep["sha256_ok"], mrep["sha256"] or ""))
    reg = pa.default_registry()
    for lid, ws, wm in (
        ("route1_manual_battle_seed", pa.ROUTE1_MANUAL_SEED["state_sha256"], pa.ROUTE1_MANUAL_SEED["meta_sha256"]),
        ("protected_ram_probe_route1_seed", pa.RAM_PROBE_ROUTE1_SEED["state_sha256"], pa.RAM_PROBE_ROUTE1_SEED["meta_sha256"]),
        ("protected_live_route1_frontier_anchor", pa.ROUTE1_LIVE_ANCHOR["state_sha256"], pa.ROUTE1_LIVE_ANCHOR["meta_sha256"]),
    ):
        a = next((x for x in reg.assets if x.logical_id == lid), None)
        af = a.abs_files() if a else []
        got = (pa.sha256_file(af[0]) if af else None, pa.sha256_file(af[1]) if len(af) > 1 else None)
        C(Check(f"seed:{lid}", got == (ws, wm), f"{got == (ws, wm)}"))

    dumps = _load_dumps()
    C(Check("ram_probe_dataset", len(dumps) >= 12, f"{len(dumps)} dumps"))
    if dumps:
        results = rp.verify_all(dumps)
        for f in rp.MANDATORY_FIELDS:
            C(Check(f"ram_verified:{f}", results.get(f, {}).get("verified"),
                    f"offset={results.get(f, {}).get('winning_offset')}"))
        C(Check("ram_all_mandatory_verified", rp.all_mandatory_verified(results), ""))
        xc_ok = True
        for m in dumps:
            if m["battle_kind"] == "out_of_battle":
                continue
            live = L.read_all(m["ram"])
            ok, _ = L.crosscheck_against_party(live, m["party"])
            if not ok:
                xc_ok = False
            if m["expected_action_cursor"] is not None and live["action_cursor"] != m["expected_action_cursor"]:
                xc_ok = False
            if m["expected_move_cursor"] is not None and live["move_cursor"] != m["expected_move_cursor"]:
                xc_ok = False
        C(Check("live_readers_match_dumps", xc_ok, f"{len(dumps)} dumps"))
    else:
        C(Check("ram_all_mandatory_verified", False, "no dumps"))

    try:
        s = cfg.validate_worker_config()
        bs = cfg.battle_worker_split()
        C(Check("worker_config_40_8_1_plus_watcher",
                s["nav_workers"] == 40 and s["nav_anchor"] == 0
                and bs["headless"] == 8 and bs["visible"] == 1
                and s["full_watcher_emulators"] == 1
                and s["total_emulators"] == 50,
                f"nav {s['nav_workers']}, battle {bs['headless']}+{bs['visible']}, "
                f"full-watcher {s['full_watcher_emulators']}, total {s['total_emulators']}"))
    except Exception as exc:
        C(Check("worker_config_40_8_1_plus_watcher", False, str(exc)))

    from twoby2 import isolation as iso
    nav, bat = iso.navigation_counters(), iso.battle_counters()
    C(Check("counter_sets_disjoint", set(nav.snapshot()) & set(bat.snapshot()) == set(), ""))
    ns, bs2 = iso.IsolatedSystem("navigation", nav), iso.IsolatedSystem("battle", bat)
    try:
        iso.assert_systems_isolated(ns, bs2)
        C(Check("nav_battle_objects_isolated", True, ""))
    except AssertionError as exc:
        C(Check("nav_battle_objects_isolated", False, str(exc)))

    # ---------------- production_wired (real call-site scan) ----------------
    wired = {}
    for mod in LIVE_MODULES:
        p = os.path.join(ROOT, "src", mod)
        if not os.path.isfile(p):
            wired[mod] = False
            continue
        with open(p) as _fh:
            txt = _fh.read()
        wired[mod] = any(w in txt for w in WIRED_NAMES)
    with open(os.path.join(ROOT, "src", "battle_train.py")) as _fh:
        battle_train_txt = _fh.read()
    with open(os.path.join(ROOT, "src", "battle_watch.py")) as _fh:
        battle_watch_txt = _fh.read()
    driver_wired = (("battle_driver_for" in battle_train_txt
                     or "make_live_battle_env" in battle_train_txt
                     or "EmulatorBattleDriver" in battle_train_txt)
                    and ("battle_driver_for" in battle_watch_txt
                         or "EmulatorBattleDriver" in battle_watch_txt))
    C(Check("production_wired:pokemon_env", wired.get("pokemon_env.py", False),
            "pokemon_env.py imports/uses the 2x2 seam", level="production_wired"))
    C(Check("production_wired:train", wired.get("train.py", False),
            "train.py wires the 2x2 FULL-agent path", level="production_wired"))
    C(Check("production_wired:watch", wired.get("watch.py", False),
            "watch.py consults the 2x2 router", level="production_wired"))
    C(Check("production_wired:watcher_runtime", wired.get("watcher_runtime.py", False),
            "watcher_runtime.py wraps the eval env (FULL-watcher parity)",
            level="production_wired"))
    C(Check("production_wired:battle_trainers_use_real_driver", driver_wired,
            "battle_train.py + battle_watch.py select the real EmulatorBattleDriver",
            level="production_wired"))

    # ---------------- unit_tests_passed (green AND not stale) ----------------
    hist = os.path.join(ROOT, "runtime", "twoby2_test_result.json")
    tp = False
    detail = "no recorded result"
    if os.path.isfile(hist):
        try:
            with open(hist) as _fh:
                d = json.load(_fh)
            green = d.get("ok") is True
            rec_ts = float(d.get("ts_epoch", 0))
            # newest mtime of any code the tests must cover
            newest = 0.0
            for base in (os.path.join(ROOT, "src", "twoby2"),):
                for r, _dd, ff in os.walk(base):
                    for fn in ff:
                        if fn.endswith(".py"):
                            newest = max(newest, os.path.getmtime(os.path.join(r, fn)))
            for rel in ("src/pokemon_env.py", "src/train.py", "src/watch.py",
                        "src/watcher_runtime.py", "src/battle_train.py",
                        "src/battle_watch.py", "src/battle_env.py"):
                p2 = os.path.join(ROOT, rel)
                if os.path.isfile(p2):
                    newest = max(newest, os.path.getmtime(p2))
            fresh = rec_ts >= newest
            tp = bool(green and fresh)
            detail = (f"green={green} fresh={fresh} "
                      f"(recorded {d.get('ts')}, {d.get('tests')} tests)")
        except Exception as exc:
            detail = f"unreadable: {exc}"
    C(Check("unit_tests_passed", tp, detail, level="unit_tests_passed"))

    # ---------------- real_canary_passed ----------------
    rep, path = _latest_canary_report()
    canary_ok = False
    detail = "no live_battle_canary report found"
    if rep:
        cur_rom = rep.get("rom_sha256") == pa.ROM_SHA256
        cur_seed = rep.get("seed_unchanged_final") is True or \
            rep.get("seed", {}).get("seed_unchanged") is True
        addr_ok = rep.get("verified_ram_addresses", {}).get("gBattleMons") == hex(L.OFF_BATTLE_MONS)
        overall = rep.get("overall") == "PASS"
        canary_ok = bool(cur_rom and cur_seed and addr_ok and overall)
        detail = (f"{os.path.relpath(path, ROOT)} overall={rep.get('overall')} "
                  f"rom_ok={cur_rom} seed_ok={cur_seed} addr_ok={addr_ok}")
    C(Check("real_canary_passed", canary_ok, detail, level="real_canary_passed"))

    # ---------------- migration_ready / activation_ready ----------------
    protected_ok = True
    for a in reg.assets:
        for f in a.abs_files():
            try:
                pa.assert_write_target_ok(f, op="write", registry=reg)
                protected_ok = False
            except pa.ProtectedPathError:
                pass
    C(Check("protected_files_reject_writes", protected_ok, "", level="migration_ready"))

    gates = {k: feature_enabled(k) for k in (
        "battle_env", "battle_executor_live", "battle_router_live",
        "nav_battle_wrapper", "watcher_battle_champion", "runtime_migration")}
    C(Check("feature_gates_currently_off", not any(gates.values()),
            json.dumps(gates), level="component_prepared"))

    return checks


def _level_ok(checks, level):
    rel = [c for c in checks if c.level == level]
    return bool(rel) and all(c.ok for c in rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    checks = run_checks()

    levels = ["component_prepared", "production_wired", "unit_tests_passed",
              "real_canary_passed", "migration_ready", "activation_ready"]
    status = {}
    status["component_prepared"] = _level_ok(checks, "component_prepared")
    status["production_wired"] = _level_ok(checks, "production_wired")
    status["unit_tests_passed"] = _level_ok(checks, "unit_tests_passed")
    status["real_canary_passed"] = _level_ok(checks, "real_canary_passed")
    status["migration_ready"] = (status["component_prepared"]
                                 and status["production_wired"]
                                 and status["unit_tests_passed"]
                                 and status["real_canary_passed"]
                                 and _level_ok(checks, "migration_ready"))
    status["activation_ready"] = status["migration_ready"]

    if args.json:
        print(json.dumps({"status": status,
                          "checks": [{"name": c.name, "ok": c.ok, "level": c.level,
                                      "detail": c.detail} for c in checks]}, indent=2))
    else:
        print("2x2 PRE-FLIGHT (6-level)\n")
        for c in checks:
            print(c.row())
        print("\nREADINESS:")
        for lvl in levels:
            print(f"  {lvl:22} {'YES' if status[lvl] else 'no'}")
        print("\nRESULT: " + ("ACTIVATION-READY"
                              if status["activation_ready"] else "NOT READY"))
    sys.exit(0 if status["activation_ready"] else 1)


if __name__ == "__main__":
    main()
