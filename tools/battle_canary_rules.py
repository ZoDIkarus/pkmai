#!/usr/bin/env python3
"""Automated battle canary — the RULE controller vs the real emulator.

Before the battle learner is restarted (Step 5 of the battle repair), the
verified rule controller must play a batch of real battles cleanly through the
fixed end-detection / menu state machine:

  * >= 18 / 20 wild battles correctly detected as a win
  * zero ``terminal_unknown`` (a battle that ended but could not be classified)
  * zero ``menu_stall``
  * no episode reaches ``max_turns`` (60)
  * total enemy KOs match total wins (one wild enemy per Route-1 battle)
  * a forced RUN is detected as ``fled`` and the emulator settles OUT_OF_BATTLE

Needs ``PKMAI_TWOBY2_LIVE=1`` (opens an isolated emulator; the running
navigation trainer / watcher / web are not touched). Writes a JSON report to
``runtime/battle/canary_rules/<ts>/report.json`` and exits non-zero on FAIL.

    PYTHONPATH=src PKMAI_TWOBY2_LIVE=1 python tools/battle_canary_rules.py [--battles 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

WIN_MIN = 18            # of --battles
MAX_TURNS = 25         # a Route-1 wild win is 1-3 turns; matches the live cap


def _scenario(scenario_id=None):
    idx = os.path.join(ROOT, "runtime", "battle", "scenarios", "index.json")
    with open(idx) as f:
        rows = (json.load(f) or {}).get("scenarios", [])
    wild = [r for r in rows
            if not r.get("is_trainer")
            and os.path.isfile(r.get("savestate_path", ""))]
    if scenario_id is not None:
        wild = [r for r in wild if r.get("id") == scenario_id]
    if not wild:
        detail = (f" with id {scenario_id!r}" if scenario_id is not None else "")
        raise SystemExit("no captured wild battle scenario" + detail + " in "
                         "runtime/battle/scenarios/index.json")
    return dict(wild[0])


def _rule_action(env, obs, ALL_MACROS):
    """Rule controller decision -> a legal macro index for this env state."""
    import battle_controller
    snap = env.driver._emu.snapshot()
    decision = battle_controller.decide(snap)
    macro = decision.get("action", "MOVE_1")
    a = ALL_MACROS.index(macro) if macro in ALL_MACROS else 0
    mask = list(obs["action_mask"])
    if not mask[a]:
        a = next((i for i, m in enumerate(mask) if m), 0)
    return a, macro, decision.get("reason", "")


def _play(env, scenario, ALL_MACROS, *, seed, force_run=False):
    try:
        obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    except Exception as exc:
        return {"turns": 0, "outcome": "reset_failed", "kos": 0,
                "terminal_unknown": False, "menu_stall": False, "invalid": 0,
                "no_damage": 0, "aborted": 0, "actions": [],
                "settled_out_of_battle": False, "error": str(exc)[:200]}
    snap0 = env.driver._emu.snapshot()
    moves0 = [m for m in ((snap0.get("player_active") or {}).get("moves") or [])
              if m.get("mechanics_known")]
    rec = {"turns": 0, "outcome": None, "kos": 0, "terminal_unknown": False,
           "menu_stall": False, "invalid": 0, "no_damage": 0, "aborted": 0,
           "readable_moves_at_start": len(moves0),
           "enemy_readable_at_start": bool(snap0.get("enemy_active")),
           "actions": []}
    done = trunc = False
    info = {}
    while not (done or trunc):
        if force_run and rec["turns"] == 0:
            a = ALL_MACROS.index("RUN")
            if not list(obs["action_mask"])[a]:
                a = next((i for i, m in enumerate(obs["action_mask"]) if m), 0)
            macro = ALL_MACROS[a]
        else:
            a, macro, _reason = _rule_action(env, obs, ALL_MACROS)
        obs, _r, done, trunc, info = env.step(a)
        rec["turns"] += 1
        rec["actions"].append(macro)
        rec["kos"] += int(bool(info.get("enemy_ko")))
        rec["invalid"] += int(bool(info.get("invalid")))
        rec["aborted"] += int(bool(info.get("aborted")))
        rec["no_damage"] += int(bool(info.get("no_damage")))
        rec["terminal_unknown"] |= bool(info.get("terminal_unknown"))
        rec["menu_stall"] |= bool(info.get("menu_stall"))
    rec["outcome"] = info.get("outcome") or ("terminated_no_outcome" if done
                                             else "truncated_no_outcome")
    try:
        rec["settled_out_of_battle"] = env.driver._emu.in_battle() is not True
    except Exception:
        rec["settled_out_of_battle"] = None
    diag = list(getattr(env.driver._emu, "diagnostics", []))
    rec["diagnostics_count"] = len(diag)
    rec["diagnostics"] = (diag[:3] + [{"...": len(diag) - 9}] + diag[-6:]
                          if len(diag) > 9 else diag)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=20)
    ap.add_argument("--scenario-id",
                    help="run the canary against this captured scenario id")
    ap.add_argument("--harvest-run-seed", type=int,
                    help="override the deterministic harvester run seed")
    args = ap.parse_args()

    from twoby2 import feature_enabled
    if not feature_enabled("battle_env"):
        raise SystemExit("PKMAI_TWOBY2_LIVE=1 is required (FEATURES['battle_env'] off)")

    from battle_env import make_live_battle_env, ALL_MACROS

    scenario = _scenario(args.scenario_id)
    if args.harvest_run_seed is not None:
        scenario["run_seed"] = args.harvest_run_seed
    out_dir = os.path.join(ROOT, "runtime", "battle", "canary_rules",
                           time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    env = make_live_battle_env(scenario_sampler=None, seed=4242, max_turns=MAX_TURNS)
    records = []
    flee = {"outcome": None, "settled_out_of_battle": False, "terminal_unknown": False,
            "menu_stall": False, "turns": 0, "kos": 0, "invalid": 0, "actions": []}
    try:
        for i in range(int(args.battles)):
            r = _play(env, scenario, ALL_MACROS, seed=4242 + i)
            records.append(r)
            print(f"  battle {i + 1:2d}/{args.battles}: {str(r['outcome']):>20} "
                  f"| {r['turns']:2d} turns | {r['kos']} KO | {r['invalid']} invalid"
                  f" | moves@start={r.get('readable_moves_at_start', '?')}"
                  + ("  TERMINAL_UNKNOWN" if r["terminal_unknown"] else "")
                  + ("  MENU_STALL" if r["menu_stall"] else "")
                  + (f"  ERR:{r['error']}" if r.get("error") else ""), flush=True)
        flee = _play(env, scenario, ALL_MACROS, seed=9999, force_run=True)
        print(f"  flee test: {flee['outcome']} "
              f"(settled_out_of_battle={flee['settled_out_of_battle']})", flush=True)
    finally:
        env.close()

    wins = sum(1 for r in records if r["outcome"] == "win")
    total_ko = sum(r["kos"] for r in records)
    checks = {
        "wins_detected": f"{wins}/{len(records)}",
        "win_min_ok": wins >= min(WIN_MIN, len(records)),
        "no_terminal_unknown": not any(r["terminal_unknown"] for r in records + [flee]),
        "no_menu_stall": not any(r["menu_stall"] for r in records + [flee]),
        "no_episode_hit_max_turns": all(r["turns"] < MAX_TURNS for r in records + [flee]),
        "kos_match_wins": total_ko == wins,
        "flee_detected": flee["outcome"] == "fled",
        "flee_settled_out_of_battle": bool(flee["settled_out_of_battle"]),
    }
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    report = {
        "overall": "PASS" if ok else "FAIL",
        "scenario_id": scenario.get("id"),
        "battles": len(records), "wins": wins, "total_enemy_kos": total_ko,
        "checks": checks,
        "records": records, "flee": flee,
        "note": ("deliberate-wipe not tested: no wipe-capable scenario "
                 "(Route-1 wild cannot KO the party). Capture a weakened-party "
                 "or trainer scenario to cover it."),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(out_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(checks, indent=2))
    print(f"\nRESULT: {report['overall']}  ->  {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
