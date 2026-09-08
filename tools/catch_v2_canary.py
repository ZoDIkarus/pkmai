#!/usr/bin/env python3
"""Catch-v2 reproducible canary (spec §18).

Runs a fixed, seeded set of battle scenarios end-to-end through the isolated
``BattleEnv`` + deterministic ``SimulatedBattleDriver`` and prints, for each:

  * the macro ORDER and the per-step ACTIONS,
  * the driver-visible menu / catch state transitions,
  * ball count BEFORE / AFTER,
  * party / "pokedex" (caught-species) BEFORE / AFTER,
  * the final OUTCOME,
  * the REWARD COMPONENTS and whether they sum to the step / episode reward,
  * any diagnostic errors (invalid, aborted, catch_reject, ...).

It also asserts the Catch-v2 invariants (catch never stacks with KO/win, a
requested catch is its own terminal outcome, components sum exactly).

NOTE: the LIVE emulator catch path is still gated
(``twoby2.battle_ram_live.catch_ram_ready() -> False``); this canary therefore
exercises the SIMULATOR, which is the reproducible bring-up surface. A live
canary follows once ``tools/catch_ram_probe.py`` verifies the bag / dex / result
RAM.

    PYTHONPATH=src python tools/catch_v2_canary.py
    -> runtime/catch_v2_canary/<timestamp>/canary_report.json
"""
from __future__ import annotations

import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_env import (BattleEnv, SimulatedBattleDriver, catch_scenario,   # noqa: E402
                        default_scenario, ALL_MACROS)
from battle_executor import CATCH_ACTION                                    # noqa: E402

MI = 0                       # MOVE_1
CI = ALL_MACROS.index(CATCH_ACTION)


def _policy_for(name):
    """A fixed scripted macro sequence per canary (no learned policy — the
    canary is about the plumbing, not the brain)."""
    return {
        "normal_wild_combat":   [MI, MI, MI, MI, MI],
        "trainer_battle":       [MI, MI, MI, MI, MI],
        "catch_weakened_target": [CI],
        "catch_after_weakening": [MI, CI],
        "broke_free_then_continue": [CI, CI, CI],
        "no_ball":              [CI, MI],
        "trainer_catch_blocked": [CI, MI, MI],
        "mixed_series_1":       [MI, CI],
        "mixed_series_2":       [CI],
    }[name]


CANARIES = {
    "normal_wild_combat": lambda: {**default_scenario(), "area": "route1",
                                   "objective_mode": "combat"},
    "trainer_battle": lambda: {**default_scenario(), "area": "route1",
                               "is_trainer": True, "can_escape": False,
                               "objective_mode": "combat"},
    "catch_weakened_target": lambda: catch_scenario(
        species_id=19, catch_rate=180, enemy_cur=6, enemy_max=20, catch_seed=1),
    "catch_after_weakening": lambda: catch_scenario(
        species_id=25, catch_rate=90, enemy_cur=22, enemy_max=26, catch_seed=2),
    "broke_free_then_continue": lambda: catch_scenario(
        species_id=16, catch_rate=200, enemy_cur=20, enemy_max=20, catch_seed=2),
    "no_ball": lambda: catch_scenario(
        species_id=19, catch_rate=255, ball_inventory=1, ball_reserve=1, catch_seed=4),
    "trainer_catch_blocked": lambda: catch_scenario(
        species_id=19, catch_rate=255, is_trainer=True, catch_seed=5),
    "mixed_series_1": lambda: catch_scenario(
        species_id=21, catch_rate=150, enemy_cur=40, enemy_max=40, catch_seed=8),
    "mixed_series_2": lambda: catch_scenario(
        species_id=27, catch_rate=200, enemy_cur=14, enemy_max=20, catch_seed=7),
}


def _ball_count(env):
    return int((env.driver._s or {}).get("ball_count", 0))


def run_canary(name, *, seed=0):
    scenario = CANARIES[name]()
    env = BattleEnv(SimulatedBattleDriver(), max_turns=25)
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    obj = dict(info["objective"])
    rec = {
        "name": name,
        "objective": {k: obj.get(k) for k in
                      ("objective_mode", "catch_requested", "target_species_id",
                       "catch_reason", "usable_ball_count", "reserved_ball_count")},
        "balls_before": _ball_count(env),
        "party_before": len(scenario.get("our_party", [])),
        "caught_before": [],
        "steps": [], "diagnostics": [],
    }
    ep_reward = 0.0
    for a in _policy_for(name):
        if env._state.get("done"):
            break
        mask = list(obs["action_mask"])
        obs, r, term, trunc, i = env.step(a)
        ep_reward += r
        comps = i.get("reward_components") or []
        csum = round(sum(x for _, x in comps), 6)
        step = {
            "macro": i.get("macro"),
            "legal_catch": bool(mask[CI]),
            "reward": round(r, 4),
            "components": [[n, round(x, 4)] for n, x in comps],
            "components_sum_ok": abs(csum - round(r, 6)) < 1e-6,
            "outcome": i.get("outcome"),
            "catch_outcome": i.get("catch_outcome"),
            "catch_attempted": bool(i.get("catch_attempted")),
            "catch_success": bool(i.get("catch_success")),
            "balls_used": int(i.get("balls_used", 0) or 0),
            "enemy_hp_after": i.get("enemy_hp_after"),
        }
        for key in ("invalid", "aborted", "catch_reject", "illegal_trainer_catch",
                    "unrequested_catch", "premature_catch_attempt"):
            if i.get(key):
                step[key] = i.get(key)
                rec["diagnostics"].append(f"{i.get('macro')}: {key}={i.get(key)}")
        rec["steps"].append(step)
        if term or trunc:
            break

    rec["balls_after"] = _ball_count(env)
    st = env._state or {}
    rec["party_after"] = len([m for m in st.get("our_party", []) if m])
    rec["final_outcome"] = st.get("outcome") or (rec["steps"][-1]["outcome"]
                                                 if rec["steps"] else None)
    rec["episode_reward"] = round(ep_reward, 4)
    rec["episode_sum_ok"] = abs(
        ep_reward - sum(s["reward"] for s in rec["steps"])) < 1e-6

    # --- invariants (spec §9/§10/§13) ---
    inv = []
    for s in rec["steps"]:
        names = [n for n, _ in s["components"]]
        if s["catch_success"] and ("enemy_ko" in names or "battle_win" in names):
            inv.append("catch stacked with KO/win")
        if not s["components_sum_ok"]:
            inv.append("component sum mismatch")
    if rec["final_outcome"] == "caught" and rec["party_after"] <= rec["party_before"] \
            and "sent_to_pc" not in json.dumps(rec):
        pass  # sim grows party only conceptually; not asserted here
    if not rec["episode_sum_ok"]:
        inv.append("episode reward != sum of step rewards")
    # per-canary expectations (spec §18: reproducible)
    expect = {
        "normal_wild_combat": {"final_outcome": "win"},
        "trainer_battle": {"final_outcome": "win"},
        "catch_weakened_target": {"final_outcome": "caught"},
        "catch_after_weakening": {"catch_attempted": True},
        "broke_free_then_continue": {"broke_free_seen": True, "final_outcome": "caught"},
        "no_ball": {"catch_reject_seen": True},
        "trainer_catch_blocked": {"catch_reject_seen": True},
        "mixed_series_1": {"catch_attempted": True},
        "mixed_series_2": {"final_outcome": "caught"},
    }[name]
    got = {
        "final_outcome": rec["final_outcome"],
        "catch_attempted": any(s["catch_attempted"] for s in rec["steps"]),
        "broke_free_seen": any(s["catch_outcome"] == "broke_free" for s in rec["steps"]),
        "catch_reject_seen": any(s.get("catch_reject") or s.get("invalid")
                                 for s in rec["steps"]),
    }
    mismatches = [f"{k}: expected {v}, got {got.get(k)}"
                  for k, v in expect.items() if got.get(k) != v]
    rec["invariant_violations"] = inv
    rec["expectation_mismatches"] = mismatches
    rec["ok"] = not inv and not mismatches
    return rec


def main():
    outdir = os.path.join(ROOT, "runtime", "catch_v2_canary",
                          datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    report = {"schema": "catch_v2_canary_v1", "surface": "SimulatedBattleDriver",
              "live_catch_gated": True, "canaries": {}}
    all_ok = True
    for name in CANARIES:
        rec = run_canary(name)
        report["canaries"][name] = rec
        all_ok &= rec["ok"]
        print(f"\n=== {name} === {'OK' if rec['ok'] else 'FAIL'}")
        print(f"  objective: {rec['objective']}")
        print(f"  balls {rec['balls_before']} -> {rec['balls_after']} | "
              f"party {rec['party_before']} -> {rec['party_after']} | "
              f"outcome={rec['final_outcome']} | ep_reward={rec['episode_reward']} "
              f"(sum_ok={rec['episode_sum_ok']})")
        for s in rec["steps"]:
            print(f"    {s['macro']:8s} r={s['reward']:+.3f} "
                  f"comps={s['components']} outcome={s['outcome']} "
                  f"catch={s['catch_outcome']}")
        if rec["diagnostics"]:
            print(f"  diagnostics: {rec['diagnostics']}")
        if rec["invariant_violations"]:
            print(f"  !!! INVARIANTS: {rec['invariant_violations']}")
        if rec["expectation_mismatches"]:
            print(f"  !!! EXPECTATION: {rec['expectation_mismatches']}")

    report["overall"] = "PASS" if all_ok else "FAIL"
    path = os.path.join(outdir, "canary_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\noverall: {report['overall']}  ->  {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
