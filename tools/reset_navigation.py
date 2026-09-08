#!/usr/bin/env python3
"""Reset the navigation models.

Default: navigation learner <- navigation champion (soft reset). Battle models,
curriculum, savestates and exploration memory are untouched.

``--fresh-obs-schema``: the navigation observation width changed
(nav_obs_v3_directed, NAV_DIM 36). Every old nav .zip is structurally
incompatible, so this archives and DELETES champion / learner / resume /
candidate / latest / champion_score / model_version / nav_horizon /
navigation_stats / trainer_status / shaping_state, then the trainer builds a
brand-new PPO on next start. Savestates, curriculum, exploration memory and the
directed movement graph (``movement_graph_v1.json``) are preserved. The
pre-reset Route-1 reach rate is recorded in the backup as the bar the new
champion must not regress below.

Dry-run by default. Pass --apply to actually do it (backup happens on --apply).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2.reset_tools import plan_navigation_reset, apply_plan  # noqa: E402

RUNTIME = os.path.join(ROOT, "runtime")
NAV_CKPT = os.path.join(RUNTIME, "navigation", "checkpoints")
NAV_STATS = os.path.join(RUNTIME, "navigation")


def _backup(actions, dry_run):
    dest = os.path.join(NAV_STATS, "backups", time.strftime("%Y%m%d_%H%M%S"))
    present = [p for _, p in actions if os.path.isfile(p)]
    reach = None
    try:
        with open(os.path.join(NAV_STATS, "nav_horizon.json")) as f:
            reach = (json.load(f).get("transition_rates") or {}).get("1")
    except Exception:
        pass
    if not present:
        return {"backup_dir": None, "copied": [], "route1_reach_rate": reach}
    if not dry_run:
        os.makedirs(dest, exist_ok=True)
        for p in present:
            shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
        with open(os.path.join(dest, "RESET_INFO.json"), "w") as f:
            json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "route1_reach_rate_baseline": reach,
                       "archived": [os.path.basename(p) for p in present]}, f, indent=2)
    return {"backup_dir": dest, "copied": [os.path.basename(p) for p in present],
            "route1_reach_rate": reach}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually perform the reset (default: dry-run)")
    ap.add_argument("--fresh-obs-schema", action="store_true",
                    help="obs width changed: fresh PPO, keep savestates + graph")
    args = ap.parse_args()
    plan = plan_navigation_reset(nav_ckpt_dir=NAV_CKPT, nav_stats_dir=NAV_STATS,
                                 fresh_obs_schema=args.fresh_obs_schema)
    backup = _backup(plan["actions"], dry_run=not args.apply)
    result = apply_plan(plan, dry_run=not args.apply)
    manifest_note = None
    if args.fresh_obs_schema:
        mf = os.path.join(RUNTIME, "model_manifest.json")
        try:
            from pokemon_env import PokemonFireRedEnv
            want = PokemonFireRedEnv.NAV_OBS_SCHEMA
            with open(mf) as f:
                m = json.load(f)
            cur = (m.get("obs_schemas") or {}).get("navigation")
            manifest_note = f"navigation obs_schema {cur} -> {want}"
            if not args.apply:
                manifest_note += "  (dry-run)"
            else:
                m.setdefault("obs_schemas", {})["navigation"] = want
                tmp = mf + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(m, f, indent=2)
                os.replace(tmp, mf)
        except Exception as exc:
            manifest_note = f"manifest update skipped: {exc}"
    print(json.dumps({
        "mode": "fresh-obs-schema" if args.fresh_obs_schema else "soft",
        "backup": backup,
        "manifest": manifest_note,
        "plan": {k: v for k, v in plan.items() if k != "actions"},
        "will_touch": [os.path.relpath(p, ROOT) for _, p in plan["actions"]],
        "result": result,
    }, indent=2, default=str))
    if not args.apply:
        print("\n(dry-run - pass --apply to execute; backup happens on --apply)")


if __name__ == "__main__":
    main()
