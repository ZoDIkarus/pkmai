#!/usr/bin/env python3
"""Reset ONLY the battle learner/candidate + battle stats.

battle learner <- battle champion (or the verified rule fallback if no PPO
champion exists yet). The scenario pool is KEPT unless --wipe-scenarios.
Navigation models, curriculum, savestates and exploration memory are untouched.

The current battle learner/resume/stats are copied to
``runtime/battle/backups/<ts>/`` first (the learner trained on the broken end
signals - kept as a backup, not resumed).

Dry-run by default. Pass --apply to actually do it.
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

from twoby2.reset_tools import plan_battle_reset, apply_plan  # noqa: E402

RUNTIME = os.path.join(ROOT, "runtime")
BATTLE_DIR = os.path.join(RUNTIME, "battle")
BATTLE_CKPT = os.path.join(BATTLE_DIR, "checkpoints")
BATTLE_STATS = BATTLE_DIR

_BACKUP_FILES = (
    os.path.join(BATTLE_CKPT, "battle_learner.zip"),
    os.path.join(BATTLE_CKPT, "battle_resume.zip"),
    os.path.join(BATTLE_CKPT, "battle_candidate.zip"),
    os.path.join(BATTLE_DIR, "battle_stats.json"),
)


def _backup(dry_run):
    dest = os.path.join(BATTLE_DIR, "backups", time.strftime("%Y%m%d_%H%M%S"))
    present = [p for p in _BACKUP_FILES if os.path.isfile(p)]
    if not present:
        return {"backup_dir": None, "copied": [], "note": "nothing to back up"}
    if not dry_run:
        os.makedirs(dest, exist_ok=True)
        for p in present:
            shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
    return {"backup_dir": dest, "copied": [os.path.basename(p) for p in present]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--wipe-scenarios", action="store_true",
                    help="also delete the battle scenario pool (default: keep)")
    args = ap.parse_args()
    backup = _backup(dry_run=not args.apply)
    plan = plan_battle_reset(battle_ckpt_dir=BATTLE_CKPT,
                             battle_stats_dir=BATTLE_STATS,
                             wipe_scenarios=args.wipe_scenarios)
    print(json.dumps({"backup": backup,
                      "plan": {k: v for k, v in plan.items() if k != "actions"},
                      "result": apply_plan(plan, dry_run=not args.apply)},
                     indent=2, default=str))
    if not args.apply:
        print("\n(dry-run — pass --apply to execute; backup happens on --apply)")


if __name__ == "__main__":
    main()
