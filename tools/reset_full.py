#!/usr/bin/env python3
"""Reset BOTH model systems (navigation + battle).

Requires an extra explicit confirmation flag (--i-really-mean-full-reset) on
top of --apply. Savestates, curriculum state, exploration memory and the
battle scenario pool are ALWAYS preserved.

Dry-run by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2.reset_tools import plan_full_reset, apply_plan  # noqa: E402

RUNTIME = os.path.join(ROOT, "runtime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--i-really-mean-full-reset", action="store_true",
                    dest="confirm")
    args = ap.parse_args()
    plan = plan_full_reset(
        nav_ckpt_dir=os.path.join(RUNTIME, "navigation", "checkpoints"),
        nav_stats_dir=os.path.join(RUNTIME, "navigation"),
        battle_ckpt_dir=os.path.join(RUNTIME, "battle", "checkpoints"),
        battle_stats_dir=os.path.join(RUNTIME, "battle"),
        confirm_full=args.confirm)
    print(json.dumps({"plan": {k: v for k, v in plan.items() if k != "actions"},
                      "result": apply_plan(plan, dry_run=not (args.apply and args.confirm))},
                     indent=2, default=str))
    if not (args.apply and args.confirm):
        print("\n(dry-run — needs BOTH --apply and --i-really-mean-full-reset)")


if __name__ == "__main__":
    main()
