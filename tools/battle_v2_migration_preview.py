#!/usr/bin/env python3
"""Print the Battle v1 -> v2 (Catch-v2) migration preview (spec §14).

APPLIES NOTHING. It reports:
  * the untouched v1 champion (stays the live combat fallback),
  * the new, fully-separate v2 files,
  * what is archived (nothing — v1 is not moved),
  * what is untouched,
  * the live schema now and after v2 passes,
  * the expected obs / action dims for both schemas.

    PYTHONPATH=src python tools/battle_v2_migration_preview.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_train import battle_migration_preview        # noqa: E402


def main():
    p = battle_migration_preview()
    print(json.dumps(p, indent=2))
    print("\nSUMMARY")
    print(f"  applies anything: {p['applies_anything']}")
    print(f"  v1 champion:      {p['current_v1_champion']['path']} "
          f"(exists={p['current_v1_champion']['exists']}, stays live)")
    print(f"  archived:         {p['archived_files'] or 'nothing'}")
    print(f"  live schema:      {p['live_schema_now']} -> "
          f"{p['live_schema_after_v2_passes']}")
    print(f"  v1 dims:          {p['expected_dims']['v1']}")
    print(f"  v2 dims:          {p['expected_dims']['v2']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
