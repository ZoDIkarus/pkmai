#!/usr/bin/env python3
"""Score collected in-battle RAM dumps against the candidate battle addresses.

    python tools/battle_dump_score.py runtime/ram_probe/<timestamp>/

Loads every ``*.ram.gz`` + ``*.json`` sidecar and runs each candidate offset
through ``twoby2.ram_battle_probe.verify_all``.

Exit code:
  0  every MANDATORY field is VERIFIED  ->  ready for the next activation step
  1  at least one mandatory field is NOT verified  ->  still blocked

Nothing is written and no address is adopted; the candidates are
reference / integration guesses only.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2 import ram_battle_probe as rp  # noqa: E402


def load_dumps(d):
    dumps = []
    for jf in sorted(glob.glob(os.path.join(d, "*.json"))):
        rf = jf[:-5] + ".ram.gz"
        if not os.path.exists(rf):
            continue
        with gzip.open(rf, "rb") as f:
            ram = f.read()
        meta = json.load(open(jf))
        meta["ram"] = ram
        dumps.append(meta)
    return dumps


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    dumps = load_dumps(d)
    print(f"loaded {len(dumps)} dumps from {d}")
    in_battle = [x for x in dumps if rp.dump_is_in_battle(x)]
    encs = rp.distinct_encounters(in_battle)
    oob = [x for x in dumps if x.get("battle_kind") == "out_of_battle"]
    print(f"  in-battle dumps: {len(in_battle)}   distinct encounters: {sorted(encs)}")
    print(f"  out-of-battle dumps: {len(oob)}")

    results = rp.verify_all(dumps)
    print()
    for field, r in results.items():
        tag = "MANDATORY" if r["mandatory"] else "optional "
        v = "VERIFIED  " if r["verified"] else "not verified"
        off = r["winning_offset"] or "-"
        print(f"  [{tag}] {field:24} {v}  offset={off}")
        if not r["verified"]:
            for a in r["attempts"]:
                print(f"      {a['candidate_offset']}: {a.get('reason','')}")

    ok = rp.all_mandatory_verified(results)
    print()
    if ok:
        print("ALL MANDATORY FIELDS VERIFIED — ready for the next activation step.")
        sys.exit(0)
    missing = [f for f in rp.MANDATORY_FIELDS if not results[f]["verified"]]
    print(f"STILL BLOCKED — not verified: {missing}")
    print("Collect more (see docs/RAM_PROBE_GUIDE.md): >= 2 distinct encounters "
          "(wild + trainer), labelled main/move/party/out-of-battle, a switch, "
          "an enemy KO, the full party in every dump.")
    sys.exit(1)


if __name__ == "__main__":
    main()
