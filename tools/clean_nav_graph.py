#!/usr/bin/env python3
"""Clean the directed movement graph of PROVABLY-WRONG block / ledge entries.

Root cause of the pollution (see git log 2026-09-08):
  * the overworld location is SAMPLED (every few agent steps), so a stretch of
    continuous walking was stored as a one-way ``jump_or_ledge`` -> ~1400 fake
    ledges;
  * a direction that produced no move DURING a battle transition / warp fade /
    walk animation was stored as ``blocked`` and the fleet ``visit_id`` union
    promoted it to a permanent ``blocked_static`` -> ~2500 fake walls, which
    walled off the Route-1 corridor and made ``target_valid=false``.

``DirectedNavGraph.sanitize()`` removes ONLY entries that are contradicted by
real evidence (a confirmed reverse walk, a legacy adjacency, a walk edge for
the same tile+action). It never deletes ``walk`` edges or legacy ``unknown``
edges. A timestamped backup is written before anything changes.

    PYTHONPATH=src python tools/clean_nav_graph.py            # dry-run
    PYTHONPATH=src python tools/clean_nav_graph.py --apply    # write + backup
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import nav_graph                                      # noqa: E402

GRAPH = os.path.join(ROOT, "runtime", "navigation", "movement_graph_v1.json")

# real Route-1 / Pallet targets (from runtime/curriculum_v20/known_transitions.json)
ROUTE1 = (3, 19)
ROUTE1_EXIT = (10, 0)
PALLET = (3, 0)
PALLET_EXIT = (12, 0)


def _route_coverage(g, mp, exit_tile, *, n=250, seed=1):
    rng = random.Random(seed)
    xs = [f for f in g._edges.get(mp, {})]
    if not xs:
        return 0.0, 0
    tiles = list({frm for (frm, _a) in g._edges.get(mp, {})})
    rng.shuffle(tiles)
    tiles = tiles[:n]
    hit = sum(1 for t in tiles
              if g.directed_bfs(mp, t, [exit_tile], now_step=5_000_000) is not None)
    return (hit / len(tiles) if tiles else 0.0), len(tiles)


def _confirmed_coverage(g, mp, exit_tile, *, n=250, seed=2):
    rng = random.Random(seed)
    tiles = list({frm for (frm, _a) in g._edges.get(mp, {})})
    rng.shuffle(tiles)
    tiles = tiles[:n]
    hit = sum(1 for t in tiles
              if g.directed_bfs(mp, t, [exit_tile], now_step=5_000_000,
                                allow_legacy=False) is not None)
    return (hit / len(tiles) if tiles else 0.0), len(tiles)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the cleaned graph")
    ap.add_argument("--graph", default=GRAPH)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.graph):
        print(f"graph not found: {args.graph}")
        return 2

    g = nav_graph.DirectedNavGraph.load(args.graph)
    before = {
        "edges": g.edge_count(),
        "ledges": sum(1 for m in g._edges.values() for e in m.values()
                      if e.kind == nav_graph.JUMP),
        "blocks": sum(len(v) for v in g._blocks.values()),
        "blocked_static": sum(1 for m in g._blocks.values() for b in m.values()
                              if b.kind == nav_graph.BLOCKED_STATIC),
    }
    r1_leg_b, r1_n = _route_coverage(g, ROUTE1, ROUTE1_EXIT)
    r1_conf_b, _ = _confirmed_coverage(g, ROUTE1, ROUTE1_EXIT)
    pl_leg_b, pl_n = _route_coverage(g, PALLET, PALLET_EXIT)

    rep = g.sanitize()

    after = {
        "edges": g.edge_count(),
        "ledges": sum(1 for m in g._edges.values() for e in m.values()
                      if e.kind == nav_graph.JUMP),
        "blocks": sum(len(v) for v in g._blocks.values()),
        "blocked_static": sum(1 for m in g._blocks.values() for b in m.values()
                              if b.kind == nav_graph.BLOCKED_STATIC),
    }
    r1_leg_a, _ = _route_coverage(g, ROUTE1, ROUTE1_EXIT)
    r1_conf_a, _ = _confirmed_coverage(g, ROUTE1, ROUTE1_EXIT)
    pl_leg_a, _ = _route_coverage(g, PALLET, PALLET_EXIT)

    out = {
        "graph": args.graph,
        "applied": False,
        "sanitize_report": rep,
        "before": before, "after": after,
        "route1_to_(10,0)": {
            "legacy_route_coverage": [round(r1_leg_b, 3), round(r1_leg_a, 3)],
            "confirmed_route_coverage": [round(r1_conf_b, 3), round(r1_conf_a, 3)],
            "tiles_sampled": r1_n,
        },
        "pallet_to_(12,0)": {
            "legacy_route_coverage": [round(pl_leg_b, 3), round(pl_leg_a, 3)],
            "tiles_sampled": pl_n,
        },
    }

    if args.apply:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bdir = os.path.join(os.path.dirname(os.path.abspath(args.graph)),
                            "backups", ts)
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(args.graph, os.path.join(bdir, "movement_graph_v1.json"))
        with open(os.path.join(bdir, "clean_nav_graph_report.json"), "w") as f:
            json.dump(out, f, indent=2)
        g.save(args.graph)
        out["applied"] = True
        out["backup_dir"] = bdir

    print(json.dumps(out, indent=2))
    if not args.apply:
        print("\n(dry-run — nothing written. Re-run with --apply to clean + back up.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
