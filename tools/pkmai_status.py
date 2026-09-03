#!/usr/bin/env python3
"""PKMAI Schnellstatus - liest die Runtime-JSONs und fasst sie kompakt zusammen.

Aufruf:
    python tools/pkmai_status.py           # aktualisiert automatisch alle 5s
    python tools/pkmai_status.py --once     # nur einmal ausgeben
    python tools/pkmai_status.py -n 10       # Intervall auf 10s setzen
Beenden mit Ctrl+C.
"""
import glob
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "runtime")


def load(*parts):
    try:
        with open(os.path.join(RT, *parts)) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def fmt(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def render():
    ts = load("trainer_status.json")
    cs = load("champion_score.json")
    gp = load("exploration_memory", "global_progress.json")
    sv = load("skill_vault_scores.json")
    m = cs.get("metrics", {})

    depth = int(gp.get("max_episode_maps", 0))
    wall = "  <-- WALL (Route 1 -> Vertania)" if depth <= 2 else "  <-- offen!"

    out = []
    out.append("=" * 60)
    out.append(f"  {datetime.now():%H:%M:%S}")
    out.append(f"  Champion      v{cs.get('version', 0):06d}   @ {fmt(cs.get('timesteps', 0))} steps")
    out.append(f"  Learner       {fmt(ts.get('learner_steps', 0))} steps"
               f"   (+{fmt(ts.get('delta_steps', 0))} seit Champion)")
    out.append(f"  Phase         {ts.get('training_phase', '?')}   "
               f"strikes={ts.get('regression_strikes', 0)}  rollbacks={ts.get('rollback_count', 0)}")
    out.append("-" * 60)
    out.append(f"  WELT-TIEFE    Aussen-Maps = {depth}{wall}")
    out.append(f"  Champion      maps={m.get('max_maps', 0)}  level={m.get('max_level', 0)}  "
               f"badges={m.get('max_badges', 0)}")
    out.append(f"  Full-Raten    Starter {m.get('full_starter_permille', 0)/10:.1f}%   "
               f"Treppe {m.get('full_stairs_permille', 0)/10:.1f}%   "
               f"Exit {m.get('full_exit_permille', 0)/10:.1f}%   "
               f"full_eps={m.get('full_episodes', 0)}")
    out.append(f"  Skill-Vault   " + "  ".join(f"{k}:{v}" for k, v in sv.items()))

    roles = Counter()
    maps_seen = Counter()
    for fp in glob.glob(os.path.join(RT, "instances_data", "*.json")):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        roles[d.get("training_objective") or d.get("agent_role") or "?"] += 1
        maps_seen[f"{d.get('bank')},{d.get('map')}"] += 1
    if roles:
        out.append("-" * 60)
        out.append("  Rollen        " + "  ".join(f"{k}:{v}" for k, v in sorted(roles.items())))
        out.append("  Agenten @Map  " + "  ".join(f"{k}:{v}" for k, v in maps_seen.most_common(6)))

    outdoor = sorted(glob.glob(os.path.join(RT, "curriculum_shared", "outdoor_*.state.gz")))
    if outdoor:
        deepest = max(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in outdoor)
        out.append(f"  Checkpoints   tiefster outdoor_{deepest}  ({len(outdoor)} Aussen-Savestates)")
    out.append("=" * 60)
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    once = "--once" in args or "-1" in args
    interval = 5
    if "-n" in args:
        try:
            interval = int(args[args.index("-n") + 1])
        except Exception:
            pass

    if once:
        print(render())
        return

    try:
        body = render()
        last_pull = time.time()
        while True:
            now = time.time()
            if now - last_pull >= interval:
                body = render()
                last_pull = now
            remaining = max(0, int(round(interval - (now - last_pull))))
            bar = "#" * (interval - remaining) + "-" * remaining
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(
                body
                + f"\n\n  naechste Aktualisierung in {remaining}s  [{bar}]"
                + "   (Ctrl+C beendet)\n"
            )
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
