#!/usr/bin/env python3
"""Compact live status for the active 2x2 Navigation/Battle architecture."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "runtime")
WIDTH = 86


def load(*parts):
    try:
        with open(os.path.join(RT, *parts)) as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def fmt(value):
    try:
        return f"{int(value):,}".replace(",", ".")
    except Exception:
        return str(value)


def _instances():
    runners, watcher = [], {}
    for path in glob.glob(os.path.join(RT, "instances_data", "inst_*.json")):
        try:
            with open(path) as f:
                row = json.load(f) or {}
            aid = int(row.get("id", -1))
        except Exception:
            continue
        if aid == 120:
            watcher = row
        elif 0 <= aid < 40:
            runners.append(row)
    return sorted(runners, key=lambda r: int(r.get("id", 0))), watcher


def render():
    nav = load("navigation", "trainer_status.json")
    nav_champ = load("navigation", "champion_score.json")
    battle = load("battle", "battle_stats.json")
    bc = battle.get("counters") or {}
    scenarios = load("battle", "scenarios", "index.json").get("scenarios") or []
    runners, watcher = _instances()
    metrics = nav.get("last_eval_metrics") or nav_champ.get("metrics") or {}

    nav_steps = int(nav.get("learner_steps", 0) or 0)
    champ_steps = int(nav.get("champion_steps", 0)
                      or nav_champ.get("timesteps", 0) or 0)
    nav_version = int(nav.get("champion_version", 0)
                      or nav_champ.get("version", 0) or 0)
    battle_version = int(battle.get("champion_version", 0) or 0)
    battle_learner_version = int(battle.get("learner_version", 0) or 0)
    battle_next_check = int(battle.get("next_promotion_step", 0) or 0)
    battle_phase = str(battle.get("phase") or "training")
    areas = sorted({str(s.get("area") or "unknown") for s in scenarios})

    in_battle = sum(bool(r.get("in_battle")) for r in runners)
    maps = max((int(r.get("visited_maps", 0) or 0)
                for r in runners), default=0)
    max_stage = max((int(r.get("world_stage", 0) or 0) for r in runners),
                    default=0)

    lines = ["=" * WIDTH,
             f"  PKMai 2x2 STATUS  ·  {datetime.now():%H:%M:%S}",
             "=" * WIDTH,
             "  🧠 NAVIGATION BRAIN · LIVE",
             f"     State: {str(nav.get('training_phase') or 'training').upper()}  |  40 FULL Agents vom Master-Start",
             f"     Learner:  {fmt(nav_steps)} Steps",
             f"     Champion: v{nav_version:06d} @ {fmt(champ_steps)} Steps",
             f"     Seit Champion: +{fmt(int(nav.get('delta_steps', nav_steps-champ_steps) or 0))}",
             f"     Progress: Stage {int(metrics.get('max_stage', 0) or 0)}  ·  Maps {int(metrics.get('max_maps', 0) or 0)}  ·  Orden {int(metrics.get('max_badges', 0) or 0)}  ·  FULL Runs {int(nav.get('recent_full_done', 0) or 0)}",
             f"     Letzte Prüfung: {str(nav.get('last_eval_result') or 'noch keine')} @ {fmt(nav.get('last_eval_at_step', 0))}",
             "-" * WIDTH,
             "  ⚔️  BATTLE BRAIN · LIVE",
             f"     State: {battle_phase.upper().replace('_', ' ')}  |  {int(battle.get('n_workers', 9) or 9)} lernende Fighter",
             f"     Training: Learner v{battle_learner_version}  ·  {fmt(bc.get('env_steps', 0))} Battle-Steps  ·  {fmt(bc.get('ppo_updates', 0))} PPO-Updates",
             "     Live:     Champion " + (f"v{battle_version}" if battle_version else "RULE FALLBACK (noch kein PPO-Champion)"),
             f"     Episoden {fmt(bc.get('episodes', 0))}  ·  Siege {fmt(bc.get('wins', 0))}  ·  K.O. {fmt(bc.get('kos', 0))}  ·  Wipes {fmt(bc.get('wipes', 0))}",
             f"     Fluchten {fmt(bc.get('flees', 0))}  ·  Timeouts/ohne Ergebnis {fmt(bc.get('timeouts', 0))}",
             f"     Szenarien: {len(scenarios)}  ·  Gebiete: {', '.join(areas) if areas else 'noch keine'}"]
    if battle_phase == "champion_evaluation":
        lines.append("     Nächste Prüfung: läuft gerade (40 echte Kämpfe; Trainings-Rollouts pausieren).")
    elif battle_next_check:
        lines.append(f"     Nächste Prüfung: bei {fmt(battle_next_check)} Battle-Steps (dann 40 echte Kämpfe)")

    lines.extend(["-" * WIDTH,
                  "  🧭 FULL FLEET",
                  f"     Telemetrie: {len(runners)}/40  ·  im Kampf: {in_battle}  ·  aktuelle Best-Stage: {max_stage}  ·  max. besuchte Maps: {maps}"])
    if watcher:
        lines.extend(["-" * WIDTH,
                      "  👁️  FULL WATCHER (Inference, lernt nicht)",
                      f"     Navigation Champion v{int(watcher.get('model_version', nav_version) or nav_version):06d}  ·  Ort {watcher.get('bank', '?')}/{watcher.get('map', '?')} @ {watcher.get('x', '?')},{watcher.get('y', '?')}  ·  Steps {fmt(watcher.get('steps', 0))}"])
    lines.extend(["=" * WIDTH, "  Ctrl-C beendet nur diese Statusanzeige."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            if not args.once:
                print("\033[2J\033[H", end="")
            print(render(), flush=True)
            if args.once:
                return
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        print("\nStatusanzeige beendet.")


if __name__ == "__main__":
    main()
