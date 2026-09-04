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

    depth = int(gp.get("max_world_stage", 0))
    stage_names = {
        0: "Start/innen", 1: "Alabastia", 2: "Route 1",
        3: "Vertania", 4: "Eichs Paket", 5: "Pokédex/Paket abgegeben",
        6: "Route 2", 7: "Vertania-Wald", 8: "Marmoria",
        9: "erster Orden",
    }
    wall = "  <-- Route-1-Wand" if depth <= 2 else ""

    stats_rows = []
    for fp in glob.glob(os.path.join(RT, "training_stats", "agent_*.json")):
        try:
            with open(fp) as f:
                stats_rows.append(json.load(f) or {})
        except Exception:
            continue

    instances = {}
    watcher = {}
    for fp in glob.glob(os.path.join(RT, "instances_data", "inst_*.json")):
        try:
            with open(fp) as f:
                data = json.load(f) or {}
            agent_id = int(data.get("id", -1))
        except Exception:
            continue
        if agent_id == 120:
            watcher = data
        elif 0 <= agent_id < 32:
            instances[agent_id] = data

    out = []
    out.append("=" * 108)
    out.append(f"  {datetime.now():%H:%M:%S}")
    out.append(f"  Champion      v{cs.get('version', 0):06d}   @ {fmt(cs.get('timesteps', 0))} steps")
    out.append(f"  Learner       {fmt(ts.get('learner_steps', 0))} steps"
               f"   (+{fmt(ts.get('delta_steps', 0))} seit Champion)")
    out.append(f"  Phase         {ts.get('training_phase', '?')}   "
               f"strikes={ts.get('regression_strikes', 0)}  rollbacks={ts.get('rollback_count', 0)}")
    out.append("-" * 108)
    out.append(f"  WELT-STUFE    {depth}: {stage_names.get(depth, '?')}{wall}")
    out.append(f"  Champion      maps={m.get('max_maps', 0)}  level={m.get('max_level', 0)}  "
               f"badges={m.get('max_badges', 0)}")
    out.append(f"  Full-Raten    Starter {m.get('full_starter_permille', 0)/10:.1f}%   "
               f"Treppe {m.get('full_stairs_permille', 0)/10:.1f}%   "
               f"Exit {m.get('full_exit_permille', 0)/10:.1f}%   "
               f"full_eps={m.get('full_episodes', 0)}")
    out.append(f"  Skill-Vault   " + "  ".join(f"{k}:{v}" for k, v in sv.items()))

    bt = {"battles_started": 0, "battles_completed": 0,
          "enemy_faints": 0, "experience_wins": 0,
          "enemy_damage_hp": 0, "v8_battle_success": 0}
    for _d in stats_rows:
        for _k in bt:
            bt[_k] += int(_d.get(_k, 0) or 0)
    out.append(
        f"  KAEMPFE       gestartet {bt['battles_started']}   "
        f"Gegner-K.O. {max(bt['enemy_faints'], bt['experience_wins'])}   "
        f"Schaden-HP {bt['enemy_damage_hp']}   "
        f"gewonnen {bt['v8_battle_success']}"
    )

    if watcher:
        wbs = watcher.get("battle_stats") or {}
        out.append("-" * 108)
        out.append(
            "  WATCHER       "
            f"steps={fmt(watcher.get('steps', 0))}  "
            f"skill={watcher.get('active_skill', '?')}  "
            f"model={watcher.get('loaded_model', '?')}  "
            f"map={watcher.get('bank')}/{watcher.get('map')} "
            f"@{watcher.get('x')},{watcher.get('y')}  "
            f"level={watcher.get('level', 0)}  "
            f"battle={int(bool(watcher.get('in_battle', 0)))}"
        )
        out.append(
            "                "
            f"Kämpfe {int(wbs.get('started', 0))}/{int(wbs.get('completed', 0))}  "
            f"EP +{int((watcher.get('exp_stats') or {}).get('gained_total', 0))}  "
            f"Reward {watcher.get('reward', 0)}  "
            f"Input {watcher.get('input', '?')}"
        )

    # ---- SHIGGY / STARTER-SKILL (Phase-4 -> Phase-5 Gate) ----
    st_health = (ts.get("live_skill_health") or {}).get("starter") or {}
    st_vault = int(sv.get("starter", 0) or 0)
    st_eff = int((ts.get("effective_skill_scores") or {}).get("starter", 0) or 0)
    st_runs = sum(int(s.get("v8_starter_episodes", 0) or 0) for s in stats_rows)
    st_wins = sum(int(s.get("v8_starter_success", 0) or 0) for s in stats_rows)
    st_agents = [
        d for d in instances.values()
        if (d.get("training_objective") or d.get("agent_role")) == "starter"
    ]
    n_have = sum(1 for d in st_agents if d.get("has_target_starter"))
    lab_max = max(
        (int((d.get("story_progress") or {}).get("pallet_oaks_lab_scene", 0) or 0)
         for d in instances.values()),
        default=0,
    )
    to_counter = Counter(
        d.get("last_stage_timeout") or "-"
        for d in st_agents if d.get("last_stage_timeout")
    )
    DONE = 880
    gate = "OFFEN" if st_vault >= DONE else f"ZU (fehlen {DONE - st_vault})"
    st_pct = (100 * st_wins / st_runs) if st_runs else 0.0
    out.append("-" * 108)
    out.append(
        f"  SHIGGY/STARTER Vault {st_vault}/{DONE}   "
        f"live {int(st_health.get('score', 0) or 0)} ({int(st_health.get('episodes', 0) or 0)} eps)   "
        f"effektiv {st_eff}   ->Phase5: {gate}"
    )
    out.append(
        f"                Lifetime {st_wins}/{st_runs} ({st_pct:.1f}%)   "
        f"Champion Full-Starter {m.get('full_starter_permille', 0)/10:.1f}%   "
        f"Lab-Szene max {lab_max}/6"
    )
    out.append(
        f"                Live-Agenten {len(st_agents)}  mit Schiggy {n_have}   "
        f"Timeouts {dict(to_counter.most_common(4)) if to_counter else '-'}"
    )

    roles = Counter()
    maps_seen = Counter()
    role_live = {}
    for d in instances.values():
        role = d.get("training_objective") or d.get("agent_role") or "?"
        roles[role] += 1
        maps_seen[f"{d.get('bank')},{d.get('map')}"] += 1
        row = role_live.setdefault(role, {
            "agents": 0, "active": 0, "started": 0, "completed": 0,
            "ko": 0, "damage": 0, "level": 0, "stage": 0,
            "steps": 0,
        })
        bs = d.get("battle_stats") or {}
        row["agents"] += 1
        row["active"] += int(bool(d.get("in_battle", 0)))
        row["started"] += int(bs.get("episode_started", 0) or 0)
        row["completed"] += int(bs.get("episode_completed", 0) or 0)
        row["ko"] += int(bs.get("enemy_faints", 0) or 0)
        row["damage"] += int(bs.get("enemy_damage_hp", 0) or 0)
        row["level"] = max(row["level"], int(d.get("level", 0) or 0))
        row["stage"] = max(row["stage"], int(d.get("world_stage", 0) or 0))
        row["steps"] += int(d.get("steps", 0) or 0)

    lifetime_keys = {
        "intro": ("v2_intro_episodes", "v2_intro_success"),
        "stairs": ("v2_stairs_episodes", "v2_stairs_success"),
        "exit": ("v2_exit_episodes", "v2_exit_success"),
        "starter": ("v8_starter_episodes", "v8_starter_success"),
        "battle": ("v8_battle_episodes", "v8_battle_success"),
        "level": ("v8_level_episodes", "v8_level_success"),
        "badge": ("v8_badge_episodes", "v8_badge_success"),
        "progress": ("v7_progress_episodes", "v7_progress_badge1"),
        "full": ("v2_full_episodes", "v7_full_badge1"),
    }
    if roles:
        out.append("-" * 108)
        out.append("  Rollen        " + "  ".join(f"{k}:{v}" for k, v in sorted(roles.items())))
        out.append("  Agenten @Map  " + "  ".join(f"{k}:{v}" for k, v in maps_seen.most_common(6)))
        out.append("  KATEGORIEN    n  aktiv   Kämpfe(Episode)   K.O.  Schaden  maxLvl  Welt  ØSteps  Runs/Erfolg")
        for role in sorted(role_live):
            row = role_live[role]
            ep_key, ok_key = lifetime_keys.get(role, (None, None))
            runs = sum(int(s.get(ep_key, 0) or 0) for s in stats_rows) if ep_key else 0
            wins = sum(int(s.get(ok_key, 0) or 0) for s in stats_rows) if ok_key else 0
            out.append(
                f"  {role:<12} {row['agents']:>2} {row['active']:>5}   "
                f"{row['started']:>5}/{row['completed']:<5}      "
                f"{row['ko']:>3}  {row['damage']:>7}  "
                f"{row['level']:>6}  {row['stage']:>4}  "
                f"{round(row['steps']/row['agents']) if row['agents'] else 0:>6}  {runs}/{wins}"
            )

    stages = sorted(glob.glob(os.path.join(RT, "curriculum_shared", "stage_*.state.gz")))
    if stages:
        deepest = max(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in stages)
        out.append(f"  Checkpoints   tiefster stage_{deepest}  ({len(stages)} validierte Stufen)")
    out.append("=" * 108)
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
