#!/usr/bin/env python3
"""PKMAI Schnellstatus - liest die Runtime-JSONs und fasst sie kompakt zusammen.

Aufruf:
    python tools/pkmai_status.py           # aktualisiert automatisch jede Sekunde
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
WIDTH = 82


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


def score(m):
    """Muss exakt zu MilestoneCheckpointCallback._score in src/train.py passen,
    damit ein Candidate ohne last_eval_result (alter Statusstand) genauso
    gegen den Champion bewertet werden kann wie beim Trainer-Neustart."""
    return (
        int(m.get("max_badges", 0)),
        int(m.get("max_stage", 0)),
        int(m.get("full_starter_permille", 0)),
        int(m.get("full_exit_permille", 0)),
        int(m.get("full_stairs_permille", 0)),
        int(m.get("full_intro_permille", 0)),
        int(m.get("max_level", 0)),
        int(m.get("max_maps", 0)),
        -int(m.get("full_best_stage_steps", 1_000_000) or 1_000_000),
    )


def load_history():
    try:
        with open(os.path.join(RT, "training_history.json")) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def brain_section(ts, cs, m):
    """Kompakte Antwort auf die Frage: lernt das Netz weiter?"""
    lines = []
    has_champion = os.path.exists(
        os.path.join(RT, "checkpoints", "pokemon_model_best.zip")
    )
    champion_label = (
        f"Champion v{cs.get('version', 0):06d} @ {fmt(cs.get('timesteps', 0))} Steps"
        if has_champion
        else "noch kein bestätigter Champion"
    )
    lines.append("  🧠 BRAIN")
    lines.append(f"     {champion_label}")
    lines.append(
        f"     Learner: {fmt(ts.get('learner_steps', 0))} Steps  "
        f"(+{fmt(ts.get('delta_steps', 0))} seit Champion)"
    )
    lines.append("     Beststand:")
    lines.append(f"       Orden:      {m.get('max_badges', 0)}")
    lines.append(f"       Weltstufe:  {m.get('max_stage', 0)}")
    lines.append(f"       Level:      {m.get('max_level', 0)}")
    lines.append(f"       Maps:       {m.get('max_maps', 0)}")
    lines.append(
        f"       Tempo:      {fmt(m.get('full_best_stage_steps', 0))} Weg-Steps"
    )

    last_eval = ts.get("last_eval_metrics") or {}
    last_eval_result = str(ts.get("last_eval_result", "") or "")
    champion_score = cs.get("score")
    if (
        last_eval
        and not last_eval_result
        and isinstance(champion_score, list)
        and score(last_eval) <= tuple(champion_score)
    ):
        # Alter Statusstand ohne last_eval_result (vor der Migration in
        # train.py geschrieben): Candidate lag unter dem Champion, also
        # eindeutig nicht uebernommen. Sonst wuerde dieser Datenpunkt in der
        # Anzeige verloren gehen, sobald der naechste Eval-Zyklus laeuft.
        last_eval_result = "rejected"
    if last_eval and last_eval_result in {"rejected", "regression"}:
        label = (
            "wegen Regression verworfen"
            if last_eval_result == "regression"
            else "nicht übernommen"
        )
        eval_step = int(ts.get("last_eval_at_step", 0) or 0)
        lines.append(f"     Letzter Candidate: {label}")
        if eval_step:
            lines.append(f"       geprüft bei: {fmt(eval_step)} Learner-Steps")
        lines.append(
            f"       erreicht:    Stufe {int(last_eval.get('max_stage', 0) or 0)}, "
            f"Level {int(last_eval.get('max_level', 0) or 0)}, "
            f"Maps {int(last_eval.get('max_maps', 0) or 0)}"
        )
        lines.append(
            f"       Bestweg:     {fmt(last_eval.get('full_best_stage_steps', 0))} "
            f"Weg-Steps ({int(last_eval.get('full_episodes', 0) or 0)} Full-Runs)"
        )

    hist = load_history()
    if not hist:
        lines.append("                 (noch keine Reward-Historie)")
    else:
        now_pt = hist[-1]
        base_idx = 0
        for i in range(len(hist) - 1, 0, -1):
            if int(hist[i].get("timesteps", 0) or 0) < int(hist[i - 1].get("timesteps", 0) or 0):
                base_idx = i
                break
        cmp_idx = max(base_idx, len(hist) - 1 - 40)
        old_pt = hist[cmp_idx]
        best_now = float(now_pt.get("best_episode_reward", 0) or 0)
        best_old = float(old_pt.get("best_episode_reward", 0) or 0)
        avg_now = float(now_pt.get("avg_episode_reward", 0) or 0)
        delta = best_now - best_old
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        lines.append(
            f"     Reward: Best {best_now:.0f} ({arrow}{delta:+.0f})  |  Ø {avg_now:.0f}"
        )
    strikes = int(ts.get("regression_strikes", 0) or 0)
    rollbacks = int(ts.get("rollback_count", 0) or 0)
    if strikes or rollbacks:
        lines.append(
            f"     ⚠ Schutz: {strikes} Regression(en), {rollbacks} Rollback(s)"
        )
    return lines


def render():
    ts = load("trainer_status.json")
    cs = load("champion_score.json")
    gp = load("exploration_memory", "global_progress.json")
    m = cs.get("metrics", {})

    depth = int(gp.get("max_world_stage", 0))
    stage_names = {
        0: "Start/innen", 1: "Alabastia", 2: "Route 1",
        3: "Vertania", 4: "Eichs Paket", 5: "Pokédex/Paket abgegeben",
        6: "Route 2", 7: "Vertania-Wald", 8: "Marmoria",
        9: "erster Orden",
    }
    wall = "  <-- Route-1-Wand" if depth <= 2 else ""

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
        elif agent_id >= 0:
            # Kein festes Envs-Limit mehr - NUM_ENVS aendert sich (32/50/...),
            # nur die Watcher-ID 120 ist reserviert und wird ausgeschlossen.
            instances[agent_id] = data

    out = []
    out.append("=" * WIDTH)
    out.append(f"  {datetime.now():%H:%M:%S}")
    out.append("=" * WIDTH)

    # 1) BRAIN zuerst: die eine Frage die zaehlt - wird das Netz besser?
    out.extend(brain_section(ts, cs, m))

    # 2) WELT: wo steht die Front gerade.
    out.append("-" * WIDTH)
    out.append(f"  🗺️ WELT        Stufe {depth}: {stage_names.get(depth, '?')}{wall}")

    # V17.2: die SCHIGGI-Sektion (Starter-Erfolgsquote) ist seit dem
    # Savestate-Start ueberfluessig geworden - der Starter ist bereits Teil
    # des fixen Episodenstarts und liegt praktisch immer bei ~100 %. War
    # reiner Log-Ballast ("Falscher Starter aktuell: 0" jede Sekunde).
    all_agents = list(instances.values())

    # 3b) K.O.-Bilanz: wie oft stirbt die Flotte gerade (Party-Wipe),
    # gegen wie oft sie selbst K.O.s landet - fleet-weite Summe ueber
    # reward_stats.run_stats aus allen Instanz-JSONs.
    party_wipes = 0
    enemy_faints = 0
    for d in all_agents:
        rs = (d.get("reward_stats") or {}).get("run_stats") or {}
        party_wipes += int(rs.get("party_wipes", 0) or 0)
        enemy_faints += int(rs.get("enemy_faints", 0) or 0)
    out.append("-" * WIDTH)
    out.append("  ☠️ K.O.-BILANZ (Summe ueber die Flotte)")
    out.append(
        f"     Eigene Party K.O. (wiped): {fmt(party_wipes)}   |   "
        f"Gegner-K.O.: {fmt(enemy_faints)}"
    )

    # 4) WATCHER: der sichtbare Einzel-Lauf.
    if watcher:
        wbs = watcher.get("battle_stats") or {}
        out.append("-" * WIDTH)
        out.append("  👁️ WATCHER")
        out.append(f"     Modell:   {watcher.get('loaded_model', '?')}")
        out.append(
            f"     Position: {watcher.get('bank')}/{watcher.get('map')} "
            f"@ {watcher.get('x')},{watcher.get('y')}  |  Level {watcher.get('level', 0)}"
        )
        out.append(
            f"     Schritte: Weg {fmt(watcher.get('route_steps', watcher.get('steps', 0)))}  |  "
            f"Kampf {fmt(watcher.get('battle_steps', 0))}"
        )
        out.append(
            f"     Reward:   {watcher.get('reward', 0)}"
        )
        fights_started = int(wbs.get("started", 0) or 0)
        fights_done = int(wbs.get("completed", 0) or 0)
        if fights_started or fights_done:
            out.append(f"     Kämpfe:   {fights_started} gestartet, {fights_done} beendet")

    maps_seen = Counter()
    for d in instances.values():
        maps_seen[f"{d.get('bank')},{d.get('map')}"] += 1
    if instances:
        out.append("-" * WIDTH)
        out.append(f"  🤖 FLOTTE       {len(instances)} Full-Agenten")
        out.append("     Auf Maps: " + "  ".join(f"{k}:{v}" for k, v in maps_seen.most_common(6)))

    stages = sorted(glob.glob(os.path.join(RT, "curriculum_shared", "stage_*.state.gz")))
    if stages:
        deepest = max(int(os.path.basename(p).split("_")[1].split(".")[0]) for p in stages)
        out.append("-" * WIDTH)
        out.append(f"  💾 CHECKPOINTS  tiefster stage_{deepest}  ({len(stages)} validierte Stufen)")
    out.append("=" * WIDTH)
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    once = "--once" in args or "-1" in args
    interval = 1
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
