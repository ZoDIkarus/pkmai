import os
import glob
import json
import threading
import tempfile
import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

app = FastAPI()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
BASE_DIR = PROJECT_ROOT
ROOMS_DIR = os.path.join(RUNTIME_DIR, "room_captures")
MAP_FILE = os.path.join(ASSETS_DIR, "maps", "kanto_map.png")
INSTANCES_DIR = os.path.join(RUNTIME_DIR, "instances_data")
VERSION_FILE = os.path.join(RUNTIME_DIR, "model_version.json")
SKELETON_FILE = os.path.join(RUNTIME_DIR, "skeleton_map.json")
HISTORY_FILE = os.path.join(RUNTIME_DIR, "training_history.json")
CHAMPION_FILE = os.path.join(RUNTIME_DIR, "champion_score.json")
TRAINER_STATUS_FILE = os.path.join(RUNTIME_DIR, "trainer_status.json")
HISTORY_LOCK = threading.Lock()
EXPLORATION_MEMORY_DIR = os.path.join(RUNTIME_DIR, "exploration_memory")
WATCHER_MAPPING_FILE = os.path.join(RUNTIME_DIR, "watcher_mapping.json")
TRAINER_STATUS_FILE = os.path.join(RUNTIME_DIR, "trainer_status.json")
WATCHER_STREAM_FILE = os.path.join(RUNTIME_DIR, "watcher.jpg")
CLUSTER_DIR = os.path.join(RUNTIME_DIR, "cluster")
CLUSTER_POLICY_FILE = os.path.join(CLUSTER_DIR, "policy.json")
CLUSTER_WORKERS_FILE = os.path.join(CLUSTER_DIR, "workers.json")


def web_bind_settings() -> tuple[str, int]:
    return os.getenv("PKMAI_WEB_BIND_HOST", "0.0.0.0"), int(os.getenv("PKMAI_WEB_PORT", "8001"))


def _live_learner_steps(fallback=0):
    try:
        with open(TRAINER_STATUS_FILE, "r") as f:
            d = json.load(f) or {}
        n = int(d.get("learner_steps", 0) or 0)
        return n if n > 0 else int(fallback or 0)
    except Exception:
        return int(fallback or 0)


@app.get("/api/champion")
def get_champion():
    try:
        with open(CHAMPION_FILE, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {"version":0,"timesteps":0,"metrics":{},"score":[]}


@app.get("/api/trainer-status")
def get_trainer_status():
    try:
        with open(TRAINER_STATUS_FILE, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {"learner_steps":0,"champion_steps":0,"champion_version":0,"delta_steps":0}


@app.get("/api/cluster-status")
def get_cluster_status():
    now = time.time()
    try:
        with open(CLUSTER_POLICY_FILE, "r", encoding="utf-8") as f:
            policy = json.load(f) or {}
    except Exception:
        policy = {}
    try:
        with open(CLUSTER_WORKERS_FILE, "r", encoding="utf-8") as f:
            worker_rows = json.load(f) or {}
    except Exception:
        worker_rows = {}

    workers = []
    for worker_id, row in worker_rows.items():
        if not isinstance(row, dict):
            continue
        last_seen = float(row.get("last_seen", 0) or 0)
        workers.append(
            {
                "worker_id": str(row.get("worker_id", worker_id)),
                "hostname": str(row.get("hostname", "")),
                "active_agents": max(0, int(row.get("active_agents", 0) or 0)),
                "fps": (
                    max(0.0, float(row["fps"]))
                    if row.get("fps") is not None
                    else None
                ),
                "policy_version": max(0, int(row.get("policy_version", 0) or 0)),
                "age_seconds": round(max(0.0, now - last_seen), 1),
                "online": now - last_seen < 15,
            }
        )
    workers.sort(key=lambda item: item["worker_id"])

    checkpoint = os.path.basename(str(policy.get("checkpoint") or ""))
    policy_mtime = os.path.getmtime(CLUSTER_POLICY_FILE) if os.path.exists(CLUSTER_POLICY_FILE) else 0
    return {
        "brain_online": bool(policy_mtime and now - policy_mtime < 45),
        "policy_version": max(0, int(policy.get("version", 0) or 0)),
        "timesteps": max(0, int(policy.get("timesteps", 0) or 0)),
        "checkpoint": checkpoint,
        "workers": workers,
    }


@app.get("/map.png")
def get_map():
    if os.path.exists(MAP_FILE):
        return FileResponse(MAP_FILE, media_type="image/png", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return Response(status_code=404)


@app.get("/watcher.jpg")
def get_watcher_frame():
    try:
        with open(WATCHER_STREAM_FILE, "rb") as f:
            image = f.read()
        return Response(
            content=image,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    except FileNotFoundError:
        return Response(status_code=404)


@app.get("/watcher", response_class=HTMLResponse)
def watcher_view():
    return HTMLResponse("""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PKMAI Watcher</title>
<style>html,body{margin:0;width:100%;height:100%;background:#0a0d14}img{width:100%;height:100%;object-fit:contain}</style>
</head><body><img id="watcher" alt="PKMAI Watcher wird geladen">
<script>const img=document.getElementById('watcher');function refresh(){img.src='/watcher.jpg?t='+Date.now()}refresh();setInterval(refresh,100)</script>
</body></html>""")


def _load_training_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-1000:]
    except Exception:
        pass
    return []


def _save_training_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    # Unique temp file prevents concurrent FastAPI requests from
    # fighting over one shared "training_history.json.tmp".
    fd, tmp = tempfile.mkstemp(
        prefix="training_history_",
        suffix=".tmp",
        dir=os.path.dirname(HISTORY_FILE),
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(history[-1000:], f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, HISTORY_FILE)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _aggregate_training_stats(instances):
    agents = [
        x for x in instances
        if isinstance(x, dict) and 0 <= int(x.get("id", -1)) <= 119
    ]

    episodes = 0
    reward_weighted_sum = 0.0
    best_reward = 0.0
    max_level = max_badges = max_maps = max_explored_tiles = 0

    run_totals = {
        "all_episodes": 0,
        "beginning_episodes": 0,
        "curriculum_episodes": 0,
        "beginning_intro_complete": 0,
        "beginning_stairs_down": 0,
        "beginning_left_house": 0,
        "beginning_grass": 0,
        "beginning_starter": 0,
        "beginning_next_map": 0,
        "beginning_loop_resets": 0,
        "curriculum_loop_resets": 0,

        # V6 Spezialisten
        "v2_intro_episodes": 0,
        "v2_intro_success": 0,
        "v2_stairs_episodes": 0,
        "v2_stairs_success": 0,
        "v2_exit_episodes": 0,
        "v2_exit_success": 0,

        # Full-Chain wird im House Bootcamp aktuell nicht trainiert,
        # bleibt aber fuer kompatible Historie sichtbar.
        "v2_full_episodes": 0,
        "v2_full_intro": 0,
        "v2_full_stairs": 0,
        "v2_full_left_house": 0,
        "battles_started": 0,
        "battles_completed": 0,
        "journey_starter": 0,
        "journey_map5": 0,
        "journey_map10": 0,
        "journey_warp5": 0,
        "journey_progress_checkpoint": 0,
        "journey_badge1": 0,
    }

    # V83_REQUIRED_COUNTERS
    for _k in ('v8_starter_episodes', 'v8_starter_success', 'v8_battle_episodes', 'v8_battle_success', 'v8_level_episodes', 'v8_level_success', 'v8_badge_episodes', 'v8_badge_success', 'v2_full_starter', 'v7_full_badge1', 'enemy_faints', 'enemy_damage_hp'):
        run_totals.setdefault(_k, 0)

    for inst in agents:
        max_level = max(max_level, int(inst.get("level", 0)))
        max_badges = max(max_badges, int(inst.get("badges", 0)))
        max_maps = max(max_maps, int(inst.get("visited_maps", 0)))
        max_explored_tiles = max(
            max_explored_tiles, int(inst.get("explored_tiles", 0))
        )

        rs = inst.get("reward_stats") or {}
        ep = int(rs.get("episodes", 0))
        avg = float(rs.get("avg_episode_reward", 0.0))
        episodes += ep
        reward_weighted_sum += avg * ep
        best_reward = max(
            best_reward,
            float(rs.get("best_episode_reward", 0.0) or 0.0)
        )

        persistent = rs.get("run_stats") or {}
        for key in run_totals:
            run_totals[key] += int(persistent.get(key, 0))

        # V7.6.1: Battle-Fallback direkt aus Agent-Telemetrie.
        # Damit Battles nicht 0 bleiben, falls reward_stats.run_stats
        # in einem Instanzfile fehlt oder hinterherhinkt.
        bs = inst.get("battle_stats") or {}
        if int(persistent.get("battles_started", 0)) <= 0:
            run_totals["battles_started"] += int(
                bs.get("started", bs.get("episode_started", 0)) or 0
            )
        if int(persistent.get("battles_completed", 0)) <= 0:
            run_totals["battles_completed"] += int(
                bs.get("completed", bs.get("episode_completed", 0)) or 0
            )

    avg_reward = reward_weighted_sum / episodes if episodes else 0.0
    beginning = run_totals["beginning_episodes"]

    def beginning_rate(key):
        if beginning <= 0:
            return 0.0
        return min(
            100.0,
            100.0 * run_totals[key] / beginning
        )

    def specialist_rate(success_key, episode_key):
        episode_count = int(run_totals.get(episode_key, 0) or 0)
        if episode_count <= 0:
            return 0.0
        success_count = int(run_totals.get(success_key, 0) or 0)
        return min(
            100.0,
            100.0 * success_count / episode_count
        )

    loop_rate = (
        100.0 * run_totals["beginning_loop_resets"] / beginning
        if beginning else 0.0
    )

    return {
        "episodes": episodes,
        "avg_episode_reward": round(avg_reward, 3),
        "best_episode_reward": round(best_reward, 3),
        "max_level": max_level,
        "max_badges": max_badges,
        "max_maps": max_maps,
        "max_explored_tiles": max_explored_tiles,
        "run_totals": run_totals,
        "beginning_success_rates": {
            "intro_complete": round(
                beginning_rate("beginning_intro_complete"), 2
            ),
            "stairs_down": round(
                beginning_rate("beginning_stairs_down"), 2
            ),
            "left_house": round(
                beginning_rate("beginning_left_house"), 2
            ),
            "north_to_grass": round(
                beginning_rate("beginning_grass"), 2
            ),
            "first_pokemon": round(
                beginning_rate("beginning_starter"), 2
            ),
            "next_outdoor_map": round(
                beginning_rate("beginning_next_map"), 2
            ),
        },
        "v6_skill_rates": {
            "intro": round(
                specialist_rate("v2_intro_success", "v2_intro_episodes"), 2
            ),
            "stairs": round(
                specialist_rate("v2_stairs_success", "v2_stairs_episodes"), 2
            ),
            "exit": round(
                specialist_rate("v2_exit_success", "v2_exit_episodes"), 2
            ),
            "full_intro": round(
                specialist_rate("v2_full_intro", "v2_full_episodes"), 2
            ),
            "full_stairs": round(
                specialist_rate("v2_full_stairs", "v2_full_episodes"), 2
            ),
            "full_exit": round(
                specialist_rate(
                    "v2_full_left_house",
                    "v2_full_episodes"
                ), 2
            ),
        },
        "beginning_loop_resets": run_totals["beginning_loop_resets"],
        "v8_skill_rates": {
            "starter": round(
                specialist_rate("v8_starter_success", "v8_starter_episodes"), 2
            ),
            "battle": round(
                specialist_rate("v8_battle_success", "v8_battle_episodes"), 2
            ),
            "level": round(
                specialist_rate("v8_level_success", "v8_level_episodes"), 2
            ),
            "badge": round(
                specialist_rate("v8_badge_success", "v8_badge_episodes"), 2
            ),
            "full_starter": round(
                specialist_rate("v2_full_starter", "v2_full_episodes"), 2
            ),
            "full_badge1": round(
                specialist_rate("v7_full_badge1", "v2_full_episodes"), 2
            ),
        },
        "beginning_loops_per_100_runs": round(loop_rate, 2),
    }


def _maybe_record_history(version_meta, instances):
    with HISTORY_LOCK:
        version = int(version_meta.get("version", 0) or 0)
        champion_steps = int(version_meta.get("timesteps", 0) or 0)
        learner_steps = 0
        try:
            with open(TRAINER_STATUS_FILE, "r") as f:
                ts = json.load(f) or {}
            learner_steps = int(ts.get("learner_steps", 0) or 0)
        except Exception:
            pass
        timesteps = learner_steps if learner_steps > 0 else champion_steps
        if timesteps <= 0:
            return
        # One live history point every 25k learner steps.
        bucket = (timesteps // 25000) * 25000
        history = _load_training_history()
        if history and int(history[-1].get("timesteps", -1)) >= bucket:
            return

        stats = _aggregate_training_stats(instances)
        history.append({
            "version": version,
            "timesteps": timesteps,
            "max_level": max(stats["max_level"], int(version_meta.get("max_level", 0))),
            "max_badges": max(stats["max_badges"], int(version_meta.get("max_badges", 0))),
            "max_maps": max(stats["max_maps"], int(version_meta.get("max_maps", 0))),
            "episodes": stats["episodes"],
            "avg_episode_reward": stats["avg_episode_reward"],
            "best_episode_reward": stats["best_episode_reward"],
            "beginning_episodes": stats["run_totals"]["beginning_episodes"],
            "curriculum_episodes": stats["run_totals"]["curriculum_episodes"],
            "beginning_loop_resets": stats["beginning_loop_resets"],
            "beginning_loops_per_100_runs":
                stats["beginning_loops_per_100_runs"],
            "beginning_success_rates": stats["beginning_success_rates"],
            "v6_skill_rates": stats["v6_skill_rates"],
            "v8_skill_rates": stats.get("v8_skill_rates", {}),
            "v6_skill_totals": {
                "intro_episodes": stats["run_totals"]["v2_intro_episodes"],
                "intro_success": stats["run_totals"]["v2_intro_success"],
                "stairs_episodes": stats["run_totals"]["v2_stairs_episodes"],
                "stairs_success": stats["run_totals"]["v2_stairs_success"],
                "exit_episodes": stats["run_totals"]["v2_exit_episodes"],
                "exit_success": stats["run_totals"]["v2_exit_success"],
            },
            "stats_schema": 3,
        })
        _save_training_history(history)


def _global_exploration_summary():
    edges=set(); maps=set(); transitions=set()
    for path in glob.glob(os.path.join(EXPLORATION_MEMORY_DIR,"agent_*.json")):
        try:
            with open(path,"r") as f: d=json.load(f)
            for x in d.get("edges",[]):
                if isinstance(x,list) and len(x)==6: edges.add(tuple(x))
            for x in d.get("maps",[]):
                if isinstance(x,list) and len(x)==2: maps.add(tuple(x))
            for x in d.get("transitions",[]):
                if isinstance(x,list) and len(x)==8: transitions.add(tuple(x))
        except Exception:
            pass
    return {"known_edges":len(edges),"known_maps":len(maps),"known_transitions":len(transitions)}

def _v8_direct_skill_stats():
    totals = {}
    stats_dir = os.path.join(RUNTIME_DIR, "training_stats")
    keys = (
        "v2_intro_episodes","v2_intro_success","v2_stairs_episodes","v2_stairs_success",
        "v2_exit_episodes","v2_exit_success","v2_full_episodes","v2_full_intro",
        "v2_full_stairs","v2_full_left_house","v2_full_starter",
        "v8_starter_episodes","v8_starter_success","v8_battle_episodes","v8_battle_success",
        "v8_level_episodes","v8_level_success","v8_badge_episodes","v8_badge_success",
        "v7_full_badge1","battles_started","battles_completed","enemy_faints","enemy_damage_hp"
    )
    for k in keys: totals[k] = 0
    for path in glob.glob(os.path.join(stats_dir, "agent_*.json")):
        try:
            with open(path, "r") as f: d = json.load(f)
            if not isinstance(d, dict): continue
            for k in keys: totals[k] += int(d.get(k, 0) or 0)
        except Exception:
            pass
    def rate(s,e):
        ee=int(totals.get(e,0)); ss=int(totals.get(s,0))
        return round(100.0*ss/ee,2) if ee else 0.0
    full_ep=int(totals.get("v2_full_episodes",0))
    return {
        "totals":totals,
        "rates":{
            "intro":rate("v2_intro_success","v2_intro_episodes"),
            "stairs":rate("v2_stairs_success","v2_stairs_episodes"),
            "exit":rate("v2_exit_success","v2_exit_episodes"),
            "starter":rate("v8_starter_success","v8_starter_episodes"),
            "battle":rate("v8_battle_success","v8_battle_episodes"),
            "level":rate("v8_level_success","v8_level_episodes"),
            "badge":rate("v8_badge_success","v8_badge_episodes"),
            "full_intro":round(100.0*totals["v2_full_intro"]/full_ep,2) if full_ep else 0.0,
            "full_stairs":round(100.0*totals["v2_full_stairs"]/full_ep,2) if full_ep else 0.0,
            "full_exit":round(100.0*totals["v2_full_left_house"]/full_ep,2) if full_ep else 0.0,
            "full_starter":round(100.0*totals["v2_full_starter"]/full_ep,2) if full_ep else 0.0,
            "full_badge1":round(100.0*totals["v7_full_badge1"]/full_ep,2) if full_ep else 0.0,
        }
    }

@app.get("/api/v8_skills")
def get_v8_skills():
    return _v8_direct_skill_stats()

def _v81_skill_health():
    totals = {}
    stats_dir = os.path.join(RUNTIME_DIR, "training_stats")
    keys = (
        "v2_intro_episodes","v2_intro_success","v2_stairs_episodes","v2_stairs_success",
        "v2_exit_episodes","v2_exit_success","v2_full_episodes","v2_full_intro",
        "v2_full_stairs","v2_full_left_house","v2_full_starter",
        "v8_starter_episodes","v8_starter_success","v8_battle_episodes","v8_battle_success",
        "v8_level_episodes","v8_level_success","v8_badge_episodes","v8_badge_success",
        "v7_full_badge1","battles_started","battles_completed","enemy_faints"
    )
    for k in keys: totals[k] = 0
    for p in glob.glob(os.path.join(stats_dir, "agent_*.json")):
        try:
            with open(p,"r") as f: d=json.load(f)
            for k in keys: totals[k] += int(d.get(k,0) or 0)
        except Exception: pass
    def rate(ok,ep):
        e=int(totals.get(ep,0)); o=int(totals.get(ok,0))
        return round(100.0*o/e,2) if e else 0.0
    full_ep=int(totals.get("v2_full_episodes",0))
    def fr(k): return round(100.0*int(totals.get(k,0))/full_ep,2) if full_ep else 0.0
    return {"skills":[
        ["Intro",rate("v2_intro_success","v2_intro_episodes"),totals["v2_intro_success"],totals["v2_intro_episodes"]],
        ["Treppe",rate("v2_stairs_success","v2_stairs_episodes"),totals["v2_stairs_success"],totals["v2_stairs_episodes"]],
        ["Haus Exit",rate("v2_exit_success","v2_exit_episodes"),totals["v2_exit_success"],totals["v2_exit_episodes"]],
        ["Starter",rate("v8_starter_success","v8_starter_episodes"),totals["v8_starter_success"],totals["v8_starter_episodes"]],
        ["Battle KO",rate("v8_battle_success","v8_battle_episodes"),totals["v8_battle_success"],totals["v8_battle_episodes"]],
        ["Level-Up",rate("v8_level_success","v8_level_episodes"),totals["v8_level_success"],totals["v8_level_episodes"]],
        ["Gym / Badge",rate("v8_badge_success","v8_badge_episodes"),totals["v8_badge_success"],totals["v8_badge_episodes"]],
        ["Full Intro",fr("v2_full_intro"),totals["v2_full_intro"],full_ep],
        ["Full Treppe",fr("v2_full_stairs"),totals["v2_full_stairs"],full_ep],
        ["Full Exit",fr("v2_full_left_house"),totals["v2_full_left_house"],full_ep],
        ["Full Starter",fr("v2_full_starter"),totals["v2_full_starter"],full_ep],
        ["Full Badge 1",fr("v7_full_badge1"),totals["v7_full_badge1"],full_ep]
    ],"totals":totals}

@app.get("/api/v81_skills")
def api_v81_skills():
    return _v81_skill_health()



def _v84_skill_stats():
    stats_dir = os.path.join(RUNTIME_DIR, "training_stats")
    keys = (
        "v2_intro_episodes","v2_intro_success",
        "v2_stairs_episodes","v2_stairs_success",
        "v2_exit_episodes","v2_exit_success",
        "v2_full_episodes","v2_full_intro","v2_full_stairs",
        "v2_full_left_house","v2_full_starter",
        "v8_starter_episodes","v8_starter_success",
        "v8_battle_episodes","v8_battle_success",
        "v8_level_episodes","v8_level_success",
        "v8_badge_episodes","v8_badge_success",
        "v7_full_badge1",
        "battles_started","battles_completed",
        "enemy_faints","enemy_damage_hp"
    )
    totals = {k: 0 for k in keys}
    agents_seen = set()

    for path in sorted(glob.glob(os.path.join(stats_dir, "agent_*.json"))):
        try:
            aid = int(os.path.basename(path)[6:8])
            if not (0 <= aid <= 119):
                continue
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            agents_seen.add(aid)
            for k in keys:
                totals[k] += int(data.get(k, 0) or 0)
        except Exception:
            pass

    def pct(ok, ep):
        e = int(totals.get(ep, 0))
        o = int(totals.get(ok, 0))
        return round(100.0 * o / e, 2) if e else 0.0

    full_ep = int(totals.get("v2_full_episodes", 0))
    def fpct(k):
        return round(100.0 * int(totals.get(k, 0)) / full_ep, 2) if full_ep else 0.0

    skills = [
        {"label":"Intro","rate":pct("v2_intro_success","v2_intro_episodes"),"success":totals["v2_intro_success"],"episodes":totals["v2_intro_episodes"]},
        {"label":"Treppe","rate":pct("v2_stairs_success","v2_stairs_episodes"),"success":totals["v2_stairs_success"],"episodes":totals["v2_stairs_episodes"]},
        {"label":"Haus Exit","rate":pct("v2_exit_success","v2_exit_episodes"),"success":totals["v2_exit_success"],"episodes":totals["v2_exit_episodes"]},
        {"label":"Erstes Pokémon","rate":pct("v8_starter_success","v8_starter_episodes"),"success":totals["v8_starter_success"],"episodes":totals["v8_starter_episodes"]},
        {"label":"Battle KO","rate":pct("v8_battle_success","v8_battle_episodes"),"success":totals["v8_battle_success"],"episodes":totals["v8_battle_episodes"]},
        {"label":"Level-Up","rate":pct("v8_level_success","v8_level_episodes"),"success":totals["v8_level_success"],"episodes":totals["v8_level_episodes"]},
        {"label":"Gym / Badge","rate":pct("v8_badge_success","v8_badge_episodes"),"success":totals["v8_badge_success"],"episodes":totals["v8_badge_episodes"]},
        {"label":"Full Intro","rate":fpct("v2_full_intro"),"success":totals["v2_full_intro"],"episodes":full_ep},
        {"label":"Full Treppe","rate":fpct("v2_full_stairs"),"success":totals["v2_full_stairs"],"episodes":full_ep},
        {"label":"Full Exit","rate":fpct("v2_full_left_house"),"success":totals["v2_full_left_house"],"episodes":full_ep},
        {"label":"Full Pokémon","rate":fpct("v2_full_starter"),"success":totals["v2_full_starter"],"episodes":full_ep},
        {"label":"Full Badge 1","rate":fpct("v7_full_badge1"),"success":totals["v7_full_badge1"],"episodes":full_ep}
    ]

    return {
        "agents_seen": len(agents_seen),
        "skills": skills,
        "battle": {
            "started": totals["battles_started"],
            "completed": totals["battles_completed"],
            "enemy_faints": totals["enemy_faints"],
            "enemy_damage_hp": totals["enemy_damage_hp"]
        }
    }

@app.get("/api/v84_skills")
def get_v84_skills():
    return _v84_skill_stats()

@app.get("/api/state")
def get_state():
    files = glob.glob(os.path.join(INSTANCES_DIR, "inst_*.json"))
    instances = []
    max_level = 0
    max_badges = 0
    total_steps = 0
    trainer_name = "Alex"
    party = []

    for f in sorted(files):
        try:
            with open(f, "r") as jf:
                data = json.load(jf)
                instances.append(data)
                if data.get("level", 0) > max_level:
                    max_level = data.get("level", 0)
                if data.get("badges", 0) > max_badges:
                    max_badges = data.get("badges", 0)
                total_steps += data.get("steps", 0)
                if data.get("id") == 99:
                    trainer_name = data.get("name", "Alex").replace(" (Watcher)", "")
                    party = data.get("party", [])
        except Exception:
            pass

    # Sortieren: Watcher (ID 99) immer an 1. Stelle, danach Agent 00 bis 39
    instances.sort(key=lambda x: (0 if x.get("id") == 99 else 1, x.get("id", 0)))

    version_meta = {"version": 0, "timesteps": 0}
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as vf:
                loaded = json.load(vf)
                if isinstance(loaded, dict):
                    version_meta.update(loaded)
        except Exception:
            pass

    try:
        training_stats = _aggregate_training_stats(instances)
    except Exception as _agg_error:
        print(f"[WEB] aggregate warning: {_agg_error}")
        training_stats = {
            "episodes": 0, "avg_episode_reward": 0.0,
            "best_episode_reward": 0.0, "max_level": 0,
            "max_badges": 0, "max_maps": 0,
            "max_explored_tiles": 0,
            "run_totals": {}, "beginning_success_rates": {},
            "v6_skill_rates": {}, "v8_skill_rates": {},
            "beginning_loop_resets": 0,
            "beginning_loops_per_100_runs": 0.0,
        }
    try:
        _maybe_record_history(version_meta, instances)
    except Exception as _history_error:
        print(f"[WEB] history warning: {_history_error}")
    try:
        global_exploration = _global_exploration_summary()
    except Exception:
        global_exploration = {
            "known_edges": 0,
            "known_maps": 0,
            "known_transitions": 0,
        }
    battle_started = int(training_stats["run_totals"].get("battles_started",0))
    battle_completed = int(training_stats["run_totals"].get("battles_completed",0))

    return {
        "trainer_name": trainer_name,
        "version": int(version_meta.get("version", 0)),
        "training_timesteps": _live_learner_steps(int(version_meta.get("timesteps", 0))),
        "max_level": max(max_level, training_stats["max_level"]),
        "max_badges": max(max_badges, training_stats["max_badges"]),
        "total_steps": total_steps,
        "training_stats": training_stats,
        "global_exploration": global_exploration,
        "battle_stats": {"started":battle_started,"completed":battle_completed},
        "journey_stats": {
            "starter": int(training_stats["run_totals"].get("journey_starter",0)),
            "map5": int(training_stats["run_totals"].get("journey_map5",0)),
            "map10": int(training_stats["run_totals"].get("journey_map10",0)),
            "warp5": int(training_stats["run_totals"].get("journey_warp5",0)),
            "progress": int(training_stats["run_totals"].get("journey_progress_checkpoint",0)),
            "badge1": int(training_stats["run_totals"].get("journey_badge1",0)),
        },
        "party": party,
        "instances": instances
    }

@app.get("/api/history")
def get_history():
    return {"history": _load_training_history()}


def _load_instances_for_skeleton():
    instances = []
    for f in sorted(glob.glob(os.path.join(INSTANCES_DIR, "inst_*.json"))):
        try:
            with open(f, "r") as jf:
                data = json.load(jf)
            if isinstance(data, dict):
                instances.append(data)
        except Exception:
            pass
    return instances


@app.get("/api/skeleton")
def get_skeleton():
    """
    RAM-Skeleton direkt aus vorhandener Agent-Telemetrie.
    Keine Screenshots, keine zusaetzlichen RAM-Reads, kein Mapper-Prozess.
    """
    instances = _load_instances_for_skeleton()
    maps = {}
    transitions = {}

    for inst in instances:
        try:
            agent_id = int(inst.get("id", -1))
        except Exception:
            continue

        points = []
        for pt in (inst.get("path") or []):
            if not isinstance(pt, (list, tuple)) or len(pt) < 4:
                continue
            try:
                points.append([
                    int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])
                ])
            except Exception:
                continue

        try:
            cur = [
                int(inst.get("bank", 0)),
                int(inst.get("map", 0)),
                int(inst.get("x", 0)),
                int(inst.get("y", 0)),
            ]
            if not points or points[-1] != cur:
                points.append(cur)
        except Exception:
            pass

        prev = None
        for bank, map_id, x, y in points:
            if not (
                0 <= bank < 128 and 0 <= map_id < 128
                and 0 <= x < 512 and 0 <= y < 512
            ):
                prev = None
                continue

            key = (bank, map_id)
            m = maps.setdefault(key, {
                "bank": bank,
                "map_id": map_id,
                "min_x": x,
                "max_x": x,
                "min_y": y,
                "max_y": y,
                "agents": set(),
                "observations": 0,
            })
            m["min_x"] = min(m["min_x"], x)
            m["max_x"] = max(m["max_x"], x)
            m["min_y"] = min(m["min_y"], y)
            m["max_y"] = max(m["max_y"], y)
            m["agents"].add(agent_id)
            m["observations"] += 1

            if prev is not None:
                pb, pm, px, py = prev
                if (pb, pm) != (bank, map_id):
                    tkey = (pb, pm, bank, map_id)
                    tr = transitions.setdefault(tkey, {
                        "from_bank": pb,
                        "from_map": pm,
                        "to_bank": bank,
                        "to_map": map_id,
                        "agents": set(),
                        "count": 0,
                    })
                    tr["agents"].add(agent_id)
                    tr["count"] += 1

            prev = (bank, map_id, x, y)

    out_maps = []
    for m in maps.values():
        m["agents"] = sorted(m["agents"])
        m["width_tiles"] = int(m["max_x"] - m["min_x"] + 1)
        m["height_tiles"] = int(m["max_y"] - m["min_y"] + 1)
        out_maps.append(m)

    out_transitions = []
    for tr in transitions.values():
        tr["agents"] = sorted(tr["agents"])
        out_transitions.append(tr)

    return {
        "maps": sorted(out_maps, key=lambda m: (m["bank"], m["map_id"])),
        "transitions": out_transitions,
    }


def _cluster_warp_points(transitions, radius=2):
    """
    Convert noisy transition samples into physical-looking warp endpoints.

    Training positions are sampled every few agent steps, so the same doorway
    can be recorded at adjacent tiles. We keep the raw transitions untouched
    and cluster only the visualization/API warp points.
    """
    candidates = []

    for t in transitions:
        if not isinstance(t, (tuple, list)) or len(t) != 8:
            continue

        a = tuple(int(v) for v in t[:4])
        b = tuple(int(v) for v in t[4:])

        # Each side is a possible physical entrance/exit.
        candidates.append({
            "bank": a[0], "map": a[1], "x": a[2], "y": a[3],
            "to_bank": b[0], "to_map": b[1],
        })
        candidates.append({
            "bank": b[0], "map": b[1], "x": b[2], "y": b[3],
            "to_bank": a[0], "to_map": a[1],
        })

    # Group only endpoints on the same source map going to the same target map.
    grouped = {}
    for p in candidates:
        key = (
            p["bank"], p["map"],
            p["to_bank"], p["to_map"],
        )
        grouped.setdefault(key, []).append(p)

    out = []
    for key, points in grouped.items():
        clusters = []

        for p in points:
            matched = None
            for c in clusters:
                # Manhattan <= radius means "same doorway area".
                if abs(p["x"] - c["cx"]) + abs(p["y"] - c["cy"]) <= radius:
                    matched = c
                    break

            if matched is None:
                clusters.append({
                    "points": [p],
                    "cx": p["x"],
                    "cy": p["y"],
                })
            else:
                matched["points"].append(p)
                xs = [q["x"] for q in matched["points"]]
                ys = [q["y"] for q in matched["points"]]
                # Median is robust against one bad sampled coordinate.
                xs.sort()
                ys.sort()
                matched["cx"] = xs[len(xs)//2]
                matched["cy"] = ys[len(ys)//2]

        for c in clusters:
            sample = c["points"][0]
            out.append({
                "bank": int(sample["bank"]),
                "map": int(sample["map"]),
                "x": int(c["cx"]),
                "y": int(c["cy"]),
                "to_bank": int(sample["to_bank"]),
                "to_map": int(sample["to_map"]),
                "samples": len(c["points"]),
            })

    out.sort(
        key=lambda p: (
            p["bank"], p["map"],
            p["x"], p["y"],
            p["to_bank"], p["to_map"],
        )
    )
    return out


@app.get("/api/global_mapping")
def get_global_mapping():
    """Union aller Training-Agents plus Watcher, exakt dedupliziert."""
    tiles = set()
    edges = set()
    maps = set()
    transitions = set()

    # 120 Trainingsagenten
    for agent_id in range(120):
        path = os.path.join(
            EXPLORATION_MEMORY_DIR,
            f"agent_{agent_id:02d}.json"
        )
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)

            for x in data.get("edges", []):
                if isinstance(x, list) and len(x) == 6:
                    edge = tuple(int(v) for v in x)
                    edges.add(edge)
                    tiles.add((edge[0], edge[1], edge[2], edge[3]))
                    tiles.add((edge[0], edge[1], edge[4], edge[5]))

            for x in data.get("maps", []):
                if isinstance(x, list) and len(x) == 2:
                    maps.add(tuple(int(v) for v in x))

            for x in data.get("transitions", []):
                if isinstance(x, list) and len(x) == 8:
                    transitions.add(tuple(int(v) for v in x))
        except Exception:
            pass

    # Watcher-Mapping ebenfalls einbeziehen.
    if os.path.exists(WATCHER_MAPPING_FILE):
        try:
            with open(WATCHER_MAPPING_FILE, "r") as f:
                data = json.load(f)
            for x in data.get("tiles", []):
                if isinstance(x, list) and len(x) == 4:
                    tiles.add(tuple(int(v) for v in x))
            for x in data.get("edges", []):
                if isinstance(x, list) and len(x) == 6:
                    edges.add(tuple(int(v) for v in x))
            for x in data.get("maps", []):
                if isinstance(x, list) and len(x) == 2:
                    maps.add(tuple(int(v) for v in x))
            for x in data.get("transitions", []):
                if isinstance(x, list) and len(x) == 8:
                    transitions.add(tuple(int(v) for v in x))
        except Exception:
            pass

    warp_points_raw = _cluster_warp_points(
        sorted(transitions),
        radius=2,
    )
    warp_points = [
        p for p in warp_points_raw
        if int(p.get("samples", 0)) >= 3
    ]

    return {
        "tiles": [list(x) for x in sorted(tiles)],
        "edges": [list(x) for x in sorted(edges)],
        "maps": [list(x) for x in sorted(maps)],
        # Raw transitions stay available for graph/history.
        "transitions": [list(x) for x in sorted(transitions)],
        # Visual warp endpoints are spatially deduplicated.
        "warp_points": warp_points,
    }


@app.get("/api/explorations")
def get_explorations():
    result = {}
    for agent_id in range(120):
        path = os.path.join(
            EXPLORATION_MEMORY_DIR,
            f"agent_{agent_id:02d}.json"
        )
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            result[str(agent_id)] = {
                "edges": data.get("edges", []),
                "maps": data.get("maps", []),
                "transitions": data.get("transitions", []),
            }
        except Exception:
            pass
    return result


@app.get("/api/exploration/{agent_id}")
def get_exploration(agent_id: int):
    if agent_id < 0 or agent_id > 119:
        return {"agent_id": agent_id, "edges": [], "maps": [], "transitions": []}

    path = os.path.join(
        EXPLORATION_MEMORY_DIR,
        f"agent_{agent_id:02d}.json"
    )
    if not os.path.exists(path):
        return {"agent_id": agent_id, "edges": [], "maps": [], "transitions": []}

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "agent_id": agent_id,
            "edges": data.get("edges", []),
            "maps": data.get("maps", []),
            "transitions": data.get("transitions", []),
        }
    except Exception:
        return {"agent_id": agent_id, "edges": [], "maps": [], "transitions": []}


@app.get("/api/rooms")
def get_rooms():
    files = glob.glob(os.path.join(ROOMS_DIR, "*.png"))
    return [{"name": os.path.splitext(os.path.basename(f))[0], "url": f"/room/{os.path.basename(f)}"} for f in sorted(files)]

@app.get("/room/{filename}")
def get_room_img(filename: str):
    p = os.path.join(ROOMS_DIR, filename)
    if os.path.exists(p):
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return Response(status_code=404)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pokemon FireRed AI Live Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        async function v8RefreshSelectedAgent() {
            try {
                const r = await fetch('/api/state', {cache:'no-store'});
                const d = await r.json();
                const list = d.instances || [];
                let inst = null;
                if (selectedAgentId !== null) {
                    inst = list.find(x => Number(x.id) === Number(selectedAgentId));
                }
                if (!inst) inst = list.find(x => Number(x.id) === 120) || null;
                if (!inst) return;

                const role = document.getElementById('v8-agent-role');
                if (role) role.textContent =
                    (inst.agent_role || inst.training_objective || 'watcher') +
                    ' · ' + (inst.name || ('Agent '+inst.id));

                const grid = document.getElementById('v8-party-grid');
                if (grid) {
                    const party = inst.party || [];
                    if (!party.length) {
                        grid.innerHTML = '<div class="v8-mon"><span>Noch kein Pokémon / keine Party-Telemetrie</span></div>';
                    } else {
                        grid.innerHTML = party.map(m => {
                            const moves = (m.moves || []).map(x => x.name || ('Move '+x.id)).join(', ');
                            return `<div class="v8-mon"><b>${m.name || ('Species #'+m.species_id)} Lv.${m.level || 0}</b><span>HP ${m.cur_hp || 0}/${m.max_hp || 0}</span><br><span>${moves || '—'}</span></div>`;
                        }).join('');
                    }
                }

                const rb = document.getElementById('v8-reward-breakdown');
                if (rb) {
                    const rs = inst.reward_stats || {};
                    const ev = rs.event_counts || {};
                    const keys = Object.keys(ev).sort((a,b)=>(ev[b]||0)-(ev[a]||0));
                    rb.innerHTML =
                        '<b style="color:#7df9b7">TRAINING REWARD DETAILS</b><br>' +
                        `Current Episode: ${Number(inst.reward || 0).toFixed(2)}<br>` +
                        `Ø Episode: ${Number(rs.avg_episode_reward || 0).toFixed(2)}<br>` +
                        `Best: ${Number(rs.best_episode_reward || 0).toFixed(2)}<br><br>` +
                        (keys.slice(0,20).map(k => `${k}: ${ev[k]}`).join('<br>') || 'noch keine Events') +
                        '<br><br><b>Letzter Step</b><br>' +
                        ((inst.reward_events || []).join('<br>') || '—');
                }
            } catch(e) {}
        }

        document.addEventListener('click', function(e) {
            if (e.target.closest('#detail-reward')) {
                const box = document.getElementById('v8-reward-breakdown');
                if (box) box.classList.toggle('open');
            }
        });
        setInterval(v8RefreshSelectedAgent, 1200);

    
        let v8SkillChart = null;
        function v8SkillCard(label,pct,success,episodes){
            return `<div class="v8-skill-card"><div class="n">${label}</div><div class="v">${Number(pct||0).toFixed(1)}%</div><div class="s">${success||0}/${episodes||0} Erfolge</div></div>`;
        }
        async function refreshV8SkillGraph(){
            try{
                const d=await (await fetch('/api/v8_skills',{cache:'no-store'})).json();
                const r=d.rates||{}, t=d.totals||{};
                const rows=[
                    ['Intro',r.intro,t.v2_intro_success,t.v2_intro_episodes],
                    ['Treppe',r.stairs,t.v2_stairs_success,t.v2_stairs_episodes],
                    ['Haus Exit',r.exit,t.v2_exit_success,t.v2_exit_episodes],
                    ['Starter',r.starter,t.v8_starter_success,t.v8_starter_episodes],
                    ['Battle KO',r.battle,t.v8_battle_success,t.v8_battle_episodes],
                    ['Level-Up',r.level,t.v8_level_success,t.v8_level_episodes],
                    ['Gym/Badge',r.badge,t.v8_badge_success,t.v8_badge_episodes],
                    ['Full Intro',r.full_intro,t.v2_full_intro,t.v2_full_episodes],
                    ['Full Treppe',r.full_stairs,t.v2_full_stairs,t.v2_full_episodes],
                    ['Full Exit',r.full_exit,t.v2_full_left_house,t.v2_full_episodes],
                    ['Full Starter',r.full_starter,t.v2_full_starter,t.v2_full_episodes],
                    ['Full Badge 1',r.full_badge1,t.v7_full_badge1,t.v2_full_episodes]
                ];
                const cards=document.getElementById('v8-skill-cards');
                if(cards) cards.innerHTML=rows.map(x=>v8SkillCard(x[0],x[1],x[2],x[3])).join('');
                const canvas=document.getElementById('v8-skill-chart');
                if(!canvas || typeof Chart==='undefined') return;
                const labels=rows.map(x=>x[0]), values=rows.map(x=>Number(x[1]||0));
                if(!v8SkillChart){
                    v8SkillChart=new Chart(canvas,{type:'bar',data:{labels,datasets:[{label:'Skill %',data:values}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{min:0,max:100,ticks:{callback:v=>v+'%'}}},plugins:{legend:{display:false}}}});
                }else{
                    v8SkillChart.data.labels=labels;
                    v8SkillChart.data.datasets[0].data=values;
                    v8SkillChart.update('none');
                }
            }catch(e){}
        }
        setInterval(refreshV8SkillGraph,1500);
        setTimeout(refreshV8SkillGraph,300);

            let v81chart=null;
        async function refreshV81Skills(){
            try{
                const d=await (await fetch('/api/v81_skills',{cache:'no-store'})).json();
                const rows=d.skills||[];
                const g=document.getElementById('v81skills');
                if(g) g.innerHTML=rows.map(x=>`<div class="v81card"><div class="v81n">${x[0]}</div><div class="v81v">${Number(x[1]||0).toFixed(1)}%</div><div class="v81s">${x[2]||0}/${x[3]||0}</div></div>`).join('');
                const c=document.getElementById('v81skillschart');
                if(c && typeof Chart!=='undefined'){
                    const labels=rows.map(x=>x[0]), vals=rows.map(x=>Number(x[1]||0));
                    if(!v81chart){v81chart=new Chart(c,{type:'bar',data:{labels,datasets:[{label:'Skill %',data:vals}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{min:0,max:100,ticks:{callback:v=>v+'%'}}},plugins:{legend:{display:false}}}})}
                    else{v81chart.data.labels=labels;v81chart.data.datasets[0].data=vals;v81chart.update('none')}
                }
            }catch(e){}
        }
        setInterval(refreshV81Skills,1500);setTimeout(refreshV81Skills,250);

    
        async function refreshV83Skills(){
            try{
                const d = await (await fetch('/api/state',{cache:'no-store'})).json();
                const ts = d.training_stats || {};
                const a = ts.v6_skill_rates || {};
                const b = ts.v8_skill_rates || {};
                const rt = ts.run_totals || {};
                const rows = [
                    ['Intro',a.intro||0,rt.v2_intro_success||0,rt.v2_intro_episodes||0],
                    ['Treppe',a.stairs||0,rt.v2_stairs_success||0,rt.v2_stairs_episodes||0],
                    ['Haus Exit',a.exit||0,rt.v2_exit_success||0,rt.v2_exit_episodes||0],
                    ['Erstes Pokémon',b.starter||0,rt.v8_starter_success||0,rt.v8_starter_episodes||0],
                    ['Battle KO',b.battle||0,rt.v8_battle_success||0,rt.v8_battle_episodes||0],
                    ['Level-Up',b.level||0,rt.v8_level_success||0,rt.v8_level_episodes||0],
                    ['Gym / Badge',b.badge||0,rt.v8_badge_success||0,rt.v8_badge_episodes||0],
                    ['Full Intro',a.full_intro||0,rt.v2_full_intro||0,rt.v2_full_episodes||0],
                    ['Full Treppe',a.full_stairs||0,rt.v2_full_stairs||0,rt.v2_full_episodes||0],
                    ['Full Exit',a.full_exit||0,rt.v2_full_left_house||0,rt.v2_full_episodes||0],
                    ['Full Pokémon',b.full_starter||0,rt.v2_full_starter||0,rt.v2_full_episodes||0],
                    ['Full Badge 1',b.full_badge1||0,rt.v7_full_badge1||0,rt.v2_full_episodes||0]
                ];
                const box=document.getElementById('v83-skill-table');
                if(box){
                    box.innerHTML=rows.map(x=>`<div style="background:#0c121a;border:1px solid #283649;border-radius:9px;padding:8px">
                        <div style="font-size:9px;color:#8291a5">${x[0]}</div>
                        <div style="font-size:19px;font-weight:900;color:#7df9b7">${Number(x[1]).toFixed(1)}%</div>
                        <div style="font-size:8px;color:#657488">${x[2]}/${x[3]}</div>
                    </div>`).join('');
                }
                const c=document.getElementById('v83-skill-canvas');
                if(c){
                    const ctx=c.getContext('2d'), W=c.width, H=c.height;
                    ctx.clearRect(0,0,W,H);
                    ctx.fillStyle='#0c121a'; ctx.fillRect(0,0,W,H);
                    const bw=(W-80)/rows.length;
                    rows.forEach((x,i)=>{
                        const p=Math.max(0,Math.min(100,Number(x[1]||0)));
                        const h=(H-55)*p/100;
                        ctx.fillStyle='#2ddf9b';
                        ctx.fillRect(45+i*bw,H-35-h,bw*.62,h);
                        ctx.save();
                        ctx.translate(45+i*bw,H-20);
                        ctx.rotate(-0.55);
                        ctx.fillStyle='#91a0b4';
                        ctx.font='11px sans-serif';
                        ctx.fillText(x[0],0,0);
                        ctx.restore();
                    });
                    ctx.strokeStyle='#344255'; ctx.beginPath();
                    ctx.moveTo(35,10);ctx.lineTo(35,H-35);ctx.lineTo(W-10,H-35);ctx.stroke();
                }
            }catch(e){ console.error('V83 skills',e); }
        }
        setInterval(refreshV83Skills,1500);
        setTimeout(refreshV83Skills,300);

    
        function v84Card(s){
            return `<div class="v84-card"><div class="n">${s.label}</div><div class="v">${Number(s.rate||0).toFixed(1)}%</div><div class="s">${s.success||0}/${s.episodes||0}</div></div>`;
        }
        function drawV84(rows){
            const c=document.getElementById('v84-skill-canvas'); if(!c)return;
            const ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#0b121a';ctx.fillRect(0,0,W,H);
            const left=55,right=20,top=18,bottom=72,plotW=W-left-right,plotH=H-top-bottom,slot=plotW/Math.max(1,rows.length);
            ctx.strokeStyle='#2a394b';ctx.lineWidth=1;
            for(let p=0;p<=100;p+=20){const y=top+plotH-(plotH*p/100);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(W-right,y);ctx.stroke();ctx.fillStyle='#73849a';ctx.font='11px sans-serif';ctx.fillText(p+'%',8,y+4);}
            rows.forEach((s,i)=>{const pct=Math.max(0,Math.min(100,Number(s.rate||0)));const h=plotH*pct/100,bw=slot*.58,x=left+i*slot+(slot-bw)/2,y=top+plotH-h;ctx.fillStyle='#28dfa0';ctx.fillRect(x,y,bw,h);ctx.fillStyle='#d9e7f5';ctx.font='10px sans-serif';ctx.fillText(pct.toFixed(1)+'%',x,y-5);ctx.save();ctx.translate(x+bw/2,H-bottom+14);ctx.rotate(-0.58);ctx.fillStyle='#8b9bb0';ctx.font='11px sans-serif';ctx.fillText(s.label,0,0);ctx.restore();});
        }
        async function refreshV84Skills(){
            try{
                const r=await fetch('/api/v84_skills',{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);
                const d=await r.json(),rows=d.skills||[],b=d.battle||{};
                const g=document.getElementById('v84-skill-grid');if(g)g.innerHTML=rows.map(v84Card).join('');
                const m=document.getElementById('v84-meta');if(m)m.innerHTML=[[d.agents_seen||0,'Agents mit Stats'],[b.started||0,'Battles gestartet'],[b.completed||0,'Battles beendet'],[b.enemy_faints||0,'Enemy KOs'],[b.enemy_damage_hp||0,'Enemy HP Damage']].map(x=>`<div><b>${x[0]}</b><span>${x[1]}</span></div>`).join('');
                drawV84(rows);
            }catch(e){const g=document.getElementById('v84-skill-grid');if(g)g.innerHTML=`<div class="v84-card" style="grid-column:1/-1"><div class="n">API FEHLER</div><div class="s">${String(e)}</div></div>`;}
        }
        setInterval(refreshV84Skills,1500);setTimeout(refreshV84Skills,250);

    </script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        /* V85 SINGLE SKILL VIEW */
        #v84-skill-panel,.v8-skill-panel,.graph-card:has(#v81skillschart),.graph-card:has(#v83-skill-canvas){display:none !important;}
        /* V84 CANONICAL SKILLS */
        .v81skills,.v8-skill-panel{display:none !important;}
        .v84-skill-panel{margin:0 0 14px 0;background:#0d141d;border:1px solid #29384a;border-radius:14px;padding:14px;}
        .v84-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
        .v84-head b{font-size:14px;color:#eef7ff}.v84-head span{font-size:10px;color:#7c8ca1}
        .v84-grid{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:8px;}
        .v84-card{background:#101821;border:1px solid #263648;border-radius:10px;padding:9px;}
        .v84-card .n{font-size:9px;color:#8191a5;text-transform:uppercase}.v84-card .v{font-size:20px;font-weight:900;color:#7df9b7;margin-top:4px}.v84-card .s{font-size:8px;color:#607085;margin-top:2px}
        .v84-meta{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:10px;}
        .v84-meta div{background:#0a1118;border:1px solid #223142;border-radius:9px;padding:8px;text-align:center;}
        .v84-meta b{display:block;font-size:16px;color:#eaf7ff}.v84-meta span{font-size:8px;color:#68788d}
        .v84-canvas-wrap{height:300px;margin-top:12px;}#v84-skill-canvas{width:100%;height:300px;}
        /* V81 NIGHT SKILLS */
        .v81skills{position:absolute;top:14px;left:14px;z-index:930;width:min(800px,calc(100vw - 300px));background:rgba(7,12,18,.94);border:1px solid #29384a;border-radius:13px;padding:9px;backdrop-filter:blur(10px)}
        .v81grid{display:grid;grid-template-columns:repeat(6,1fr);gap:5px}.v81card{background:#101821;border:1px solid #243244;border-radius:8px;padding:6px}.v81n{font-size:8px;color:#8190a3}.v81v{font-size:15px;font-weight:900;color:#79f5b2}.v81s{font-size:7px;color:#617085}
        /* V8 SKILL GRAPH FIX */
        .v8-skill-panel{background:#111821;border:1px solid #263446;border-radius:12px;padding:12px;margin-bottom:12px}
        .v8-skill-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
        .v8-skill-title b{font-size:13px}.v8-skill-title span{font-size:9px;color:#7f8ea1}
        .v8-skill-cards{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:7px;margin-bottom:12px}
        .v8-skill-card{background:#0b1118;border:1px solid #243244;border-radius:9px;padding:8px}
        .v8-skill-card .n{font-size:8px;color:#7b8a9d;text-transform:uppercase}
        .v8-skill-card .v{font-size:18px;font-weight:900;color:#7df9b7;margin-top:3px}
        .v8-skill-card .s{font-size:8px;color:#647286;margin-top:2px}
        .v8-skill-chart-wrap{height:280px;position:relative}
        /* V8 NIGHT MAP */
        #map-view {
            background: radial-gradient(circle at 50% 45%, #14252d 0%, #091117 46%, #05080d 100%);
        }
        #map-view .leaflet-image-layer {
            filter: brightness(.38) saturate(.72) contrast(1.28);
        }
        .v8-agent-party {
            margin-top:8px;padding:8px;border-radius:10px;
            background:rgba(8,13,19,.94);border:1px solid #263446;
        }
        .v8-agent-party-title {
            display:flex;justify-content:space-between;font-size:10px;
            color:#8392a8;margin-bottom:7px;text-transform:uppercase;
        }
        .v8-party-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
        .v8-mon {padding:6px;border:1px solid #263446;border-radius:8px;background:#101821}
        .v8-mon b {display:block;font-size:10px;color:#edf7ff}
        .v8-mon span {font-size:9px;color:#8796aa}
        .v8-reward-breakdown {
            display:none;margin-top:8px;padding:8px;border-radius:10px;
            background:#091017;border:1px solid #243345;font-size:9px;
            max-height:170px;overflow:auto;
        }
        .v8-reward-breakdown.open {display:block}
        #detail-reward {cursor:pointer;color:#7df9b7}
        body { margin: 0; background: #0c0e14; color: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        header { 
            padding: 8px 16px; 
            background: #151821; 
            border-bottom: 1px solid #232738; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            z-index: 100;
        }
        .header-left { display: flex; align-items: center; gap: 14px; }
        .logo-title { font-weight: 800; font-size: 15px; color: #00e676; letter-spacing: 0.5px; }
        
        .badge-bar { display: flex; align-items: center; gap: 5px; background: #0e1017; padding: 4px 8px; border-radius: 20px; border: 1px solid #232738; }
        .badge-slot{width:28px;height:28px;border-radius:9px;background:linear-gradient(145deg,#202432,#10131c);border:1px solid #353b50;display:flex;align-items:center;justify-content:center;font-size:14px;filter:grayscale(1);opacity:.34;transform:scale(.94);transition:.25s;box-shadow:inset 0 0 10px rgba(0,0,0,.45)}
        .badge-slot.active{filter:none;opacity:1;transform:scale(1);border-color:#ffd54f;background:linear-gradient(145deg,#3b3420,#17191f);box-shadow:0 0 13px rgba(255,213,79,.5)}
        .live-global{position:absolute;top:16px;right:16px;z-index:950;min-width:245px;background:rgba(12,14,20,.91);border:1px solid #2b3143;border-radius:12px;padding:10px;backdrop-filter:blur(10px);box-shadow:0 10px 28px rgba(0,0,0,.48)}
        .live-global-title{display:flex;justify-content:space-between;align-items:center;font-size:11px;font-weight:800;color:#00e676;margin-bottom:8px;letter-spacing:.4px}
        .live-dot{width:8px;height:8px;border-radius:50%;background:#00e676;display:inline-block;box-shadow:0 0 8px #00e676;margin-right:5px}
        .live-global-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
        .live-stat{background:#11151e;border:1px solid #242a3a;border-radius:8px;padding:7px}
        .live-stat .lv{font-size:15px;font-weight:800;color:#fff}.live-stat .lk{font-size:8px;color:#747d94;text-transform:uppercase;margin-top:2px;letter-spacing:.5px}

        .journey-wrap{margin-bottom:14px;background:#11151e;border:1px solid #242a3a;border-radius:12px;padding:12px}
        .journey-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
        .journey-title h3{margin:0;font-size:13px}.journey-title span{font-size:9px;color:#7f8799}
        .journey-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
        .journey-card{background:#0d1118;border:1px solid #252b3b;border-radius:10px;padding:10px;min-height:82px;position:relative}
        .journey-card.done{border-color:#00e676;box-shadow:0 0 14px rgba(0,230,118,.12)}
        .journey-card.locked{opacity:.55;filter:grayscale(.35)}
        .journey-icon{font-size:22px;margin-bottom:6px}.journey-name{font-size:10px;font-weight:800}
        .journey-sub{font-size:8px;color:#6f788e;margin-top:2px}.journey-value{position:absolute;right:8px;top:8px;font-size:10px;font-weight:800;color:#00e676}
        .journey-bar{height:4px;background:#202635;border-radius:99px;overflow:hidden;margin-top:8px}.journey-fill{height:100%;background:linear-gradient(90deg,#00e676,#4dd0e1);width:0%}
        @media(max-width:1200px){.journey-grid{grid-template-columns:repeat(4,1fr)}}

        .team-bar { display: flex; align-items: center; gap: 6px; background: #0e1017; padding: 4px 8px; border-radius: 8px; border: 1px solid #232738; }
        .team-slot {
            width: 42px;
            height: 42px;
            background: #181c28;
            border-radius: 6px;
            border: 1px dashed #2f354a;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .team-slot.filled {
            border: 1px solid #00e676;
            background: radial-gradient(circle, #1f2736 0%, #12151f 100%);
        }
        .team-slot img {
            width: 32px;
            height: 32px;
            image-rendering: pixelated;
        }
        .team-slot .lvl-tag {
            position: absolute;
            bottom: 1px;
            right: 2px;
            background: rgba(0,0,0,0.85);
            color: #ffd700;
            font-size: 8px;
            font-weight: bold;
            padding: 0 2px;
            border-radius: 2px;
        }
        .team-slot .hp-bar-bg {
            position: absolute;
            top: 2px;
            left: 3px;
            right: 3px;
            height: 2px;
            background: #333;
            border-radius: 1px;
            overflow: hidden;
        }
        .team-slot .hp-bar-fill {
            height: 100%;
            background: #00e676;
            width: 100%;
        }

        .team-slot.filled { cursor:pointer; }
        .team-slot.filled:hover { transform:translateY(-1px); box-shadow:0 0 12px rgba(0,230,118,.25); }
        .team-label { font-size:9px; color:#7f8799; margin-left:2px; cursor:pointer; }
        .poke-modal-backdrop {
            display:none; position:fixed; inset:0; z-index:5000;
            background:rgba(4,6,10,.72); backdrop-filter:blur(5px);
            align-items:center; justify-content:center;
        }
        .poke-modal-backdrop.open { display:flex; }
        .poke-modal {
            width:min(560px,92vw); background:#11151e; border:1px solid #31384b;
            border-radius:16px; box-shadow:0 24px 80px rgba(0,0,0,.65); overflow:hidden;
        }
        .poke-modal-head {
            display:flex; align-items:center; gap:14px; padding:16px;
            background:linear-gradient(135deg,#17202b,#10131b); border-bottom:1px solid #252c3b;
        }
        .poke-modal-head img { width:88px; height:88px; image-rendering:pixelated; }
        .poke-title { font-size:21px; font-weight:900; }
        .poke-meta { font-size:11px; color:#8791a8; margin-top:3px; }
        .poke-close { margin-left:auto; align-self:flex-start; border:0; border-radius:8px; padding:7px 10px; background:#252b39; color:#fff; cursor:pointer; }
        .poke-body { padding:16px; display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .poke-card { background:#0c1017; border:1px solid #252c3a; border-radius:10px; padding:11px; }
        .poke-card h4 { margin:0 0 9px; font-size:11px; color:#00e676; text-transform:uppercase; letter-spacing:.5px; }
        .poke-hp-text { display:flex; justify-content:space-between; font-size:11px; margin-bottom:6px; }
        .poke-hp-bg { height:8px; background:#262c37; border-radius:99px; overflow:hidden; }
        .poke-hp-fill { height:100%; }
        .poke-stat-row,.poke-move-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:11px; }
        .poke-stat-row:last-child,.poke-move-row:last-child { border-bottom:0; }
        .poke-move-id { color:#677086; font-size:9px; }
        @media(max-width:650px){.poke-body{grid-template-columns:1fr}}

        /* AGENT FILTER CONTROL */
        .filter-control {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #0e1017;
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid #232738;
            font-size: 11px;
        }
        .filter-input {
            width: 44px;
            background: #181c28;
            border: 1px solid #3b4258;
            color: #00e676;
            font-weight: bold;
            font-size: 12px;
            text-align: center;
            border-radius: 4px;
            padding: 2px;
        }
        .filter-btn {
            background: #232738;
            border: none;
            color: #aaa;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 10px;
            cursor: pointer;
        }
        .filter-btn:hover { background: #3b4258; color: #fff; }

        .header-right { display: flex; align-items: center; gap: 10px; }
        .tabs { display: flex; gap: 6px; }
        .tab-btn { background: #232738; color: #aaa; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 11px; transition: all 0.2s; }
        .tab-btn:hover { color: #fff; background: #2f344a; }
        .tab-btn.active { background: #00e676; color: #000; font-weight: 700; }

        #main-container { flex: 1; position: relative; overflow: hidden; }
        #map-view, #rooms-view, #graphs-view { width: 100%; height: 100%; position: absolute; top:0; left:0; }
        #rooms-view { display: none; overflow-y: auto; padding: 24px; }
        #graphs-view { display:none; overflow-y:auto; padding:18px 22px 30px; background:#0c0e14; }
        .graphs-kpis { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:10px; margin-bottom:12px; }
        .graphs-kpi { background:#151821; border:1px solid #232738; border-radius:9px; padding:10px 12px; }
        .graphs-kpi .v { font-size:20px; font-weight:800; color:#00e676; }
        .graphs-kpi .k { font-size:9px; color:#7f879b; text-transform:uppercase; letter-spacing:.6px; margin-top:3px; }
        .graphs-grid { display:grid; grid-template-columns:repeat(2,minmax(320px,1fr)); gap:12px; }
        .graph-card { background:#151821; border:1px solid #232738; border-radius:10px; padding:12px; min-height:300px; }
        .graph-title { font-weight:700; font-size:13px; margin-bottom:2px; }
        .graph-sub { color:#7f879b; font-size:9px; margin-bottom:8px; }
        .graph-canvas-wrap { position:relative; height:245px; }
        .room-grid {
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
            gap:18px;
        }
        .room-card {
            background:#151821;
            border:1px solid #232738;
            border-radius:10px;
            overflow:hidden;
        }
        .room-card .title {
            padding:10px 12px 3px;
            font-weight:700;
            font-size:14px;
            color:#00e676;
        }
        .room-card .sub {
            padding:0 12px 9px;
            color:#8f98ad;
            font-size:10px;
        }
        .room-canvas-wrap {
            padding:10px;
            background:#0e1017;
            border-top:1px solid #232738;
        }
        .room-map-canvas {
            width:100%;
            height:auto;
            display:block;
            image-rendering:pixelated;
            background:#0b0d12;
            border-radius:6px;
        }
        .room-legend {
            display:flex;
            flex-wrap:wrap;
            gap:12px;
            padding:8px 12px 11px;
            color:#aab2c3;
            font-size:10px;
        }
        .room-legend i {
            width:10px;
            height:10px;
            display:inline-block;
            margin-right:4px;
            vertical-align:-1px;
        }
        .room-legend .walked { background:#43a047; border-radius:2px; }
        .room-legend .warp { background:#ffca28; border-radius:50%; }
        .room-legend .edge { background:#66bb6a; height:3px; vertical-align:2px; }

        .hud-overlay { 
            position: absolute; 
            bottom: 20px; 
            right: 20px; 
            background: rgba(21, 24, 33, 0.94); 
            border: 1px solid #232738; 
            padding: 14px; 
            border-radius: 10px; 
            z-index: 1000; 
            font-size: 11px; 
            width: 320px; 
            max-height: 380px; 
            overflow-y: auto; 
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            backdrop-filter: blur(8px);
        }
        .hud-title { font-weight: 700; color: #00e676; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #232738; padding-bottom: 6px; }
        .agent-row { display: flex; justify-content: space-between; padding: 5px 6px; border-bottom: 1px solid rgba(255,255,255,0.05); cursor: pointer; border-radius: 4px; }
        .agent-row:hover { background: rgba(41,121,255,0.12); }
        .agent-row.selected { background: rgba(41,121,255,0.24); outline: 1px solid rgba(41,121,255,0.45); }
        .agent-watcher { color: #00e676; font-weight: 800; background: rgba(0,230,118,0.1); padding: 4px 6px; border-radius: 4px; margin-bottom: 4px; }
        .agent-watcher.selected { background: rgba(0,230,118,0.22); outline: 1px solid rgba(0,230,118,0.45); }
        .agent-badge { background: #232738; padding: 2px 5px; border-radius: 4px; font-size: 9px; color: #888; }
        .agent-color-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:7px; flex:0 0 auto; box-shadow:0 0 0 1px rgba(255,255,255,.18); }
        .agent-name-wrap { display:flex; align-items:center; min-width:0; }
        .focus-hint { font-size:9px; color:#777; margin-top:6px; }
        .step-list { max-height:128px; overflow-y:auto; background:#0e1017; border:1px solid #232738; border-radius:6px; margin-top:6px; }
        .step-row { display:grid; grid-template-columns:34px 1fr 58px; gap:6px; padding:4px 6px; border-bottom:1px solid rgba(255,255,255,.04); font-size:9px; color:#aeb5c7; }
        .step-row:last-child { border-bottom:none; }
        .step-dir { font-weight:800; color:#fff; }
        .step-coord { color:#7f879b; text-align:right; }

        .detail-panel {
            position: absolute;
            left: 20px;
            bottom: 20px;
            width: 390px;
            background: rgba(21,24,33,0.94);
            border: 1px solid #232738;
            border-radius: 10px;
            padding: 12px;
            z-index: 1000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            backdrop-filter: blur(8px);
        }
        .detail-title { display:flex; justify-content:space-between; align-items:center; color:#00e676; font-weight:700; margin-bottom:8px; }
        .detail-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(70px,1fr)); gap:6px; margin-bottom:8px; }
        .detail-stat { background:#0e1017; border:1px solid #232738; border-radius:6px; padding:6px; text-align:center; }
        .detail-stat .v { font-size:13px; font-weight:700; color:#fff; }
        .detail-stat .k { font-size:8px; color:#777; text-transform:uppercase; letter-spacing:.5px; }
        .chart-wrap { background:#0e1017; border:1px solid #232738; border-radius:6px; padding:5px; margin-top:6px; }
        .chart-label { font-size:9px; color:#888; margin:0 0 3px 4px; }
        .mini-chart { width:100%; height:72px; display:block; }
    </style>
<style id="champion-night-style">
#champion-night-card{position:fixed;right:390px;top:72px;z-index:4800;width:285px;padding:10px 12px;border-radius:14px;background:linear-gradient(145deg,rgba(8,17,28,.96),rgba(15,27,41,.93));border:1px solid rgba(68,214,255,.28);box-shadow:0 14px 42px rgba(0,0,0,.34);backdrop-filter:blur(12px);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#champion-night-card.minimized{width:185px}
#champion-night-card.minimized .cn-grid,#champion-night-card.minimized .cn-foot{display:none}
.cn-title{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:10px;font-weight:900;letter-spacing:.65px;color:#55e4a3}
.cn-title-right{display:flex;align-items:center;gap:7px}.cn-toggle{border:0;background:rgba(255,255,255,.08);color:#a9bdd2;border-radius:7px;padding:2px 7px;cursor:pointer;font-size:12px}
.cn-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.cn-grid div{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.065);border-radius:9px;padding:7px 3px;text-align:center}
.cn-grid b{display:block;font-size:13px;color:#f4f8ff}.cn-grid small{font-size:7px;color:#8191a7;text-transform:uppercase}
.cn-foot{margin-top:7px;font-size:7px;color:#74879d}
@media(max-width:1200px){#champion-night-card{right:18px;top:145px}}
</style>
<style id="pkmai-window-tools-style">
.pkmai-float-tools{position:absolute;right:8px;top:7px;display:flex;gap:4px;z-index:30}
.pkmai-float-tools button{border:0;border-radius:7px;background:rgba(255,255,255,.08);color:#b8c7d8;font-size:11px;line-height:1;padding:4px 7px;cursor:pointer}
.pkmai-float-tools button:hover{background:rgba(255,255,255,.15)}
.pkmai-movable{touch-action:none}
.pkmai-minimized{height:38px!important;min-height:38px!important;overflow:hidden!important}
.pkmai-hidden{display:none!important}
#pkmai-hidden-tray{position:fixed;left:10px;bottom:10px;z-index:6000;display:flex;gap:6px;flex-wrap:wrap;max-width:calc(100vw - 20px)}
#pkmai-hidden-tray button{border:1px solid rgba(77,208,225,.25);background:rgba(10,18,28,.94);color:#aee8dd;border-radius:9px;padding:7px 10px;font-size:10px;cursor:pointer}
@media(max-width:820px){
  body{overflow:auto!important}
  #map-view,#rooms-view,#graphs-view{position:relative!important;min-height:calc(100vh - 54px)!important;height:auto!important}
  .pkmai-movable{position:relative!important;left:auto!important;right:auto!important;top:auto!important;bottom:auto!important;width:calc(100% - 16px)!important;max-width:none!important;margin:8px!important;transform:none!important}
  #champion-night-card{position:relative!important;right:auto!important;top:auto!important;width:calc(100% - 16px)!important;margin:8px!important}
  .graphs-grid{grid-template-columns:1fr!important}
  .journey-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
}
@media(max-width:480px){
  .journey-grid{grid-template-columns:1fr!important}
  .graph-card{min-height:320px!important}
}
</style>
<style id="learner-truth-style">
#learner-truth-hud{position:fixed;left:16px;bottom:16px;z-index:4900;width:300px;padding:10px 12px;border-radius:13px;background:rgba(7,14,23,.96);border:1px solid rgba(82,222,172,.30);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#learner-truth-hud.minimized .lth-body{display:none}.lth-head{display:flex;justify-content:space-between;align-items:center;color:#58e3a5;font-size:10px;font-weight:900}.lth-head button{border:0;background:rgba(255,255,255,.08);color:#b7c7d8;border-radius:7px;padding:2px 7px}
.lth-body{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}.lth-cell{background:rgba(255,255,255,.045);border-radius:8px;padding:7px 5px;text-align:center}.lth-cell b{display:block;font-size:13px;color:#f5f8fc}.lth-cell small{font-size:7px;color:#8393a8;text-transform:uppercase}
@media(max-width:820px){#learner-truth-hud{position:relative!important;left:auto!important;bottom:auto!important;width:calc(100% - 16px)!important;margin:8px!important}}
</style>
</head>
<body>
<div id="learner-truth-hud"><div class="lth-head"><span>🧠 TRAINER · LIVE</span><button onclick="document.getElementById('learner-truth-hud').classList.toggle('minimized')">−</button></div><div class="lth-body"><div class="lth-cell"><b id="lth-learner">0</b><small>Learner Steps</small></div><div class="lth-cell"><b id="lth-champion">0</b><small>Champion Steps</small></div><div class="lth-cell"><b id="lth-delta">0</b><small>Seit Champion</small></div></div></div>
<div id="champion-night-card">
  <div class="cn-title"><span>🏆 FRONTIER CHAMPION</span><span class="cn-title-right"><span id="cn-ver">v0</span><button class="cn-toggle" onclick="document.getElementById('champion-night-card').classList.toggle('minimized')">−</button></span></div>
  <div class="cn-grid">
    <div><b id="cn-steps">0</b><small>Steps</small></div>
    <div><b id="cn-full">0%</b><small>Full Exit</small></div>
    <div><b id="cn-starter">0%</b><small>Full Starter</small></div>
    <div><b id="cn-badge">0</b><small>Badge</small></div>
  </div>
  <div class="cn-foot">30 Episoden + 4 Full · neue Full-Meilensteine sofort</div>
</div>
    <header>
        <div class="header-left">
            <div class="logo-title">⚡ PKMAI <span id="model-ver" style="color:#2979ff;">v000001</span></div>
            
            <div style="font-size: 12px; color:#888;">Trainer: <b id="hud-trainer" style="color:#00e676;">Alex</b></div>

            <span class="team-label" onclick="openFirstPokemon()">POKÉMON TEAM</span>
            <div class="team-bar">
                <div class="team-slot" id="slot-0"><span style="font-size: 9px; color: #444;">1</span></div>
                <div class="team-slot" id="slot-1"><span style="font-size: 9px; color: #444;">2</span></div>
                <div class="team-slot" id="slot-2"><span style="font-size: 9px; color: #444;">3</span></div>
                <div class="team-slot" id="slot-3"><span style="font-size: 9px; color: #444;">4</span></div>
                <div class="team-slot" id="slot-4"><span style="font-size: 9px; color: #444;">5</span></div>
                <div class="team-slot" id="slot-5"><span style="font-size: 9px; color: #444;">6</span></div>
            </div>

            <div class="badge-bar" title="Kanto Orden">
                <div class="badge-slot" id="badge-1" title="Felsorden">🪨</div>
                <div class="badge-slot" id="badge-2" title="Quellorden">💧</div>
                <div class="badge-slot" id="badge-3" title="Donnerorden">⚡</div>
                <div class="badge-slot" id="badge-4" title="Farborden">🌿</div>
                <div class="badge-slot" id="badge-5" title="Seelenorden">☠️</div>
                <div class="badge-slot" id="badge-6" title="Sumpforden">🔮</div>
                <div class="badge-slot" id="badge-7" title="Vulkanorden">🔥</div>
                <div class="badge-slot" id="badge-8" title="Erdorden">🌍</div>
            </div>

            <!-- AGENTEN FILTER INPUT -->
            <div class="filter-control">
                <span>Zeige Agenten:</span>
                <input type="number" id="agent-limit" class="filter-input" value="120" min="0" max="120" onchange="updateFilter(this.value)">
                <button class="filter-btn" onclick="setFilter(5)">5</button>
                <button class="filter-btn" onclick="setFilter(10)">10</button>
                <button class="filter-btn" onclick="setFilter(20)">20</button>
                <button class="filter-btn" onclick="setFilter(120)">Alle 120</button>
            </div>
        </div>

        <div class="header-right">
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('map', event)">🗺️ Overworld Map</button>
                <button class="tab-btn" onclick="showTab('rooms', event)">🏠 Indoor Mapping</button>
                <button class="tab-btn" onclick="showTab('graphs', event)">📈 Graphs</button>
            </div>
        </div>
    </header>

    <div id="main-container">
        <div id="map-view"><div class="v81skills"><div style="font-size:11px;font-weight:800;margin-bottom:6px">V8.4 TRAINING SKILLS · 120 AGENTS</div><div class="v81grid" id="v81skills"></div></div>
            <div class="live-global">
                <div class="live-global-title"><span><span class="live-dot"></span>GLOBAL AI</span><span id="live-model">v0</span></div>
                <div class="live-global-grid">
                    <div class="live-stat"><div class="lv" id="live-maps">0</div><div class="lk">🌍 Maps</div></div>
                    <div class="live-stat"><div class="lv" id="live-warps">0</div><div class="lk">🚪 Warps</div></div>
                    <div class="live-stat"><div class="lv" id="live-edges">0</div><div class="lk">🧭 Edges</div></div>
                    <div class="live-stat"><div class="lv" id="live-battles">0</div><div class="lk">⚔ Battles</div></div>
                    <div class="live-stat"><div class="lv" id="live-finished">0</div><div class="lk">✅ Finished</div></div>
                    <div class="live-stat"><div class="lv" id="live-steps">0</div><div class="lk">🧠 PPO Steps</div></div>
                </div>
            </div>
        </div>
        <div id="rooms-view"><div class="room-grid" id="room-grid"></div></div>
        <div id="graphs-view"><div class="v84-skill-panel" id="v84-skill-panel">
            <div class="v84-head"><b>V8.4 – Trainings-Skills · 120 Agents</b><span>direkt aus agent_00…119 Training-Stats</span></div>
            <div class="v84-grid" id="v84-skill-grid"></div>
            <div class="v84-meta" id="v84-meta"></div>
            <div class="v84-canvas-wrap"><canvas id="v84-skill-canvas" width="1500" height="300"></canvas></div>
        </div><div class="graph-card" style="min-height:390px;margin-bottom:12px">
            <div class="graph-title">V8.3 – Vollständige Skill-Statistik</div>
            <div class="graph-sub">Intro → Treppe → Exit → Erstes Pokémon → Battle → Level → Gym + Full Chain</div>
            <div id="v83-skill-table" style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:12px 0"></div>
            <canvas id="v83-skill-canvas" width="1200" height="250" style="width:100%;height:250px"></canvas>
        </div><div class="graph-card" style="min-height:350px;margin-bottom:12px"><div class="graph-title">V8.1 – Alle Skill-Raten</div><div class="graph-sub">Spezialisten + Full-Chain</div><div class="graph-canvas-wrap" style="height:300px"><canvas id="v81skillschart"></canvas></div></div><div class="v8-skill-panel"><div class="v8-skill-title"><b>V8 Skill Matrix</b><span>direkt aus Training-Stats · nicht Watcher</span></div><div class="v8-skill-cards" id="v8-skill-cards"></div><div class="v8-skill-chart-wrap"><canvas id="v8-skill-chart"></canvas></div></div>
            <div class="journey-wrap">
                <div class="journey-title"><h3>🚀 Journey Skills</h3><span>Early Game → Vertania → Wald → Orden 1</span></div>
                <div class="journey-grid">
                    <div class="journey-card" id="journey-starter"><div class="journey-value" id="jv-starter">0%</div><div class="journey-icon">🐣</div><div class="journey-name">Starter</div><div class="journey-sub">Starter zuverlässig erhalten</div><div class="journey-bar"><div class="journey-fill" id="jf-starter"></div></div></div>
                    <div class="journey-card" id="journey-battle"><div class="journey-value" id="jv-battle">0%</div><div class="journey-icon">⚔️</div><div class="journey-name">Battles</div><div class="journey-sub">Kämpfe starten & beenden</div><div class="journey-bar"><div class="journey-fill" id="jf-battle"></div></div></div>
                    <div class="journey-card" id="journey-map5"><div class="journey-value" id="jv-map5">0/5</div><div class="journey-icon">🗺️</div><div class="journey-name">5 Maps</div><div class="journey-sub">Global erkannte Maps</div><div class="journey-bar"><div class="journey-fill" id="jf-map5"></div></div></div>
                    <div class="journey-card" id="journey-warps"><div class="journey-value" id="jv-warps">0/5</div><div class="journey-icon">🚪</div><div class="journey-name">Warps</div><div class="journey-sub">Übergänge entdeckt</div><div class="journey-bar"><div class="journey-fill" id="jf-warps"></div></div></div>
                    <div class="journey-card" id="journey-progress"><div class="journey-value" id="jv-progress">0</div><div class="journey-icon">🌉</div><div class="journey-name">Progress Bridge</div><div class="journey-sub">Fortschritt-Checkpoints</div><div class="journey-bar"><div class="journey-fill" id="jf-progress"></div></div></div>
                    <div class="journey-card" id="journey-map10"><div class="journey-value" id="jv-map10">0/10</div><div class="journey-icon">🌲</div><div class="journey-name">Forest Push</div><div class="journey-sub">Über die ersten Maps hinaus</div><div class="journey-bar"><div class="journey-fill" id="jf-map10"></div></div></div>
                    <div class="journey-card locked" id="journey-badge1"><div class="journey-value" id="jv-badge1">LOCKED</div><div class="journey-icon">🪨</div><div class="journey-name">Orden 1</div><div class="journey-sub">Felsorden</div><div class="journey-bar"><div class="journey-fill" id="jf-badge1"></div></div></div>
                </div>
            </div>
            <div class="graphs-kpis">
                <div class="graphs-kpi"><div class="v" id="g-steps">0</div><div class="k">PPO Steps</div></div>
                <div class="graphs-kpi"><div class="v" id="g-version">v0</div><div class="k">Modell</div></div>
                <div class="graphs-kpi"><div class="v" id="g-episodes">0 / 0</div><div class="k">Beginning / Alle Episoden</div></div>
                <div class="graphs-kpi"><div class="v" id="g-avgreward">0</div><div class="k">Ø Reward</div></div>
                <div class="graphs-kpi"><div class="v" id="g-house">0%</div><div class="k">Intro Skill</div></div>
                <div class="graphs-kpi"><div class="v" id="g-starter">0%</div><div class="k">Treppen Skill</div></div>
            </div>
            <div class="graphs-grid">
                <div class="graph-card"><div class="graph-title">Lernkurve</div><div class="graph-sub">Ø Episode-Reward über echte PPO-Trainingsschritte.</div><div class="graph-canvas-wrap"><canvas id="graph-reward"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">V9.0 – Clean Night 120</div><div class="graph-sub">Kumulative Erfolgsquote aller Spezialisten + Full-Chain. Sinkende Linien = neue Runs sind aktuell schwächer als der bisherige Durchschnitt.</div><div class="graph-canvas-wrap"><canvas id="graph-success"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">Spiel-Fortschritt</div><div class="graph-sub">Bestes Level, Orden und Maps je Modellstand.</div><div class="graph-canvas-wrap"><canvas id="graph-progress"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">Festfahren / Anti-Loop</div><div class="graph-sub">Loops pro 100 echte Beginning-Runs; Curriculum wird separat gezählt.</div><div class="graph-canvas-wrap"><canvas id="graph-loops"></canvas></div></div>
            </div>
        </div>
        <div class="hud-overlay" id="hud">Lade 35 Agenten...</div>

        <div class="detail-panel" id="detail-panel">
            <div class="detail-title">
                <span id="detail-name">Watcher</span>
                <span id="detail-room" style="font-size:9px;color:#777;">-</span>
            </div>
            <div class="detail-stats">
                <div class="detail-stat"><div class="v" id="detail-steps">0</div><div class="k">Steps</div></div>
                <div class="detail-stat"><div class="v" id="detail-reward">0</div><div class="k">Episode Reward</div></div>
                <div class="detail-stat"><div class="v" id="detail-level">0</div><div class="k">Level</div></div>
                <div class="detail-stat"><div class="v" id="detail-maps">0</div><div class="k">Episode Maps</div></div>
                <div class="detail-stat"><div class="v" id="detail-edges">0</div><div class="k">Known Edges</div></div>
                <div class="detail-stat"><div class="v" id="detail-knownmaps">0</div><div class="k">Known Maps</div></div>
                <div class="detail-stat"><div class="v" id="detail-transitions">0</div><div class="k">Transitions</div></div>
                <div class="detail-stat"><div class="v" id="detail-battles">0</div><div class="k">⚔ Battles</div></div>
                <div class="detail-stat"><div class="v" id="detail-battle-done">0</div><div class="k">✅ Finished</div></div>
            </div>
            <div class="chart-wrap">
                <div class="chart-label">Letztes Reward-Event</div>
                <div id="detail-event" style="font-size:12px;color:#00e676;padding:6px 8px;">-</div>
            </div>
            <div class="v8-agent-party" id="v8-agent-party">
                <div class="v8-agent-party-title">
                    <span>Selected Agent Team</span>
                    <span id="v8-agent-role">-</span>
                </div>
                <div class="v8-party-grid" id="v8-party-grid">
                    <div class="v8-mon"><span>Agent anklicken</span></div>
                </div>
            </div>
            <div class="v8-reward-breakdown" id="v8-reward-breakdown"></div>
            <div class="chart-wrap">
                <div class="chart-label">Reward-Verlauf</div>
                <canvas id="reward-chart" class="mini-chart" width="360" height="72"></canvas>
            </div>
            <div class="chart-wrap">
                <div class="chart-label">Steps / Lernfortschritt</div>
                <canvas id="steps-chart" class="mini-chart" width="360" height="72"></canvas>
            </div>
            <div class="chart-wrap">
                <div class="chart-label">Letzte 1-Tile-Schritte</div>
                <div class="step-list" id="step-list"><div class="step-row"><span>-</span><span>warte auf Route</span><span></span></div></div>
            </div>
        </div>
    </div>

    <div class="poke-modal-backdrop" id="poke-modal-bg" onclick="closePokemonModal(event)">
        <div class="poke-modal" onclick="event.stopPropagation()">
            <div class="poke-modal-head">
                <img id="poke-detail-sprite" src="" alt="">
                <div>
                    <div class="poke-title" id="poke-detail-name">Pokémon</div>
                    <div class="poke-meta" id="poke-detail-meta">Lv. ?</div>
                </div>
                <button class="poke-close" onclick="closePokemonModal()">✕</button>
            </div>
            <div class="poke-body">
                <div class="poke-card">
                    <h4>❤️ Health</h4>
                    <div class="poke-hp-text"><span id="poke-detail-hp">0 / 0 HP</span><span id="poke-detail-hp-state">—</span></div>
                    <div class="poke-hp-bg"><div class="poke-hp-fill" id="poke-detail-hp-fill"></div></div>
                    <div style="margin-top:10px;font-size:11px;display:flex;justify-content:space-between">
                        <span style="color:#7f8799">✨ Total EXP</span>
                        <b id="poke-detail-exp">0</b>
                    </div>
                    <div style="margin-top:4px;font-size:10px;display:flex;justify-content:space-between">
                        <span style="color:#7f8799">Letzter EXP Gain</span>
                        <b id="poke-detail-exp-delta" style="color:#00e676">+0</b>
                    </div>
                </div>
                <div class="poke-card">
                    <h4>📊 Stats</h4>
                    <div id="poke-detail-stats"></div>
                </div>
                <div class="poke-card" style="grid-column:1/-1">
                    <h4>⚔️ Moves</h4>
                    <div id="poke-detail-moves"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let maxVisibleAgents = 120;
        let selectedAgentId = null; // null = alle Agenten sichtbar.
        let latestInstances = [];
        const historyByAgent = {};

        function setFilter(n) {
            document.getElementById('agent-limit').value = n;
            maxVisibleAgents = parseInt(n);
        }

        function updateFilter(val) {
            maxVisibleAgents = parseInt(val) || 0;
        }

        const map = L.map('map-view', {
            crs: L.CRS.Simple,
            minZoom: -3,
            maxZoom: 3,
            zoomControl: true,
            attributionControl: false
        });
        const bounds = [[0, 0], [3000, 3000]];
        map.setMaxBounds([[-500,-500],[3500,3500]]);
        map.fitBounds([[900, 1100], [2600, 2050]]);

        // Overworld uses the normal Kanto image only.
        const overworldBackground = L.imageOverlay(
            '/map.png',
            [[0, 0], [3000, 3000]],
            { opacity: 0.62, interactive: false }
        ).addTo(map);

        const MAP_OFFSETS = {
            '3,0': [1410, 2320],
            '3,19': [1500, 1850],
            '3,1': [1500, 1350],
            '3,20': [1500, 950],
            '3,2': [1500, 480],
            '4,0': [1410, 2320],
            '4,1': [1410, 2320],
            '4,2': [1580, 2490],
            '4,3': [1590, 2320],
            '5,0': [1350, 1350],
            '5,1': [1650, 1350],
        };

        let agentMarkers = {};
        let agentPolylines = {};
        let persistentExplorationLayers = {};
        let persistentExplorationSignatures = {};
        let persistentEdgeLayers = {};
        let persistentTransitionLayers = {};
        let skeletonRects = {};
        let skeletonTransitionLines = [];

        let agentStepDots = {};
        let globalWarpLayer = L.layerGroup().addTo(map);
        let globalMappingSignature = "";

        function agentColor(id) {
            if (Number(id) === 99) return '#00e676';
            const hue = (Number(id) * 137.508) % 360;
            return `hsl(${hue.toFixed(1)}, 78%, 58%)`;
        }

        function isAgentVisible(id, isWatcher, renderedTrainCount) {
            if (selectedAgentId !== null) {
                return Number(id) === Number(selectedAgentId);
            }
            return isWatcher || renderedTrainCount < maxVisibleAgents;
        }

        function getLeafletCoords(bank, mapId, x, y) {
            const key = bank + ',' + mapId;
            const base = MAP_OFFSETS[key] || [1410, 2320];
            const px = base[0] + (x * 12);
            const py = base[1] + (y * 12);
            return [3000 - py, px];
        }

        let currentTab = 'map';

        function showTab(t, e) {
            currentTab = t;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (e && e.target) e.target.classList.add('active');

            document.getElementById('map-view').style.display = t === 'map' ? 'block' : 'none';
            document.getElementById('rooms-view').style.display = t === 'rooms' ? 'block' : 'none';
            document.getElementById('graphs-view').style.display = t === 'graphs' ? 'block' : 'none';

            const showOverlays = t === 'map';
            document.getElementById('hud').style.display = showOverlays ? 'block' : 'none';
            document.getElementById('detail-panel').style.display = showOverlays ? 'block' : 'none';

            if (t === 'rooms') {
                updateGlobalMapping(true);
            }
            if (t === 'graphs') loadTrainingGraphs();
            if (t === 'map') {
                updateGlobalMapping(true);
                setTimeout(() => map.invalidateSize(), 50);
            }
        }

        async function updateSkeleton() {
            try {
                const res = await fetch('/api/skeleton?t=' + Date.now());
                const sk = await res.json();
                const seen = new Set();

                (sk.maps || []).forEach(m => {
                    const key = `${m.bank},${m.map_id}`;
                    seen.add(key);

                    const p1 = getLeafletCoords(
                        m.bank, m.map_id, m.min_x, m.min_y
                    );
                    const p2 = getLeafletCoords(
                        m.bank, m.map_id, m.max_x + 1, m.max_y + 1
                    );

                    const bounds = [
                        [Math.min(p1[0], p2[0]), Math.min(p1[1], p2[1])],
                        [Math.max(p1[0], p2[0]), Math.max(p1[1], p2[1])]
                    ];

                    const label =
                        `Bank ${m.bank} / Map ${m.map_id}<br>` +
                        `${m.width_tiles}×${m.height_tiles} Tiles<br>` +
                        `${(m.agents || []).length} Agents`;

                    if (!skeletonRects[key]) {
                        skeletonRects[key] = L.rectangle(bounds, {
                            color: '#ffd54f',
                            weight: 1.5,
                            opacity: 0.8,
                            fillColor: '#ffd54f',
                            fillOpacity: 0.035,
                            dashArray: '6,5',
                            interactive: true
                        }).bindTooltip(label).addTo(map);
                    } else {
                        skeletonRects[key].setBounds(bounds);
                        skeletonRects[key].setTooltipContent(label);
                    }
                });

                Object.keys(skeletonRects).forEach(key => {
                    if (!seen.has(key)) {
                        map.removeLayer(skeletonRects[key]);
                        delete skeletonRects[key];
                    }
                });

                skeletonTransitionLines.forEach(line => map.removeLayer(line));
                skeletonTransitionLines = [];

                (sk.transitions || []).forEach(t => {
                    const fromKey = `${t.from_bank},${t.from_map}`;
                    const toKey = `${t.to_bank},${t.to_map}`;
                    const a = skeletonRects[fromKey];
                    const b = skeletonRects[toKey];
                    if (!a || !b) return;

                    const ca = a.getBounds().getCenter();
                    const cb = b.getBounds().getCenter();
                    const line = L.polyline([ca, cb], {
                        color: '#ffca28',
                        weight: 1,
                        opacity: 0.25,
                        dashArray: '3,6',
                        interactive: false
                    }).addTo(map);
                    skeletonTransitionLines.push(line);
                });
            } catch(e) {}
        }
        setInterval(updateSkeleton, 5000);
        updateSkeleton();

        function getSelectedInstance() {
            if (selectedAgentId !== null) {
                return latestInstances.find(i => Number(i.id) === Number(selectedAgentId))
                    || null;
            }
            // Ohne Kartenfokus zeigt das Detailpanel weiterhin den Watcher.
            return latestInstances.find(i => Number(i.id) === 99)
                || latestInstances[0]
                || null;
        }

        function selectAgent(id) {
            const n = Number(id);
            selectedAgentId = (selectedAgentId === n) ? null : n;
            renderSelectedAgent();
            updatePersistentExploration(true);
            updateDashboard();
        }

        let latestParty = [];

        function hpState(hpPct) {
            if (hpPct <= 0) return ['FAINTED','#ff1744'];
            if (hpPct < 30) return ['CRITICAL','#ff1744'];
            if (hpPct < 60) return ['CAUTION','#ffea00'];
            return ['HEALTHY','#00e676'];
        }

        function openPokemonDetail(index) {
            const p = latestParty[index];
            if (!p) return;

            const spriteUrl = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/${p.id}.png`;
            const curHp = Number(p.cur_hp || 0);
            const maxHp = Number(p.max_hp || 0);
            const hpPct = maxHp > 0 ? Math.max(0,Math.min(100,(curHp/maxHp)*100)) : 0;
            const [stateName,stateColor] = hpState(hpPct);
            const stats = p.stats || {};
            const moves = Array.isArray(p.moves) ? p.moves : [];

            document.getElementById('poke-detail-sprite').src = spriteUrl;
            document.getElementById('poke-detail-name').innerText = p.name || `Species #${p.id || '?'}`;
            document.getElementById('poke-detail-meta').innerText =
                `Lv. ${p.level || '?'} · Slot ${Number(p.slot || 0)+1} · Species #${p.id || '?'}`;
            document.getElementById('poke-detail-hp').innerText = `${curHp} / ${maxHp} HP`;
            document.getElementById('poke-detail-exp').innerText =
                Number(p.experience || 0).toLocaleString();
            document.getElementById('poke-detail-exp-delta').innerText =
                `+${Number(p.exp_delta || 0).toLocaleString()}`;
            const stateEl=document.getElementById('poke-detail-hp-state');
            stateEl.innerText=stateName; stateEl.style.color=stateColor;
            const hpFill=document.getElementById('poke-detail-hp-fill');
            hpFill.style.width=`${hpPct}%`; hpFill.style.background=stateColor;

            document.getElementById('poke-detail-stats').innerHTML = [
                ['Attack', stats.attack],
                ['Defense', stats.defense],
                ['Speed', stats.speed],
                ['Sp. Attack', stats.sp_attack],
                ['Sp. Defense', stats.sp_defense],
            ].map(([n,v])=>`<div class="poke-stat-row"><span>${n}</span><b>${v ?? '—'}</b></div>`).join('');

            document.getElementById('poke-detail-moves').innerHTML =
                moves.length
                ? moves.map(m=>`<div class="poke-move-row"><span>${m.name || 'Move'} <span class="poke-move-id">#${m.id || '?'}</span></span><b>PP ${m.pp ?? '—'}</b></div>`).join('')
                : '<div style="font-size:11px;color:#727b90">Noch keine Move-Telemetrie verfügbar.</div>';

            document.getElementById('poke-modal-bg').classList.add('open');
        }

        function openFirstPokemon() {
            if (latestParty.length) openPokemonDetail(0);
        }

        function closePokemonModal(ev) {
            if (ev && ev.target && ev.target.id !== 'poke-modal-bg') return;
            document.getElementById('poke-modal-bg').classList.remove('open');
        }

        function updateParty(party) {
            party = party || [];
            latestParty = party;
            for (let i = 0; i < 6; i++) {
                const slot = document.getElementById(`slot-${i}`);
                if (i < party.length) {
                    const p = party[i];
                    const spriteUrl = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/${p.id}.png`;
                    const maxHp = Number(p.max_hp || 0);
                    const curHp = Number(p.cur_hp || 0);
                    const hpPct = maxHp > 0 ? Math.max(0, Math.min(100, Math.round((curHp / maxHp) * 100))) : 100;
                    const hpColor = hpPct > 50 ? '#00e676' : (hpPct > 20 ? '#ffea00' : '#ff1744');

                    slot.className = 'team-slot filled';
                    slot.onclick = () => openPokemonDetail(i);
                    slot.title = `${p.name || 'Pokemon'} (Lv. ${p.level || '?'}) - HP: ${curHp}/${maxHp} - EXP: ${Number(p.experience || 0).toLocaleString()}`;
                    slot.innerHTML = `
                        <div class="hp-bar-bg"><div class="hp-bar-fill" style="width:${hpPct}%; background:${hpColor};"></div></div>
                        <img src="${spriteUrl}" alt="${p.name || 'Pokemon'}">
                        <div class="lvl-tag">Lv.${p.level || '?'}</div>
                    `;
                } else {
                    slot.className = 'team-slot';
                    slot.onclick = null;
                    slot.title = `Slot ${i+1} (Leer / keine Party-Telemetrie)`;
                    slot.innerHTML = `<span style="font-size: 9px; color: #444;">${i+1}</span>`;
                }
            }
        }

        function drawMiniChart(canvasId, values) {
            const c = document.getElementById(canvasId);
            if (!c) return;
            const ctx = c.getContext('2d');
            const w = c.width, h = c.height;
            ctx.clearRect(0, 0, w, h);

            ctx.strokeStyle = '#242a3b';
            ctx.lineWidth = 1;
            for (let y = 18; y < h; y += 18) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            if (!values || values.length < 2) return;
            const min = Math.min(...values);
            const max = Math.max(...values);
            const span = Math.max(1e-9, max - min);

            ctx.strokeStyle = '#00e676';
            ctx.lineWidth = 2;
            ctx.beginPath();
            values.forEach((v, idx) => {
                const x = idx * (w / Math.max(1, values.length - 1));
                const y = h - 4 - ((v - min) / span) * (h - 8);
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        function pushHistory(inst) {
            if (!inst) return;
            if (!historyByAgent[inst.id]) {
                historyByAgent[inst.id] = { reward: [], steps: [], lastSteps: null };
            }
            const h = historyByAgent[inst.id];
            const steps = Number(inst.steps || 0);
            const reward = Number(inst.reward || 0);

            // Nur einen neuen Punkt speichern, wenn sich die Steps geaendert haben.
            if (h.lastSteps !== steps) {
                h.steps.push(steps);
                h.reward.push(reward);
                h.lastSteps = steps;
                if (h.steps.length > 180) h.steps.shift();
                if (h.reward.length > 180) h.reward.shift();
            }
        }

        function renderSelectedAgent() {
            const inst = getSelectedInstance();
            if (!inst) return;

            document.getElementById('hud-trainer').innerText =
                (inst.name || 'Unknown').replace(' (Watcher)', '');

            const party = inst.party || [];
            updateParty(party);

            const badges = Number(inst.badges || 0);
            for (let i = 1; i <= 8; i++) {
                const el = document.getElementById(`badge-${i}`);
                if (i <= badges) el.classList.add('active');
                else el.classList.remove('active');
            }

            document.getElementById('detail-name').innerText = inst.name || `Agent ${inst.id}`;
            document.getElementById('detail-room').innerText = inst.room || `Bank ${inst.bank} / Map ${inst.map}`;
            document.getElementById('detail-steps').innerText = Number(inst.steps || 0).toLocaleString();
            document.getElementById('detail-reward').innerText = Number(inst.reward || 0).toFixed(2);
            document.getElementById('detail-level').innerText = Number(inst.level || 0);
            document.getElementById('detail-maps').innerText = Number(inst.visited_maps || 0);

            const pe = inst.persistent_exploration || {};
            document.getElementById('detail-edges').innerText =
                Number(pe.known_edges || 0).toLocaleString();
            document.getElementById('detail-knownmaps').innerText =
                Number(pe.known_maps || 0).toLocaleString();
            document.getElementById('detail-transitions').innerText =
                Number(pe.known_transitions || 0).toLocaleString();
            const bs = inst.battle_stats || {};
            document.getElementById('detail-battles').innerText = Number(bs.started || 0).toLocaleString();
            document.getElementById('detail-battle-done').innerText = Number(bs.completed || 0).toLocaleString();

            const navState = pe.exit_seek_active
                ? `EXIT SEEK (${Number(pe.steps_since_new_edge || 0)} stale)`
                : `EXPLORE (${Number(pe.steps_since_new_edge || 0)} stale)`;

            const events = Array.isArray(inst.reward_events) ? inst.reward_events : [];
            document.getElementById('detail-event').innerText =
                `${navState} | ${events.length ? events[events.length - 1] : '-'}`;

            const h = historyByAgent[inst.id] || {reward:[], steps:[]};
            drawMiniChart('reward-chart', h.reward);
            drawMiniChart('steps-chart', h.steps);

            const stepList = document.getElementById('step-list');
            const path = Array.isArray(inst.path) ? inst.path : [];
            const rows = [];
            const start = Math.max(1, path.length - 24);
            for (let idx = start; idx < path.length; idx++) {
                const a = path[idx - 1];
                const b = path[idx];
                if (!a || !b || a.length < 4 || b.length < 4) continue;

                const sameMap = Number(a[0]) === Number(b[0]) && Number(a[1]) === Number(b[1]);
                const dx = Number(b[2]) - Number(a[2]);
                const dy = Number(b[3]) - Number(a[3]);
                let dir = 'MAP';
                if (sameMap && Math.abs(dx) + Math.abs(dy) === 1) {
                    if (dx === 1) dir = '→';
                    else if (dx === -1) dir = '←';
                    else if (dy === 1) dir = '↓';
                    else if (dy === -1) dir = '↑';
                } else if (sameMap) {
                    dir = '·';
                }
                rows.push(
                    `<div class="step-row">` +
                    `<span class="step-dir">${dir}</span>` +
                    `<span>B${b[0]} M${b[1]}</span>` +
                    `<span class="step-coord">${b[2]},${b[3]}</span>` +
                    `</div>`
                );
            }
            stepList.innerHTML = rows.length
                ? rows.reverse().join('')
                : '<div class="step-row"><span>-</span><span>noch keine 1-Tile-Schritte</span><span></span></div>';
        }

        const trainingCharts = {};

        function graphOptions(percent=false) {
            return {
                responsive:true, maintainAspectRatio:false, animation:false,
                interaction:{mode:'index',intersect:false},
                plugins:{legend:{labels:{color:'#b8bfd0',boxWidth:12}}},
                scales:{
                    x:{ticks:{color:'#798196',maxTicksLimit:10},grid:{color:'rgba(255,255,255,.045)'},title:{display:true,text:'PPO Steps',color:'#798196'}},
                    y:{beginAtZero:percent,suggestedMax:percent?100:undefined,ticks:{color:'#798196',callback:percent?(v=>v+'%'):undefined},grid:{color:'rgba(255,255,255,.045)'}}
                }
            };
        }

        function upsertTrainingChart(id, labels, datasets, percent=false) {
            const canvas=document.getElementById(id);
            if (!canvas || typeof Chart==='undefined') return;
            if (trainingCharts[id]) {
                trainingCharts[id].data.labels=labels;
                trainingCharts[id].data.datasets=datasets;
                trainingCharts[id].update('none');
            } else {
                trainingCharts[id]=new Chart(canvas,{type:'line',data:{labels,datasets},options:graphOptions(percent)});
            }
        }

        function setJourneyCard(id,text,pct,done=false){
            const card=document.getElementById('journey-'+id), val=document.getElementById('jv-'+id), fill=document.getElementById('jf-'+id);
            if(val) val.innerText=text;
            if(fill) fill.style.width=`${Math.max(0,Math.min(100,pct))}%`;
            if(card){card.classList.toggle('done',!!done);if(done)card.classList.remove('locked');}
        }
        function updateJourneySkills(state){
            const st=state.training_stats||{}, rt=st.run_totals||{}, gx=state.global_exploration||{}, bs=state.battle_stats||{}, jr=state.journey_stats||{};
            const fullRuns=Number(rt.v2_full_episodes||0), fullStarter=Number(rt.v2_full_starter||0);
            const starterPct=fullRuns>0?100*fullStarter/fullRuns:0;
            const battles=Number(bs.started||0), completed=Number(bs.completed||0);
            const battlePct=battles>0?100*completed/battles:0;
            const maps=Number(gx.known_maps||0), warps=Number(gx.known_transitions||0), progress=Number(jr.progress||0);
            const badge1=Number(state.max_badges||0)>=1||Number(jr.badge1||0)>0;
            setJourneyCard('starter',`${starterPct.toFixed(1)}%`,starterPct,starterPct>=90);
            setJourneyCard('battle',`${battlePct.toFixed(1)}%`,battlePct,battles>=3&&battlePct>=70);
            setJourneyCard('map5',`${maps}/5`,100*maps/5,maps>=5);
            setJourneyCard('warps',`${warps}/5`,100*warps/5,warps>=5);
            setJourneyCard('progress',`${progress}`,Math.min(100,progress*12.5),progress>=3);
            setJourneyCard('map10',`${maps}/10`,100*maps/10,maps>=10);
            setJourneyCard('badge1',badge1?'DONE':'LOCKED',badge1?100:0,badge1);
        }

        async function loadTrainingGraphs() {
            try {
                const [sr,hr]=await Promise.all([
                    fetch('/api/state?t='+Date.now()),
                    fetch('/api/history?t='+Date.now())
                ]);
                const state=await sr.json();
                const histPayload=await hr.json();
                const hist=histPayload.history||[];
                updateJourneySkills(state);
                const st=state.training_stats||{};
                const rates=st.beginning_success_rates||{};
                const skillRates=st.v6_skill_rates||{};

                document.getElementById('g-steps').innerText=Number(state.training_timesteps||0).toLocaleString();
                document.getElementById('g-version').innerText=`v${String(state.version||0).padStart(6,'0')}`;
                const rt=st.run_totals||{};
                const skillRuns =
                    Number(rt.v2_intro_episodes||0) +
                    Number(rt.v2_stairs_episodes||0) +
                    Number(rt.v2_exit_episodes||0);
                document.getElementById('g-episodes').innerText=
                    `${skillRuns.toLocaleString()} / ${Number(rt.all_episodes||0).toLocaleString()}`;
                document.getElementById('g-avgreward').innerText=Number(st.avg_episode_reward||0).toFixed(1);
                document.getElementById('g-house').innerText=`${Number(skillRates.intro||0).toFixed(1)}%`;
                document.getElementById('g-starter').innerText=`${Number(skillRates.stairs||0).toFixed(1)}%`;

                if (!hist.length) return;
                const labels=hist.map(p=>Number(p.timesteps||0).toLocaleString());

                upsertTrainingChart('graph-reward',labels,[
                    {label:'Ø Episode Reward',data:hist.map(p=>Number(p.avg_episode_reward||0)),borderWidth:2,pointRadius:1.5,tension:.22}
                ]);

                const cleanHist=hist.filter(p=>Number(p.stats_schema||0)>=3);
                const cleanLabels=cleanHist.map(p=>Number(p.timesteps||0).toLocaleString());

                upsertTrainingChart('graph-success',cleanLabels,[
                    {label:'Intro Skill',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).intro||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Treppen Skill',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).stairs||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Exit Skill',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).exit||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Full Intro',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).full_intro||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Full Treppe',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).full_stairs||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Full Haus raus',data:cleanHist.map(p=>Number((p.v6_skill_rates||{}).full_exit||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Erstes Pokémon',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).starter||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Battle KO',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).battle||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Level-Up',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).level||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Gym / Badge',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).badge||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Full Pokémon',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).full_starter||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Full Badge 1',data:cleanHist.map(p=>Number((p.v8_skill_rates||{}).full_badge1||0)),borderWidth:2,pointRadius:1,tension:.2}
                ],true);

                const runningMax=(arr)=>{let m=0;return arr.map(v=>{m=Math.max(m,Number(v||0));return m;});};
                upsertTrainingChart('graph-progress',labels,[
                    {label:'Max Level',data:runningMax(hist.map(p=>p.max_level)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Orden',data:runningMax(hist.map(p=>p.max_badges)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Maps',data:runningMax(hist.map(p=>p.max_maps)),borderWidth:2,pointRadius:1,tension:.2}
                ]);

                const loopHist=hist.filter(p=>Number(p.stats_schema||0)>=2);
                const loopLabels=loopHist.map(p=>Number(p.timesteps||0).toLocaleString());
                upsertTrainingChart('graph-loops',loopLabels,[
                    {label:'Loops pro 100 Beginning-Runs',data:loopHist.map(p=>Number(p.beginning_loops_per_100_runs||0)),borderWidth:2,pointRadius:1,tension:.2}
                ]);
            } catch(e) {
                console.error('Graph load failed',e);
            }
        }

        setInterval(()=>{ if(currentTab==='graphs') loadTrainingGraphs(); },5000);

        function clearPersistentAgentLayer(id) {
            if (persistentExplorationLayers[id]) {
                map.removeLayer(persistentExplorationLayers[id]);
                delete persistentExplorationLayers[id];
            }
            delete persistentExplorationSignatures[id];
        }

        function persistentEdgeKey(agentId, e) {
            if (!Array.isArray(e) || e.length !== 6) return null;

            let x1 = Number(e[2]), y1 = Number(e[3]);
            let x2 = Number(e[4]), y2 = Number(e[5]);

            // A->B und B->A sind dieselbe Kante.
            if (x2 < x1 || (x2 === x1 && y2 < y1)) {
                [x1, x2] = [x2, x1];
                [y1, y2] = [y2, y1];
            }

            return `${agentId}:${Number(e[0])}:${Number(e[1])}:${x1}:${y1}:${x2}:${y2}`;
        }

        function persistentTransitionKey(agentId, t) {
            if (!Array.isArray(t) || t.length !== 8) return null;

            let a = [
                Number(t[0]), Number(t[1]),
                Number(t[2]), Number(t[3])
            ];
            let b = [
                Number(t[4]), Number(t[5]),
                Number(t[6]), Number(t[7])
            ];

            const sa = a.join(',');
            const sb = b.join(',');
            if (sb < sa) [a, b] = [b, a];

            return `${agentId}:${a.join(':')}:${b.join(':')}`;
        }

        function clearPersistentAgentGeometry(agentId) {
            const prefix = `${agentId}:`;

            Object.keys(persistentEdgeLayers).forEach(key => {
                if (key.startsWith(prefix)) {
                    map.removeLayer(persistentEdgeLayers[key]);
                    delete persistentEdgeLayers[key];
                }
            });

            Object.keys(persistentTransitionLayers).forEach(key => {
                if (key.startsWith(prefix)) {
                    map.removeLayer(persistentTransitionLayers[key]);
                    delete persistentTransitionLayers[key];
                }
            });
        }

        async function updatePersistentExploration(force=false) {
            try {
                const res = await fetch('/api/explorations?t=' + Date.now());
                const all = await res.json();

                const wantedEdgeKeys = new Set();
                const wantedTransitionKeys = new Set();
                const visibleAgentIds = new Set();

                Object.entries(all || {}).forEach(([idStr, data]) => {
                    const id = Number(idStr);
                    if (!Number.isFinite(id)) return;

                    const isFocused =
                        selectedAgentId !== null
                        && Number(selectedAgentId) === id;

                    const shouldShow =
                        selectedAgentId === null
                        ? id < maxVisibleAgents
                        : isFocused;

                    if (!shouldShow) return;

                    visibleAgentIds.add(id);

                    const edges = Array.isArray(data.edges) ? data.edges : [];
                    const transitions = Array.isArray(data.transitions)
                        ? data.transitions : [];

                    const color = agentColor(id);

                    edges.forEach(e => {
                        const key = persistentEdgeKey(id, e);
                        if (!key) return;

                        wantedEdgeKeys.add(key);

                        // Wenn diese Kante bereits existiert: NICHT noch einmal zeichnen.
                        if (persistentEdgeLayers[key]) return;

                        const p1 = getLeafletCoords(
                            Number(e[0]), Number(e[1]),
                            Number(e[2]), Number(e[3])
                        );
                        const p2 = getLeafletCoords(
                            Number(e[0]), Number(e[1]),
                            Number(e[4]), Number(e[5])
                        );

                        persistentEdgeLayers[key] = L.polyline([p1, p2], {
                            color,
                            weight: isFocused ? 2.0 : 1.25,
                            opacity: isFocused ? 0.95 : 0.52,
                            interactive: false
                        }).addTo(map);
                    });

                    transitions.forEach(t => {
                        const key = persistentTransitionKey(id, t);
                        if (!key) return;

                        wantedTransitionKeys.add(key);

                        if (persistentTransitionLayers[key]) return;

                        const p1 = getLeafletCoords(
                            Number(t[0]), Number(t[1]),
                            Number(t[2]), Number(t[3])
                        );
                        const p2 = getLeafletCoords(
                            Number(t[4]), Number(t[5]),
                            Number(t[6]), Number(t[7])
                        );

                        persistentTransitionLayers[key] =
                            L.polyline([p1, p2], {
                                color,
                                weight: isFocused ? 2.8 : 1.8,
                                opacity: isFocused ? 1.0 : 0.72,
                                dashArray: '7,5',
                                interactive: false
                            }).addTo(map);
                    });
                });

                // Nur Geometrie entfernen, die nicht mehr sichtbar sein soll.
                Object.keys(persistentEdgeLayers).forEach(key => {
                    if (!wantedEdgeKeys.has(key)) {
                        map.removeLayer(persistentEdgeLayers[key]);
                        delete persistentEdgeLayers[key];
                    }
                });

                Object.keys(persistentTransitionLayers).forEach(key => {
                    if (!wantedTransitionKeys.has(key)) {
                        map.removeLayer(persistentTransitionLayers[key]);
                        delete persistentTransitionLayers[key];
                    }
                });

            } catch(e) {
                console.error('Persistent exploration load failed', e);
            }
        }

        setInterval(() => {
            if (currentTab === 'map') {
                updatePersistentExploration(false);
            }
        }, 2000);

        let latestGlobalMapping = {
            tiles: [],
            edges: [],
            maps: [],
            transitions: [],
            warp_points: []
        };

        function canonicalTransitionLabel(t) {
            return `Bank ${t[0]} Map ${t[1]} (${t[2]},${t[3]}) → ` +
                   `Bank ${t[4]} Map ${t[5]} (${t[6]},${t[7]})`;
        }

        async function updateGlobalMapping(force=false) {
            try {
                const res = await fetch('/api/global_mapping?t=' + Date.now());
                const data = await res.json();
                latestGlobalMapping = {
                    tiles: Array.isArray(data.tiles) ? data.tiles : [],
                    edges: Array.isArray(data.edges) ? data.edges : [],
                    maps: Array.isArray(data.maps) ? data.maps : [],
                    transitions: Array.isArray(data.transitions)
                        ? data.transitions : [],
                    warp_points: Array.isArray(data.warp_points)
                        ? data.warp_points : []
                };

                const setLiveMap=(id,v)=>{
                    const el=document.getElementById(id);
                    if(el) el.innerText=v;
                };
                setLiveMap('live-maps', Number(latestGlobalMapping.maps.length).toLocaleString());
                setLiveMap('live-warps', '—');
                setLiveMap('live-edges', Number(latestGlobalMapping.edges.length).toLocaleString());

                const signature =
                    `${latestGlobalMapping.tiles.length}:` +
                    `${latestGlobalMapping.edges.length}:` +
                    `${latestGlobalMapping.warp_points.length}`;

                if (!force && signature === globalMappingSignature) {
                    if (currentTab === 'rooms') renderIndoorMapping();
                    return;
                }
                globalMappingSignature = signature;

                // V7.4.5: Warp-Marker auf der Global Map vorerst deaktiviert.
                // Rohdaten/Transitions bleiben erhalten und werden NICHT geloescht.
                globalWarpLayer.clearLayers();

                if (currentTab === 'rooms') {
                    renderIndoorMapping();
                }
            } catch(e) {
                console.error('Global mapping load failed', e);
            }
        }

        function renderIndoorMapping() {
            const grid = document.getElementById('room-grid');
            if (!grid) return;

            const tiles = latestGlobalMapping.tiles || [];
            const edges = latestGlobalMapping.edges || [];
            const transitions = latestGlobalMapping.transitions || [];

            const rooms = new Map();

            function roomFor(bank, mapId) {
                const key = `${bank},${mapId}`;
                if (!rooms.has(key)) {
                    rooms.set(key, {
                        bank:Number(bank),
                        mapId:Number(mapId),
                        tiles:new Set(),
                        edges:[],
                        warps:[]
                    });
                }
                return rooms.get(key);
            }

            tiles.forEach(t => {
                if (!Array.isArray(t) || t.length !== 4) return;
                const bank = Number(t[0]);
                const mapId = Number(t[1]);
                if (bank === 3) return;
                roomFor(bank, mapId).tiles.add(
                    `${Number(t[2])},${Number(t[3])}`
                );
            });

            edges.forEach(e => {
                if (!Array.isArray(e) || e.length !== 6) return;
                const bank = Number(e[0]);
                const mapId = Number(e[1]);
                if (bank === 3) return;
                roomFor(bank, mapId).edges.push(e.map(Number));
            });

            transitions.forEach(t => {
                if (!Array.isArray(t) || t.length !== 8) return;
                const aBank = Number(t[0]), aMap = Number(t[1]);
                const bBank = Number(t[4]), bMap = Number(t[5]);

                if (aBank !== 3) {
                    roomFor(aBank, aMap).warps.push({
                        x:Number(t[2]), y:Number(t[3]),
                        toBank:bBank, toMap:bMap,
                        toX:Number(t[6]), toY:Number(t[7])
                    });
                }
                if (bBank !== 3) {
                    roomFor(bBank, bMap).warps.push({
                        x:Number(t[6]), y:Number(t[7]),
                        toBank:aBank, toMap:aMap,
                        toX:Number(t[2]), toY:Number(t[3])
                    });
                }
            });

            const roomList = [...rooms.values()]
                .filter(r => r.tiles.size || r.edges.length || r.warps.length)
                .sort((a,b) => a.bank-b.bank || a.mapId-b.mapId);

            if (!roomList.length) {
                grid.innerHTML =
                    '<div style="color:#8f98ad;padding:20px">' +
                    'Noch keine Indoor-Tiles seit dem Mapping-Reset entdeckt.' +
                    '</div>';
                return;
            }

            grid.innerHTML = '';

            roomList.forEach((room, idx) => {
                const coords = [...room.tiles].map(s => s.split(',').map(Number));

                room.edges.forEach(e => {
                    coords.push([e[2],e[3]], [e[4],e[5]]);
                });
                room.warps.forEach(w => coords.push([w.x,w.y]));

                if (!coords.length) return;

                let minX = Math.min(...coords.map(p => p[0]));
                let maxX = Math.max(...coords.map(p => p[0]));
                let minY = Math.min(...coords.map(p => p[1]));
                let maxY = Math.max(...coords.map(p => p[1]));

                minX -= 1; maxX += 1;
                minY -= 1; maxY += 1;

                const widthTiles = Math.max(1, maxX-minX+1);
                const heightTiles = Math.max(1, maxY-minY+1);
                const cell = Math.max(
                    10,
                    Math.min(
                        26,
                        Math.floor(520 / Math.max(widthTiles, heightTiles))
                    )
                );

                const canvas = document.createElement('canvas');
                canvas.className = 'room-map-canvas';
                canvas.width = widthTiles * cell;
                canvas.height = heightTiles * cell;

                const card = document.createElement('div');
                card.className = 'room-card';

                const title = document.createElement('div');
                title.className = 'title';
                title.textContent = `Indoor · Bank ${room.bank} / Map ${room.mapId}`;

                const sub = document.createElement('div');
                sub.className = 'sub';
                sub.textContent =
                    `${room.tiles.size} Felder · ` +
                    `${room.edges.length} Kanten · ` +
                    `${room.warps.length} Warp-Punkte`;

                const wrap = document.createElement('div');
                wrap.className = 'room-canvas-wrap';
                wrap.appendChild(canvas);

                const legend = document.createElement('div');
                legend.className = 'room-legend';
                legend.innerHTML =
                    '<span><i class="walked"></i>erkundet</span>' +
                    '<span><i class="edge"></i>begehbare Kante</span>' +
                    '<span><i class="warp"></i>Warp/Treppe/Tür</span>';

                card.appendChild(title);
                card.appendChild(sub);
                card.appendChild(wrap);
                card.appendChild(legend);
                grid.appendChild(card);

                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#0b0d12';
                ctx.fillRect(0,0,canvas.width,canvas.height);

                function px(x) { return (x-minX)*cell; }
                function py(y) { return (y-minY)*cell; }
                function cx(x) { return px(x)+cell/2; }
                function cy(y) { return py(y)+cell/2; }

                // Grid.
                ctx.strokeStyle = '#232738';
                ctx.lineWidth = 1;
                for (let x=0; x<=widthTiles; x++) {
                    ctx.beginPath();
                    ctx.moveTo(x*cell,0);
                    ctx.lineTo(x*cell,canvas.height);
                    ctx.stroke();
                }
                for (let y=0; y<=heightTiles; y++) {
                    ctx.beginPath();
                    ctx.moveTo(0,y*cell);
                    ctx.lineTo(canvas.width,y*cell);
                    ctx.stroke();
                }

                // Green explored fields.
                ctx.fillStyle = '#2e7d32';
                room.tiles.forEach(s => {
                    const [x,y] = s.split(',').map(Number);
                    ctx.fillRect(
                        px(x)+1, py(y)+1,
                        Math.max(1,cell-2), Math.max(1,cell-2)
                    );
                });

                // Successful traversable edges.
                ctx.strokeStyle = '#81c784';
                ctx.lineWidth = Math.max(2,cell*0.12);
                room.edges.forEach(e => {
                    ctx.beginPath();
                    ctx.moveTo(cx(e[2]), cy(e[3]));
                    ctx.lineTo(cx(e[4]), cy(e[5]));
                    ctx.stroke();
                });

                // Warp points.
                room.warps.forEach(w => {
                    ctx.beginPath();
                    ctx.fillStyle = '#ffca28';
                    ctx.arc(
                        cx(w.x), cy(w.y),
                        Math.max(4,cell*0.28),
                        0, Math.PI*2
                    );
                    ctx.fill();

                    ctx.strokeStyle = '#fff3c4';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });
            });
        }

        setInterval(() => {
            if (currentTab === 'map' || currentTab === 'rooms') {
                updateGlobalMapping(false);
            }
        }, 2000);

        async function updateDashboard() {
            try {
                const res = await fetch('/api/state?t=' + Date.now());
                const state = await res.json();
                
                                const setLiveState=(id,v)=>{
                    const el=document.getElementById(id);
                    if(el) el.innerText=v;
                };
                const globalBattles=state.battle_stats||{};
                setLiveState('live-model', `v${String(state.version||0).padStart(6,'0')}`);
                setLiveState('live-battles', Number(globalBattles.started||0).toLocaleString());
                setLiveState('live-finished', Number(globalBattles.completed||0).toLocaleString());
                setLiveState('live-steps', Number(state.training_timesteps||0).toLocaleString());

document.getElementById('model-ver').innerText = `v${String(state.version).padStart(6, '0')}`;

                const instances = state.instances || [];
                latestInstances = instances;
                instances.forEach(pushHistory);

                // Falls Watcher noch kein party-Feld in seiner Instanz hat,
                // die alte API-Fallback-Party nur dem Watcher zuordnen.
                const watcher = instances.find(i => i.id === 99);
                if (watcher && (!watcher.party || watcher.party.length === 0) && state.party) {
                    watcher.party = state.party;
                }

                // Fokus nur aufheben, wenn der fokussierte Agent nicht mehr existiert.
                if (
                    selectedAgentId !== null
                    && !instances.some(i => Number(i.id) === Number(selectedAgentId))
                ) {
                    selectedAgentId = null;
                }
                let hudHtml = `
                    <div class="hud-title">
                        <span>Aktive Instanzen (${instances.length})</span>
                        <span class="agent-badge">Max Speed</span>
                    </div>
                `;

                let renderedTrainCount = 0;

                instances.forEach(inst => {
                    const isWatcher = Number(inst.id) === 99;
                    const shouldRender = isAgentVisible(
                        inst.id, isWatcher, renderedTrainCount
                    );

                    if (!isWatcher && selectedAgentId === null && shouldRender) {
                        renderedTrainCount++;
                    }

                    const markerColor = agentColor(inst.id);
                    const selectedClass =
                        Number(inst.id) === Number(selectedAgentId) ? 'selected' : '';
                    const nameClass = isWatcher ? 'agent-watcher' : '';

                    // HUD-Liste bleibt immer klickbar, auch wenn ein anderer Agent
                    // im Fokus ist. So kann man direkt umschalten.
                    hudHtml += `
                        <div class="agent-row ${nameClass} ${selectedClass}" onclick="selectAgent(${inst.id})">
                            <span class="agent-name-wrap">
                                <span class="agent-color-dot" style="background:${markerColor}"></span>
                                <span>${inst.name}</span>
                            </span>
                            <span>${inst.room} (${inst.steps} ep)</span>
                        </div>
                    `;

                    if (shouldRender) {
                        const curPos = getLeafletCoords(
                            inst.bank, inst.map, inst.x, inst.y
                        );

                        if (!agentMarkers[inst.id]) {
                            agentMarkers[inst.id] = L.circleMarker(curPos, {
                                radius: isWatcher ? 8 : 5,
                                color: markerColor,
                                fillColor: markerColor,
                                fillOpacity: isWatcher ? 1.0 : 0.9,
                                weight: 2
                            }).bindPopup(
                                `<b>${inst.name}</b><br>${inst.room}`
                            ).addTo(map);
                            agentMarkers[inst.id].on(
                                'click', () => selectAgent(inst.id)
                            );
                        } else {
                            agentMarkers[inst.id].setLatLng(curPos);
                            agentMarkers[inst.id].setStyle({
                                color: markerColor,
                                fillColor: markerColor
                            });
                            agentMarkers[inst.id].setPopupContent(
                                `<b>${inst.name}</b><br>${inst.room}`
                            );
                        }

                        // Alte Route/Schrittpunkte jedes Refresh sauber ersetzen.
                        if (agentPolylines[inst.id]) {
                            map.removeLayer(agentPolylines[inst.id]);
                            delete agentPolylines[inst.id];
                        }
                        if (agentStepDots[inst.id]) {
                            agentStepDots[inst.id].forEach(d => map.removeLayer(d));
                            delete agentStepDots[inst.id];
                        }

                        const path = Array.isArray(inst.path) ? inst.path : [];
                        const segs = [];
                        let curSeg = [];
                        let last = null;
                        const dots = [];

                        path.forEach(pt => {
                            if (!pt || pt.length < 4) return;
                            const pb = Number(pt[0]);
                            const pm = Number(pt[1]);
                            const px = Number(pt[2]);
                            const py = Number(pt[3]);
                            const pos = getLeafletCoords(pb, pm, px, py);

                            if (last) {
                                const sameMap =
                                    last.bank === pb && last.map === pm;
                                const oneTile =
                                    Math.abs(px - last.x) + Math.abs(py - last.y) === 1;

                                if (!sameMap || !oneTile) {
                                    if (curSeg.length > 1) segs.push(curSeg);
                                    curSeg = [];
                                }
                            }

                            curSeg.push(pos);

                            // Bei Fokus: jeden einzelnen RAM-Schritt als Punkt zeigen.
                            if (
                                selectedAgentId !== null
                                && Number(inst.id) === Number(selectedAgentId)
                                && isWatcher
                            ) {
                                dots.push(
                                    L.circleMarker(pos, {
                                        radius: 2.2,
                                        color: markerColor,
                                        fillColor: markerColor,
                                        fillOpacity: 0.95,
                                        weight: 0.5,
                                        interactive: false
                                    }).addTo(map)
                                );
                            }

                            last = {bank:pb, map:pm, x:px, y:py};
                        });
                        if (curSeg.length > 1) segs.push(curSeg);

                        if (segs.length > 0 && isWatcher) {
                            // Trainingsagenten werden NICHT mehr aus recent_path
                            // gezeichnet. Ihre Karte kommt ausschliesslich aus dem
                            // persistenten Exploration-Memory und verschwindet nie.
                            agentPolylines[inst.id] = L.polyline(segs, {
                                color: markerColor,
                                weight: 2.5,
                                opacity: 0.8
                            }).addTo(map);
                        }
                        agentStepDots[inst.id] = dots;
                    } else {
                        if (agentMarkers[inst.id]) {
                            map.removeLayer(agentMarkers[inst.id]);
                            delete agentMarkers[inst.id];
                        }
                        if (agentPolylines[inst.id]) {
                            map.removeLayer(agentPolylines[inst.id]);
                            delete agentPolylines[inst.id];
                        }
                        if (agentStepDots[inst.id]) {
                            agentStepDots[inst.id].forEach(d => map.removeLayer(d));
                            delete agentStepDots[inst.id];
                        }
                    }
                });

                hudHtml += `
                    <div class="focus-hint">
                        Klick auf Agent = nur diesen anzeigen · nochmal klicken = alle.
                        Watcher-Telemetrie läuft immer weiter.
                    </div>
                `;

                document.getElementById('hud').innerHTML = hudHtml;
                renderSelectedAgent();
                if (currentTab === 'map') {
                    updatePersistentExploration(false);
                }
            } catch(e) {
                console.error(e);
            }
        }
        setInterval(updateDashboard, 1000);

    </script>
<script id="champion-night-js">
async function refreshChampionNight(){
  try{
    const r=await fetch('/api/champion?ts='+Date.now());
    const c=await r.json(),m=c.metrics||{};
    const fmt=n=>Number(n||0).toLocaleString('de-DE');
    document.getElementById('cn-ver').textContent='v'+String(c.version||0).padStart(6,'0');
    document.getElementById('cn-steps').textContent=fmt(c.timesteps||0);
    document.getElementById('cn-full').textContent=(Number(m.full_exit_permille||0)/10).toFixed(1)+'%';
    document.getElementById('cn-starter').textContent=(Number(m.full_starter_permille||0)/10).toFixed(1)+'%';
    document.getElementById('cn-badge').textContent=String(m.max_badges||0);
  }catch(e){}
}
setInterval(refreshChampionNight,2000);refreshChampionNight();
</script>
<div id="pkmai-hidden-tray"></div>
<script id="pkmai-window-tools-js">
(function(){
  const tray=document.getElementById('pkmai-hidden-tray');
  const preferred=['#champion-night-card','.v81skills','.live-global','.instance-panel','.agents-panel','.agent-list','.global-ai','.watcher-panel'];
  function titleOf(el){const t=el.querySelector('.cn-title,.graph-title,.live-global-title,h1,h2,h3,b,strong');return (t&&t.textContent.trim())||el.id||'Fenster';}
  function restoreButton(el,label){const b=document.createElement('button');b.textContent='↩ '+label;b.onclick=()=>{el.classList.remove('pkmai-hidden');b.remove();};tray.appendChild(b);}
  function addTools(el){
    if(!el||el.dataset.pkmaiTools==='1')return;
    el.dataset.pkmaiTools='1';el.classList.add('pkmai-movable');
    if(getComputedStyle(el).position==='static')el.style.position='relative';
    const tools=document.createElement('div');tools.className='pkmai-float-tools';
    const min=document.createElement('button');min.textContent='−';min.title='Minimieren';
    const hide=document.createElement('button');hide.textContent='×';hide.title='Ausblenden';
    tools.append(min,hide);el.appendChild(tools);
    min.onclick=e=>{e.stopPropagation();el.classList.toggle('pkmai-minimized');};
    hide.onclick=e=>{e.stopPropagation();const label=titleOf(el);el.classList.add('pkmai-hidden');restoreButton(el,label);};
    if(matchMedia('(max-width:820px)').matches)return;
    let drag=false,ox=0,oy=0;
    el.addEventListener('pointerdown',e=>{
      if(e.target.closest('button,input,canvas,a'))return;
      drag=true;el.setPointerCapture(e.pointerId);
      const r=el.getBoundingClientRect();ox=e.clientX-r.left;oy=e.clientY-r.top;
      el.style.position='fixed';el.style.margin='0';
    });
    el.addEventListener('pointermove',e=>{
      if(!drag)return;
      const x=Math.max(0,Math.min(innerWidth-el.offsetWidth,e.clientX-ox));
      const y=Math.max(52,Math.min(innerHeight-40,e.clientY-oy));
      el.style.left=x+'px';el.style.top=y+'px';el.style.right='auto';el.style.bottom='auto';
    });
    el.addEventListener('pointerup',()=>drag=false);
    el.addEventListener('pointercancel',()=>drag=false);
  }
  function scan(){
    preferred.forEach(sel=>document.querySelectorAll(sel).forEach(addTools));
    document.querySelectorAll('body > div,#map-view > div,#graphs-view > div').forEach(el=>{
      const cs=getComputedStyle(el);
      if((cs.position==='fixed'||cs.position==='absolute')&&el.offsetWidth>180&&el.offsetHeight>80)addTools(el);
    });
  }
  scan();setInterval(scan,2000);
})();
</script>
<script id="learner-truth-js">async function refreshLearnerTruth(){try{const r=await fetch('/api/trainer-status?ts='+Date.now());const d=await r.json();const f=n=>Number(n||0).toLocaleString('de-DE');document.getElementById('lth-learner').textContent=f(d.learner_steps);document.getElementById('lth-champion').textContent=f(d.champion_steps);document.getElementById('lth-delta').textContent=(Number(d.delta_steps||0)>=0?'+':'')+f(d.delta_steps);}catch(e){}}setInterval(refreshLearnerTruth,1000);refreshLearnerTruth();</script>


<style id="pkmai-cluster-style">
  #cluster-dashboard{margin:16px;background:#101827;border:1px solid #2a3b55;border-radius:12px;padding:14px;color:#dbeafe;font:13px system-ui,sans-serif}
  #cluster-dashboard h2{margin:0 0 10px;font-size:16px;color:#93c5fd}.cluster-summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}.cluster-chip{background:#17243a;border-radius:7px;padding:7px 9px}.cluster-ok{color:#86efac}.cluster-stale{color:#fca5a5}.cluster-workers{display:grid;gap:7px}.cluster-worker{display:grid;grid-template-columns:1fr auto auto auto;gap:10px;align-items:center;background:#17243a;border-radius:7px;padding:8px}.cluster-worker small{color:#93a4bc}
</style>
<section id="cluster-dashboard" aria-live="polite">
  <h2>Cluster Brain &amp; Worker</h2>
  <div id="cluster-summary" class="cluster-summary">Cluster-Status wird geladen …</div>
  <div id="cluster-workers" class="cluster-workers"></div>
</section>
<script id="pkmai-cluster-dashboard-js">
  async function refreshClusterDashboard(){
    try{
      const r=await fetch('/api/cluster-status?ts='+Date.now());
      const d=await r.json(), fmt=n=>Number(n||0).toLocaleString('de-DE'), esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
      const online=d.brain_online?'cluster-ok':'cluster-stale';
      document.getElementById('cluster-summary').innerHTML=`<span class="cluster-chip ${online}">Brain: ${d.brain_online?'online':'offline'}</span><span class="cluster-chip">Policy v${d.policy_version||0}</span><span class="cluster-chip">${fmt(d.timesteps)} Steps</span><span class="cluster-chip">Checkpoint: ${d.checkpoint||'—'}</span>`;
      document.getElementById('cluster-workers').innerHTML=(d.workers||[]).map(w=>{const fps=w.fps===null?'FPS —':`${Number(w.fps||0).toFixed(1)} FPS`;return `<div class="cluster-worker"><span><b>${esc(w.worker_id)}</b><br><small>${esc(w.hostname||'unbekannter Host')} · ${w.age_seconds}s</small></span><span class="${w.online?'cluster-ok':'cluster-stale'}">${w.online?'online':'stale'}</span><span>${w.active_agents} Emulatoren</span><span>${fps} · v${w.policy_version||0}</span></div>`;}).join('')||'<div class="cluster-worker">Keine Worker registriert.</div>';
    }catch(e){document.getElementById('cluster-summary').textContent='Cluster-Status nicht erreichbar.';}
  }
  refreshClusterDashboard();setInterval(refreshClusterDashboard,2000);
</script>
</body>
</html>
    """

if __name__ == '__main__':
    host, port = web_bind_settings()
    uvicorn.run(app, host=host, port=port, log_level="warning")
