import os
import asyncio
import collections
import glob
import json
import threading
import tempfile
from datetime import datetime, timezone
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

app = FastAPI()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
BASE_DIR = PROJECT_ROOT
ROOMS_DIR = os.path.join(RUNTIME_DIR, "room_captures")
MAP_FILE = os.path.join(ASSETS_DIR, "maps", "kanto_map.png")
INSTANCES_DIR = os.path.join(RUNTIME_DIR, "instances_data")
NAV_RUNTIME_DIR = os.path.join(RUNTIME_DIR, "navigation")
BATTLE_RUNTIME_DIR = os.path.join(RUNTIME_DIR, "battle")
VERSION_FILE = os.path.join(NAV_RUNTIME_DIR, "model_version.json")
SKELETON_FILE = os.path.join(RUNTIME_DIR, "skeleton_map.json")
HISTORY_FILE = os.path.join(RUNTIME_DIR, "training_history.json")
CHAMPION_FILE = os.path.join(NAV_RUNTIME_DIR, "champion_score.json")
TRAINER_STATUS_FILE = os.path.join(NAV_RUNTIME_DIR, "trainer_status.json")
NAV_GLOBAL_FILE = os.path.join(NAV_RUNTIME_DIR, "nav_global.json")
MOVEMENT_GRAPH_FILE = os.path.join(NAV_RUNTIME_DIR, "movement_graph_v1.json")
BATTLE_STATUS_FILE = os.path.join(BATTLE_RUNTIME_DIR, "battle_stats.json")
BATTLE_CHAMPION_FILE = os.path.join(
    BATTLE_RUNTIME_DIR, "checkpoints", "battle_champion.zip"
)
BATTLE_SCENARIOS_FILE = os.path.join(BATTLE_RUNTIME_DIR, "scenarios", "index.json")
HISTORY_LOCK = threading.Lock()
EXPLORATION_MEMORY_DIR = os.path.join(RUNTIME_DIR, "exploration_memory")
WATCHER_MAPPING_FILE = os.path.join(RUNTIME_DIR, "watcher_mapping.json")
WATCHER_FRAME_FILE = os.path.join(RUNTIME_DIR, "watcher.jpg")
MAPPER_FRAME_FILE = os.path.join(RUNTIME_DIR, "mapper.jpg")
MAPPER_DIR = os.path.join(RUNTIME_DIR, "mapper")
MAPPER_ATLAS_FILE = os.path.join(MAPPER_DIR, "kanto_map.png")
MAPPER_STATUS_FILE = os.path.join(RUNTIME_DIR, "mapper_status.json")
MAPPER_MAPS_DIR = os.path.join(MAPPER_DIR, "stitched_maps")


def _load_json(path, default=None):
    try:
        with open(path, "r") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else (default or {})
    except Exception:
        return default or {}

def _live_learner_steps(fallback=0):
    try:
        with open(TRAINER_STATUS_FILE, "r") as f:
            d = json.load(f) or {}
        n = int(d.get("learner_steps", 0) or 0)
        return n if n > 0 else int(fallback or 0)
    except Exception:
        return int(fallback or 0)


def _load_version_meta():
    """Aktive Version; vor der ersten Promotion aus Champion-Baseline lesen."""
    default = {"version": 0, "timesteps": 0}
    for source in (VERSION_FILE, CHAMPION_FILE):
        try:
            with open(source, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                default.update(loaded)
                return default
        except Exception:
            continue
    return default


def _file_freshness(path, declared=None):
    """Return a truthful timestamp and age for one live status file."""
    stamp = declared
    if not stamp:
        try:
            stamp = datetime.fromtimestamp(
                os.path.getmtime(path), tz=timezone.utc
            ).isoformat()
        except OSError:
            return {"updated_at": None, "age_seconds": None}
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    except (TypeError, ValueError):
        age = None
    return {"updated_at": stamp, "age_seconds": age}


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


@app.get("/api/brains")
def get_brains():
    """Single truthful status source for the live 2x2 architecture."""
    nav = _load_json(TRAINER_STATUS_FILE)
    nav_champion = _load_json(CHAMPION_FILE)
    battle = _load_json(BATTLE_STATUS_FILE)
    battle_counts = battle.get("counters") or {}
    # The first proven PPO champion predates battle_stats.json's version field
    # in some migrated runs.  A present champion artifact is v1 at minimum;
    # later promotions always publish their explicit version in battle_stats.
    battle_champion_version = int(battle.get("champion_version", 0) or 0)
    if battle_champion_version <= 0 and os.path.isfile(BATTLE_CHAMPION_FILE):
        battle_champion_version = 1
    scenarios = (_load_json(BATTLE_SCENARIOS_FILE).get("scenarios") or [])
    areas = sorted({str(s.get("area") or "unknown") for s in scenarios})
    nav_freshness = _file_freshness(TRAINER_STATUS_FILE, nav.get("updated"))
    battle_freshness = _file_freshness(BATTLE_STATUS_FILE, battle.get("updated"))
    fresh_nav_workers = 0
    now = datetime.now(timezone.utc).timestamp()
    _nav_tel = []
    for path in glob.glob(os.path.join(INSTANCES_DIR, "inst_*.json")):
        try:
            agent_id = int(os.path.basename(path)[5:-5])
            if 0 <= agent_id < 40 and now - os.path.getmtime(path) < 15:
                fresh_nav_workers += 1
                with open(path) as _f:
                    _d = json.load(_f)
                if _d.get("navigation"):
                    _nav_tel.append((agent_id, _d))
        except (OSError, ValueError):
            continue

    def _directed_nav_summary():
        n = len(_nav_tel) or 1
        on_r1 = sum(1 for _, d in _nav_tel if (d.get("bank"), d.get("map")) == (3, 19))
        in_pallet = sum(1 for _, d in _nav_tel if (d.get("bank"), d.get("map")) == (3, 0))
        loops = sum(int((d.get("navigation") or {}).get("region_loops", 0)) for _, d in _nav_tel)
        ledges = sum(int((d.get("navigation") or {}).get("ledge_jumps", 0)) for _, d in _nav_tel)
        frozen = sum(1 for _, d in _nav_tel if (d.get("navigation") or {}).get("shaping_frozen"))
        tos = collections.Counter((d.get("navigation") or {}).get("timeout_reason")
                                  for _, d in _nav_tel if (d.get("navigation") or {}).get("timeout_reason"))
        r1r = next((v.get("navigation", {}).get("route1_reach_rate")
                    for _, v in _nav_tel if v.get("navigation", {}).get("route1_reach_rate") is not None), None)
        return {
            "agents": len(_nav_tel),
            "on_route1": on_r1, "in_pallet": in_pallet,
            "pct_on_route1": round(100 * on_r1 / n),
            "region_loops": loops, "ledge_jumps": ledges,
            "shaping_frozen_agents": frozen,
            "timeout_reasons": dict(tos),
            "route1_reach_rate": r1r,
            "movement_graph_edges": max((int((d.get("navigation") or {}).get(
                "movement_graph_edges", 0)) for _, d in _nav_tel), default=0),
            "per_agent": [
                {"id": aid,
                 "target": (d.get("navigation") or {}).get("target"),
                 "source": (d.get("navigation") or {}).get("target_source"),
                 "graph_distance": (d.get("navigation") or {}).get("graph_distance"),
                 "best": (d.get("navigation") or {}).get("best_graph_distance"),
                 "since_improve": (d.get("navigation") or {}).get("steps_since_improve"),
                 "next_hop": (d.get("navigation") or {}).get("next_hop_action"),
                 "blocked": (d.get("navigation") or {}).get("blocked_directions"),
                 "loop": (d.get("navigation") or {}).get("loop_kind"),
                 "timeout": (d.get("navigation") or {}).get("timeout_reason"),
                 "recovery": (d.get("navigation") or {}).get("recovery_mode")}
                for aid, d in sorted(_nav_tel)],
        }
    return {
        "schema": "twoby2_brains_v1",
        "navigation": {
            "workers": 40,
            "telemetry_workers": fresh_nav_workers,
            "learner_steps": int(nav.get("learner_steps", 0) or 0),
            "champion_steps": int(nav.get("champion_steps", 0)
                                  or nav_champion.get("timesteps", 0) or 0),
            "since_champion": int(nav.get("delta_steps", 0) or 0),
            "learner_version": int(nav.get("champion_version", 0) or 0) + 1,
            "champion_version": int(nav.get("champion_version", 0)
                                    or nav_champion.get("version", 0) or 0),
            "phase": str(nav.get("training_phase") or "training"),
            "completed_runs": int(nav.get("recent_full_done", 0) or 0),
            "last_eval_result": str(nav.get("last_eval_result") or "none"),
            "last_eval_at_step": int(nav.get("last_eval_at_step", 0) or 0),
            "metrics": nav.get("last_eval_metrics") or nav_champion.get("metrics") or {},
            "directed": _directed_nav_summary(),
            **nav_freshness,
        },
        "battle": {
            "workers": int(battle.get("n_workers", 9) or 9),
            "learner_steps": int(battle_counts.get("env_steps", 0) or 0),
            "ppo_updates": int(battle_counts.get("ppo_updates", 0) or 0),
            "episodes": int(battle_counts.get("episodes", 0) or 0),
            "wins": int(battle_counts.get("wins", 0) or 0),
            "wipes": int(battle_counts.get("wipes", 0) or 0),
            "flees": int(battle_counts.get("flees", 0) or 0),
            "timeouts": int(battle_counts.get("timeouts", 0) or 0),
            "kos": int(battle_counts.get("kos", 0) or 0),
            "learner_version": int(battle.get("learner_version", 0) or 0),
            "champion_version": battle_champion_version,
            "champion_source": ("ppo" if battle_champion_version
                                else "rule fallback"),
            "phase": str(battle.get("phase") or "training"),
            "next_promotion_step": int(battle.get("next_promotion_step", 0) or 0),
            "scenario_count": len(scenarios),
            "areas": areas,
            # spec §1B: the ACTUAL live battle-policy source — a v1 PPO champion
            # or the rule fallback, plus the schema and (on a fallback) why.
            "policy": _battle_policy_view(),
            # rolling window (last 100 / 200 battles) + reward breakdown
            "rolling": battle.get("rolling") or {},
            # last champion check: candidate vs champion + why it was rejected
            "last_promotion": battle.get("last_promotion") or {},
            "last_eval": battle.get("last_live_eval") or {},
            # --- Catch-v2 telemetry (spec §16). Never computed from log text —
            # these are the counters BattleCounters recorded where BattleEnv
            # produced the reward. ``schema`` shows whether the live champion
            # is still combat-only (v1) or the catch-capable v2.
            "catch": _battle_catch_view(battle, battle_counts),
            **battle_freshness,
        },
        # spec ZIEL C: wild-encounter + shiny counters, separate per agent
        # class, read straight from the persisted event-counter files. Nothing
        # here is reconstructed from logs.
        "shiny": _shiny_view(),
    }


def _shiny_view():
    try:
        from twoby2.shiny_counters import aggregate_all
        from twoby2 import shiny_ram
        agg = aggregate_all()
        t = agg["totals"]
        return {
            "ram_verified": bool(shiny_ram.SHINY_RAM_VERIFIED),
            "ram_status": ("verified" if shiny_ram.SHINY_RAM_VERIFIED
                           else "shiny_ram_unverified"),
            "seen": int(t.get("shiny_encounters_verified", 0)),
            "caught": int(t.get("shiny_caught", 0)),
            "lost": int(t.get("shiny_ko", 0)) + int(t.get("shiny_fled", 0))
            + int(t.get("shiny_lost_wipe", 0)),
            "unknown": int(t.get("shiny_ram_unknown", 0)),
            "unresolved": int(t.get("shiny_unresolved", 0)),
            "wild_encounters": int(t.get("wild_encounters", 0)),
            "by_agent_class": {cls: v["counters"]
                               for cls, v in agg["per_agent_class"].items()},
            "recent_shinies": agg["recent_shinies"][:8],
        }
    except Exception:
        return {"ram_verified": False, "ram_status": "shiny_ram_unverified",
                "seen": 0, "caught": 0, "lost": 0, "unknown": 0,
                "wild_encounters": 0, "by_agent_class": {}, "recent_shinies": []}


def _battle_policy_view():
    """The live battle-policy status written by
    ``twoby2.live_integration.LatestBattleChampionPolicy`` (spec §1B)."""
    doc = _load_json(os.path.join(BATTLE_RUNTIME_DIR, "battle_policy_status.json"))
    return {
        "source": doc.get("battle_policy_source", "unknown"),
        "schema": doc.get("battle_policy_schema"),
        "champion_dims": doc.get("battle_policy_champion_dims"),
        "fallback_reason": doc.get("battle_policy_fallback_reason", ""),
        "updated": doc.get("updated"),
    }


def _battle_catch_view(battle, battle_counts):
    """The Catch-v2 dashboard block (spec §16). Reads the persisted counters +
    rolling catch section; computes nothing from logs."""
    rolling = (battle.get("rolling") or {}).get("catch") or {}
    manifest = _load_json(os.path.join(BATTLE_RUNTIME_DIR,
                                      "battle_model_manifest.json"))
    lp = battle.get("last_promotion") or {}
    return {
        "live_schema": str(manifest.get("live_schema") or "v1"),
        "v2_training_active": bool(manifest.get("v2", {}).get("training_schema") == "v2"),
        "objective_distribution": rolling.get("objective_distribution") or {},
        "combat_episodes": int(battle_counts.get("combat_episodes", 0) or 0),
        "catch_episodes": int(battle_counts.get("catch_episodes", 0) or 0),
        "catch_requested_episodes": int(battle_counts.get("catch_requested_episodes", 0) or 0),
        "catch_orders": int(battle_counts.get("catch_orders", 0) or 0),
        "catch_attempts": int(battle_counts.get("catch_attempts", 0) or 0),
        "catch_successes": int(battle_counts.get("catch_successes", 0) or 0),
        "catch_failures": int(battle_counts.get("catch_failures", 0) or 0),
        "catch_target_kos": int(battle_counts.get("catch_target_kos", 0) or 0),
        "premature_catches": int(battle_counts.get("premature_catches", 0) or 0),
        "unrequested_catches": int(battle_counts.get("unrequested_catches", 0) or 0),
        "trainer_catch_attempts": int(battle_counts.get("trainer_catch_attempts", 0) or 0),
        "balls_used": int(battle_counts.get("balls_used", 0) or 0),
        "balls_per_successful_catch": rolling.get("balls_per_successful_catch", 0.0),
        "catch_success_rate_when_requested": rolling.get("catch_success_rate_when_requested", 0.0),
        "reward_sum_mismatch_steps": int(battle_counts.get("reward_sum_mismatch_steps", 0) or 0),
        "last_catch_gate": lp.get("catch_reasons") or lp.get("catch_eval") or {},
    }


@app.get("/api/movement_graph")
def get_movement_graph(bank: int = 3, map_id: int = 19):
    """The directed movement graph for one map: walk / jump_or_ledge / blocked
    edges as (from, to, action, kind) for the map viz."""
    path = os.path.join(RUNTIME_DIR, "navigation", "movement_graph_v1.json")
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {"bank": bank, "map": map_id, "edges": [], "blocked": []}
    key = f"{bank},{map_id}"
    edges, blocked = [], []
    for row in (raw.get("edges", {}) or {}).get(key, []):
        frm, action, e = row[0], row[1], row[2]
        if e.get("kind") in ("blocked_static", "blocked_dynamic"):
            blocked.append({"xy": frm, "action": action, "kind": e["kind"]})
        else:
            edges.append({"from": frm, "to": e["to"], "action": action,
                          "kind": e.get("kind"), "legacy": e.get("legacy", False),
                          "confidence": e.get("confidence", 0)})
    for row in (raw.get("blocks", {}) or {}).get(key, []):
        blocked.append({"xy": row[0], "action": row[1],
                        "kind": row[2].get("kind", "blocked_dynamic")})
    return {"bank": bank, "map": map_id, "generation": raw.get("generation", 0),
            "edges": edges, "blocked": blocked}


@app.get("/map.png")
def get_map():
    if os.path.exists(MAP_FILE):
        return FileResponse(MAP_FILE, media_type="image/png", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return Response(status_code=404)


@app.get("/watcher.jpg")
def get_watcher_jpeg():
    """Return the newest complete watcher frame without caching."""
    try:
        with open(WATCHER_FRAME_FILE, "rb") as f:
            jpeg = f.read()
        if not jpeg:
            raise OSError("empty watcher frame")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    except OSError:
        return Response(status_code=404)


async def watcher_mjpeg_frames():
    last_revision = None
    while True:
        try:
            # Open once: fstat and read refer to the same atomic snapshot.
            with open(WATCHER_FRAME_FILE, "rb") as frame:
                stat = os.fstat(frame.fileno())
                revision = (stat.st_ino, stat.st_mtime_ns)
                jpeg = frame.read() if revision != last_revision else b""
            if jpeg:
                last_revision = revision
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except OSError:
            pass
        await asyncio.sleep(1 / 60)


@app.get("/watcher.mjpg")
def get_watcher_stream():
    return StreamingResponse(
        watcher_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/watcher-emulator.jpg")
def get_watcher_emulator_jpeg():
    try:
        with open(os.path.join(RUNTIME_DIR, "watcher_emulator.jpg"), "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    except OSError:
        return Response(status_code=404)


@app.get("/dashboard-language.js")
def get_dashboard_language():
    return FileResponse(os.path.join(ASSETS_DIR, "ui", "dashboard-language.js"),
                        media_type="application/javascript", headers={"Cache-Control": "no-cache"})


# Reale Orden-Sprites (Kanto 1-8, vom Nutzer in assets/ui/ abgelegt) statt
# Emoji-Platzhalter. Whitelist statt generischem Static-Mount, damit ueber
# diese Route keine beliebigen Dateien aus assets/ui/ erreichbar sind.
BADGE_FILES = {
    1: "01_boulder_badge.png",
    2: "02_cascade_badge.png",
    3: "03_thunder_badge.png",
    4: "04_rainbow_badge.png",
    5: "05_soul_badge.png",
    6: "06_marsh_badge.png",
    7: "07_volcano_badge.png",
    8: "08_earth_badge.png",
}


@app.get("/badges/{badge_id}.png")
def get_badge_png(badge_id: int):
    filename = BADGE_FILES.get(badge_id)
    if not filename:
        return Response(status_code=404)
    path = os.path.join(ASSETS_DIR, "ui", filename)
    if not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/battler.png")
def get_battler_png():
    """Whitelisted Battle-agent marker used throughout the dashboard."""
    path = os.path.join(ASSETS_DIR, "maps", "Battler.png")
    if not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/mapper.jpg")
def get_mapper_jpeg():
    if os.path.exists(MAPPER_FRAME_FILE):
        return FileResponse(
            MAPPER_FRAME_FILE, media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(status_code=404)


@app.get("/mapper-atlas.png")
def get_mapper_atlas():
    if os.path.exists(MAPPER_ATLAS_FILE):
        return FileResponse(
            MAPPER_ATLAS_FILE, media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(status_code=404)


@app.get("/api/mapper")
def get_mapper_status():
    try:
        with open(MAPPER_STATUS_FILE, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {"running": False, "name": "Mapper"}


@app.get("/api/mapper/maps")
def get_mapper_maps():
    """Metadaten der echten Screenshot-Karten fuer das Leaflet-Overlay."""
    result = []
    try:
        names = sorted(os.listdir(MAPPER_MAPS_DIR))
    except OSError:
        names = []
    for name in names:
        if not name.startswith("bank_") or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(MAPPER_MAPS_DIR, name), "r") as f:
                meta = json.load(f) or {}
            bank = int(meta["bank"])
            map_id = int(meta["map_id"])
            image_path = os.path.join(MAPPER_MAPS_DIR, name[:-5] + ".png")
            if not os.path.isfile(image_path) or meta.get("invalid_extent"):
                continue
            stat = os.stat(image_path)
            result.append({
                "bank": bank,
                "map_id": map_id,
                "min_x": int(meta["min_x"]),
                "max_x": int(meta["max_x"]),
                "min_y": int(meta["min_y"]),
                "max_y": int(meta["max_y"]),
                "known_world_tiles": int(meta.get("known_world_tiles", 0)),
                # Screenshot-Stitching bleibt bis zur automatischen
                # Kalibrierung nur Vorschau. Falsche Bilder duerfen nicht mehr
                # unter die exakten RAM-Marker gelegt werden.
                "alignment_confident": bool(meta.get("alignment_confident", False)),
                "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
                "url": f"/mapper-map/{bank}/{map_id}.png",
            })
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return {"maps": result}


@app.get("/api/mapper/tiles")
def get_mapper_tiles():
    """Exakte, animationsunabhaengige RAM-Tilemap des Frontier-Mappers."""
    path = os.path.join(MAPPER_DIR, "exploration_memory.json")
    try:
        with open(path, "r") as f:
            data = json.load(f) or {}
        tiles = []
        for item in data.get("positions", []):
            if not isinstance(item, list) or len(item) != 4:
                continue
            bank, map_id, x, y = (int(v) for v in item)
            if 0 <= x < 512 and 0 <= y < 512:
                tiles.append([bank, map_id, x, y])
        return {"tiles": tiles, "count": len(tiles)}
    except Exception:
        return {"tiles": [], "count": 0}


@app.get("/mapper-map/{bank}/{map_id}.png")
def get_mapper_map_image(bank: int, map_id: int):
    if not (0 <= bank <= 999 and 0 <= map_id <= 999):
        return Response(status_code=404)
    path = os.path.join(
        MAPPER_MAPS_DIR,
        f"bank_{bank:03d}_map_{map_id:03d}.png",
    )
    if os.path.isfile(path):
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(status_code=404)


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


def _confirmed_movement_graph_summary():
    """Return persistent exploration rather than per-episode tile counts."""
    raw = _load_json(MOVEMENT_GRAPH_FILE)
    confirmed_edges = 0
    confirmed_maps = 0
    for rows in (raw.get("edges") or {}).values():
        map_edges = 0
        for row in rows if isinstance(rows, list) else ():
            edge = row[2] if isinstance(row, list) and len(row) >= 3 else {}
            if not isinstance(edge, dict) or edge.get("legacy", False):
                continue
            if int(edge.get("confidence", 0) or 0) <= 0:
                continue
            if edge.get("kind") not in ("walk", "jump_or_ledge", "warp"):
                continue
            map_edges += 1
        if map_edges:
            confirmed_maps += 1
            confirmed_edges += map_edges
    return {"confirmed_edges": confirmed_edges, "confirmed_maps": confirmed_maps}


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
    for _k in ('v8_starter_episodes', 'v8_starter_success', 'v8_battle_episodes', 'v8_battle_success', 'v8_level_episodes', 'v8_level_success', 'v8_badge_episodes', 'v8_badge_success', 'v2_full_starter', 'v7_full_badge1', 'enemy_faints', 'enemy_damage_hp', 'party_wipes'):
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
        last_ts = int(history[-1].get("timesteps", -1)) if history else -1
        # V15.3: nach einem Learner-Reset (z.B. Reset auf den Champion) faellt
        # timesteps zurueck unter den letzten aufgezeichneten Wert. Die alte
        # Wächter-Bedingung ">= bucket" blieb dann fuer immer wahr (der alte
        # Spitzenwert wird nie wieder erreicht) -> die Historie fror auf dem
        # Vor-Reset-Stand ein und zeigte nie wieder echte, aktuelle Werte.
        # Ein Rueckwaertssprung heisst "neuer Lauf" -> immer neu aufzeichnen.
        graph = _confirmed_movement_graph_summary()
        run_id = int(_load_json(NAV_GLOBAL_FILE).get("training_run_id", 0) or 0)
        # Upgrade the active bucket once so the truthful graph curve starts
        # immediately after deployment without inventing old measurements.
        upgrade_current_bucket = bool(
            history and last_ts == bucket
            and "movement_graph_edges" not in history[-1]
        )
        if history and last_ts >= bucket and timesteps >= last_ts and not upgrade_current_bucket:
            return

        stats = _aggregate_training_stats(instances)
        point = {
            "version": version,
            "timesteps": timesteps,
            "max_level": max(stats["max_level"], int(version_meta.get("max_level", 0))),
            "max_badges": max(stats["max_badges"], int(version_meta.get("max_badges", 0))),
            "max_maps": max(stats["max_maps"], int(version_meta.get("max_maps", 0))),
            "max_stage": max(
                [int(i.get("world_stage", 0) or 0) for i in instances] or [0]
            ),
            "max_explored_tiles": stats["max_explored_tiles"],
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
            "movement_graph_edges": graph["confirmed_edges"],
            "movement_graph_maps": graph["confirmed_maps"],
            "training_run_id": run_id,
            "stats_schema": 4,
        }
        if upgrade_current_bucket:
            history[-1] = point
        else:
            history.append(point)
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
                agent_id = int(data.get("id", -1))
                if agent_id != 120 and not (0 <= agent_id < 40):
                    continue
                instances.append(data)
                if data.get("level", 0) > max_level:
                    max_level = data.get("level", 0)
                if data.get("badges", 0) > max_badges:
                    max_badges = data.get("badges", 0)
                total_steps += data.get("steps", 0)
                if data.get("id") == 120:
                    trainer_name = data.get("name", "Alex").replace(" (Watcher)", "")
                    party = data.get("party", [])
        except Exception:
            pass

    # Nach ID sortieren (Watcher id 120 landet damit natuerlich unten).
    instances.sort(key=lambda x: x.get("id", 0))

    version_meta = _load_version_meta()

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

    # V13.3: Flotten-Status live aus den Instanz-JSONs.
    _role_counts = {}
    _loc = {"outdoor": 0, "indoor": 0, "battle": 0, "unknown": 0}
    _depth_events = []
    _depth_breakthroughs = 0
    for _inst in instances:
        if _inst.get("id") == 120:
            continue
        _r = str(_inst.get("training_role") or _inst.get("agent_role")
                 or _inst.get("training_objective") or "?")
        _role_counts[_r] = _role_counts.get(_r, 0) + 1
        _b = int(_inst.get("bank", 0) or 0)
        if int(_inst.get("in_battle", 0) or 0):
            _loc["battle"] += 1
        elif _b == 3:
            _loc["outdoor"] += 1
        elif _b == 0:
            _loc["unknown"] += 1
        else:
            _loc["indoor"] += 1
        _depth_breakthroughs += int(
            ((_inst.get("reward_stats") or {}).get("event_counts") or {}).get("world_depth", 0) or 0
        )
        for _ev in (_inst.get("reward_events") or []):
            if isinstance(_ev, str) and _ev.startswith("world_depth"):
                _depth_events.append({"id": _inst.get("id"), "ev": _ev})
    _enemy_ko = int(training_stats["run_totals"].get("enemy_faints", 0))
    _enemy_dmg = int(training_stats["run_totals"].get("enemy_damage_hp", 0))

    # V15: explizite Welt-Stufe + tiefster validierter Stage-Checkpoint.
    world_depth = 0
    deepest_outdoor = 0
    try:
        with open(os.path.join(RUNTIME_DIR, "exploration_memory", "global_progress.json")) as _f:
            _progress = json.load(_f) or {}
            world_depth = int(_progress.get("max_world_stage", 0))
            if _progress.get("progress_schema") != "geography_v1":
                world_depth = {4: 1, 5: 1, 6: 4, 7: 5, 8: 6, 9: 6}.get(world_depth, world_depth)
    except Exception:
        pass
    try:
        import glob as _glob
        for _p in _glob.glob(os.path.join(RUNTIME_DIR, "curriculum_shared", "stage_*.state.gz")):
            try:
                with open(_p[:-9] + ".meta.json") as _meta_file:
                    _meta = json.load(_meta_file)
                _stage = int(_meta.get("stage", 0))
                _actual = {(3,0):1,(3,19):2,(3,1):3,(3,20):4,(1,0):5,(3,2):6}.get(
                    (_meta.get("bank"), _meta.get("map")), 0)
                if _actual == _stage and _meta.get("has_starter"):
                    deepest_outdoor = max(deepest_outdoor, _stage)
            except Exception:
                pass
    except Exception:
        pass

    champion_snapshot = get_champion()
    champion_metrics = champion_snapshot.get("metrics") or {}

    return {
        "trainer_name": trainer_name,
        "version": int(version_meta.get("version", 0)),
        "training_timesteps": _live_learner_steps(int(version_meta.get("timesteps", 0))),
        "max_level": max(max_level, training_stats["max_level"]),
        "max_badges": max(max_badges, training_stats["max_badges"]),
        "world_depth": world_depth,
        "deepest_outdoor_checkpoint": deepest_outdoor,
        "champion_speed": {
            "version": int(champion_snapshot.get("version", 0) or 0),
            "steps": int(champion_snapshot.get("timesteps", 0) or 0),
            "best_stage_steps": int(
                champion_metrics.get("full_best_stage_steps", 0) or 0
            ),
        },
        "fleet": {
            "count": len([i for i in instances if i.get("id") != 120]),
            "roles": _role_counts,
            "location": _loc,
            "enemy_ko": _enemy_ko,
            "enemy_damage_hp": _enemy_dmg,
            "depth_breakthroughs": _depth_breakthroughs,
            "depth_events": _depth_events[-8:],
        },
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
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <title>Pokemon FireRed AI Live Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
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
        .header-left { display:grid;grid-template-columns:auto auto auto auto;align-items:stretch;gap:10px; }
        .identity-card,.header-group{background:#0e1017;border:1px solid #292f40;border-radius:10px;padding:7px 9px}
        .identity-card{display:flex;flex-direction:column;justify-content:center;min-width:150px}
        .logo-title { font-weight: 900; font-size: 18px; color: #00e676; letter-spacing: 0.5px; }
        .trainer-line{font-size:13px;color:#8e98aa;margin-top:5px}
        .header-group-label{display:block;font-size:9px;font-weight:800;color:#8793a8;letter-spacing:.7px;text-transform:uppercase;margin-bottom:5px}
        
        .badge-bar { display:grid;grid-template-columns:repeat(4,28px);align-items:center;gap:4px; }
        .badge-slot{width:28px;height:28px;border-radius:7px;background:linear-gradient(145deg,#202432,#10131c);border:1px solid #353b50;display:flex;align-items:center;justify-content:center;font-size:14px;filter:grayscale(1);opacity:.34;transform:scale(.94);transition:.25s;box-shadow:inset 0 0 10px rgba(0,0,0,.45)}
        .badge-slot.active{filter:none;opacity:1;transform:scale(1);border-color:#ffd54f;background:linear-gradient(145deg,#3b3420,#17191f);box-shadow:0 0 13px rgba(255,213,79,.5)}
        .badge-icon{width:100%;height:100%;object-fit:contain;padding:3px;box-sizing:border-box}
        .live-global{grid-column:1/-1;position:static;background:#0b0e14;border-top:1px solid #283142;padding:6px 12px}
        .live-global-title{display:flex;justify-content:space-between;align-items:center;font-size:11px;font-weight:800;color:#00e676;margin-bottom:8px;letter-spacing:.4px}
        .live-dot{width:8px;height:8px;border-radius:50%;background:#00e676;display:inline-block;box-shadow:0 0 8px #00e676;margin-right:5px}
        .live-global-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}
        .live-stat{background:#11151e;border:1px solid #242a3a;border-radius:8px;padding:6px 8px;text-align:center}
        .live-stat .lv{font-size:12px;font-weight:800;color:#fff}.live-stat .lk{font-size:7px;color:#747d94;text-transform:uppercase;margin-top:2px;letter-spacing:.5px}
        .battle-fighter-marker{background:transparent;border:0}
        .battle-fighter-marker .battle-map-icon{position:relative;width:38px;height:38px;border-radius:9px;overflow:hidden;border:1px solid #ff8a65;box-shadow:0 0 9px rgba(255,61,0,.9);background:#090909}
        .battle-fighter-marker img{display:block;width:100%;height:100%;object-fit:cover;image-rendering:pixelated}
        .battle-fighter-marker b{position:absolute;right:1px;bottom:1px;min-width:12px;padding:1px 3px;border-radius:5px;background:rgba(0,0,0,.88);color:#fff;font:900 9px/12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-align:center}
        .battler-icon{width:17px;height:17px;object-fit:cover;border-radius:4px;vertical-align:-4px;margin-right:4px;image-rendering:pixelated}

        .journey-wrap{margin-bottom:14px;background:#11151e;border:1px solid #242a3a;border-radius:12px;padding:12px}
        .journey-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
        .journey-title h3{margin:0;font-size:13px}.journey-title span{font-size:9px;color:#7f8799}
        .journey-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
        .journey-card{background:#0d1118;border:1px solid #252b3b;border-radius:10px;padding:10px;min-height:82px;position:relative}
        .journey-card.done{border-color:#00e676;box-shadow:0 0 14px rgba(0,230,118,.12)}
        .journey-card.locked{opacity:.55;filter:grayscale(.35)}
        .journey-icon{font-size:22px;margin-bottom:6px}.journey-name{font-size:10px;font-weight:800}
        .journey-icon img{width:28px;height:28px;object-fit:contain;image-rendering:pixelated}
        .journey-sub{font-size:8px;color:#6f788e;margin-top:2px}.journey-value{position:absolute;right:8px;top:8px;font-size:10px;font-weight:800;color:#00e676}
        .journey-bar{height:4px;background:#202635;border-radius:99px;overflow:hidden;margin-top:8px}.journey-fill{height:100%;background:linear-gradient(90deg,#00e676,#4dd0e1);width:0%}
        @media(max-width:1200px){.journey-grid{grid-template-columns:repeat(4,1fr)}}

        .team-bar { display:grid;grid-template-columns:repeat(3,36px);align-items:center;gap:4px; }
        .team-slot {
            width: 36px;
            height: 36px;
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
            width: 29px;
            height: 29px;
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
        .team-label { font-size:9px; color:#7f8799; cursor:pointer; }
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
            display:flex;
            flex-direction:column;
            justify-content:center;
            gap: 6px;
            background: #0e1017;
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid #232738;
            font-size:11px;
        }
        .filter-buttons{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}
        .filter-btn {
            background: #232738;
            border: none;
            color: #aaa;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
        }
        .filter-btn:hover { background: #3b4258; color: #fff; }
        .filter-btn.active{background:#00e676;color:#07120c}

        .header-right { display: flex; align-items: center; gap: 10px; }
        .tabs { display: grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap: 6px; width:100%; }
        .tab-btn { background: #232738; color: #aaa; border: none; padding: 9px 14px; border-radius: 7px; cursor: pointer; font-weight: 700; font-size: 13px; transition: all 0.2s; }
        .tab-btn:hover { color: #fff; background: #2f344a; }
        .tab-btn.active { background: #00e676; color: #000; font-weight: 700; }

        #main-container { flex: 1; position: relative; overflow: hidden; }
        #map-view, #rooms-view, #graphs-view, #watcher-view, #mapper-view, #status-view { width: 100%; height: 100%; position: absolute; top:0; left:0; }
        #rooms-view { display: none; overflow-y: auto; padding: 24px; }
        #graphs-view { display:none; overflow-y:auto; padding:18px 22px 30px; background:#0c0e14; }
        #watcher-view { display: none; overflow-y: auto; }
        .watcher-page{padding:14px;max-width:1100px;margin:0 auto}
        .watcher-page-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:10px;flex-wrap:wrap}
        .watcher-page-head b{color:#00e676;font-size:14px}
        .watcher-page-head span{display:block;color:#8f98ad;font-size:10px;margin-top:2px}
        .watcher-page-head a{color:#7c8db5;font-size:10px;white-space:nowrap}
        /* V17.2: das rohe Composite-Bild ist Spiel+Team+LiveMap nebeneinander
           (1458x638, siehe watch.py GAME_PANEL_W+TEAM_PANEL_W+MAP_PANEL_W).
           Im Web brauchen wir nur Spiel+Team (838px) - die LiveMap rechts
           macht das Layout kaputt (v.a. mobil) und wird hier nicht gebraucht.
           Container mit fester Seitenzahl (838:638) schneidet den Rest per
           overflow:hidden ab; das Bild selbst wird proportional so breit
           skaliert, dass genau der 838px-Ausschnitt die Containerbreite
           fuellt - funktioniert responsiv auf Desktop UND Mobil ohne
           gesonderten Breakpoint. */
        .wt-stream-wrap{width:100%;max-width:420px;margin:0 auto 16px;overflow:hidden;position:relative;aspect-ratio:838/638;border-radius:8px;border:1px solid #293148;background:#0b0e14}
        @media(min-width:601px){.wt-stream-wrap{max-width:760px}}
        #watcher-stream{position:absolute;left:0;top:0;width:calc(100% * 1458 / 838);max-width:none;height:auto;display:block;image-rendering:pixelated}
        .wt-picker-head{display:flex;justify-content:space-between;align-items:center;margin:2px 0 8px}
        .wt-picker-head b{color:#fff;font-size:12px}
        .wt-picker-head span{color:#7f879b;font-size:10px}
        .wt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:6px;margin-bottom:14px}
        .wt-chip{background:#151821;border:1px solid #293148;border-radius:8px;padding:7px 8px;cursor:pointer;transition:.12s;text-align:left}
        .wt-chip:hover{border-color:#3d4a6b;background:#1a1f2b}
        .wt-chip.sel{border-color:#00e676;background:rgba(0,230,118,.12)}
        .wt-chip.watcher{border-color:#7c4dff}
        .wt-chip .c-id{font-weight:800;font-size:11px;color:#dfe6f2}
        .wt-chip .c-role{font-size:9px;color:#8f98ad;text-transform:uppercase;letter-spacing:.4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .wt-chip .c-meta{font-size:10px;color:#9fb0d0;margin-top:3px}
        .wt-detail{background:#12161f;border:1px solid #2b3346;border-radius:12px;padding:14px}
        .wt-detail h4{margin:0 0 2px;color:#00e676;font-size:14px}
        .wt-detail .wt-sub{color:#8f98ad;font-size:10px;margin:2px 0 12px;line-height:1.5}
        .wt-dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:8px;margin-bottom:12px}
        .wt-cell{background:#0b0e14;border:1px solid #232738;border-radius:8px;padding:8px}
        .wt-cell .v{font-size:15px;font-weight:800;color:#fff;word-break:break-word}
        .wt-cell .k{font-size:9px;color:#8b93a7;margin-top:2px;text-transform:uppercase;letter-spacing:.3px}
        .wt-events{background:#0b0e14;border:1px solid #232738;border-radius:8px;padding:9px;font-family:ui-monospace,Menlo,monospace;font-size:10px;line-height:1.75;color:#9fb0d0;max-height:170px;overflow:auto}
        .wt-events .pos{color:#5fe08a}
        .wt-events .neg{color:#ff7a7a}
        .wt-close{float:right;background:#232738;border:none;color:#aaa;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:11px}
        .wt-starter-sprite{float:right;width:56px;height:56px;margin:0 8px 6px 10px;image-rendering:pixelated;background:#0b0e14;border:1px solid #232738;border-radius:8px}
        @media(max-width:600px){
          .watcher-page{padding:10px}
          .wt-grid{grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:5px}
          .wt-dgrid{grid-template-columns:repeat(3,1fr)}
          .wt-stream-wrap{margin-bottom:12px;max-width:100%}
        }
        #mapper-view { display:none; overflow-y:auto; padding:18px; box-sizing:border-box; }
        .mapper-grid { display:grid; grid-template-columns:minmax(360px,1fr) minmax(360px,1fr); gap:16px; }
        .mapper-card { background:#11151f; border:1px solid #273043; border-radius:12px; padding:12px; min-width:0; }
        .mapper-card h3 { margin:0 0 5px; color:#63e6ad; font-size:14px; }
        .mapper-card p { margin:0 0 10px; color:#8e99ad; font-size:11px; }
        .mapper-card img { display:block; width:100%; height:auto; image-rendering:pixelated; border-radius:8px; background:#090c12; }
        .path-toggle.active { background:#00e676 !important; color:#07120c !important; }
        #status-view { display:none; overflow-y:auto; padding:18px 22px 32px; background:#0c0e14; }
        .status-title { display:flex; justify-content:space-between; align-items:end; margin-bottom:14px; }
        .status-title h2 { margin:0; color:#fff; font-size:20px; }
        .status-title span { color:#7f879b; font-size:10px; }
        .status-summary, .status-role-grid, .status-agent-grid { display:grid; gap:10px; }
        .status-summary { display:block; margin-bottom:14px; }
        .status-brain-group{background:#0e1118;border:1px solid #293148;border-radius:13px;padding:12px;margin-bottom:12px}
        .status-brain-group.nav{border-color:rgba(82,222,172,.34)}.status-brain-group.battle{border-color:rgba(255,116,82,.38)}
        .status-brain-head{display:flex;justify-content:space-between;align-items:center;font-size:14px;font-weight:900;margin-bottom:10px}
        .status-brain-head span{font-size:11px;color:#8f9db0}.status-brain-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}
        .rew-break{margin-top:11px;background:#151821;border:1px solid #293148;border-radius:11px;padding:11px}
        .rew-break-h{font-size:10px;color:#8b93a7;text-transform:uppercase;letter-spacing:.4px;margin-bottom:8px}
        .rew-row{display:grid;grid-template-columns:88px 1fr 54px;align-items:center;gap:8px;margin:3px 0;font-size:11px}
        .rew-k{color:#aeb8c8}.rew-v{text-align:right;color:#dfe6f0;font-variant-numeric:tabular-nums}
        .rew-bar{height:9px;background:#0e1118;border-radius:5px;overflow:hidden}.rew-bar i{display:block;height:100%}
        .bp-reason{margin-top:10px;padding:9px 11px;background:#171a12;border:1px solid #3a4327;border-radius:9px;font-size:11px;color:#cdd7c2;line-height:1.5}
        .bp-reason b{color:#e7efd9}
        .status-kpi, .status-watcher, .status-role, .status-agent { background:#151821; border:1px solid #293148; border-radius:11px; padding:12px; }
        .status-kpi .big { color:#00e676; font-size:21px; font-weight:900; }
        .status-kpi .small { color:#8b93a7; font-size:10px; margin-top:3px; }
        .status-kpi.bp-up { border-color:#1f6b3f; }
        .status-kpi.bp-up .big { color:#5fe08a; }
        .status-kpi.bp-down { border-color:#7a2f2f; }
        .status-kpi.bp-down .big { color:#ff7a7a; }
        .status-kpi .bp-delta { font-size:11px; font-weight:700; margin-top:2px; }
        .bp-versions { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
        .bp-ver-chip { background:#151821; border:1px solid #293148; border-radius:9px; padding:8px 12px; font-size:12px; }
        .bp-ver-chip b { color:#00e676; display:block; font-size:13px; }
        .bp-ver-chip span { color:#c3cadb; }
        .bp-ver-sep { color:#4a5266; font-size:14px; }
        .status-watcher { border-color:#7c4dff; margin-bottom:14px; }
        .status-watcher h3, .status-section-title { margin:0 0 9px; color:#fff; font-size:13px; }
        .status-watcher-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:7px; }
        .status-pill { background:#0e1119; border-radius:7px; padding:7px 9px; color:#bac3d8; font-size:11px; }
        .status-pill b { color:#fff; }
        .status-role-grid { grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); margin-bottom:15px; }
        .status-role-head { display:flex; justify-content:space-between; font-weight:800; color:#fff; margin-bottom:8px; }
        .status-role-head b { color:#00e676; }
        .status-role-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:5px; }
        .status-role-stats span { background:#0e1119; padding:6px 4px; border-radius:6px; text-align:center; color:#929caf; font-size:9px; }
        .status-role-stats strong { display:block; color:#fff; font-size:14px; margin-bottom:2px; }
        .status-agent-grid { grid-template-columns:repeat(auto-fit,minmax(245px,1fr)); max-height:25vh; overflow-y:auto; align-content:start; padding:3px 5px 3px 0; scrollbar-gutter:stable; }
        .status-agent { cursor:pointer; transition:border-color .12s, background .12s; }
        .status-agent:hover { border-color:#3d4a6b; }
        .status-agent.sel { border-color:#00e676; background:rgba(0,230,118,.10); }
        .status-agent { padding:9px 10px; }
        .status-agent-top { display:flex; justify-content:space-between; color:#fff; font-size:11px; font-weight:800; }
        .status-agent-meta { color:#8d96aa; font-size:9px; margin-top:5px; line-height:1.5; }
        .status-agent.fighting { border-color:#ff5252; box-shadow:inset 3px 0 #ff5252; }
        .fleet-panel { background:linear-gradient(180deg,#12161f,#0d1017); border:1px solid #2b3346; border-radius:12px; padding:14px 16px; margin-bottom:14px; }
        .fleet-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
        .fleet-head b { font-size:14px; color:#00e676; letter-spacing:.5px; }
        .fleet-head span { font-size:9px; color:#7f879b; text-transform:uppercase; letter-spacing:.6px; }
        .fleet-depth { display:flex; align-items:center; gap:14px; background:#0b0e14; border:1px solid #232738; border-radius:10px; padding:12px 14px; margin-bottom:12px; flex-wrap:wrap; }
        .fleet-depth-num { font-size:38px; font-weight:900; color:#ffd54f; line-height:1; min-width:44px; text-align:center; }
        .fleet-depth-lbl { flex:1; min-width:150px; }
        .fleet-depth-lbl > div:first-child { font-size:16px; font-weight:800; color:#fff; }
        .fleet-depth-hint { font-size:10px; color:#7f879b; margin-top:3px; }
        .fleet-depth-hint b { color:#9fb0d0; }
        .fleet-depth-track { display:flex; gap:4px; width:100%; margin-top:4px; }
        .fleet-depth-track i { flex:1; height:8px; border-radius:3px; background:#232738; font-style:normal; position:relative; }
        .fleet-depth-track i.on { background:#00e676; }
        .fleet-depth-track i.cur { background:#ffd54f; box-shadow:0 0 6px #ffd54f88; }
        .fleet-grid { display:grid; grid-template-columns:repeat(7,1fr); gap:8px; margin-bottom:12px; }
        .fleet-cell { background:#151821; border:1px solid #232738; border-radius:9px; padding:9px 6px; text-align:center; }
        .fleet-cell .fc-v { font-size:19px; font-weight:800; color:#00e676; }
        .fleet-cell .fc-k { font-size:9px; color:#8b93a7; margin-top:2px; }
        .fleet-roles { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
        .fleet-roles span { font-size:10px; background:#1b2030; border:1px solid #2b3346; border-radius:20px; padding:3px 9px; color:#b9c2d8; }
        .fleet-roles span b { color:#fff; }
        .fleet-events { display:flex; flex-direction:column; gap:3px; }
        .fleet-events div { font-size:10px; color:#ffd54f; font-family:ui-monospace,monospace; }
        .fleet-events div.none { color:#5b6478; }
        @media(max-width:820px){ .fleet-grid{grid-template-columns:repeat(3,1fr)!important} }
        .graphs-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:12px; }
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
            top: 20px;
            width: 390px;
            max-height: calc(100% - 40px);
            overflow-y: auto;
            background: rgba(21,24,33,0.94);
            border: 1px solid #232738;
            border-radius: 10px;
            padding: 12px;
            z-index: 1000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            backdrop-filter: blur(8px);
        }
        /* V17.2: verdeckte bisher dauerhaft die linke Kartenhaelfte ("links
           abgeschnitten"). Eingeklappt bleibt nur die Titelzeile sichtbar. */
        .detail-panel.collapsed { max-height: none; overflow: visible; width: auto; }
        .detail-panel.collapsed > *:not(.detail-title) { display: none; }
        .detail-collapse-btn {
            background: #232738; color: #aaa; border: none; border-radius: 6px;
            width: 22px; height: 22px; cursor: pointer; font-size: 12px; line-height: 1;
        }
        .detail-collapse-btn:hover { color: #fff; background: #2f344a; }
        .detail-title { display:flex; justify-content:space-between; align-items:center; gap:8px; color:#00e676; font-weight:700; margin-bottom:8px; }
        .detail-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(70px,1fr)); gap:6px; margin-bottom:8px; }
        .detail-stat { background:#0e1017; border:1px solid #232738; border-radius:6px; padding:6px; text-align:center; }
        .detail-stat .v { font-size:13px; font-weight:700; color:#fff; }
        .detail-stat .k { font-size:8px; color:#777; text-transform:uppercase; letter-spacing:.5px; }
        .chart-wrap { background:#0e1017; border:1px solid #232738; border-radius:6px; padding:5px; margin-top:6px; }
        .chart-label { font-size:9px; color:#888; margin:0 0 3px 4px; }
        .mini-chart { width:100%; height:72px; display:block; }

        .agent-filter-bar {
            position:absolute; left:50%; top:10px; transform:translateX(-50%);
            z-index:1001;
            display:flex; flex-wrap:wrap; gap:6px; align-items:center;
            justify-content:center;
            background:rgba(21,24,33,0.94); border:1px solid #232738;
            border-radius:10px; padding:8px;
            box-shadow:0 8px 24px rgba(0,0,0,0.5); backdrop-filter:blur(8px);
            max-width:min(760px, calc(100vw - 40px));
        }
        .agent-filter-bar select {
            background:#0e1017; color:#cdd6e5; border:1px solid #2f344a;
            border-radius:6px; font-size:11px; padding:5px 6px; max-width:160px;
        }
        .agent-filter-bar .af-reset {
            background:#2f344a; color:#fff; border:0; border-radius:6px;
            font-size:13px; line-height:1; padding:5px 9px; cursor:pointer;
        }
        .agent-filter-bar.filtered { border-color:#00e676; }
    </style>
<style id="champion-night-style">
#brain-summary-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:8px 12px;background:#0c0e14;border-bottom:1px solid #232738;flex:none}
#champion-night-card,#battle-brain-card{position:static;width:auto;padding:10px 12px;border-radius:14px;background:linear-gradient(145deg,rgba(8,17,28,.96),rgba(15,27,41,.93));border:1px solid rgba(68,214,255,.28);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#battle-brain-card{border-color:rgba(255,116,82,.32)}
#battle-brain-card .cn-grid{grid-template-columns:repeat(3,1fr)}
#champion-night-card.minimized{width:185px}
#champion-night-card.minimized .cn-grid,#champion-night-card.minimized .cn-foot{display:none}
.cn-title{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:10px;font-weight:900;letter-spacing:.65px;color:#55e4a3}
.cn-title-right{display:flex;align-items:center;gap:7px}.cn-toggle{border:0;background:rgba(255,255,255,.08);color:#a9bdd2;border-radius:7px;padding:2px 7px;cursor:pointer;font-size:12px}
.cn-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}
.cn-grid div{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.065);border-radius:9px;padding:7px 3px;text-align:center}
.cn-grid b{display:block;font-size:13px;color:#f4f8ff}.cn-grid small{font-size:7px;color:#8191a7;text-transform:uppercase}
.cn-foot{margin-top:7px;font-size:7px;color:#74879d}
.cn-sub{margin-top:6px;font-size:8.5px;color:#cbd9ea;line-height:1.55;letter-spacing:.2px}
.cn-sub b{color:#ffd7c4;font-weight:800}
.cn-sub .sep{color:#4a5f76;margin:0 5px}
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
.leaflet-container{touch-action:none}

/* =====================  HANDY / MOBILE  =====================
   Grundsatz: alles ist ein normaler vertikaler Scroll aus festen
   Bloecken. NUR die Leaflet-Karte behaelt eine feste Hoehe (Pinch-Zoom).
   Nichts schwebt, nichts ueberlappt, nichts ist verschiebbar oder
   minimierbar. */
@media(max-width:820px){
  html{-webkit-text-size-adjust:100%}
  body{height:auto!important;min-height:100vh!important;overflow-x:hidden!important;overflow-y:auto!important;-webkit-overflow-scrolling:touch;display:flex!important;flex-direction:column!important}

  /* Reihenfolge: Header zuerst, dann die Info-Kacheln, dann der Inhalt */
  header{order:1!important}
  #brain-summary-row{order:2!important}
  #main-container{order:3!important}

  /* HEADER: klebt oben, Tabs zuerst und immer sichtbar */
  header{position:sticky!important;top:0!important;z-index:300!important;flex-direction:column!important;align-items:stretch!important;gap:6px!important;padding:6px 8px!important}
  .header-left{order:2!important;flex-wrap:wrap!important;gap:6px 10px!important;align-items:center!important}
  .header-right{order:1!important;width:100%!important}
  .tabs{display:flex!important;width:100%!important;gap:4px!important;flex-wrap:nowrap!important}
  .tabs .tab-btn{flex:1 1 0!important;min-width:0!important;font-size:10px!important;padding:8px 4px!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}
  .logo-title{font-size:13px!important}
  .team-label{font-size:9px!important}
  .team-bar{gap:3px!important}
  .team-slot{width:24px!important;height:24px!important}
  .badge-bar{padding:3px 5px!important;gap:2px!important}
  .badge-slot{width:19px!important;height:19px!important;font-size:10px!important}
  .filter-control{width:100%!important;flex-wrap:wrap!important}

  /* MAIN: kein 100vh-Kaefig mehr, alles im Fluss */
  #main-container{position:static!important;overflow:visible!important;flex:none!important;height:auto!important}
  #graphs-view,#status-view,#watcher-view{position:relative!important;display:none;height:auto!important;min-height:auto!important;overflow:visible!important;padding:10px 10px calc(72px + env(safe-area-inset-bottom))!important}
  /* Leaflet braucht feste Hoehe, sonst kollabiert die Karte / kein Zoom */
  #map-view{position:relative!important;height:72vh!important;min-height:360px!important;overflow:hidden!important}

  /* ALLE Panels: feste Kacheln - nicht schwebend, nicht verschiebbar,
     nicht minimierbar */
  #brain-summary-row,#learner-truth-hud,#champion-night-card,#battle-brain-card,.pkmai-movable,.hud-overlay,.detail-panel,
  .agent-filter-bar,.pkmai-mobile-tile,.live-global,.v81skills{
    position:static!important;inset:auto!important;left:auto!important;right:auto!important;top:auto!important;bottom:auto!important;
    transform:none!important;width:auto!important;max-width:none!important;max-height:none!important;
    z-index:auto!important;box-shadow:none!important;backdrop-filter:none!important;
  }
  #brain-summary-row{grid-template-columns:1fr!important;gap:8px!important;margin:0!important;padding:8px!important}
  #brain-summary-row #learner-truth-hud,#brain-summary-row #champion-night-card,#brain-summary-row #battle-brain-card{margin:0!important}
  .pkmai-mobile-tile{display:block!important}
  .agent-filter-bar select{flex:1 1 42%!important;max-width:none!important;font-size:12px!important;padding:8px 6px!important}
  /* Verschiebe- / Minimier- / Ausblende-Knoepfe komplett weg */
  .pkmai-float-tools,#pkmai-hidden-tray,
  #learner-truth-hud .lth-head button,#champion-night-card .cn-toggle{display:none!important}
  /* falls vorher minimiert/ausgeblendet: erzwungen wieder voll anzeigen */
  .pkmai-minimized{height:auto!important;min-height:0!important;overflow:visible!important}
  .pkmai-hidden{display:block!important}
  #learner-truth-hud.minimized .lth-body{display:grid!important}
  #champion-night-card.minimized .cn-grid{display:grid!important}
  #champion-night-card.minimized .cn-foot{display:block!important}
  #champion-night-card.minimized{width:auto!important}

  .graphs-grid{grid-template-columns:1fr!important}
  .graphs-kpis{grid-template-columns:repeat(2,1fr)!important}
  .journey-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
  .room-grid{grid-template-columns:1fr!important}
  .mapper-grid{grid-template-columns:1fr!important}
  .agent-row,.agent-watcher{padding:11px 8px!important;font-size:12px!important}
  .step-list{max-height:220px!important}
}
@media(max-width:480px){
  .journey-grid{grid-template-columns:1fr!important}
  .graph-card{min-height:320px!important}
  .tabs .tab-btn{font-size:9px!important;padding:8px 3px!important}
  .team-slot{width:21px!important;height:21px!important}
}
@media(max-width:820px){
  /* Kein horizontales Scrollen der ganzen Seite auf dem Handy. */
  html,body{overflow-x:hidden!important;max-width:100vw!important}
  #map-view,#rooms-view,#graphs-view,#watcher-view,#mapper-view,#status-view{overflow-x:hidden!important}
  .v81skills,.v8-skill-cards,.v84-grid{flex-wrap:wrap!important;overflow-x:auto!important;max-width:100%!important}
  .v81card{flex:0 0 auto!important}
  canvas{max-width:100%!important}
}
</style>
<style id="learner-truth-style">
#learner-truth-hud{position:fixed;left:16px;bottom:16px;z-index:4900;width:300px;padding:10px 12px;border-radius:13px;background:rgba(7,14,23,.96);border:1px solid rgba(82,222,172,.30);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#learner-truth-hud.minimized .lth-body{display:none}.lth-head{display:flex;justify-content:space-between;align-items:center;color:#58e3a5;font-size:10px;font-weight:900}.lth-head button{border:0;background:rgba(255,255,255,.08);color:#b7c7d8;border-radius:7px;padding:2px 7px}
.lth-body{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}.lth-cell{background:rgba(255,255,255,.045);border-radius:8px;padding:7px 5px;text-align:center}.lth-cell b{display:block;font-size:13px;color:#f5f8fc}.lth-cell small{font-size:7px;color:#8393a8;text-transform:uppercase}
@media(min-width:821px){#learner-truth-hud{position:static;inset:auto;width:auto;z-index:auto}}
@media(max-width:820px){#learner-truth-hud{position:static!important;left:auto!important;right:auto!important;top:auto!important;bottom:auto!important;transform:none!important;width:auto!important;z-index:auto!important}}
</style>

<style id="continuous-page-style">
html,body{height:100%!important;min-height:0!important;overflow:hidden!important}
body{height:100dvh!important;display:flex!important;flex-direction:column!important;background:#0c0e14}
header{flex:none;flex-wrap:wrap;gap:8px;position:relative!important}
.header-left{gap:10px;flex:1 1 auto;grid-template-columns:auto auto auto auto!important}
.header-right{flex:1 0 100%;width:100%}
.header-right .tab-btn{width:100%}
.filter-control{min-width:128px}
.team-label small{display:block;color:#657187;font-size:8px;margin-top:2px;text-transform:none;letter-spacing:0}
.battle-champion-mini{margin-top:9px;padding-top:8px;border-top:1px solid #283142}
.battle-champion-mini .cn-grid{grid-template-columns:repeat(3,1fr)}
.agent-list-meta{display:block;text-align:right;line-height:1.35;color:#aeb8ca;font-size:10px}
.cn-title,.lth-head{font-size:12px!important}.cn-grid b,.lth-cell b{font-size:16px!important}
.cn-grid small,.lth-cell small{font-size:9px!important}.cn-foot{font-size:9px!important}
.live-stat .lv{font-size:14px!important}.live-stat .lk{font-size:9px!important}
.detail-stat .v{font-size:16px!important}.detail-stat .k{font-size:9px!important}
#brain-summary-row{flex:none;width:100%;box-sizing:border-box;grid-template-columns:1fr 1fr 1fr!important;gap:0!important;padding:0!important;margin:0!important}
#learner-truth-hud,#champion-night-card,#battle-brain-card,.live-global,.hud-overlay,.detail-panel,.agent-filter-bar{
position:static!important;inset:auto!important;transform:none!important;width:auto!important;max-width:none!important;max-height:none!important;
background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;backdrop-filter:none!important;
padding:12px!important;margin:0!important;box-sizing:border-box}
#brain-summary-row>div{min-width:0;border-right:1px solid #283142!important}
.live-global{grid-column:1/-1!important;border-right:0!important;border-top:1px solid #283142!important;padding:5px 10px!important}
.live-global-grid{grid-template-columns:repeat(6,minmax(0,1fr));gap:4px}
.cn-grid div,.lth-cell,.live-stat,.detail-stat{background:transparent!important;border:0!important;border-radius:0!important}
#main-container{position:relative!important;flex:1 1 0!important;min-height:0!important;overflow:hidden!important;height:auto!important;order:3}
#map-workspace{display:grid;grid-template-columns:minmax(230px,26%) minmax(0,1fr) minmax(210px,22%);height:100%;width:100%;min-height:0}
#alex-watcher-column{min-width:0;min-height:0;overflow-y:auto;border-right:1px solid #283142}
#map-column{display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}
#map-column #map-view{position:relative!important;inset:auto!important;flex:1 1 0;height:auto!important;min-height:0!important;width:100%;overflow:hidden!important}
#map-column .agent-filter-bar{flex:none;justify-content:flex-start;gap:5px;border-bottom:1px solid #283142!important}
#map-column .agent-filter-bar select{min-width:0;max-width:170px;font-size:12px}
#map-workspace #hud{min-width:0;min-height:0;overflow-y:auto!important;border-left:1px solid #283142!important}
#alex-watcher-column .wt-stream-wrap{width:100%;max-width:none;margin:0;border:0;border-radius:0}
#alex-watcher-stream{position:static;width:100%;height:auto;display:block;image-rendering:pixelated}
#alex-watcher-column .wt-stream-wrap{aspect-ratio:3/2}
#alex-watcher-column .v8-agent-party{display:none}
#language-toggle{position:absolute;right:7px;top:6px;z-index:2;background:#303236;color:#eee;border:1px solid #555;border-radius:5px;padding:3px 7px;font-size:10px;font-weight:800;cursor:pointer}
.badge-group{position:relative;padding-right:42px!important}
#watcher-view .wt-stream-wrap{aspect-ratio:1200/570;max-width:1000px}
#watcher-stream{width:100%;position:static}
.leaflet-container{background:#202124!important}
.map-name-label{background:rgba(10,14,20,.75);color:#cfe0cf;font:700 11px/1.3 -apple-system,BlinkMacSystemFont,sans-serif;padding:2px 7px;border-radius:4px;white-space:nowrap;border:1px solid rgba(127,135,155,.4);width:auto!important;height:auto!important;pointer-events:none}
.room-map-canvas{background:#202124}
.alex-watcher-title{padding:12px;color:#00e676;font-size:12px;font-weight:700;border-bottom:1px solid #283142}
#alex-watcher-column #detail-panel{border-top:1px solid #283142!important}
#rooms-view,#graphs-view,#watcher-view,#mapper-view,#status-view{position:absolute!important;inset:0;height:100%!important;overflow-y:auto!important;box-sizing:border-box}
.pkmai-float-tools,#pkmai-hidden-tray{display:none!important}
.hud-overlay{font-size:12px!important}.hud-overlay .agent-row{padding:10px 4px}
/* V17.3: eigene Agenten-Kachel-Sektion fuer die mobile Kartenansicht -
   auf Desktop unsichtbar, dort erledigt die rechte Sidebar (#hud) das. */
#mobile-agent-section{display:none}
@media(max-width:820px){
/* Desktop ist ein fixes Ein-Bildschirm-Layout (kein Body-Scroll noetig,
   alles per flex/grid exakt in die Viewport-Hoehe gepresst). Auf dem Handy
   muss die Seite dagegen ganz normal scrollbar sein, sonst verschwinden
   Watcher + Agenten-Kacheln unterhalb der Karte komplett unerreichbar. */
html,body{height:auto!important;overflow:auto!important;overflow-y:auto!important}
#main-container{overflow:visible!important;flex:none!important}
header{padding:6px!important;gap:6px!important}
.header-left{display:grid!important;grid-template-columns:1fr 1fr!important;width:100%!important;gap:6px!important}
.identity-card{grid-column:1/-1;min-width:0!important}.filter-control{grid-column:1/-1}
.team-bar{grid-template-columns:repeat(3,34px)!important}.team-slot{width:34px!important;height:34px!important}
.badge-bar{grid-template-columns:repeat(4,25px)!important}.badge-slot{width:25px!important;height:25px!important}
#brain-summary-row{grid-template-columns:1fr!important}
#brain-summary-row>div{padding:9px!important;border-right:0!important;border-bottom:1px solid #283142!important}
.live-global-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important}
.cn-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
.lth-body{grid-template-columns:1fr}.cn-foot{display:none}
/* Handy: Karte oben (volle Breite), dann Watcher, dann Agenten als
   antippbare Kacheln statt der schmalen Listen-Sidebar. */
#map-workspace{display:flex!important;flex-direction:column!important;height:auto!important;min-height:0!important;overflow:visible!important}
#map-column{order:1;flex:none!important;min-width:0}
#map-column #map-view{height:52vh!important;min-height:300px!important;flex:none!important;width:100%!important}
#alex-watcher-column{order:2;border-right:0!important;border-bottom:1px solid #283142!important;min-width:0}
#alex-watcher-column #detail-panel{display:none!important}
#map-workspace #hud{display:none!important}
#mobile-agent-section{display:block!important;order:3;padding:8px;border-top:1px solid #283142}
#map-workspace #hud,#alex-watcher-column #detail-panel{padding:6px!important}
#map-column .agent-filter-bar{padding:5px!important}
#map-column .agent-filter-bar select{flex:1 1 40%!important;padding:4px!important;font-size:9px!important}
.detail-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
.agent-row{flex-wrap:wrap;overflow-wrap:anywhere}
}
@media(min-width:821px) and (max-width:1400px){
  .header-left{grid-template-columns:auto auto auto!important}.filter-control{grid-column:1/-1;flex-direction:row!important;justify-content:flex-start!important}
  .header-right{align-self:flex-start}.tab-btn{font-size:12px!important;padding:8px 10px!important}
}
</style>
<script src="/dashboard-language.js"></script>
</head>
<body>
    <header>
        <div class="header-left">
            <div class="identity-card"><div class="logo-title">⚡ PKMAI <span id="model-ver" style="color:#5795ff;">v000001</span></div>
              <div class="trainer-line">Trainer: <b id="hud-trainer" style="color:#00e676;">Alex</b></div>
            </div>
            <div class="header-group team-group"><span class="header-group-label team-label" onclick="openFirstPokemon()">Pokémon-Team<small>anklicken für Details</small></span><div class="team-bar">
                <div class="team-slot" id="slot-0"><span style="font-size: 9px; color: #444;">1</span></div>
                <div class="team-slot" id="slot-1"><span style="font-size: 9px; color: #444;">2</span></div>
                <div class="team-slot" id="slot-2"><span style="font-size: 9px; color: #444;">3</span></div>
                <div class="team-slot" id="slot-3"><span style="font-size: 9px; color: #444;">4</span></div>
                <div class="team-slot" id="slot-4"><span style="font-size: 9px; color: #444;">5</span></div>
                <div class="team-slot" id="slot-5"><span style="font-size: 9px; color: #444;">6</span></div>
            </div></div>

            <div class="header-group badge-group"><button id="language-toggle" aria-label="Switch language / Sprache wechseln">DE</button><span class="header-group-label">Kanto-Orden · 1 bis 8</span><div class="badge-bar" title="Kanto Orden">
                <div class="badge-slot" id="badge-1" title="1 · Felsorden"><img class="badge-icon" src="/badges/1.png?v=kanto2" alt="Felsorden"></div>
                <div class="badge-slot" id="badge-2" title="2 · Quellorden"><img class="badge-icon" src="/badges/2.png?v=kanto2" alt="Quellorden"></div>
                <div class="badge-slot" id="badge-3" title="3 · Donnerorden"><img class="badge-icon" src="/badges/3.png?v=kanto2" alt="Donnerorden"></div>
                <div class="badge-slot" id="badge-4" title="4 · Farborden"><img class="badge-icon" src="/badges/4.png?v=kanto2" alt="Farborden"></div>
                <div class="badge-slot" id="badge-5" title="5 · Seelenorden"><img class="badge-icon" src="/badges/5.png?v=kanto2" alt="Seelenorden"></div>
                <div class="badge-slot" id="badge-6" title="6 · Sumpforden"><img class="badge-icon" src="/badges/6.png?v=kanto2" alt="Sumpforden"></div>
                <div class="badge-slot" id="badge-7" title="7 · Vulkanorden"><img class="badge-icon" src="/badges/7.png?v=kanto2" alt="Vulkanorden"></div>
                <div class="badge-slot" id="badge-8" title="8 · Erdorden"><img class="badge-icon" src="/badges/8.png?v=kanto2" alt="Erdorden"></div>
            </div></div>

            <div class="filter-control">
                <span class="header-group-label">Agents anzeigen</span>
                <div class="filter-buttons"><button class="filter-btn" onclick="setFilter(5)">5</button>
                  <button class="filter-btn" onclick="setFilter(10)">10</button>
                  <button class="filter-btn" onclick="setFilter(20)">20</button>
                  <button class="filter-btn" onclick="setFilter(9999)">Alle</button></div>
            </div>
        </div>

        <div class="header-right">
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('map', event)">🗺️ Overview</button>
                <button class="tab-btn" onclick="showTab('graphs', event)">📈 Learning</button>
                <button class="tab-btn" onclick="showTab('status', event)">📊 Status</button>
                <button class="tab-btn" onclick="showTab('watcher', event)">👁️ Watcher</button>
            </div>
        </div>
    </header>
<div id="brain-summary-row">
<div id="learner-truth-hud"><div class="lth-head"><span>🧠 NAVIGATION BRAIN · LIVE</span><span id="nav-brain-state">TRAINING</span></div><div class="lth-body"><div class="lth-cell"><b id="lth-learner">0</b><small>Learner Steps</small></div><div class="lth-cell"><b id="lth-champion">0</b><small>Champion Steps</small></div><div class="lth-cell"><b id="lth-delta">0</b><small>Seit Champion</small></div></div></div>
<div id="champion-night-card">
  <div class="cn-title"><span>🏆 NAVIGATION CHAMPION</span><span class="cn-title-right"><span id="cn-ver">v0</span></span></div>
  <div class="cn-grid">
    <div><b id="cn-steps">0</b><small>Steps</small></div>
    <div><b id="cn-stage">0</b><small>Stage</small></div>
    <div><b id="cn-maps">0</b><small>Maps</small></div>
    <div><b id="cn-runs">0</b><small>Full Runs</small></div>
  </div>
  <div class="cn-foot">Promotion nur durch reproduzierbaren geografischen Fortschritt</div>
  <div class="battle-champion-mini">
    <div class="cn-title"><span><img class="battler-icon" src="/battler.png" alt="">BATTLE CHAMPION</span><span id="cn-battle-ver">RULE</span></div>
    <div class="cn-grid">
      <div><b id="cn-battle-win">–</b><small>Siegrate</small></div>
      <div><b id="cn-battle-turns">–</b><small>Ø Sieger-Züge</small></div>
      <div><b id="cn-battle-hp">–</b><small>Rest-HP</small></div>
    </div>
  </div>
</div>
<div id="battle-brain-card">
  <div class="cn-title"><span><img class="battler-icon" src="/battler.png" alt="">BATTLE BRAIN · LIVE</span><span class="cn-title-right"><span id="bb-state">TRAINING</span></span></div>
  <div class="cn-sub">Training <b id="bb-learner-ver">v0</b> · <b id="bb-steps">0</b> Steps<span class="sep">|</span>Live <b id="bb-champion">RULE</b><span class="sep">|</span>Nächste Prüfung <b id="bb-next-check">–</b></div>
  <div class="cn-sub">Ø Rwd/100 <b id="bb-avgrew">0.0</b><span class="sep">|</span>letzte 200: <b id="bb-roll">– / – / –</b> (Sieg/Wipe/Flucht)<span class="sep">|</span>Prüfung <b id="bb-lastcheck">–</b></div>
  <div class="cn-grid">
    <div><b id="bb-updates">0</b><small>PPO Updates</small></div>
    <div><b id="bb-episodes">0</b><small>Kämpfe</small></div>
    <div><b id="bb-kos">0</b><small>Gegner-K.O.</small></div>
    <div><b id="bb-wins">0</b><small>Siege</small></div>
    <div><b id="bb-wipes">0</b><small>Wipes</small></div>
    <div><b id="bb-flees">0</b><small>Fluchten</small></div>
  </div>
  <div class="cn-foot"><span id="bb-workers">9 Battle</span> · <span id="bb-scenarios">0 Szenarien</span> · <span id="bb-reason">–</span></div>
</div>
<div class="live-global" id="system-pulse">
  <div class="live-global-grid">
    <div class="live-stat"><div class="lv" id="pulse-nav-eval">–</div><div class="lk">🧠 Nav-Prüfung</div></div>
    <div class="live-stat"><div class="lv" id="pulse-battle-outcomes">0</div><div class="lk">⚔ Fluchten / Timeouts</div></div>
    <div class="live-stat"><div class="lv" id="pulse-nav-workers">0 / 40</div><div class="lk">🧭 FULL aktiv</div></div>
    <div class="live-stat"><div class="lv" id="pulse-battle-workers">0 / 9</div><div class="lk"><img class="battler-icon" src="/battler.png" alt="">Battler aktiv</div></div>
    <div class="live-stat"><div class="lv" id="pulse-scenarios">0</div><div class="lk">🧪 Kampfszenarien</div></div>
    <div class="live-stat"><div class="lv" id="pulse-runs">0</div><div class="lk">✅ FULL Runs</div></div>
  </div>
</div>
</div>

    <div id="main-container">
        <div id="map-view"><div class="v81skills" hidden><div id="v81skills"></div><span id="v81-agent-count"></span></div></div>
        <div id="watcher-view"><section class="watcher-page">
            <div class="watcher-page-head"><div><b>● LIVE WATCHER</b><span>End-to-End-Screenshot + Live-Stats jedes Agenten</span></div><a href="/watcher.jpg" target="_blank" rel="noopener">JPEG ↗</a></div>
            <div class="wt-stream-wrap"><img id="watcher-stream" src="/watcher.jpg" alt="Live-Bild des Watchers"></div>
            <div class="wt-detail" id="wt-detail" hidden></div>
            <div class="wt-picker-head"><b>Agenten – antippen für Live-Stats</b><span id="wt-count">–</span></div>
            <div class="wt-grid" id="wt-grid"></div>
        </section></div>
        <!-- Mapper-Tab entfernt: der Mapper laeuft standardmaessig nicht
             (start_all.sh --no-mapper), eine dauerleere Ansicht verwirrt nur. -->
        <div id="status-view">
            <div class="status-title"><h2>📊 2×2 Status</h2><span>Navigation Brain · Battle Brain · 40 FULL Agents</span></div>
            <div class="status-summary" id="status-summary"></div>
            <div class="status-watcher" id="status-watcher"></div>
            <h3 class="status-section-title">🧭 FULL Agents <span style="font-weight:400;color:#7f879b;font-size:11px">– alle 40 trainieren Navigation vom Master-Start</span></h3>
            <div class="wt-detail" id="status-agent-detail" hidden style="margin-bottom:12px"></div>
            <div class="status-agent-grid" id="status-agent-grid"></div>
        </div>
        <div id="graphs-view"><div class="fleet-panel" id="fleet-panel">
            <div class="fleet-head">
                <b>🧭 FLOTTEN-STATUS</b>
                <span id="fleet-sub">Live aus den Agenten</span>
            </div>
            <div class="fleet-depth">
                <div class="fleet-depth-num" id="fleet-depth-num">0</div>
                <div class="fleet-depth-lbl">
                    <div id="fleet-depth-name">–</div>
                    <div class="fleet-depth-hint">Welt-Tiefe · tiefster Checkpoint <b id="fleet-cp">outdoor_0</b></div>
                </div>
                <div class="fleet-depth-track" id="fleet-depth-track"></div>
            </div>
            <div class="fleet-grid">
                <div class="fleet-cell"><div class="fc-v" id="fleet-full">0</div><div class="fc-k">🧭 FULL Telemetrie</div></div>
                <div class="fleet-cell"><div class="fc-v" id="fleet-outdoor">0</div><div class="fc-k">🌳 gerade draußen</div></div>
                <div class="fleet-cell"><div class="fc-v" id="fleet-brk">0</div><div class="fc-k">🚩 Tiefen-Durchbrüche</div></div>
                <div class="fleet-cell"><div class="fc-v" id="fleet-tiles">0</div><div class="fc-k">🟩 erkundete Tiles</div></div>
                <div class="fleet-cell"><div class="fc-v" id="fleet-fighters">0</div><div class="fc-k"><img class="battler-icon" src="/battler.png" alt="">Battle</div></div>
                <div class="fleet-cell"><div class="fc-v" id="fleet-scenarios">0</div><div class="fc-k">🧪 Szenarien</div></div>
            </div>
            <div class="fleet-roles" id="fleet-roles"></div>
            <div class="fleet-events" id="fleet-events"></div>
        </div>
            <div class="journey-wrap">
                <div class="journey-title"><h3>🗺️ Fortschritt</h3><span>nur Geografie und Story</span></div>
                <div class="journey-grid">
                    <div class="journey-card" id="journey-stage"><div class="journey-value" id="jv-stage">0</div><div class="journey-icon">🚩</div><div class="journey-name">Stage</div><div class="journey-sub">Tiefster bestätigter Fortschritt</div></div>
                    <div class="journey-card" id="journey-maps"><div class="journey-value" id="jv-maps">0</div><div class="journey-icon">🗺️</div><div class="journey-name">Progress-Maps</div><div class="journey-sub">Champion-Evaluation, ohne Häuserzählung</div></div>
                    <div class="journey-card" id="journey-badges"><div class="journey-value" id="jv-badges">0/8</div><div class="journey-icon"><img src="/badges/1.png?v=kanto2" alt="Felsorden"></div><div class="journey-name">Nächster Orden: Felsorden</div><div class="journey-sub">Kanto-Orden in richtiger Reihenfolge</div><div class="journey-bar"><div class="journey-fill" id="jf-badges"></div></div></div>
                </div>
            </div>
            <div class="graphs-kpis">
                <div class="graphs-kpi"><div class="v" id="g-steps">0</div><div class="k">Navigation Steps</div></div>
                <div class="graphs-kpi"><div class="v" id="g-version">v0</div><div class="k">Navigation Champion</div></div>
                <div class="graphs-kpi"><div class="v" id="g-champ-delta">0</div><div class="k">Seit Champion</div></div>
                <div class="graphs-kpi"><div class="v" id="g-episodes">0</div><div class="k">FULL Runs</div></div>
                <div class="graphs-kpi"><div class="v" id="g-battle-steps">0</div><div class="k">Battle Steps</div></div>
                <div class="graphs-kpi"><div class="v" id="g-battle-updates">0</div><div class="k">Battle PPO Updates</div></div>
                <div class="graphs-kpi"><div class="v" id="g-battle-episodes">0</div><div class="k">Battle Episoden</div></div>
                <div class="graphs-kpi"><div class="v" id="g-battle-kos">0</div><div class="k">Battle K.O.</div></div>
                <div class="graphs-kpi"><div class="v" id="g-world-depth">0</div><div class="k">Geografische Stage</div></div>
            </div>
            <div class="graphs-grid">
                <div class="graph-card"><div class="graph-title">Navigation Learning</div><div class="graph-sub">Episoden-Reward des aktuellen Trainer-Laufs; Neustarts werden nicht verbunden.</div><div class="graph-canvas-wrap"><canvas id="graph-reward"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">Exploration Growth</div><div class="graph-sub">Emulator-bestätigte, persistente Kanten und Maps im gerichteten Bewegungsgraphen.</div><div class="graph-canvas-wrap"><canvas id="graph-success"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">Geografischer Fortschritt</div><div class="graph-sub">Maps, Story-Stages und Orden; Level zählt nicht als Navigation.</div><div class="graph-canvas-wrap"><canvas id="graph-progress"></canvas></div></div>
                <div class="graph-card"><div class="graph-title">Battle Learning</div><div class="graph-sub">Battle-Schritte, Episoden und K.O. seit Öffnen dieser Ansicht.</div><div class="graph-canvas-wrap"><canvas id="graph-battle"></canvas></div></div>
            </div>
        </div>
        <div class="agent-filter-bar" id="agent-filter-bar">
            <select id="af-map" onchange="setAgentFilter('map',this.value)">
                <option value="">Alle Orte</option>
            </select>
            <select id="af-stage" onchange="setAgentFilter('stage',this.value)">
                <option value="">Alle Stages</option>
            </select>
            <select id="af-sort" onchange="setAgentFilter('sort',this.value)">
                <option value="">Sortierung: Standard</option>
                <option value="progress">Weitester Fortschritt</option>
                <option value="steps">Meiste Run-Steps</option>
                <option value="reward">Höchster Reward</option>
            </select>
            <button id="path-toggle" class="af-reset path-toggle" onclick="toggleAgentPaths()" title="Zusätzliche echte Nachbarschritte der Savestate-Runner anzeigen">👣 Weitere Wege: aus</button>
            <button class="af-reset" onclick="resetAgentFilter()">×</button>
        </div>
        <div class="hud-overlay" id="hud">Lade 35 Agenten...</div>

        <div class="detail-panel" id="detail-panel">
            <div class="detail-title">
                <span id="detail-name">Watcher</span>
                <span id="detail-room" style="font-size:9px;color:#777;flex:1">-</span>
                <button class="detail-collapse-btn" id="detail-collapse-btn" onclick="toggleDetailPanel()" title="Ein-/ausklappen (Karte freigeben)">▾</button>
            </div>
            <div class="detail-stats">
                <div class="detail-stat"><div class="v" id="detail-steps">0</div><div class="k">Steps</div></div>
                <div class="detail-stat"><div class="v" id="detail-reward">0</div><div class="k">Episode Reward</div></div>
                <div class="detail-stat"><div class="v" id="detail-stage">0</div><div class="k">Geografische Stage</div></div>
                <div class="detail-stat"><div class="v" id="detail-position">0,0</div><div class="k">Position X,Y</div></div>
                <div class="detail-stat"><div class="v" id="detail-tiles">0</div><div class="k">Erkundete Tiles</div></div>
                <div class="detail-stat"><div class="v" id="detail-runsteps">0</div><div class="k">Run-Steps</div></div>
                <div class="detail-stat"><div class="v" id="detail-remaining">–</div><div class="k">Steps bis Reset</div></div>
                <div class="detail-stat"><div class="v" id="detail-inbattle">–</div><div class="k">Im Kampf</div></div>
                <div class="detail-stat"><div class="v" id="detail-badges">0 / 8</div><div class="k">Orden</div></div>
            </div>
            <div class="chart-wrap">
                <div class="chart-label">Letztes Reward-Event</div>
                <div id="detail-event" style="font-size:12px;color:#00e676;padding:6px 8px;">-</div>
            </div>
            <div class="chart-wrap">
                <div class="chart-label">Letzte 10 Rewards · Extremwerte (⬆ höchster + / ⬇ tiefster −)</div>
                <div class="wt-events" id="detail-recent-rewards"><div>-</div></div>
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
        let maxVisibleAgents = 9999; // "Alle" - kein fest verdrahtetes Envs-Limit mehr
        let selectedAgentId = null; // null = alle Agenten sichtbar.
        let latestInstances = [];
        let showAgentPaths = false;
        const historyByAgent = {};

        function setFilter(n) {
            maxVisibleAgents = parseInt(n);
            document.querySelectorAll('.filter-btn').forEach(b=>
                b.classList.toggle('active', (n===9999 && b.textContent.trim()==='Alle') || b.textContent.trim()===String(n))
            );
        }

        function updateFilter(val) {
            maxVisibleAgents = parseInt(val) || 0;
        }

        function clearAgentPaths() {
            Object.keys(agentPolylines).forEach(id => {
                map.removeLayer(agentPolylines[id]);
                delete agentPolylines[id];
            });
            Object.keys(agentStepDots).forEach(id => {
                agentStepDots[id].forEach(dot => map.removeLayer(dot));
                delete agentStepDots[id];
            });
            Object.keys(persistentEdgeLayers).forEach(key => {
                map.removeLayer(persistentEdgeLayers[key]);
                delete persistentEdgeLayers[key];
            });
            Object.keys(persistentTransitionLayers).forEach(key => {
                map.removeLayer(persistentTransitionLayers[key]);
                delete persistentTransitionLayers[key];
            });
        }

        function toggleAgentPaths() {
            showAgentPaths = !showAgentPaths;
            const button = document.getElementById('path-toggle');
            if (button) {
                button.textContent = showAgentPaths ? '👣 Weitere Wege: an' : '👣 Weitere Wege: aus';
                button.classList.toggle('active', showAgentPaths);
            }
            if (!showAgentPaths) clearAgentPaths();
            updateDashboard();
        }

        // V17.2: das Detail-Panel deckte dauerhaft die linke Kartenhaelfte
        // ab. Eingeklappt bleibt nur die Titelzeile, Karte ist voll sichtbar.
        function applyDetailPanelCollapsed(collapsed) {
            const panel = document.getElementById('detail-panel');
            const btn = document.getElementById('detail-collapse-btn');
            if (panel) panel.classList.toggle('collapsed', collapsed);
            if (btn) btn.textContent = collapsed ? '▸' : '▾';
        }
        let detailPanelCollapsed = false;
        try {
            detailPanelCollapsed = localStorage.getItem('pkmai_detail_collapsed') === '1';
        } catch (_) {}
        function toggleDetailPanel() {
            detailPanelCollapsed = !detailPanelCollapsed;
            applyDetailPanelCollapsed(detailPanelCollapsed);
            try {
                localStorage.setItem('pkmai_detail_collapsed', detailPanelCollapsed ? '1' : '0');
            } catch (_) {}
        }
        applyDetailPanelCollapsed(detailPanelCollapsed);

        // ---------- Agenten-Filter ----------
        const agentFilter = { role:'', map:'', starter:'', stage:'', sort:'' };

        function setAgentFilter(key, val) {
            agentFilter[key] = val;
            const bar = document.getElementById('agent-filter-bar');
            if (bar) bar.classList.toggle('filtered',
                !!(agentFilter.role || agentFilter.map || agentFilter.starter
                   || agentFilter.stage || agentFilter.sort));
            if (typeof updateDashboard === 'function') updateDashboard();
        }

        function resetAgentFilter() {
            Object.keys(agentFilter).forEach(k => agentFilter[k] = '');
            ['af-map','af-stage','af-sort'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            const bar = document.getElementById('agent-filter-bar');
            if (bar) bar.classList.remove('filtered');
            if (typeof updateDashboard === 'function') updateDashboard();
        }

        function agentPassesFilter(i) {
            const f = agentFilter;
            if (f.role && String(i.training_objective || i.agent_role || '') !== f.role) return false;
            if (f.map && String(i.room || '') !== f.map) return false;
            if (f.starter === '1' && !i.has_starter) return false;
            if (f.starter === '0' && i.has_starter) return false;
            if (f.stage && String(i.story_stage || '') !== f.stage) return false;
            return true;
        }

        const STAGE_ORDER = {
            INTRO:0, F2_TO_STAIRS:1, F1_TO_EXIT:2, OUTDOOR:3,
            STARTER:4, PROGRESS:5, BATTLE:6, BADGE1:7
        };
        function agentProgressRank(i) {
            let r = (STAGE_ORDER[i.story_stage] || 0) * 1000;
            r += Number(i.visited_maps || 0) * 40;
            r += Number(i.level || 0) * 5;
            r += Number(i.badges || 0) * 500;
            if (i.has_starter) r += 300;
            return r;
        }

        function agentSortKey(i) {
            switch (agentFilter.sort) {
                case 'progress': return agentProgressRank(i);
                case 'steps':    return Number(i.steps || 0);
                case 'reward':   return Number(i.reward || 0);
                default:         return 0;
            }
        }

        function syncAgentFilterOptions(instances) {
            const fill = (id, values) => {
                const sel = document.getElementById(id);
                if (!sel) return;
                const have = new Set([...sel.options].map(o => o.value));
                [...values].sort().forEach(v => {
                    if (v && !have.has(v)) {
                        const o = document.createElement('option');
                        o.value = v; o.textContent = v;
                        sel.appendChild(o);
                    }
                });
            };
            fill('af-map', new Set(instances.map(i => String(i.room || '')).filter(Boolean)));
            fill('af-stage', new Set(instances.map(i => String(i.story_stage || '')).filter(Boolean)));
        }

        // Slightly closer fixed zoom; panning follows the discovered world.
        const FIXED_ZOOM = 0.5;
        const TILE_UNIT = 12;
        const MAP_MARGIN_UNITS = 4 * TILE_UNIT;
        // V17.3: auf dem Handy darf man ein kleines Stueck pinch-zoomen
        // (sonst wirkt die Karte auf dem kleinen Bildschirm zu winzig/grob),
        // aber nicht ins Unendliche - Desktop bleibt hart auf FIXED_ZOOM
        // gesperrt wie bisher.
        const IS_MOBILE_VIEWPORT = window.innerWidth <= 820;
        const MAP_MIN_ZOOM = FIXED_ZOOM;
        const MAP_MAX_ZOOM = IS_MOBILE_VIEWPORT ? FIXED_ZOOM + 1.5 : FIXED_ZOOM;

        const map = L.map('map-view', {
            crs: L.CRS.Simple,
            zoomSnap: 0.5,
            minZoom: MAP_MIN_ZOOM,
            maxZoom: MAP_MAX_ZOOM,
            zoomControl: false,
            scrollWheelZoom: false,
            doubleClickZoom: false,
            touchZoom: IS_MOBILE_VIEWPORT,
            boxZoom: false,
            keyboard: false,
            dragging: true,
            inertia: false,
            maxBoundsViscosity: 1.0,
            attributionControl: false
        });
        const bounds = [[0, 0], [3000, 3000]];
        // Start = Alabastia/Oaks Labor (der Savestate-Startpunkt jeder
        // Episode), nicht irgendein geratener Punkt - dort haelt sich die
        // Flotte zu Beginn eines Laufs am meisten auf.
        const START_CENTER = [680, 1500];
        map.setView(START_CENTER, FIXED_ZOOM);

        // Beim ersten Laden ist der Container u.U. noch 0px hoch (CSS/Layout
        // noch nicht fertig) -> Leaflet rendert dann eine 0x0-Karte. Nach dem
        // Layout neu vermessen. Wichtig fuer Handy.
        function _initMapSize() {
            map.invalidateSize();
        }
        window.addEventListener('load', () => {
            _initMapSize();
            setTimeout(_initMapSize, 250);
            setTimeout(_initMapSize, 800);
        });
        if (document.readyState === 'complete') setTimeout(_initMapSize, 100);

        // Optionales Kanto-Hintergrundbild. Fehlt es (kein legaler Map-Asset
        // vorhanden -> assets/maps/kanto_map.png), bleibt die Karte trotzdem
        // nutzbar: die abgelaufenen Tiles / Kanten werden ohnehin live
        // darueber gezeichnet ("Fog of War" aus echten Agenten-Daten).
        let overworldBackground = null;
        (function tryLoadBackground(){
            const probe = new Image();
            probe.onload = () => {
                overworldBackground = L.imageOverlay(
                    '/map.png', [[0, 0], [3000, 3000]],
                    { opacity: 0.55, interactive: false }
                ).addTo(map);
            };
            probe.onerror = () => {
                // Kein Bild -> dezentes Raster als Orientierung.
                const c = document.querySelector('#map-view');
                if (c) c.style.background =
                    'repeating-linear-gradient(0deg,#0c0e14,#0c0e14 39px,#12151f 40px),'
                  + 'repeating-linear-gradient(90deg,#0c0e14,#0c0e14 39px,#12151f 40px)';
            };
            probe.src = '/map.png?probe=' + Date.now();
        })();

        // V17.3: X/Y aus echten, haeufig bestaetigten Transitions in
        // /api/global_mapping berechnet (nicht geraten), damit Ausgang einer
        // Map und Eingang der naechsten wirklich aneinanderliegen statt
        // versetzt zu sein - z.B. Pallet-Nordausgang bei lokal x=12-13 traf
        // in 70 Beispielen zuverlaessig auf Route-1-Suedeingang bei
        // ebenfalls x=12-13, aber die alten Offsets hatten Route 1 um 90
        // Einheiten nach rechts verschoben. Vertania->Route 2 (8 Samples,
        // Vertania-Nordausgang x=19-22 trifft auf Route-2-Suedeingang
        // y=78-79) ergab denselben Fehler in Y: die alte feste Zahl (950)
        // lag viel zu nah an Vertania - Route 1 und 2 wirkten dadurch am
        // oberen Kartenrand zusammengequetscht. Wald/Marmoria haben noch
        // KEINE eigenen Transitions-Samples (die Flotte ist noch nicht so
        // weit) - deren Position ist eine grobe Schaetzung und wird
        // korrigiert, sobald echte Daten reinkommen.
        const MAP_OFFSETS = {
            '3,0': [1410, 2320],   // Pallet Town (outdoor, Referenzpunkt)
            '3,19': [1410, 1850],  // Route 1
            '3,1': [1272, 1388],   // Viridian City (outdoor)
            '3,20': [1410, 440],   // Route 2 (aus echten Transitions berechnet)
            '1,0': [1210, 300],    // Viridian Forest (Schaetzung, noch keine Daten)
            '3,2': [1410, 100],    // Pewter/Marmoria City (Schaetzung, noch keine Daten)

            // Innenraeume um Alabastia verteilt: Abstand ist NICHT mehr eine
            // gespiegelte Schaetzung, sondern aus /api/global_mapping direkt
            // gemessen. Alabastia (3,0) selbst deckt real x=2..21 ab, d.h.
            // seine echte Ost-Kante liegt bei 1410 + 22*TILE_UNIT = 1674
            // (dieselbe +1-Randkachel-Konvention wie bei getLeafletCoords(
            // maxX+1, maxY+1) oben). Reds Haus (4,0) deckt selbst x=1..12 ab
            // (eigene Breite ~13 Kacheln), Rivalenhaus (4,2) x=0..12 -
            // beide Gebaeude bekommen denselben, klar benannten
            // BUILDING_GAP von 4 Kacheln Luft zur echten Stadt-Kante, statt
            // nur am Anker-Punkt vorbeizuschrammen (das fuehrte vorher dazu,
            // dass das Rivalenhaus sichtbar in Alabastia hineinragte).
            '4,0': [1206, 2320],   // Alabastia - Reds Haus F1 (West, 4 Kacheln Abstand zur Stadt)
            '4,1': [1206, 2452],   // Alabastia - Reds Haus F2 (darunter, 1 Kachel Abstand)
            '4,2': [1722, 2320],   // Alabastia - Haus des Rivalen (Ost, 4 Kacheln Abstand zur Stadt)
            '4,3': [1722, 2452],   // Alabastia - Professor Eichs Labor (unter dem Rivalenhaus, 1 Kachel Abstand)

            // Vertania-Gebaeude in zwei Saeulen NEBEN die Stadt geklebt, wie
            // bei Alabastia. Vertania (3,1) ist real gross (x0..41) - die alten
            // Offsets (5,1/5,3/5,4/5,5 bei px 1272..1512) lagen MITTEN in der
            // gerenderten Stadt. Stadt-px-Spanne ~1272..1776; West-Saeule bei
            // px 1050 (Gebaeude ~13 Kacheln breit -> endet ~1206, 5 Kacheln
            // Luft), Ost-Saeule bei px 1830 (~4 Kacheln Luft zur Ost-Kante).
            // py bleibt im Stadt-Band 1388..1868.
            '5,0': [1050, 1388],   // Vertania City - Wohnhaus (West, oben)
            '5,2': [1050, 1524],   // Vertania City - Schule (West, mitte)
            '5,4': [1050, 1660],   // Vertania City - Center Erdgeschoss (West, unten)
            '5,5': [1050, 1784],   // Vertania City - Center Obergeschoss (West, unter dem EG)
            '5,1': [1830, 1388],   // Vertania City - Arena (Ost, oben)
            '5,3': [1830, 1524],   // Vertania City - Pokemon-Markt (Ost, mitte)
        };

        let agentMarkers = {};
        let battleFighterMarkers = {};
        let agentPolylines = {};
        let mapperMapLayers = {};
        let mapperMapSignatures = {};
        let mapperCoverageLayer = L.layerGroup().addTo(map);
        let mapperCoverageSignature = '';
        let persistentExplorationLayers = {};
        let persistentExplorationSignatures = {};
        let persistentEdgeLayers = {};
        let persistentTransitionLayers = {};
        let skeletonRects = {};
        let skeletonTransitionLines = [];

        let agentStepDots = {};
        let globalWarpLayer = L.layerGroup().addTo(map);
        let globalMappingSignature = "";

        // V17.3: ersetzt die Pfad-Linien (fielen nach recent_path-Cap von
        // 300 Punkten immer wieder weg) durch dauerhafte gruene Quadrate je
        // besuchter Kachel - waechst nur, wird nie entfernt. Dazu ein
        // beschrifteter Rahmen je entdeckter Map ("Pallet Town" etc.).
        let globalTileLayer = L.layerGroup().addTo(map);
        let globalTileDrawn = new Set();
        let mapNameLayers = {};
        let mapNameSignatures = {};

        function agentColor(id) {
            if (Number(id) === 120) return '#00e676';
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

        // Recompute from live discoveries, allowing only four tiles of empty
        // margin. Small worlds get a viewport-sized bound to stay centered.
        function recomputeDynamicMapBounds() {
            let b = null;
            const extend = (latlng) => {
                b = b ? b.extend(latlng) : L.latLngBounds(latlng, latlng);
            };
            const knownKey = (bank, mapId) => MAP_OFFSETS.hasOwnProperty(
                Number(bank) + ',' + Number(mapId)
            );

            (latestGlobalMapping.edges || []).forEach(e => {
                if (!Array.isArray(e) || e.length !== 6) return;
                if (!knownKey(e[0], e[1])) return;
                extend(getLeafletCoords(e[0], e[1], e[2], e[3]));
                extend(getLeafletCoords(e[0], e[1], e[4], e[5]));
            });
            (latestGlobalMapping.tiles || []).forEach(t => {
                if (!Array.isArray(t) || t.length !== 4) return;
                if (!knownKey(t[0], t[1])) return;
                extend(getLeafletCoords(t[0], t[1], t[2], t[3]));
            });
            (latestInstances || []).forEach(i => {
                if (!knownKey(i.bank, i.map)) return;
                extend(getLeafletCoords(i.bank, i.map, i.x, i.y));
            });

            if (!b || !b.isValid()) {
                // Noch keine Live-Daten da - Startbereich Alabastia/Route 1.
                extend(getLeafletCoords(3, 0, 0, 0));
                extend(getLeafletCoords(3, 19, 20, 40));
            }

            const size = map.getSize();
            if (!size.x || !size.y) return;
            const scale = map.options.crs.scale(FIXED_ZOOM);
            const center = b.getCenter();
            const halfWidth = Math.max(
                (b.getEast() - b.getWest()) / 2 + MAP_MARGIN_UNITS,
                size.x / scale / 2
            );
            const halfHeight = Math.max(
                (b.getNorth() - b.getSouth()) / 2 + MAP_MARGIN_UNITS,
                size.y / scale / 2
            );
            map.setMaxBounds([
                [center.lat - halfHeight, center.lng - halfWidth],
                [center.lat + halfHeight, center.lng + halfWidth]
            ]);
        }
        map.on('resize', recomputeDynamicMapBounds);

        // V17.3: dauerhafte gruene Quadrate statt Pfad-Linien (siehe oben) -
        // eine Kachel wird nie wieder entfernt, sobald sie einmal besucht
        // wurde. Zusaetzlich ein beschrifteter, gestrichelter Rahmen je
        // entdeckter Map ("Pallet Town" etc.) um die bisher bekannte
        // Ausdehnung dieser Map - "billig" aus denselben Tile-Daten
        // abgeleitet, keine echten Kartenbilder noetig.
        function updateGlobalTileCoverage() {
            const tiles = latestGlobalMapping.tiles || [];
            if (!tiles.length) return;

            const boxes = {};
            tiles.forEach(t => {
                if (!Array.isArray(t) || t.length !== 4) return;
                const key = `${Number(t[0])},${Number(t[1])}`;
                if (!MAP_OFFSETS[key]) return;
                const x = Number(t[2]), y = Number(t[3]);

                const tileKey = `${key}:${x}:${y}`;
                if (!globalTileDrawn.has(tileKey)) {
                    globalTileDrawn.add(tileKey);
                    const a = getLeafletCoords(t[0], t[1], x, y);
                    const b2 = getLeafletCoords(t[0], t[1], x + 1, y + 1);
                    L.rectangle([
                        [Math.min(a[0], b2[0]), Math.min(a[1], b2[1])],
                        [Math.max(a[0], b2[0]), Math.max(a[1], b2[1])]
                    ], {
                        stroke: false,
                        fill: true,
                        fillColor: '#00e676',
                        fillOpacity: 0.35,
                        interactive: false
                    }).addTo(globalTileLayer);
                }

                if (!boxes[key]) {
                    boxes[key] = {
                        bank: Number(t[0]), mapId: Number(t[1]),
                        minX: x, maxX: x, minY: y, maxY: y
                    };
                } else {
                    const b = boxes[key];
                    b.minX = Math.min(b.minX, x); b.maxX = Math.max(b.maxX, x);
                    b.minY = Math.min(b.minY, y); b.maxY = Math.max(b.maxY, y);
                }
            });

            Object.entries(boxes).forEach(([key, b]) => {
                const signature = `${b.minX}:${b.minY}:${b.maxX}:${b.maxY}`;
                if (mapNameSignatures[key] === signature) return;
                mapNameSignatures[key] = signature;

                if (mapNameLayers[key]) {
                    mapNameLayers[key].forEach(l => map.removeLayer(l));
                }

                const a = getLeafletCoords(b.bank, b.mapId, b.minX, b.minY);
                const c = getLeafletCoords(b.bank, b.mapId, b.maxX + 1, b.maxY + 1);
                const bounds = [
                    [Math.min(a[0], c[0]), Math.min(a[1], c[1])],
                    [Math.max(a[0], c[0]), Math.max(a[1], c[1])]
                ];
                const outline = L.rectangle(bounds, {
                    color: '#7f879b',
                    weight: 1,
                    fill: false,
                    dashArray: '4,4',
                    interactive: false
                }).addTo(map);

                const name = (typeof placeName === 'function')
                    ? placeName(b.bank, b.mapId)
                    : `Bank ${b.bank} / Map ${b.mapId}`;
                const labelMarker = L.marker([bounds[1][0], bounds[0][1]], {
                    icon: L.divIcon({
                        className: 'map-name-label',
                        html: name,
                        iconAnchor: [0, 0]
                    }),
                    interactive: false
                }).addTo(map);

                mapNameLayers[key] = [outline, labelMarker];
            });
        }

        async function updateMapperMapOverlays(force=false) {
            try {
                const response = await fetch('/api/mapper/maps?t=' + Date.now());
                const payload = await response.json();
                const seen = new Set();
                (payload.maps || []).forEach(item => {
                    const key = `${Number(item.bank)},${Number(item.map_id)}`;
                    // Eine unbekannte Map niemals mit dem Alabastia-Fallback
                    // an eine falsche Position malen.
                    if (!MAP_OFFSETS[key] || !item.alignment_confident) return;
                    seen.add(key);
                    const signature = String(item.revision || '');
                    if (!force && mapperMapSignatures[key] === signature) return;
                    if (mapperMapLayers[key]) map.removeLayer(mapperMapLayers[key]);

                    const a = getLeafletCoords(
                        item.bank, item.map_id, item.min_x, item.min_y
                    );
                    const b = getLeafletCoords(
                        item.bank, item.map_id,
                        Number(item.max_x) + 1, Number(item.max_y) + 1
                    );
                    const imageBounds = [
                        [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
                        [Math.max(a[0], b[0]), Math.max(a[1], b[1])]
                    ];
                    mapperMapLayers[key] = L.imageOverlay(
                        `${item.url}?rev=${encodeURIComponent(signature)}`,
                        imageBounds,
                        {opacity:1.0, interactive:false}
                    ).addTo(map);
                    mapperMapSignatures[key] = signature;
                });
                Object.keys(mapperMapLayers).forEach(key => {
                    if (!seen.has(key)) {
                        map.removeLayer(mapperMapLayers[key]);
                        delete mapperMapLayers[key];
                        delete mapperMapSignatures[key];
                    }
                });
            } catch (error) {
                console.error('Mapper map overlay failed', error);
            }
        }

        async function updateMapperTileCoverage(force=false) {
            try {
                const response = await fetch('/api/mapper/tiles?t=' + Date.now());
                const payload = await response.json();
                const tiles = Array.isArray(payload.tiles) ? payload.tiles : [];
                const signature = `${tiles.length}:` + (
                    tiles.length ? tiles[tiles.length - 1].join(',') : '-'
                );
                if (!force && signature === mapperCoverageSignature) return;
                mapperCoverageSignature = signature;
                mapperCoverageLayer.clearLayers();
                tiles.forEach(tile => {
                    const key = `${Number(tile[0])},${Number(tile[1])}`;
                    if (!MAP_OFFSETS[key]) return;
                    const a = getLeafletCoords(tile[0], tile[1], tile[2], tile[3]);
                    const b = getLeafletCoords(
                        tile[0], tile[1], Number(tile[2]) + 1, Number(tile[3]) + 1
                    );
                    L.rectangle([
                        [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
                        [Math.max(a[0], b[0]), Math.max(a[1], b[1])]
                    ], {
                        stroke:false,
                        fill:true,
                        fillColor:'#26c6da',
                        fillOpacity:0.28,
                        interactive:false
                    }).addTo(mapperCoverageLayer);
                });
            } catch (error) {
                console.error('Mapper tile coverage failed', error);
            }
        }

        let currentTab = 'map';

        function refreshWatcherStream() {
            // V18: nur noch der Watcher-Tab zeigt das Live-Bild.
            const img = document.getElementById('watcher-stream');
            if (!img) return;
            if (currentTab !== 'watcher' || document.hidden) {
                if (img.dataset.live) {
                    img.removeAttribute('src');
                    delete img.dataset.live;
                }
                return;
            }
            if (img.dataset.live) return;
            img.dataset.live = '1';
            img.onerror = () => {
                delete img.dataset.live;
                setTimeout(refreshWatcherStream, 1000);
            };
            img.src = '/watcher.mjpg';
        }

        // --- klickbare Agenten-Liste + Live-Stats (Watcher- UND Status-Tab) ---
        let wtSelected = null;
        function agentDisplayName(d) {
            const id = Number(d && d.id);
            if (id === 120) return 'Watcher';
            return 'Full ' + String(id + 1).padStart(2, '0');
        }
        function wtPick(id) {
            const opening = wtSelected !== id;
            wtSelected = opening ? id : null;
            renderWatcherTab();
            renderAgentDetail('status-agent-detail');
            document.querySelectorAll('.status-agent').forEach(el => {
                el.classList.toggle('sel', Number(el.dataset.aid) === wtSelected);
            });
            // Beim Oeffnen das Stats-Fenster ins Bild scrollen, statt dass es
            // unten am Ende der langen Liste unsichtbar bleibt.
            if (opening) {
                setTimeout(() => {
                    const boxId = currentTab === 'status'
                        ? 'status-agent-detail'
                        : currentTab === 'map'
                        ? 'map-agent-detail'
                        : 'wt-detail';
                    const box = document.getElementById(boxId);
                    if (box && !box.hidden) box.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 30);
            }
        }
        function agentChipsHtml(list) {
            return list.map(d => {
                const id = Number(d.id);
                const isW = id === 120;
                const role = isW ? 'watcher' : (d.training_objective || d.agent_role || '?');
                const rew = Number(d.reward || 0);
                const lvl = Number(d.level || 0);
                const stage = Number(d.world_stage || 0);
                const cls = 'wt-chip' + (wtSelected === id ? ' sel' : '') + (isW ? ' watcher' : '');
                const battle = Number(d.in_battle || 0)
                    ? ' <img class="battler-icon" src="/battler.png" alt="Battle">' : '';
                return '<div class="' + cls + '" onclick="wtPick(' + id + ')">'
                    + '<div class="c-id">' + (isW ? '👁 Watcher' : agentDisplayName(d)) + battle + '</div>'
                    + '<div class="c-role">' + role + '</div>'
                    + '<div class="c-meta">S' + stage + ' · L' + lvl + ' · <b style="color:' + (rew >= 0 ? '#5fe08a' : '#ff7a7a') + '">' + (rew >= 0 ? '+' : '') + rew.toFixed(0) + '</b></div>'
                    + '</div>';
            }).join('');
        }
        function renderWatcherTab() {
            const list = (latestInstances || []).slice().sort((a, b) => {
                if (Number(a.id) === 120) return -1;
                if (Number(b.id) === 120) return 1;
                return Number(a.id) - Number(b.id);
            });
            const html = agentChipsHtml(list);

            const grid = document.getElementById('wt-grid');
            if (grid) grid.innerHTML = html;
            const cnt = document.getElementById('wt-count');
            if (cnt) cnt.textContent = list.length + ' aktiv';
            renderAgentDetail('wt-detail');

            // V17.3: dieselben Kacheln nochmal fuer die mobile Kartenansicht -
            // eigene IDs, damit Handy (Karte) und Desktop (Watcher-Tab)
            // unabhaengig voneinander funktionieren.
            const mgrid = document.getElementById('map-agent-grid');
            if (mgrid) mgrid.innerHTML = html;
            const mcnt = document.getElementById('map-agent-count');
            if (mcnt) mcnt.textContent = list.length + ' aktiv';
            renderAgentDetail('map-agent-detail');
        }
        function renderAgentDetail(boxId) {
            const box = document.getElementById(boxId);
            if (!box) return;
            const d = (latestInstances || []).find(x => Number(x.id) === wtSelected);
            if (wtSelected === null || !d) { box.hidden = true; return; }
            box.hidden = false;
            const isW = wtSelected === 120;
            const sp = d.story_progress || {};
            const bs = d.battle_stats || {};
            const evs = Array.isArray(d.reward_events) ? d.reward_events : [];
            const evHtml = evs.slice(-16).reverse().map(e => {
                const s = (typeof e === 'string') ? e : (e && (e.type || e.name) ? (e.type || e.name) : JSON.stringify(e));
                const neg = /:-|(-[0-9])/.test(s);
                return '<div class="' + (neg ? 'neg' : 'pos') + '">' + s + '</div>';
            }).join('') || '<div>–</div>';
            const rew = Number(d.reward || 0);
            const starter = d.has_target_starter ? '✅ Schiggi' : (d.has_starter ? '≈ (falsch?)' : '—');
            const starterSpeciesId = Number(
                d.starter_species_id || (d.party && d.party[0] && d.party[0].species_id) || 0
            );
            const starterSpriteHtml = starterSpeciesId > 0
                ? '<img class="wt-starter-sprite" alt="Starter" src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-iii/firered-leafgreen/'
                    + starterSpeciesId + '.png" onerror="this.style.display=&quot;none&quot;">'
                : '';
            box.innerHTML =
                '<button class="wt-close" onclick="wtPick(' + wtSelected + ')">✕</button>'
                + starterSpriteHtml
                + '<h4>' + (isW ? '👁 Watcher' : agentDisplayName(d)) + '</h4>'
                + '<div class="wt-sub">' + (d.training_objective || d.agent_role || '?') + ' · ' + (d.room || ('Bank ' + d.bank + ' / Map ' + d.map)) + ' @ ' + d.x + ',' + d.y + ' · Start: ' + (d.episode_start || '?') + '</div>'
                + '<div class="wt-dgrid">'
                + cell(rew.toFixed(1), 'Episode-Reward', rew >= 0 ? '#5fe08a' : '#ff7a7a')
                + cell(Number(d.steps || 0).toLocaleString(), 'Steps')
                + cell(Number(d.world_stage || 0), 'Welt-Stufe')
                + cell(Number(d.level || 0), 'Level')
                + cell(Number(d.badges || 0), 'Orden')
                + cell(Number(d.episode_steps_remaining || 0).toLocaleString(), 'Steps bis Reset')
                + cell((sp.pallet_oaks_lab_scene || 0) + '/6', 'Eich-Szene')
                + cell(Number(bs.started || 0) + '/' + Number(bs.completed || 0), 'Kämpfe s/f')
                + cell(String(d.input || d.effective_action || '–'), 'Taste')
                + '</div>'
                + (d.last_stage_timeout ? '<div class="wt-sub">⏱ letzter Abbruch: <b>' + d.last_stage_timeout + '</b></div>' : '')
                + '<div class="wt-events">' + evHtml + '</div>';
            function cell(v, k, color) {
                return '<div class="wt-cell"><div class="v"' + (color ? ' style="color:' + color + '"' : '') + '>' + v + '</div><div class="k">' + k + '</div></div>';
            }
        }
        async function refreshMapperStream() {
            const stamp = Date.now();
            const live = document.getElementById('mapper-stream');
            const atlas = document.getElementById('mapper-atlas');
            if (live) live.src = '/mapper.jpg?ts=' + stamp;
            if (atlas) atlas.src = '/mapper-atlas.png?ts=' + stamp;
            try {
                const response = await fetch('/api/mapper?ts=' + stamp);
                const s = await response.json();
                const label = document.getElementById('mapper-status');
                if (label) label.textContent = s.running
                    ? `${s.map || 'Map unbekannt'} · ${Number(s.new_positions || 0).toLocaleString()} neue Felder · ${Number(s.new_visual_tiles || 0).toLocaleString()} Bild-Tiles · ${s.last_event || ''}`
                    : 'Mapper ist angehalten oder startet gerade.';
            } catch (_) {}
        }
        setInterval(() => {
            refreshWatcherStream();
            if (currentTab === 'mapper') refreshMapperStream();
        }, 500);

        function showTab(t, e) {
            currentTab = t;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (e && e.target) e.target.classList.add('active');

            document.getElementById('map-view').style.display = t === 'map' ? 'block' : 'none';
            document.getElementById('graphs-view').style.display = t === 'graphs' ? 'block' : 'none';
            document.getElementById('status-view').style.display = t === 'status' ? 'block' : 'none';
            document.getElementById('watcher-view').style.display = t === 'watcher' ? 'block' : 'none';

            const showOverlays = t === 'map';
            const workspace = document.getElementById('map-workspace');
            if (workspace) workspace.style.display = showOverlays ? 'grid' : 'none';
            document.getElementById('hud').style.display = showOverlays ? 'block' : 'none';
            document.getElementById('detail-panel').style.display = showOverlays ? 'block' : 'none';
            const afb = document.getElementById('agent-filter-bar');
            if (afb) afb.style.display = showOverlays ? 'flex' : 'none';
            // Mobile: .live-global/.v81skills werden aus #map-view raus- und als
            // feste Kacheln in #main-container gehaengt (pkmai-mobile-relayout).
            // Dadurch haengen sie an KEINER Tab-View mehr und ueberlagerten bisher
            // jeden anderen Tab (das alte schwebende Panel am Handy).
            // Inline !important noetig, um die CSS-Regel .pkmai-mobile-tile{display:
            // block!important} zu schlagen.
            document.querySelectorAll('.pkmai-mobile-tile').forEach(el => {
                el.style.setProperty('display', showOverlays ? 'block' : 'none', 'important');
            });
            // Live-Brain- und Champion-Karten gehoeren nur auf die Map. Auf der
            // Graphs-Seite stehen dieselben Zahlen jetzt in der KPI-Zeile.
            const brainRow = document.getElementById('brain-summary-row');
            if (brainRow) brainRow.style.display = showOverlays ? 'grid' : 'none';

            if (t === 'graphs') loadTrainingGraphs();
            if (t === 'watcher') { refreshWatcherStream(); renderWatcherTab(); }
            if (t === 'mapper') refreshMapperStream();
            if (t === 'map') {
                updateGlobalMapping(true);
                updateMapperMapOverlays(true);
                updateMapperTileCoverage(true);
                renderWatcherTab();
                setTimeout(() => map.invalidateSize(), 50);
                setTimeout(() => map.invalidateSize(), 350);
            }
        }

        // Handy: nach Dreh / Resize die Leaflet-Karte neu vermessen, sonst
        // bleibt sie grau oder laesst sich nicht mehr zoomen.
        let _mapResizeT;
        function _remeasureMap() {
            clearTimeout(_mapResizeT);
            _mapResizeT = setTimeout(() => {
                if (currentTab === 'map') map.invalidateSize();
            }, 200);
        }
        window.addEventListener('resize', _remeasureMap);
        window.addEventListener('orientationchange', _remeasureMap);

        // V17.3: die gelb gestrichelten "Skeleton"-Kartenrahmen + Verbindungs-
        // linien liefen unabhaengig von den neuen gruenen Tile-Quadraten und
        // Namensrahmen und sorgten fuer verwirrende doppelte/ueberlagerte
        // Umrandungen. Auf Wunsch abgeschaltet - bestehende Layer einmalig
        // entfernt, keine Interval-Aktualisierung mehr.
        function updateSkeleton() {
            Object.keys(skeletonRects).forEach(key => {
                map.removeLayer(skeletonRects[key]);
                delete skeletonRects[key];
            });
            skeletonTransitionLines.forEach(line => map.removeLayer(line));
            skeletonTransitionLines = [];
        }
        updateSkeleton();

        function getSelectedInstance() {
            if (selectedAgentId !== null) {
                return latestInstances.find(i => Number(i.id) === Number(selectedAgentId))
                    || null;
            }
            // Ohne Kartenfokus zeigt das Detailpanel weiterhin den Watcher.
            return latestInstances.find(i => Number(i.id) === 120)
                || latestInstances[0]
                || null;
        }

        function selectAgent(id) {
            const n = Number(id);
            selectedAgentId = (selectedAgentId === n) ? null : n;
            // Sofortiges, leichtes Feedback aus dem Cache. Die schweren
            // Refreshs (Karten-Marker, HUD-Liste, Exploration-Layer) macht der
            // 1s-Intervall gleich von selbst - kein synchroner
            // updateDashboard() + force-Redraw mehr, der beim Klick die halbe
            // Seite neu aufbaut und die Stats kurz leer blinken laesst.
            renderSelectedAgent();
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

            const isW = Number(inst.id) === 120;

            // Team, Orden und Name gehoeren zum ausgewaehlten Client. Daten
            // des Watchers duerfen nicht bei einem Trainingsagenten erscheinen.
            const headInst = inst;
            document.getElementById('hud-trainer').innerText =
                (isW ? (headInst.name || 'Alex').replace(' (Watcher)', '') : agentDisplayName(headInst)) + ' (Live)';

            updateParty(headInst.party || []);

            const badges = Number(headInst.badges || 0);
            for (let i = 1; i <= 8; i++) {
                const el = document.getElementById(`badge-${i}`);
                if (i <= badges) el.classList.add('active');
                else el.classList.remove('active');
            }

            document.getElementById('detail-name').innerText = agentDisplayName(inst);
            document.getElementById('detail-room').innerText = inst.room || `Bank ${inst.bank} / Map ${inst.map}`;
            document.getElementById('detail-steps').innerText = Number(inst.steps || 0).toLocaleString();
            document.getElementById('detail-reward').innerText = Number(inst.reward || 0).toFixed(2);
            const pe = inst.persistent_exploration || {};
            document.getElementById('detail-stage').innerText = Number(inst.world_stage || 0);
            document.getElementById('detail-position').innerText = `${Number(inst.x||0)},${Number(inst.y||0)}`;
            document.getElementById('detail-tiles').innerText = Number(inst.explored_tiles || 0).toLocaleString();
            document.getElementById('detail-runsteps').innerText = Number(inst.ppo_episode_steps || inst.steps || 0).toLocaleString();
            document.getElementById('detail-remaining').innerText = Number(
                inst.episode_steps_remaining || 0
            ).toLocaleString('de-DE');
            document.getElementById('detail-inbattle').innerText = inst.in_battle ? 'JA' : 'NEIN';
            document.getElementById('detail-badges').innerText = `${Number(inst.badges||0)} / 8`;

            const navState = pe.exit_seek_active
                ? `EXIT SEEK (${Number(pe.steps_since_new_edge || 0)} stale)`
                : `EXPLORE (${Number(pe.steps_since_new_edge || 0)} stale)`;

            const events = Array.isArray(inst.reward_events) ? inst.reward_events : [];
            document.getElementById('detail-event').innerText =
                `${navState} | ${events.length ? events[events.length - 1] : '-'}`;

            // V18: die letzten 10 Reward-Events des angeklickten Agenten -
            // schneller Blick, was der gerade tatsaechlich macht.
            // 2026-09-07 (user): zusaetzlich der hoechste positive und der
            // tiefste negative Einzel-Reward aus dem Fenster (letzte ~40),
            // damit man die grossen Ausschlaege sieht, auch wenn sie schon
            // aus den letzten 10 herausgescrollt sind.
            const recentBox = document.getElementById('detail-recent-rewards');
            if (recentBox) {
                const parseEv = (raw) => {
                    const s = String(raw);
                    const m = s.match(/([+-]\d+(?:\.\d+)?)\s*$/);
                    if (!m) return null;
                    const val = parseFloat(m[1]);
                    if (!isFinite(val)) return null;
                    let body = s.slice(0, m.index).replace(/:$/, '');
                    let step = '';
                    const sm = body.match(/^(\d+):(.*)$/);
                    if (sm) { step = sm[1]; body = sm[2]; }
                    return { val, label: body, step };
                };
                const fmtV = (v) => (v > 0 ? '+' : '') +
                    (Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3));
                let hi = null, lo = null;
                for (const raw of events) {
                    const p = parseEv(raw);
                    if (!p) continue;
                    if (p.val > 0 && (!hi || p.val > hi.val)) hi = p;
                    if (p.val < 0 && (!lo || p.val < lo.val)) lo = p;
                }
                let head = '';
                if (hi) head += '<div class="pos" style="border-bottom:1px dashed #232738">⬆ ' +
                    fmtV(hi.val) + '  ' + hi.label + (hi.step ? ' · s' + hi.step : '') + '</div>';
                if (lo) head += '<div class="neg" style="border-bottom:1px dashed #232738">⬇ ' +
                    fmtV(lo.val) + '  ' + lo.label + (lo.step ? ' · s' + lo.step : '') + '</div>';
                const list = events.slice(-10).reverse().map(e => {
                    const s = (typeof e === 'string') ? e
                        : (e && (e.type || e.name) ? (e.type || e.name) : JSON.stringify(e));
                    const neg = /:-|-[0-9]/.test(s);
                    return '<div class="' + (neg ? 'neg' : 'pos') + '">' + s + '</div>';
                }).join('');
                recentBox.innerHTML = (head + list) || '<div>-</div>';
            }

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

        const FLEET_DEPTH_NAMES = {
            0:'Spielanfang / Alabastia-Innen', 1:'Alabastia (außen)', 2:'Route 1',
            3:'Vertania City', 4:'Route 2', 5:'Vertania-Wald', 6:'Marmoria City'
        };
        const FLEET_ROLE_LABELS = {
            intro:'Intro', stairs:'Treppe', exit:'Haus-Exit', starter:'Starter',
            starter_rush:'Starter', battle:'Kampf', level:'Level', progress:'Progress',
            full:'FULL Navigation', badge:'Orden', scout:'Frontier Scout'
        };
        const STATUS_ROLE_ICONS = {
            intro:'🎬', stairs:'🪜', exit:'🚪', starter:'🐣', battle:'⚔️',
            level:'⬆️', progress:'🧭', full:'🏁', badge:'🪨', scout:'🔭'
        };
        const NAV_KEY_NAMES = ['A','B','START','↑','↓','←','→'];
        function statusEsc(value){
            return String(value ?? '–').replace(/[&<>"']/g, c=>({
                '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
            })[c]);
        }
        function renderStatusDashboard(state){
            const instances=(state.instances||[]).slice();
            const watcher=instances.find(i=>Number(i.id)===120);
            // Kein festes Envs-Limit mehr - NUM_ENVS aendert sich (32/50/...),
            // nur die Watcher-ID 120 ist reserviert und wird ausgeschlossen.
            const runners=instances.filter(i=>Number(i.id)>=0&&Number(i.id)!==120)
                .sort((a,b)=>Number(a.id)-Number(b.id));
            const rt=((state.training_stats||{}).run_totals)||{};
            const battle=state.battle_stats||{};
            const summary=document.getElementById('status-summary');
            if(summary) summary.innerHTML=[
                ['🧠',Number(state.training_timesteps||0).toLocaleString('de-DE'),'Learner Steps'],
                ['🏆','v'+String(state.version||0),'Champion'],
                ['🗺️',Number(state.world_depth||0),FLEET_DEPTH_NAMES[Number(state.world_depth||0)]||'Weltstufe'],
                ['⚔️',Number(battle.started||0).toLocaleString('de-DE'),'Kämpfe gesamt'],
                ['💥',Number(rt.enemy_faints||0).toLocaleString('de-DE'),'Gegner-K.O.'],
                ['☠️',Number(rt.party_wipes||0).toLocaleString('de-DE'),'Eigene K.O. (Party wiped)'],
                ['🩸',Number(rt.enemy_damage_hp||0).toLocaleString('de-DE'),'Schaden-HP'],
                ['🏁',Number((state.champion_speed||{}).best_stage_steps||0)>0
                    ? Number(state.champion_speed.best_stage_steps).toLocaleString('de-DE')
                    : '–','Champion: beste Full-Steps']
            ].map(x=>`<div class="status-kpi"><div class="big">${x[0]} ${x[1]}</div><div class="small">${x[2]}</div></div>`).join('');

            const watcherEl=document.getElementById('status-watcher');
            if(watcherEl){
                const wb=(watcher&&watcher.battle_stats)||{};
                const we=(watcher&&watcher.exp_stats)||{};
                watcherEl.innerHTML=watcher ? `
                    <h3>👁️ Watcher separat</h3>
                    <div class="status-watcher-grid">
                      <div class="status-pill">🧠 Modell <b>${statusEsc(watcher.loaded_model)}</b></div>
                      <div class="status-pill">📈 Learner <b>${Number(watcher.learner_steps||0).toLocaleString('de-DE')}</b> Steps</div>
                      <div class="status-pill">🔄 Netz nachgeladen <b>${Number(watcher.brain_reloads||0)}×</b></div>
                      <div class="status-pill">🏆 Champion <b>v${String(watcher.model_version||0)}</b></div>
                      <div class="status-pill">👣 Lauf-Steps <b>${Number(watcher.steps||0).toLocaleString('de-DE')}</b></div>
                      <div class="status-pill">📍 Ort <b>${statusEsc(watcher.bank)}/${statusEsc(watcher.map)} @ ${statusEsc(watcher.x)},${statusEsc(watcher.y)}</b></div>
                      <div class="status-pill">⭐ Level <b>${Number(watcher.level||0)}</b></div>
                      <div class="status-pill">⚔️ Kämpfe <b>${Number(wb.started||0)} / ${Number(wb.completed||0)}</b></div>
                      <div class="status-pill">✨ EP <b>+${Number(we.gained_total||0)}</b></div>
                      <div class="status-pill">🎁 Reward <b>${Number(watcher.reward||0).toFixed(1)}</b></div>
                    </div>` : '<h3>👁️ Watcher separat</h3><div class="status-pill">Noch keine Telemetrie</div>';
            }

            const groups={};
            runners.forEach(i=>{
                const role=String(i.training_objective||i.agent_role||'?');
                const g=groups[role]||(groups[role]={n:0,active:0,started:0,done:0,ko:0,damage:0,level:0,stage:0,steps:0});
                const bs=i.battle_stats||{};
                g.n++; g.active+=Number(!!i.in_battle);
                g.started+=Number(bs.episode_started||0); g.done+=Number(bs.episode_completed||0);
                g.ko+=Number(bs.enemy_faints||0); g.damage+=Number(bs.enemy_damage_hp||0);
                g.level=Math.max(g.level,Number(i.level||0)); g.stage=Math.max(g.stage,Number(i.world_stage||0));
                g.steps+=Number(i.steps||0);
            });
            const runKeys={
                intro:['v2_intro_episodes','v2_intro_success'],stairs:['v2_stairs_episodes','v2_stairs_success'],
                exit:['v2_exit_episodes','v2_exit_success'],starter:['v8_starter_episodes','v8_starter_success'],
                battle:['v8_battle_episodes','v8_battle_success'],level:['v8_level_episodes','v8_level_success'],
                badge:['v8_badge_episodes','v8_badge_success'],progress:['v7_progress_episodes','v7_progress_badge1'],
                full:['v2_full_episodes','v7_full_badge1']
            };
            const order=['full','progress','battle','level','badge','starter','exit','stairs','intro'];
            const roleGrid=document.getElementById('status-role-grid');
            if(roleGrid) roleGrid.innerHTML=Object.keys(groups).sort((a,b)=>order.indexOf(a)-order.indexOf(b)).map(role=>{
                const g=groups[role], keys=runKeys[role]||[];
                return `<div class="status-role"><div class="status-role-head"><span>${STATUS_ROLE_ICONS[role]||'🤖'} ${statusEsc(FLEET_ROLE_LABELS[role]||role)}</span><b>${g.n} Agenten</b></div>
                  <div class="status-role-stats">
                    <span><strong>${g.active}</strong>im Kampf</span><span><strong>${g.started}/${g.done}</strong>Kämpfe</span>
                    <span><strong>${g.ko}</strong>K.O.</span><span><strong>${g.damage}</strong>Schaden</span>
                    <span><strong>${g.level}</strong>max Level</span><span><strong>${g.stage}</strong>Welt</span>
                    <span><strong>${g.n?Math.round(g.steps/g.n).toLocaleString('de-DE'):0}</strong>Ø Ep-Steps</span>
                    <span><strong>${Number(rt[keys[0]]||0)}</strong>Runs</span><span><strong>${Number(rt[keys[1]]||0)}</strong>Erfolg</span>
                  </div></div>`;
            }).join('');

            const agentGrid=document.getElementById('status-agent-grid');
            if(agentGrid) agentGrid.innerHTML=runners.map(i=>{
                const role=String(i.training_objective||i.agent_role||'?'), bs=i.battle_stats||{};
                const rew=Number(i.reward||0);
                const sel=Number(i.id)===wtSelected?' sel':'';
                return `<div class="status-agent ${i.in_battle?'fighting':''}${sel}" data-aid="${Number(i.id)}" onclick="wtPick(${Number(i.id)})">
                  <div class="status-agent-top"><span>${STATUS_ROLE_ICONS[role]||'🤖'} ${agentDisplayName(i)}</span><span style="color:${rew>=0?'#5fe08a':'#ff7a7a'}">${rew>=0?'+':''}${rew.toFixed(0)}</span></div>
                  <div class="status-agent-meta">⌨️ ${NAV_KEY_NAMES[Number(i.effective_action)]||'–'} · 👣 ${Number(i.steps||0).toLocaleString('de-DE')} Ep-Steps · 🎬 ${statusEsc(i.story_stage)} · 🐣 ${i.has_target_starter?'Schiggi':(i.has_starter?'falsch?':'nein')}<br>
                  📍 ${statusEsc(i.bank)}/${statusEsc(i.map)} @ ${statusEsc(i.x)},${statusEsc(i.y)} · 🌍 Welt ${Number(i.world_stage||0)} · ⭐ Lv ${Number(i.level||0)}<br>
                  🥊 ${Number(bs.episode_started||0)}/${Number(bs.episode_completed||0)} · 💥 ${Number(bs.enemy_faints||0)} · 🩸 ${Number(bs.enemy_damage_hp||0)} · 💾 ${statusEsc(i.episode_start)}</div>
                </div>`;
            }).join('');
            renderAgentDetail('status-agent-detail');
            if (typeof latestBrains !== 'undefined' && latestBrains.navigation) {
                renderBrainArchitecture(latestBrains);
            }
        }

        // --- Brain Progress: wird das Netz wirklich besser? Fest in die
        // Status-Kacheln eingebaut (kein schwebendes Extra-Fenster). Zeigt
        // Champion (letzte bestaetigte Bestmarke) + Live-Brain jetzt vs. vor
        // einer Weile, farbig nach Richtung. /api/history ist klein und
        // aendert sich nur alle 25k Steps - eigenes, selten laufendes Fetch.
        let _bpHistCache = null, _bpHistFetchedAt = 0;
        async function renderBrainProgress(state) {
            const box = document.getElementById('brain-progress');
            if (!box) return;
            const now = Date.now();
            if (!_bpHistCache || now - _bpHistFetchedAt > 5000) {
                try {
                    const r = await fetch('/api/history?t=' + now);
                    _bpHistCache = (await r.json()).history || [];
                    _bpHistFetchedAt = now;
                } catch (e) { /* alte Daten weiterverwenden */ }
            }
            const hist = _bpHistCache || [];
            if (!hist.length) { box.innerHTML = ''; return; }
            const nowPoint = hist[hist.length - 1];
            // Letzter Learner-Reset = letzter Rueckwaertssprung der Steps.
            // Davor liegt ein anderer Trainingslauf - fuer "wird's besser"
            // zaehlt nur die Zeit seit dem aktuellen Lauf.
            let baseIdx = 0;
            for (let i = hist.length - 1; i > 0; i--) {
                if (Number(hist[i].timesteps) < Number(hist[i - 1].timesteps)) { baseIdx = i; break; }
            }
            // Innerhalb des aktuellen Laufs: ein Punkt von vor einer Weile
            // (bis zu 40 Eintraege = ~1 Mio Steps zurueck), nie vor dem Reset.
            const cmpIdx = Math.max(baseIdx, hist.length - 1 - 40);
            const oldPoint = hist[cmpIdx];
            const tile = (icon, big, small, deltaTxt, dir) => {
                const cls = dir > 0 ? ' bp-up' : (dir < 0 ? ' bp-down' : '');
                const deltaHtml = deltaTxt
                    ? '<div class="bp-delta" style="color:' + (dir > 0 ? '#5fe08a' : dir < 0 ? '#ff7a7a' : '#8b93a7') + '">' + deltaTxt + '</div>'
                    : '';
                return '<div class="status-kpi' + cls + '"><div class="big">' + icon + ' ' + big + '</div><div class="small">' + small + '</div>' + deltaHtml + '</div>';
            };
            const bestNow = Number(nowPoint.best_episode_reward || 0);
            const bestOld = Number((oldPoint || nowPoint).best_episode_reward || 0);
            const bestDelta = bestNow - bestOld;
            const avgNow = Number(nowPoint.avg_episode_reward || 0);
            const avgOld = Number((oldPoint || nowPoint).avg_episode_reward || 0);
            const avgDelta = avgNow - avgOld;
            const lvlNow = Number(nowPoint.max_level || 0);
            const lvlOld = Number((oldPoint || nowPoint).max_level || 0);
            const champVer = 'v' + String(state.version || 0);
            const champSteps = Number((state.champion_speed || {}).steps || 0);
            box.innerHTML =
                tile('🏆', champVer, 'Champion (bestätigt, Steps ' + champSteps.toLocaleString('de-DE') + ')', '', 0)
                + tile(
                    '🧠', bestNow.toFixed(0), 'Bester Live-Reward jetzt',
                    (bestDelta >= 0 ? '+' : '') + bestDelta.toFixed(0) + ' seit vorhin', Math.sign(bestDelta)
                )
                + tile(
                    '📊', avgNow.toFixed(0), 'Ø Live-Reward jetzt',
                    (avgDelta >= 0 ? '+' : '') + avgDelta.toFixed(0) + ' seit vorhin', Math.sign(avgDelta)
                )
                + tile(
                    '⭐', lvlNow, 'Höchstes Level jetzt',
                    lvlNow === lvlOld ? '' : ((lvlNow > lvlOld ? '+' : '') + (lvlNow - lvlOld) + ' seit vorhin'),
                    Math.sign(lvlNow - lvlOld)
                );

            // Letzte 2-3 Brain-Versionen im Vergleich: pro Version der letzte
            // (reifste) beobachtete beste Reward-Wert, chronologisch.
            const byVersion = [];
            for (const p of hist) {
                const v = Number(p.version || 0);
                if (byVersion.length && byVersion[byVersion.length - 1].v === v) {
                    byVersion[byVersion.length - 1].p = p;
                } else {
                    byVersion.push({ v, p });
                }
            }
            const lastVersions = byVersion.slice(-3);
            const vEl = document.getElementById('brain-progress-versions');
            if (vEl && lastVersions.length) {
                vEl.innerHTML = lastVersions.map((entry, i) => {
                    const r = Number(entry.p.best_episode_reward || 0);
                    const prev = i > 0 ? Number(lastVersions[i - 1].p.best_episode_reward || 0) : null;
                    const arrow = prev === null ? '' : (r > prev ? ' <span style="color:#5fe08a">▲</span>' : r < prev ? ' <span style="color:#ff7a7a">▼</span>' : ' <span style="color:#8b93a7">▬</span>');
                    return '<div class="bp-ver-chip"><b>v' + String(entry.v) + '</b><span>' + r.toFixed(0) + ' Reward' + arrow + '</span></div>';
                }).join('<span class="bp-ver-sep">→</span>');
            }
        }
        function updateFleetPanel(state){
            const f = state.fleet || {};
            const loc = f.location || {};
            const wd = Number(state.world_depth || 0);
            const cp = Number(state.deepest_outdoor_checkpoint || 0);
            const setTxt = (id,v)=>{ const e=document.getElementById(id); if(e) e.innerText=v; };
            setTxt('fleet-depth-num', wd);
            setTxt('fleet-depth-name', FLEET_DEPTH_NAMES[wd] || ('Tiefe '+wd));
            setTxt('fleet-cp', 'stage_'+cp);
            setTxt('fleet-full', Number((state.instances||[]).filter(i=>Number(i.id)!==120).length));
            setTxt('fleet-outdoor', loc.outdoor||0);
            setTxt('fleet-brk', f.depth_breakthroughs||0);
            setTxt('fleet-tiles', Number((state.training_stats||{}).max_explored_tiles||0));
            setTxt('fleet-fighters', Number((latestBrains.battle||{}).workers||0));
            setTxt('fleet-scenarios', Number((latestBrains.battle||{}).scenario_count||0));
            const on=document.getElementById('fleet-outdoor');
            if(on) on.style.color = '#00e676';
            const track = document.getElementById('fleet-depth-track');
            if(track){
                let h='';
                // Geographic stages; badges are tracked separately.
                for(let i=1;i<=6;i++){
                    const cls = i<wd ? 'on' : (i===wd ? 'cur' : '');
                    h += `<i class="${cls}" title="${FLEET_DEPTH_NAMES[i]||''}"></i>`;
                }
                track.innerHTML = h;
            }
            const rolesEl = document.getElementById('fleet-roles');
            if(rolesEl){
                const roles = f.roles || {};
                const order = ['scout','progress','battle','level','full','starter','exit','stairs','intro'];
                const keys = Object.keys(roles).sort((a,b)=>{
                    const ia=order.indexOf(a), ib=order.indexOf(b);
                    return (ia<0?99:ia)-(ib<0?99:ib);
                });
                rolesEl.innerHTML = keys.map(k=>
                    `<span>${FLEET_ROLE_LABELS[k]||k} <b>${roles[k]}</b></span>`
                ).join('') || '<span>–</span>';
            }
            const evEl = document.getElementById('fleet-events');
            if(evEl){
                const evs = f.depth_events || [];
                evEl.innerHTML = evs.length
                    ? evs.slice().reverse().map(e=>`<div>Agent ${e.id}: ${e.ev}</div>`).join('')
                    : '<div class="none">noch kein world_depth-Event in diesem Zyklus</div>';
            }
        }
        function updateJourneySkills(state, brains){
            const setTxt=(id,v)=>{ const e=document.getElementById(id); if(e) e.innerText=v; };
            const metrics=((brains||{}).navigation||{}).metrics||{};
            const maps=Number(metrics.max_maps||0);
            const stage=Number(metrics.max_stage||state.world_depth||0);
            const badges=Number(metrics.max_badges||state.max_badges||0);
            setTxt('jv-maps', maps);
            setTxt('jv-stage', stage);
            setTxt('jv-badges', `${badges}/8`);
            const badgeFill=document.getElementById('jf-badges');
            if(badgeFill) badgeFill.style.width=`${Math.min(100,100*badges/8)}%`;
        }

        async function loadTrainingGraphs() {
            // V17.3: /api/state und /api/history liefen als EIN Promise.all -
            // ein einzelner langsamer/fehlgeschlagener History-Abruf (z.B.
            // direkt nach einem Webserver-Neustart) verhinderte dann auch das
            // sofortige Rendern der Fleet-/Weltstufen-Karte, obwohl deren
            // Daten laengst da waren. Jetzt unabhaengig, damit ein Ausfall
            // beim einen den anderen nicht mit runterreisst.
            let state = {};
            try {
                const sr = await fetch('/api/state?t='+Date.now());
                state = await sr.json();
            } catch (e) {
                console.error('State load failed', e);
            }
            let hist = [];
            let brains = {};
            try {
                const hr = await fetch('/api/history?t='+Date.now());
                const histPayload = await hr.json();
                hist = histPayload.history || [];
            } catch (e) {
                console.error('History load failed', e);
            }
            try {
                const br = await fetch('/api/brains?t='+Date.now());
                brains = await br.json();
            } catch (e) {
                console.error('Brain status load failed', e);
            }
            try {
                updateJourneySkills(state, brains);
                const st=state.training_stats||{};
                const rates=st.beginning_success_rates||{};
                const skillRates=st.v6_skill_rates||{};

                const nav=brains.navigation||{}, bat=brains.battle||{};
                document.getElementById('g-steps').innerText=Number(nav.learner_steps||state.training_timesteps||0).toLocaleString('de-DE');
                document.getElementById('g-version').innerText=`v${String(nav.champion_version||state.version||0)}`;
                document.getElementById('g-champ-delta').innerText='+'+Number(nav.since_champion||0).toLocaleString('de-DE');
                document.getElementById('g-episodes').innerText=Number(nav.completed_runs||0).toLocaleString('de-DE');
                document.getElementById('g-battle-steps').innerText=Number(bat.learner_steps||0).toLocaleString('de-DE');
                document.getElementById('g-battle-updates').innerText=Number(bat.ppo_updates||0).toLocaleString('de-DE');
                document.getElementById('g-battle-episodes').innerText=Number(bat.episodes||0).toLocaleString('de-DE');
                document.getElementById('g-battle-kos').innerText=Number(bat.kos||0).toLocaleString('de-DE');
                const wd=Number(state.world_depth||0);
                const gwd=document.getElementById('g-world-depth');
                if(gwd){gwd.innerText=wd; gwd.style.color = wd>=3 ? '#00e676' : (wd>=2 ? '#ffea00' : '#ff8a65');}
                try { updateFleetPanel(state); } catch(e) {}

                if (!hist.length) return;
                // Never connect incompatible trainer/reset runs. Episode
                // counters are process-local; learner timesteps may reset too.
                let runStart=0;
                for(let i=1;i<hist.length;i++){
                    const prev=hist[i-1], cur=hist[i];
                    const explicitRunChange = Number(prev.training_run_id||0)>0 && Number(cur.training_run_id||0)>0 && Number(prev.training_run_id)!==Number(cur.training_run_id);
                    const stepReset = Number(cur.timesteps||0)<Number(prev.timesteps||0);
                    const episodeReset = Number(cur.episodes||0)<Number(prev.episodes||0);
                    if(explicitRunChange || stepReset || episodeReset) runStart=i;
                }
                const rewardHist=hist.slice(runStart).filter(p=>Number(p.episodes||0)>0);
                const labels=rewardHist.map(p=>Number(p.timesteps||0).toLocaleString());

                upsertTrainingChart('graph-reward',labels,[
                    {label:'Ø Episode Reward',data:rewardHist.map(p=>Number(p.avg_episode_reward||0)),borderWidth:2,pointRadius:1.5,tension:.22}
                ]);

                const cleanHist=hist.filter(p=>Number(p.stats_schema||0)>=4 && p.movement_graph_edges!=null);
                const cleanLabels=cleanHist.map(p=>Number(p.timesteps||0).toLocaleString());

                upsertTrainingChart('graph-success',cleanLabels,[
                    {label:'Bestätigte Kanten',data:cleanHist.map(p=>Number(p.movement_graph_edges||0)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Graph-Maps',data:cleanHist.map(p=>Number(p.movement_graph_maps||0)),borderWidth:2,pointRadius:1,tension:.2}
                ]);

                const runningMax=(arr)=>{let m=0;return arr.map(v=>{m=Math.max(m,Number(v||0));return m;});};
                upsertTrainingChart('graph-progress',labels,[
                    {label:'Stage',data:runningMax(rewardHist.map(p=>p.max_stage)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Orden',data:runningMax(rewardHist.map(p=>p.max_badges)),borderWidth:2,pointRadius:1,tension:.2},
                    {label:'Maps',data:runningMax(rewardHist.map(p=>p.max_maps)),borderWidth:2,pointRadius:1,tension:.2}
                ]);
                window.__battleGraphHistory = window.__battleGraphHistory || [];
                window.__battleGraphHistory.push({t:new Date().toLocaleTimeString('de-DE'),s:Number(bat.learner_steps||0),e:Number(bat.episodes||0),k:Number(bat.kos||0)});
                window.__battleGraphHistory = window.__battleGraphHistory.slice(-120);
                const bh=window.__battleGraphHistory;
                upsertTrainingChart('graph-battle',bh.map(p=>p.t),[
                    {label:'Steps / 1000',data:bh.map(p=>p.s/1000),borderWidth:2,pointRadius:0,tension:.2},
                    {label:'Episoden',data:bh.map(p=>p.e),borderWidth:2,pointRadius:0,tension:.2},
                    {label:'K.O.',data:bh.map(p=>p.k),borderWidth:2,pointRadius:0,tension:.2}
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
                // Historische Trainingskanten sind optisch Agenten-Wege und
                // bleiben deshalb bis zum ausdruecklichen Schalter unsichtbar.
                if (!showAgentPaths) {
                    Object.keys(persistentEdgeLayers).forEach(key => {
                        map.removeLayer(persistentEdgeLayers[key]);
                        delete persistentEdgeLayers[key];
                    });
                    Object.keys(persistentTransitionLayers).forEach(key => {
                        map.removeLayer(persistentTransitionLayers[key]);
                        delete persistentTransitionLayers[key];
                    });
                    return;
                }
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
                updateMapperMapOverlays(false);
                updateMapperTileCoverage(false);
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

                recomputeDynamicMapBounds();
                updateGlobalTileCoverage();

                const signature =
                    `${latestGlobalMapping.tiles.length}:` +
                    `${latestGlobalMapping.edges.length}:` +
                    `${latestGlobalMapping.warp_points.length}`;

                if (!force && signature === globalMappingSignature) {
                    return;
                }
                globalMappingSignature = signature;

                // V7.4.5: Warp-Marker auf der Global Map vorerst deaktiviert.
                // Rohdaten/Transitions bleiben erhalten und werden NICHT geloescht.
                globalWarpLayer.clearLayers();

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

            const roomList = [...rooms.values()]
                .filter(r => r.tiles.size || r.edges.length)
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
                title.textContent = placeName(room.bank, room.mapId);

                const sub = document.createElement('div');
                sub.className = 'sub';
                sub.textContent =
                    `${room.tiles.size} Felder · ` +
                    `${room.edges.length} Kanten`;

                const wrap = document.createElement('div');
                wrap.className = 'room-canvas-wrap';
                wrap.appendChild(canvas);

                const legend = document.createElement('div');
                legend.className = 'room-legend';
                legend.innerHTML =
                    '<span><i class="walked"></i>erkundet</span>' +
                    '<span><i class="edge"></i>begehbare Kante</span>';

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
                ctx.fillStyle = '#596269';
                room.tiles.forEach(s => {
                    const [x,y] = s.split(',').map(Number);
                    ctx.fillRect(
                        px(x)+1, py(y)+1,
                        Math.max(1,cell-2), Math.max(1,cell-2)
                    );
                });

                // Successful traversable edges.
                ctx.strokeStyle = '#b4c7cf';
                ctx.lineWidth = Math.max(2,cell*0.12);
                room.edges.forEach(e => {
                    ctx.beginPath();
                    ctx.moveTo(cx(e[2]), cy(e[3]));
                    ctx.lineTo(cx(e[4]), cy(e[5]));
                    ctx.stroke();
                });


            });
        }

        setInterval(() => {
            if (currentTab === 'map') {
                updateGlobalMapping(false);
            }
        }, 2000);

        async function updateDashboard() {
            try {
                const res = await fetch('/api/state?t=' + Date.now());
                const state = await res.json();
                
document.getElementById('model-ver').innerText = `v${String(state.version)}`;

                window.__trainerName = state.trainer_name || 'Alex';
                const instances = state.instances || [];
                instances.forEach(i => { i.room = placeName(i.bank, i.map); });
                latestInstances = instances;
                const _v81cnt = document.getElementById('v81-agent-count');
                if (_v81cnt) _v81cnt.textContent = instances.filter(i => Number(i.id) !== 120).length;
                if (currentTab === 'watcher' || currentTab === 'map') renderWatcherTab();
                renderStatusDashboard(state);
                instances.forEach(pushHistory);
                syncAgentFilterOptions(instances);

                // Der sichtbare End-to-End-Watcher ist immer der erste Eintrag.
                // Die gewaehlte Sortierung gilt danach fuer alle Runner.
                const watcherFirst = instances.filter(i => Number(i.id) === 120);
                let workInstances = instances.filter(i => Number(i.id) !== 120);
                if (agentFilter.sort) {
                    workInstances.sort((a, b) => agentSortKey(b) - agentSortKey(a));
                } else {
                    workInstances.sort((a, b) => Number(a.id) - Number(b.id));
                }
                workInstances = watcherFirst.concat(workInstances);

                // Falls Watcher noch kein party-Feld in seiner Instanz hat,
                // die alte API-Fallback-Party nur dem Watcher zuordnen.
                const watcher = instances.find(i => i.id === 120);
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
                const _anyFilter = !!(agentFilter.role || agentFilter.map
                    || agentFilter.starter || agentFilter.stage);
                const _shown = _anyFilter
                    ? workInstances.filter(i => Number(i.id) === 120 || agentPassesFilter(i)).length
                    : instances.length;
                let hudHtml = `
                    <div class="hud-title">
                        <span>40 Full + 9 Battle · ${Math.max(0,_shown-1)}/40 Full live + Watcher${agentFilter.sort ? ' ↓' : ''}</span>
                        <span class="agent-badge">Max Speed</span>
                    </div>
                `;

                let renderedTrainCount = 0;
                let listedCount = 0;

                workInstances.forEach(inst => {
                    const isWatcher = Number(inst.id) === 120;
                    const isSelected = Number(inst.id) === Number(selectedAgentId);
                    const keep = isWatcher || isSelected || agentPassesFilter(inst);

                    // V17.4: direkt nach einem Reset ist die RAM-Position noch
                    // nicht gueltig (bank/map/x/y = 0) - das mappte bisher auf
                    // den MAP_OFFSETS-Fallback und erzeugte einen stoerenden
                    // Punkt direkt neben Pallet Town. Marker erst zeigen,
                    // sobald eine echte Position ausgelesen wurde.
                    const hasValidPos = inst.ram_valid !== false;
                    const shouldRender = keep && hasValidPos && isAgentVisible(
                        inst.id, isWatcher, renderedTrainCount
                    );

                    if (!isWatcher && selectedAgentId === null && shouldRender) {
                        renderedTrainCount++;
                    }

                    const markerColor = agentColor(inst.id);
                    const selectedClass = isSelected ? 'selected' : '';
                    const nameClass = '';

                    // HUD-Liste bleibt immer klickbar, auch wenn ein anderer Agent
                    // im Fokus ist. So kann man direkt umschalten.
                    if (keep) {
                        listedCount++;
                        const starterTag = inst.has_starter
                            ? '<span class="agent-badge" style="color:#7df9b7">🐣</span>' : '';
                        hudHtml += `
                        <div class="agent-row ${nameClass} ${selectedClass}" onclick="selectAgent(${inst.id})">
                            <span class="agent-name-wrap">
                                <span class="agent-color-dot" style="background:${markerColor}"></span>
                                <span>${agentDisplayName(inst)}</span>
                            </span>
                            <span class="agent-list-meta">${starterTag}${inst.room}<br>${inst.story_stage || ''}<br>${Number(inst.steps || 0).toLocaleString('de-DE')} Steps</span>
                        </div>
                    `;
                    }

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
                                `<b>${agentDisplayName(inst)}</b><br>${inst.room}`
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
                                `<b>${agentDisplayName(inst)}</b><br>${inst.room}`
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

                        // V17.3: Linien aus recent_path (max. 300 Punkte) fielen
                        // nach 300 Schritten am Anfang immer wieder weg - fuer
                        // den Watcher (laeuft dauerhaft) sah das wie "Linie
                        // verschwindet" aus. Keine Pfad-Linien mehr; besuchte
                        // Kacheln werden stattdessen dauerhaft als gruene
                        // Quadrate in updateGlobalTileCoverage() markiert.
                        const path = Array.isArray(inst.path) ? inst.path : [];
                        const dots = [];

                        if (
                            selectedAgentId !== null
                            && Number(inst.id) === Number(selectedAgentId)
                            && isWatcher
                        ) {
                            path.forEach(pt => {
                                if (!pt || pt.length < 4) return;
                                const pos = getLeafletCoords(
                                    Number(pt[0]), Number(pt[1]),
                                    Number(pt[2]), Number(pt[3])
                                );
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
                            });
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
<script id="twoby2-brains-js">
let latestBrains={};
const brainFmt=n=>Number(n||0).toLocaleString('de-DE');
const BATTLE_AREA_ANCHORS={
  route1:[3,19,10,18],viridian:[3,1,20,20],route2:[3,20,10,20],
  forest:[1,0,17,35],pewter:[3,2,20,30]
};
function brainFreshness(x){
  if(!x||x.updated_at==null)return 'keine Daten';
  const date=new Date(x.updated_at),clock=Number.isNaN(date.getTime())?'–':date.toLocaleTimeString('de-DE');
  const age=Number(x.age_seconds);
  return `${clock} · ${Number.isFinite(age)?age+'s':'?'}`;
}
function renderBattleFightersOnMap(b){
  if(typeof map==='undefined'||typeof L==='undefined')return;
  const areas=(b.areas||[]).filter(a=>BATTLE_AREA_ANCHORS[a]);
  const wanted=new Set();
  if(!areas.length){
    Object.keys(battleFighterMarkers).forEach(k=>{map.removeLayer(battleFighterMarkers[k]);delete battleFighterMarkers[k];});
    return;
  }
  // One marker per actual training area, not one marker per worker. Nine
  // isolated workers at the same scenario anchor otherwise obscure the map.
  for(const area of areas){
    const base=BATTLE_AREA_ANCHORS[area];
    const key=String(area);
    const pos=getLeafletCoords(base[0],base[1],base[2],base[3]);
    wanted.add(key);
    const popup=`<b><img class="battler-icon" src="/battler.png" alt="">Battle-Training</b><br>${area}<br>${brainFmt(b.workers)} Battle-Agenten · Szenario-Anker`;
    if(!battleFighterMarkers[key]){
      const icon=L.divIcon({className:'battle-fighter-marker',html:`<div class="battle-map-icon"><img src="/battler.png" alt="Battle"></div>`,iconSize:[38,38],iconAnchor:[19,19]});
      battleFighterMarkers[key]=L.marker(pos,{icon,zIndexOffset:1000})
        .bindPopup(popup).addTo(map);
    }else{
      battleFighterMarkers[key].setLatLng(pos).setPopupContent(popup);
    }
  }
  Object.keys(battleFighterMarkers).forEach(k=>{if(!wanted.has(k)){map.removeLayer(battleFighterMarkers[k]);delete battleFighterMarkers[k];}});
}
function renderBrainArchitecture(d){
  latestBrains=d||{};
  const n=latestBrains.navigation||{},b=latestBrains.battle||{},m=n.metrics||{};
  const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
  set('lth-learner',brainFmt(n.learner_steps));
  set('lth-champion',brainFmt(n.champion_steps));
  set('lth-delta',(Number(n.since_champion||0)>=0?'+':'')+brainFmt(n.since_champion));
  set('nav-brain-state',String(n.phase||'training').replaceAll('_',' ').toUpperCase());
  set('cn-ver','v'+String(n.champion_version||0));
  set('cn-steps',brainFmt(n.champion_steps));
  set('cn-stage',String(m.max_stage||0));
  set('cn-maps',String(m.max_maps||0));
  set('cn-runs',brainFmt(n.completed_runs));
  set('bb-steps',brainFmt(b.learner_steps));
  set('bb-updates',brainFmt(b.ppo_updates));
  set('bb-episodes',brainFmt(b.episodes));
  set('bb-kos',brainFmt(b.kos));
  set('bb-wins',brainFmt(b.wins));
  set('bb-wipes',brainFmt(b.wipes));
  set('bb-flees',brainFmt(b.flees));
  set('bb-learner-ver','v'+String(b.learner_version||0));
  set('bb-champion',b.champion_version?'v'+String(b.champion_version):'RULE');
  set('bb-next-check',b.next_promotion_step?brainFmt(b.next_promotion_step)+' Steps'
      :(String(b.phase||'')==='champion_evaluation'?'läuft …':'–'));
  set('bb-state',String(b.phase||'training').replaceAll('_',' ').toUpperCase());
  set('bb-workers',brainFmt(b.workers)+' Battle');
  set('bb-scenarios',brainFmt(b.scenario_count)+' Szenarien');
  {const rl=b.rolling||{}, lp=b.last_promotion||{}, p=v=>(Number(v||0)*100).toFixed(0)+'%';
   set('bb-avgrew',(Number(rl.avg_reward_last_100||0)).toFixed(2));
   set('bb-roll',rl.window?`${p(rl.win_rate_200)} / ${p(rl.wipe_rate_200)} / ${p(rl.flee_rate_200)}`:'– / – / –');
   set('bb-lastcheck',('promote'in lp)?(lp.promote?'✅ v'+String(b.champion_version):'❌ abgelehnt'):'–');
   set('bb-reason',lp.reason?String(lp.reason).slice(0,120):'Promotion nur über echte Kampfergebnisse');}
  {const ev=b.last_eval||{}, rl=b.rolling||{}, p=v=>(Number(v||0)*100).toFixed(0)+'%';
   set('cn-battle-ver',b.champion_version?'v'+String(b.champion_version):'RULE');
   set('cn-battle-win',p(ev.win_rate ?? rl.win_rate_200));
   set('cn-battle-turns',Number(ev.avg_turns_on_win ?? rl.avg_turns_on_win ?? 0).toFixed(1));
   set('cn-battle-hp',p(ev.avg_residual_hp_on_win));}
  set('pulse-nav-eval',String(n.last_eval_result||'keine').toUpperCase()+' @ '+brainFmt(n.last_eval_at_step));
  set('pulse-battle-outcomes',brainFmt(b.flees)+' / '+brainFmt(b.timeouts));
  set('pulse-nav-workers',brainFmt(n.telemetry_workers)+' / '+brainFmt(n.workers));
  set('pulse-battle-workers',brainFmt(b.workers)+' / 9');
  set('pulse-scenarios',brainFmt(b.scenario_count));
  set('pulse-runs',brainFmt(n.completed_runs));
  renderBattleFightersOnMap(b);
  const status=document.getElementById('status-summary');
  if(status){
    const cards=rows=>rows.map(x=>`<div class="status-kpi"><div class="big">${x[0]} ${x[1]}</div><div class="small">${x[2]}</div></div>`).join('');
    const nd=n.directed||{};
    const pct=v=>(Number(v||0)*100).toFixed(0)+'%';
    const navRows=[
      ['🧠',brainFmt(n.learner_steps),'Learner Steps'],
      ['🏆','v'+String(n.champion_version||0),'Champion · '+brainFmt(n.champion_steps)+' Steps'],
      ['➕',brainFmt(n.since_champion),'Seit Champion'],
      ['📍',brainFmt(m.max_stage),'Geografische Stage'],
      ['🧭',(nd.on_route1||0)+'/'+(nd.agents||0)+' ('+(nd.pct_on_route1||0)+'%)','Agenten auf Route 1'],
      ['🏚️',String(nd.in_pallet||0),'noch in Pallet Town'],
      ['↩️',String(nd.region_loops||0)+' / '+String(nd.ledge_jumps||0),'Region-Loops / Ledge-Sprünge'],
      ['🧊',String(nd.shaping_frozen_agents||0),'im Recovery-Freeze'],
      ['🕸️',brainFmt(nd.movement_graph_edges||0),'gerichtete Graph-Kanten'],
      ['🎯',nd.route1_reach_rate!=null?pct(nd.route1_reach_rate):'–','Route-1-Reachrate'],
      ['⏱️',Object.entries(nd.timeout_reasons||{}).map(([k,v])=>k+':'+v).join(' ')||'keine','aktive Timeout-Gründe'],
      ['✅',brainFmt(n.completed_runs),'abgeschlossene FULL Runs'],
      ['🔬',String(n.last_eval_result||'keine').toUpperCase(),'letzte Prüfung @ '+brainFmt(n.last_eval_at_step)]
    ];
    const navAgentRows=(nd.per_agent||[]).slice(0,40).map(a=>
      `<div class="rew-row" style="grid-template-columns:34px 1fr 1fr 60px">`+
      `<span class="rew-k">#${a.id}</span>`+
      `<span class="rew-k">→ ${a.next_hop||'–'}  d=${a.graph_distance??'–'}/${a.best??'–'}</span>`+
      `<span class="rew-k">${a.recovery?'RECOVERY ':''}${a.loop||a.timeout||''} ${(a.blocked||[]).length?'⛔'+a.blocked.join(''):''}</span>`+
      `<span class="rew-v">${a.since_improve??''}</span></div>`).join('');
    const roll=b.rolling||{}, ev=b.last_eval||{}, lp=b.last_promotion||{};
    const rcomp=roll.reward_components||{};
    const promoTxt=(lp&&('promote'in lp))?(lp.promote?'PROMOTED v'+String(b.champion_version):'REJECTED'):'noch keine';
    const battleRows=[
      ['⚔️','v'+String(b.learner_version||0)+' · '+brainFmt(b.learner_steps),'Training · Learner Steps'],
      ['🏆',b.champion_version?'v'+String(b.champion_version):'RULE','Live · Battle Champion'],
      ['🔬',b.next_promotion_step?brainFmt(b.next_promotion_step):'–','Nächste Champion-Prüfung (Steps)'],
      ['🔁',brainFmt(b.ppo_updates),'PPO Updates'],
      ['<img class="battler-icon" src="/battler.png" alt="">',brainFmt(b.episodes),'Battle Episoden (gesamt)'],
      ['💰',(Number(roll.avg_reward_last_100||0)).toFixed(2),'Ø Reward · letzte 100'],
      ['🏅',pct(roll.win_rate_200),'Siegrate · letzte 200'],
      ['☠️',pct(roll.wipe_rate_200),'Wipe-Rate · letzte 200'],
      ['🏃',pct(roll.flee_rate_200),'Flucht-Rate · letzte 200'],
      ['⏱️',(Number(roll.avg_turns_on_win||0)).toFixed(1),'Ø Züge bei Sieg'],
      ['❤️',pct(ev.avg_residual_hp_on_win),'Ø Rest-HP bei Sieg (letzte Prüfung)'],
      ['⏳',brainFmt(b.timeouts),'echte Timeouts (gesamt)'],
      ['🧪',brainFmt(b.scenario_count)+' / '+brainFmt(ev.scenario_coverage||0),'Szenarien vorhanden / in Prüfung'],
      ['🔎',promoTxt,'Ergebnis letzte Prüfung'],
    ];
    const rewBars=[
      ['Schaden',rcomp.damage,1],['Gegner-KO',rcomp.enemy_ko,1],['Sieg',rcomp.battle_win,1],
      ['Faint',rcomp.own_faint,-1],['Wipe',rcomp.wipe,-1],['Flucht',rcomp.fled,-1],
      ['Zug-Kosten',rcomp.turn_cost,-1],['ungültig',rcomp.invalid_action,-1],['Switch-Loop',rcomp.switch_loop,-1],
    ];
    const maxAbs=Math.max(1,...rewBars.map(x=>Math.abs(Number(x[1]||0))));
    const rewHtml=rewBars.map(x=>{const v=Number(x[1]||0);const w=(Math.abs(v)/maxAbs*100).toFixed(0);
      return `<div class="rew-row"><span class="rew-k">${x[0]}</span><span class="rew-bar"><i style="width:${w}%;background:${v>=0?'#3fb96b':'#d9534f'}"></i></span><span class="rew-v">${v>=0?'+':''}${v.toFixed(1)}</span></div>`;}).join('');
    const reason=lp.reason?`<div class="bp-reason"><b>${lp.promote?'Prüfung bestanden':'Ablehnungsgrund'}:</b> ${String(lp.reason)}</div>`:'';
    // spec ZIEL C.9: wild-encounter + shiny counters, separate per agent class,
    // straight from the persisted event counters. Clearly flag unverified RAM.
    const sh=latestBrains.shiny||{};
    const scl=sh.by_agent_class||{};
    const shName={battle_fighter:'Battle-Fighter',full_agent:'FULL-Agenten',watcher:'Watcher'};
    const shRow=(cls)=>{const c=scl[cls]||{};
      const lost=(Number(c.shiny_ko||0)+Number(c.shiny_fled||0)+Number(c.shiny_lost_wipe||0));
      return `<div class="rew-row" style="grid-template-columns:120px repeat(5,1fr)">`+
        `<span class="rew-k">${shName[cls]||cls}</span>`+
        `<span class="rew-k">Enc ${brainFmt(c.wild_encounters||0)}</span>`+
        `<span class="rew-k">✨ ${brainFmt(c.shiny_encounters_verified||0)}</span>`+
        `<span class="rew-k">🎣 ${brainFmt(c.shiny_caught||0)}</span>`+
        `<span class="rew-k">💔 ${brainFmt(lost)}</span>`+
        `<span class="rew-v">? ${brainFmt(c.shiny_ram_unknown||0)}</span></div>`;};
    const ramBad=!(sh.ram_verified);
    const shBanner=ramBad
      ? `<div class="bp-reason" style="border-color:#d9534f;color:#ffb3b3"><b>SHINY-RAM NICHT VERIFIZIERT</b> — PID/TID/SID für BPRD unbestätigt. `+
        `Kein Shiny wird gezählt oder belohnt, kein Fang. Status: ${sh.ram_status||'shiny_ram_unverified'}. `+
        `Freischalten via <code>tools/shiny_ram_probe.py</code>.</div>`
      : `<div class="bp-reason" style="border-color:#3fb96b;color:#b7f2c9"><b>Shiny-RAM verifiziert</b></div>`;
    const shList=(sh.recent_shinies||[]).slice(0,6).map(r=>
      `<div class="rew-row" style="grid-template-columns:1fr 1fr 1fr 1fr"><span class="rew-k">#${r.species_id||'?'} L${r.level||'?'}</span>`+
      `<span class="rew-k">${r.area||'?'}</span><span class="rew-k">${r.agent_class||'?'}</span>`+
      `<span class="rew-v">${r.outcome||'pending'}</span></div>`).join('');
    const shinySection=`<section class="status-brain-group battle"><div class="status-brain-head">✨ Shiny &amp; Wild-Encounter <span>${ramBad?'RAM UNVERIFIZIERT':'RAM OK'}</span></div>`+
      shBanner+
      `<div class="rew-break"><div class="rew-break-h">pro Agent-Klasse · Encounter · ✨ gesehen (verifiziert) · 🎣 gefangen · 💔 verloren (KO/Flucht/Wipe) · ? unbekannt/unverifiziert</div>`+
      shRow('battle_fighter')+shRow('full_agent')+shRow('watcher')+`</div>`+
      (shList?`<div class="rew-break"><div class="rew-break-h">Letzte verifizierte Shinys</div>${shList}</div>`:'')+
      `</section>`;
    status.innerHTML=`<section class="status-brain-group nav"><div class="status-brain-head">🧠 Navigation Brain <span>${String(n.phase||'training').replaceAll('_',' ').toUpperCase()}</span></div><div class="status-brain-grid">${cards(navRows)}</div>`+
      (navAgentRows?`<div class="rew-break"><div class="rew-break-h">Gerichtete Navigation · pro Agent (Next-Hop · Distanz/Best · Loop/Timeout · Steps seit Verbesserung)</div>${navAgentRows}</div>`:'')+
      `</section>`+
      `<section class="status-brain-group battle"><div class="status-brain-head">⚔️ Battle Brain <span>${String(b.phase||'training').replaceAll('_',' ').toUpperCase()}</span></div><div class="status-brain-grid">${cards(battleRows)}</div>`+
      `<div class="rew-break"><div class="rew-break-h">Reward-Aufschlüsselung (seit Prozessstart · Summe ${Number(rcomp&&roll.reward_sum||0).toFixed(0)})</div>${rewHtml}</div>${reason}</section>`+
      shinySection;
  }
}
async function refreshBrains(){try{const r=await fetch('/api/brains?ts='+Date.now());renderBrainArchitecture(await r.json());}catch(e){}}
setInterval(refreshBrains,1000);refreshBrains();
</script>
<div id="pkmai-hidden-tray"></div>

<script id="fixed-workspace-layout">
(function(){
  const main=document.getElementById('main-container');
  const workspace=document.createElement('div');
  workspace.id='map-workspace';
  const left=document.createElement('aside');
  left.id='alex-watcher-column';
  left.setAttribute('aria-label', 'Alex Watcher');
  // V18: Das Live-Bild des Watchers laeuft nur noch im Watcher-Tab. In der
  // Kartenansicht zeigt diese Spalte ausschliesslich das Detail-Feld des
  // angeklickten Agenten (inkl. der letzten 10 Reward-Events).
  left.innerHTML='<div class="alex-watcher-title">● AGENT · DETAIL</div>';
  left.appendChild(document.getElementById('detail-panel'));
  const center=document.createElement('section');
  center.id='map-column';
  center.setAttribute('aria-label','Live-Weltkarte');
  document.getElementById('brain-summary-row').appendChild(document.querySelector('.live-global'));
  center.appendChild(document.getElementById('agent-filter-bar'));
  center.appendChild(document.getElementById('map-view'));
  // V17.3: nur fuer Handy - Karte/Watcher bleiben auf Desktop-Groesse,
  // aber darunter erscheinen die Agenten als anklickbare Kacheln (wie im
  // Status-Tab) statt der schmalen Listen-Sidebar. Eigene IDs statt die
  // Watcher-Tab-Elemente zu verschieben, damit beide Tabs unabhaengig
  // funktionieren.
  const mobileAgents=document.createElement('section');
  mobileAgents.id='mobile-agent-section';
  mobileAgents.setAttribute('aria-label','Agenten');
  mobileAgents.innerHTML='<div class="wt-picker-head"><b>Agenten – antippen für Live-Stats</b><span id="map-agent-count">–</span></div>'
    + '<div class="wt-detail" id="map-agent-detail" hidden></div>'
    + '<div class="wt-grid" id="map-agent-grid"></div>';
  workspace.append(left,center,document.getElementById('hud'),mobileAgents);
  main.prepend(workspace);
  requestAnimationFrame(()=>{map.invalidateSize();recomputeDynamicMapBounds();});
})();
</script>


</body>
</html>
    """

if __name__ == '__main__':
    # Echtes Dual-Stack: EIN IPv6-Socket mit IPV6_V6ONLY=0 nimmt IPv6 UND
    # (per IPv4-mapped) IPv4 auf demselben Port an. Noetig, weil das
    # Mobilfunknetz die myfritz-Adresse per IPv6 aufloest, das LAN aber per
    # IPv4 zugreift. macOS bindet "::" sonst IPv6-only.
    import socket as _socket

    _sock = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
    try:
        _sock.setsockopt(_socket.IPPROTO_IPV6, _socket.IPV6_V6ONLY, 0)
    except OSError:
        pass
    _sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    _sock.bind(("::", 8001))
    _sock.listen(256)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[_sock])
