"""Live dashboard and public telemetry for the local dynamic rollout cluster."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from curriculum import GOAL_CATALOG, load_status

app = FastAPI(title="PKMAI Dashboard")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
WATCHER_STREAM_FILE = RUNTIME_DIR / "watcher.jpg"
WATCHER_STATUS_FILE = RUNTIME_DIR / "watcher.json"
CLUSTER_DIR = RUNTIME_DIR / "cluster"
CLUSTER_POLICY_FILE = CLUSTER_DIR / "policy.json"
CLUSTER_WORKERS_FILE = CLUSTER_DIR / "workers.json"
KNOWN_TILES_FILE = CLUSTER_DIR / "known_tiles.json"
EXPLORATION_MEMORY_DIR = RUNTIME_DIR / "exploration_memory"
CURRICULUM_QUALITY_FILE = RUNTIME_DIR / "curriculum_quality.json"
STAGE_EVALUATION_FILE = CLUSTER_DIR / "stage_evaluation.json"
LAST_WATCHER_FRAME: bytes | None = None


def web_bind_settings() -> tuple[str, int]:
    return os.getenv("PKMAI_WEB_BIND_HOST", "0.0.0.0"), int(
        os.getenv("PKMAI_WEB_PORT", "8001")
    )


def _load_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _public_events(values) -> list[str]:
    return [str(value)[:120] for value in (values or []) if isinstance(value, str)][-64:]


def _public_reward_trace(values) -> list[dict]:
    result = []
    for row in (values or [])[-12:]:
        if not isinstance(row, dict):
            continue
        try:
            reward = float(row.get("reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        result.append(
            {
                "step": max(0, int(row.get("step", 0) or 0)),
                "action": max(0, int(row.get("action", 0) or 0)),
                "reward": round(reward, 4),
                "events": _public_events(row.get("events")),
            }
        )
    return result


def _public_stage_evaluation(value) -> dict:
    value = value if isinstance(value, dict) else {}
    stages = value.get("stages") if isinstance(value.get("stages"), dict) else {}
    public_stages = {}
    for key, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        try:
            public_stages[str(key)[:48]] = {
                "episodes": max(0, int(stage.get("episodes", 0) or 0)),
                "successes": max(0, int(stage.get("successes", 0) or 0)),
                "success_rate": round(min(1.0, max(0.0, float(stage.get("success_rate", 0.0) or 0.0))), 4),
                "median_success_steps": max(0, int(stage["median_success_steps"])) if stage.get("median_success_steps") is not None else None,
            }
        except (TypeError, ValueError):
            continue
    status = str(value.get("status", "unavailable"))
    return {
        "policy_version": max(0, int(value.get("policy_version", 0) or 0)),
        "seed": max(0, int(value.get("seed", 0) or 0)),
        "episodes_per_stage": max(0, int(value.get("episodes_per_stage", 0) or 0)),
        "status": status if status in {"running", "blocked", "complete"} else "unavailable",
        "blocked_stage": str(value.get("blocked_stage", ""))[:48],
        "stages": public_stages,
    }


def get_watchers() -> dict:
    status = _load_json(WATCHER_STATUS_FILE)
    position = status.get("position") if isinstance(status.get("position"), dict) else {}
    active_goal = status.get("active_goal") if isinstance(status.get("active_goal"), dict) else {}
    active_goal_key = str(active_goal.get("key", "unknown"))[:32]
    active_goal_label = str(active_goal.get("label", "Wird bestimmt"))[:96]
    try:
        updated_at = WATCHER_STATUS_FILE.stat().st_mtime
    except OSError:
        updated_at = 0
    return {
        "watchers": [
            {
                "id": "dynamic-watcher",
                "name": "Best Dynamic Brain",
                "policy_version": max(0, int(status.get("policy_version", 0) or 0)),
                "action": str(status.get("action", "waiting")),
                "online": bool(updated_at and time.time() - updated_at < 5),
                "stream_url": "/watcher.jpg",
                "reward": round(float(status.get("reward", 0.0) or 0.0), 3),
                "episode_reward": round(float(status.get("episode_reward", 0.0) or 0.0), 3),
                "reward_events": _public_events(status.get("reward_events")),
                "active_goal": {
                    "key": active_goal_key,
                    "label": active_goal_label,
                },
                "episode_steps": max(0, int(status.get("episode_steps", 0) or 0)),
                "in_battle": bool(status.get("in_battle", False)),
                "position": {
                    "valid": bool(position.get("valid", False)),
                    "map_bank": max(0, int(position.get("map_bank", 0) or 0)),
                    "map_id": max(0, int(position.get("map_id", 0) or 0)),
                    "x": max(0, int(position.get("x", 0) or 0)),
                    "y": max(0, int(position.get("y", 0) or 0)),
                },
                "milestones": sorted(
                    str(value)
                    for value in (status.get("milestones") or [])
                    if isinstance(value, str) and value.replace("_", "").isalnum()
                )[:16],
            }
        ]
    }


@app.get("/api/watchers")
def get_watchers_api() -> dict:
    return get_watchers()


@app.get("/api/cluster-status")
def get_cluster_status() -> dict:
    now = time.time()
    policy = _load_json(CLUSTER_POLICY_FILE)
    stage_evaluation = _public_stage_evaluation(_load_json(STAGE_EVALUATION_FILE))
    worker_rows = _load_json(CLUSTER_WORKERS_FILE)
    workers = []
    for worker_id, row in worker_rows.items():
        if not isinstance(row, dict):
            continue
        last_seen = float(row.get("last_seen", 0) or 0)
        raw_position = row.get("position") if isinstance(row.get("position"), dict) else {}
        milestones = [
            str(value)
            for value in (row.get("milestones") or [])
            if isinstance(value, str) and value.replace("_", "").isalnum()
        ][:16]
        workers.append(
            {
                "worker_id": str(row.get("worker_id", worker_id)),
                "hostname": str(row.get("hostname", "")),
                "active_agents": max(0, int(row.get("active_agents", 0) or 0)),
                "fps": max(0.0, float(row["fps"])) if row.get("fps") is not None else None,
                "policy_version": max(0, int(row.get("policy_version", 0) or 0)),
                "position": {
                    "valid": bool(raw_position.get("valid", False)),
                    "map_bank": max(0, int(raw_position.get("map_bank", 0) or 0)),
                    "map_id": max(0, int(raw_position.get("map_id", 0) or 0)),
                    "x": max(0, int(raw_position.get("x", 0) or 0)),
                    "y": max(0, int(raw_position.get("y", 0) or 0)),
                },
                "last_action": max(0, int(row.get("last_action", 0) or 0)),
                "last_reward": round(float(row.get("last_reward", 0.0) or 0.0), 3),
                "episode_steps": max(0, int(row.get("episode_steps", 0) or 0)),
                "in_battle": bool(row.get("in_battle", False)),
                "milestones": sorted(set(milestones)),
                "training_objective": str(row.get("training_objective", "unknown"))[:32],
                "training_role": str(row.get("training_role", "unknown"))[:32],
                "story_stage": str(row.get("story_stage", "unknown"))[:48],
                "last_reward_events": _public_events(row.get("last_reward_events")),
                "episode_reward": round(float(row.get("episode_reward", 0.0) or 0.0), 3),
                "reward_trace": _public_reward_trace(row.get("reward_trace")),
                "age_seconds": round(max(0.0, now - last_seen), 1),
                "online": now - last_seen < 15,
            }
        )
    workers.sort(key=lambda item: item["worker_id"])
    observed_milestones = {
        milestone for worker in workers for milestone in worker["milestones"]
    }
    known_tiles = _load_json(KNOWN_TILES_FILE)
    known_warps = []
    warp_sources = list(EXPLORATION_MEMORY_DIR.glob("agent_*.json"))
    warp_sources += list((RUNTIME_DIR / "curriculum_shared" / "confirmed_story_warps_v2").glob("*.json"))

    for memory_file in warp_sources:
        memory = _load_json(memory_file)
        transitions = []
        if isinstance(memory, dict):
            transitions.extend(memory.get("transitions", []))

            if isinstance(memory.get("transition"), list):
                transitions.append(memory["transition"])
        for transition in transitions:
            if isinstance(transition, list) and len(transition) == 8:
                for endpoint in (transition[:4], transition[4:]):
                    if endpoint not in known_warps:
                        known_warps.append(endpoint)
    objective_counts = Counter(
        worker["training_objective"]
        for worker in workers
        if worker["training_objective"] not in {"", "unknown"}
    )
    observed_events = {
        event.split(":", 1)[0]
        for worker in workers
        for event in (
            worker["last_reward_events"]
            + [event for trace in worker["reward_trace"] for event in trace["events"]]
        )
    }
    curriculum_stages = load_status(str(CURRICULUM_QUALITY_FILE)).get("stages", {})
    goals = [
        {
            "key": key,
            "label": label,
            "category": category,
            "objective": objective,
            "observed": evidence in observed_milestones or evidence in observed_events,
            "average_steps": (
                max(0, int((curriculum_stages.get(key) or {}).get("average_steps")))
                if (curriculum_stages.get(key) or {}).get("average_steps") is not None
                else None
            ),
            "active_trainers": 0,
        }
        for key, label, category, objective, evidence in GOAL_CATALOG
    ]
    for objective, count in objective_counts.items():
        candidates = [goal for goal in goals if goal["objective"] == objective]
        if candidates:
            next(goal for goal in candidates if not goal["observed"] or goal is candidates[-1])["active_trainers"] = count
    for goal in goals:
        del goal["objective"]
    try:
        policy_mtime = CLUSTER_POLICY_FILE.stat().st_mtime
    except OSError:
        policy_mtime = 0
    return {
        "brain_online": bool(policy_mtime and now - policy_mtime < 45),
        "policy_version": max(0, int(policy.get("version", 0) or 0)),
        "timesteps": max(0, int(policy.get("timesteps", 0) or 0)),
        "checkpoint": Path(str(policy.get("checkpoint") or "")).name,
        "goals": goals,
        "learning_objectives": [
            {"key": objective, "trainers": objective_counts[objective]}
            for objective in sorted(objective_counts)
        ],
        "workers": workers,
        "known_tiles": known_tiles if isinstance(known_tiles, list) else [],
        "known_warps": known_warps,
        "stage_evaluation": stage_evaluation,
    }


@app.get("/watcher.jpg")
def get_watcher_frame():
    global LAST_WATCHER_FRAME
    try:
        frame = WATCHER_STREAM_FILE.read_bytes()
        if not frame:
            raise OSError("empty watcher frame")
        LAST_WATCHER_FRAME = frame
    except (FileNotFoundError, OSError):
        frame = LAST_WATCHER_FRAME
        if frame is None:
            return Response(status_code=404)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/watcher", include_in_schema=False)
def watcher_dashboard_redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=307)


@app.get("/watcher-observer", response_class=HTMLResponse, include_in_schema=False)
def watcher_view() -> str:
    return """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PKMAI · Live Watcher</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#101010;color:#e7e7e7}*{box-sizing:border-box}body{margin:0;background:#101010;min-height:100vh}header{height:69px;border-bottom:1px solid #292929;padding:8px 20px}h1{margin:0;font-size:20px;letter-spacing:.02em}.subtitle{color:#cdbf9e;font-weight:600;font-size:13px;margin-top:10px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 278px;gap:9px;padding:0 12px 10px}.screen{height:calc(100vh - 79px);min-height:540px;background:#000;display:grid;place-items:center;overflow:hidden}.screen img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}.sidebar{display:grid;grid-template-rows:auto minmax(0,1fr);gap:9px;padding-top:4px}.panel{background:#272324;border:1px solid #464143;padding:10px}.mode{display:inline-block;background:#67df68;color:#123d17;font-weight:800;padding:6px 12px;font-size:12px}.model{font-size:12px;color:#b5afb0;margin-top:8px}.state{display:grid;gap:7px;margin-top:16px;font-size:13px}.row{display:flex;gap:8px;align-items:center}.dot{width:12px;height:12px;border-radius:50%;background:#6ce775;flex:none}.dot.warn{background:#ddc364}.dot.bad{background:#e94b65}.positive{color:#6ce775}.negative{color:#e94b65}.hint{font-size:10px;color:#8b8586;margin-top:12px}.event-head{font-size:13px;color:#9ae29a;font-weight:800;margin-bottom:8px}.events{display:grid;gap:9px;overflow:auto;max-height:calc(100vh - 340px)}.event{display:grid;grid-template-columns:12px 1fr auto;gap:7px;align-items:center;color:#c5bfc0;font-size:12px}.event b{font-weight:650}.event em{font-style:normal;color:#e94b65}.event em.good{color:#6ce775}@media(max-width:760px){.layout{grid-template-columns:1fr}.screen{height:58vh;min-height:360px}.events{max-height:240px}}
.map-atlas{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.map-card{border:1px solid #26364d;border-radius:12px;background:#0b111b;padding:12px}.map-card h3{margin-bottom:8px}.map-card .coordinate-map{position:relative;min-height:260px;background:#0b111b;border:1px solid #31425c;border-radius:8px;overflow:hidden}.map-card .dot{position:absolute;width:16px;height:16px;border-radius:50%;background:#45dc9a;border:2px solid #071018;transform:translate(-50%,-50%);font-size:8px;text-align:center}.map-card .dot.battle{background:#ff8b8b}.map-card .watcher-dot{background:#ff9f43;border-color:#fff0d0}.map-card .known-tile{position:absolute;background:#45dc9a;border:0;opacity:.78}.map-card .known-warp{position:absolute;background:#ffd447;border:2px solid #8a6500;z-index:2;pointer-events:none}</style></head><body>
<header><h1>PKMAI - LIVE WATCHER</h1><div class="subtitle" id="summary">Beste Dynamic Policy wird geladen …</div></header>
<main class="layout"><section class="screen"><img id="watcher-frame" alt="Live PKMAI Watcher"></section><aside class="sidebar"><section class="panel"><span class="mode" id="watcher-mode">OVERWORLD</span><div class="model" id="watcher-model">dynamic_policy_best</div><div class="state" id="watcher-state"></div><div class="hint">Gleiche Policy-Verteilung wie die Rollout-Trainer. Inferenz only, kein Lernen.</div></section><section class="panel"><div class="event-head">REWARD EVENTS <span style="float:right;color:#7d7778;font-size:10px">neueste zuerst</span></div><div class="events" id="watcher-events"></div></section></aside></main>
<script>
const $=id=>document.getElementById(id);let last='';function el(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n}function pos(w){const p=w.position||{};return p.valid?`Route B${p.map_bank}/M${p.map_id} · ${p.x},${p.y}`:'Position wird gelesen'}function render(w){$('summary').textContent=`Best Brain v${w.policy_version} | Episode ${w.episode_steps||0} Schritte | Aktion ${w.action}`;$('watcher-mode').textContent=w.in_battle?'BATTLE':'OVERWORLD';$('watcher-model').textContent=`dynamic_policy_best · Policy v${w.policy_version}`;const state=$('watcher-state');state.replaceChildren();const rows=[['Episode '+(w.episode_steps||0),'dot'],['Reward '+Number(w.reward||0).toFixed(3),'dot '+(w.reward>=0?'':'bad')],['Aktion '+w.action,'dot '+(w.action==='START'?'warn':'')],[pos(w),'dot'],[w.in_battle?'Battle aktiv':'Keine Battle','dot '+(w.in_battle?'bad':'')]];rows.forEach(([label,cls])=>{const r=el('div',undefined,'row');r.append(el('span','',cls),el('span',label));state.append(r)});const events=$('watcher-events');events.replaceChildren();const eventData=[['Letzte Policy-Aktion',w.action,Number(w.reward||0)],['Episode-Schritt',String(w.episode_steps||0),0],['Kartenstatus',pos(w),0],...((w.milestones||[]).slice(-4).reverse().map(x=>['Milestone',x,0])];eventData.forEach(([name,value,reward])=>{const r=el('div',undefined,'event');r.append(el('span','','dot '+(reward<0?'bad':reward>0?'':'warn')),el('b',name+' · '+value),el('em',(reward>=0?'+':'')+reward.toFixed(3),reward>0?'good':''));events.append(r)})}async function tick(){try{const payload=await fetch('/api/watchers?t='+Date.now(),{cache:'no-store'}).then(r=>r.json());const w=payload.watchers?.[0];if(!w)return;render(w);$('watcher-frame').src=w.stream_url+'?t='+Date.now()}catch{ $('summary').textContent='Watcher-Status nicht erreichbar' }}tick();setInterval(tick,150);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PKMAI Dashboard</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#080b10;color:#edf4ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#0c1320,#080b10 60%);min-height:100vh}header{padding:16px 24px;border-bottom:1px solid #26364d;display:flex;gap:18px;justify-content:space-between;align-items:center}h1,h2,h3,p{margin:0}.sub,.muted{color:#9eafc3;font-size:13px}.nav{display:flex;gap:7px;flex-wrap:wrap}.nav button,.map-select{border:1px solid #31425c;background:#111a28;color:#c8d7eb;border-radius:8px;padding:8px 10px;cursor:pointer}.nav button.active{border-color:#45dc9a;background:#153126;color:#84efb4}.summary{display:flex;gap:8px;flex-wrap:wrap}.chip,.badge{padding:5px 8px;border:1px solid #26364d;border-radius:999px;background:#111a28;font-size:12px}.online{color:#73edab}.offline{color:#ff9b9b}main{padding:20px;max-width:1500px;margin:auto}.page{display:none}.page.active{display:block}.page-head{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:15px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.card,.panel{border:1px solid #26364d;border-radius:12px;background:rgba(15,24,38,.88);padding:14px}.metric{font-size:24px;font-weight:750;margin-top:7px}.grid{display:grid;grid-template-columns:minmax(220px,300px) minmax(0,1fr);gap:16px}.watcher{width:100%;text-align:left;margin-bottom:8px;border:1px solid #26364d;border-radius:9px;padding:11px;background:#101826;color:inherit;cursor:pointer}.watcher.selected{border-color:#45dc9a}.preview{background:#05070a;border:1px solid #26364d;border-radius:12px;min-height:500px;display:grid;place-items:center;overflow:hidden}.preview img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{text-align:left;padding:10px;border-bottom:1px solid #26364d;white-space:nowrap}.table th{color:#9eafc3;font-weight:600}.trainer-row{cursor:pointer}.trainer-row:hover,.trainer-row.selected{background:#153126}.map-layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px}.coordinate-map{position:relative;min-height:540px;border:1px solid #26364d;border-radius:12px;background:linear-gradient(#122238 1px,transparent 1px),linear-gradient(90deg,#122238 1px,transparent 1px),#09111c;background-size:32px 32px;overflow:hidden}.coordinate-map:before{content:'Kartenkoordinaten · keine Bildkarte';position:absolute;left:12px;top:10px;color:#7790ac;font-size:12px}.dot{position:absolute;width:25px;height:25px;border-radius:50%;border:2px solid #d9ffe9;background:#157b51;color:#fff;font-size:9px;display:grid;place-items:center;transform:translate(-50%,-50%);cursor:default}.dot.battle{background:#af513f}.legend{display:grid;gap:8px}.goal{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:12px 0;border-bottom:1px solid #26364d}.goal:last-child{border-bottom:0}.empty{color:#9eafc3;padding:18px;text-align:center}.observer{display:grid;grid-template-columns:minmax(0,1fr) 285px;gap:14px}.observer-screen{min-height:560px;background:#000;border:1px solid #26364d;border-radius:12px;overflow:hidden;display:grid;place-items:start}.observer-screen img{width:100%;height:100%;object-fit:contain;object-position:top left;image-rendering:pixelated}.observer-side{display:grid;align-content:start;gap:12px}.observer-mode{display:inline-block;padding:6px 10px;border-radius:6px;background:#57d873;color:#092716;font-size:12px;font-weight:800}.observer-state{display:grid;gap:8px;margin-top:14px;font-size:13px}.observer-row{display:flex;gap:8px;align-items:center}.observer-row i{width:10px;height:10px;border-radius:50%;background:#73edab;flex:none}.observer-row i.warn{background:#f2ce67}.observer-row i.bad{background:#ff7b86}.event-list{display:grid;gap:9px;max-height:360px;overflow:auto;padding-right:6px}.event{display:grid;grid-template-columns:10px 1fr auto;gap:7px;align-items:center;font-size:12px}.event i{width:10px;height:10px;border-radius:50%;background:#f2ce67}.event i.bad{background:#ff7b86}.event i.good{background:#73edab}.event b{font-weight:650}.event em{font-style:normal;color:#9eafc3}.event em.good{color:#73edab}@media(max-width:800px){header{align-items:start;flex-direction:column}.grid,.map-layout,.observer{grid-template-columns:1fr}.preview{min-height:340px}.observer-screen{min-height:360px}main{padding:14px}}
.map-atlas{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.map-card{border:1px solid #26364d;border-radius:12px;background:#0b111b;padding:12px}.map-card h3{margin-bottom:8px}.map-card .coordinate-map{position:relative;min-height:260px;background:#0b111b;border:1px solid #31425c;border-radius:8px;overflow:hidden}.map-card .dot{position:absolute;width:16px;height:16px;border-radius:50%;background:#45dc9a;border:2px solid #071018;transform:translate(-50%,-50%);font-size:8px;text-align:center}.map-card .dot.battle{background:#ff8b8b}.map-card .watcher-dot{background:#ff9f43;border-color:#fff0d0}.map-card .known-tile{position:absolute;background:#45dc9a;border:0;opacity:.78}.map-card .known-warp{position:absolute;background:#ffd447;border:2px solid #8a6500;z-index:2;pointer-events:none}</style></head><body>
<header><div><h1>PKMAI · Lokaler Trainingscluster</h1><div class="sub">Live-Telemetrie ohne private Runtime- oder Modellpfade</div></div><nav class="nav" id="nav"><button class="active" data-page="watchers">Watcher</button><button data-page="trainers">Trainer</button><button data-page="overworld">Overworld</button><button data-page="stats">Live-Statistik</button><button data-page="goals">Lernziele</button></nav><div class="summary" id="summary"></div></header>
<main>
<section class="page active" id="page-watchers"><div class="page-head"><div><h2>PKMAI · Live Watcher</h2><p class="muted" id="watcher-summary">Beste veröffentlichte Policy wird geladen …</p></div></div><div class="observer"><section class="observer-screen"><img id="watcher-frame" alt="Live PKMAI Watcher"></section><aside class="observer-side"><section class="panel"><span class="observer-mode" id="watcher-mode">OVERWORLD</span><div class="muted" id="watcher-model" style="margin-top:10px">dynamic_policy_best</div><div id="watcher-state" class="observer-state"></div><p class="muted" style="margin-top:14px">Gleiche Policy-Verteilung wie die Rollout-Trainer. Inferenz only, kein Lernen.</p></section><section class="panel"><h3 style="color:#84efb4">REWARD EVENTS</h3><div id="watcher-events" class="event-list" style="margin-top:12px"></div></section><section class="panel"><h3>Aktive Watcher</h3><div id="watcher-list" class="muted" style="margin-top:10px">Wird geladen …</div></section></aside></div></section>
<section class="page" id="page-trainers"><div class="page-head"><div><h2>Trainer</h2><p class="muted">Anklicken: Skill, Etappe und die jüngsten Reward-Rechnungen prüfen</p></div></div><div class="panel table-wrap"><table class="table"><thead><tr><th>Trainer</th><th>Status</th><th>Skill</th><th>Etappe</th><th>Policy</th><th>FPS</th><th>Position</th><th>Aktion</th><th>Letzter Step</th><th>Episoden-Reward</th><th>Episode Steps</th><th>Battle</th><th>Alter</th></tr></thead><tbody id="trainer-rows"></tbody></table></div><section class="panel" id="trainer-detail" style="margin-top:16px"><h3>Trainer-Details</h3><div class="muted" id="trainer-detail-summary" style="margin-top:8px">Trainer auswählen …</div><div class="event-list" id="trainer-last-events" style="margin-top:12px"></div><h3 style="margin-top:16px">Reward-Extremwerte (höchster + / tiefster −)</h3><div class="event-list" id="trainer-reward-extremes" style="margin-top:10px"></div><h3 style="margin-top:16px">Letzte Reward-Rechnungen</h3><div class="event-list" id="trainer-reward-trace" style="margin-top:10px"></div></section></section>
<section class="page" id="page-overworld"><div class="page-head"><div><h2>Overworld-Atlas</h2><p class="muted">Alle aktuell bekannten Karten mit gemeinsamen lokalen Koordinaten</p></div></div><div class="map-atlas" id="map-atlas"></div><div class="panel" style="margin-top:14px"><h3>Karten-Legende</h3><div id="map-legend" class="legend muted"></div></div></section>
<section class="page" id="page-stats"><div class="page-head"><div><h2>Live-Statistik</h2><p class="muted">Aggregierte Signale aus Brain und Trainerfleet</p></div></div><div class="cards" id="stat-cards"></div></section>
<section class="page" id="page-goals"><div class="page-head"><div><h2>Lernziele & Etappen</h2><p class="muted">Zielstatus und die aktuelle Trainerverteilung sind getrennt dargestellt.</p></div></div><div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))"><section class="panel"><h3>Zielkatalog</h3><div class="muted" style="margin-top:6px">Erreichte und nächste Curriculum-Ziele</div><div id="goal-catalog"></div></section><section class="panel"><h3>Aktive Trainingsaufträge</h3><div class="muted" style="margin-top:6px">Jeder der 10 Trainer erscheint genau einmal.</div><div id="active-training-objectives"></div></section></div></section>
</main>
<script>
const state={watchers:[],cluster:{workers:[]},selectedWatcher:null,selectedTrainer:null,selectedMap:null};const A=['A','B','START','UP','DOWN','LEFT','RIGHT'];
const $=id=>document.getElementById(id),esc=v=>String(v??'');const position=w=>w.position?.valid?`B${w.position.map_bank} M${w.position.map_id} · ${w.position.x},${w.position.y}`:'keine Koordinate';
function make(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function renderSummary(){const c=state.cluster,w=c.workers||[],online=w.filter(x=>x.online).length;$('summary').replaceChildren(make('span',c.brain_online?'Brain online':'Brain offline','chip '+(c.brain_online?'online':'offline')),make('span',`Policy v${c.policy_version||0}`,'chip'),make('span',`${online}/${w.length} Trainer`,'chip'),make('span',`${Number(c.timesteps||0).toLocaleString('de-DE')} Steps`,'chip'))}
function watcherPosition(w){const p=w.position||{};return p.valid?`B${p.map_bank} M${p.map_id} · ${p.x},${p.y}`:'Position wird gelesen'}function renderWatcherObserver(w){const goal=w.active_goal?.label||'Wird bestimmt';$('watcher-summary').textContent=`Best Brain v${w.policy_version} · Lernziel: ${goal} · Episode ${w.episode_steps||0} Schritte · Aktion ${w.action}`;$('watcher-mode').textContent=w.in_battle?'BATTLE':'OVERWORLD';$('watcher-model').textContent=`dynamic_policy_best · Policy v${w.policy_version}`;const stateRoot=$('watcher-state');stateRoot.replaceChildren();[['Aktuelles Lernziel: '+goal,''],['Episode '+(w.episode_steps||0),''],['Trainings-Reward (laufende Episode) '+Number(w.episode_reward||0).toFixed(3),w.episode_reward<0?'bad':''],['Aktion '+w.action,w.action==='START'?'warn':''],[watcherPosition(w),''],[w.in_battle?'Battle aktiv':'Keine Battle',w.in_battle?'bad':'']].forEach(([label,kind])=>{const row=make('div',undefined,'observer-row');const dot=make('i');if(kind)dot.className=kind;row.append(dot,make('span',label));stateRoot.append(row)});const events=$('watcher-events');events.replaceChildren();[['Aktuelles Lernziel',goal,0],['Letzte Policy-Aktion',w.action,Number(w.reward||0)],['Episode-Schritt',String(w.episode_steps||0),0],['Kartenstatus',watcherPosition(w),0],...((w.reward_events||[]).slice().reverse().map(x=>['Trainings-Event',x,Number((x.match(/:([+-]\\d+(?:\\.\\d+)?)/)||[,0])[1])])),...((w.milestones||[]).slice(-4).reverse().map(x=>['Milestone',x,0]))].forEach(([name,value,reward])=>{const row=make('div',undefined,'event'),dot=make('i');dot.className=reward<0?'bad':reward>0?'good':'';row.append(dot,make('b',name+' · '+value),make('em',(reward>=0?'+':'')+reward.toFixed(3),reward>0?'good':''));events.append(row)});$('watcher-frame').src=w.stream_url+'?t='+Date.now()}function renderWatchers(){const root=$('watcher-list');root.replaceChildren();if(!state.watchers.length){root.append(make('div','Keine Watcher verfügbar.','empty'));return}if(!state.selectedWatcher)state.selectedWatcher=state.watchers[0].id;state.watchers.forEach(w=>{const b=make('button',undefined,'watcher'+(w.id===state.selectedWatcher?' selected':''));b.type='button';b.append(make('strong',w.name));b.append(make('div',`Policy v${w.policy_version} · ${w.active_goal?.label||'Wird bestimmt'} · ${w.online?'live':'offline'}`,'muted'));b.onclick=()=>{state.selectedWatcher=w.id;renderWatchers()};root.append(b)});renderWatcherObserver(state.watchers.find(w=>w.id===state.selectedWatcher)||state.watchers[0])}
function renderTrainerDetail(w){const summary=$('trainer-detail-summary'),events=$('trainer-last-events'),extremes=$('trainer-reward-extremes'),trace=$('trainer-reward-trace');const parseEv=(raw)=>{const s=String(raw),m=s.match(/([+-]\d+(?:\.\d+)?)\s*$/);if(!m)return null;const val=Number(m[1]);if(!Number.isFinite(val))return null;let label=s.slice(0,m.index).replace(/:$/,''),step='';const sm=label.match(/^(\d+):(.*)$/);if(sm){step=sm[1];label=sm[2]}return {val,label,step}};const fmtV=v=>(v>0?'+':'')+(Math.abs(v)>=1?v.toFixed(1):v.toFixed(3));if(!w){summary.textContent='Trainer auswählen …';events.replaceChildren();extremes.replaceChildren();trace.replaceChildren();return}summary.textContent=`${w.worker_id} · ${w.fps===null||w.fps===undefined?'FPS wird gemessen':Number(w.fps).toFixed(2)+' FPS'} · Skill ${w.training_objective||'unbekannt'} · Rolle ${w.training_role||'unbekannt'} · Etappe ${w.story_stage||'unbekannt'} · Episode ${w.episode_steps||0} · Gesamt ${Number(w.episode_reward||0).toFixed(3)}`;events.replaceChildren();const last=w.last_reward_events||[];if(!last.length)events.append(make('div','Keine Reward-Events im letzten Step.','muted'));last.slice().reverse().forEach(x=>events.append(make('div',x,'event')));const allEvents=[...last,...(w.reward_trace||[]).flatMap(x=>x.events||[])];let hi=null,lo=null;allEvents.map(parseEv).filter(Boolean).forEach(p=>{if(p.val>0&&(!hi||p.val>hi.val))hi=p;if(p.val<0&&(!lo||p.val<lo.val))lo=p});extremes.replaceChildren();[['⬆',hi,'good'],['⬇',lo,'bad']].forEach(([arrow,p,kind])=>{if(!p)return;const row=make('div',undefined,'event'),dot=make('i');dot.className=kind;row.append(dot,make('b',arrow+' '+p.label+(p.step?' · Schritt '+p.step:'')),make('em',fmtV(p.val),kind==='good'?'good':''));extremes.append(row)});if(!extremes.children.length)extremes.append(make('div','Keine numerischen Reward-Events im aktuellen Rollout.','muted'));trace.replaceChildren();const rows=w.reward_trace||[];if(!rows.length)trace.append(make('div','Noch keine Reward-Rechnungen im aktuellen Rollout.','muted'));rows.slice().reverse().forEach(x=>{const eventText=(x.events||[]).join(' · ')||'keine Events';trace.append(make('div',`Schritt ${x.step} · ${A[x.action]||x.action} · ${Number(x.reward||0).toFixed(4)} · ${eventText}`,'event'))})}
function renderTrainers(){const root=$('trainer-rows'),workers=state.cluster.workers||[];root.replaceChildren();if(!workers.some(w=>w.worker_id===state.selectedTrainer))state.selectedTrainer=workers[0]?.worker_id||null;workers.forEach(w=>{const r=document.createElement('tr');r.className='trainer-row'+(w.worker_id===state.selectedTrainer?' selected':'');r.tabIndex=0;r.title='Trainer-Details anzeigen';[w.worker_id,w.online?'online':'offline',`${w.training_objective||'unbekannt'} · ${w.training_role||'–'}`,w.story_stage||'–',`v${w.policy_version}`,w.fps===null||w.fps===undefined?'–':Number(w.fps).toFixed(2),position(w),A[w.last_action]||w.last_action,Number(w.last_reward).toFixed(3),Number(w.episode_reward||0).toFixed(2),w.episode_steps,w.in_battle?'ja':'–',w.age_seconds+' s'].forEach((v,i)=>r.append(make('td',v,i===1?(w.online?'online':'offline'):'')));r.onclick=()=>{state.selectedTrainer=w.worker_id;renderTrainers()};r.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();r.click()}};root.append(r)});if(!root.children.length)root.append(make('tr','Keine Trainertelemetrie.','empty'));renderTrainerDetail(workers.find(w=>w.worker_id===state.selectedTrainer))}
function renderMap(){const workers=state.cluster.workers||[],groups={};(state.cluster.known_tiles||[]).forEach(t=>{if(!Array.isArray(t)||t.length!==4)return;const [b,m,x,y]=t,k=`B${b} · M${m}`;(groups[k]??=[]).push({position:{valid:true,map_bank:b,map_id:m,x,y},known:true})});workers.filter(w=>w.position?.valid).forEach(w=>{const p=w.position,k=`B${p.map_bank} · M${p.map_id}`;(groups[k]??=[]).push(w)});(state.watchers||[]).filter(w=>w.position?.valid).forEach(w=>{const p=w.position,k=`B${p.map_bank} · M${p.map_id}`;(groups[k]??=[]).push({position:p,worker_id:w.id||'watcher',watcher:true})});const atlas=$('map-atlas'),legend=$('map-legend');atlas.replaceChildren();legend.replaceChildren();Object.entries(groups).forEach(([key,rows])=>{const card=make('section',undefined,'map-card'),title=make('h3',key),map=make('div',undefined,'coordinate-map');const xs=rows.map(w=>w.position.x),ys=rows.map(w=>w.position.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),rangeX=Math.max(1,maxX-minX),rangeY=Math.max(1,maxY-minY);const known=new Map(rows.filter(w=>w.known).map(w=>[`${w.position.x},${w.position.y}`,w])),tileSize=80/(Math.max(rangeX,rangeY)+1);known.forEach((w,key)=>{const [x,y]=key.split(',').map(Number),tile=make('div');tile.className='known-tile';tile.style.left=(10+(x-minX)*tileSize)+'%';tile.style.top=(10+(y-minY)*tileSize)+'%';tile.style.width=tileSize+'%';tile.style.height=tileSize+'%';map.append(tile)});(state.cluster.known_warps||[]).filter(w=>Array.isArray(w)&&w.length===4&&`B${w[0]} · M${w[1]}`===key).forEach(w=>{const warp=make('div');warp.className='known-warp';warp.style.left=(10+(w[2]-minX)*tileSize)+'%';warp.style.top=(10+(w[3]-minY)*tileSize)+'%';warp.style.width=tileSize+'%';warp.style.height=tileSize+'%';warp.title='Bekannter Warp';map.append(warp)});rows.filter(w=>!w.known).forEach(w=>{const p=w.position,d=make('div',w.worker_id.replace('local-trainer-',''),'dot'+(w.watcher?' watcher-dot':'')+(w.in_battle?' battle':''));d.style.left=(10+(p.x-minX)*tileSize+tileSize/2)+'%';d.style.top=(10+(p.y-minY)*tileSize+tileSize/2)+'%';d.title=`${w.worker_id}: ${position(w)}`;map.append(d);legend.append(make('div',`${w.worker_id} · ${key} · ${p.x},${p.y}${w.in_battle?' · Battle':''}`))});card.append(title,make('div',`Bekannte Fläche: X ${minX}–${maxX} · Y ${minY}–${maxY}`,'muted'),map);atlas.append(card)});legend.append(make('div','Grün · bekannte Tiles'),make('div','Schwarz · Rand der bekannten Fläche'),make('div','Gelb · bekannte Warps'),make('div','Blau · bekannte Ziele'));if(!Object.keys(groups).length){atlas.append(make('div','Warte auf bekannte Overworld-Tiles.','empty'));legend.append(make('div','Noch keine persistenten Kartendaten vorhanden.','muted'))}}
function renderStats(){const w=state.cluster.workers||[],valid=w.filter(x=>x.position?.valid),battle=w.filter(x=>x.in_battle),rewards=w.length?w.reduce((a,x)=>a+Number(x.last_reward||0),0)/w.length:0,episodeRewards=w.length?w.reduce((a,x)=>a+Number(x.episode_reward||0),0)/w.length:0,ev=state.cluster.stage_evaluation||{},evStatus=ev.status||'unavailable',evProgress=Object.values(ev.stages||{}).reduce((n,s)=>n+Number(s.episodes||0),0),evLabel=evStatus==='blocked'?`blocked · ${ev.blocked_stage||'unbekannt'}`:`${evStatus} · ${evProgress}/${Number(ev.episodes_per_stage||0)*4}`;const stats=[['Trainer online',w.filter(x=>x.online).length+'/'+w.length],['Policy-Version','v'+(state.cluster.policy_version||0)],['Trainingsschritte',Number(state.cluster.timesteps||0).toLocaleString('de-DE')],['Evaluator',evLabel],['Evaluator-Policy',ev.policy_version?'v'+ev.policy_version:'–'],['Kartenpositionen',valid.length],['Aktive Battles',battle.length],['Ø letzter Reward',rewards.toFixed(3)],['Ø Episoden-Reward',episodeRewards.toFixed(2)]];$('stat-cards').replaceChildren(...stats.map(([k,v])=>{const e=make('div',undefined,'card');e.append(make('div',k,'muted'),make('div',v,'metric'));return e}))}
function renderGoals(){const goals=state.cluster.goals||[],objectives=state.cluster.learning_objectives||[],catalog=$('goal-catalog'),active=$('active-training-objectives');catalog.replaceChildren(...goals.map(g=>{const e=make('div',undefined,'goal'),status=[g.category,g.observed?'Milestone gespeichert':'noch kein Signal'].filter(Boolean).join(' · '),average=g.average_steps===null||g.average_steps===undefined?'Ø Schritte: –':`Ø Schritte: ${Number(g.average_steps).toLocaleString('de-DE')}`;const details=make('div');details.append(make('div',g.label),make('div',average,'muted'));e.append(details,make('span',status,'badge '+(g.observed?'online':'')));return e}));active.replaceChildren(...objectives.map(o=>{const e=make('div',undefined,'goal');e.append(make('div',o.key),make('span',o.trainers+' Trainer','badge online'));return e}));const total=objectives.reduce((sum,o)=>sum+Number(o.trainers||0),0),summary=make('div',undefined,'goal');summary.append(make('div','Trainer gesamt'),make('span',total+' eindeutig zugeordnet','badge online'));active.append(summary)}
function render(){renderSummary();renderWatchers();renderTrainers();renderMap();renderStats();renderGoals()}async function refresh(){try{const [watchers,cluster]=await Promise.all([fetch('/api/watchers?t='+Date.now(),{cache:'no-store'}).then(r=>r.json()),fetch('/api/cluster-status?t='+Date.now(),{cache:'no-store'}).then(r=>r.json())]);state.watchers=watchers.watchers||[];state.cluster=cluster;render()}catch{ $('summary').replaceChildren(make('span','Cluster nicht erreichbar','chip offline')) }}
$('nav').onclick=e=>{const page=e.target.dataset.page;if(!page)return;document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+page))};refresh();setInterval(refresh,1500);setInterval(()=>{const active=state.watchers.find(w=>w.id===state.selectedWatcher)||state.watchers[0];if(active)$('watcher-frame').src=active.stream_url+'?t='+Date.now()},150);
</script></body></html>"""


if __name__ == "__main__":
    host, port = web_bind_settings()
    uvicorn.run(app, host=host, port=port, log_level="warning")
