"""Live dashboard and public telemetry for the local dynamic rollout cluster."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI(title="PKMAI Dashboard")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
WATCHER_STREAM_FILE = RUNTIME_DIR / "watcher.jpg"
WATCHER_STATUS_FILE = RUNTIME_DIR / "watcher.json"
CLUSTER_DIR = RUNTIME_DIR / "cluster"
CLUSTER_POLICY_FILE = CLUSTER_DIR / "policy.json"
CLUSTER_WORKERS_FILE = CLUSTER_DIR / "workers.json"


def web_bind_settings() -> tuple[str, int]:
    return os.getenv("PKMAI_WEB_BIND_HOST", "0.0.0.0"), int(
        os.getenv("PKMAI_WEB_PORT", "8001")
    )


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def get_watchers() -> dict:
    status = _load_json(WATCHER_STATUS_FILE)
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
                "age_seconds": round(max(0.0, now - last_seen), 1),
                "online": now - last_seen < 15,
            }
        )
    workers.sort(key=lambda item: item["worker_id"])
    try:
        policy_mtime = CLUSTER_POLICY_FILE.stat().st_mtime
    except OSError:
        policy_mtime = 0
    return {
        "brain_online": bool(policy_mtime and now - policy_mtime < 45),
        "policy_version": max(0, int(policy.get("version", 0) or 0)),
        "timesteps": max(0, int(policy.get("timesteps", 0) or 0)),
        "checkpoint": Path(str(policy.get("checkpoint") or "")).name,
        "workers": workers,
    }


@app.get("/watcher.jpg")
def get_watcher_frame():
    try:
        return FileResponse(
            WATCHER_STREAM_FILE,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    except (FileNotFoundError, OSError):
        return Response(status_code=404)


@app.get("/watcher", response_class=HTMLResponse)
def watcher_view() -> str:
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>PKMAI Watcher</title><style>html,body{margin:0;height:100%;background:#080b10}img{width:100%;height:100%;object-fit:contain}</style></head><body><img id="watcher" alt="PKMAI Watcher wird geladen"><script>const image=document.getElementById('watcher');setInterval(()=>image.src='/watcher.jpg?t='+Date.now(),100);</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PKMAI Dashboard</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#080b10;color:#edf4ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#0c1320,#080b10 60%);min-height:100vh}header{padding:16px 24px;border-bottom:1px solid #26364d;display:flex;gap:18px;justify-content:space-between;align-items:center}h1,h2,h3,p{margin:0}.sub,.muted{color:#9eafc3;font-size:13px}.nav{display:flex;gap:7px;flex-wrap:wrap}.nav button,.map-select{border:1px solid #31425c;background:#111a28;color:#c8d7eb;border-radius:8px;padding:8px 10px;cursor:pointer}.nav button.active{border-color:#45dc9a;background:#153126;color:#84efb4}.summary{display:flex;gap:8px;flex-wrap:wrap}.chip,.badge{padding:5px 8px;border:1px solid #26364d;border-radius:999px;background:#111a28;font-size:12px}.online{color:#73edab}.offline{color:#ff9b9b}main{padding:20px;max-width:1500px;margin:auto}.page{display:none}.page.active{display:block}.page-head{display:flex;justify-content:space-between;gap:12px;align-items:end;margin-bottom:15px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.card,.panel{border:1px solid #26364d;border-radius:12px;background:rgba(15,24,38,.88);padding:14px}.metric{font-size:24px;font-weight:750;margin-top:7px}.grid{display:grid;grid-template-columns:minmax(220px,300px) minmax(0,1fr);gap:16px}.watcher{width:100%;text-align:left;margin-bottom:8px;border:1px solid #26364d;border-radius:9px;padding:11px;background:#101826;color:inherit;cursor:pointer}.watcher.selected{border-color:#45dc9a}.preview{background:#05070a;border:1px solid #26364d;border-radius:12px;min-height:500px;display:grid;place-items:center;overflow:hidden}.preview img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{text-align:left;padding:10px;border-bottom:1px solid #26364d;white-space:nowrap}.table th{color:#9eafc3;font-weight:600}.map-layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px}.coordinate-map{position:relative;min-height:540px;border:1px solid #26364d;border-radius:12px;background:linear-gradient(#122238 1px,transparent 1px),linear-gradient(90deg,#122238 1px,transparent 1px),#09111c;background-size:32px 32px;overflow:hidden}.coordinate-map:before{content:'Kartenkoordinaten · keine Bildkarte';position:absolute;left:12px;top:10px;color:#7790ac;font-size:12px}.dot{position:absolute;width:25px;height:25px;border-radius:50%;border:2px solid #d9ffe9;background:#157b51;color:#fff;font-size:9px;display:grid;place-items:center;transform:translate(-50%,-50%);cursor:default}.dot.battle{background:#af513f}.legend{display:grid;gap:8px}.goal{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:12px 0;border-bottom:1px solid #26364d}.goal:last-child{border-bottom:0}.empty{color:#9eafc3;padding:18px;text-align:center}@media(max-width:800px){header{align-items:start;flex-direction:column}.grid,.map-layout{grid-template-columns:1fr}.preview{min-height:340px}main{padding:14px}}
</style></head><body>
<header><div><h1>PKMAI · Lokaler Trainingscluster</h1><div class="sub">Live-Telemetrie ohne private Runtime- oder Modellpfade</div></div><nav class="nav" id="nav"><button class="active" data-page="watchers">Watcher</button><button data-page="trainers">Trainer</button><button data-page="overworld">Overworld</button><button data-page="stats">Live-Statistik</button><button data-page="goals">Lernziele</button></nav><div class="summary" id="summary"></div></header>
<main>
<section class="page active" id="page-watchers"><div class="page-head"><div><h2>Watcher</h2><p class="muted">Beste veröffentlichte Policy in einem separaten Emulator</p></div></div><div class="grid"><div class="panel"><h3>Aktive Watcher</h3><div id="watcher-list" class="muted">Wird geladen …</div></div><div class="preview"><img id="watcher-preview" alt="Live Watcher"></div></div></section>
<section class="page" id="page-trainers"><div class="page-head"><div><h2>Trainer</h2><p class="muted">Individuelle lokale Rollout-Worker und ihre letzte Telemetrie</p></div></div><div class="panel table-wrap"><table class="table"><thead><tr><th>Trainer</th><th>Status</th><th>Policy</th><th>Position</th><th>Aktion</th><th>Reward</th><th>Episode</th><th>Battle</th><th>Alter</th></tr></thead><tbody id="trainer-rows"></tbody></table></div></section>
<section class="page" id="page-overworld"><div class="page-head"><div><h2>Overworld</h2><p class="muted">Aktuelle Trainerpositionen auf der gewählten Kartenkoordinaten-Ebene</p></div><select class="map-select" id="map-select"></select></div><div class="map-layout"><div class="coordinate-map" id="coordinate-map"></div><div class="panel"><h3>Karten-Legende</h3><div id="map-legend" class="legend muted"></div></div></div></section>
<section class="page" id="page-stats"><div class="page-head"><div><h2>Live-Statistik</h2><p class="muted">Aggregierte Signale aus Brain und Trainerfleet</p></div></div><div class="cards" id="stat-cards"></div></section>
<section class="page" id="page-goals"><div class="page-head"><div><h2>Lernziele & Etappen</h2><p class="muted">Status basiert auf den von Trainern veröffentlichten Milestone-Signalen</p></div></div><div class="panel" id="goals"></div></section>
</main>
<script>
const state={watchers:[],cluster:{workers:[]},selectedWatcher:null,selectedMap:null};const A=['A','B','START','UP','DOWN','LEFT','RIGHT'];
const $=id=>document.getElementById(id),esc=v=>String(v??'');const position=w=>w.position?.valid?`B${w.position.map_bank} M${w.position.map_id} · ${w.position.x},${w.position.y}`:'keine Koordinate';
function make(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function renderSummary(){const c=state.cluster,w=c.workers||[],online=w.filter(x=>x.online).length;$('summary').replaceChildren(make('span',c.brain_online?'Brain online':'Brain offline','chip '+(c.brain_online?'online':'offline')),make('span',`Policy v${c.policy_version||0}`,'chip'),make('span',`${online}/${w.length} Trainer`,'chip'),make('span',`${Number(c.timesteps||0).toLocaleString('de-DE')} Steps`,'chip'))}
function renderWatchers(){const root=$('watcher-list');root.replaceChildren();if(!state.watchers.length){root.append(make('div','Keine Watcher verfügbar.','empty'));return}if(!state.selectedWatcher)state.selectedWatcher=state.watchers[0].id;state.watchers.forEach(w=>{const b=make('button',undefined,'watcher'+(w.id===state.selectedWatcher?' selected':''));b.type='button';b.append(make('strong',w.name));b.append(make('div',`Policy v${w.policy_version} · ${w.action} · ${w.online?'live':'offline'}`,'muted'));b.onclick=()=>{state.selectedWatcher=w.id;renderWatchers()};root.append(b)});const active=state.watchers.find(w=>w.id===state.selectedWatcher)||state.watchers[0];$('watcher-preview').src=active.stream_url+'?t='+Date.now()}
function renderTrainers(){const root=$('trainer-rows');root.replaceChildren();state.cluster.workers.forEach(w=>{const r=document.createElement('tr');[w.worker_id,w.online?'online':'offline',`v${w.policy_version}`,position(w),A[w.last_action]||w.last_action,Number(w.last_reward).toFixed(3),w.episode_steps,w.in_battle?'ja':'–',w.age_seconds+' s'].forEach((v,i)=>r.append(make('td',v,i===1?(w.online?'online':'offline'):'')));root.append(r)});if(!root.children.length)root.append(make('tr','Keine Trainertelemetrie.','empty'))}
function mapKey(w){const p=w.position;return p?.valid?`B${p.map_bank} · M${p.map_id}`:null}function renderMap(){const workers=state.cluster.workers.filter(w=>w.position?.valid),keys=[...new Set(workers.map(mapKey).filter(Boolean))].sort();const select=$('map-select');if(!keys.includes(state.selectedMap))state.selectedMap=keys[0]||null;select.replaceChildren(...keys.map(k=>{const o=make('option',k);o.value=k;o.selected=k===state.selectedMap;return o}));select.disabled=!keys.length;const canvas=$('coordinate-map'),legend=$('map-legend');canvas.replaceChildren();legend.replaceChildren();workers.filter(w=>mapKey(w)===state.selectedMap).forEach(w=>{const p=w.position,d=make('div',w.worker_id.replace('local-trainer-',''), 'dot'+(w.in_battle?' battle':''));d.style.left=(8+(p.x%64)/64*84)+'%';d.style.top=(10+(p.y%64)/64*80)+'%';d.title=`${w.worker_id}: ${position(w)}`;canvas.append(d);legend.append(make('div',`${w.worker_id} · ${p.x},${p.y}${w.in_battle?' · Battle':''}`))});if(!workers.length){canvas.append(make('div','Warte auf gültige Overworld-Koordinaten.','empty'));legend.append(make('div','Positionen erscheinen nach dem nächsten Trainer-Heartbeat.','muted'))}}
function renderStats(){const w=state.cluster.workers||[],valid=w.filter(x=>x.position?.valid),battle=w.filter(x=>x.in_battle),rewards=w.length?w.reduce((a,x)=>a+Number(x.last_reward||0),0)/w.length:0;const stats=[['Trainer online',w.filter(x=>x.online).length+'/'+w.length],['Policy-Version','v'+(state.cluster.policy_version||0)],['Trainingsschritte',Number(state.cluster.timesteps||0).toLocaleString('de-DE')],['Kartenpositionen',valid.length],['Aktive Battles',battle.length],['Ø letzter Reward',rewards.toFixed(3)]];$('stat-cards').replaceChildren(...stats.map(([k,v])=>{const e=make('div',undefined,'card');e.append(make('div',k,'muted'),make('div',v,'metric'));return e}))}
function renderGoals(){const found=new Set(state.cluster.workers.flatMap(w=>w.milestones||[]));const goals=[['intro_complete','Intro abschließen'],['stairs_down','Treppe erreichen'],['left_house','Haus verlassen'],['starter','Starter erhalten']];const root=$('goals');root.replaceChildren(...goals.map(([key,label])=>{const e=make('div',undefined,'goal');e.append(make('div',label),make('span',found.has(key)?'Milestone gespeichert':'noch kein Signal','badge '+(found.has(key)?'online':'')));return e}));const flow=make('div',undefined,'goal');flow.append(make('div','Kontinuierliche Policy-Lernschleife'),make('span',state.cluster.brain_online?'aktiv':'wartet','badge '+(state.cluster.brain_online?'online':'offline')));root.append(flow)}
function render(){renderSummary();renderWatchers();renderTrainers();renderMap();renderStats();renderGoals()}async function refresh(){try{const [watchers,cluster]=await Promise.all([fetch('/api/watchers?t='+Date.now(),{cache:'no-store'}).then(r=>r.json()),fetch('/api/cluster-status?t='+Date.now(),{cache:'no-store'}).then(r=>r.json())]);state.watchers=watchers.watchers||[];state.cluster=cluster;render()}catch{ $('summary').replaceChildren(make('span','Cluster nicht erreichbar','chip offline')) }}
$('nav').onclick=e=>{const page=e.target.dataset.page;if(!page)return;document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+page))};$('map-select').onchange=e=>{state.selectedMap=e.target.value;renderMap()};refresh();setInterval(refresh,1500);setInterval(()=>{if(state.selectedWatcher)renderWatchers()},150);
</script></body></html>"""


if __name__ == "__main__":
    host, port = web_bind_settings()
    uvicorn.run(app, host=host, port=port, log_level="warning")
