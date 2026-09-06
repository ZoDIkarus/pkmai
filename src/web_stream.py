"""Small live dashboard for the local dynamic rollout cluster."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI(title="PKMAI Watchers")
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
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PKMAI Watchers</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#080b10; color:#edf4ff; }
    * { box-sizing:border-box; } body { margin:0; min-height:100vh; background:linear-gradient(145deg,#0c1320,#080b10 55%); }
    header { padding:18px 24px; border-bottom:1px solid #233044; display:flex; justify-content:space-between; gap:16px; align-items:center; }
    h1 { font-size:18px; margin:0; letter-spacing:.03em; } .sub { color:#91a0b5; font-size:12px; margin-top:4px; }
    #cluster-summary { display:flex; flex-wrap:wrap; gap:8px; justify-content:end; }
    .chip { background:#121b29; border:1px solid #26364d; border-radius:999px; padding:6px 10px; font-size:12px; color:#b9c8dc; }
    .online { color:#73edab; } .offline { color:#ff9b9b; }
    main { display:grid; grid-template-columns:minmax(230px,300px) minmax(0,1fr); min-height:calc(100vh - 76px); }
    aside { padding:18px; border-right:1px solid #233044; background:rgba(10,15,23,.78); }
    aside h2 { margin:0 0 12px; font-size:13px; color:#9cb0c9; text-transform:uppercase; letter-spacing:.1em; }
    #watcher-list { display:grid; gap:9px; }
    .watcher { width:100%; border:1px solid #26364d; border-radius:10px; padding:12px; text-align:left; background:#101826; color:inherit; cursor:pointer; }
    .watcher:hover,.watcher.selected { border-color:#45dc9a; background:#132237; box-shadow:0 0 0 1px rgba(69,220,154,.22); }
    .watcher-name { font-weight:750; font-size:14px; display:flex; justify-content:space-between; gap:8px; }
    .watcher-meta { margin-top:6px; color:#9eafc3; font-size:12px; display:flex; justify-content:space-between; }
    section { padding:18px; display:grid; grid-template-rows:auto minmax(0,1fr); gap:12px; min-height:0; }
    .title { display:flex; align-items:baseline; justify-content:space-between; gap:12px; } .title h2 { margin:0; font-size:18px; } #watcher-detail { color:#9eafc3; font-size:13px; }
    .preview { min-height:0; border:1px solid #26364d; border-radius:14px; background:#05070a; overflow:hidden; display:grid; place-items:center; }
    #watcher-preview { width:100%; height:100%; max-height:calc(100vh - 160px); object-fit:contain; image-rendering:pixelated; }
    .empty { color:#91a0b5; text-align:center; padding:24px; }
    @media (max-width:720px) { header { align-items:start; flex-direction:column; } #cluster-summary { justify-content:start; } main { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid #233044; } #watcher-list { grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); } section { min-height:65vh; } }
  </style>
</head>
<body>
  <header><div><h1>PKMAI · Watchers</h1><div class="sub">Sichtbare Ausführung der besten veröffentlichten Policy</div></div><div id="cluster-summary"><span class="chip">Cluster wird geladen …</span></div></header>
  <main>
    <aside><h2>Watcher</h2><div id="watcher-list"><div class="empty">Watcher werden geladen …</div></div></aside>
    <section><div class="title"><h2 id="watcher-title">Watcher</h2><span id="watcher-detail">Wird geladen …</span></div><div class="preview"><img id="watcher-preview" alt="Live Watcher"><div id="empty-state" class="empty" hidden>Kein Watcher verfügbar.</div></div></section>
  </main>
  <script>
    const list=document.getElementById('watcher-list'), preview=document.getElementById('watcher-preview'), title=document.getElementById('watcher-title'), detail=document.getElementById('watcher-detail'), empty=document.getElementById('empty-state');
    let selectedId=null, activeStream='';
    const text=value=>String(value ?? '');
    function choose(watcher) { selectedId=watcher.id; activeStream=watcher.stream_url; title.textContent=watcher.name; detail.textContent=`Policy v${watcher.policy_version} · Aktion: ${watcher.action}`; preview.hidden=false; empty.hidden=true; renderWatchers(window.currentWatchers || []); }
    function renderWatchers(watchers) { list.innerHTML=''; if (!watchers.length) { list.innerHTML='<div class="empty">Keine Watcher konfiguriert.</div>'; preview.hidden=true; empty.hidden=false; return; } watchers.forEach(watcher=>{ const item=document.createElement('button'); item.className='watcher'+(watcher.id===selectedId?' selected':''); item.type='button'; const state=watcher.online?'online':'offline'; item.innerHTML=`<div class="watcher-name"><span></span><span class="${state}"></span></div><div class="watcher-meta"><span></span><span></span></div>`; const spans=item.querySelectorAll('span'); spans[0].textContent=text(watcher.name); spans[1].textContent=watcher.online?'live':'offline'; spans[2].textContent=`Policy v${watcher.policy_version}`; spans[3].textContent=text(watcher.action); item.onclick=()=>choose(watcher); list.appendChild(item); }); }
    async function refreshWatchers() { try { const response=await fetch('/api/watchers?t='+Date.now(),{cache:'no-store'}); const payload=await response.json(); const watchers=payload.watchers || []; window.currentWatchers=watchers; const current=watchers.find(watcher=>watcher.id===selectedId); if (!current && watchers[0]) choose(watchers[0]); else { renderWatchers(watchers); if (current) { title.textContent=current.name; detail.textContent=`Policy v${current.policy_version} · Aktion: ${current.action}`; activeStream=current.stream_url; } } } catch { list.innerHTML='<div class="empty">Watcher-Status nicht erreichbar.</div>'; } }
    async function refreshCluster() { try { const response=await fetch('/api/cluster-status?t='+Date.now(),{cache:'no-store'}); const cluster=await response.json(); const online=(cluster.workers || []).filter(worker=>worker.online).length; document.getElementById('cluster-summary').innerHTML=`<span class="chip ${cluster.brain_online?'online':'offline'}">Brain ${cluster.brain_online?'online':'offline'}</span><span class="chip">Policy v${cluster.policy_version}</span><span class="chip">${online}/${(cluster.workers || []).length} Trainer online</span><span class="chip">${Number(cluster.timesteps || 0).toLocaleString('de-DE')} Steps</span>`; } catch { document.getElementById('cluster-summary').innerHTML='<span class="chip offline">Cluster nicht erreichbar</span>'; } }
    function refreshFrame() { if (activeStream) preview.src=activeStream+'?t='+Date.now(); }
    refreshWatchers(); refreshCluster(); refreshFrame(); setInterval(refreshWatchers,1000); setInterval(refreshCluster,2000); setInterval(refreshFrame,120);
  </script>
</body></html>"""


if __name__ == "__main__":
    host, port = web_bind_settings()
    uvicorn.run(app, host=host, port=port, log_level="warning")
