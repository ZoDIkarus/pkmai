#!/usr/bin/env python3
import json
import os
import secrets
import socket
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
CLUSTER_DIR = RUNTIME_DIR / "cluster"
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

KEY_FILE = CLUSTER_DIR / "cluster_key.txt"
STATE_FILE = CLUSTER_DIR / "workers.json"

HOST = "0.0.0.0"
PORT = int(os.getenv("PKMAI_CLUSTER_PORT", "8765"))

if not KEY_FILE.exists():
    KEY_FILE.write_text(secrets.token_urlsafe(32) + "\n")
    try:
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass

CLUSTER_KEY = KEY_FILE.read_text().strip()

app = FastAPI(title="PKMAI LAN Cluster", version="1.0")
LOCK = threading.Lock()
WORKERS = {}

def _load_state():
    global WORKERS
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text())
        if isinstance(data, dict):
            WORKERS = data
    except Exception:
        WORKERS = {}

def _save_state():
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(WORKERS, indent=2))
    os.replace(tmp, STATE_FILE)

def _check_key(x_pkmai_key):
    if not x_pkmai_key or not secrets.compare_digest(x_pkmai_key, CLUSTER_KEY):
        raise HTTPException(status_code=401, detail="invalid cluster key")

def _local_ip():
    # No external traffic required; UDP connect only selects interface.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

@app.get("/health")
def health():
    now = time.time()
    with LOCK:
        online = sum(
            1 for w in WORKERS.values()
            if now - float(w.get("last_seen", 0)) < 15
        )
    return {
        "ok": True,
        "role": "master",
        "host": _local_ip(),
        "port": PORT,
        "workers_online": online,
    }

@app.post("/api/worker/heartbeat")
def worker_heartbeat(payload: dict, x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)

    worker_id = str(payload.get("worker_id", "")).strip()
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id required")

    now = time.time()
    record = {
        "worker_id": worker_id,
        "hostname": str(payload.get("hostname", "")),
        "os": str(payload.get("os", "")),
        "python": str(payload.get("python", "")),
        "status": str(payload.get("status", "online")),
        "requested_agents": int(payload.get("requested_agents", 0)),
        "active_agents": int(payload.get("active_agents", 0)),
        "fps": float(payload.get("fps", 0.0)),
        "build": str(payload.get("build", "")),
        "brain_version": str(payload.get("brain_version", "")),
        "last_seen": now,
    }

    with LOCK:
        WORKERS[worker_id] = record
        _save_state()

    return {
        "ok": True,
        "server_time": now,
        "worker_id": worker_id,
    }

@app.get("/api/workers")
def workers(x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    now = time.time()
    with LOCK:
        rows = []
        for w in WORKERS.values():
            item = dict(w)
            age = max(0.0, now - float(item.get("last_seen", 0)))
            item["age_seconds"] = round(age, 1)
            item["online"] = age < 15
            rows.append(item)
    rows.sort(key=lambda x: x["worker_id"])
    return {"workers": rows}

@app.get("/api/cluster")
def cluster(x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    now = time.time()
    with LOCK:
        online = [
            w for w in WORKERS.values()
            if now - float(w.get("last_seen", 0)) < 15
        ]
    return {
        "worker_count": len(online),
        "remote_agents": sum(int(w.get("active_agents", 0)) for w in online),
        "remote_fps": round(sum(float(w.get("fps", 0.0)) for w in online), 2),
    }

if __name__ == "__main__":
    _load_state()
    print("🌐 PKMAI LAN Cluster Master V1")
    print(f"🔑 Key: {CLUSTER_KEY}")
    print(f"🖥️  LAN: http://{_local_ip()}:{PORT}")
    print("🧠 Brain/Training werden von V1 NICHT verändert.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
