#!/usr/bin/env python3
"""Authenticated PKMAI cluster control plane; the brain remains the only writer."""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
import uvicorn

from cluster_config import (
    ClusterCompatibilityError,
    ClusterSettings,
    build_environment_signature,
    validate_worker_registration,
)
from rollout_protocol import decode_rollout, write_rollout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
CLUSTER_DIR = RUNTIME_DIR / "cluster"
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
KEY_FILE = Path(os.getenv("PKMAI_CLUSTER_KEY_FILE", CLUSTER_DIR / "cluster_key.txt"))
STATE_FILE = CLUSTER_DIR / "workers.json"
POLICY_FILE = CLUSTER_DIR / "policy.json"
MODEL_FILE = CLUSTER_DIR / "dynamic_policy.pt"
ROLLOUT_INBOX = CLUSTER_DIR / "rollout_inbox"
PORT = int(os.getenv("PKMAI_CLUSTER_PORT", "8765"))
WORKER_TTL_SECONDS = max(15, int(os.getenv("PKMAI_WORKER_TTL_SECONDS", "60")))


def _load_or_create_key() -> str:
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        os.chmod(KEY_FILE, 0o600)
    return KEY_FILE.read_text(encoding="utf-8").strip()


CLUSTER_KEY = _load_or_create_key()
SETTINGS = ClusterSettings(
    environment_signature=build_environment_signature(
        observation_shape=(64, 64, 1), nav_features=28, action_count=7
    )
)
app = FastAPI(title="PKMAI Cluster Control", version="2")
LOCK = threading.Lock()
WORKERS: dict[str, dict] = {}


def _policy_version() -> int:
    try:
        return int(json.loads(POLICY_FILE.read_text(encoding="utf-8")).get("version", 0))
    except Exception:
        return 0


def load_policy_artifact() -> bytes:
    if not MODEL_FILE.is_file():
        raise HTTPException(status_code=503, detail="dynamic policy is not published yet")
    return MODEL_FILE.read_bytes()


def _save_state() -> None:
    temp = STATE_FILE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(WORKERS, sort_keys=True), encoding="utf-8")
    os.replace(temp, STATE_FILE)


def _prune_workers(now: float) -> None:
    stale = [
        worker_id
        for worker_id, row in WORKERS.items()
        if now - float(row.get("last_seen", 0) or 0) >= WORKER_TTL_SECONDS
    ]
    for worker_id in stale:
        del WORKERS[worker_id]
    if stale:
        _save_state()


def _check_key(value: str | None) -> None:
    if not value or not secrets.compare_digest(value, CLUSTER_KEY):
        raise HTTPException(status_code=401, detail="invalid cluster key")


def _record(payload: dict, decision_reason: str) -> dict:
    worker_id = str(payload.get("worker_id", "")).strip()
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id required")
    raw_position = payload.get("position") if isinstance(payload.get("position"), dict) else {}
    milestones = [
        str(value)
        for value in (payload.get("milestones") or [])
        if isinstance(value, str) and value.replace("_", "").isalnum()
    ][:16]
    record = {
        "worker_id": worker_id,
        "hostname": str(payload.get("hostname", "")),
        "build": str(payload.get("build", "")),
        "signature": str(payload.get("signature", "")),
        "active_agents": max(0, int(payload.get("active_agents", 0))),
        "fps": (
            max(0.0, float(payload["fps"]))
            if payload.get("fps") is not None
            else None
        ),
        "policy_version": max(0, int(payload.get("policy_version", 0))),
        "position": {
            "valid": bool(raw_position.get("valid", False)),
            "map_bank": max(0, int(raw_position.get("map_bank", 0) or 0)),
            "map_id": max(0, int(raw_position.get("map_id", 0) or 0)),
            "x": max(0, int(raw_position.get("x", 0) or 0)),
            "y": max(0, int(raw_position.get("y", 0) or 0)),
        },
        "last_action": max(0, int(payload.get("last_action", 0) or 0)),
        "last_reward": (
            float(payload.get("last_reward", 0.0))
            if math.isfinite(float(payload.get("last_reward", 0.0) or 0.0))
            else 0.0
        ),
        "episode_steps": max(0, int(payload.get("episode_steps", 0) or 0)),
        "in_battle": bool(payload.get("in_battle", False)),
        "milestones": sorted(set(milestones)),
        "status": decision_reason,
        "last_seen": time.time(),
    }
    with LOCK:
        WORKERS[worker_id] = record
        _save_state()
    return record


def store_rollout_upload(worker_id: str, batch) -> Path:
    return write_rollout(ROLLOUT_INBOX, worker_id, batch)


def store_rollout_payload(worker_id: str, payload: bytes) -> tuple[Path, int]:
    batch = decode_rollout(payload)
    return store_rollout_upload(worker_id, batch), int(len(batch["actions"]))


@app.get("/health")
def health():
    now = time.time()
    with LOCK:
        _prune_workers(now)
        online = sum(now - float(row.get("last_seen", 0)) < 15 for row in WORKERS.values())
    return {"ok": True, "role": "control-plane", "workers_online": online, "policy_version": _policy_version()}


@app.post("/api/worker/register")
def register(payload: dict, x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    try:
        decision = validate_worker_registration(
            SETTINGS,
            worker_signature=str(payload.get("signature", "")),
            worker_policy_version=int(payload.get("policy_version", 0)),
            master_policy_version=_policy_version(),
        )
    except ClusterCompatibilityError as exc:
        _record(payload, "rejected")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _record(payload, decision.reason)
    return {"accepted": decision.accepted, "reason": decision.reason, "policy_version": decision.policy_version}


@app.post("/api/worker/heartbeat")
def heartbeat(payload: dict, x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    _record(payload, "online")
    return {"ok": True, "policy_version": _policy_version()}


@app.get("/api/policy")
def policy(x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    return Response(content=load_policy_artifact(), media_type="application/octet-stream")


@app.post("/api/rollout/{worker_id}")
async def rollout_upload(
    worker_id: str,
    request: Request,
    x_pkmai_key: str | None = Header(default=None),
):
    _check_key(x_pkmai_key)
    try:
        _, samples = store_rollout_payload(worker_id, await request.body())
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid rollout: {exc}") from exc
    return {"accepted": True, "samples": samples}


@app.get("/api/cluster")
def cluster(x_pkmai_key: str | None = Header(default=None)):
    _check_key(x_pkmai_key)
    now = time.time()
    with LOCK:
        _prune_workers(now)
        workers = [dict(row, online=now - float(row.get("last_seen", 0)) < 15) for row in WORKERS.values()]
    return {"policy_version": _policy_version(), "workers": sorted(workers, key=lambda item: item["worker_id"])}


if __name__ == "__main__":
    app_port = PORT
    app_host = os.getenv("PKMAI_CLUSTER_HOST", "0.0.0.0")
    uvicorn.run(app, host=app_host, port=app_port, log_level="info")
