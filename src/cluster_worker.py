#!/usr/bin/env python3
"""Remote PKMAI rollout-node registration and heartbeat client."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cluster_config import build_environment_signature

MASTER_URL = os.getenv("PKMAI_CLUSTER_MASTER_URL", "http://10.10.15.112:8765").rstrip("/")
KEY_FILE = Path(os.getenv("PKMAI_CLUSTER_KEY_FILE", "local/cluster_key.txt"))
WORKER_ID = os.getenv("PKMAI_WORKER_ID", socket.gethostname())
ACTIVE_AGENTS = max(1, int(os.getenv("PKMAI_WORKER_AGENTS", "1")))
HEARTBEAT_SECONDS = max(3, int(os.getenv("PKMAI_HEARTBEAT_SECONDS", "10")))


def build_id() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def payload(policy_version: int = 0) -> dict:
    return {
        "worker_id": WORKER_ID,
        "hostname": socket.gethostname(),
        "build": build_id(),
        "signature": build_environment_signature(
            observation_shape=(64, 64, 1), nav_features=28, action_count=7
        ),
        "active_agents": ACTIVE_AGENTS,
        "policy_version": policy_version,
        "platform": platform.platform(),
    }


def request(path: str, body: dict) -> dict:
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    data = json.dumps(body).encode("utf-8")
    req = Request(
        f"{MASTER_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "X-PKMAI-Key": key},
    )
    with urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if not KEY_FILE.exists():
        raise SystemExit(f"cluster key file missing: {KEY_FILE}")
    try:
        registration = request("/api/worker/register", payload())
    except HTTPError as exc:
        raise SystemExit(f"registration rejected: {exc.read().decode('utf-8', 'replace')}") from exc
    except URLError as exc:
        raise SystemExit(f"master unavailable: {exc.reason}") from exc
    version = int(registration["policy_version"])
    if not registration.get("accepted"):
        if registration.get("reason") != "policy_reload_required":
            raise SystemExit(f"registration not accepted: {registration.get('reason')}")
        print(f"policy reload required; adopting version={version}", flush=True)
    else:
        print(f"registered worker={WORKER_ID} policy_version={version}", flush=True)
    while True:
        time.sleep(HEARTBEAT_SECONDS)
        try:
            response = request("/api/worker/heartbeat", payload(version))
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"heartbeat failed; retrying: {exc}", flush=True)
            continue
        version = int(response.get("policy_version", version))


if __name__ == "__main__":
    main()
