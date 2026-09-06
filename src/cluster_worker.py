#!/usr/bin/env python3
"""Independent PKMAI rollout worker; no Ray runtime or optimizer state."""

from __future__ import annotations

import io
import json
import os
import platform
import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch.distributions import Categorical

from cluster_config import build_environment_signature
from dynamic_policy import PKMAIPolicy
from pokemon_env import PokemonFireRedEnv
from rollout_protocol import encode_rollout

MASTER_URL = os.getenv("PKMAI_CLUSTER_MASTER_URL", "http://10.10.15.1:8765").rstrip("/")
KEY_FILE = Path(os.getenv("PKMAI_CLUSTER_KEY_FILE", "local/cluster_key.txt"))
WORKER_ID = os.getenv("PKMAI_WORKER_ID", socket.gethostname())
ACTIVE_AGENTS = max(1, int(os.getenv("PKMAI_WORKER_AGENTS", "1")))
HEARTBEAT_SECONDS = max(3, int(os.getenv("PKMAI_HEARTBEAT_SECONDS", "10")))
ROLLOUT_STEPS = max(8, int(os.getenv("PKMAI_ROLLOUT_STEPS", "32")))


def _public_reward_events(events) -> list[str]:
    return [
        str(value)[:120]
        for value in (events or [])
        if isinstance(value, str)
    ][-8:]


def live_telemetry(
    env: PokemonFireRedEnv,
    *,
    action: int,
    reward: float,
    info: dict | None = None,
    reward_trace: list[dict] | None = None,
) -> dict:
    info = info or {}
    location = getattr(env, "cached_loc", {}) or {}
    valid = bool(location.get("valid", False))
    return {
        "position": {
            "valid": valid,
            "map_bank": max(0, int(location.get("map_bank", 0) or 0)),
            "map_id": max(0, int(location.get("map_id", 0) or 0)),
            "x": max(0, int(location.get("x_pos", 0) or 0)),
            "y": max(0, int(location.get("y_pos", 0) or 0)),
        },
        "last_action": max(0, int(action)),
        "last_reward": float(reward),
        "episode_steps": max(0, int(getattr(env, "total_steps", 0) or 0)),
        "in_battle": bool(getattr(env, "last_in_battle", False)),
        "milestones": sorted(str(value) for value in getattr(env, "saved_milestones", ()) or ()),
        "training_objective": str(info.get("training_objective", "unknown"))[:32],
        "training_role": str(info.get("training_role", "unknown"))[:32],
        "story_stage": str(info.get("story_stage", "unknown"))[:48],
        "last_reward_events": _public_reward_events(info.get("reward_events")),
        "episode_reward": round(float(info.get("episode_reward", 0.0) or 0.0), 3),
        "reward_trace": list(reward_trace or [])[-12:],
    }


def worker_rank(hostname: str | None = None, explicit_rank: str | None = None) -> int:
    configured_rank = explicit_rank if explicit_rank is not None else os.getenv("PKMAI_WORKER_RANK")
    if configured_rank:
        try:
            return max(0, int(configured_rank))
        except ValueError:
            pass
    match = re.search(r"-(\d+)$", hostname or socket.gethostname())
    return max(0, int(match.group(1)) - 1) if match else 0


def build_id() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def payload(policy_version: int = 0, telemetry: dict | None = None) -> dict:
    result = {
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
    if telemetry:
        result.update(telemetry)
    return result


def _key() -> str:
    return KEY_FILE.read_text(encoding="utf-8").strip()


def request_json(path: str, body: dict) -> dict:
    req = Request(
        f"{MASTER_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-PKMAI-Key": _key()},
    )
    with urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def request_bytes(path: str) -> bytes:
    req = Request(f"{MASTER_URL}{path}", headers={"X-PKMAI-Key": _key()})
    with urlopen(req, timeout=30) as response:
        return response.read()


def upload_rollout(batch: dict[str, np.ndarray]) -> dict:
    req = Request(
        f"{MASTER_URL}/api/rollout/{WORKER_ID}",
        data=encode_rollout(batch),
        method="POST",
        headers={"Content-Type": "application/octet-stream", "X-PKMAI-Key": _key()},
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def choose_action(policy: PKMAIPolicy, observation: dict) -> tuple[int, float, float]:
    image = torch.from_numpy(np.asarray(observation["image"], dtype=np.uint8))[None, ...]
    nav = torch.from_numpy(np.asarray(observation["nav"], dtype=np.float32))[None, ...]
    with torch.no_grad():
        logits, values = policy(image, nav)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
    return int(action.item()), float(distribution.log_prob(action).item()), float(values.item())


def load_policy() -> tuple[PKMAIPolicy, int]:
    artifact = torch.load(io.BytesIO(request_bytes("/api/policy")), map_location="cpu")
    policy = PKMAIPolicy()
    policy.load_state_dict(artifact["state_dict"])
    policy.eval()
    return policy, int(artifact["version"])


def collect_rollout(env: PokemonFireRedEnv, policy: PKMAIPolicy, observation: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    rows = {name: [] for name in ("images", "nav", "actions", "rewards", "dones", "log_probs", "values")}
    telemetry = live_telemetry(env, action=0, reward=0.0)
    reward_trace = []
    for _ in range(ROLLOUT_STEPS):
        action, log_prob, value = choose_action(policy, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        rows["images"].append(np.asarray(observation["image"], dtype=np.uint8))
        rows["nav"].append(np.asarray(observation["nav"], dtype=np.float32))
        rows["actions"].append(action)
        rows["rewards"].append(float(reward))
        done = bool(terminated or truncated)
        rows["dones"].append(done)
        rows["log_probs"].append(log_prob)
        rows["values"].append(value)
        reward_trace.append(
            {
                "step": max(0, int(info.get("episode_steps", getattr(env, "total_steps", 0)) or 0)),
                "action": max(0, int(action)),
                "reward": round(float(reward), 4),
                "events": _public_reward_events(info.get("reward_events")),
            }
        )
        telemetry = live_telemetry(
            env, action=action, reward=reward, info=info, reward_trace=reward_trace
        )
        observation = env.reset()[0] if done else next_observation
    return {
        "images": np.asarray(rows["images"], dtype=np.uint8),
        "nav": np.asarray(rows["nav"], dtype=np.float32),
        "actions": np.asarray(rows["actions"], dtype=np.int64),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "dones": np.asarray(rows["dones"], dtype=np.bool_),
        "log_probs": np.asarray(rows["log_probs"], dtype=np.float32),
        "values": np.asarray(rows["values"], dtype=np.float32),
    }, observation, telemetry


def main() -> None:
    if not KEY_FILE.exists():
        raise SystemExit(f"cluster key file missing: {KEY_FILE}")
    fleet_size = max(ACTIVE_AGENTS, int(os.getenv("PKMAI_WORKER_FLEET_SIZE", ACTIVE_AGENTS)))
    env = PokemonFireRedEnv(rank=worker_rank(), agent_count=fleet_size)
    observation, _ = env.reset()
    registration = request_json("/api/worker/register", payload())
    policy = None
    version = int(registration.get("policy_version", -1))
    last_heartbeat = 0.0
    telemetry = None
    try:
        while True:
            try:
                response = request_json("/api/worker/heartbeat", payload(version, telemetry))
                target_version = int(response.get("policy_version", version))
                if policy is None or target_version != version:
                    policy, version = load_policy()
                    print(f"loaded policy version={version}", flush=True)
                last_heartbeat = time.monotonic()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"master/policy unavailable; retrying: {exc}", flush=True)
                time.sleep(HEARTBEAT_SECONDS)
                continue
            batch, observation, telemetry = collect_rollout(env, policy, observation)
            try:
                response = upload_rollout(batch)
                print(f"uploaded samples={response.get('samples', 0)} policy={version}", flush=True)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                print(f"rollout upload failed; retrying: {exc}", flush=True)
            if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                continue
    finally:
        env.close()


if __name__ == "__main__":
    main()
