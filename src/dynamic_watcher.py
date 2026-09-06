"""Visible, non-training rollout that follows the best published dynamic brain."""

from __future__ import annotations

import os
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dynamic_policy import PKMAIPolicy
from pokemon_env import PokemonFireRedEnv
from watcher_stream import write_watcher_stream_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = PROJECT_ROOT / "runtime" / "cluster"
BEST_MODEL_FILE = CLUSTER_DIR / "dynamic_policy_best.pt"
LATEST_MODEL_FILE = CLUSTER_DIR / "dynamic_policy.pt"
WATCHER_STATUS_FILE = PROJECT_ROOT / "runtime" / "watcher.json"
ACTION_NAMES = ("A", "B", "START", "UP", "DOWN", "LEFT", "RIGHT")


def select_published_model(best_path: Path = BEST_MODEL_FILE, latest_path: Path = LATEST_MODEL_FILE) -> Path:
    if best_path.is_file():
        return best_path
    if latest_path.is_file():
        return latest_path
    raise FileNotFoundError("no published dynamic brain is available")


def model_signature(path: Path) -> tuple[Path, int, int]:
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def load_published_policy(path: Path) -> tuple[PKMAIPolicy, int]:
    artifact = torch.load(path, map_location="cpu")
    policy = PKMAIPolicy()
    policy.load_state_dict(artifact["state_dict"])
    policy.eval()
    return policy, int(artifact["version"])


def choose_watcher_action(
    policy: PKMAIPolicy, observation: dict, generator: torch.Generator | None = None
) -> int:
    image = torch.from_numpy(np.asarray(observation["image"], dtype=np.uint8))[None, ...]
    nav = torch.from_numpy(np.asarray(observation["nav"], dtype=np.float32))[None, ...]
    with torch.no_grad():
        logits, _ = policy(image, nav)
        probabilities = torch.softmax(logits, dim=1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def watcher_telemetry(
    env: PokemonFireRedEnv, reward: float, reward_events: list[str] | None = None
) -> dict:
    location = getattr(env, "cached_loc", {}) or {}
    return {
        "reward": round(float(reward), 3),
        "episode_steps": max(0, int(getattr(env, "total_steps", 0) or 0)),
        "in_battle": bool(getattr(env, "last_in_battle", False)),
        "position": {
            "valid": bool(location.get("valid", False)),
            "map_bank": max(0, int(location.get("map_bank", 0) or 0)),
            "map_id": max(0, int(location.get("map_id", 0) or 0)),
            "x": max(0, int(location.get("x_pos", 0) or 0)),
            "y": max(0, int(location.get("y_pos", 0) or 0)),
        },
        "milestones": sorted(str(value) for value in getattr(env, "saved_milestones", ()) or ()),
        "reward_events": [
            str(event)[:120]
            for event in (reward_events or [])
            if isinstance(event, str)
        ][-8:],
    }


def append_recent_reward_events(
    recent_events: list[str], current_events: list[str] | None
) -> list[str]:
    """Keep recent public reward events visible beyond their single trigger step."""
    return [
        str(event)[:120]
        for event in [*recent_events, *(current_events or [])]
        if isinstance(event, str)
    ][-8:]


def write_watcher_status(
    status_file: Path, policy_version: int, action: int, telemetry: dict | None = None
) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_file.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "id": "dynamic-watcher",
                "policy_version": policy_version,
                "action": ACTION_NAMES[action],
                **(telemetry or {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, status_file)


def annotate_frame(screen: np.ndarray, policy_version: int, action: int) -> np.ndarray:
    return cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)


def main() -> None:
    reload_seconds = max(0.1, float(os.getenv("PKMAI_WATCHER_RELOAD_SECONDS", "1.0")))
    watcher_rank = max(0, int(os.getenv("PKMAI_WATCHER_RANK", "120")))
    fleet_size = max(1, int(os.getenv("PKMAI_WATCHER_FLEET_SIZE", "121")))
    env = PokemonFireRedEnv(rank=watcher_rank, agent_count=fleet_size)
    env.EXPLORATION_MEMORY_ENABLED = False
    policy: PKMAIPolicy | None = None
    policy_version = -1
    signature = None
    last_reload = 0.0
    recent_reward_events: list[str] = []
    observation, _ = env.reset()
    try:
        while True:
            now = time.monotonic()
            if policy is None or now - last_reload >= reload_seconds:
                try:
                    path = select_published_model()
                    candidate_signature = model_signature(path)
                    if candidate_signature != signature:
                        policy, policy_version = load_published_policy(path)
                        signature = candidate_signature
                        print(f"watcher loaded best brain version={policy_version}", flush=True)
                    last_reload = now
                except (FileNotFoundError, OSError, RuntimeError, KeyError) as exc:
                    print(f"watcher waiting for best brain: {exc}", flush=True)
                    time.sleep(reload_seconds)
                    continue
            action = choose_watcher_action(policy, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            recent_reward_events = append_recent_reward_events(
                recent_reward_events,
                info.get("reward_events"),
            )
            write_watcher_stream_frame(
                annotate_frame(env.env.get_screen(), policy_version, action),
                PROJECT_ROOT / "runtime" / "watcher.jpg",
            )
            write_watcher_status(
                WATCHER_STATUS_FILE,
                policy_version,
                action,
                watcher_telemetry(env, reward, recent_reward_events),
            )
            if terminated or truncated:
                observation, _ = env.reset()
                print("watcher reset to the true initial game state", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
