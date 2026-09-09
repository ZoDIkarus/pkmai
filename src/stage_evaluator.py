"""Independent fixed-seed stage evaluation for published dynamic policies."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dynamic_policy import PKMAIPolicy
from pokemon_env import PokemonFireRedEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = PROJECT_ROOT / "runtime" / "cluster"
MODEL_FILE = CLUSTER_DIR / "dynamic_policy.pt"
RESULT_FILE = CLUSTER_DIR / "stage_evaluation.json"
STAGES = (
    ("intro_complete", "beginning", "intro"),
    ("stairs_down", "intro_complete", "stairs"),
    ("left_house", "stairs_down", "exit"),
    ("starter", "left_house", "starter"),
)
STAGE_STEP_LIMITS = {
    "intro_complete": 256,
    "stairs_down": 640,
    "left_house": 768,
    "starter": 1024,
}


def stage_step_limit(stage):
    """Bound held-out episodes without changing training horizons."""
    return int(STAGE_STEP_LIMITS[str(stage)])


def summarize_stage_results(results):
    summary = {}
    for stage in sorted({str(row["stage"]) for row in results}):
        rows = [row for row in results if row["stage"] == stage]
        successful = [int(row["steps"]) for row in rows if row["success"]]
        summary[stage] = {
            "episodes": len(rows),
            "successes": len(successful),
            "success_rate": len(successful) / max(1, len(rows)),
            "median_success_steps": float(np.median(successful)) if successful else None,
        }
    return summary


def load_policy():
    artifact = torch.load(MODEL_FILE, map_location="cpu")
    policy = PKMAIPolicy()
    policy.load_state_dict(artifact["state_dict"])
    policy.eval()
    return policy, int(artifact["version"])


def write_result(payload):
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, RESULT_FILE)


def evaluate_stage(policy, stage, start, objective, seed, episodes, on_result=None):
    results = []
    for episode in range(episodes):
        env = PokemonFireRedEnv(
            rank=1000 + episode,
            agent_count=1,
            episode_start_override=start,
            training_objective_override=objective,
        )
        try:
            observation, _ = env.reset(seed=seed + episode)
            generator = torch.Generator().manual_seed(seed + episode)
            for _ in range(stage_step_limit(stage)):
                image = torch.from_numpy(np.asarray(observation["image"], dtype=np.uint8))[None, ...]
                nav = torch.from_numpy(np.asarray(observation["nav"], dtype=np.float32))[None, ...]
                with torch.no_grad():
                    logits, _ = policy(image, nav)
                    action = int(torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).item())
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    result = {"stage": stage, "success": bool(info.get("objective_success", False)), "steps": int(info.get("episode_steps", env.total_steps))}
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                    break
            else:
                result = {
                    "stage": stage,
                    "success": False,
                    "steps": stage_step_limit(stage),
                    "truncation_reason": "evaluator_step_limit",
                }
                results.append(result)
                if on_result is not None:
                    on_result(result)
        finally:
            env.close()
    return results


def main():
    episodes = max(1, int(os.getenv("PKMAI_STAGE_EVAL_EPISODES", "10")))
    seed = int(os.getenv("PKMAI_STAGE_EVAL_SEED", "1729"))
    last_version = -1
    while True:
        try:
            policy, version = load_policy()
            if version != last_version:
                rows = []
                payload = {"policy_version": version, "seed": seed, "episodes_per_stage": episodes, "status": "running", "stages": {}}
                write_result(payload)
                for index, (stage, start, objective) in enumerate(STAGES):
                    def publish_partial(result):
                        rows.append(result)
                        payload["stages"] = summarize_stage_results(rows)
                        payload["updated_at"] = time.time()
                        write_result(payload)
                    evaluate_stage(policy, stage, start, objective, seed + index * 1000, episodes, publish_partial)
                    stage_summary = payload["stages"].get(stage, {})
                    if stage_summary.get("episodes", 0) >= 3 and stage_summary.get("successes", 0) == 0:
                        payload["status"] = "blocked"
                        payload["blocked_stage"] = stage
                        payload["updated_at"] = time.time()
                        write_result(payload)
                        break
                else:
                    payload["status"] = "complete"
                    payload["updated_at"] = time.time()
                    write_result(payload)
                print(json.dumps({"evaluated_policy_version": version}), flush=True)
                last_version = version
        except (OSError, RuntimeError, KeyError) as exc:
            print(f"stage evaluator waiting: {exc}", flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
