"""Central dynamic PPO learner: consumes uploads; never creates rollout workers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from dynamic_policy import PKMAIPolicy
from rollout_protocol import consume_rollouts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = PROJECT_ROOT / "runtime" / "cluster"
INBOX = CLUSTER_DIR / "rollout_inbox"
POLICY_FILE = CLUSTER_DIR / "policy.json"
MODEL_FILE = CLUSTER_DIR / "dynamic_policy.pt"
BEST_MODEL_FILE = CLUSTER_DIR / "dynamic_policy_best.pt"
BEST_SCORE_FILE = CLUSTER_DIR / "best_policy_score.json"
CHECKPOINTS_DIR = CLUSTER_DIR / "brain_checkpoints"
ACTION_EXPLORATION_FLOOR = min(
    0.35, max(0.0, float(os.getenv("PKMAI_ACTION_EXPLORATION_FLOOR", "0.35")))
)
MIN_STAGE_EVALUATION_EPISODES = 10
MIN_STAGE_SUCCESS_RATE = 0.60
EVALUATION_STAGE_KEYS = ("intro_complete", "stairs_down", "left_house", "starter")


def combine_rollouts(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Keep PPO updates broad enough that one tiny rollout cannot steer policy."""
    if not batches:
        raise ValueError("at least one rollout batch is required")
    return {
        name: np.concatenate([np.asarray(batch[name]) for batch in batches], axis=0)
        for name in batches[0]
    }


def load_best_mean_reward(path: Path = BEST_SCORE_FILE) -> tuple[int, float, float]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (int(value.get("successes", 0)), float(value.get("mean_reward", 0.0)), float(value.get("speed", 0.0)))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return (-1, float("-inf"), float("inf"))


def stage_gate_allows_promotion(candidate, baseline) -> bool:
    """Protect verified stages; missing episode evidence is not a pass."""
    for code in (1, 2, 3, 4):
        current = (candidate or {}).get(code)
        previous = (baseline or {}).get(str(code), (baseline or {}).get(code))
        if not previous:
            continue
        if (
            previous.get("samples", 0) < MIN_STAGE_EVALUATION_EPISODES
            or previous.get("success_rate", 0.0) < MIN_STAGE_SUCCESS_RATE
        ):
            continue
        if not current or current.get("samples", 0) < MIN_STAGE_EVALUATION_EPISODES:
            return False
        if current.get("success_rate", 0.0) < MIN_STAGE_SUCCESS_RATE:
            return False
    return True


def complete_evaluation_is_promotable(evaluation, candidate_version) -> bool:
    """Accept only complete held-out evidence for the exact frozen candidate."""
    if not isinstance(evaluation, dict) or evaluation.get("status") != "complete":
        return False
    if int(evaluation.get("policy_version", -1)) != int(candidate_version):
        return False
    episodes = int(evaluation.get("episodes_per_stage", 0) or 0)
    stages = evaluation.get("stages")
    if episodes < MIN_STAGE_EVALUATION_EPISODES or not isinstance(stages, dict):
        return False
    for stage in EVALUATION_STAGE_KEYS:
        result = stages.get(stage)
        if not isinstance(result, dict):
            return False
        if int(result.get("episodes", 0) or 0) < episodes:
            return False
        if float(result.get("success_rate", 0.0) or 0.0) < MIN_STAGE_SUCCESS_RATE:
            return False
    return True


def rollout_quality(batch: dict[str, np.ndarray], mean_reward: float) -> tuple[int, float, float]:
    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    if "objective_success" in batch:
        success_indices = np.flatnonzero(np.asarray(batch["objective_success"], dtype=np.bool_))
    else:
        success_indices = np.flatnonzero(rewards >= 100.0)
    speed = float(success_indices[0] + 1) if len(success_indices) else float("inf")
    return (int(len(success_indices)), float(mean_reward), -speed)


def rollout_stage_summary(batch: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    """Summarize completed objective episodes, never individual action steps."""
    codes = np.asarray(batch.get("objective_code", []), dtype=np.int8)
    successes = np.asarray(batch.get("objective_success", []), dtype=np.bool_)
    steps = np.asarray(batch.get("success_steps", []), dtype=np.int32)
    if not len(codes) or len(successes) != len(codes) or len(steps) != len(codes):
        return {}
    terminals = np.asarray(batch.get("dones", np.ones(len(codes), dtype=np.bool_)), dtype=np.bool_)
    if len(terminals) != len(codes):
        return {}
    result = {}
    for code in sorted(set(int(value) for value in codes[terminals])):
        mask = (codes == code) & terminals
        success_mask = mask & successes
        result[code] = {
            "samples": float(mask.sum()),
            "successes": float(success_mask.sum()),
            "success_rate": float(success_mask.sum() / max(1, mask.sum())),
            "median_success_steps": float(np.median(steps[success_mask])) if success_mask.any() else float("inf"),
        }
    return result


class DynamicLearner:
    def __init__(self, learning_rate: float = 3e-4) -> None:
        self.model = PKMAIPolicy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.version = 0
        self.timesteps = 0

    def restore_latest(self) -> bool:
        if not MODEL_FILE.is_file():
            return False
        artifact = torch.load(MODEL_FILE, map_location="cpu")
        self.model.load_state_dict(artifact["state_dict"])
        self.version = int(artifact["version"])
        try:
            self.timesteps = int(json.loads(POLICY_FILE.read_text(encoding="utf-8")).get("timesteps", 0))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.timesteps = 0
        return True

    def learn(self, batch: dict[str, np.ndarray]) -> dict[str, float | int]:
        images = torch.from_numpy(np.asarray(batch["images"]))
        nav = torch.from_numpy(np.asarray(batch["nav"], dtype=np.float32))
        actions = torch.from_numpy(np.asarray(batch["actions"], dtype=np.int64))
        rewards = torch.from_numpy(np.asarray(batch["rewards"], dtype=np.float32))
        dones = torch.from_numpy(np.asarray(batch["dones"], dtype=np.bool_))
        old_log_probs = torch.from_numpy(np.asarray(batch["log_probs"], dtype=np.float32))
        old_values = torch.from_numpy(np.asarray(batch["values"], dtype=np.float32))

        returns = torch.empty_like(rewards)
        running = torch.tensor(0.0)
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + 0.99 * running * (~dones[index])
            returns[index] = running
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        logits, values = self.model(images, nav)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = (
            (1.0 - ACTION_EXPLORATION_FLOOR) * probabilities
            + ACTION_EXPLORATION_FLOOR / logits.shape[-1]
        )
        distribution = Categorical(probs=probabilities)
        ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.minimum(ratio * advantages, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.version += 1
        self.timesteps += len(actions)
        return {
            "samples": len(actions),
            "loss": float(loss.detach()),
            "mean_reward": float(rewards.mean()),
        }

    def publish(self, checkpoint: str | None = None, best: bool = False, quality: tuple[int, float, float] | None = None, stage_summary: dict | None = None) -> None:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {"version": self.version, "state_dict": self.model.state_dict()}
        for model_file in (MODEL_FILE, BEST_MODEL_FILE) if best else (MODEL_FILE,):
            temporary_model = model_file.with_suffix(".pt.tmp")
            torch.save(artifact, temporary_model)
            os.replace(temporary_model, model_file)
        if best and quality is not None:
            temporary_score = BEST_SCORE_FILE.with_suffix(".json.tmp")
            temporary_score.write_text(json.dumps({"successes": quality[0], "mean_reward": quality[1], "speed": quality[2], "stage_summary": stage_summary or {} }), encoding="utf-8")
            os.replace(temporary_score, BEST_SCORE_FILE)
        payload = {
            "version": self.version,
            "timesteps": self.timesteps,
            "checkpoint": checkpoint,
            "updated_at": time.time(),
            "mode": "dynamic-rollout",
        }
        temporary_policy = POLICY_FILE.with_suffix(".json.tmp")
        temporary_policy.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary_policy, POLICY_FILE)


def main() -> None:
    checkpoint_every = max(1, int(os.getenv("PKMAI_CLUSTER_CHECKPOINT_EVERY", "50")))
    batches_per_update = max(2, int(os.getenv("PKMAI_CLUSTER_BATCHES_PER_UPDATE", "8")))
    learner = DynamicLearner()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_quality = load_best_mean_reward()
    try:
        best_stage_summary = json.loads(BEST_SCORE_FILE.read_text(encoding="utf-8")).get("stage_summary", {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        best_stage_summary = {}
    learner.restore_latest()
    learner.publish()
    pending_batches = []
    while True:
        consumed = 0
        for _, batch in consume_rollouts(INBOX, limit=8):
            pending_batches.append(batch)
            consumed += 1
            if len(pending_batches) < batches_per_update:
                continue
            combined = combine_rollouts(pending_batches)
            metrics = learner.learn(combined)
            pending_batches.clear()
            checkpoint = None
            if learner.version % checkpoint_every == 0:
                checkpoint_path = CHECKPOINTS_DIR / f"dynamic-v{learner.version:08d}.pt"
                torch.save({"version": learner.version, "state_dict": learner.model.state_dict()}, checkpoint_path)
                checkpoint = str(checkpoint_path)
            quality = rollout_quality(combined, float(metrics["mean_reward"]))
            stage_summary = rollout_stage_summary(combined)
            is_best = quality > best_quality and stage_gate_allows_promotion(stage_summary, best_stage_summary)
            if is_best:
                best_quality = quality
                best_stage_summary = stage_summary
            learner.publish(checkpoint, best=is_best, quality=quality if is_best else None, stage_summary=stage_summary if is_best else None)
            print(json.dumps({"policy_version": learner.version, "timesteps": learner.timesteps}), flush=True)
        if not consumed:
            time.sleep(0.25)


if __name__ == "__main__":
    main()
