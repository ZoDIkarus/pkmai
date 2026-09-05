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
CHECKPOINTS_DIR = CLUSTER_DIR / "brain_checkpoints"


class DynamicLearner:
    def __init__(self, learning_rate: float = 3e-4) -> None:
        self.model = PKMAIPolicy()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.version = 0
        self.timesteps = 0

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
        distribution = Categorical(logits=logits)
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

    def publish(self, checkpoint: str | None = None, best: bool = False) -> None:
        CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
        artifact = {"version": self.version, "state_dict": self.model.state_dict()}
        for model_file in (MODEL_FILE, BEST_MODEL_FILE) if best else (MODEL_FILE,):
            temporary_model = model_file.with_suffix(".pt.tmp")
            torch.save(artifact, temporary_model)
            os.replace(temporary_model, model_file)
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
    learner = DynamicLearner()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_mean_reward = float("-inf")
    learner.publish(best=True)
    while True:
        consumed = 0
        for _, batch in consume_rollouts(INBOX, limit=8):
            metrics = learner.learn(batch)
            consumed += 1
            checkpoint = None
            if learner.version % checkpoint_every == 0:
                checkpoint_path = CHECKPOINTS_DIR / f"dynamic-v{learner.version:08d}.pt"
                torch.save({"version": learner.version, "state_dict": learner.model.state_dict()}, checkpoint_path)
                checkpoint = str(checkpoint_path)
            is_best = float(metrics["mean_reward"]) >= best_mean_reward
            if is_best:
                best_mean_reward = float(metrics["mean_reward"])
            learner.publish(checkpoint, best=is_best)
            print(json.dumps({"policy_version": learner.version, "timesteps": learner.timesteps}), flush=True)
        if not consumed:
            time.sleep(0.25)


if __name__ == "__main__":
    main()
