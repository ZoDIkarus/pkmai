#!/usr/bin/env python3
"""Central RLlib PPO learner. Remote Ray nodes provide rollout environments."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from cluster_config import build_environment_signature
from pokemon_env import PokemonFireRedEnv
from gymnasium import spaces
from rllib_model import PKMAIDictCNN

class ClusteredPokemonEnv(PokemonFireRedEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8),
            "nav": spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32),
        })

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = PROJECT_ROOT / "runtime" / "cluster"
CHECKPOINTS_DIR = CLUSTER_DIR / "brain_checkpoints"
POLICY_FILE = CLUSTER_DIR / "policy.json"


def persist_checkpoint(checkpoint, target: Path) -> str:
    temporary = target.parent / f".{target.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    checkpoint.to_directory(str(temporary))
    os.replace(temporary, target)
    return str(target)


def make_cluster_env(config: dict):
    return ClusteredPokemonEnv(
        rank=int(config.get("rank", 0)),
        agent_count=int(config.get("agent_count", 1)),
    )


def publish_policy(
    version: int, checkpoint: str | None = None, timesteps: int = 0
) -> None:
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "checkpoint": checkpoint,
        "timesteps": timesteps,
        "signature": build_environment_signature(
            observation_shape=(64, 64, 1), nav_features=28, action_count=7
        ),
        "updated_at": time.time(),
    }
    temporary = POLICY_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, POLICY_FILE)


def main() -> None:
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    env_runners = max(1, int(os.getenv("PKMAI_CLUSTER_ENV_RUNNERS", "64")))
    checkpoint_every = max(1, int(os.getenv("PKMAI_CLUSTER_CHECKPOINT_EVERY", "10")))
    register_env("pkmai_cluster_env", make_cluster_env)
    ModelCatalog.register_custom_model("pkmai_dict_cnn", PKMAIDictCNN)
    ray.init(address=os.getenv("RAY_ADDRESS", "auto"), ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("pkmai_cluster_env", env_config={"agent_count": 1})
        .framework("torch")
        .training(model={"custom_model": "pkmai_dict_cnn"})
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=env_runners,
            num_envs_per_env_runner=1,
            sample_timeout_s=30,
            create_env_on_local_worker=False,
        )
    )

    config.train_batch_size = 128
    config.minibatch_size = 64
    config.num_epochs = 1
    config.ignore_env_runner_failures = True
    config.restart_failed_env_runners = True
    algorithm = config.build()
    version = 0
    last_checkpoint = None
    publish_policy(version, last_checkpoint, timesteps=0)
    try:
        while True:
            result = algorithm.train()
            version += 1
            if version % checkpoint_every == 0:
                target = CHECKPOINTS_DIR / (
                    f"policy-v{version:08d}-{int(time.time())}"
                )
                checkpoint_result = algorithm.save()
                if checkpoint_result.checkpoint is None:
                    raise RuntimeError("RLlib returned no checkpoint")
                last_checkpoint = persist_checkpoint(checkpoint_result.checkpoint, target)
            timesteps = int(result.get("num_env_steps_sampled_lifetime", 0) or 0)
            publish_policy(version, last_checkpoint, timesteps=timesteps)
            print(json.dumps({"policy_version": version, "timesteps": timesteps}), flush=True)
    finally:
        algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
