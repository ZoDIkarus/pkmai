#!/usr/bin/env python3
"""Rebase navigation learner files on champion weights with fresh counters."""
from __future__ import annotations

import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from stable_baselines3 import PPO

CKPT = os.path.join(ROOT, "runtime", "navigation", "checkpoints")
CHAMPION = os.path.join(CKPT, "navigation_champion.zip")
TARGETS = ("navigation_learner.zip", "navigation_candidate.zip",
           "navigation_resume.zip")


def policy_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    champion = PPO.load(CHAMPION, device="cpu")
    expected = policy_digest(champion)
    champion.num_timesteps = 0
    champion._episode_num = 0
    champion._n_updates = 0
    os.makedirs(CKPT, exist_ok=True)
    for name in TARGETS:
        path = os.path.join(CKPT, name)
        tmp_zip = path[:-4] + f".{os.getpid()}.tmp.zip"
        champion.save(tmp_zip)
        check = PPO.load(tmp_zip, device="cpu")
        if check.num_timesteps != 0 or policy_digest(check) != expected:
            os.unlink(tmp_zip)
            raise RuntimeError(f"verification failed for {name}")
        os.replace(tmp_zip, path)
        print(f"{name}: weights={expected[:16]} steps=0")


if __name__ == "__main__":
    main()
