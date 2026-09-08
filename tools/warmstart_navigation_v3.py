#!/usr/bin/env python3
"""Warm-start the 36-wide navigation learner from a proven 31-wide policy.

The five directed-navigation inputs are appended to the existing ``nav``
vector.  Copying the old first-layer columns and zero-initialising only those
five new columns therefore preserves the old policy exactly while leaving the
new inputs trainable.  Dry-run is the default; ``--apply`` atomically replaces
only the fresh v3 learner checkpoint after making a backup.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(
    ROOT, "runtime", "navigation", "backups", "20260908_120710",
    "navigation_champion.zip",
)
DEFAULT_TARGET = os.path.join(
    ROOT, "runtime", "navigation", "checkpoints", "navigation_learner.zip",
)
WIDENED_KEYS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.value_net.0.weight",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def migrate_policy_state(old_policy, new_policy):
    """Return a v3 state dict with old behaviour and five zero new columns."""
    old = old_policy.state_dict()
    new = new_policy.state_dict()
    migrated = {}
    widened = {}
    for name, new_tensor in new.items():
        old_tensor = old.get(name)
        if name in WIDENED_KEYS:
            if old_tensor is None or old_tensor.ndim != 2:
                raise RuntimeError(f"missing old widened tensor: {name}")
            if new_tensor.shape[0] != old_tensor.shape[0]:
                raise RuntimeError(f"row mismatch for {name}")
            extra = int(new_tensor.shape[1] - old_tensor.shape[1])
            if extra != 5:
                raise RuntimeError(
                    f"{name}: expected exactly five appended nav inputs, got {extra}")
            out = torch.zeros_like(new_tensor)
            out[:, : old_tensor.shape[1]] = old_tensor
            migrated[name] = out
            widened[name] = extra
        elif old_tensor is not None and old_tensor.shape == new_tensor.shape:
            migrated[name] = old_tensor.clone()
        elif old_tensor is None:
            raise RuntimeError(f"new parameter has no old counterpart: {name}")
        else:
            raise RuntimeError(
                f"unexpected shape change for {name}: "
                f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}")
    return migrated, widened


def verify_equivalence(old_model, new_model):
    """Verify logits and values are unchanged when new inputs are zero."""
    rng = np.random.default_rng(20260908)
    old_obs = {
        "image": rng.integers(0, 256, size=(4, 4, 64, 64), dtype=np.uint8),
        "nav": rng.uniform(-1, 1, size=(4, 31)).astype(np.float32),
    }
    new_obs = {
        "image": old_obs["image"].copy(),
        "nav": np.concatenate(
            [old_obs["nav"], np.zeros((4, 5), dtype=np.float32)], axis=1),
    }

    def outputs(model, obs):
        policy = model.policy
        converted, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            features = policy.extract_features(converted)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            return policy.action_net(latent_pi), policy.value_net(latent_vf)

    old_logits, old_values = outputs(old_model, old_obs)
    new_logits, new_values = outputs(new_model, new_obs)
    logit_diff = float((old_logits - new_logits).abs().max().cpu())
    value_diff = float((old_values - new_values).abs().max().cpu())
    return {
        "ok": logit_diff <= 1e-6 and value_diff <= 1e-6,
        "max_logit_diff": logit_diff,
        "max_value_diff": value_diff,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    source = os.path.realpath(args.source)
    target = os.path.realpath(args.target)
    if not os.path.isfile(source) or not os.path.isfile(target):
        raise SystemExit(f"missing source or target: {source} / {target}")

    old_model = PPO.load(source, device="cpu")
    new_model = PPO.load(target, device="cpu")
    old_dim = int(old_model.observation_space["nav"].shape[0])
    new_dim = int(new_model.observation_space["nav"].shape[0])
    if (old_dim, new_dim) != (31, 36):
        raise SystemExit(f"expected nav 31 -> 36, got {old_dim} -> {new_dim}")
    if old_model.action_space != new_model.action_space:
        raise SystemExit("navigation action spaces differ")

    migrated, widened = migrate_policy_state(old_model.policy, new_model.policy)
    new_model.policy.load_state_dict(migrated, strict=True)
    equivalence = verify_equivalence(old_model, new_model)
    if not equivalence["ok"]:
        raise SystemExit(f"policy equivalence failed: {equivalence}")

    report = {
        "source": source,
        "source_sha256": _sha256(source),
        "target": target,
        "target_sha256_before": _sha256(target),
        "nav_dims": [old_dim, new_dim],
        "widened": widened,
        "equivalence": equivalence,
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(report)
        return 0

    backup_dir = os.path.join(
        os.path.dirname(os.path.dirname(target)), "backups",
        time.strftime("warmstart_%Y%m%d_%H%M%S"),
    )
    os.makedirs(backup_dir, exist_ok=False)
    backup = os.path.join(backup_dir, os.path.basename(target))
    shutil.copy2(target, backup)

    # Preserve the fresh run's counters/optimizer schedule, replacing weights
    # only.  This is a warm start, not a resurrection of the 18M-step counter.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(target)) as tmpdir:
        tmp = os.path.join(tmpdir, "navigation_learner.zip")
        new_model.save(tmp)
        check = PPO.load(tmp, device="cpu")
        checked = verify_equivalence(old_model, check)
        if not checked["ok"] or int(check.observation_space["nav"].shape[0]) != 36:
            raise RuntimeError(f"saved warm start failed verification: {checked}")
        os.replace(tmp, target)

    report["backup"] = backup
    report["target_sha256_after"] = _sha256(target)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
