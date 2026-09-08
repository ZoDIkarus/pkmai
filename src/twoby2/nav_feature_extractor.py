"""Versioned custom feature extractor that adds the RAM coordinate-map branch
to the existing navigation policy WITHOUT disturbing the trained weights.

Contract: with a freshly-migrated policy, for *any* observation whose ``map``
channel is present, the action logits and value output are bit-identical to the
original policy's — because the widened first MLP layer has **zero columns** for
the map features. The map branch itself is normally initialised (so it is
actually trainable); its output is multiplied by those zero columns and
contributes nothing until training moves the columns off zero.

``OBS_SCHEMA_V1`` = image + nav (old).  ``OBS_SCHEMA_V2`` = + map (new).
"""
from __future__ import annotations

import numpy as np

OBS_SCHEMA_V1 = "nav_obs_v1"
OBS_SCHEMA_V2 = "nav_obs_v2_coordmap"

MAP_BRANCH_OUT = 32
EXTRACTOR_VERSION = 2


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_v2_observation_space(v1_space, *, map_channels, map_size=64):
    """v1 Dict{image,nav} -> v2 Dict{image,nav,map}."""
    from gymnasium import spaces
    d = dict(v1_space.spaces)
    d["map"] = spaces.Box(low=0, high=255,
                          shape=(int(map_channels), int(map_size), int(map_size)),
                          dtype=np.uint8)
    return spaces.Dict(d)


def make_map_branch(map_channels, map_size=64, out_dim=MAP_BRANCH_OUT):
    torch, nn = _torch()

    class MapBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(map_channels, 8, kernel_size=4, stride=4), nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, map_channels, map_size, map_size)
                flat = self.cnn(dummy).shape[1]
            self.proj = nn.Linear(flat, out_dim)
            # NOTE: the map branch is normally initialised. Equivalence with the
            # old policy is guaranteed by the *widened MLP first layer* having
            # zero columns for the map features (see ``migrate_state_dict``);
            # zero-initialising the branch too would make it un-trainable
            # (both gradient paths gated by zeros).

        def forward(self, x):
            return self.proj(self.cnn(x.float() / 255.0))

    return MapBranch()


def make_v2_extractor_class(map_channels, map_size=64, map_out=MAP_BRANCH_OUT):
    """Return a ``CombinedExtractor`` subclass that keeps SB3's exact submodule
    naming (``extractors.image`` / ``extractors.nav``) so the trained weights
    load verbatim, and appends a zero-init ``extractors.map`` branch LAST so the
    concatenated feature vector is ``[<old features>, <map features>]``.
    """
    torch, nn = _torch()
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import CombinedExtractor

    class NavMapCombinedExtractor(CombinedExtractor):
        version = EXTRACTOR_VERSION
        obs_schema = OBS_SCHEMA_V2

        def __init__(self, observation_space, cnn_output_dim=256,
                     normalized_image=False):
            # build image+nav exactly as stock SB3 would (map excluded)
            without_map = spaces.Dict(
                {k: v for k, v in observation_space.spaces.items() if k != "map"})
            super().__init__(without_map, cnn_output_dim=cnn_output_dim,
                             normalized_image=normalized_image)
            self._observation_space = observation_space
            self._old_dim = self._features_dim
            # append the map branch LAST
            self.extractors["map"] = make_map_branch(map_channels, map_size, map_out)
            self._features_dim = self._old_dim + map_out
            self._concat_order = [k for k in observation_space.spaces if k != "map"] + ["map"]

        def forward(self, observations):
            parts = [self.extractors[k](observations[k]) for k in self._concat_order]
            return torch.cat(parts, dim=1)

    return NavMapCombinedExtractor


# --------------------------------------------------------------------------
# weight migration
# --------------------------------------------------------------------------
_IMAGE_PREFIXES = (
    "features_extractor.extractors.image.",
    "pi_features_extractor.extractors.image.",
    "vf_features_extractor.extractors.image.",
)
_WIDENED_KEYS = ("mlp_extractor.policy_net.0.weight",
                 "mlp_extractor.value_net.0.weight")


def migrate_state_dict(old_sd, new_sd, *, old_feature_dim):
    """Return a new state dict for the v2 policy.

    * image-CNN + nav params copied verbatim (same names, same shapes);
    * ``mlp_extractor.{policy,value}_net.0.weight`` widened: old columns kept,
      new map columns set to 0;
    * every other shared param (mlp layer 2, action_net, value_net, log_std)
      copied verbatim;
    * the new ``*.map_branch.*`` params keep their zero init.
    Returns ``(migrated_sd, report)``.
    """
    torch, _ = _torch()
    migrated = {k: v.clone() for k, v in new_sd.items()}
    report = {"copied": 0, "widened": 0, "kept_new": 0, "missing_in_old": []}

    for name, new_tensor in new_sd.items():
        if ".extractors.map." in name or "map_branch" in name:
            report["kept_new"] += 1
            continue
        if name in _WIDENED_KEYS:
            old_w = old_sd.get(name)
            if old_w is None:
                report["missing_in_old"].append(name)
                continue
            w = torch.zeros_like(new_tensor)
            w[:, :old_w.shape[1]] = old_w
            migrated[name] = w
            report["widened"] += 1
            continue
        old_tensor = old_sd.get(name)
        if old_tensor is None:
            report["missing_in_old"].append(name)
            continue
        if old_tensor.shape != new_tensor.shape:
            report["missing_in_old"].append(f"{name}(shape {tuple(old_tensor.shape)}"
                                             f"!={tuple(new_tensor.shape)})")
            continue
        migrated[name] = old_tensor.clone()
        report["copied"] += 1

    report["old_feature_dim"] = old_feature_dim
    report["ok"] = not report["missing_in_old"]
    return migrated, report


def build_migrated_policy(old_policy, *, map_channels, map_size=64, device="cpu"):
    """Given a loaded SB3 ``MultiInputActorCriticPolicy`` (old obs = image+nav),
    build a v2 policy (obs += map) and migrate the weights. Returns
    ``(new_policy, report)``."""
    torch, _ = _torch()
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    v2_space = build_v2_observation_space(
        old_policy.observation_space, map_channels=map_channels, map_size=map_size)
    extractor_cls = make_v2_extractor_class(map_channels, map_size)
    new_policy = MultiInputActorCriticPolicy(
        v2_space, old_policy.action_space, lambda _: 3e-4,
        features_extractor_class=extractor_cls).to(device)

    migrated_sd, report = migrate_state_dict(
        old_policy.state_dict(), new_policy.state_dict(),
        old_feature_dim=old_policy.features_dim)
    missing, unexpected = new_policy.load_state_dict(migrated_sd, strict=False)
    # 'missing' should be empty; 'unexpected' likewise. map_branch keeps its
    # (zero) init and is not in `missing` because it IS in migrated_sd.
    report["load_missing"] = [m for m in missing]
    report["load_unexpected"] = [u for u in unexpected]
    report["new_feature_dim"] = new_policy.features_dim
    new_policy.eval()
    old_policy.eval()
    return new_policy, report


def logit_value_equivalence(old_policy, new_policy, obs_batch, *, atol=1e-4):
    """Max |Δ| of action logits and value output between the two policies for
    ``obs_batch`` (dict of np arrays; must contain 'image','nav' and 'map').
    Returns ``(ok, max_logit_diff, max_value_diff)``."""
    torch, _ = _torch()
    with torch.no_grad():
        o_old = {k: torch.as_tensor(v) for k, v in obs_batch.items()
                 if k in ("image", "nav")}
        o_new = {k: torch.as_tensor(v) for k, v in obs_batch.items()}

        def logits_values(policy, obs):
            feats = policy.extract_features(obs, policy.features_extractor) \
                if not policy.share_features_extractor else \
                policy.extract_features(obs)
            if isinstance(feats, tuple):
                pi_f, vf_f = feats
            else:
                pi_f = vf_f = feats
            lat_pi = policy.mlp_extractor.forward_actor(pi_f)
            lat_vf = policy.mlp_extractor.forward_critic(vf_f)
            return policy.action_net(lat_pi), policy.value_net(lat_vf)

        ol, ov = logits_values(old_policy, o_old)
        nl, nv = logits_values(new_policy, o_new)
        dl = (ol - nl).abs().max().item()
        dv = (ov - nv).abs().max().item()
    return (dl <= atol and dv <= atol), dl, dv


def migrate_navigation_policy(old_ppo_path, *, map_channels, map_size=64,
                              device="cpu"):
    """Load the existing navigation PPO checkpoint and return
    ``(new_policy, report, verify)`` where ``verify(obs_batch)`` checks
    logit/value equivalence against the original policy."""
    import warnings
    warnings.filterwarnings("ignore")
    from stable_baselines3 import PPO

    old = PPO.load(old_ppo_path, device=device)
    new_policy, report = build_migrated_policy(
        old.policy, map_channels=map_channels, map_size=map_size, device=device)

    def verify(obs_batch, atol=1e-4):
        return logit_value_equivalence(old.policy, new_policy, obs_batch, atol=atol)

    return new_policy, report, verify
