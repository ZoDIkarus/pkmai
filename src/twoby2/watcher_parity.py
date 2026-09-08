"""Full-Watcher / Full-Worker parity (2×2).

The normal watcher stays **inference-only** (no rollouts, no optimizer, no
learning counters), but in the new path it must run through the *exact same
central components* as a full navigation worker — one implementation, not a
copied second one:

  protected post-parcel master · action map · hold/release frames · observation
  · NavCoordMap · reset baseline · reward components (display only) ·
  terminated/truncated · NavigationBattleWrapper · battle router · blackout /
  recovery.

This module defines the shared component bundle and a deterministic parity
check: two copies of the same state, identical seed, identical action sequence
-> identical observation / reward components / position / story flags / HP-PP /
battle detection / terminated-truncated / world state.
"""
from __future__ import annotations

# The canonical list of components a full worker and the full watcher MUST
# share (same object/spec, never a re-implementation).
SHARED_COMPONENTS = (
    "protected_post_parcel_master",
    "action_map",
    "hold_release_frames",
    "observation_builder",
    "nav_coord_map",
    "reset_baseline",
    "reward_components",
    "termination_truncation",
    "navigation_battle_wrapper",
    "battle_router",
    "blackout_recovery",
)

# The ONLY things the watcher differs in.
WATCHER_DIFFERENCES = ("no_rollouts", "no_optimizer", "no_learning_counters",
                       "rewards_are_display_only")


class WatcherRole:
    """Marks an env instance as watcher (inference-only). The shared components
    are identical; this only flips the learning behaviour off."""

    def __init__(self):
        self.is_watcher = True
        self.collects_rollouts = False
        self.has_optimizer = False
        self.updates_learning_counters = False
        self.rewards_are_display_only = True


class FullRole:
    def __init__(self):
        self.is_watcher = False
        self.collects_rollouts = True
        self.has_optimizer = True
        self.updates_learning_counters = True
        self.rewards_are_display_only = False


# --------------------------------------------------------------------------
# deterministic parity check
# --------------------------------------------------------------------------
PARITY_FIELDS = (
    "observation", "reward_components", "position", "story_flags",
    "hp_pp", "battle_detected", "terminated", "truncated", "world_state",
)


def step_trace(env_like, actions):
    """Run ``actions`` on a full-worker-shaped env and record the parity fields
    per step. ``env_like`` must expose ``reset(seed=...)`` and ``step(a)``
    returning ``(obs, reward_components, terminated, truncated, info)`` where
    ``info`` carries ``position / story_flags / hp_pp / battle_detected /
    world_state``.
    """
    trace = []
    env_like.reset(seed=0)
    for a in actions:
        obs, rc, term, trunc, info = env_like.step(a)
        trace.append({
            "observation": _hashable(obs),
            "reward_components": _hashable(rc),
            "position": info.get("position"),
            "story_flags": tuple(sorted(info.get("story_flags") or ())),
            "hp_pp": _hashable(info.get("hp_pp")),
            "battle_detected": bool(info.get("battle_detected")),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "world_state": _hashable(info.get("world_state")),
        })
        if term or trunc:
            break
    return trace


def assert_parity(full_trace, watcher_trace):
    """Raise AssertionError on the first differing step/field. A watcher and a
    full worker must be bit-identical on everything except learning."""
    if len(full_trace) != len(watcher_trace):
        raise AssertionError(
            f"trace length differs: full {len(full_trace)} vs watcher {len(watcher_trace)}")
    for i, (f, w) in enumerate(zip(full_trace, watcher_trace)):
        for field in PARITY_FIELDS:
            if f.get(field) != w.get(field):
                raise AssertionError(
                    f"parity break at step {i}, field {field!r}: "
                    f"{f.get(field)!r} != {w.get(field)!r}")
    return True


def _hashable(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return ("ndarray", x.shape, hash(x.tobytes()))
    except Exception:
        pass
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    return x
