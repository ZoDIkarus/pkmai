"""Hard isolation between the Navigation and Battle systems.

The two systems share *nothing* that carries gradient, reward or progress:
separate rollout buffers, separate optimizers, separate step/metric counters,
separate model files. Navigation may only receive a battle *summary*
(:mod:`twoby2.battle_summary`); the Battle system may only *read* captured
scenarios.

These helpers make the isolation testable: a battle step that mutates any
navigation counter, or a battle worker that can address a navigation model
path, is a bug that fails a test here.
"""
from __future__ import annotations

import copy
import re

# --- model-path ownership ------------------------------------------------
_NAV_PATH_PAT = re.compile(
    r"(navigation_(learner|champion|candidate)|pokemon_model_(latest|champion|resume|candidate)"
    r"|nav_optimizer|nav_horizon|curriculum|exploration_memory|model_version)",
    re.I)
_BATTLE_PATH_PAT = re.compile(
    r"(battle_(learner|champion|candidate)|battle_optimizer|battle_scenarios"
    r"|battle_rollout)", re.I)


class ModelPathGuard:
    """Which system may write which file. Fail-closed: if a path looks like it
    belongs to the *other* system, the write is refused."""

    @staticmethod
    def battle_may_write(path):
        p = str(path)
        if _NAV_PATH_PAT.search(p):
            return False
        return bool(_BATTLE_PATH_PAT.search(p)) or "battle" in p.lower()

    @staticmethod
    def navigation_may_write(path):
        p = str(path)
        if _BATTLE_PATH_PAT.search(p):
            return False
        return bool(_NAV_PATH_PAT.search(p)) or "nav" in p.lower()


# --- per-system counters ----------------------------------------------
_NAV_COUNTERS = (
    "nav_env_steps", "nav_ppo_updates", "nav_rollout_samples",
    "nav_episodes", "world_stage", "mastered_stage", "tiles_seen",
    "warps_used", "frontier_score", "story_progress", "nav_reward_total",
)
_BATTLE_COUNTERS = (
    "battle_env_steps", "battle_ppo_updates", "battle_rollout_samples",
    "battle_episodes", "battle_wins", "battle_kos", "battle_reward_total",
)


class SystemCounters:
    def __init__(self, keys):
        self._keys = tuple(keys)
        self._c = {k: 0 for k in self._keys}

    def add(self, key, n=1):
        if key not in self._c:
            raise KeyError(f"{key} is not a counter of this system")
        self._c[key] += n

    def set(self, key, v):
        if key not in self._c:
            raise KeyError(f"{key} is not a counter of this system")
        self._c[key] = v

    def snapshot(self):
        return dict(self._c)

    def get(self, key):
        return self._c[key]


def navigation_counters():
    return SystemCounters(_NAV_COUNTERS)


def battle_counters():
    return SystemCounters(_BATTLE_COUNTERS)


# --- isolated system handles ----------------------------------------
class IsolatedSystem:
    """A minimal stand-in for one training system: its own counters, its own
    rollout buffer object, its own optimizer token."""

    def __init__(self, name, counters):
        self.name = name
        self.counters = counters
        self.rollout_buffer = []          # distinct list object per system
        self.optimizer_token = object()   # identity-only optimizer marker
        self.model_paths = set()

    def record_rollout(self, sample):
        self.rollout_buffer.append(sample)
        self.counters.add(f"{'nav' if self.name == 'navigation' else 'battle'}_rollout_samples", 1)


def assert_systems_isolated(nav, battle):
    """Raise AssertionError if the two systems share a buffer or optimizer."""
    if nav.rollout_buffer is battle.rollout_buffer:
        raise AssertionError("navigation and battle share a rollout buffer")
    if nav.optimizer_token is battle.optimizer_token:
        raise AssertionError("navigation and battle share an optimizer")
    if nav.counters is battle.counters:
        raise AssertionError("navigation and battle share a counter set")
    return True


class NavigationImmutableDuringBattle:
    """Context manager: snapshot navigation counters + rollout length on enter,
    assert they are byte-identical on exit. Wrap any battle-side work with it in
    tests and at the seam."""

    def __init__(self, nav_system):
        self.nav = nav_system
        self._counters = None
        self._rollout_len = None
        self._buffer_copy = None

    def __enter__(self):
        self._counters = self.nav.counters.snapshot()
        self._rollout_len = len(self.nav.rollout_buffer)
        self._buffer_copy = copy.deepcopy(self.nav.rollout_buffer)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        now = self.nav.counters.snapshot()
        if now != self._counters:
            raise AssertionError(
                f"battle work mutated navigation counters: "
                f"{_diff(self._counters, now)}")
        if len(self.nav.rollout_buffer) != self._rollout_len:
            raise AssertionError(
                "battle work appended to the navigation rollout buffer")
        if self.nav.rollout_buffer != self._buffer_copy:
            raise AssertionError("battle work modified navigation rollout contents")
        return False


def _diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
