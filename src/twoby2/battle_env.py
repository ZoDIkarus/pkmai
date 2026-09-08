"""Compatibility shim — the real battle environment now lives in
``src/battle_env.py`` (top-level, isolated). This module re-exports it plus a
couple of contract markers that other twoby2 code / tests reference.
"""
from __future__ import annotations

from battle_env import (  # noqa: F401
    BattleEnv, SimulatedBattleDriver, BattleDriver, BattleRewardConfig,
    OBS_DIM, OBS_SCHEMA, default_scenario,
)
from battle_executor import ALL_MACROS as MACRO_ACTIONS, action_mask  # noqa: F401


def capture_is_reward_free():
    """Contract marker: capturing a scenario from a navigation run must never
    grant a reward to either PPO. Enforced in
    :meth:`twoby2.scenario_pool.ScenarioPool.capture` (no reward return path)
    and asserted by tests."""
    return True


class BattleEnvSpec:
    """Kept for callers that only want the static description. The runnable env
    is :class:`battle_env.BattleEnv`."""

    schema = OBS_SCHEMA
    macro_actions = MACRO_ACTIONS

    @staticmethod
    def action_mask(snapshot):
        """Fail-closed: ``None`` / unreadable snapshot -> all-zero vector,
        never a crash."""
        return action_mask(snapshot or {})

    @staticmethod
    def make_env(**kw):
        return BattleEnv(**kw)
