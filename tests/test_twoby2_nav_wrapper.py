import warnings
warnings.filterwarnings("ignore")

import unittest

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from twoby2.nav_wrapper import NavigationBattleWrapper, BattleDriver


class _FakeNavEnv(gym.Env):
    """Minimal nav env: overworld steps advance ``route_steps``; action 1
    'walks into grass' and the next step is a battle."""
    observation_space = spaces.Box(-1, 1, (4,), np.float32)
    action_space = spaces.Discrete(3)

    class _U:
        route_steps = 0

    def __init__(self):
        self.unwrapped_state = _FakeNavEnv._U()
        self._battle_pending = False

    @property
    def unwrapped(self):
        return self.unwrapped_state

    def reset(self, *, seed=None, options=None):
        self.unwrapped_state.route_steps = 0
        self._battle_pending = False
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.unwrapped_state.route_steps += 1
        obs = np.full(4, 0.1, np.float32)
        if action == 1:
            self._battle_pending = True
        return obs, 0.2, False, False, {"pre_battle": True}


class _FakeDriver(BattleDriver):
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def in_battle(self, env):
        return getattr(env, "_battle_pending", False)

    def play_battle(self, env):
        self.calls += 1
        env._battle_pending = False
        # a real driver would advance the emulator; it must NOT touch route_steps
        return dict(self.result)


BATTLE_WIN = {
    "outcome": "win", "party_hp_lost": 8, "party_hp_fraction_lost": 0.2,
    "own_faints": 0, "enemy_faints": 1, "turns": 4, "duration_steps": 120,
    "post_battle_obs": np.full(4, 0.9, np.float32),
    "resulting_world_state": {"map_group": 3, "map_id": 0, "x": 5, "y": 6},
}
BATTLE_WIPE = {**BATTLE_WIN, "outcome": "wipe", "party_hp_fraction_lost": 1.0,
               "blackout": True, "terminated": True,
               "resulting_world_state": {"map_group": 3, "map_id": 1, "x": 1, "y": 1}}


class NavWrapperTests(unittest.TestCase):
    def test_plain_overworld_step_is_untouched(self):
        w = NavigationBattleWrapper(_FakeNavEnv(), _FakeDriver(BATTLE_WIN))
        w.reset()
        obs, r, term, trunc, info = w.step(0)     # no battle
        self.assertAlmostEqual(r, 0.2)
        self.assertNotIn("battle_summary", info)
        self.assertEqual(w.env.unwrapped.route_steps, 1)

    def test_battle_subepisode_does_not_advance_route_steps(self):
        drv = _FakeDriver(BATTLE_WIN)
        w = NavigationBattleWrapper(_FakeNavEnv(), drv)
        w.reset()
        obs, r, term, trunc, info = w.step(1)      # walk into grass -> battle
        self.assertEqual(drv.calls, 1)
        # the triggering step is ONE real overworld step; the battle adds none
        self.assertEqual(w.env.unwrapped.route_steps, 1)
        self.assertEqual(w.total_battle_steps, 120)

    def test_navigation_gets_only_a_safe_summary_and_a_coarse_reward(self):
        w = NavigationBattleWrapper(_FakeNavEnv(), _FakeDriver(BATTLE_WIN))
        w.reset()
        obs, r, term, trunc, info = w.step(1)
        s = info["battle_summary"]
        self.assertEqual(set(s) - _ALLOWED, set())
        self.assertEqual(s["outcome"], "win")
        self.assertLessEqual(r, 0.2)   # overworld 0.2 + strategic (win -> <= 0)
        self.assertEqual(info["semi_mdp"]["kind"], "battle_subepisode")
        self.assertEqual(info["semi_mdp"]["duration_steps"], 120)

    def test_post_battle_observation_is_the_real_one(self):
        w = NavigationBattleWrapper(_FakeNavEnv(), _FakeDriver(BATTLE_WIN))
        w.reset()
        obs, *_ = w.step(1)
        self.assertTrue(np.allclose(obs, 0.9))     # post-battle obs, not 0.1

    def test_wipe_terminates_and_returns_respawn_state(self):
        w = NavigationBattleWrapper(_FakeNavEnv(), _FakeDriver(BATTLE_WIPE))
        w.reset()
        obs, r, term, trunc, info = w.step(1)
        self.assertTrue(term)
        self.assertEqual(info["battle_summary"]["resulting_world_state"]["map_id"], 1)
        self.assertLess(r, 0)

    def test_leaky_battle_result_is_stripped_before_navigation(self):
        leaky = {**BATTLE_WIN, "per_turn_reward": [0.1, 0.2]}
        w = NavigationBattleWrapper(_FakeNavEnv(), _FakeDriver(leaky))
        w.reset()
        obs, r, term, trunc, info = w.step(1)
        self.assertNotIn("per_turn_reward", info["battle_summary"])


_ALLOWED = {
    "outcome", "party_hp_lost", "party_hp_fraction_lost", "party_hp_total_after",
    "pp_remaining_total", "pp_spent", "items_used", "own_faints", "enemy_faints",
    "turns", "fled", "blackout", "resulting_world_state", "battle_kind",
    "duration_steps",
    # Catch-v2 strategic projection (spec §3)
    "objective_mode", "catch_requested", "catch_success", "caught_species_id",
    "target_species_id", "is_new_species_this_run",
}


if __name__ == "__main__":
    unittest.main()
