"""The gated 2x2 live seam: real wiring, no-op while gates are OFF, and the
navigation/battle separation it provides."""
import warnings
warnings.filterwarnings("ignore")

import unittest
from unittest import mock

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from twoby2 import live_integration as li
from twoby2 import FEATURES


class _NavEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (4,), np.float32)
    action_space = spaces.Discrete(3)

    class _U:
        route_steps = 0

    def __init__(self):
        self.unwrapped_state = _NavEnv._U()
        self._battle = False

    @property
    def unwrapped(self):
        return self.unwrapped_state

    def reset(self, *, seed=None, options=None):
        self.unwrapped_state.route_steps = 0
        self._battle = False
        return np.zeros(4, np.float32), {}

    def step(self, a):
        self.unwrapped_state.route_steps += 1
        if a == 1:
            self._battle = True
        return np.full(4, 0.2, np.float32), 0.3, False, False, {}


class _FakeEmuDriver:
    """stand-in for EmulatorBattleDriver used by the adapter"""
    def __init__(self, env):
        self.env = env
        self.max_turns = 80

    def in_battle(self):
        return getattr(self.env, "_battle", False)

    def play_battle(self, policy_fn, obs_fn):
        self.env._battle = False
        return {"outcome": "win", "turns": 3, "duration_steps": 90,
                "party_hp_fraction_lost": 0.1}


class GatedSeamTests(unittest.TestCase):
    def test_maybe_wrap_is_a_noop_while_gate_off(self):
        self.assertFalse(FEATURES["nav_battle_wrapper"])
        env = _NavEnv()
        out = li.maybe_wrap_full_agent(env, learning=True, battle_policy=lambda o: 0)
        self.assertIs(out, env)          # unchanged

    def test_battle_driver_for_defaults_to_simulated_while_gate_off(self):
        from battle_env import SimulatedBattleDriver
        self.assertIsInstance(li.battle_driver_for(None), SimulatedBattleDriver)

    def test_battle_driver_for_refuses_simulated_in_live_path(self):
        with mock.patch.object(li, "feature_enabled", return_value=True):
            with self.assertRaises(li.LiveIntegrationBlocked):
                li.battle_driver_for(None, allow_simulated_fallback=False)

    def test_gate_on_requires_a_real_battle_policy(self):
        with mock.patch.object(li, "feature_enabled", return_value=True):
            with self.assertRaises(li.LiveIntegrationBlocked):
                li.maybe_wrap_full_agent(_NavEnv(), battle_policy=None)

    def test_gate_on_wraps_with_navigation_battle_wrapper_and_real_driver(self):
        with mock.patch.object(li, "feature_enabled", return_value=True):
            env = _NavEnv()
            wrapped = li.maybe_wrap_full_agent(
                env, learning=True, battle_policy=lambda o: 0,
                emulator_driver_factory=lambda e: _FakeEmuDriver(e))
            from twoby2.nav_wrapper import NavigationBattleWrapper
            self.assertIsInstance(wrapped, NavigationBattleWrapper)
            wrapped.reset()
            # a normal overworld step
            obs, r, term, trunc, info = wrapped.step(0)
            self.assertEqual(env.unwrapped.route_steps, 1)
            self.assertNotIn("battle_summary", info)
            # action 1 triggers a battle -> real driver plays it, route_steps NOT bumped by battle
            obs, r, term, trunc, info = wrapped.step(1)
            self.assertEqual(env.unwrapped.route_steps, 2)   # only the 2 nav steps
            self.assertIn("battle_summary", info)
            s = info["battle_summary"]
            # navigation gets only the coarse summary; no per-turn / KO / win term
            self.assertLessEqual(set(s), {
                "outcome", "party_hp_lost", "party_hp_fraction_lost",
                "party_hp_total_after", "pp_remaining_total", "pp_spent",
                "items_used", "own_faints", "enemy_faints", "turns", "fled",
                "blackout", "resulting_world_state", "battle_kind", "duration_steps",
                "objective_mode", "catch_requested", "catch_success",
                "caught_species_id", "target_species_id", "is_new_species_this_run"})
            self.assertEqual(info["semi_mdp"]["duration_steps"], 90)

    def test_watcher_variant_marks_learning_false(self):
        with mock.patch.object(li, "feature_enabled", return_value=True):
            wrapped = li.maybe_wrap_full_agent(
                _NavEnv(), learning=False, battle_policy=lambda o: 0,
                emulator_driver_factory=lambda e: _FakeEmuDriver(e))
            self.assertFalse(wrapped._twoby2_learning)

    def test_integration_status_reports_no_simulated_driver_in_live_path(self):
        st = li.integration_status()
        self.assertFalse(st["simulated_driver_in_live_path"])
        self.assertIn("EmulatorBattleDriver", st["battle_driver"])


class LiveModuleWiringTests(unittest.TestCase):
    def test_train_make_env_calls_the_seam(self):
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("_twoby2_wrap(env", src)
        # guarded import
        self.assertIn("from twoby2.live_integration import maybe_wrap_full_agent", src)

    def test_battle_train_selects_real_driver_when_gate_on(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "battle_train.py").read_text()
        self.assertIn("make_live_battle_env", src)
        self.assertIn("assert_no_simulated_driver_in_live_path", src)

    def test_pokemon_env_step_sets_split_telemetry(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "pokemon_env.py").read_text()
        self.assertIn('info["twoby2_split"]', src)
        self.assertIn("nav_battle_wrapper", src)


if __name__ == "__main__":
    unittest.main()
