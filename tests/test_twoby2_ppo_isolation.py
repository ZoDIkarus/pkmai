"""Real 2×2 isolation: two actual small PPO models must share nothing and a
step of one must not perturb the other."""
import warnings
warnings.filterwarnings("ignore")

import unittest

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from twoby2.ppo_isolation import (assert_models_isolated, param_snapshot,
                                  params_unchanged, max_param_delta)
from twoby2.isolation import (navigation_counters, battle_counters,
                              IsolatedSystem, assert_systems_isolated,
                              NavigationImmutableDuringBattle, ModelPathGuard)


class _TinyNav(gym.Env):
    observation_space = spaces.Box(-1, 1, (16,), np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, a):
        return (self.observation_space.sample(), float(np.random.rand()),
                False, np.random.rand() < 0.1, {})


class _TinyBattle(gym.Env):
    observation_space = spaces.Box(-1, 1, (10,), np.float32)
    action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, a):
        return (self.observation_space.sample(), float(np.random.rand()),
                False, np.random.rand() < 0.1, {})


def _nav_ppo(seed=0):
    return PPO("MlpPolicy", DummyVecEnv([_TinyNav]), n_steps=32,
               batch_size=32, n_epochs=1, seed=seed, device="cpu", verbose=0)


def _battle_ppo(seed=1):
    return PPO("MlpPolicy", DummyVecEnv([_TinyBattle]), n_steps=32,
               batch_size=32, n_epochs=1, seed=seed, device="cpu", verbose=0)


class RealPPOIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nav = _nav_ppo()
        cls.battle = _battle_ppo()

    def test_no_shared_parameters_or_optimizer(self):
        assert_models_isolated(self.nav, self.battle)
        self.assertNotEqual(id(self.nav.policy.optimizer),
                            id(self.battle.policy.optimizer))
        self.assertEqual(
            {id(p) for p in self.nav.policy.parameters()}
            & {id(p) for p in self.battle.policy.parameters()}, set())

    def test_battle_optimizer_step_does_not_touch_navigation_params(self):
        nav_snap = param_snapshot(self.nav)
        self.battle.learn(total_timesteps=64, reset_num_timesteps=False)
        self.assertTrue(params_unchanged(self.nav, nav_snap))
        self.assertEqual(max_param_delta(self.nav, nav_snap), 0.0)

    def test_navigation_optimizer_step_does_not_touch_battle_params(self):
        bat_snap = param_snapshot(self.battle)
        self.nav.learn(total_timesteps=64, reset_num_timesteps=False)
        self.assertTrue(params_unchanged(self.battle, bat_snap))

    def test_counter_sets_are_disjoint_and_typed(self):
        n, b = navigation_counters(), battle_counters()
        self.assertEqual(set(n.snapshot()) & set(b.snapshot()), set())
        with self.assertRaises(KeyError):
            b.add("world_stage", 1)

    def test_battle_work_cannot_move_navigation_counters(self):
        nav_sys = IsolatedSystem("navigation", navigation_counters())
        bat_sys = IsolatedSystem("battle", battle_counters())
        assert_systems_isolated(nav_sys, bat_sys)
        nav_sys.counters.add("nav_env_steps", 100)
        with NavigationImmutableDuringBattle(nav_sys):
            for _ in range(30):
                bat_sys.counters.add("battle_env_steps", 1)
                bat_sys.record_rollout({"x": 1})

    def test_model_path_guard_blocks_cross_writes(self):
        self.assertFalse(ModelPathGuard.battle_may_write("runtime/navigation/checkpoints/navigation_learner.zip"))
        self.assertFalse(ModelPathGuard.navigation_may_write("runtime/battle/checkpoints/battle_champion.zip"))
        self.assertTrue(ModelPathGuard.battle_may_write("runtime/battle/checkpoints/battle_learner.zip"))


if __name__ == "__main__":
    unittest.main()
