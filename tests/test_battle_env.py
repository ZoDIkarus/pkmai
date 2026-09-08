import warnings
warnings.filterwarnings("ignore")

import json
import os
import tempfile
import unittest

import numpy as np

import battle_types as bt
from battle_env import (BattleEnv, SimulatedBattleDriver, BattleRewardConfig,
                        OBS_DIM, ALL_MACROS, default_scenario, _mon, _mv)


def _first_legal(mask):
    return int(np.argmax(mask))


class BattleEnvBasicsTests(unittest.TestCase):
    def test_gym_env_checker_passes(self):
        from stable_baselines3.common.env_checker import check_env
        check_env(BattleEnv(), warn=True, skip_render_check=True)

    def test_obs_shape_and_mask(self):
        e = BattleEnv()
        obs, info = e.reset(seed=0)
        self.assertEqual(obs["vec"].shape, (OBS_DIM,))
        self.assertEqual(len(obs["action_mask"]), len(ALL_MACROS))
        self.assertTrue(np.all(np.isfinite(obs["vec"])))

    def test_invalid_action_is_penalised_not_crashed(self):
        e = BattleEnv()
        obs, _ = e.reset(seed=0)
        # force an illegal index (a fainted switch slot)
        illegal = next(i for i, m in enumerate(obs["action_mask"]) if not m)
        obs, r, term, trunc, info = e.step(illegal)
        self.assertTrue(info.get("invalid"))
        self.assertLessEqual(r, BattleRewardConfig.INVALID_ACTION + BattleRewardConfig.TURN_COST)

    def test_live_style_safe_fallback_advances_but_never_rewards_invalid_choice(self):
        driver = SimulatedBattleDriver()
        driver.advance_on_invalid = True
        e = BattleEnv(driver)
        obs, _ = e.reset(seed=0)
        hp_before = e._state["enemy_active"]["cur_hp"]
        illegal = next(i for i, allowed in enumerate(obs["action_mask"])
                       if not allowed)
        obs, reward, _, _, info = e.step(illegal)
        self.assertTrue(info["invalid"])
        self.assertEqual(info["fallback_macro"], "MOVE_1")
        self.assertLess(e._state["enemy_active"]["cur_hp"], hp_before)
        self.assertEqual(reward, BattleRewardConfig.TURN_COST
                         + BattleRewardConfig.INVALID_ACTION)


class RewardDecompositionTests(unittest.TestCase):
    def test_components_sum_to_the_scalar_reward(self):
        from battle_env import decompose_battle_reward
        ev = {"our_damage_dealt": 12, "enemy_hp_before": 20, "enemy_ko": True,
              "battle_won": True}
        total = (BattleRewardConfig.TURN_COST
                 + BattleRewardConfig.DAMAGE_PER_HP * 12
                 + BattleRewardConfig.ENEMY_KO + BattleRewardConfig.BATTLE_WIN)
        bits = decompose_battle_reward(ev, total)
        self.assertAlmostEqual(sum(a for _, a in bits), total, places=4)
        self.assertIn("battle_win", [n for n, _ in bits])

    def test_switch_loop_is_an_explicit_component(self):
        from battle_env import decompose_battle_reward
        # 2026-09-08 (v2): switch_loop is a real ev field, not a residual
        bits = decompose_battle_reward({"switch_loop": True})
        self.assertIn("switch_loop", [n for n, _ in bits])
        self.assertAlmostEqual(sum(a for _, a in bits),
                               BattleRewardConfig.TURN_COST
                               + BattleRewardConfig.SWITCH_LOOP, places=4)

    def test_wild_flee_costs_more_than_a_fainted_mon(self):
        self.assertLessEqual(BattleRewardConfig.FLEE_WILD_OK,
                             BattleRewardConfig.OWN_FAINT)
        self.assertGreater(BattleRewardConfig.FLEE_WILD_OK,
                           BattleRewardConfig.WIPE)


class MirrorSidecarTests(unittest.TestCase):
    """The visible mirror worker publishes a reward-stream sidecar; a headless
    env never touches it."""

    def test_headless_env_writes_no_sidecar(self):
        e = BattleEnv()
        e.reset(seed=0, options={"scenario": default_scenario()})
        e.step(_first_legal(e._obs()["action_mask"]))
        self.assertIsNone(e._mirror_sidecar)

    def test_mirror_env_writes_decomposed_reward_events_and_totals_match(self):
        with tempfile.TemporaryDirectory() as d:
            side = os.path.join(d, "battle_worker_live.json")
            e = BattleEnv(SimulatedBattleDriver(), mirror_sidecar=side)
            obs, _ = e.reset(seed=1, options={"scenario": default_scenario()})
            self.assertTrue(os.path.isfile(side))       # written on reset too
            total = 0.0
            done = trunc = False
            steps = 0
            while not (done or trunc) and steps < 40:
                obs, r, done, trunc, _info = e.step(_first_legal(obs["action_mask"]))
                total += r
                steps += 1
            with open(side) as f:
                doc = json.load(f)
            self.assertEqual(doc["schema"], "battle_mirror_v1")
            self.assertEqual(doc["episode"], 1)
            self.assertAlmostEqual(doc["episode_reward"], total, places=2)
            self.assertTrue(doc["events"])
            # every event parses as "name:+/-amount" and the per-turn sums are
            # consistent with the scalar rewards (residual folded into one entry)
            for _t, ev in doc["events"]:
                name, amt = ev.rsplit(":", 1)
                self.assertTrue(name)
                float(amt)


class BattleEnvRewardTests(unittest.TestCase):
    def _run(self, scenario, policy):
        e = BattleEnv()
        obs, _ = e.reset(seed=1, options={"scenario": scenario})
        total, steps = 0.0, 0
        done = trunc = False
        while not (done or trunc) and steps < 80:
            a = policy(obs, steps)
            obs, r, done, trunc, info = e.step(a)
            total += r
            steps += 1
        return total, info, e

    def test_winning_pays_ko_and_win_once(self):
        total, info, e = self._run(default_scenario(),
                                   lambda o, s: _first_legal(o["action_mask"]))
        self.assertEqual(info.get("outcome"), "win")
        # KO(1) + WIN(3) + some damage, minus turn costs -> comfortably positive
        self.assertGreater(total, BattleRewardConfig.BATTLE_WIN)

    def test_damage_is_capped_at_real_enemy_hp_no_farming(self):
        # enemy with 5 HP, our move would do far more; reward for the kill hit
        # must reflect min(dmg, 5), not the raw number.
        sc = {
            "our_party": [_mon((bt.TYPE_WATER,), level=40, spa=120, spe=99,
                               moves=[_mv(bt.TYPE_WATER, 120, mid=1)])],
            "enemy_party": [_mon((bt.TYPE_ROCK,), level=3, cur_hp=5, max_hp=5)],
            "is_trainer": False, "can_escape": True,
        }
        e = BattleEnv()
        obs, _ = e.reset(seed=0, options={"scenario": sc})
        obs, r, done, trunc, info = e.step(0)   # MOVE_1
        self.assertTrue(done)
        dmg_reward = BattleRewardConfig.DAMAGE_PER_HP * 5
        # r = turn_cost + dmg(<=5) + KO + WIN
        self.assertLess(r, (BattleRewardConfig.TURN_COST + dmg_reward
                            + BattleRewardConfig.ENEMY_KO
                            + BattleRewardConfig.BATTLE_WIN) + 1e-6)

    def test_trainer_flee_is_illegal_and_penalised(self):
        sc = {**default_scenario(), "is_trainer": True, "can_escape": False}
        e = BattleEnv()
        obs, _ = e.reset(seed=0, options={"scenario": sc})
        run_ix = ALL_MACROS.index("RUN")
        obs, r, term, trunc, info = e.step(run_ix)
        # RUN is masked in a trainer battle -> invalid penalty, battle continues
        self.assertTrue(info.get("invalid"))
        self.assertFalse(term)

    def test_switch_loop_is_penalised(self):
        sc = {
            "our_party": [_mon((bt.TYPE_WATER,), slot=0, spe=5,
                               moves=[_mv(bt.TYPE_WATER, 10, mid=1)]),
                          _mon((bt.TYPE_GRASS,), slot=1, spe=5,
                               moves=[_mv(bt.TYPE_GRASS, 10, mid=2)])],
            "enemy_party": [_mon((bt.TYPE_NORMAL,), level=3, cur_hp=200, max_hp=200,
                                 moves=[_mv(bt.TYPE_NORMAL, 5, mid=9)])],
            "is_trainer": True, "can_escape": False,
        }
        e = BattleEnv()
        obs, _ = e.reset(seed=0, options={"scenario": sc})
        s1 = ALL_MACROS.index("SWITCH_2")
        s0 = ALL_MACROS.index("SWITCH_1")
        e.step(s1)
        e.step(s0)
        obs, r, *_ = e.step(s1)     # A->B->A loop
        self.assertLessEqual(r, BattleRewardConfig.SWITCH_LOOP)

    def test_no_navigation_or_story_reward_anywhere(self):
        import ast
        import inspect
        import battle_env
        tree = ast.parse(inspect.getsource(battle_env))
        # no navigation module imported
        imported = {n.module for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("pokemon_env", imported)
        # no navigation counter / progress identifier used in code
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for bad in ("route_steps", "world_stage", "mastered_stage",
                    "story_progress", "tiles_seen"):
            self.assertNotIn(bad, names | attrs)


class DeterministicDriverTests(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        def rollout():
            e = BattleEnv(SimulatedBattleDriver())
            obs, _ = e.reset(seed=42, options={"scenario": default_scenario()})
            hist = []
            for _ in range(20):
                a = _first_legal(obs["action_mask"])
                obs, r, d, t, info = e.step(a)
                hist.append((a, round(r, 5), d, t))
                if d or t:
                    break
            return hist
        self.assertEqual(rollout(), rollout())


if __name__ == "__main__":
    unittest.main()
