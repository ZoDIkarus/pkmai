"""Go-live mandatory v1/v2 compatibility gate (13 tests).

A v1 (combat-only, 116 / 11) battle model must keep working after the restart:
the env it runs on, the observation it is fed and the live champion adapter all
have to be v1. A v2 (140 / 12) model must never be loaded against a v1 env and
vice versa — no padding, no truncation, and a fallback must publish its reason.
"""
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

import battle_train as bt
from battle_env import (BattleEnv, SimulatedBattleDriver, battle_schema_spec,
                        default_scenario, catch_scenario)
from battle_executor import CATCH_ACTION, ALL_MACROS, ALL_MACROS_V1

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runtime", "battle", "checkpoints")
V1_CHAMPION = os.path.join(CKPT, "battle_champion.zip")
V1_RESUME = os.path.join(CKPT, "battle_resume.zip")
V1_LEARNER = os.path.join(CKPT, "battle_learner.zip")

_HAVE_V1 = all(os.path.isfile(p) for p in (V1_CHAMPION, V1_RESUME, V1_LEARNER))
_V1_IS_V1 = _HAVE_V1 and bt.read_battle_model_spaces(V1_CHAMPION) == (116, 11)

_TMP = tempfile.TemporaryDirectory()
_V2_MODEL = os.path.join(_TMP.name, "v2.zip")


def setUpModule():
    # one tiny v2 model for the cross-rejection tests
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec = DummyVecEnv([lambda: BattleEnv(SimulatedBattleDriver(), schema="v2")])
    m = bt.build_battle_ppo(vec, seed=0)
    m.save(_V2_MODEL)


def tearDownModule():
    _TMP.cleanup()


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _predict_loop(model, env, schema, n=6):
    obs, _ = env.reset(seed=0, options={"scenario": default_scenario()})
    seen = []
    for _ in range(n):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        a, _ = model.predict(obs, deterministic=True, action_masks=mask)
        seen.append(int(a))
        obs, r, term, trunc, i = env.step(int(a))
        if term or trunc:
            break
    return seen


class SchemaShapeTests(unittest.TestCase):
    def test_1_v1_env_is_exactly_116_11(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v1")
        e.reset(options={"scenario": default_scenario()})
        self.assertEqual(e.observation_space["vec"].shape, (116,))
        self.assertEqual(e.action_space.n, 11)
        self.assertEqual(len(e._obs()["action_mask"]), 11)
        self.assertEqual(e._obs()["vec"].shape, (116,))
        self.assertEqual(e._obs_schema, "battle_obs_v1")

    def test_2_v2_env_is_exactly_140_12(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v2")
        e.reset(options={"scenario": catch_scenario()})
        self.assertEqual(e.observation_space["vec"].shape, (140,))
        self.assertEqual(e.action_space.n, 12)
        self.assertEqual(len(e._obs()["action_mask"]), 12)

    def test_schema_spec_matches(self):
        self.assertEqual(battle_schema_spec("v1")[0], ALL_MACROS_V1)
        self.assertEqual(battle_schema_spec("v2")[0], ALL_MACROS)


@unittest.skipUnless(_V1_IS_V1, "no live v1 battle_champion.zip (116/11)")
class LiveV1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sb3_contrib import MaskablePPO
        cls.model = MaskablePPO.load(V1_CHAMPION, device="cpu")

    def test_3_v1_resume_loads_against_a_v1_env(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib import MaskablePPO
        vec = DummyVecEnv([lambda: BattleEnv(SimulatedBattleDriver(), schema="v1")])
        m = MaskablePPO.load(V1_RESUME, env=vec, device="cpu")
        self.assertEqual(m.observation_space["vec"].shape[0], 116)
        self.assertEqual(m.action_space.n, 11)

    def test_4_v1_champion_predicts_on_a_real_v1_observation(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v1")
        seen = _predict_loop(self.model, e, "v1")
        self.assertTrue(seen)

    def test_5_v1_prediction_is_only_index_0_to_10(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v1")
        for a in _predict_loop(self.model, e, "v1", n=10):
            self.assertIn(a, range(0, 11))

    def test_6_v1_can_never_return_catch(self):
        self.assertNotIn(CATCH_ACTION, ALL_MACROS_V1)
        e = BattleEnv(SimulatedBattleDriver(), schema="v1")
        # even a catch scenario cannot make a v1 env expose CATCH
        e.reset(options={"scenario": catch_scenario()})
        self.assertEqual(len(e._legal_mask()), 11)

    def test_8_v1_model_is_refused_by_a_v2_trainer(self):
        with self.assertRaises(bt.BattleSchemaError):
            bt.battle_schema_preflight(V1_CHAMPION, expect_obs_dim=140,
                                       expect_n_actions=12, schema="v2")
        # and passes the v1 preflight silently
        bt.battle_schema_preflight(V1_CHAMPION, expect_obs_dim=116,
                                   expect_n_actions=11, schema="v1")

    def test_13_champion_learner_resume_hashes_unchanged(self):
        # loading + preflighting must not rewrite the files
        before = {p: _sha(p) for p in (V1_CHAMPION, V1_RESUME, V1_LEARNER)}
        for p in before:
            bt.battle_schema_preflight(p, expect_obs_dim=116, expect_n_actions=11,
                                       schema="v1")
        from sb3_contrib import MaskablePPO
        MaskablePPO.load(V1_CHAMPION, device="cpu")
        after = {p: _sha(p) for p in (V1_CHAMPION, V1_RESUME, V1_LEARNER)}
        self.assertEqual(before, after)


class CrossRejectionTests(unittest.TestCase):
    def test_7_v2_model_is_refused_by_a_v1_trainer(self):
        with self.assertRaises(bt.BattleSchemaError):
            bt.battle_schema_preflight(_V2_MODEL, expect_obs_dim=116,
                                       expect_n_actions=11, schema="v1")
        bt.battle_schema_preflight(_V2_MODEL, expect_obs_dim=140,
                                   expect_n_actions=12, schema="v2")

    def test_read_spaces_on_the_tiny_v2_model(self):
        self.assertEqual(bt.read_battle_model_spaces(_V2_MODEL), (140, 12))


class LiveChampionAdapterTests(unittest.TestCase):
    def _policy(self):
        from twoby2.live_integration import LatestBattleChampionPolicy
        return LatestBattleChampionPolicy(path=V1_CHAMPION)

    @unittest.skipUnless(_V1_IS_V1, "no live v1 champion")
    def test_9_live_policy_loads_the_v1_champion_as_ppo(self):
        p = self._policy()
        m = p._load_current()
        self.assertIsNotNone(m)
        self.assertEqual(p._schema, "v1")
        self.assertEqual(p._dims, (116, 11))

    def test_10_predict_failure_publishes_a_visible_fallback_reason(self):
        from twoby2 import live_integration as li
        p = li.LatestBattleChampionPolicy(path=os.path.join(_TMP.name, "nope.zip"))
        act = p.pin()
        self.assertIs(act, li.rule_battle_policy)
        with open(li._BATTLE_POLICY_STATUS) as f:
            doc = json.load(f)
        self.assertEqual(doc["battle_policy_source"], "rule")
        self.assertTrue(doc["battle_policy_fallback_reason"])

    def test_11_catch_is_masked_in_every_live_path_while_ram_unverified(self):
        from twoby2 import battle_ram_live as L
        from twoby2.emulator_battle_driver import EmulatorBattleDriver
        self.assertFalse(L.catch_ram_ready(None)[0])
        drv = EmulatorBattleDriver.__new__(EmulatorBattleDriver)
        drv._objective = {"catch_requested": True, "target_species_id": 19,
                          "usable_ball_count": 3, "party_has_space": True,
                          "pc_capture_supported": False}
        drv._macros = ALL_MACROS_V1
        drv.schema = "v1"
        snap = {"player_active": {"moves": [{"id": 1, "pp": 5, "mechanics_known": True,
                                             "power": 40}]},
                "enemy_active": {"species_id": 19, "cur_hp": 10},
                "is_trainer": False, "can_escape": True, "active_slot": 0,
                "catch_ram_ready": True}
        legal = EmulatorBattleDriver.legal_macros(drv, snap)
        self.assertNotIn(CATCH_ACTION, legal)   # v1 macro list has no CATCH slot


class V1CanaryTests(unittest.TestCase):
    def test_12_v1_battle_runs_without_a_shape_error(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v1", max_turns=25)
        obs, _ = e.reset(seed=1, options={"scenario": default_scenario()})
        total = 0.0
        for _ in range(25):
            mask = obs["action_mask"]
            a = next((i for i, m in enumerate(mask) if m), 0)
            obs, r, term, trunc, i = e.step(a)
            total += r
            self.assertEqual(obs["vec"].shape, (116,))
            self.assertEqual(len(obs["action_mask"]), 11)
            if term or trunc:
                break
        self.assertIsInstance(total, float)


if __name__ == "__main__":
    unittest.main()
