"""Observation + weight migration against the REAL navigation champion."""
import warnings
warnings.filterwarnings("ignore")

import os
import unittest

import numpy as np

CHAMP = os.path.join(os.path.dirname(__file__), "..", "runtime", "checkpoints",
                     "pokemon_model_champion.zip")
CHAMP = os.path.abspath(CHAMP)

from twoby2.nav_feature_extractor import (  # noqa: E402
    migrate_navigation_policy, build_v2_observation_space, OBS_SCHEMA_V2,
    MAP_BRANCH_OUT)


@unittest.skipUnless(os.path.exists(CHAMP), "no champion checkpoint present")
class RealCheckpointMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        new_policy, report, verify = migrate_navigation_policy(CHAMP, map_channels=6)
        cls.new_policy = new_policy
        cls.report = report
        cls._verify = staticmethod(verify)

    def verify(self, *a, **kw):
        return type(self)._verify(*a, **kw)

    def _batch(self, n=8, seed=0, map_value="random"):
        rng = np.random.default_rng(seed)
        if map_value == "zero":
            mp = np.zeros((n, 6, 64, 64), dtype=np.uint8)
        else:
            mp = rng.integers(0, 256, (n, 6, 64, 64), dtype=np.uint8)
        return {
            "image": rng.integers(0, 256, (n, 4, 64, 64), dtype=np.uint8),
            "nav": rng.uniform(-1, 1, (n, 31)).astype(np.float32),
            "map": mp,
        }

    def test_weight_report_is_clean(self):
        r = self.report
        self.assertEqual(r["load_missing"], [])
        self.assertEqual(r["load_unexpected"], [])
        self.assertEqual(r["missing_in_old"], [])   # every shared param mapped
        self.assertEqual(r["widened"], 2)            # policy_net.0 + value_net.0
        self.assertGreater(r["copied"], 20)
        self.assertEqual(r["new_feature_dim"] - r["old_feature_dim"], MAP_BRANCH_OUT)

    def test_logits_and_values_identical_with_zero_map(self):
        ok, dl, dv = self.verify(self._batch(map_value="zero"))
        self.assertTrue(ok)
        self.assertLess(dl, 1e-4)
        self.assertLess(dv, 1e-4)

    def test_logits_and_values_identical_with_random_map(self):
        # the map branch is zero-initialised -> ANY map input is a no-op at init
        ok, dl, dv = self.verify(self._batch(seed=7, map_value="random"))
        self.assertTrue(ok)
        self.assertEqual((dl, dv), (0.0, 0.0))

    def test_v2_obs_space_adds_only_the_map_key(self):
        from stable_baselines3 import PPO
        old = PPO.load(CHAMP, device="cpu")
        v2 = build_v2_observation_space(old.observation_space, map_channels=6)
        self.assertEqual(set(v2.spaces) - set(old.observation_space.spaces), {"map"})
        self.assertEqual(v2.spaces["map"].shape, (6, 64, 64))

    def test_new_policy_is_trainable_widened_columns_then_map_branch(self):
        import torch
        pol = self.new_policy
        vcol_before = pol.mlp_extractor.value_net[0].weight.detach()[:, 287:].clone()
        pcol_before = pol.mlp_extractor.policy_net[0].weight.detach()[:, 287:].clone()
        map_before = {n: p.detach().clone() for n, p in pol.named_parameters()
                      if "extractors.map" in n}
        pol.train()
        for step in range(3):
            obs = {k: torch.as_tensor(v) for k, v in self._batch(seed=step).items()}
            _, values, log_prob = pol(obs)
            loss = values.pow(2).mean() - log_prob.mean()   # value + policy signal
            pol.optimizer.zero_grad()
            loss.backward()
            pol.optimizer.step()
        vcol_after = pol.mlp_extractor.value_net[0].weight.detach()[:, 287:]
        pcol_after = pol.mlp_extractor.policy_net[0].weight.detach()[:, 287:]
        self.assertGreater((vcol_after - vcol_before).abs().max().item(), 0.0,
                           "widened value-net map columns never left zero")
        self.assertGreater((pcol_after - pcol_before).abs().max().item(), 0.0,
                           "widened policy-net map columns never left zero")
        moved = any(not torch.equal(p.detach(), map_before[n])
                    for n, p in pol.named_parameters() if n in map_before)
        self.assertTrue(moved, "map branch never trained after a few steps")


if __name__ == "__main__":
    unittest.main()
