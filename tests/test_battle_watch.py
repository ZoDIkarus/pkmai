import warnings
warnings.filterwarnings("ignore")

import unittest

import battle_watch as bw
from battle_env import SimulatedBattleDriver
from twoby2.config import (BATTLE_HEADLESS_WORKERS, BATTLE_VISIBLE_WORKERS,
                           BATTLE_WORKERS)


class VisibleBattleWorkerTests(unittest.TestCase):
    def test_9_plus_1_equals_10_and_visible_is_not_extra(self):
        self.assertEqual(BATTLE_HEADLESS_WORKERS, 8)
        self.assertEqual(BATTLE_VISIBLE_WORKERS, 1)
        self.assertEqual(BATTLE_HEADLESS_WORKERS + BATTLE_VISIBLE_WORKERS,
                         BATTLE_WORKERS)
        self.assertEqual(bw.worker_descriptor()["counts_toward"],
                         "BATTLE_WORKERS (the 9th worker, not an extra)")

    def test_no_own_optimizer_or_model_uses_central_learner(self):
        w = bw.VisibleBattleWorker(learner_rollout_sink=lambda x: None,
                                   policy_provider=lambda: object(),
                                   scenario_provider=lambda: {})
        self.assertTrue(w.uses_central_learner_only())
        self.assertIsNone(w._own_optimizer)
        self.assertIsNone(w._own_model)

    def test_render_is_declared_side_effect_free(self):
        w = bw.VisibleBattleWorker(learner_rollout_sink=lambda x: None,
                                   policy_provider=lambda: object(),
                                   scenario_provider=lambda: {})
        self.assertTrue(w.render_is_side_effect_free())
        self.assertEqual(w.fps, 60)

    def test_policy_is_pinned_per_battle(self):
        calls = []
        w = bw.VisibleBattleWorker(learner_rollout_sink=lambda x: None,
                                   policy_provider=lambda: calls.append(1) or f"snap{len(calls)}",
                                   scenario_provider=lambda: {})
        p1 = w.pin_policy_for_battle()
        self.assertEqual(w._pinned_policy, p1)

    def test_fail_closed_while_emulator_driver_is_blocked(self):
        # simulated driver -> NOT real training
        w = bw.VisibleBattleWorker(learner_rollout_sink=lambda x: None,
                                   policy_provider=lambda: object(),
                                   scenario_provider=lambda: {},
                                   driver=SimulatedBattleDriver())
        self.assertFalse(w.is_real_training_ready())
        with self.assertRaises(RuntimeError):
            w.collect_battle()

    def test_no_driver_is_also_fail_closed(self):
        w = bw.VisibleBattleWorker(learner_rollout_sink=lambda x: None,
                                   policy_provider=lambda: object(),
                                   scenario_provider=lambda: {})
        self.assertFalse(w.is_real_training_ready())
        with self.assertRaises(RuntimeError):
            w.collect_battle()


if __name__ == "__main__":
    unittest.main()
