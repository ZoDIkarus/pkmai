import unittest

import numpy as np

from dynamic_brain import DynamicLearner


class DynamicLearnerTests(unittest.TestCase):
    def test_trains_only_from_supplied_rollout_batch(self):
        learner = DynamicLearner()
        before = learner.version
        metrics = learner.learn(
            {
                "images": np.zeros((4, 1, 64, 64), dtype=np.uint8),
                "nav": np.zeros((4, 28), dtype=np.float32),
                "actions": np.array([0, 1, 2, 3], dtype=np.int64),
                "rewards": np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float32),
                "dones": np.array([False, False, False, True]),
                "log_probs": np.array([-1.9, -1.9, -1.9, -1.9], dtype=np.float32),
                "values": np.zeros(4, dtype=np.float32),
            }
        )
        self.assertEqual(learner.version, before + 1)
        self.assertEqual(metrics["samples"], 4)
