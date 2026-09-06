import tempfile
import unittest
from pathlib import Path

import numpy as np

import dynamic_brain
from dynamic_brain import DynamicLearner, combine_rollouts


class DynamicLearnerTests(unittest.TestCase):
    def test_combines_several_rollouts_before_a_policy_update(self):
        batch = {
            "images": np.zeros((2, 1, 64, 64), dtype=np.uint8),
            "nav": np.zeros((2, 28), dtype=np.float32),
            "actions": np.array([0, 1], dtype=np.int64),
            "rewards": np.array([0.1, 0.2], dtype=np.float32),
            "dones": np.array([False, True]),
            "log_probs": np.array([-1.9, -1.9], dtype=np.float32),
            "values": np.zeros(2, dtype=np.float32),
        }

        combined = combine_rollouts([batch, batch, batch])

        self.assertEqual(len(combined["actions"]), 6)
        self.assertEqual(combined["actions"].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(tuple(combined["images"].shape), (6, 1, 64, 64))

    def test_restores_the_latest_brain_before_republishing_best(self):
        with tempfile.TemporaryDirectory() as directory:
            original_model = dynamic_brain.MODEL_FILE
            original_policy = dynamic_brain.POLICY_FILE
            root = Path(directory)
            dynamic_brain.MODEL_FILE = root / "dynamic_policy.pt"
            dynamic_brain.POLICY_FILE = root / "policy.json"
            try:
                source = DynamicLearner()
                source.version = 17
                source.publish()
                restored = DynamicLearner()

                self.assertTrue(restored.restore_latest())
                self.assertEqual(restored.version, 17)
            finally:
                dynamic_brain.MODEL_FILE = original_model
                dynamic_brain.POLICY_FILE = original_policy

    def test_publishes_a_best_brain_artifact_for_the_watcher(self):
        with tempfile.TemporaryDirectory() as directory:
            original_model = dynamic_brain.MODEL_FILE
            original_best = dynamic_brain.BEST_MODEL_FILE
            original_policy = dynamic_brain.POLICY_FILE
            root = Path(directory)
            dynamic_brain.MODEL_FILE = root / "dynamic_policy.pt"
            dynamic_brain.BEST_MODEL_FILE = root / "dynamic_policy_best.pt"
            dynamic_brain.POLICY_FILE = root / "policy.json"
            try:
                DynamicLearner().publish(best=True)
                self.assertTrue(dynamic_brain.BEST_MODEL_FILE.is_file())
            finally:
                dynamic_brain.MODEL_FILE = original_model
                dynamic_brain.BEST_MODEL_FILE = original_best
                dynamic_brain.POLICY_FILE = original_policy

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
