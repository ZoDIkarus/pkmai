import tempfile
import unittest
from pathlib import Path

import numpy as np

import dynamic_brain
from dynamic_brain import DynamicLearner, combine_rollouts, load_best_mean_reward


class DynamicLearnerTests(unittest.TestCase):
    def test_missing_best_score_does_not_invent_a_quality_value(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_best_mean_reward(Path(directory) / "missing.json"), (-1, float("-inf"), float("inf")))

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

    def test_rollout_quality_uses_explicit_objective_success(self):
        quality = dynamic_brain.rollout_quality(
            {"rewards": np.array([200.0, 0.0], dtype=np.float32),
             "objective_success": np.array([False, True], dtype=np.bool_)},
            0.0,
        )
        self.assertEqual(quality[0], 1)

    def test_rollout_stage_summary_separates_objectives_from_rewards(self):
        summary = dynamic_brain.rollout_stage_summary({
            "objective_code": np.array([1, 1, 2], dtype=np.int8),
            "objective_success": np.array([True, False, True], dtype=np.bool_),
            "success_steps": np.array([40, -1, 80], dtype=np.int32),
        })
        self.assertEqual(summary[1]["successes"], 1.0)
        self.assertEqual(summary[1]["success_rate"], 0.5)
        self.assertEqual(summary[2]["median_success_steps"], 80.0)

    def test_stage_summary_counts_only_terminal_episode_outcomes(self):
        summary = dynamic_brain.rollout_stage_summary({
            "objective_code": np.array([2, 2, 2, 3, 3], dtype=np.int8),
            "objective_success": np.array([False, False, True, False, False], dtype=np.bool_),
            "success_steps": np.array([-1, -1, 80, -1, -1], dtype=np.int32),
            "dones": np.array([False, False, True, False, True], dtype=np.bool_),
        })

        self.assertEqual(summary[2]["samples"], 1.0)
        self.assertEqual(summary[2]["successes"], 1.0)
        self.assertEqual(summary[3]["samples"], 1.0)
        self.assertEqual(summary[3]["success_rate"], 0.0)

    def test_stage_gate_blocks_regression_of_confirmed_stage(self):
        self.assertFalse(dynamic_brain.stage_gate_allows_promotion(
            {1: {"samples": 64, "success_rate": 0.2}},
            {"1": {"samples": 64, "success_rate": 0.9}},
        ))

    def test_stage_gate_requires_evidence_for_every_confirmed_stage(self):
        self.assertFalse(dynamic_brain.stage_gate_allows_promotion(
            {2: {"samples": 64, "success_rate": 0.9}},
            {"1": {"samples": 64, "success_rate": 0.9}},
        ))

    def test_stage_gate_blocks_insufficient_episode_evidence(self):
        self.assertFalse(dynamic_brain.stage_gate_allows_promotion(
            {2: {"samples": 8, "success_rate": 0.0}},
            {"2": {"samples": 64, "success_rate": 0.9}},
        ))
