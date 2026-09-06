import unittest

import numpy as np
import torch

from cluster_worker import choose_action, live_telemetry, worker_rank
from dynamic_policy import PKMAIPolicy


class DynamicWorkerTests(unittest.TestCase):
    def test_live_telemetry_reports_current_overworld_position_and_progress(self):
        class Environment:
            cached_loc = {"valid": True, "map_bank": 3, "map_id": 7, "x_pos": 14, "y_pos": 22}
            total_steps = 128
            last_in_battle = 1
            saved_milestones = {"intro_complete", "stairs_down"}

        telemetry = live_telemetry(Environment(), action=6, reward=1.25)

        self.assertEqual(
            telemetry["position"],
            {"valid": True, "map_bank": 3, "map_id": 7, "x": 14, "y": 22},
        )
        self.assertEqual(telemetry["last_action"], 6)
        self.assertEqual(telemetry["episode_steps"], 128)
        self.assertTrue(telemetry["in_battle"])
        self.assertEqual(telemetry["milestones"], ["intro_complete", "stairs_down"])

    def test_scaled_compose_hostnames_map_to_distinct_zero_based_ranks(self):
        self.assertEqual(worker_rank("pkmai2-cluster-worker-1"), 0)
        self.assertEqual(worker_rank("pkmai2-cluster-worker-10"), 9)
        self.assertEqual(worker_rank("remote-worker"), 0)
        self.assertEqual(worker_rank("any-host", explicit_rank="7"), 7)

    def test_policy_selects_a_valid_discrete_action(self):
        policy = PKMAIPolicy()
        action, log_prob, value = choose_action(
            policy,
            {"image": np.zeros((1, 64, 64), dtype=np.uint8), "nav": np.zeros(28, dtype=np.float32)},
        )
        self.assertIn(action, range(7))
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
