import unittest

import numpy as np
import torch

from cluster_worker import choose_action, worker_rank
from dynamic_policy import PKMAIPolicy


class DynamicWorkerTests(unittest.TestCase):
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
