import unittest

import numpy as np
import torch

from cluster_worker import choose_action
from dynamic_policy import PKMAIPolicy


class DynamicWorkerTests(unittest.TestCase):
    def test_policy_selects_a_valid_discrete_action(self):
        policy = PKMAIPolicy()
        action, log_prob, value = choose_action(
            policy,
            {"image": np.zeros((1, 64, 64), dtype=np.uint8), "nav": np.zeros(28, dtype=np.float32)},
        )
        self.assertIn(action, range(7))
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
