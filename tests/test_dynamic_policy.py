import unittest

import torch

from dynamic_policy import PKMAIPolicy


class DynamicPolicyTests(unittest.TestCase):
    def test_returns_action_logits_and_values_for_pkmai_observations(self):
        model = PKMAIPolicy()
        logits, values = model(
            torch.zeros((2, 1, 64, 64), dtype=torch.uint8),
            torch.zeros((2, 28), dtype=torch.float32),
        )
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(values.shape), (2,))
