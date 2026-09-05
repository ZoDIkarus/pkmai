import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dynamic_watcher import choose_watcher_action, select_published_model


class DynamicWatcherTests(unittest.TestCase):
    def test_samples_the_policy_distribution_instead_of_always_taking_argmax(self):
        class BalancedPolicy:
            def __call__(self, image, nav):
                return torch.zeros((1, 7)), torch.zeros(1)

        observation = {
            "image": np.zeros((1, 64, 64), dtype=np.uint8),
            "nav": np.zeros(28, dtype=np.float32),
        }
        generator = torch.Generator().manual_seed(0)

        self.assertEqual(choose_watcher_action(BalancedPolicy(), observation, generator), 6)

    def test_prefers_the_best_published_brain_over_the_latest_brain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "dynamic_policy.pt"
            best = root / "dynamic_policy_best.pt"
            latest.write_bytes(b"latest")
            best.write_bytes(b"best")

            self.assertEqual(select_published_model(best, latest), best)

    def test_uses_latest_brain_until_a_best_publication_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "dynamic_policy.pt"
            best = root / "dynamic_policy_best.pt"
            latest.write_bytes(b"latest")

            self.assertEqual(select_published_model(best, latest), latest)


if __name__ == "__main__":
    unittest.main()
