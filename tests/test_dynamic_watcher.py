import tempfile
import unittest
from pathlib import Path

from dynamic_watcher import select_published_model


class DynamicWatcherTests(unittest.TestCase):
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
