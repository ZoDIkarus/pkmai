import tempfile
import unittest
from pathlib import Path

from reward_state import claim_event


class RewardStateTests(unittest.TestCase):
    def test_claim_event_pays_exactly_once_across_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertTrue(claim_event(path, "center_5_4"))
            self.assertFalse(claim_event(path, "center_5_4"))
            self.assertTrue(claim_event(path, "mart_5_3"))


if __name__ == "__main__":
    unittest.main()
