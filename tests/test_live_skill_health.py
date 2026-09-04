import unittest
from collections import deque

from train import MilestoneCheckpointCallback


class LiveSkillHealthTests(unittest.TestCase):
    def test_old_failures_do_not_poison_new_retention(self):
        callback = object.__new__(MilestoneCheckpointCallback)
        callback.recent = deque(
            ({"role": "starter", "starter": 1} for _ in range(20)),
            maxlen=600,
        )
        callback.skill_health_seed = {
            "starter": {"success": 0, "episodes": 10_000}
        }
        health = callback._live_skill_health()
        self.assertEqual(health["starter"], {"score": 1000, "episodes": 20})

    def test_only_latest_64_episodes_define_retention(self):
        callback = object.__new__(MilestoneCheckpointCallback)
        callback.recent = deque(
            ([{"role": "starter", "starter": 0}] * 20)
            + ([{"role": "starter", "starter": 1}] * 64),
            maxlen=600,
        )
        callback.skill_health_seed = {}
        health = callback._live_skill_health()
        self.assertEqual(health["starter"], {"score": 1000, "episodes": 64})


if __name__ == "__main__":
    unittest.main()
