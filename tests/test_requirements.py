import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RequirementsTests(unittest.TestCase):
    def test_local_dynamic_cluster_does_not_install_retired_ray_or_sb3_stacks(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertNotIn("ray[", requirements)
        self.assertNotIn("stable-baselines3", requirements)


if __name__ == "__main__":
    unittest.main()
