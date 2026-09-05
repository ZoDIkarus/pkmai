import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RequirementsTests(unittest.TestCase):
    def test_ray_installs_default_runtime_agents_for_remote_workers(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("ray[default,rllib]==2.58.0", requirements)


if __name__ == "__main__":
    unittest.main()
