import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ShellScriptLineEndingTests(unittest.TestCase):
    def test_shell_scripts_use_lf_line_endings_for_linux_containers(self):
        for script in (PROJECT_ROOT / "scripts").glob("*.sh"):
            with self.subTest(script=script.name):
                self.assertNotIn(b"\r", script.read_bytes())

    def test_cluster_brain_starts_the_dynamic_batch_learner(self):
        brain_script = (PROJECT_ROOT / "scripts" / "start_cluster_brain.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("src/dynamic_brain.py", brain_script)
        self.assertNotIn("ray start", brain_script)

    def test_cluster_worker_starts_without_a_ray_runtime(self):
        worker_script = (PROJECT_ROOT / "scripts" / "start_cluster_worker.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("src/cluster_worker.py", worker_script)
        self.assertNotIn("ray start", worker_script)


if __name__ == "__main__":
    unittest.main()
