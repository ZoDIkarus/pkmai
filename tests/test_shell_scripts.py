import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ShellScriptLineEndingTests(unittest.TestCase):
    def test_shell_scripts_use_lf_line_endings_for_linux_containers(self):
        for script in (PROJECT_ROOT / "scripts").glob("*.sh"):
            with self.subTest(script=script.name):
                self.assertNotIn(b"\r", script.read_bytes())

    def test_cluster_worker_advertises_the_configured_cpu_capacity_to_ray(self):
        worker_script = (PROJECT_ROOT / "scripts" / "start_cluster_worker.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('RAY_WORKER_CPUS="${PKMAI_WORKER_CPUS%%.*}"', worker_script)
        self.assertIn('--num-cpus="$RAY_WORKER_CPUS"', worker_script)


if __name__ == "__main__":
    unittest.main()
