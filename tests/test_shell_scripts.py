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

    def test_local_cluster_compose_uses_loopback_control_plane(self):
        compose_text = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
        master_section = compose_text.split("  cluster-master:\n", 1)[1].split("  cluster-brain:\n", 1)[0]

        self.assertIn('PKMAI_CLUSTER_HOST: "${PKMAI_CLUSTER_HOST:-127.0.0.1}"', master_section)
        self.assertNotIn("cluster-worker:", compose_text)

    def test_remote_worker_compose_does_not_require_removed_ray_settings(self):
        compose_text = (PROJECT_ROOT / "compose.remote-worker.yaml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("PKMAI_RAY_ADDRESS", compose_text)
        self.assertNotIn("RAY_NODE_IP", compose_text)

    def test_local_launcher_assigns_explicit_worker_ids_and_ranks(self):
        launcher = (PROJECT_ROOT / "scripts" / "start_local_trainers.sh").read_text(encoding="utf-8")
        self.assertIn('PKMAI_WORKER_ID="local-trainer-${rank}"', launcher)
        self.assertIn('PKMAI_WORKER_RANK="$rank"', launcher)
        self.assertIn('PKMAI_WORKER_FLEET_SIZE="$trainer_count"', launcher)
        self.assertIn('--network host', launcher)


if __name__ == "__main__":
    unittest.main()
