import unittest

from cluster_config import load_cluster_runtime_settings


class ClusterRuntimeSettingsTests(unittest.TestCase):
    def test_uses_deployment_runner_limit(self):
        settings = load_cluster_runtime_settings(
            {"PKMAI_CLUSTER_ENV_RUNNERS": "11", "PKMAI_CLUSTER_AGENTS_PER_RUNNER": "1"}
        )
        self.assertEqual(settings.env_runners, 11)
        self.assertEqual(settings.agents_per_runner, 1)
