import unittest

from cluster_config import (
    ClusterCompatibilityError,
    ClusterSettings,
    build_environment_signature,
    validate_worker_registration,
)


class ClusterConfigTests(unittest.TestCase):
    def test_same_environment_produces_a_stable_signature(self):
        first = build_environment_signature(
            observation_shape=(64, 64, 1), nav_features=28, action_count=7
        )
        second = build_environment_signature(
            observation_shape=(64, 64, 1), nav_features=28, action_count=7
        )

        self.assertEqual(first, second)

    def test_changed_observation_is_rejected(self):
        settings = ClusterSettings(environment_signature="expected", version_window=2)

        with self.assertRaisesRegex(ClusterCompatibilityError, "signature"):
            validate_worker_registration(
                settings,
                worker_signature="wrong",
                worker_policy_version=1,
                master_policy_version=1,
            )

    def test_stale_policy_is_told_to_reload(self):
        settings = ClusterSettings(environment_signature="expected", version_window=2)

        decision = validate_worker_registration(
            settings,
            worker_signature="expected",
            worker_policy_version=1,
            master_policy_version=4,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "policy_reload_required")

    def test_current_policy_is_accepted(self):
        settings = ClusterSettings(environment_signature="expected", version_window=2)

        decision = validate_worker_registration(
            settings,
            worker_signature="expected",
            worker_policy_version=3,
            master_policy_version=4,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.policy_version, 4)
