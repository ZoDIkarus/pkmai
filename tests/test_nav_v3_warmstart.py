import importlib.util
import os
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "warmstart_navigation_v3",
    os.path.join(ROOT, "tools", "warmstart_navigation_v3.py"),
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class _Policy:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class NavigationV3WarmstartTests(unittest.TestCase):
    def test_only_five_appended_columns_are_zero_initialized(self):
        old = {
            "mlp_extractor.policy_net.0.weight": torch.arange(12).reshape(3, 4),
            "mlp_extractor.value_net.0.weight": torch.arange(8).reshape(2, 4),
            "action_net.weight": torch.ones(7, 3),
        }
        new = {
            "mlp_extractor.policy_net.0.weight": torch.full((3, 9), 99),
            "mlp_extractor.value_net.0.weight": torch.full((2, 9), 99),
            "action_net.weight": torch.zeros(7, 3),
        }
        migrated, widened = MOD.migrate_policy_state(_Policy(old), _Policy(new))
        self.assertEqual(set(widened), set(MOD.WIDENED_KEYS))
        for key in MOD.WIDENED_KEYS:
            old_width = old[key].shape[1]
            self.assertTrue(torch.equal(migrated[key][:, :old_width], old[key]))
            self.assertEqual(torch.count_nonzero(migrated[key][:, old_width:]), 0)
        self.assertTrue(torch.equal(migrated["action_net.weight"], old["action_net.weight"]))

    def test_unexpected_schema_growth_fails_closed(self):
        old = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(2, 4),
            "mlp_extractor.value_net.0.weight": torch.zeros(2, 4),
        }
        new = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(2, 8),
            "mlp_extractor.value_net.0.weight": torch.zeros(2, 8),
        }
        with self.assertRaisesRegex(RuntimeError, "exactly five"):
            MOD.migrate_policy_state(_Policy(old), _Policy(new))


if __name__ == "__main__":
    unittest.main()
