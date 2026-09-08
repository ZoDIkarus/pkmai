import unittest
from collections import deque
from unittest.mock import Mock

import train


class Route1RegressionGuardTests(unittest.TestCase):
    def test_route1_regression_counts_strikes_and_rolls_back_on_third(self):
        callback = train.MilestoneCheckpointCallback.__new__(
            train.MilestoneCheckpointCallback
        )
        metrics = {"episodes": 8, "full_episodes": 8}
        callback.min_eval_episodes = 8
        callback.min_full_episodes = 8
        callback.num_timesteps = 123
        callback.regression_strikes = 0
        callback.steps_since_champion_update = 0
        callback._champion_route1_reach = 0.84
        callback._metrics = Mock(return_value=metrics)
        callback._score = Mock(return_value=(1,))
        callback._protected_regression = Mock(return_value=False)
        callback._route1_reach_rate = Mock(return_value=0.66)
        callback._rollback_to_champion = Mock(return_value=True)
        callback._maybe_advance_nav_horizon = Mock()
        callback.model = Mock()
        callback.recent = deque([1])
        callback.recent_full = deque([1])

        for _ in range(3):
            self.assertTrue(callback._evaluate())

        self.assertEqual(callback.last_eval_result, "route1_reach_regressed")
        self.assertEqual(callback.regression_strikes, 3)
        callback._rollback_to_champion.assert_called_once_with()
        callback._maybe_advance_nav_horizon.assert_not_called()


if __name__ == "__main__":
    unittest.main()
