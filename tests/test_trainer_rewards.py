import unittest

from trainer_rewards import TrainerRewards


class TrainerRewardsTests(unittest.TestCase):
    def test_start_and_win_each_pay_once_after_an_ongoing_trainer_battle(self):
        rewards = TrainerRewards()
        self.assertEqual(rewards.update(True, True, 12, 0), [("trainer_battle_start", 50.0)])
        self.assertEqual(rewards.update(True, True, 12, 0), [])
        self.assertEqual(rewards.update(True, True, 12, 1), [("trainer_battle_won", 50.0)])
        self.assertEqual(rewards.update(False, False, 12, 1), [])
        self.assertEqual(rewards.update(True, True, 12, 0), [])


if __name__ == "__main__":
    unittest.main()
