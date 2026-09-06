import unittest

from pokemon_env import PokemonFireRedEnv, short_cycle_repeats, stuck_loop_penalty


class ShortCycleRepeatsTests(unittest.TestCase):
    def test_detects_repeating_backtrack_cycle(self):
        a = (3, 0, 10, 10)
        b = (3, 0, 11, 10)
        self.assertEqual(short_cycle_repeats([a, b, a, b, a, b]), 2)

    def test_ignores_single_necessary_backtrack(self):
        a = (3, 0, 10, 10)
        b = (3, 0, 11, 10)
        self.assertEqual(short_cycle_repeats([a, b, a]), 0)

    def test_ignores_non_repeating_route(self):
        path = [
            (3, 0, 10, 10),
            (3, 0, 11, 10),
            (3, 0, 12, 10),
            (3, 0, 12, 9),
            (3, 0, 13, 9),
            (3, 0, 14, 9),
        ]
        self.assertEqual(short_cycle_repeats(path), 0)


class V17RewardTuningTests(unittest.TestCase):
    def test_gameplay_time_and_combat_rewards_favor_progress(self):
        self.assertEqual(PokemonFireRedEnv.GAMEPLAY_STEP_COST, -0.001)
        self.assertEqual(PokemonFireRedEnv.ENEMY_FAINT_REWARD, 2.0)
        self.assertEqual(PokemonFireRedEnv.LEVEL_GAIN_REWARD, 10.0)

    def test_stuck_penalty_grows_continuously_without_a_discontinuity(self):
        self.assertEqual(stuck_loop_penalty(59), 0.0)
        self.assertEqual(stuck_loop_penalty(60), -0.001)
        self.assertEqual(stuck_loop_penalty(180), -0.121)
        self.assertEqual(stuck_loop_penalty(400), -0.341)


if __name__ == "__main__":
    unittest.main()
