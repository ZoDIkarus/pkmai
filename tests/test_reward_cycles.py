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
    def test_local_ten_worker_curriculum_reserves_two_frontier_scouts(self):
        env = object.__new__(PokemonFireRedEnv)
        env.rank = 8
        env.agent_count = 10
        env.saved_milestones = {
            "intro_complete", "stairs_down", "left_house", "starter", "progress_3",
        }
        env.full_chain_ready = False
        self.assertEqual(PokemonFireRedEnv._agent_role(env)[0], "scout")

    def test_frontier_scout_resumes_the_deepest_progress_checkpoint(self):
        env = object.__new__(PokemonFireRedEnv)
        env.rank = 8
        env.agent_count = 10
        env.saved_milestones = {
            "intro_complete", "stairs_down", "left_house", "starter", "progress_3",
        }
        env._discover_saved_milestones = lambda: list(env.saved_milestones)
        env._champion_full_starter_ready = lambda: False
        self.assertEqual(PokemonFireRedEnv._choose_episode_start(env), "progress_3")

    def test_frontier_scout_uses_the_shared_full_policy_context(self):
        env = object.__new__(PokemonFireRedEnv)
        env.training_objective = "scout"
        self.assertEqual(PokemonFireRedEnv._policy_objective(env), "full")

    def test_frontier_scout_gets_a_longer_stall_budget(self):
        env = object.__new__(PokemonFireRedEnv)
        env.training_objective = "scout"
        self.assertEqual(
            PokemonFireRedEnv._progress_stall_limit(env),
            PokemonFireRedEnv.SCOUT_STALL_TIMEOUT,
        )

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
