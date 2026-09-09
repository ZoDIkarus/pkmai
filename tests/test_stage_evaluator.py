import unittest

from stage_evaluator import should_block_stage, stage_step_limit, summarize_stage_results


class StageEvaluatorTests(unittest.TestCase):
    def test_summarizes_fixed_seed_episode_outcomes_by_stage(self):
        summary = summarize_stage_results([
            {"stage": "stairs_down", "success": True, "steps": 100},
            {"stage": "stairs_down", "success": False, "steps": 900},
            {"stage": "stairs_down", "success": True, "steps": 200},
        ])

        self.assertEqual(summary["stairs_down"]["episodes"], 3)
        self.assertEqual(summary["stairs_down"]["successes"], 2)
        self.assertAlmostEqual(summary["stairs_down"]["success_rate"], 2 / 3)
        self.assertEqual(summary["stairs_down"]["median_success_steps"], 150.0)

    def test_stage_step_limits_bound_each_fixed_seed_episode(self):
        self.assertEqual(stage_step_limit("intro_complete"), 256)
        self.assertEqual(stage_step_limit("stairs_down"), 640)
        self.assertEqual(stage_step_limit("left_house"), 768)
        self.assertEqual(stage_step_limit("starter"), 1024)

    def test_three_terminal_failures_block_downstream_evaluation_early(self):
        self.assertFalse(should_block_stage({"episodes": 2, "successes": 0}))
        self.assertFalse(should_block_stage({"episodes": 3, "successes": 1}))
        self.assertTrue(should_block_stage({"episodes": 3, "successes": 0}))


if __name__ == "__main__":
    unittest.main()
