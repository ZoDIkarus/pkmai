import unittest

from stage_evaluator import summarize_stage_results


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


if __name__ == "__main__":
    unittest.main()
