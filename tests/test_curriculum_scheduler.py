import unittest

from curriculum import (
    local_frontier_roles,
    curriculum_roles,
    stage_speed_bonus,
    stage_is_confirmed,
    watcher_completion_stage,
    quality_snapshot,
    quality_is_better,
    adaptive_stage_limit,
    curriculum_progress_report,
)


class CurriculumRolesTests(unittest.TestCase):
    def test_local_frontier_uses_two_battle_specialists_after_the_starter_chain(self):
        roles = local_frontier_roles(
            10,
            {"intro_complete", "stairs_down", "left_house", "starter", "progress_3"},
        )
        self.assertEqual(len(roles), 10)
        self.assertEqual(roles.count("battle"), 2)
        self.assertEqual(roles.count("progress"), 4)
        self.assertEqual(roles[:4], ("intro", "stairs", "exit", "starter"))

    def test_uses_intro_frontier_until_intro_state_exists(self):
        roles = curriculum_roles(10, set(), {})
        self.assertEqual(roles.count("intro"), 8)
        self.assertEqual(roles.count("stairs"), 2)

    def test_revalidates_earliest_saved_stage_before_using_later_frontier(self):
        states = {"intro_complete", "stairs_down", "left_house"}
        self.assertEqual(curriculum_roles(10, states, {}).count("intro"), 8)
        status = {"intro_complete": {"recent": [True] * 7 + [False] * 3}}
        self.assertEqual(curriculum_roles(10, states, status).count("stairs"), 8)

    def test_watcher_intro_gate_holds_dynamic_frontier_until_visible_pass(self):
        states = {"intro_complete", "stairs_down", "left_house"}
        status = {"intro_complete": {"recent": [True] * 7 + [False] * 3}}
        blocked = curriculum_roles(10, states, status, {"intro_complete": {"passed": False}})
        released = curriculum_roles(10, states, status, {"intro_complete": {"passed": True}})
        self.assertEqual(blocked.count("intro"), 8)
        self.assertEqual(released.count("stairs"), 8)

    def test_keeps_starter_frontier_until_it_is_confirmed(self):
        states = {"intro_complete", "stairs_down", "left_house", "starter"}
        status = {
            "intro_complete": {"recent": [True] * 7 + [False] * 3},
            "stairs_down": {"recent": [True] * 7 + [False] * 3},
            "left_house": {"recent": [True] * 7 + [False] * 3},
            "starter": {"recent": [True] * 6},
        }
        roles = curriculum_roles(10, states, status)
        self.assertEqual(roles.count("starter"), 7)

    def test_advances_after_confirmed_starter_but_keeps_regression_workers(self):
        states = {"intro_complete", "stairs_down", "left_house", "starter"}
        status = {
            "intro_complete": {"recent": [True] * 7 + [False] * 3},
            "stairs_down": {"recent": [True] * 7 + [False] * 3},
            "left_house": {"recent": [True] * 7 + [False] * 3},
            "starter": {"recent": [True] * 7 + [False] * 3},
        }
        roles = curriculum_roles(10, states, status)
        self.assertGreaterEqual(roles.count("starter"), 1)
        self.assertGreaterEqual(roles.count("exit"), 1)
        self.assertIn("progress", roles)

    def test_stage_confirmation_requires_enough_recent_successes(self):
        self.assertFalse(stage_is_confirmed({"recent": [True] * 6}))
        self.assertFalse(stage_is_confirmed({"recent": [True] * 5 + [False] * 5}))
        self.assertTrue(stage_is_confirmed({"recent": [True] * 7 + [False] * 3}))

    def test_watcher_completes_current_best_stage_from_the_beginning(self):
        states = {"intro_complete", "stairs_down", "left_house", "starter"}
        # A saved achievement is the end-to-end demo target even before its
        # rolling training quality is confirmed.
        self.assertEqual(watcher_completion_stage(states, {}), "starter")
        self.assertEqual(watcher_completion_stage({"left_house"}, {}), "left_house")

    def test_quality_prefers_success_rate_then_faster_successes(self):
        slower = quality_snapshot({"starter": {"recent": [True] * 7 + [False] * 3, "median_success_steps": 800}})
        faster = quality_snapshot({"starter": {"recent": [True] * 7 + [False] * 3, "median_success_steps": 400}})
        reliable = quality_snapshot({"starter": {"recent": [True] * 8 + [False] * 2, "median_success_steps": 900}})
        self.assertTrue(quality_is_better(faster, slower))
        self.assertTrue(quality_is_better(reliable, faster))
        self.assertFalse(quality_is_better(slower, faster))

    def test_adaptive_limit_is_twice_best_but_never_exceeds_safe_fallback(self):
        self.assertEqual(adaptive_stage_limit(None, 800), 800)
        self.assertEqual(adaptive_stage_limit(320, 800), 640)
        self.assertEqual(adaptive_stage_limit(600, 800), 800)

    def test_speed_bonus_rewards_fast_success_without_penalizing_at_target(self):
        self.assertEqual(stage_speed_bonus(0, 500, 80.0), 80.0)
        self.assertEqual(stage_speed_bonus(250, 500, 80.0), 40.0)
        self.assertEqual(stage_speed_bonus(500, 500, 80.0), 0.0)

    def test_progress_report_explains_reliability_and_stage_start_steps(self):
        report = curriculum_progress_report(
            {
                "intro_complete": {
                    "recent": [True] * 8 + [False] * 2,
                    "attempts": 25,
                    "successes": 19,
                    "median_success_steps": 320,
                },
            }
        )
        self.assertIn("Intro abschließen", report)
        self.assertIn("8/10 = 80%", report)
        self.assertIn("320 Schritte ab Stage-Start", report)
        self.assertIn("bestätigt", report)


if __name__ == "__main__":
    unittest.main()
