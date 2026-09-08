import json
import os
import tempfile
import unittest

import twoby2.reset_tools as rt


class ResetMatrixTests(unittest.TestCase):
    def _dirs(self, d):
        for sub in ("nav/ck", "nav", "bat/ck", "bat"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        open(os.path.join(d, "nav/ck", "navigation_champion.zip"), "w").write("navchamp")
        open(os.path.join(d, "bat/ck", "battle_champion.zip"), "w").write("batchamp")
        return dict(nav_ckpt=os.path.join(d, "nav/ck"),
                    nav_stats=os.path.join(d, "nav"),
                    bat_ckpt=os.path.join(d, "bat/ck"),
                    bat_stats=os.path.join(d, "bat"))

    def test_navigation_reset_only_touches_navigation(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            plan = rt.plan_navigation_reset(nav_ckpt_dir=p["nav_ckpt"],
                                            nav_stats_dir=p["nav_stats"])
            for path in plan["touch"]:
                self.assertNotIn("bat", os.path.relpath(path, d).split(os.sep)[0])
            self.assertIn("savestates", plan["preserves"])
            res = rt.apply_plan(plan, dry_run=False)
            self.assertTrue(os.path.exists(os.path.join(p["nav_ckpt"], "navigation_learner.zip")))
            self.assertEqual(open(os.path.join(p["nav_ckpt"], "navigation_learner.zip")).read(),
                             "navchamp")
            self.assertTrue(os.path.exists(os.path.join(p["bat_ckpt"], "battle_champion.zip")))

    def test_fresh_obs_schema_deletes_champion_but_preserves_the_movement_graph(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            for f in ("navigation_learner.zip", "navigation_latest.zip"):
                open(os.path.join(p["nav_ckpt"], f), "w").write("x")
            open(os.path.join(p["nav_stats"], "champion_score.json"), "w").write("{}")
            open(os.path.join(p["nav_stats"], "movement_graph_v1.json"), "w").write("{}")
            plan = rt.plan_navigation_reset(nav_ckpt_dir=p["nav_ckpt"],
                                            nav_stats_dir=p["nav_stats"],
                                            fresh_obs_schema=True)
            self.assertTrue(all(a == "delete" for a, _ in plan["actions"]))
            self.assertIn("movement_graph_v1.json", plan["preserves"])
            rt.apply_plan(plan, dry_run=False)
            self.assertFalse(os.path.exists(os.path.join(p["nav_ckpt"], "navigation_champion.zip")))
            self.assertFalse(os.path.exists(os.path.join(p["nav_stats"], "champion_score.json")))
            self.assertTrue(os.path.exists(os.path.join(p["nav_stats"], "movement_graph_v1.json")))
            self.assertTrue(os.path.exists(os.path.join(p["bat_ckpt"], "battle_champion.zip")))

    def test_battle_reset_without_a_ppo_champion_removes_the_stale_learner(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            os.unlink(os.path.join(p["bat_ckpt"], "battle_champion.zip"))  # rule fallback
            learner = os.path.join(p["bat_ckpt"], "battle_learner.zip")
            resume = os.path.join(p["bat_ckpt"], "battle_resume.zip")
            open(learner, "w").write("old-broken-learner")
            open(resume, "w").write("old-broken-resume")
            plan = rt.plan_battle_reset(battle_ckpt_dir=p["bat_ckpt"],
                                        battle_stats_dir=p["bat_stats"])
            rt.apply_plan(plan, dry_run=False)
            self.assertFalse(os.path.exists(learner))
            self.assertFalse(os.path.exists(resume))

    def test_battle_reset_keeps_scenarios_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            open(os.path.join(p["bat_stats"], "scenario_pool.json"), "w").write("{}")
            plan = rt.plan_battle_reset(battle_ckpt_dir=p["bat_ckpt"],
                                        battle_stats_dir=p["bat_stats"])
            self.assertIn("scenario_pool.json", plan["preserves"])
            rt.apply_plan(plan, dry_run=False)
            self.assertTrue(os.path.exists(os.path.join(p["bat_stats"], "scenario_pool.json")))

    def test_battle_reset_wipe_scenarios_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            pool = os.path.join(p["bat_stats"], "scenario_pool.json")
            open(pool, "w").write("{}")
            plan = rt.plan_battle_reset(battle_ckpt_dir=p["bat_ckpt"],
                                        battle_stats_dir=p["bat_stats"],
                                        wipe_scenarios=True)
            rt.apply_plan(plan, dry_run=False)
            self.assertFalse(os.path.exists(pool))

    def test_full_reset_refused_without_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            plan = rt.plan_full_reset(nav_ckpt_dir=p["nav_ckpt"], nav_stats_dir=p["nav_stats"],
                                      battle_ckpt_dir=p["bat_ckpt"], battle_stats_dir=p["bat_stats"])
            self.assertTrue(plan["refused"])
            self.assertFalse(rt.apply_plan(plan, dry_run=False)["applied"])

    def test_full_reset_with_confirmation_preserves_data(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            plan = rt.plan_full_reset(nav_ckpt_dir=p["nav_ckpt"], nav_stats_dir=p["nav_stats"],
                                      battle_ckpt_dir=p["bat_ckpt"], battle_stats_dir=p["bat_stats"],
                                      confirm_full=True)
            self.assertFalse(plan.get("refused"))
            self.assertIn("savestates", plan["preserves"])
            self.assertIn("exploration_memory", plan["preserves"])

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._dirs(d)
            plan = rt.plan_navigation_reset(nav_ckpt_dir=p["nav_ckpt"],
                                            nav_stats_dir=p["nav_stats"])
            res = rt.apply_plan(plan, dry_run=True)
            self.assertFalse(res["applied"])
            self.assertFalse(os.path.exists(os.path.join(p["nav_ckpt"], "navigation_learner.zip")))


if __name__ == "__main__":
    unittest.main()
