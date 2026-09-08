import os
import tempfile
import unittest

import twoby2.orchestration as orch
from twoby2 import feature_enabled


class OrchestrationTests(unittest.TestCase):
    def test_emulator_budget_is_40_plus_9_battle_plus_1_full_watcher(self):
        b = orch.total_emulator_budget()
        self.assertEqual(b["navigation"], 40)
        self.assertEqual(b["battle_headless"], 8)
        self.assertEqual(b["battle_visible"], 1)
        self.assertEqual(b["battle_total"], 9)
        self.assertEqual(b["full_watcher"], 1)
        self.assertEqual(b["grand_total"], 50)
        self.assertLessEqual(b["grand_total"], b["cap"])

    def test_process_set_has_separate_pid_files(self):
        pids = [p["pid_file"] for p in orch.PROCESSES.values()]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertIn("battle_trainer_visible", orch.PROCESSES)
        self.assertEqual(orch.PROCESSES["battle_trainer_visible"]["script"],
                         "tools/battle_mirror_watch.py")
        self.assertEqual(orch.PROCESSES["battle_trainer_visible"]["workers"], 0)

    def test_double_start_guard(self):
        with tempfile.TemporaryDirectory() as d:
            # no pid file -> ok
            self.assertTrue(orch.check_no_double_start("web", root=d))
            pf = os.path.join(d, orch.PROCESSES["web"]["pid_file"])
            os.makedirs(os.path.dirname(pf), exist_ok=True)
            open(pf, "w").write(str(os.getpid()))    # our own pid = alive
            with self.assertRaises(orch.DoubleStartError):
                orch.check_no_double_start("web", root=d)
            # stale pid -> ok again
            open(pf, "w").write("999999999")
            self.assertTrue(orch.check_no_double_start("web", root=d))

    def test_status_shape_shows_workers_routes_versions_gates_blockers(self):
        rgp = {"active_routes": ["route1", "route2"],
               "regression_core_routes": [],
               "visible_watcher": {"route": "route2", "scenario_signature": "abc"}}
        s = orch.status_shape(route_group_plan=rgp,
                              nav_versions={"learner": 5, "champion": 4},
                              battle_versions={"learner": 3, "champion": 2})
        self.assertEqual(s["workers"], {"navigation": 40, "battle_headless": 8,
                                        "battle_visible": 1})
        self.assertEqual(s["route_groups"], ["route1", "route2"])
        self.assertEqual(s["battle_watcher_route"], "route2")
        self.assertEqual(s["navigation_champion_version"], 4)
        self.assertEqual(s["battle_learner_version"], 3)
        self.assertIn("gBattleMons", s["ram_blockers"])
        self.assertFalse(any(s["feature_gates"].values()))
        self.assertFalse(s["live"])

    def test_nothing_is_started(self):
        # the module exposes no run/spawn/Popen
        import inspect
        src = inspect.getsource(orch)
        for bad in ("subprocess", "Popen", "os.system", "osascript", "spawn"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
