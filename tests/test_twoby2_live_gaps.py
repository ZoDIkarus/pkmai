"""Remaining 2x2 live-gap regressions (F13-F19) not covered elsewhere."""
import importlib.util
import json
import os
import tempfile
import unittest

from twoby2 import router as R
from twoby2 import reward_split as rs

_CAN = os.path.join(os.path.dirname(__file__), "..", "tools", "live_battle_canary.py")
_spec = importlib.util.spec_from_file_location("live_battle_canary", _CAN)
canary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canary)


class RewardBufferSeparationTests(unittest.TestCase):
    def test_no_battle_reward_into_navigation_buffer(self):             # F13
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"new_coord": 0.3, "damage_dealt": 1.0})
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"enemy_ko": 1.0})
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"battle_win": 3.0})

    def test_no_navigation_reward_into_battle_buffer(self):             # F14
        for bad in ({"tile": 0.05}, {"new_map": 50.0}, {"story_flag": 1.0},
                    {"world_stage": 1.0}, {"navigation_transition": 1.0}):
            with self.assertRaises(rs.RewardChannelError):
                rs.battle_reward({"damage_dealt": 0.5, **bad})

    def test_clean_channels_still_work(self):
        self.assertAlmostEqual(rs.navigation_reward({"new_coord": 0.3, "wipe": -3.0}), -2.7)
        self.assertAlmostEqual(rs.battle_reward({"damage_dealt": 0.4, "enemy_ko": 1.0}), 1.4)


class FullAgentAndWatcherSameChampionTests(unittest.TestCase):
    def test_full_worker_and_full_watcher_use_the_same_pinned_champions(self):  # F15
        bc = R.PolicyVersion(4, "bsha", "ppo", 7, "battle_obs_v1")
        worker = R.BattlePolicyRouter(consumer_mode=R.NAV_TRAINING,
                                      pinned_battle_champion=bc,
                                      battle_champion_valid=True,
                                      battle_champion_source="ppo")
        watcher = R.BattlePolicyRouter(consumer_mode=R.WATCHER,
                                       pinned_battle_champion=bc,
                                       battle_champion_valid=True,
                                       battle_champion_source="ppo")
        wr = worker.route(in_battle=True, execution_check=(True, []))
        wa = watcher.route(in_battle=True, execution_check=(True, []))
        self.assertEqual(wr["policy"], R.POLICY_BATTLE_CHAMPION)
        self.assertEqual(wa["policy"], R.POLICY_BATTLE_CHAMPION)
        self.assertEqual(wr["battle_policy_version"], wa["battle_policy_version"])
        # overworld: worker=learner, watcher=champion, but neither loads the battle learner
        self.assertFalse(worker.route(in_battle=True, execution_check=(True, []))["loads_battle_learner"])
        self.assertFalse(watcher.route(in_battle=True, execution_check=(True, []))["loads_battle_learner"])

    def test_watcher_never_loads_a_learner(self):
        self.assertFalse(R.WatcherBattlePin().loads_learner())


class CanaryArtifactTests(unittest.TestCase):
    def test_canary_uses_the_registered_protected_probe_seed(self):
        from twoby2 import protected_assets as pa
        self.assertEqual(canary.SEED_REL, pa.RAM_PROBE_ROUTE1_SEED["state"])
        self.assertEqual(canary.SEED_SHA, pa.RAM_PROBE_ROUTE1_SEED["state_sha256"])
        # the canary imports the REAL driver, not a simulated one
        import inspect
        src = inspect.getsource(canary)
        self.assertIn("EmulatorBattleDriver", src)
        self.assertNotIn("SimulatedBattleDriver", src)
        self.assertNotIn("FakeDriver", src)

    def test_state_change_detector(self):
        before = {"in_battle": True, "menu_state": "move",
                  "enemy_active": {"cur_hp": 20}, "player_active": {"cur_hp": 25, "moves": [{"pp": 10}]}}
        after_hp = {"in_battle": True, "menu_state": "main",
                    "enemy_active": {"cur_hp": 12}, "player_active": {"cur_hp": 25, "moves": [{"pp": 10}]}}
        d = canary._state_changed(after_hp, after_hp, 20, 25, [10])
        # helper signature: (before, after, ehp0, php0, pp0)
        d2 = canary._state_changed(before, after_hp, 20, 25, [10])
        self.assertTrue(d2["enemy_hp_changed"])
        self.assertTrue(d2["any"])
        d3 = canary._state_changed(before, {"in_battle": False, "menu_state": None,
                                            "enemy_active": {}, "player_active": {}},
                                   20, 25, [10])
        self.assertTrue(d3["any"])   # battle ended -> counts as progressed

    def test_stale_report_is_not_a_pass_for_preflight(self):           # F18
        _p = os.path.join(os.path.dirname(__file__), "..", "tools", "twoby2_preflight.py")
        s = importlib.util.spec_from_file_location("pf2", _p)
        pf = importlib.util.module_from_spec(s)
        s.loader.exec_module(pf)
        with tempfile.TemporaryDirectory() as d:
            cdir = os.path.join(d, "x"); os.makedirs(cdir)
            json.dump({"overall": "FAIL"}, open(os.path.join(cdir, "canary_report.json"), "w"))
            old = pf.CANARY_ROOT
            try:
                pf.CANARY_ROOT = d
                by = {c.name: c for c in pf.run_checks()}
                self.assertFalse(by["real_canary_passed"].ok)
            finally:
                pf.CANARY_ROOT = old


class ProtectedFilesUnchangedTests(unittest.TestCase):
    def test_master_and_route1_seeds_are_byte_identical(self):         # F19
        from twoby2 import protected_assets as pa
        self.assertTrue(pa.verify_master()["sha256_ok"])
        reg = pa.default_registry()
        for lid, want in (
            ("route1_manual_battle_seed", pa.ROUTE1_MANUAL_SEED["state_sha256"]),
            ("protected_ram_probe_route1_seed", pa.RAM_PROBE_ROUTE1_SEED["state_sha256"]),
        ):
            a = next(x for x in reg.assets if x.logical_id == lid)
            self.assertEqual(pa.sha256_file(a.abs_files()[0]), want, lid)


if __name__ == "__main__":
    unittest.main()
