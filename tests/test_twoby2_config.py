import os
import unittest

import twoby2.config as cfg


class WorkerConfigTests(unittest.TestCase):
    def test_final_architecture_is_40_beginning_0_anchor_9_battle(self):
        self.assertEqual(cfg.NAV_WORKERS, 40)
        self.assertEqual(cfg.NAV_BEGINNING_WORKERS, 40)
        self.assertEqual(cfg.NAV_ANCHOR_WORKERS, 0)
        self.assertEqual(cfg.BATTLE_WORKERS, 9)
        self.assertEqual(cfg.MAX_TOTAL_EMULATORS, 50)
        s = cfg.validate_worker_config()
        self.assertEqual(s["nav_beginning"], 40)
        self.assertEqual(s["nav_anchor"], 0)
        self.assertEqual(s["total_emulators"], 50)

    def test_40_plus_9_plus_full_watcher_must_not_exceed_50(self):
        self.assertEqual(cfg.NAV_WORKERS + cfg.BATTLE_WORKERS
                         + cfg.FULL_WATCHER_EMULATORS, cfg.MAX_TOTAL_EMULATORS)
        with self.assertRaises(cfg.WorkerConfigError):
            cfg.validate_worker_config(nav_workers=42, nav_beginning=42,
                                       nav_anchor=0, battle_workers=10)

    def test_any_anchor_worker_is_rejected(self):
        with self.assertRaises(cfg.WorkerConfigError):
            cfg.validate_worker_config(nav_workers=40, nav_beginning=32,
                                       nav_anchor=8, battle_workers=10)

    def test_all_nav_beginning_required(self):
        with self.assertRaises(cfg.WorkerConfigError):
            cfg.validate_worker_config(nav_workers=40, nav_beginning=39,
                                       nav_anchor=0, battle_workers=10)

    def test_needs_a_battle_worker(self):
        with self.assertRaises(cfg.WorkerConfigError):
            cfg.validate_worker_config(battle_workers=0)

    def test_roster_exact_counts_and_identical_starts(self):
        r = cfg.worker_roster()
        self.assertEqual(len(r), 49)
        nav = [w for w in r if w["system"] == "navigation"]
        bat = [w for w in r if w["system"] == "battle"]
        self.assertEqual(len(nav), 40)
        self.assertEqual(len(bat), 9)
        self.assertTrue(all(w["start_kind"] == "beginning" for w in nav))
        self.assertEqual({w["start_state"] for w in nav}, {cfg.CANONICAL_NAV_START})
        self.assertEqual(sum(w["start_kind"] == "anchor" for w in r), 0)
        cfg.assert_all_navigation_starts_identical(r)
        self.assertEqual(r, cfg.worker_roster())

    def test_assert_all_navigation_starts_identical_rejects_a_stray_start(self):
        r = cfg.worker_roster()
        r[0]["start_state"] = "stage_2.state"
        with self.assertRaises(cfg.WorkerConfigError):
            cfg.assert_all_navigation_starts_identical(r)

    def test_battle_split_is_8_headless_plus_1_visible(self):
        self.assertEqual(cfg.BATTLE_HEADLESS_WORKERS, 8)
        self.assertEqual(cfg.BATTLE_VISIBLE_WORKERS, 1)
        s = cfg.battle_worker_split()
        self.assertEqual((s["headless"], s["visible"], s["total"]), (8, 1, 9))
        r = cfg.worker_roster()
        battle = [w for w in r if w["system"] == "battle"]
        self.assertEqual(sum(w["render"] == "headless" for w in battle), 8)
        self.assertEqual(sum(w["render"] == "visible" for w in battle), 1)
        vis = next(w for w in battle if w["render"] == "visible")
        self.assertEqual(vis["window_title"], "PKMai – BATTLE")

    def test_canonical_nav_start_resolves_via_registry_and_verifies_sha(self):
        p = cfg.canonical_nav_start_path()
        self.assertTrue(p.endswith("StartGame.state"))
        self.assertTrue(os.path.isabs(p))


if __name__ == "__main__":
    unittest.main()
