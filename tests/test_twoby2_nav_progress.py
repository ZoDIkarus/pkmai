import unittest

from twoby2 import nav_progress as npg
from twoby2 import promotion as promo


class NavProgressExcludesLevelTests(unittest.TestCase):
    def test_level_up_without_geo_change_is_zero_progress(self):        # F1
        before = {"map_bank": 3, "map_id": 19, "x": 10, "y": 6,
                  "seen_coords": {(3, 19, 10, 6)}, "stage": 2, "badges": 0}
        after = {**before, "party_total_level": 40, "level": 12, "kos": 7}
        d = npg.navigation_progress_delta(before, after)
        self.assertEqual(d["total"], 0)
        self.assertTrue(d["same_geographic_position"])

    def test_xp_gain_without_geo_change_is_zero_progress(self):         # F2
        before = {"map_id": 19, "seen_coords": {(1,)}, "story_flags": {"a"}}
        after = {**before, "experience_reward": 999, "xp": 12345}
        self.assertEqual(npg.navigation_progress_delta(before, after)["total"], 0)

    def test_identical_navigation_different_levels_same_score(self):    # F4
        route = {"seen_coords": {(1,), (2,), (3,)}, "maps": {"m1", "m2"},
                 "stage": 2, "badges": 0}
        low = npg.navigation_progress_delta(
            {"seen_coords": set(), "maps": set(), "stage": 1, "badges": 0},
            {**route, "party_total_level": 12})
        high = npg.navigation_progress_delta(
            {"seen_coords": set(), "maps": set(), "stage": 1, "badges": 0},
            {**route, "party_total_level": 60, "kos": 200})
        self.assertEqual(low, high)

    def test_real_geo_progress_still_counts(self):
        d = npg.navigation_progress_delta(
            {"seen_coords": {(1,)}, "maps": {"m1"}, "stage": 2, "badges": 0},
            {"seen_coords": {(1,), (2,)}, "maps": {"m1", "m2"}, "stage": 3, "badges": 1})
        self.assertEqual(d["new_coords"], 1)
        self.assertEqual(d["new_maps"], 1)
        self.assertEqual(d["stage_transitions"], 1)
        self.assertEqual(d["new_badges"], 1)
        self.assertGreaterEqual(d["total"], 4)

    def test_assert_no_level_signal(self):
        npg.assert_no_level_signal({"new_coord": 1, "story_flag": 1})
        for bad in ({"level_up": 1}, {"team_level_up": 1}, {"enemy_ko": 1},
                    {"battle_win": 1}, {"party_total_level": 30},
                    {"experience_reward": 1}):
            with self.assertRaises(npg.NavProgressLeak):
                npg.assert_no_level_signal(bad)


class NavPromotionIgnoresLevelTests(unittest.TestCase):
    def _cand(self, runs, **extra):
        d = {"beginning_runs": runs,
             "early_rates": {"leave_oak_lab": 0.95, "pallet_route1": 0.95,
                             "route1_viridian": 0.9},
             "early_samples": {"leave_oak_lab": 200, "pallet_route1": 200,
                               "route1_viridian": 200}}
        d.update(extra)
        return d

    def _runs(self, n, stage=4):
        return [{"start_kind": "beginning", "completed": True, "evaluable": True,
                 "max_stage": stage, "transitions": {1: 0.95, 2: 0.92, 3: 0.9}}
                for _ in range(n)]

    def test_100_wild_wins_same_place_cannot_promote(self):             # F3
        champ = {"max_stage": 3, "early_rates":
                 {"leave_oak_lab": 0.95, "pallet_route1": 0.95, "route1_viridian": 0.9}}
        # candidate has NO geographic improvement, only 100 KOs + level
        cand = self._cand(
            [{"start_kind": "beginning", "completed": True, "evaluable": True,
              "max_stage": 3, "transitions": {1: 0.95, 2: 0.5},   # weak onward
              "wild_wins": 100, "kos": 300, "party_total_level": 55}
             for _ in range(150)])
        d = promo.evaluate_promotion(champion=champ, candidate=cand)
        self.assertFalse(d.promote)

    def test_promotion_result_carries_no_level_fields(self):            # F5
        champ = {"max_stage": 3, "level": 11, "kos": 4,
                 "early_rates": {"leave_oak_lab": 0.95, "pallet_route1": 0.95,
                                 "route1_viridian": 0.9}}
        cand = self._cand(self._runs(150), level=60, kos=999,
                          party_total_level=180)
        d = promo.evaluate_promotion(champion=champ, candidate=cand)
        for k in d:
            self.assertNotIn("level", str(k).lower())
            self.assertNotIn("kos", str(k).lower())
        # a deeper candidate that retained the early game still promotes
        self.assertTrue(d.promote, d["reason"])

    def test_strip_level_from_metrics_is_recursive(self):
        m = {"max_stage": 4, "party_total_level": 30,
             "eval": {"win_rate": 0.7, "level_up": 5, "battle_kos": 3},
             "early_rates": {"pallet_route1": 0.9, "team_level_up": 2}}
        out = npg.strip_level_from_promotion_metrics(m)
        self.assertNotIn("party_total_level", out)
        self.assertNotIn("level_up", out["eval"])
        self.assertNotIn("battle_kos", out["eval"])
        self.assertNotIn("team_level_up", out["early_rates"])
        self.assertIn("win_rate", out["eval"])
        self.assertIn("pallet_route1", out["early_rates"])


if __name__ == "__main__":
    unittest.main()
