import unittest

import twoby2.battle_promotion as bp


def metrics(win=0.75, trainer=0.7, wild=0.8, wipe=0.05, resid_win=0.55,
            turns_win=6, flee_rate=0.0, invalid=0, aborted=0, menu_stalls=0,
            unreadable=0, terminal_unknown=0, timeouts=0, trainer_flees=0,
            loops=0, per_area=None, per_kind=None, episodes=250, cov=5,
            n_trainer=2, n_wild=3):
    return {
        "episodes": episodes, "win_rate": win, "trainer_win_rate": trainer,
        "wild_win_rate": wild, "wipe_rate": wipe,
        "flee_rate": flee_rate,
        "avg_residual_hp_on_win": resid_win, "avg_turns_on_win": turns_win,
        "invalid_actions": invalid, "aborted_macros": aborted,
        "menu_stalls": menu_stalls, "unreadable_states": unreadable,
        "terminal_unknown": terminal_unknown, "timeouts": timeouts,
        "trainer_flees": trainer_flees, "switch_loops": loops,
        "per_area_win_rate": per_area or {"route1": win, "route22": win},
        "per_kind_win_rate": per_kind or {"trainer": trainer, "wild": wild},
        "scenario_coverage": cov, "n_trainer": n_trainer, "n_wild": n_wild,
    }


class CoverageGateTests(unittest.TestCase):
    def test_needs_enough_eval_episodes(self):
        d = bp.evaluate_battle_promotion(candidate=metrics(episodes=50),
                                         champion=metrics())
        self.assertFalse(d.promote)
        self.assertIn("eval episodes", d["reason"])

    def test_needs_several_distinct_scenarios(self):
        d = bp.evaluate_battle_promotion(candidate=metrics(cov=1),
                                         champion=metrics())
        self.assertFalse(d.promote)
        self.assertIn("distinct eval scenarios", d["reason"])

    def test_needs_both_trainer_and_wild(self):
        d = bp.evaluate_battle_promotion(candidate=metrics(n_trainer=0),
                                         champion=metrics())
        self.assertFalse(d.promote)
        self.assertIn("trainer battle", d["reason"])


class SafetyGateTests(unittest.TestCase):
    def test_any_diagnosed_fault_hard_blocks_even_with_a_huge_win_gain(self):
        for k in bp.SAFETY_ZERO_METRICS:
            cand = metrics(win=0.99, wild=0.99, trainer=0.99)
            cand[k] = 1
            d = bp.evaluate_battle_promotion(candidate=cand,
                                             champion=metrics(win=0.3))
            self.assertFalse(d.promote, k)
            self.assertIn("safety gate", d["reason"])
            self.assertIn(k, d["reason"])

    def test_wild_flee_in_eval_blocks_promotion(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.9, flee_rate=0.05),
            champion=metrics(win=0.5))
        self.assertFalse(d.promote)
        self.assertIn("flee_rate", d["reason"])


class PerformanceLadderTests(unittest.TestCase):
    def test_lucky_win_without_margin_does_not_promote(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.76), champion=metrics(win=0.75))
        self.assertFalse(d.promote)
        self.assertIn("no meaningful improvement", d["reason"])

    def test_three_point_win_gain_promotes(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.82, wild=0.85, trainer=0.78),
            champion=metrics(win=0.75, wild=0.8, trainer=0.7))
        self.assertTrue(d.promote, d["reason"])

    def test_same_winrate_more_residual_hp_on_win_promotes(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.75, resid_win=0.7),
            champion=metrics(win=0.75, resid_win=0.5))
        self.assertTrue(d.promote, d["reason"])

    def test_same_winrate_same_hp_fewer_turns_promotes(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.75, resid_win=0.55, turns_win=5.0),
            champion=metrics(win=0.75, resid_win=0.55, turns_win=6.0))
        self.assertTrue(d.promote, d["reason"])

    def test_residual_hp_gain_of_two_points_is_not_enough(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.75, resid_win=0.57),
            champion=metrics(win=0.75, resid_win=0.55))
        self.assertFalse(d.promote)


class RegressionTests(unittest.TestCase):
    def test_per_area_regression_blocks_promotion(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.85, per_area={"route1": 0.9, "route22": 0.4}),
            champion=metrics(win=0.72, per_area={"route1": 0.7, "route22": 0.72}))
        self.assertFalse(d.promote)
        self.assertTrue(any("route22" in str(r) for r in d["regressions"]))

    def test_per_kind_regression_blocks_promotion(self):
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.85, trainer=0.4,
                              per_kind={"trainer": 0.4, "wild": 0.95}),
            champion=metrics(win=0.72, trainer=0.7,
                             per_kind={"trainer": 0.7, "wild": 0.74}))
        self.assertFalse(d.promote)
        self.assertTrue(any("trainer" in str(r) for r in d["regressions"]))

    def test_core_regression_and_multiple_regressions_is_hard(self):
        cand_core = metrics(win=0.4, trainer=0.3, wild=0.4,
                            per_kind={"trainer": 0.3, "wild": 0.4})
        champ_core = metrics(win=0.85, trainer=0.85, wild=0.85,
                             per_kind={"trainer": 0.85, "wild": 0.85})
        d = bp.evaluate_battle_promotion(
            candidate=metrics(win=0.5, trainer=0.4, wild=0.5,
                              per_kind={"trainer": 0.4, "wild": 0.5}),
            champion=metrics(win=0.85, trainer=0.85, wild=0.85,
                             per_kind={"trainer": 0.85, "wild": 0.85}),
            candidate_core=cand_core, champion_core=champ_core)
        self.assertFalse(d.promote)
        self.assertTrue(d["hard_regression"])


class FirstChampionTests(unittest.TestCase):
    def test_first_champion_needs_clean_safe_covered_and_80pct(self):
        d = bp.first_champion_ok(metrics(win=0.85, wipe=0.05))
        self.assertTrue(d.promote, d["reason"])

    def test_first_champion_blocked_by_low_coverage(self):
        d = bp.first_champion_ok(metrics(win=0.95, cov=1, n_trainer=0))
        self.assertFalse(d.promote)

    def test_first_champion_blocked_by_a_single_invalid_action(self):
        m = metrics(win=0.95)
        m["invalid_actions"] = 1
        d = bp.first_champion_ok(m)
        self.assertFalse(d.promote)
        self.assertIn("invalid_actions", d["reason"])

    def test_first_champion_blocked_below_80pct_wins(self):
        d = bp.first_champion_ok(metrics(win=0.7))
        self.assertFalse(d.promote)
        self.assertIn("0.80", d["reason"])


class ResetScopeTests(unittest.TestCase):
    def test_reset_scope_is_battle_only(self):
        sc = bp.reset_scope_for_hard_regression()
        self.assertEqual(sc["resets"], ["battle_learner <- battle_champion"])
        self.assertIn("navigation_champion", sc["never_touches"])
        self.assertIn("scenario_pool", sc["never_touches"])


if __name__ == "__main__":
    unittest.main()
