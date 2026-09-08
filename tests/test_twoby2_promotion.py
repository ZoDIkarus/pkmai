import unittest

import twoby2.promotion as promo
from twoby2.retention import EARLY_TRANSITIONS


def beginning_runs(n, max_stage=3, transitions=None):
    transitions = transitions or {1: 0.95, 2: 0.9}
    return [{"start_kind": "beginning", "completed": True, "evaluable": True,
             "max_stage": max_stage, "transitions": dict(transitions)}
            for _ in range(n)]


def champ(max_stage=3, early=0.95, version=5, **extra):
    d = {"max_stage": max_stage, "version": version,
         "early_rates": {k: early for k in EARLY_TRANSITIONS},
         "stage_reach_rate": 0.9}
    d.update(extra)
    return d


def cand(runs, early=0.95, samples=150, anchor_runs=0, **extra):
    d = {
        "beginning_runs": runs,
        "anchor_runs": [{"start_kind": "anchor", "max_stage": 6}
                        for _ in range(anchor_runs)],
        "early_rates": {k: early for k in EARLY_TRANSITIONS},
        "early_samples": {k: samples for k in EARLY_TRANSITIONS},
        "version": 6,
    }
    d.update(extra)
    return d


class PromotionTests(unittest.TestCase):
    def test_needs_100_beginning_full_runs(self):
        d = promo.evaluate_promotion(champion=champ(),
                                     candidate=cand(beginning_runs(40, max_stage=4)))
        self.assertFalse(d.promote)
        self.assertIn("beginning-full runs", d["reason"])

    def test_anchor_runs_are_rejected_and_counted_never_mixed_in(self):
        runs = beginning_runs(150, max_stage=3, transitions={1: 0.95, 2: 0.92})
        d = promo.evaluate_promotion(champion=champ(max_stage=3),
                                     candidate=cand(runs, anchor_runs=99))
        self.assertEqual(d["beginning_full_runs"], 150)
        self.assertEqual(d["anchor_runs"], 99)
        self.assertFalse(d["used_anchor_for_eval"])
        self.assertIn("rejected 99 anchor runs", d["reason"])

    def test_anchor_only_candidate_cannot_promote(self):
        d = promo.evaluate_promotion(champion=champ(), candidate=cand([], anchor_runs=500))
        self.assertFalse(d.promote)
        self.assertFalse(d["used_anchor_for_eval"])

    def test_incomparable_battle_versions_block_promotion(self):
        c = champ(max_stage=3, battle_champion_version=3, battle_champion_sha256="aaa")
        d = promo.evaluate_promotion(
            champion=c,
            candidate=cand(beginning_runs(150, max_stage=4),
                           battle_champion_version=4, battle_champion_sha256="bbb"))
        self.assertFalse(d.promote)
        self.assertIn("different battle-champion versions", d["reason"])
        self.assertFalse(d["comparable"])

    def test_same_battle_version_is_comparable(self):
        c = champ(max_stage=3, battle_champion_version=3, battle_champion_sha256="aaa")
        d = promo.evaluate_promotion(
            champion=c,
            candidate=cand(beginning_runs(150, max_stage=4,
                                          transitions={1: 0.95, 2: 0.92, 3: 0.88}),
                           battle_champion_version=3, battle_champion_sha256="aaa"))
        self.assertTrue(d.promote, d["reason"])

    def test_single_lucky_deep_run_is_not_credited_as_depth(self):
        runs = beginning_runs(120, max_stage=4, transitions={1: 0.6, 2: 0.6, 3: 0.6})
        runs[0]["max_stage"] = 6
        d = promo.evaluate_promotion(champion=champ(max_stage=4), candidate=cand(runs))
        self.assertIn("noise", d["reason"])
        self.assertFalse(d.promote)

    def test_deeper_but_forgot_early_game_is_rejected(self):
        runs = beginning_runs(150, max_stage=4)
        c = cand(runs)
        c["early_rates"]["pallet_route1"] = 0.4   # forgot a post-parcel gate
        d = promo.evaluate_promotion(champion=champ(max_stage=3), candidate=c)
        self.assertFalse(d.promote)
        self.assertFalse(d["retention"]["passed"])

    def test_precanonical_regression_does_not_block(self):
        runs = beginning_runs(150, max_stage=4,
                              transitions={1: 0.95, 2: 0.92, 3: 0.88})
        c = cand(runs)
        c["early_rates"]["starter"] = 0.1        # precanonical -> ignored
        c["early_rates"]["intro"] = 0.1
        d = promo.evaluate_promotion(champion=champ(max_stage=3), candidate=c)
        self.assertTrue(d.promote, d["reason"])

    def test_deeper_with_retained_early_game_is_accepted(self):
        runs = beginning_runs(150, max_stage=4,
                              transitions={1: 0.95, 2: 0.92, 3: 0.88})
        d = promo.evaluate_promotion(champion=champ(max_stage=3), candidate=cand(runs))
        self.assertTrue(d.promote, d["reason"])
        self.assertEqual(d["candidate_max_stage"], 4)

    def test_weak_onward_transition_blocks_promotion(self):
        runs = beginning_runs(150, max_stage=4, transitions={1: 0.95, 2: 0.5})
        d = promo.evaluate_promotion(champion=champ(max_stage=3), candidate=cand(runs))
        self.assertFalse(d.promote)
        self.assertIn("transition", d["reason"])

    def test_hard_regression_reset_scope_is_navigation_learner_only(self):
        sc = promo.nav_hard_regression_reset_scope()
        self.assertEqual(sc["resets"], ["navigation_learner <- navigation_champion"])
        self.assertIn("battle_champion", sc["never_touches"])
        self.assertIn("savestates", sc["never_touches"])


if __name__ == "__main__":
    unittest.main()
