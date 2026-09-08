import unittest

import twoby2.retention as ret


class RetentionGateTests(unittest.TestCase):
    def test_active_chain_starts_post_parcel_and_precanonical_excluded(self):
        # corrected 2026-09-07: precanonical (intro/stairs/left_house/starter)
        # is before the master state and is no longer a retention gate.
        self.assertEqual(ret.ACTIVE_EARLY_TRANSITIONS,
                         ("leave_oak_lab", "pallet_route1", "route1_viridian"))
        self.assertEqual(ret.EARLY_TRANSITIONS, ret.ACTIVE_EARLY_TRANSITIONS)
        for k in ("intro", "stairs_down", "left_house", "starter"):
            self.assertEqual(ret.classify_transition(k), "precanonical")

    def test_pass_when_candidate_matches_champion(self):
        rates = {k: 0.95 for k in ret.EARLY_TRANSITIONS}
        samples = {k: 200 for k in ret.EARLY_TRANSITIONS}
        r = ret.evaluate_retention(rates, rates, samples)
        self.assertTrue(r["passed"])
        self.assertFalse(r["hard_failure"])

    def test_absolute_drop_is_a_regression(self):
        champ = {k: 0.95 for k in ret.EARLY_TRANSITIONS}
        cand = dict(champ, pallet_route1=0.70)          # -0.25
        samples = {k: 200 for k in ret.EARLY_TRANSITIONS}
        r = ret.evaluate_retention(champ, cand, samples)
        self.assertFalse(r["passed"])
        self.assertIn("pallet_route1", r["regressions"])

    def test_undersampled_candidate_is_fail_closed(self):
        champ = {k: 0.95 for k in ret.EARLY_TRANSITIONS}
        samples = {k: 200 for k in ret.EARLY_TRANSITIONS}
        samples["route1_viridian"] = 5
        r = ret.evaluate_retention(champ, champ, samples)
        self.assertFalse(r["passed"])
        self.assertIn("route1_viridian", r["regressions"])

    def test_two_regressions_make_a_hard_failure(self):
        champ = {k: 0.95 for k in ret.EARLY_TRANSITIONS}
        cand = dict(champ, leave_oak_lab=0.5, pallet_route1=0.5)
        samples = {k: 200 for k in ret.EARLY_TRANSITIONS}
        r = ret.evaluate_retention(champ, cand, samples)
        self.assertTrue(r["hard_failure"])
        self.assertGreaterEqual(r["regression_count"], 2)

    def test_permille_inputs_are_tolerated(self):
        champ = {k: 950 for k in ret.EARLY_TRANSITIONS}   # permille
        cand = {k: 0.95 for k in ret.EARLY_TRANSITIONS}   # fraction
        samples = {k: 200 for k in ret.EARLY_TRANSITIONS}
        self.assertTrue(ret.evaluate_retention(champ, cand, samples)["passed"])

    def test_precanonical_keys_never_become_a_gate(self):
        gates = ret.default_gates(mastered_keys=("intro", "starter",
                                                 "leave_oak_lab", "pallet_route1"))
        self.assertEqual([g.key for g in gates], ["leave_oak_lab", "pallet_route1"])

    def test_missing_intro_starter_metrics_do_not_block(self):
        champ = {"leave_oak_lab": 0.95, "pallet_route1": 0.95,
                 "route1_viridian": 0.9, "intro": 0.99, "starter": 0.99}
        cand = {"leave_oak_lab": 0.95, "pallet_route1": 0.95, "route1_viridian": 0.9}
        samples = {k: 200 for k in champ}
        r = ret.evaluate_retention(champ, cand, samples)
        self.assertTrue(r["passed"])
        self.assertEqual(sorted(r["precanonical_ignored"]), ["intro", "starter"])


if __name__ == "__main__":
    unittest.main()
