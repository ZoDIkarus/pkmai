import unittest
from checkpoint_health import (
    party_health, may_replace_frontier, frontier_viable,
    FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN,
)


def mon(hp=20, maximum=20, status=0, pp=10):
    return dict(checksum_ok=True, cur_hp=hp, max_hp=maximum, status=status, moves=[{'pp':pp}])


class CheckpointHealthTests(unittest.TestCase):
    def test_two_fainted_and_two_hp_last_survivor_is_not_ready(self):
        self.assertFalse(party_health([mon(0,31),mon(0,16),mon(2,15)])['party_ready'])

    def test_all_members_need_health_and_usable_pp(self):
        for party in ([], [mon(3)], [mon(status=8)], [mon(pp=0)], [mon(),mon(0)]):
            self.assertFalse(party_health(party)['party_ready'])
        self.assertTrue(party_health([mon(),mon(18)])['party_ready'])

    def test_healthy_reclaims_hurt_anchor_at_equal_score_but_never_regresses(self):
        # 2026-09-07: a healthy party reclaims a hurt/legacy anchor WITHOUT a
        # new distance record (equal score is enough) - but it may never walk
        # the anchor's frontier_score backwards.
        old=dict(party_ready=False,frontier_score=100,frontier_metric_version=2)
        self.assertTrue(may_replace_frontier(old,100,party_health([mon(),mon()]),2))
        self.assertTrue(may_replace_frontier(old,120,party_health([mon(),mon()]),2))
        self.assertFalse(may_replace_frontier(old,30,party_health([mon(),mon()]),2))
        # a legacy anchor whose metric version is stale scores as 0 -> reclaimable
        stale=dict(party_ready=False,frontier_score=100,frontier_metric_version=0)
        self.assertTrue(may_replace_frontier(stale,5,party_health([mon(),mon()]),2))

    def test_no_progress_record_can_publish_dying_party(self):
        self.assertFalse(may_replace_frontier({},1000,party_health([mon(2)]),2))

    def test_health_refresh_at_same_depth_and_no_regression(self):
        old=dict(party_ready=True,party_min_hp_ratio=.8,frontier_score=40,frontier_metric_version=2)
        healthy=party_health([mon()])
        self.assertTrue(may_replace_frontier(old,40,healthy,2))
        self.assertFalse(may_replace_frontier(old,30,healthy,2))
        old['party_min_hp_ratio']=1
        self.assertFalse(may_replace_frontier(old,40,healthy,2))


class FrontierViabilityTests(unittest.TestCase):
    """2026-09-07: hurt-but-playable parties may anchor stage_frontier_<n>."""

    def test_case2_viable_but_not_ready_party_is_frontier_viable(self):
        # two alive at 60% (below the strict 80% bar), both can still act
        # -> frontier_viable, NOT party_ready.
        party = [mon(12, 20), mon(12, 20)]
        h = party_health(party)
        self.assertFalse(h['party_ready'])
        self.assertTrue(h['frontier_viable'])
        self.assertTrue(frontier_viable(party))
        # a fainted third member still leaves 2 alive and >=50% total
        party3 = [mon(16, 20), mon(16, 20), mon(0, 20)]
        self.assertFalse(party_health(party3)['party_ready'])
        self.assertTrue(party_health(party3)['frontier_viable'])

    def test_case3_dead_or_unplayable_party_is_not_frontier_viable(self):
        for party in (
            [mon(0, 20), mon(0, 20), mon(0, 20)],          # all fainted
            [mon(11, 20), mon(0, 20), mon(0, 20)],         # only one alive
            [mon(3, 20), mon(3, 20), mon(3, 20)],          # alive but ~15% HP
            [mon(11, 20, pp=0), mon(11, 20, pp=0)],        # no usable PP
        ):
            h = party_health(party)
            self.assertFalse(h['frontier_viable'], party)
            self.assertFalse(frontier_viable(party), party)

    def test_strict_ready_party_is_always_frontier_viable(self):
        self.assertTrue(party_health([mon(), mon(18)])['frontier_viable'])

    def test_case4_weak_anchor_cannot_replace_healthy_below_min_gain(self):
        old = dict(party_ready=True, party_min_hp_ratio=.9,
                   frontier_score=40.0, frontier_metric_version=2)
        weak = party_health([mon(11, 20), mon(11, 20)])   # viable, not ready
        self.assertFalse(weak['party_ready'])
        gain = FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN
        self.assertFalse(may_replace_frontier(old, 40.0, weak, 2))
        self.assertFalse(may_replace_frontier(old, 40.0 + gain - 0.01, weak, 2))

    def test_case5_weak_but_viable_anchor_advances_at_min_gain(self):
        old = dict(party_ready=True, party_min_hp_ratio=.9,
                   frontier_score=40.0, frontier_metric_version=2)
        weak = party_health([mon(11, 20), mon(11, 20)])
        self.assertTrue(may_replace_frontier(
            old, 40.0 + FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN, weak, 2))

    def test_case6_weaker_state_never_replaces_healthier_at_equal_score(self):
        old = dict(party_ready=True, party_min_hp_ratio=.9,
                   frontier_score=55.0, frontier_metric_version=2)
        weak = party_health([mon(11, 20), mon(11, 20)])
        self.assertFalse(may_replace_frontier(old, 55.0, weak, 2))
        self.assertFalse(may_replace_frontier(old, 50.0, weak, 2))

    def test_case7_legacy_party_ready_true_meta_still_valid(self):
        # Old anchor meta has party_ready True but none of the new keys.
        legacy = dict(party_ready=True, party_min_hp_ratio=.85,
                      frontier_score=30.0, frontier_metric_version=2)
        healthy = party_health([mon()])
        self.assertTrue(may_replace_frontier(legacy, 31.0, healthy, 2))   # deeper
        self.assertFalse(may_replace_frontier(legacy, 20.0, healthy, 2))  # shallower
        # a deliberate hurt anchor still cannot displace it cheaply
        weak = party_health([mon(11, 20), mon(11, 20)])
        self.assertFalse(may_replace_frontier(legacy, 31.0, weak, 2))

    def test_hurt_anchor_advances_only_on_real_progress_between_hurt_states(self):
        old = dict(party_ready=False, frontier_viable=True,
                   frontier_score=60.0, frontier_metric_version=2)
        weak = party_health([mon(11, 20), mon(11, 20)])
        self.assertFalse(may_replace_frontier(old, 61.0, weak, 2))
        self.assertTrue(may_replace_frontier(
            old, 60.0 + FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN, weak, 2))
        # a healthy party reclaims it without throwing away progress
        healthy = party_health([mon(), mon()])
        self.assertTrue(may_replace_frontier(old, 60.0, healthy, 2))
        self.assertFalse(may_replace_frontier(old, 50.0, healthy, 2))

    def test_healthy_candidate_just_below_score_cannot_replace_hurt_anchor(self):
        old = dict(party_ready=False, frontier_viable=True,
                   frontier_score=60.0, frontier_metric_version=2)
        healthy = party_health([mon(), mon()])
        self.assertFalse(may_replace_frontier(old, 60.0 - 0.01, healthy, 2))
        self.assertTrue(may_replace_frontier(old, 60.0, healthy, 2))

    def test_no_replacement_path_ever_lowers_frontier_score(self):
        healthy = party_health([mon(), mon()])
        weak = party_health([mon(11, 20), mon(11, 20)])
        anchors = [
            dict(party_ready=True, party_min_hp_ratio=.9,
                 frontier_score=50.0, frontier_metric_version=2),
            dict(party_ready=False, frontier_viable=True,
                 frontier_score=50.0, frontier_metric_version=2),
            dict(party_ready=False, frontier_score=50.0, frontier_metric_version=2),
            dict(party_ready=False, frontier_score=50.0, frontier_metric_version=0),
        ]
        for old in anchors:
            old_effective = (old['frontier_score']
                             if old.get('frontier_metric_version', 0) >= 2 else 0.0)
            for cand in (healthy, weak):
                for cand_score in (0.0, 10.0, 49.99, 50.0, 60.0):
                    if may_replace_frontier(old, cand_score, cand, 2):
                        self.assertGreaterEqual(
                            cand_score, old_effective,
                            f"replacement lowered score: {old} <- {cand_score}")

class ShapingConsistencyTests(unittest.TestCase):
    def test_leaving_and_returning_does_not_repay_same_approach(self):
        from target_shaper_v20 import TargetShaper
        s=TargetShaper(backtrack_margin=12)
        s.update('pallet',20)
        self.assertGreater(s.update('pallet',10)[0],0)
        s.update('route1',30)
        s.update('pallet',20)
        self.assertEqual(s.update('pallet',10)[0],0)
        self.assertGreater(s.update('pallet',9)[0],0)
        s.reset()
        s.update('pallet',20)
        self.assertGreater(s.update('pallet',10)[0],0)
