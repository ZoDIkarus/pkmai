import unittest
from checkpoint_health import party_health, may_replace_frontier


def mon(hp=20, maximum=20, status=0, pp=10):
    return dict(checksum_ok=True, cur_hp=hp, max_hp=maximum, status=status, moves=[{'pp':pp}])


class CheckpointHealthTests(unittest.TestCase):
    def test_two_fainted_and_two_hp_last_survivor_is_not_ready(self):
        self.assertFalse(party_health([mon(0,31),mon(0,16),mon(2,15)])['party_ready'])

    def test_all_members_need_health_and_usable_pp(self):
        for party in ([], [mon(3)], [mon(status=8)], [mon(pp=0)], [mon(),mon(0)]):
            self.assertFalse(party_health(party)['party_ready'])
        self.assertTrue(party_health([mon(),mon(18)])['party_ready'])

    def test_healthy_replaces_poisoned_anchor_without_new_score_record(self):
        old=dict(party_ready=False,frontier_score=100,frontier_metric_version=2)
        self.assertTrue(may_replace_frontier(old,30,party_health([mon()]),2))

    def test_no_progress_record_can_publish_dying_party(self):
        self.assertFalse(may_replace_frontier({},1000,party_health([mon(2)]),2))

    def test_health_refresh_at_same_depth_and_no_regression(self):
        old=dict(party_ready=True,party_min_hp_ratio=.8,frontier_score=40,frontier_metric_version=2)
        healthy=party_health([mon()])
        self.assertTrue(may_replace_frontier(old,40,healthy,2))
        self.assertFalse(may_replace_frontier(old,30,healthy,2))
        old['party_min_hp_ratio']=1
        self.assertFalse(may_replace_frontier(old,40,healthy,2))

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
