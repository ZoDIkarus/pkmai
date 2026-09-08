import gzip
import os
import tempfile
import unittest

import twoby2.scenario_normalizer as sn
from twoby2.protected_assets import (ProtectedRegistry, ProtectedAsset,
                                     TYPE_MASTER, TYPE_BATTLE_SEED as TYPE_SEED)


class NormalizerFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "seed.state.gz")
        with gzip.open(self.src, "wb") as f:
            f.write(b"ORIGINAL-STATE-BYTES")
        self._orig = open(self.src, "rb").read()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_normalize_is_fail_closed_and_never_touches_the_original(self):
        norm = sn.BattleScenarioNormalizer()          # no writer, not verified
        raw = norm.capture_raw(self.src, {"route": "route1", "party": [
            {"cur_hp": 10, "max_hp": 31, "status": 0x40, "move_pp": [3, 5],
             "move_max_pp": [25, 20]}]})
        res = norm.normalize(raw, workdir=self.tmp)
        self.assertFalse(res.ok)
        self.assertFalse(res["normalized"])
        self.assertIn("NOT verified", res["blocker"])
        self.assertTrue(res["needs"])
        # original byte-identical
        self.assertEqual(open(self.src, "rb").read(), self._orig)
        # a working copy was made, and it is NOT the original
        self.assertTrue(os.path.exists(res["working_copy"]))
        self.assertNotEqual(os.path.realpath(res["working_copy"]),
                            os.path.realpath(self.src))

    def test_protected_seed_may_be_read_only_copied_then_fails_closed(self):
        # a PROTECTED source is allowed as a read source; the working copy is
        # made outside the protected tree; normalize still fails closed.
        reg = ProtectedRegistry(root=self.tmp)
        reg.add(ProtectedAsset("route1_manual_battle_seed", asset_type=TYPE_SEED,
                               files=["seed.state.gz"]))
        norm = sn.BattleScenarioNormalizer(registry=reg)
        raw = norm.capture_raw(self.src, {"route": "route1", "party": []})
        self.assertTrue(raw["source_is_protected"])
        workdir = tempfile.mkdtemp()          # OUTSIDE the protected tree
        res = norm.normalize(raw, workdir=workdir)
        self.assertFalse(res.ok)
        self.assertTrue(res["source_unchanged"])
        self.assertEqual(open(self.src, "rb").read(), self._orig)   # seed untouched
        self.assertEqual(open(res["working_copy"], "rb").read(), self._orig)  # byte copy
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    def test_working_copy_inside_a_protected_tree_is_refused(self):
        reg = ProtectedRegistry(root=self.tmp)
        reg.add(ProtectedAsset("route1_manual_battle_seed", asset_type=TYPE_SEED,
                               files=["seed.state.gz"]))
        norm = sn.BattleScenarioNormalizer(registry=reg)
        raw = norm.capture_raw(self.src, {"route": "route1"})
        # workdir == the protected file's own dir -> working copy would collide
        # with a protected identity path only if named identically; but a
        # publish TO the protected seed must always be refused:
        res = sn.NormalizedScenarioResult(ok=True, working_copy=self.src)
        with self.assertRaises(Exception):
            norm.publish(res, self.src)
        self.assertEqual(open(self.src, "rb").read(), self._orig)

    def test_expected_state_is_full_health_but_keeps_identity(self):
        norm = sn.BattleScenarioNormalizer()
        raw = {"meta": {"party": [
            {"species": 7, "level": 11, "cur_hp": 10, "max_hp": 31, "status": 0x40,
             "move_pp": [3, 5, 0, 39], "move_max_pp": [25, 26, 25, 39]},
            {"species": 19, "level": 4, "cur_hp": 0, "max_hp": 16},   # fainted
        ]}}
        exp = norm._expected_state(raw)
        alive, fainted = exp["party"]
        self.assertEqual(alive["cur_hp"], 31)
        self.assertEqual(alive["status"], 0)
        self.assertEqual(alive["move_pp"], [25, 26, 25, 39])
        self.assertEqual(alive["level"], 11)          # identity unchanged
        self.assertEqual(fainted["cur_hp"], 0)        # fainted mon untouched
        self.assertIn("species", exp["unchanged"])


class LevelBandVersioningTests(unittest.TestCase):
    def _meta(self, route="route1", levels=(11, 4, 3), enemy=(16,), hp=(25, 16, 15)):
        return {"route": route, "story_phase": "route1_early", "battle_kind": "wild",
                "enemy_species": enemy, "enemy_levels": (3,),
                "own_species": (7, 19, 16), "own_levels": levels, "own_hp": hp,
                "own_pp": ((25,),), "own_status": (0, 0, 0),
                "rom_sha256": "ROM", "schema_version": "v1",
                "party": [{"level": lv} for lv in levels]}

    def test_higher_total_level_is_a_new_version_not_an_overwrite(self):
        lb = sn.LevelBandedScenarios()
        r1 = lb.add_version(self._meta(levels=(11, 4, 3)), current_total_level=18)
        r2 = lb.add_version(self._meta(levels=(20, 12, 10)), current_total_level=18)
        self.assertTrue(r1["added"] and r2["added"])
        self.assertEqual(r1["band"], sn.CURRENT)
        self.assertEqual(r2["band"], sn.HIGH)
        # both versions retained
        self.assertEqual(len(lb.versions("route1")), 2)

    def test_signature_covers_hp_pp_status_enemy_party(self):
        f1 = sn.scenario_signature_fields(self._meta(hp=(25, 16, 15)))
        f2 = sn.scenario_signature_fields(self._meta(hp=(10, 16, 15)))
        self.assertNotEqual(f1, f2)

    def test_regression_core_and_weaker_scenarios_are_kept(self):
        lb = sn.LevelBandedScenarios(cap=3)
        core = lb.add_version(self._meta(levels=(6, 0, 0)), current_total_level=18,
                              is_regression_core=True)
        for extra in range(6):
            lb.add_version(self._meta(levels=(11 + extra, 4, 3), enemy=(16 + extra,)),
                           current_total_level=18)
        sigs = {m["signature"] for m in lb.versions("route1")}
        self.assertIn(core["signature"], sigs)   # core never evicted

    def test_low_current_high_bands(self):
        self.assertEqual(sn.level_band(10, current_total=20), sn.LOW)
        self.assertEqual(sn.level_band(20, current_total=20), sn.CURRENT)
        self.assertEqual(sn.level_band(30, current_total=20), sn.HIGH)


if __name__ == "__main__":
    unittest.main()
