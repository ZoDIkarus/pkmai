import json
import os
import tempfile
import unittest

import twoby2.protected_assets as pa
from twoby2 import reset_tools as rt


class MasterProtectionTests(unittest.TestCase):
    def test_registry_has_master_seed_and_live_anchor(self):
        reg = pa.default_registry()
        ids = {a.logical_id for a in reg.assets}
        self.assertEqual(ids, {"canonical_post_parcel_master",
                               "route1_manual_battle_seed",
                               "protected_live_route1_frontier_anchor",
                               "protected_ram_probe_route1_seed"})
        master = next(a for a in reg.assets if a.asset_type == pa.TYPE_MASTER)
        self.assertTrue(master.protected_from_delete
                        and master.protected_from_replace
                        and master.protected_from_normalize)

    def test_master_exists_and_sha256_matches(self):
        rep = pa.verify_master()
        self.assertTrue(rep["exists"])
        self.assertTrue(rep["sha256_ok"], rep)
        self.assertFalse(rep["is_symlink"])

    def test_route1_seed_is_the_confirmed_manual_backup_not_ambiguous(self):
        reg = pa.default_registry()
        seed = next(a for a in reg.assets if a.logical_id == "route1_manual_battle_seed")
        self.assertFalse(seed.candidate_ambiguity)          # user-confirmed
        self.assertEqual(seed.asset_type, pa.TYPE_BATTLE_SEED)
        self.assertTrue(any("healthy_frontier_20260906_214032" in f for f in seed.files))
        self.assertEqual(pa.ROUTE1_MANUAL_SEED["state_sha256"],
                         "4244fa04db32fcd313ab159930787a613367904d5bd0f9c6ad8a0654ff6b0b48")
        self.assertEqual(pa.ROUTE1_MANUAL_SEED["meta_sha256"],
                         "37087431bf762a4cdaf2c885b800d194629f06b5bcd0e782d99c520d6fd57682")

    def test_live_frontier_anchor_is_a_separate_identity_and_not_a_manual_seed(self):
        reg = pa.default_registry()
        anchor = next(a for a in reg.assets
                      if a.logical_id == "protected_live_route1_frontier_anchor")
        self.assertEqual(anchor.asset_type, pa.TYPE_LIVE_ANCHOR)
        self.assertNotEqual(anchor.asset_type, pa.TYPE_BATTLE_SEED)
        self.assertTrue(any("runtime/curriculum_shared/stage_frontier_2" in f
                            for f in anchor.files))
        # still fully protected against destructive ops
        for op in ("delete", "normalize", "overwrite"):
            with self.assertRaises(pa.ProtectedPathError):
                pa.assert_not_protected(anchor.files[0], op=op, registry=reg)

    def test_frozen_ram_probe_seed_is_separate_and_protected(self):
        reg = pa.default_registry()
        seed = next(a for a in reg.assets
                    if a.logical_id == "protected_ram_probe_route1_seed")
        self.assertEqual(seed.asset_type, pa.TYPE_PROBE_SEED)
        self.assertTrue(any("brain_backups/ram_probe_route1_seed" in f
                            for f in seed.files))
        self.assertEqual(
            pa.sha256_file(seed.abs_files()[0]),
            pa.RAM_PROBE_ROUTE1_SEED["state_sha256"])
        for op in ("delete", "normalize", "overwrite", "publish"):
            with self.assertRaises(pa.ProtectedPathError):
                pa.assert_not_protected(seed.files[0], op=op, registry=reg)

    def test_every_mutating_op_on_the_master_is_refused(self):
        for op in pa.MUTATING_OPS:
            with self.assertRaises(pa.ProtectedPathError):
                pa.assert_not_protected(pa.MASTER_REL, op=op)

    def test_alias_dotdot_and_symlink_cannot_bypass(self):
        # .. alias
        with self.assertRaises(pa.ProtectedPathError):
            pa.assert_not_protected(
                "local/custom_integrations/PokemonFireRed-Gba/../PokemonFireRed-Gba/StartGame.state",
                op="delete")
        # absolute path
        with self.assertRaises(pa.ProtectedPathError):
            pa.assert_not_protected(
                os.path.join(pa.PROJECT_ROOT, pa.MASTER_REL), op="overwrite")
        # symlink into the file
        with tempfile.TemporaryDirectory() as d:
            link = os.path.join(d, "sneaky_link.state")
            os.symlink(os.path.join(pa.PROJECT_ROOT, pa.MASTER_REL), link)
            with self.assertRaises(pa.ProtectedPathError):
                pa.assert_not_protected(link, op="normalize")

    def test_unprotected_path_is_allowed(self):
        self.assertTrue(pa.assert_not_protected(
            "runtime/battle/checkpoints/battle_learner.zip", op="delete"))

    def test_dry_run_registry_writes_nothing(self):
        reg = pa.default_registry()
        out = reg.dump_dry_run("runtime/protected_savestates.json")
        self.assertIn("would_write", out)
        self.assertFalse(os.path.exists("runtime/protected_savestates.json"))
        # serialisable
        json.dumps(out)


class ResetsPreserveProtectedTests(unittest.TestCase):
    """Temp copies of the protected files; every reset/wipe/migration path must
    leave them byte-identical."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # fake project layout with the two protected files
        self.master = os.path.join(self.tmp, "local/custom_integrations/PokemonFireRed-Gba/StartGame.state")
        self.seed_state = os.path.join(self.tmp, "runtime/curriculum_shared/stage_frontier_2.state.gz")
        self.seed_meta = os.path.join(self.tmp, "runtime/curriculum_shared/stage_frontier_2.meta.json")
        for p, content in ((self.master, b"MASTER"), (self.seed_state, b"SEEDSTATE"),
                           (self.seed_meta, b'{"seed":true}')):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(content)
        # a live anchor pair too
        self.anchor_state = os.path.join(self.tmp, "runtime/curriculum_shared/live_anchor.state.gz")
        self.anchor_meta = os.path.join(self.tmp, "runtime/curriculum_shared/live_anchor.meta.json")
        for p, content in ((self.anchor_state, b"ANCHOR"), (self.anchor_meta, b"{}")):
            open(p, "wb").write(content)
        # a registry rooted at the temp project
        self.reg = pa.ProtectedRegistry(root=self.tmp)
        self.reg.add(pa.ProtectedAsset("canonical_post_parcel_master",
            asset_type=pa.TYPE_MASTER,
            files=["local/custom_integrations/PokemonFireRed-Gba/StartGame.state"]))
        self.reg.add(pa.ProtectedAsset("route1_manual_battle_seed",
            asset_type=pa.TYPE_BATTLE_SEED,
            files=["runtime/curriculum_shared/stage_frontier_2.state.gz",
                   "runtime/curriculum_shared/stage_frontier_2.meta.json"]))
        self.reg.add(pa.ProtectedAsset("protected_live_route1_frontier_anchor",
            asset_type=pa.TYPE_LIVE_ANCHOR,
            files=["runtime/curriculum_shared/live_anchor.state.gz",
                   "runtime/curriculum_shared/live_anchor.meta.json"]))
        self._hashes = {p: open(p, "rb").read() for p in
                        (self.master, self.seed_state, self.seed_meta,
                         self.anchor_state, self.anchor_meta)}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_all_intact(self):
        for p, want in self._hashes.items():
            self.assertEqual(open(p, "rb").read(), want, f"{p} changed!")

    def _nav_dirs(self):
        for sub in ("nav/ck", "nav", "bat/ck", "bat"):
            os.makedirs(os.path.join(self.tmp, sub), exist_ok=True)
        open(os.path.join(self.tmp, "nav/ck/navigation_champion.zip"), "w").write("c")
        return dict(nav_ckpt=os.path.join(self.tmp, "nav/ck"),
                    nav_stats=os.path.join(self.tmp, "nav"),
                    bat_ckpt=os.path.join(self.tmp, "bat/ck"),
                    bat_stats=os.path.join(self.tmp, "bat"))

    def test_navigation_reset_preserves_both(self):
        p = self._nav_dirs()
        plan = rt.plan_navigation_reset(nav_ckpt_dir=p["nav_ckpt"], nav_stats_dir=p["nav_stats"])
        rt.apply_plan(plan, dry_run=False, registry=self.reg)
        self._assert_all_intact()

    def test_battle_reset_and_scenario_wipe_preserve_both(self):
        p = self._nav_dirs()
        open(os.path.join(p["bat_stats"], "scenario_pool.json"), "w").write("{}")
        plan = rt.plan_battle_reset(battle_ckpt_dir=p["bat_ckpt"], battle_stats_dir=p["bat_stats"],
                                    wipe_scenarios=True)
        rt.apply_plan(plan, dry_run=False, registry=self.reg)
        self._assert_all_intact()

    def test_full_reset_preserves_both(self):
        p = self._nav_dirs()
        plan = rt.plan_full_reset(nav_ckpt_dir=p["nav_ckpt"], nav_stats_dir=p["nav_stats"],
                                  battle_ckpt_dir=p["bat_ckpt"], battle_stats_dir=p["bat_stats"],
                                  confirm_full=True)
        rt.apply_plan(plan, dry_run=False, registry=self.reg)
        self._assert_all_intact()

    def test_a_reset_plan_that_targets_the_seed_is_refused(self):
        # a hostile / buggy plan that lists the protected seed as a delete target
        bad_plan = rt.ResetPlan(scope="battle", actions=[
            ("delete", self.seed_state)], touch=[self.seed_state])
        with self.assertRaises(pa.ProtectedPathError):
            rt.apply_plan(bad_plan, dry_run=False, registry=self.reg)
        self._assert_all_intact()

    def test_migration_atomic_copy_refuses_protected_target(self):
        # simulate migrate_to_2x2._atomic_copy guard
        with self.assertRaises(pa.ProtectedPathError):
            pa.assert_not_protected(self.master, op="replace", registry=self.reg)

    def test_live_anchor_also_preserved_by_full_reset(self):
        p = self._nav_dirs()
        plan = rt.plan_full_reset(nav_ckpt_dir=p["nav_ckpt"], nav_stats_dir=p["nav_stats"],
                                  battle_ckpt_dir=p["bat_ckpt"], battle_stats_dir=p["bat_stats"],
                                  confirm_full=True)
        rt.apply_plan(plan, dry_run=False, registry=self.reg)
        self._assert_all_intact()


class ReadOnlyCopyTests(unittest.TestCase):
    """A protected file may be READ and byte-copied to an unprotected dest;
    it may never be a write/publish target."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.seed = os.path.join(self.tmp, "seed/stage_frontier_2.state.gz")
        os.makedirs(os.path.dirname(self.seed), exist_ok=True)
        open(self.seed, "wb").write(b"HEALTHY-SEED-BYTES" * 100)
        self.reg = pa.ProtectedRegistry(root=self.tmp)
        self.reg.add(pa.ProtectedAsset("route1_manual_battle_seed",
            asset_type=pa.TYPE_BATTLE_SEED, files=["seed/stage_frontier_2.state.gz"]))
        self._orig = open(self.seed, "rb").read()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_only_copy_of_protected_seed_is_allowed_and_verified(self):
        dst = os.path.join(self.tmp, "work/working.state.gz")
        rep = pa.read_only_copy(self.seed, dst, registry=self.reg)
        self.assertEqual(open(dst, "rb").read(), self._orig)      # byte identical
        self.assertEqual(open(self.seed, "rb").read(), self._orig)  # source untouched
        self.assertEqual(rep["bytes"], len(self._orig))
        self.assertEqual(rep["sha256"], pa.sha256_file(self.seed))

    def test_copy_destination_may_not_be_a_protected_path(self):
        with self.assertRaises(pa.ProtectedPathError):
            pa.read_only_copy(self.seed, self.seed, registry=self.reg)

    def test_direct_write_to_protected_seed_is_still_blocked(self):
        for op in ("normalize", "overwrite", "replace", "rename", "move",
                   "delete", "publish"):
            with self.assertRaises(pa.ProtectedPathError):
                pa.assert_write_target_ok(self.seed, op=op, registry=self.reg)

    def test_assert_not_protected_allows_a_read_op(self):
        self.assertTrue(pa.assert_not_protected(self.seed, op="read", registry=self.reg))
        self.assertTrue(pa.assert_not_protected(self.seed, op="copy_source", registry=self.reg))

    def test_sha_mismatch_during_copy_is_a_hard_error(self):
        # a copy whose source "changes" mid-flight (simulated via a bad dest sha)
        dst = os.path.join(self.tmp, "work/x.gz")
        real = pa.sha256_file
        try:
            calls = {"n": 0}
            def flaky(path):
                calls["n"] += 1
                return "deadbeef" if calls["n"] == 3 else real(path)
            pa.sha256_file = flaky
            with self.assertRaises(pa.Sha256Mismatch):
                pa.read_only_copy(self.seed, dst, registry=self.reg)
        finally:
            pa.sha256_file = real


if __name__ == "__main__":
    unittest.main()
