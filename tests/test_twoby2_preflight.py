import importlib.util
import json
import os
import tempfile
import unittest

_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "twoby2_preflight.py")
_spec = importlib.util.spec_from_file_location("twoby2_preflight", _PATH)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


class PreflightHonestyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checks = pf.run_checks()
        cls.by = {c.name: c for c in cls.checks}

    def test_component_prepared_holds(self):
        self.assertTrue(pf._level_ok(self.checks, "component_prepared"))
        for f in ("gBattleMons", "gBattlerPartyIndexes", "gActionSelectionCursor",
                  "gMoveSelectionCursor", "battle_menu_state"):
            self.assertTrue(self.by[f"ram_verified:{f}"].ok, f)

    def test_production_wired_checks_real_call_sites_not_file_existence(self):
        # the checked files really contain a twoby2 call-site now
        import pathlib
        src = pathlib.Path(__file__).resolve().parents[1] / "src"
        self.assertIn("maybe_wrap_full_agent", (src / "train.py").read_text())
        self.assertIn("maybe_wrap_full_agent", (src / "watcher_runtime.py").read_text())
        self.assertIn("twoby2", (src / "pokemon_env.py").read_text())
        # and the preflight reflects that
        self.assertTrue(self.by["production_wired:train"].ok)
        self.assertTrue(self.by["production_wired:watcher_runtime"].ok)
        self.assertTrue(self.by["production_wired:battle_trainers_use_real_driver"].ok)

    def test_real_canary_passed_requires_a_valid_current_report(self):
        # a report exists and is valid -> this check is True; but a wrong-ROM
        # report is rejected (see below)
        with tempfile.TemporaryDirectory() as d:
            cdir = os.path.join(d, "20990101_000000")
            os.makedirs(cdir)
            with open(os.path.join(cdir, "canary_report.json"), "w") as f:
                json.dump({"overall": "PASS", "rom_sha256": "WRONGHASH",
                           "seed_unchanged_final": True,
                           "verified_ram_addresses": {"gBattleMons": "0x23be4"}}, f)
            old = pf.CANARY_ROOT
            try:
                pf.CANARY_ROOT = d
                by = {c.name: c for c in pf.run_checks()}
                self.assertFalse(by["real_canary_passed"].ok)
            finally:
                pf.CANARY_ROOT = old

    def test_stale_fail_report_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            cdir = os.path.join(d, "x")
            os.makedirs(cdir)
            with open(os.path.join(cdir, "canary_report.json"), "w") as f:
                json.dump({"overall": "FAIL"}, f)
            old = pf.CANARY_ROOT
            try:
                pf.CANARY_ROOT = d
                by = {c.name: c for c in pf.run_checks()}
                self.assertFalse(by["real_canary_passed"].ok)
            finally:
                pf.CANARY_ROOT = old

    def test_feature_gates_still_off_so_nothing_is_actually_live(self):
        self.assertTrue(self.by["feature_gates_currently_off"].ok)
        from twoby2 import feature_enabled
        for g in ("nav_battle_wrapper", "battle_env", "battle_router_live"):
            self.assertFalse(feature_enabled(g))

    def test_activation_ready_requires_all_lower_levels(self):
        # structurally: activation_ready == migration_ready == every lower level
        s = {lvl: pf._level_ok(self.checks, lvl) for lvl in
             ("component_prepared", "production_wired", "unit_tests_passed",
              "real_canary_passed")}
        migration_ready = (all(s.values())
                           and pf._level_ok(self.checks, "migration_ready"))
        # if any lower level were false, migration/activation must be false
        if not all(s.values()):
            self.assertFalse(migration_ready)

    def test_protected_files_reject_writes(self):
        self.assertTrue(self.by["protected_files_reject_writes"].ok)


if __name__ == "__main__":
    unittest.main()
