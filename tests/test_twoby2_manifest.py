import json
import os
import tempfile
import unittest
from unittest import mock

import twoby2.manifest as mf


def full_manifest(**over):
    kw = dict(
        rom_sha256="ROM",
        navigation_champion={"version": 10, "steps": 5_000_000, "sha256": "navc"},
        navigation_learner={"version": 11, "steps": 5_100_000, "sha256": "navl"},
        navigation_candidate={"version": 11, "steps": 5_100_000, "sha256": "navd"},
        battle_learner={"version": 4, "steps": 200_000, "sha256": "batl"},
        battle_candidate={"version": 4, "steps": 200_000, "sha256": "batd"},
        battle_champion={"version": 3, "steps": 100_000, "sha256": "batc"},
        battle_champion_source="ppo", generation=7, combo_version=2)
    kw.update(over)
    return mf.build_manifest(**kw)


class ManifestTests(unittest.TestCase):
    def test_full_manifest_validates(self):
        self.assertEqual(mf.validate_manifest(full_manifest()), [])

    def test_missing_pieces_are_flagged(self):
        self.assertTrue(mf.validate_manifest({"schema": mf.SCHEMA}))
        self.assertTrue(mf.validate_manifest("nope"))
        self.assertTrue(mf.validate_manifest({}))

    def test_evals_comparable_empty_is_false(self):
        self.assertFalse(mf.evals_comparable({}, {}))
        self.assertFalse(mf.evals_comparable(full_manifest(), {}))

    def test_evals_comparable_requires_same_battle_version_sha_source_obs_rom(self):
        a = full_manifest()
        b = full_manifest()
        self.assertTrue(mf.evals_comparable(a, b))
        # different battle sha
        self.assertFalse(mf.evals_comparable(
            a, full_manifest(battle_champion={"version": 3, "steps": 1, "sha256": "OTHER"})))
        # different nav obs schema
        self.assertFalse(mf.evals_comparable(a, full_manifest(nav_obs_schema="nav_obs_v2")))
        # different ROM
        self.assertFalse(mf.evals_comparable(a, full_manifest(rom_sha256="OTHERROM")))

    def test_needs_rebaseline_on_version_or_hash_change(self):
        a = full_manifest()
        self.assertFalse(mf.needs_rebaseline(a, 3, "batc"))
        self.assertTrue(mf.needs_rebaseline(a, 4, "batc"))
        self.assertTrue(mf.needs_rebaseline(a, 3, "different"))

    def test_pin_for_navigation_generation(self):
        p = mf.pin_for_navigation_generation(full_manifest())
        self.assertEqual(p["battle_champion_version"], 3)
        self.assertEqual(p["battle_champion_sha256"], "batc")
        self.assertEqual(p["generation"], 7)

    def test_atomic_write_rejects_invalid_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "model_manifest.json")
            with self.assertRaises(ValueError):
                mf.atomic_write(path, {"schema": "bad"})
            self.assertFalse(os.path.exists(path))
            mf.atomic_write(path, full_manifest())
            with open(path) as fh:
                self.assertEqual(json.load(fh)["schema"], mf.SCHEMA)
            self.assertEqual(
                [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp.json")],
                [])

    def test_crash_between_tempwrite_and_replace_leaves_old_file_intact(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model_manifest.json")
            mf.atomic_write(path, full_manifest(generation=1))
            orig = open(path).read()
            with mock.patch("twoby2.manifest.os.replace",
                            side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    mf.atomic_write(path, full_manifest(generation=2))
            # old file untouched, no stray temp
            self.assertEqual(open(path).read(), orig)
            self.assertEqual(
                [f for f in os.listdir(d) if f.endswith(".tmp.json")], [])

    def test_module_has_no_hardcoded_runtime_path(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(mf))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                self.assertNotIn("runtime/", n.value)
        sig = inspect.signature(mf.atomic_write)
        self.assertEqual(sig.parameters["path"].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
