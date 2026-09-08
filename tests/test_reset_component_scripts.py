import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
SPEC = importlib.util.spec_from_file_location(
    "reset_component", os.path.join(ROOT, "tools", "reset_component.py")
)
reset_component = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reset_component)


class ComponentResetTests(unittest.TestCase):
    def test_apply_backs_up_then_deletes_only_explicit_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = os.path.join(tmp, "runtime")
            backups = os.path.join(tmp, "backups")
            os.makedirs(os.path.join(runtime, "navigation", "checkpoints"))
            target = os.path.join(
                runtime, "navigation", "checkpoints", "navigation_champion.zip"
            )
            preserved = os.path.join(runtime, "curriculum_shared", "stage_1.state.gz")
            os.makedirs(os.path.dirname(preserved))
            with open(target, "wb") as handle:
                handle.write(b"brain")
            with open(preserved, "wb") as handle:
                handle.write(b"savestate")

            with patch.multiple(
                reset_component,
                ROOT=tmp,
                RUNTIME=runtime,
                BACKUP_ROOT=backups,
            ):
                destination, manifest = reset_component.create_backup(
                    "nav-brain", [target]
                )
                deleted = reset_component.delete_targets([target])

            self.assertFalse(os.path.exists(target))
            self.assertTrue(os.path.exists(preserved))
            self.assertTrue(os.path.isfile(os.path.join(
                destination, "runtime_before_reset.tar.gz"
            )))
            self.assertEqual(len(manifest["archive_sha256"]), 64)
            self.assertEqual(deleted, [
                "runtime/navigation/checkpoints/navigation_champion.zip"
            ])

    def test_full_backup_archives_preserved_runtime_data_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = os.path.join(tmp, "runtime")
            backups = os.path.join(tmp, "backups")
            seed = os.path.join(runtime, "battle", "scenarios", "seed.state.gz")
            os.makedirs(os.path.dirname(seed))
            with open(seed, "wb") as handle:
                handle.write(b"seed")
            with patch.multiple(
                reset_component,
                ROOT=tmp,
                RUNTIME=runtime,
                BACKUP_ROOT=backups,
            ):
                destination, manifest = reset_component.create_backup("full", [])
            paths = {row["path"] for row in manifest["files"]}
            self.assertIn("runtime/battle/scenarios/seed.state.gz", paths)
            with open(os.path.join(destination, "BACKUP_MANIFEST.json")) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["scope"], "full")

    def test_full_script_requires_exact_written_confirmation(self):
        with open(os.path.join(ROOT, "scripts", "reset_all.sh")) as handle:
            script = handle.read()
        self.assertIn("KOMPLETT RESET", script)
        self.assertIn("reset_run_scope full", script)

    def test_only_full_reset_includes_shiny_telemetry(self):
        self.assertNotIn(reset_component._p("shiny"),
                         reset_component.SCOPES["nav-brain"])
        self.assertNotIn(reset_component._p("shiny"),
                         reset_component.SCOPES["battle-brain"])
        self.assertIn(reset_component._p("shiny"),
                      reset_component.SCOPES["full"])

    def test_readme_contains_all_copy_paste_commands(self):
        with open(os.path.join(ROOT, "scripts", "README.md")) as handle:
            readme = handle.read()
        for name in (
            "start_all.sh", "stop_all.sh", "reset_nav_brain.sh",
            "reset_map_global.sh", "reset_battle_brain.sh", "reset_all.sh",
        ):
            self.assertIn(f"bash scripts/{name}", readme)


if __name__ == "__main__":
    unittest.main()
