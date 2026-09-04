import json
import os
import tempfile
import unittest
from unittest.mock import patch

import watch


class WatcherModelRoutingTests(unittest.TestCase):
    def test_watcher_prefers_protected_vault_over_raw_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            resume = os.path.join(tmp, "resume.zip")
            vault = os.path.join(tmp, "progress.zip")
            open(resume, "wb").close()
            with (
                patch.object(watch, "RESUME_MODEL", resume),
                patch.dict(watch.SKILL_MODELS, {"progress": vault}),
            ):
                self.assertEqual(watch.get_watcher_model_path("progress"), resume)
                open(vault, "wb").close()
                self.assertEqual(watch.get_watcher_model_path("progress"), vault)

    def test_watcher_falls_back_to_stage_vault_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            resume = os.path.join(tmp, "missing-resume.zip")
            vault = os.path.join(tmp, "stairs.zip")
            open(vault, "wb").close()
            with (
                patch.object(watch, "RESUME_MODEL", resume),
                patch.dict(watch.SKILL_MODELS, {"stairs": vault}),
            ):
                self.assertEqual(watch.get_watcher_model_path("stairs"), vault)

    def test_champion_sidecar_prevents_false_version_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            version_file = os.path.join(tmp, "missing-version.json")
            champion_file = os.path.join(tmp, "champion.json")
            with open(champion_file, "w") as f:
                json.dump({"version": 7}, f)
            with (
                patch.object(watch, "VERSION_FILE", version_file),
                patch.object(watch, "CHAMPION_FILE", champion_file),
            ):
                self.assertEqual(watch.get_latest_version(), 7)


if __name__ == "__main__":
    unittest.main()
