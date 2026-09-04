import tempfile
import unittest
from pathlib import Path

from cluster_brain import persist_checkpoint


class FakeCheckpoint:
    def to_directory(self, path):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "state.txt").write_text("ready", encoding="utf-8")


class ClusterCheckpointTests(unittest.TestCase):
    def test_checkpoint_is_published_as_a_complete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint"
            result = persist_checkpoint(FakeCheckpoint(), target)

            self.assertEqual(result, str(target))
            self.assertEqual((target / "state.txt").read_text(encoding="utf-8"), "ready")
            self.assertFalse((Path(directory) / ".checkpoint.tmp").exists())
