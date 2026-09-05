import tempfile
import unittest
from pathlib import Path

import numpy as np

import cluster_master


class RolloutUploadTests(unittest.TestCase):
    def test_authenticated_upload_spools_a_valid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            original = cluster_master.ROLLOUT_INBOX
            cluster_master.ROLLOUT_INBOX = Path(directory)
            try:
                payload = cluster_master.store_rollout_upload(
                    "worker-a",
                    {
                        "images": np.zeros((1, 1, 64, 64), dtype=np.uint8),
                        "nav": np.zeros((1, 28), dtype=np.float32),
                        "actions": np.array([1], dtype=np.int64),
                        "rewards": np.array([1.0], dtype=np.float32),
                        "dones": np.array([True]),
                        "log_probs": np.array([-0.2], dtype=np.float32),
                        "values": np.array([0.1], dtype=np.float32),
                    },
                )
                self.assertEqual(payload.name.endswith("worker-a.npz"), True)
                self.assertTrue(payload.exists())
            finally:
                cluster_master.ROLLOUT_INBOX = original
