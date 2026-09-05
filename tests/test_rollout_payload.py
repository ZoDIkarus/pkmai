import tempfile
import unittest
from pathlib import Path

import numpy as np

import cluster_master
from rollout_protocol import encode_rollout


class RolloutPayloadTests(unittest.TestCase):
    def test_spools_valid_encoded_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            original = cluster_master.ROLLOUT_INBOX
            cluster_master.ROLLOUT_INBOX = Path(directory)
            try:
                batch = {
                    "images": np.zeros((1, 1, 64, 64), dtype=np.uint8),
                    "nav": np.zeros((1, 28), dtype=np.float32),
                    "actions": np.array([1], dtype=np.int64),
                    "rewards": np.array([1.0], dtype=np.float32),
                    "dones": np.array([True]),
                    "log_probs": np.array([-0.2], dtype=np.float32),
                    "values": np.array([0.1], dtype=np.float32),
                }
                path, count = cluster_master.store_rollout_payload(
                    "worker-a", encode_rollout(batch)
                )
                self.assertTrue(path.exists())
                self.assertEqual(count, 1)
            finally:
                cluster_master.ROLLOUT_INBOX = original
