import tempfile
import unittest
from pathlib import Path

import numpy as np

from rollout_protocol import consume_rollouts, write_rollout


class RolloutProtocolTests(unittest.TestCase):
    def test_round_trip_consumes_an_uploaded_batch_once(self):
        with tempfile.TemporaryDirectory() as directory:
            inbox = Path(directory)
            batch = {
                "images": np.zeros((2, 1, 64, 64), dtype=np.uint8),
                "nav": np.zeros((2, 28), dtype=np.float32),
                "actions": np.array([1, 2], dtype=np.int64),
                "rewards": np.array([0.5, 1.0], dtype=np.float32),
                "dones": np.array([False, True]),
                "log_probs": np.array([-0.2, -0.1], dtype=np.float32),
                "values": np.array([0.1, 0.2], dtype=np.float32),
            }

            write_rollout(inbox, "worker-a", batch)
            received = list(consume_rollouts(inbox, limit=1))

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], "worker-a")
            np.testing.assert_array_equal(received[0][1]["actions"], batch["actions"])
            self.assertEqual(list(consume_rollouts(inbox, limit=1)), [])
