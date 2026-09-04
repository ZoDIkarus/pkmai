import json
import tempfile
import time
import unittest
from pathlib import Path

import web_stream


class ClusterStatusApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.original_dir = web_stream.CLUSTER_DIR
        self.original_policy = web_stream.CLUSTER_POLICY_FILE
        self.original_workers = web_stream.CLUSTER_WORKERS_FILE
        web_stream.CLUSTER_DIR = root
        web_stream.CLUSTER_POLICY_FILE = root / "policy.json"
        web_stream.CLUSTER_WORKERS_FILE = root / "workers.json"

    def tearDown(self):
        web_stream.CLUSTER_DIR = self.original_dir
        web_stream.CLUSTER_POLICY_FILE = self.original_policy
        web_stream.CLUSTER_WORKERS_FILE = self.original_workers
        self.directory.cleanup()

    def test_cluster_status_omits_secret_paths_and_marks_live_worker(self):
        now = time.time()
        web_stream.CLUSTER_POLICY_FILE.write_text(
            json.dumps({"version": 12, "checkpoint": "/private/path/policy-v12"})
        )
        web_stream.CLUSTER_WORKERS_FILE.write_text(
            json.dumps(
                {
                    "worker-a": {
                        "worker_id": "worker-a",
                        "hostname": "worker-host",
                        "active_agents": 1,
                        "fps": 24.5,
                        "policy_version": 12,
                        "last_seen": now,
                        "signature": "must-not-leak",
                    }
                }
            )
        )

        payload = web_stream.get_cluster_status()

        self.assertEqual(payload["policy_version"], 12)
        self.assertEqual(payload["checkpoint"], "policy-v12")
        self.assertTrue(payload["brain_online"])
        self.assertEqual(payload["workers"][0]["worker_id"], "worker-a")
        self.assertTrue(payload["workers"][0]["online"])
        self.assertNotIn("signature", payload["workers"][0])
