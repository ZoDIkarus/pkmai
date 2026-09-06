import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import web_stream


class ClusterStatusApiTests(unittest.TestCase):
    def test_web_bind_settings_default_to_external_port_8001(self):
        original_host = os.environ.pop("PKMAI_WEB_HOST", None)
        original_port = os.environ.pop("PKMAI_WEB_PORT", None)
        try:
            self.assertEqual(web_stream.web_bind_settings(), ("0.0.0.0", 8001))
        finally:
            if original_host is not None:
                os.environ["PKMAI_WEB_HOST"] = original_host
            if original_port is not None:
                os.environ["PKMAI_WEB_PORT"] = original_port

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.original_dir = web_stream.CLUSTER_DIR
        self.original_policy = web_stream.CLUSTER_POLICY_FILE
        self.original_workers = web_stream.CLUSTER_WORKERS_FILE
        self.original_watcher_status = web_stream.WATCHER_STATUS_FILE
        self.original_watcher_stream = web_stream.WATCHER_STREAM_FILE
        web_stream.CLUSTER_DIR = root
        web_stream.CLUSTER_POLICY_FILE = root / "policy.json"
        web_stream.CLUSTER_WORKERS_FILE = root / "workers.json"
        web_stream.WATCHER_STATUS_FILE = root / "watcher.json"
        web_stream.WATCHER_STREAM_FILE = root / "watcher.jpg"

    def tearDown(self):
        web_stream.CLUSTER_DIR = self.original_dir
        web_stream.CLUSTER_POLICY_FILE = self.original_policy
        web_stream.CLUSTER_WORKERS_FILE = self.original_workers
        web_stream.WATCHER_STATUS_FILE = self.original_watcher_status
        web_stream.WATCHER_STREAM_FILE = self.original_watcher_stream
        self.directory.cleanup()

    def test_watcher_list_exposes_the_live_best_brain_without_private_paths(self):
        web_stream.WATCHER_STATUS_FILE.write_text(
            json.dumps(
                {
                    "id": "dynamic-watcher",
                    "policy_version": 42,
                    "action": "RIGHT",
                    "model_path": "/private/runtime/cluster/dynamic_policy_best.pt",
                }
            )
        )

        payload = web_stream.get_watchers()

        self.assertEqual(payload["watchers"][0]["id"], "dynamic-watcher")
        self.assertEqual(payload["watchers"][0]["policy_version"], 42)
        self.assertEqual(payload["watchers"][0]["action"], "RIGHT")
        self.assertTrue(payload["watchers"][0]["online"])
        self.assertEqual(payload["watchers"][0]["stream_url"], "/watcher.jpg")
        self.assertNotIn("model_path", payload["watchers"][0])

    def test_dashboard_starts_with_a_watcher_list_and_first_live_preview(self):
        page = web_stream.index()

        self.assertIn('id="watcher-list"', page)
        for element_id in ("watcher-frame", "watcher-state", "watcher-events"):
            self.assertIn(f'id="{element_id}"', page)
        self.assertIn("/api/watchers", page)
        self.assertIn("watchers[0]", page)

    def test_legacy_watcher_url_returns_to_the_dashboard(self):
        response = web_stream.watcher_dashboard_redirect()

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/")

    def test_watcher_frame_reads_a_complete_jpeg_before_responding(self):
        frame = b"\xff\xd8complete-frame\xff\xd9"
        web_stream.WATCHER_STREAM_FILE.write_bytes(frame)

        response = web_stream.get_watcher_frame()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, frame)
        self.assertEqual(response.headers["content-length"], str(len(frame)))

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
                        "position": {"valid": True, "map_bank": 3, "map_id": 7, "x": 14, "y": 22},
                        "last_action": 6,
                        "last_reward": 1.25,
                        "episode_steps": 128,
                        "in_battle": True,
                        "milestones": ["intro_complete", "stairs_down"],
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
        self.assertEqual(payload["workers"][0]["position"]["map_id"], 7)
        self.assertEqual(payload["workers"][0]["milestones"], ["intro_complete", "stairs_down"])
        self.assertNotIn("signature", payload["workers"][0])

    def test_dashboard_has_the_five_operational_pages(self):
        page = web_stream.index()

        for label in ("Watcher", "Trainer", "Overworld", "Live-Statistik", "Lernziele"):
            self.assertIn(label, page)
        self.assertIn('data-page="trainers"', page)
        self.assertIn('data-page="overworld"', page)
        self.assertIn('data-page="goals"', page)
