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
        self.original_last_watcher_frame = web_stream.LAST_WATCHER_FRAME
        web_stream.CLUSTER_DIR = root
        web_stream.CLUSTER_POLICY_FILE = root / "policy.json"
        web_stream.CLUSTER_WORKERS_FILE = root / "workers.json"
        web_stream.WATCHER_STATUS_FILE = root / "watcher.json"
        web_stream.WATCHER_STREAM_FILE = root / "watcher.jpg"
        web_stream.LAST_WATCHER_FRAME = None

    def tearDown(self):
        web_stream.CLUSTER_DIR = self.original_dir
        web_stream.CLUSTER_POLICY_FILE = self.original_policy
        web_stream.CLUSTER_WORKERS_FILE = self.original_workers
        web_stream.WATCHER_STATUS_FILE = self.original_watcher_status
        web_stream.WATCHER_STREAM_FILE = self.original_watcher_stream
        web_stream.LAST_WATCHER_FRAME = self.original_last_watcher_frame
        self.directory.cleanup()

    def test_watcher_list_exposes_the_live_best_brain_without_private_paths(self):
        web_stream.WATCHER_STATUS_FILE.write_text(
            json.dumps(
                {
                    "id": "dynamic-watcher",
                    "policy_version": 42,
                    "action": "RIGHT",
                    "reward_events": ["battle_hp_stagnant:-0.50"],
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
        self.assertEqual(
            payload["watchers"][0]["reward_events"], ["battle_hp_stagnant:-0.50"]
        )
        self.assertNotIn("model_path", payload["watchers"][0])

    def test_dashboard_starts_with_a_watcher_list_and_first_live_preview(self):
        page = web_stream.index()

        self.assertIn('id="watcher-list"', page)
        for element_id in ("watcher-frame", "watcher-state", "watcher-events"):
            self.assertIn(f'id="{element_id}"', page)
        self.assertIn("/api/watchers", page)
        self.assertIn("watchers[0]", page)
        self.assertIn("w.reward_events", page)
        self.assertIn("Trainings-Reward", page)

    def test_dashboard_event_list_uses_a_closed_milestone_spread_expression(self):
        page = web_stream.index()

        self.assertIn(
            "...((w.milestones||[]).slice(-4).reverse().map(x=>['Milestone',x,0]))]",
            page,
        )

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

    def test_watcher_frame_reuses_the_last_complete_frame_during_a_replace_gap(self):
        frame = b"\xff\xd8previous-complete-frame\xff\xd9"
        web_stream.WATCHER_STREAM_FILE.write_bytes(frame)
        web_stream.get_watcher_frame()
        web_stream.WATCHER_STREAM_FILE.unlink()

        response = web_stream.get_watcher_frame()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, frame)

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
                        "training_objective": "stairs",
                        "training_role": "scout",
                        "story_stage": "F2_TO_STAIRS",
                        "last_reward_events": ["new_tile:+0.20"],
                        "episode_reward": 12.5,
                        "reward_trace": [
                            {"step": 127, "action": 6, "reward": 0.2, "events": ["new_tile:+0.20"]}
                        ],
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
        self.assertEqual(payload["workers"][0]["training_objective"], "stairs")
        self.assertEqual(payload["workers"][0]["training_role"], "scout")
        self.assertEqual(payload["workers"][0]["last_reward_events"], ["new_tile:+0.20"])
        self.assertEqual(payload["workers"][0]["reward_trace"][0]["events"], ["new_tile:+0.20"])
        self.assertNotIn("signature", payload["workers"][0])

    def test_cluster_status_exposes_current_curriculum_goals_and_active_objectives(self):
        now = time.time()
        web_stream.CLUSTER_WORKERS_FILE.write_text(
            json.dumps(
                {
                    "worker-a": {
                        "worker_id": "worker-a",
                        "last_seen": now,
                        "milestones": ["intro_complete"],
                        "training_objective": "intro",
                    },
                    "worker-b": {
                        "worker_id": "worker-b",
                        "last_seen": now,
                        "training_objective": "scout",
                    },
                }
            )
        )

        payload = web_stream.get_cluster_status()

        self.assertEqual(
            payload["goals"][0],
            {
                "key": "intro_complete",
                "label": "Intro abschließen",
                "category": "Story",
                "observed": True,
                "active_trainers": 1,
            },
        )
        self.assertEqual(payload["learning_objectives"], [{"key": "intro", "trainers": 1}, {"key": "scout", "trainers": 1}])

    def test_cluster_status_includes_brock_and_frontier_goal_catalog(self):
        now = time.time()
        web_stream.CLUSTER_WORKERS_FILE.write_text(
            json.dumps(
                {
                    "worker-a": {
                        "worker_id": "worker-a",
                        "last_seen": now,
                        "milestones": ["maps_3", "level_7", "progress_1"],
                        "last_reward_events": ["brock_battle_start:+500"],
                        "training_objective": "battle",
                    }
                }
            )
        )

        goals = {goal["key"]: goal for goal in web_stream.get_cluster_status()["goals"]}

        self.assertTrue(goals["maps_3"]["observed"])
        self.assertTrue(goals["level_7"]["observed"])
        self.assertTrue(goals["progress_1"]["observed"])
        self.assertTrue(goals["brock_rush"]["observed"])
        self.assertEqual(goals["brock_rush"]["active_trainers"], 1)

    def test_dashboard_has_the_five_operational_pages(self):
        page = web_stream.index()

        for label in ("Watcher", "Trainer", "Overworld", "Live-Statistik", "Lernziele"):
            self.assertIn(label, page)
        self.assertIn('data-page="trainers"', page)
        self.assertIn('data-page="overworld"', page)
        self.assertIn('data-page="goals"', page)

    def test_dashboard_goals_render_live_goal_and_objective_payloads(self):
        page = web_stream.index()

        self.assertIn("state.cluster.goals", page)
        self.assertIn("state.cluster.learning_objectives", page)
        self.assertIn("active_trainers", page)

    def test_trainer_page_has_selectable_details_and_reward_trace(self):
        page = web_stream.index()

        self.assertIn('id="trainer-detail"', page)
        self.assertIn('id="trainer-reward-trace"', page)
        self.assertIn("selectedTrainer", page)
        self.assertIn("reward_trace", page)
        self.assertIn("training_objective", page)

    def test_trainer_page_labels_current_fps(self):
        page = web_stream.index()

        self.assertIn("FPS", page)
        self.assertIn("w.fps", page)

    def test_trainer_page_distinguishes_last_step_from_episode_reward(self):
        page = web_stream.index()

        self.assertIn("Letzter Step", page)
        self.assertIn("Episoden-Reward", page)
        self.assertIn("Ø Episoden-Reward", page)
