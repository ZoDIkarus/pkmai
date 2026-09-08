import unittest

from twoby2.nav_chunk_rollout import NavChunkedRollout, NAV_PPO_N_STEPS
from twoby2 import horizon as hz


class ChunkedRolloutTests(unittest.TestCase):
    def test_ppo_n_steps_is_512_and_independent_of_horizon(self):       # F6
        self.assertEqual(NAV_PPO_N_STEPS, 512)
        self.assertEqual(hz.NAV_EPISODE_HORIZONS[-1], 163840)

    def test_update_every_512_decisions_even_in_a_163840_step_episode(self):  # F6
        r = NavChunkedRollout()
        horizon = 163840
        updates = 0
        for _ in range(horizon):
            full = r.on_navigation_decision()
            if full:
                out = r.close_chunk(episode_terminated=False, episode_truncated=False,
                                    last_value=1.5)
                self.assertTrue(out["is_chunk_boundary_only"])
                self.assertFalse(out["reset_env"])
                self.assertEqual(out["bootstrap_value"], 1.5)
                updates += 1
        self.assertEqual(updates, horizon // 512)
        self.assertEqual(r.updates, updates)

    def test_chunk_boundary_is_not_terminal_or_truncation(self):        # F7
        r = NavChunkedRollout()
        for _ in range(512):
            r.on_navigation_decision()
        out = r.close_chunk(episode_terminated=False, episode_truncated=False,
                            last_value=2.0)
        self.assertFalse(out["is_real_terminal"])
        self.assertFalse(out["is_time_limit_truncation"])
        self.assertFalse(out["reset_env"])
        self.assertNotEqual(r.episode_route_steps, 0)   # episode keeps running

    def test_terminated_zeroes_bootstrap(self):                          # F8
        r = NavChunkedRollout()
        for _ in range(200):
            r.on_navigation_decision()
        out = r.close_chunk(episode_terminated=True, episode_truncated=False,
                            last_value=9.9, terminal_value=9.9)
        self.assertEqual(out["bootstrap_value"], 0.0)
        self.assertTrue(out["reset_env"])
        self.assertEqual(r.episode_route_steps, 0)

    def test_real_timelimit_truncation_bootstraps_from_terminal_obs(self):  # F9
        r = NavChunkedRollout()
        for _ in range(300):
            r.on_navigation_decision()
        out = r.close_chunk(episode_terminated=False, episode_truncated=True,
                            last_value=3.0, terminal_value=4.25)
        self.assertTrue(out["is_time_limit_truncation"])
        self.assertEqual(out["bootstrap_value"], 4.25)
        self.assertFalse(out["reset_env"])   # env reset handled by the horizon path
        self.assertEqual(r.episode_route_steps, 0)

    def test_horizon_truncation_detection(self):
        r = NavChunkedRollout()
        for _ in range(2000):
            r.on_navigation_decision()
        self.assertTrue(r.episode_truncated_by_horizon(2000))
        self.assertFalse(r.episode_truncated_by_horizon(4000))

    def test_battle_subepisode_adds_nothing_to_nav_counters(self):      # F10, F11
        r = NavChunkedRollout()
        for _ in range(100):
            r.on_navigation_decision()
        before = (r.decisions_in_chunk, r.total_nav_decisions, r.episode_route_steps)
        out = r.on_battle_subepisode(battle_decisions=14, battle_button_presses=47)
        self.assertEqual(out["nav_decisions_added"], 0)
        self.assertFalse(out["chunk_advanced"])
        self.assertEqual((r.decisions_in_chunk, r.total_nav_decisions,
                          r.episode_route_steps), before)
        self.assertEqual(out["battle_decisions"], 14)

    def test_watcher_step_never_increments_a_learner_counter(self):     # F12
        r = NavChunkedRollout()
        for _ in range(50):
            r.on_navigation_decision()
        snap = (r.total_nav_decisions, r.updates, r.chunks_emitted)
        for _ in range(1000):
            out = r.on_watcher_step()
            self.assertEqual(out["nav_decisions_added"], 0)
            self.assertEqual(out["updates_added"], 0)
        self.assertEqual((r.total_nav_decisions, r.updates, r.chunks_emitted), snap)


if __name__ == "__main__":
    unittest.main()
