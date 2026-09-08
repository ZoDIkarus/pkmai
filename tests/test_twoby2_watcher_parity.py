import unittest

import twoby2.watcher_parity as wp


class _SharedEnv:
    """A full-worker-shaped env built from the SAME components for both roles.
    Deterministic given (state copy, seed, actions)."""

    def __init__(self, role, *, state):
        self.role = role
        self._state0 = dict(state)
        self._s = None
        self._seed = 0
        self._t = 0

    def reset(self, *, seed=0):
        self._seed = seed
        self._s = dict(self._state0)
        self._t = 0

    def step(self, a):
        self._t += 1
        # a pure, deterministic transition shared by both roles
        self._s["x"] = self._s["x"] + (1 if a == "RIGHT" else 0)
        self._s["hp"] = max(0, self._s["hp"] - (5 if a == "FIGHT" else 0))
        battle = a == "FIGHT"
        if battle and self._s["hp"] <= 0:
            self._s["flags"] = sorted(set(self._s["flags"]) | {"blacked_out"})
        obs = [self._s["x"], self._s["hp"], self._t]
        reward_components = {"new_coord": 0.3 if a == "RIGHT" else 0.0,
                             "wipe": -3.0 if "blacked_out" in self._s["flags"] else 0.0}
        term = "blacked_out" in self._s["flags"]
        trunc = self._t >= 10
        info = {
            "position": (self._s.get("map", 3), self._s["x"], self._s["y"]),
            "story_flags": self._s["flags"],
            "hp_pp": {"hp": self._s["hp"]},
            "battle_detected": battle,
            "world_state": {"map": self._s.get("map", 3), "x": self._s["x"]},
        }
        return obs, reward_components, term, trunc, info


STATE = {"x": 0, "y": 5, "hp": 12, "flags": ["delivered_parcel"], "map": 3}
ACTIONS = ["RIGHT", "RIGHT", "FIGHT", "FIGHT", "FIGHT", "RIGHT"]


class WatcherParityTests(unittest.TestCase):
    def test_shared_component_list_and_differences(self):
        self.assertIn("reset_baseline", wp.SHARED_COMPONENTS)
        self.assertIn("navigation_battle_wrapper", wp.SHARED_COMPONENTS)
        self.assertIn("battle_router", wp.SHARED_COMPONENTS)
        self.assertEqual(set(wp.WATCHER_DIFFERENCES),
                         {"no_rollouts", "no_optimizer", "no_learning_counters",
                          "rewards_are_display_only"})

    def test_watcher_and_full_are_bit_identical_except_learning(self):
        full = _SharedEnv(wp.FullRole(), state=STATE)
        watcher = _SharedEnv(wp.WatcherRole(), state=STATE)
        ft = wp.step_trace(full, ACTIONS)
        wt = wp.step_trace(watcher, ACTIONS)
        wp.assert_parity(ft, wt)          # raises on any difference

    def test_parity_check_catches_a_real_divergence(self):
        full = _SharedEnv(wp.FullRole(), state=STATE)
        watcher = _SharedEnv(wp.WatcherRole(), state={**STATE, "hp": 999})
        with self.assertRaises(AssertionError):
            wp.assert_parity(wp.step_trace(full, ACTIONS),
                             wp.step_trace(watcher, ACTIONS))

    def test_watcher_role_flags(self):
        w = wp.WatcherRole()
        self.assertTrue(w.is_watcher and w.rewards_are_display_only)
        self.assertFalse(w.collects_rollouts or w.has_optimizer
                         or w.updates_learning_counters)


if __name__ == "__main__":
    unittest.main()
