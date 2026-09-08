"""The navigation-horizon advancement gate is actually wired into the trainer.

twoby2.horizon was fully coded but nothing triggered it - nav_horizon.json stayed
empty and the episode length was frozen at the migration rung forever. These
tests exercise MilestoneCheckpointCallback._maybe_advance_nav_horizon end to end.
"""
import json
import os
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import train
import twoby2
from train import MilestoneCheckpointCallback
from twoby2.horizon import NAV_EPISODE_HORIZONS, index_at_or_below_horizon
from pokemon_env import PokemonFireRedEnv


def _stub(path, *, runs_at_rung, eval_result="rejected"):
    cb = object.__new__(MilestoneCheckpointCallback)
    cb._nav_horizon_path = path
    cb._horizon_runs_at_rung = runs_at_rung
    cb._nav_horizon_snapshot = {}
    cb.version = 5
    cb.min_full_episodes = 8
    cb.last_eval_result = eval_result
    # champion with a fully reproduced early game (retention intact)
    cb.champion_metrics = {
        "full_intro_permille": 1000, "full_stairs_permille": 1000,
        "full_exit_permille": 1000, "full_starter_permille": 1000,
        "max_stage": 2,
    }
    # 30 completed FULL runs, every one reaching Route 1 (stage 2)
    cb.recent_full = deque(
        [{"stage": 2, "steps": 8000 + i, "maps": 6} for i in range(30)],
        maxlen=256,
    )
    return cb


_METRICS = {
    "full_episodes": 30, "max_stage": 2,
    "full_intro_permille": 1000, "full_stairs_permille": 1000,
    "full_exit_permille": 1000, "full_starter_permille": 1000,
}

_START_INDEX = index_at_or_below_horizon(PokemonFireRedEnv.LONG_FULL_PROBE_STEPS)


class NavHorizonWiringTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "nav_horizon.json")
        # feature gate ON, trainer in 2x2 live mode
        self._prev = twoby2.FEATURES.get("adaptive_nav_horizon", False)
        twoby2.FEATURES["adaptive_nav_horizon"] = True
        self._live = patch.object(train, "_TWOBY2_LIVE", True)
        self._live.start()
        # Keep unit tests independent of the concurrently changing live
        # curriculum file. Gate-specific transition rates are tested with a
        # deterministic healthy Route-1 transition here.
        self._curriculum = patch(
            "curriculum_v20.CurriculumState.load",
            return_value=SimpleNamespace(
                transitions={1: SimpleNamespace(success_rate=1.0)}
            ),
        )
        self._curriculum.start()

    def tearDown(self):
        self._curriculum.stop()
        twoby2.FEATURES["adaptive_nav_horizon"] = self._prev
        self._live.stop()

    def test_advances_exactly_one_rung_when_the_gate_passes(self):
        cb = _stub(self.path, runs_at_rung=90)   # +30 this eval -> 120 >= 100
        cb._maybe_advance_nav_horizon(_METRICS)

        self.assertTrue(os.path.exists(self.path))
        saved = json.load(open(self.path))
        self.assertEqual(saved["current_horizon_index"], _START_INDEX + 1)
        self.assertEqual(saved["current_horizon"],
                         NAV_EPISODE_HORIZONS[_START_INDEX + 1])
        self.assertEqual(cb._horizon_runs_at_rung, 0)   # counter reset on advance
        self.assertTrue(cb._nav_horizon_snapshot["advanced"])

    def test_holds_when_too_few_evaluated_full_runs(self):
        cb = _stub(self.path, runs_at_rung=10)   # +30 -> 40 < 100
        cb._maybe_advance_nav_horizon(_METRICS)

        saved = json.load(open(self.path))
        self.assertEqual(saved["current_horizon_index"], _START_INDEX)
        self.assertEqual(cb._horizon_runs_at_rung, 40)
        self.assertFalse(cb._nav_horizon_snapshot["advanced"])
        self.assertTrue(any("full runs" in r
                            for r in cb._nav_horizon_snapshot["blocked_by"]))

    def test_holds_when_a_hard_candidate_regression_was_seen(self):
        cb = _stub(self.path, runs_at_rung=200, eval_result="regression")
        cb._maybe_advance_nav_horizon(_METRICS)
        saved = json.load(open(self.path))
        self.assertEqual(saved["current_horizon_index"], _START_INDEX)
        self.assertFalse(cb._nav_horizon_snapshot["advanced"])

    def test_noop_when_feature_gate_is_off(self):
        twoby2.FEATURES["adaptive_nav_horizon"] = False
        cb = _stub(self.path, runs_at_rung=500)
        cb._maybe_advance_nav_horizon(_METRICS)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(cb._nav_horizon_snapshot, {})

    def test_noop_when_not_twoby2_live(self):
        self._live.stop()
        with patch.object(train, "_TWOBY2_LIVE", False):
            cb = _stub(self.path, runs_at_rung=500)
            cb._maybe_advance_nav_horizon(_METRICS)
        self._live.start()
        self.assertFalse(os.path.exists(self.path))

    def test_horizon_never_regresses_across_a_reload(self):
        # advance once
        cb = _stub(self.path, runs_at_rung=200)
        cb._maybe_advance_nav_horizon(_METRICS)
        advanced_index = json.load(open(self.path))["current_horizon_index"]
        self.assertEqual(advanced_index, _START_INDEX + 1)

        # a later eval that would NOT pass the gate must not lower the rung
        cb2 = _stub(self.path, runs_at_rung=0)
        cb2._maybe_advance_nav_horizon({**_METRICS, "full_episodes": 5})
        self.assertEqual(json.load(open(self.path))["current_horizon_index"],
                         advanced_index)


class NavHorizonTruncationSemanticsTests(unittest.TestCase):
    """step() ends a FULL / watcher run at the adaptive horizon as a TimeLimit
    *truncation* (PPO bootstraps V(s)), not a terminal - and the frozen
    'long_full_32k' cut at 32768 is gone."""

    def _src(self):
        import inspect
        return inspect.getsource(PokemonFireRedEnv.step)

    def test_frozen_long_full_32k_cut_is_removed(self):
        src = self._src()
        self.assertNotIn("long_full_32k:truncate", src)
        self.assertNotRegex(src, r'route_steps >= self\.LONG_FULL_PROBE_STEPS')

    def test_horizon_truncates_full_and_watcher(self):
        src = self._src()
        self.assertIn('"nav_horizon"', src)
        self.assertRegex(
            src,
            r'training_objective in \("full", "watcher"\)\s*\n\s*'
            r'and self\.route_steps >= episode_limit')

    def test_step_budget_no_longer_terminates_full_runs(self):
        src = self._src()
        # the unconditional "or self.route_steps >= episode_limit" terminal is
        # gone - it is now guarded to the non-full (legacy) objectives only
        self.assertNotRegex(
            src,
            r'terminated = bool\(\s*\n\s*objective_done\s*\n\s*'
            r'or _role_wipe_terminal\s*\n\s*or self\.route_steps >= episode_limit')
        self.assertRegex(
            src,
            r'training_objective not in \("full", "watcher"\)\s*\n\s*'
            r'and self\.route_steps >= episode_limit')


if __name__ == "__main__":
    unittest.main()
