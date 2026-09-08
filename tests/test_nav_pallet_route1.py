"""Integration-level checks for the directed-nav Pallet -> Route 1 fix.

Covers plan items D/E/F/G + the mandatory wipe/recovery and next-hop
non-farmability invariants. The RAM-driven behaviour is verified by a real-env
smoke; the reward maths is verified against ``NavShapingState`` /
``RegionLoopGuard`` directly (the env just wires those in).
"""
import inspect
import os
import tempfile
import unittest

import pokemon_env
from pokemon_env import PokemonFireRedEnv as Env
from nav_shaping_state import NavAgentState, objective_key
from loop_guard import RegionLoopGuard
import nav_graph
from nav_graph import DirectedNavGraph, UP, DOWN, LEFT, RIGHT

KP = objective_key(1, 1, "pallet_exit_edge", [(12, 0)])       # Pallet exit
KR = objective_key(1, 2, "known_transition_s2", [(12, 39)])   # Route 1 -> Viridian


def _agent(tmp, *, run_id=1):
    return NavAgentState.load(os.path.join(tmp, "shaping", "a.json"), run_id=run_id)


class SingleSourceOfTruthTests(unittest.TestCase):
    def test_obs_and_reward_both_call_nav_objective(self):
        vec_src = inspect.getsource(Env._build_nav_vector)
        step_src = inspect.getsource(Env.step)
        self.assertIn("self._nav_objective(", vec_src)
        self.assertIn("obj = self._nav_obj", step_src)
        # the old undirected drip is gone
        self.assertNotIn('f"route_approach:{_appr', step_src)
        self.assertNotIn("self.target_shaper.update", step_src)

    def test_schema_and_dim(self):
        self.assertEqual(Env.NAV_OBS_SCHEMA, "nav_obs_v3_directed")
        self.assertIn("self.NAV_DIM = 36", inspect.getsource(Env.__init__))

    def test_progress_deadlines_are_wired(self):
        s = inspect.getsource(Env.step)
        for tok in ("pallet_exit_timeout", "route1_no_progress_timeout",
                    "route1_ledge_loop", "recovery_no_progress_timeout",
                    "_nav_clock", "_recovery_clock", "_nav_shaping_frozen"):
            self.assertIn(tok, s)
        # the clock must exclude battles / menus
        self.assertIn("not in_battle and gameplay_ready", s)

    def test_next_hop_reward_is_checked_at_the_start_tile(self):
        s = inspect.getsource(Env.step)
        # item 7: compare the executed action to the PREVIOUS objective's hop
        self.assertIn("_prev_nh", s)
        self.assertIn("prev.get(\"next_hop_action\")", s)
        self.assertNotIn("obj.get(\"next_hop_action\")\n"
                         "                                        if prev", s)

    def test_legacy_route_cannot_arm_reward_or_timeout(self):
        # obj.valid is confirmed-route-only; the timeout + reward blocks gate on
        # obj.get("valid"), never obj.get("display_reachable")
        s = inspect.getsource(Env.step)
        self.assertIn('obj is not None and obj.get("valid")', s)
        self.assertIn('_obj is not None and _obj.get("valid")', s)
        self.assertNotIn('display_reachable")\n                and self.left_house', s)


class CatchDedupTests(unittest.TestCase):
    def test_catch_reward_is_small_flat_and_persistently_deduped(self):
        # review item 1: no more +54/+56/+58 re-farm over resets.
        self.assertLessEqual(Env.SPECIES_CAUGHT_FIRST_REWARD, 1.0)
        self.assertEqual(Env.SPECIES_CAUGHT_LEVEL_BONUS, 0.0)   # no grind bonus
        self.assertLess(Env.SPECIES_CAUGHT_DUPLICATE_PENALTY, 0.0)
        s = inspect.getsource(Env.step)
        self.assertIn("_load_nav_agent().mark_caught", s)
        self.assertIn("species_caught_first", s)
        # the catch reward must stay well below any geographic checkpoint
        self.assertLess(Env.SPECIES_CAUGHT_FIRST_REWARD, Env.PROVEN_EXIT_REWARD)

    def test_nav_agent_catch_dedup_survives_reset_and_wipe(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            self.assertTrue(s.mark_caught(19))
            s.on_wipe(KR)
            s.record_distance(KR, 4)     # "new episode / stage"
            s.save()
            self.assertFalse(_agent(td, run_id=1).mark_caught(19))   # still deduped
            self.assertTrue(_agent(td, run_id=2).mark_caught(19))    # fresh run


class WipeNoFarmInvariantTests(unittest.TestCase):
    """Model the env's shaping gate: positive nav shaping is 0 during recovery
    until strictly past the pre-wipe directed-distance highwater; the wipe
    penalty is never repaid by the walk back."""

    WIPE_PENALTY = -3.0
    PROGRESS = Env.DIRECTED_PROGRESS_REWARD

    def _walk(self, s, key, distances, *, recovering):
        """Total positive shaping paid over a directed-distance sequence,
        modelling the env's per-key gate."""
        paid = 0.0
        for d in distances:
            frozen = recovering and s.shaping_frozen(key, d)
            if recovering and not s.shaping_frozen(key, d):
                recovering = False
                s.clear_pre_wipe(key)
            if s.record_distance(key, d) and not frozen:
                paid += self.PROGRESS
        return paid

    def test_wipe_then_walk_back_to_pre_wipe_best_is_net_negative(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            gained = self._walk(s, KR, [14, 13, 12, 11, 10], recovering=False)
            s.on_wipe(KR)
            back = self._walk(s, KR, [40, 30, 20, 12, 10], recovering=True)
            self.assertEqual(back, 0.0)
            self.assertLess(self.WIPE_PENALTY + gained + back, 0.0)

    def test_multiple_wipes_and_walkbacks_pay_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            self._walk(s, KR, [20, 15, 10], recovering=False)
            for _ in range(3):
                s.on_wipe(KR)
                self.assertEqual(
                    self._walk(s, KR, [50, 30, 15, 10], recovering=True), 0.0)

    def test_progress_past_the_pre_wipe_highwater_pays_once(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            self._walk(s, KR, [20, 10], recovering=False)
            s.on_wipe(KR)
            paid = self._walk(s, KR, [40, 12, 10, 9, 8, 7], recovering=True)
            self.assertAlmostEqual(paid, 3 * self.PROGRESS)   # 9, 8, 7
            self.assertEqual(self._walk(s, KR, [8, 9, 8], recovering=False), 0.0)

    def test_a_wipe_on_one_objective_does_not_freeze_another(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            s.record_distance(KP, 5)
            s.on_wipe(KP)
            self.assertTrue(s.shaping_frozen(KP, 6))
            self.assertFalse(s.shaping_frozen(KR, 99))
            self.assertTrue(self._walk(s, KR, [10, 9], recovering=False) > 0)


class NextHopNonFarmableTests(unittest.TestCase):
    def test_next_hop_edge_is_run_wide_never_repaid(self):
        with tempfile.TemporaryDirectory() as td:
            s = _agent(td)
            s.mark_next_hop_paid(3, 0, (12, 5), UP)
            self.assertTrue(s.next_hop_already_paid(3, 0, (12, 5), UP))
            s.on_wipe(KP)                              # wipe: still paid
            s.record_distance(KR, 3)                   # stage change: still paid
            self.assertTrue(s.next_hop_already_paid(3, 0, (12, 5), UP))
            s.save()
            self.assertTrue(_agent(td, run_id=1)
                            .next_hop_already_paid(3, 0, (12, 5), UP))
            # only a fresh training run clears it
            self.assertFalse(_agent(td, run_id=2)
                             .next_hop_already_paid(3, 0, (12, 5), UP))


class RealEnvSmokeTests(unittest.TestCase):
    def test_env_runs_end_to_end_with_the_new_obs(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        # never write the directed graph / shaping state into the live runtime
        for attr in ("MOVEMENT_GRAPH_FILE", "NAV_GLOBAL_FILE", "NAV_AGENT_DIR"):
            orig = getattr(pokemon_env, attr)
            setattr(pokemon_env, attr,
                    os.path.join(self._td.name, os.path.basename(orig)))
            self.addCleanup(setattr, pokemon_env, attr, orig)
        try:
            e = Env(rank=0, n_envs=1, shared_edges={}, shared_maps={},
                    shared_transitions={}, shared_progress={}, shared_lock=None,
                    shared_species={}, shared_tiles={})
        except Exception as exc:                       # no ROM in this env
            self.skipTest(f"cannot build a real env: {exc}")
        try:
            import numpy as np
            e.training_objective = "full"
            obs, _ = e.reset()
            self.assertEqual(obs["nav"].shape, (36,))
            rng = np.random.default_rng(0)
            events = []
            for _ in range(80):
                obs, r, term, trunc, info = e.step(int(rng.integers(0, 7)))
                self.assertEqual(obs["nav"].shape, (36,))
                self.assertTrue(np.isfinite(obs["nav"]).all())
                events += info.get("reward_events", [])
                if term or trunc:
                    obs, _ = e.reset()
            names = {x.split(":")[1] if x[:1].isdigit() and x.count(":") >= 2
                     else x.split(":")[0] for x in events}
            self.assertNotIn("route_approach", names)     # old drip gone
            self.assertGreater(e._nav_graph.edge_count(), 0)
            tel = e._nav_telemetry(3, 0, 5, 5)
            for k in ("target", "graph_distance", "next_hop_action",
                      "blocked_directions", "route1_reach_rate"):
                self.assertIn(k, tel)
        finally:
            e.close()


if __name__ == "__main__":
    unittest.main()
