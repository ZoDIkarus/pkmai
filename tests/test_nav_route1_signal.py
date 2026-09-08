"""Navigation-progress blocker fixes (2026-09-08).

Root cause: on Route 1 ``target_valid=false`` (no confirmed directed route to
the warp-trigger exit ``(10,0)``) so the PPO got NO directional signal north,
while the per-episode ``stage_advance 1->2`` +250 magnet held agents at the
entry. Fixes:
  * geometric potential-based approach shaping toward a VERIFIED exit while the
    directed path is incomplete (symmetric, telescoping, clamped);
  * ``stage_advance`` deduplicated RUN-WIDE (not per episode);
  * directed-graph pollution cleanup (ledges + false blocks) + permanent
    static blocks + no move/block recording in battle / menu / warp frames.
"""
import os
import tempfile
import unittest

import pokemon_env
from pokemon_env import PokemonFireRedEnv as Env
from nav_shaping_state import NavAgentState
import nav_graph
from nav_graph import DirectedNavGraph, UP, DOWN, LEFT, RIGHT


MAP = (3, 19)
T = (10, 0)                       # Route-1 north exit (a warp-trigger tile)


def _obj(*, manhattan, key="k", mode="geometric", verified=True, valid=False):
    return {"target_manhattan": manhattan, "key": key, "approach_mode": mode,
            "target_verified": verified, "valid": valid}


class GeometricApproachTests(unittest.TestCase):
    def test_closer_is_positive_farther_is_the_mirror_negative(self):
        pos = Env._geometric_approach_component(
            _obj(manhattan=10), _obj(manhattan=9), moved_tiles=1)
        neg = Env._geometric_approach_component(
            _obj(manhattan=9), _obj(manhattan=10), moved_tiles=1)
        self.assertGreater(pos, 0)
        self.assertAlmostEqual(pos, -neg, places=6)
        self.assertAlmostEqual(pos, Env.GEO_APPROACH_REWARD, places=6)

    def test_a_to_b_to_a_nets_zero_before_step_cost(self):
        # 10 -> 8 -> 10  (two closer, two farther)
        legs = [(10, 8), (8, 10)]
        total = sum(Env._geometric_approach_component(
            _obj(manhattan=a), _obj(manhattan=b), moved_tiles=abs(a - b))
            for a, b in legs)
        self.assertAlmostEqual(total, 0.0, places=6)
        # with the ordinary step cost every round trip is strictly negative
        self.assertLess(total + 2 * Env.NAVIGATION_STEP_COST
                        if hasattr(Env, "NAVIGATION_STEP_COST") else total - 0.01,
                        0.0)

    def test_standstill_and_objective_change_pay_zero(self):
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=5), _obj(manhattan=5), moved_tiles=0), 0.0)
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=5, key="a"), _obj(manhattan=3, key="b"),
            moved_tiles=2), 0.0)

    def test_only_geometric_mode_and_verified_target_pay(self):
        # a confirmed (valid) route -> the directed component owns it, geo pays 0
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=5, mode="directed"),
            _obj(manhattan=3, mode="directed"), moved_tiles=2), 0.0)
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=5, verified=False), _obj(manhattan=3, verified=False),
            moved_tiles=2), 0.0)

    def test_implausible_jump_is_clamped_out(self):
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=40), _obj(manhattan=2), moved_tiles=1), 0.0)
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=10), _obj(manhattan=9),
            moved_tiles=Env.GEO_APPROACH_MAX_DELTA + 1), 0.0)

    def test_recovery_freeze_pays_zero(self):
        self.assertEqual(Env._geometric_approach_component(
            _obj(manhattan=10), _obj(manhattan=8), moved_tiles=2,
            recovery_frozen=True), 0.0)

    def test_repeated_ledge_hop_south_never_nets_positive(self):
        # jump south (+manhattan) then climb back (-manhattan): 0 before cost
        legs = [(6, 9), (9, 6)]
        total = sum(Env._geometric_approach_component(
            _obj(manhattan=a), _obj(manhattan=b), moved_tiles=abs(a - b))
            for a, b in legs)
        self.assertLessEqual(total, 0.0)


class StageAdvanceDedupTests(unittest.TestCase):
    def _st(self, td, run_id=1):
        return NavAgentState.load(os.path.join(td, "a.json"), run_id=run_id)

    def test_stage_advance_pays_once_per_run_not_per_episode(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._st(td)
            self.assertFalse(s.stage_advance_already_paid(1, 2))
            s.mark_stage_advance_paid(1, 2)
            self.assertTrue(s.stage_advance_already_paid(1, 2))
            s.save()
            # a new episode reloads the SAME run -> still paid
            self.assertTrue(self._st(td, run_id=1).stage_advance_already_paid(1, 2))

    def test_wipe_and_savestate_reload_do_not_renew_the_claim(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._st(td)
            s.mark_stage_advance_paid(1, 2)
            s.on_wipe("kR")                       # a wipe
            s.record_distance("kR", 5)            # "new episode"
            s.save()
            reloaded = self._st(td, run_id=1)     # savestate reload
            self.assertTrue(reloaded.stage_advance_already_paid(1, 2))

    def test_a_genuine_fresh_run_clears_it(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._st(td, run_id=1)
            s.mark_stage_advance_paid(1, 2)
            s.save()
            self.assertFalse(self._st(td, run_id=2).stage_advance_already_paid(1, 2))

    def test_env_source_gates_stage_reward_on_run_wide_dedup(self):
        import inspect
        src = inspect.getsource(Env.step)
        self.assertIn("stage_advance_already_paid", src)
        self.assertIn("mark_stage_advance_paid", src)


class GraphSanitizeTests(unittest.TestCase):
    def test_all_ledges_removed_static_blocks_reduced(self):
        g = DirectedNavGraph()
        # a straight corridor walked both ways
        for y in range(5, 0, -1):
            g.observe_move(MAP, (10, y), UP, (10, y - 1), step=1)
            g.observe_move(MAP, (10, y - 1), DOWN, (10, y), step=1)
        # a fake ledge (multi-tile) - not recorded any more, but simulate an
        # old stored one + a false block on a corridor tile
        e = nav_graph.MovementEdge((10, 8), nav_graph.JUMP, confidence=1)
        g._edges.setdefault(MAP, {})[((10, 4), DOWN)] = e
        for v in range(4):
            g.observe_block(MAP, (10, 3), UP, step=v, visit_id=v)  # false wall
        rep = g.sanitize()
        self.assertEqual(rep["ledge_edges_removed"], 1)
        # (10,3)-UP contradicted by the real reverse walk (10,2)-DOWN->(10,3)
        self.assertGreaterEqual(rep["blocks_removed_reverse_walk"]
                                + rep["blocks_removed_walk_contradiction"], 1)
        self.assertFalse(g.is_blocked(MAP, (10, 3), UP))

    def test_bidirectional_walk_lets_bfs_route_a_one_way_walked_corridor(self):
        g = DirectedNavGraph()
        # only ever walked NORTHBOUND (12,5)->...->(10,0)
        path = [(12, 5), (12, 4), (12, 3), (11, 3), (10, 3), (10, 2), (10, 1), (10, 0)]
        for a, b in zip(path, path[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            act = next(k for k, v in nav_graph.DELTA.items() if v == (dx, dy))
            g.observe_move(MAP, a, act, b, step=1)
        # southbound has no stored edges - but BFS from the south still routes
        r = g.directed_bfs(MAP, (12, 5), [T], now_step=5000)
        self.assertIsNotNone(r)
        self.assertTrue(r["confirmed"])

    def test_static_block_needs_multiple_distinct_confirmations(self):
        g = DirectedNavGraph()
        for _ in range(10):              # same encounter, many frames
            g.observe_block(MAP, (4, 4), RIGHT, step=1, visit_id=7)
        self.assertFalse(g.is_blocked(MAP, (4, 4), RIGHT))
        for v in (100, 200, 300):       # 3 distinct (rank-prefixed) encounters
            g.observe_block(MAP, (4, 4), RIGHT, step=v, visit_id=v)
        self.assertTrue(g.is_blocked(MAP, (4, 4), RIGHT))


class MovementRecordingContextTests(unittest.TestCase):
    def test_battle_menu_and_warp_frames_are_excluded_in_source(self):
        import inspect
        src = inspect.getsource(Env.step)
        # the guard that gates observe_move / observe_block
        self.assertIn("_clean_nav_ctx", src)
        self.assertIn("not in_battle and not battle_just_ended", src)
        self.assertIn("_warp_settle", src)
        self.assertIn("effective_action == requested_action", src)

    def test_rank_prefixed_visit_id_in_source(self):
        import inspect
        src = inspect.getsource(Env.step)
        self.assertIn("rank", src.split("observe_block")[0].rsplit("_vid", 1)[0][-400:])


_CB = "MilestoneCheckpointCallback"


def _train_cb():
    import train
    return getattr(train, _CB)


class PromotionAuthorityTests(unittest.TestCase):
    def test_train_does_not_import_the_twoby2_promotion_module(self):
        import inspect
        import train
        src = inspect.getsource(train)
        self.assertNotIn("from twoby2.promotion", src)
        self.assertNotIn("import twoby2.promotion", src)
        self.assertNotIn("evaluate_promotion(", src)

    def test_twoby2_promotion_is_marked_not_wired(self):
        import inspect
        import twoby2.promotion as p
        self.assertIn("NOT WIRED INTO THE LIVE TRAINER", inspect.getsource(p))
        self.assertEqual(p._LIVE_AUTHORITY, "train._score")

    def test_score_is_geographic_only_no_level_xp_ko(self):
        import inspect
        src = inspect.getsource(_train_cb()._score)
        for banned in ("level", "experience", "kos", "battle_win",
                       "enemy_faints", "damage"):
            self.assertNotIn(banned, src.lower())
        for wanted in ("max_stage", "max_badges", "permille"):
            self.assertIn(wanted, src)

    def test_horizon_uses_published_champion_version_not_learner(self):
        import inspect
        src = inspect.getsource(_train_cb()._maybe_advance_nav_horizon)
        self.assertIn("_published_champion_version", src)
        self.assertNotIn("navigation_champion_version=int(self.version)", src)

    def test_rejection_detail_is_recorded(self):
        import inspect
        src = inspect.getsource(_train_cb()._evaluate)
        self.assertIn("last_eval_detail", src)


class NavObjectiveGeometricFallbackTests(unittest.TestCase):
    """The obs must keep target + dx/dy even with an incomplete directed path."""

    def test_source_keeps_target_and_geometric_fields(self):
        import inspect
        src = inspect.getsource(Env._nav_objective)
        self.assertIn("approach_mode", src)
        self.assertIn('"geometric"', src)
        self.assertIn("target_manhattan", src)
        # target_valid stays confirmed-route-only (honest)
        self.assertIn("valid=True, directed=True, confirmed=True", src)
        self.assertIn("_verified and not out[\"valid\"]", src)


if __name__ == "__main__":
    unittest.main()
