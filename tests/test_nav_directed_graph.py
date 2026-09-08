import json
import os
import tempfile
import unittest

import nav_graph as ng
from nav_graph import DirectedNavGraph, UP, DOWN, LEFT, RIGHT
from nav_shaping_state import NavAgentState, NavGlobalState, objective_key
from loop_guard import RegionLoopGuard

K1 = objective_key(1, 1, "pallet_exit_edge", [(12, 0)])
K2 = objective_key(1, 2, "known_transition_s2", [(12, 39)])

MAP = (3, 19)   # Route 1


class DirectedEdgeTests(unittest.TestCase):
    def test_walk_edge_is_directed_and_only_bidir_when_both_observed(self):
        g = DirectedNavGraph()
        g.observe_move(MAP, (5, 10), RIGHT, (6, 10), step=1)
        # A -> B known; B -> A not yet
        e = g.edge_at(MAP, (5, 10), RIGHT)
        self.assertEqual(e.kind, "walk")
        self.assertFalse(e.both_dirs_observed)
        self.assertIsNone(g.edge_at(MAP, (6, 10), LEFT))
        # now observe the reverse
        g.observe_move(MAP, (6, 10), LEFT, (5, 10), step=2)
        self.assertTrue(g.edge_at(MAP, (5, 10), RIGHT).both_dirs_observed)
        self.assertTrue(g.edge_at(MAP, (6, 10), LEFT).both_dirs_observed)

    def test_multi_tile_delta_is_not_recorded(self):
        # The overworld location is SAMPLED (not read per frame), so a >1-tile
        # delta is continuous walking / a missed warp / a battle transition -
        # never a trustworthy single edge. It records NOTHING (the old code
        # stored it as a one-way `jump_or_ledge` and polluted the graph).
        g = DirectedNavGraph()
        self.assertEqual(g.observe_move(MAP, (10, 4), DOWN, (10, 7), step=1),
                         "ambiguous_multi_tile")
        self.assertIsNone(g.edge_at(MAP, (10, 4), DOWN))
        self.assertIsNone(g.edge_at(MAP, (10, 7), UP))
        # a 1+1 diagonal delta is also discarded (records nothing)
        self.assertIn(g.observe_move(MAP, (2, 2), RIGHT, (3, 3), step=1),
                      ("ambiguous_multi_tile", "ambiguous_diagonal"))
        self.assertIsNone(g.edge_at(MAP, (2, 2), RIGHT))

    def test_edge_action_is_derived_from_geometry_not_the_button(self):
        # 4-step sampling attaches the last pressed button, not the direction
        # that produced the net move. The stored edge action follows the delta.
        g = DirectedNavGraph()
        g.observe_move(MAP, (5, 10), DOWN, (6, 10), step=1)   # pressed DOWN, moved RIGHT
        self.assertIsNone(g.edge_at(MAP, (5, 10), DOWN))
        e = g.edge_at(MAP, (5, 10), RIGHT)
        self.assertIsNotNone(e)
        self.assertEqual(e.to, (6, 10))

    def test_a_later_move_clears_a_block(self):
        g = DirectedNavGraph()
        for v in range(3):
            g.observe_block(MAP, (7, 7), UP, step=100 + v * 500, visit_id=v)
        self.assertTrue(g.is_blocked(MAP, (7, 7), UP, now_step=2000))
        g.observe_move(MAP, (7, 7), UP, (7, 6), step=3000)
        self.assertFalse(g.is_blocked(MAP, (7, 7), UP, now_step=3000))

    def test_consecutive_no_move_steps_do_not_forge_a_static_block(self):
        g = DirectedNavGraph()
        # same visit_id (NPC blocking for a few frames at one encounter)
        for step in range(10):
            g.observe_block(MAP, (7, 7), RIGHT, step=step, visit_id=0)
        self.assertFalse(g.is_blocked(MAP, (7, 7), RIGHT, now_step=10))
        # three temporally independent encounters -> static
        for v in (1, 2, 3):
            g.observe_block(MAP, (7, 7), RIGHT, step=v * 400, visit_id=v)
        self.assertTrue(g.is_blocked(MAP, (7, 7), RIGHT, now_step=1300))

    def test_static_block_is_permanent_until_a_move_clears_it(self):
        # A wall does not move. The env's now_step RESETS every episode while a
        # persisted last_confirmed_step does not, so a step-based TTL made every
        # block read as active for a 6000-step window of EVERY later episode -
        # that is exactly what walled off the Route-1 corridor.
        g = DirectedNavGraph()
        for v in (0, 1, 2):
            g.observe_block(MAP, (1, 1), LEFT, step=v, visit_id=v)
        self.assertTrue(g.is_blocked(MAP, (1, 1), LEFT, now_step=100))
        self.assertTrue(g.is_blocked(MAP, (1, 1), LEFT, now_step=10_000_000))
        g.observe_move(MAP, (1, 1), LEFT, (0, 1), step=5)     # a real move clears it
        self.assertFalse(g.is_blocked(MAP, (1, 1), LEFT, now_step=6))

    def test_dynamic_block_never_cuts_a_route(self):
        g = DirectedNavGraph()
        g.observe_block(MAP, (4, 4), UP, step=1, visit_id=0)   # blocked once -> dynamic
        self.assertFalse(g.is_blocked(MAP, (4, 4), UP, now_step=2))

    def test_legacy_edges_import_as_unknown_and_are_only_a_fallback(self):
        g = DirectedNavGraph()
        n = g.import_legacy_edges([(3, 19, 5, 10, 6, 10)])
        self.assertEqual(n, 2)          # both directions, both "unknown"
        self.assertEqual(g.edge_at(MAP, (5, 10), RIGHT).kind, "unknown")
        self.assertTrue(g.edge_at(MAP, (5, 10), RIGHT).legacy)
        # strict routing ignores legacy; fallback routing uses it
        self.assertIsNone(g._bfs(MAP, (5, 10), {(6, 10)}, 0, 100, False))
        self.assertIsNotNone(g._bfs(MAP, (5, 10), {(6, 10)}, 0, 100, True))


def _route1_wall_graph():
    """Fixture, north-is-progress (target ``T = (10,0)``):

      * correct path: from S(9,7) go RIGHT along y=7 to x=14, UP to y=0, LEFT to T.
      * left approach S -> (9,4) is a dead end: UP is a wall, the only exit is a
        one-way ledge ``(9,4) --DOWN--> (9,12)`` that drops the agent far south.
      * a recovery corridor exists along y=12 back to x=14 and up - much longer.
    """
    g = DirectedNavGraph()

    def walk_both(a, b, act):
        g.observe_move(MAP, a, act, b, step=1)
        g.observe_move(MAP, b, ng.REVERSE[act], a, step=1)

    # correct path
    for x in range(9, 14):
        walk_both((x, 7), (x + 1, 7), RIGHT)          # y=7 corridor 9..14
    for y in range(12, 0, -1):
        walk_both((14, y), (14, y - 1), UP)           # x=14 column 12..0
    for x in range(14, 10, -1):
        walk_both((x, 0), (x - 1, 0), LEFT)           # y=0 row 14..10 -> T
    # left dead-end approach: walkable both ways up to (9,5); the final hop to
    # (9,4) is one-way (you can walk up but the only way down is the ledge)
    walk_both((9, 7), (9, 6), UP)
    walk_both((9, 6), (9, 5), UP)
    g.observe_move(MAP, (9, 5), UP, (9, 4), step=1)
    # one-way ledge off the dead end
    g.observe_move(MAP, (9, 4), DOWN, (9, 12), step=1)
    # southern recovery corridor y=12 from x=9 to x=14
    for x in range(9, 14):
        walk_both((x, 12), (x + 1, 12), RIGHT)
    return g


class Route1DirectedBFSTests(unittest.TestCase):
    def test_directed_bfs_takes_the_right_path_around_the_wall(self):
        g = _route1_wall_graph()
        res = g.directed_bfs(MAP, (9, 7), [(10, 0)])
        self.assertIsNotNone(res)
        self.assertEqual(res["next_action"], RIGHT)   # never UP into the dead end
        self.assertEqual(res["distance"], 16)         # 5 right + 7 up + 4 left

    def test_multi_tile_ledge_hop_is_not_stored(self):
        g = _route1_wall_graph()
        # the >1-tile hop off the dead end is not a recorded edge either way
        self.assertIsNone(g.edge_at(MAP, (9, 4), DOWN))
        self.assertIsNone(g.edge_at(MAP, (9, 12), UP))

    def test_recovery_route_after_an_accidental_jump(self):
        g = _route1_wall_graph()
        res = g.directed_bfs(MAP, (9, 12), [(10, 0)])
        self.assertIsNotNone(res)
        self.assertEqual(res["next_action"], RIGHT)

    def test_next_hop_direction_changes_after_one_step(self):
        # a corner: from S the first hop is RIGHT, but from the very next tile
        # the optimal hop is UP. An off-by-one (checking the objective at the
        # NEW tile against the action that came FROM the old tile) would miss it.
        g = DirectedNavGraph()

        def wb(a, b, act):
            g.observe_move(MAP, a, act, b, step=1)
            g.observe_move(MAP, b, ng.REVERSE[act], a, step=1)
        wb((0, 5), (1, 5), RIGHT)          # S -> corner
        wb((1, 5), (1, 0), UP)             # corner -> ... (5 up)
        for y in range(5, 0, -1):
            wb((1, y), (1, y - 1), UP)
        r0 = g.directed_bfs(MAP, (0, 5), [(1, 0)])
        r1 = g.directed_bfs(MAP, (1, 5), [(1, 0)])
        self.assertEqual(r0["next_action"], RIGHT)
        self.assertEqual(r1["next_action"], UP)

    def test_legacy_only_route_is_not_confirmed(self):
        g = DirectedNavGraph()
        g.import_legacy_edges([(3, 19, 0, 0, 1, 0), (3, 19, 1, 0, 2, 0)])
        res = g.directed_bfs(MAP, (0, 0), [(2, 0)])
        self.assertIsNotNone(res)
        self.assertFalse(res["confirmed"])          # display only
        self.assertIsNone(g.directed_bfs(MAP, (0, 0), [(2, 0)], allow_legacy=False))
        # one real observation confirms that hop
        g.observe_move(MAP, (0, 0), RIGHT, (1, 0), step=5)
        g.observe_move(MAP, (1, 0), RIGHT, (2, 0), step=6)
        self.assertTrue(g.directed_bfs(MAP, (0, 0), [(2, 0)])["confirmed"])

    def test_going_up_the_dead_end_and_ledging_worsens_the_directed_distance(self):
        g = _route1_wall_graph()
        d_start = g.directed_bfs(MAP, (9, 7), [(10, 0)])["distance"]        # 16
        d_deadend = g.directed_bfs(MAP, (9, 4), [(10, 0)])["distance"]      # via ledge
        d_landing = g.directed_bfs(MAP, (9, 12), [(10, 0)])["distance"]
        self.assertGreater(d_deadend, d_start)
        self.assertGreater(d_landing, d_start)


class RegionLoopGuardTests(unittest.TestCase):
    def test_ledge_cycle_is_detected_and_truncates(self):
        guard = RegionLoopGuard(min_steps=20, truncate_steps=60)
        cycle = [((9, 7), UP), ((9, 6), UP), ((9, 5), UP), ((9, 4), DOWN),
                 ((9, 8), UP)]
        res = None
        for i in range(200):
            pos, act = cycle[i % len(cycle)]
            res = guard.update(pos, act, directed_distance=12)
            if res["truncate"]:
                break
        self.assertTrue(res["loop"])
        self.assertEqual(res["kind"], "ledge_loop")
        self.assertTrue(res["truncate"])

    def test_a_real_detour_that_keeps_improving_is_not_flagged(self):
        guard = RegionLoopGuard(min_steps=20)
        for i in range(400):
            # every few steps the directed distance improves -> progress_event
            if i % 5 == 0:
                guard.progress_event()
            r = guard.update((i % 30, i % 7), RIGHT, directed_distance=100 - i)
            self.assertFalse(r["loop"])


class NavAgentStateTests(unittest.TestCase):
    def _s(self, d, *, run_id=1):
        return NavAgentState.load(os.path.join(d, "shaping", "agent_00.json"),
                                  run_id=run_id)

    def test_highwater_is_per_objective_key_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._s(d)
            self.assertTrue(s.record_distance(K1, 20))
            self.assertFalse(s.record_distance(K1, 20))
            self.assertFalse(s.record_distance(K1, 25))
            self.assertTrue(s.record_distance(K1, 18))
            # distance 0 at the OLD objective must not block the NEW one
            self.assertTrue(s.record_distance(K1, 0))
            self.assertTrue(s.record_distance(K2, 30))
            s.save()
            s2 = self._s(d)
            self.assertEqual(s2.best_distance(K1), 0)
            self.assertEqual(s2.best_distance(K2), 30)
            self.assertFalse(s2.record_distance(K2, 31))
            self.assertTrue(s2.record_distance(K2, 29))

    def test_wipe_freezes_shaping_per_key_until_strictly_past(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._s(d)
            s.record_distance(K1, 10)
            s.on_wipe(K1)
            self.assertTrue(s.shaping_frozen(K1, 40))
            self.assertTrue(s.shaping_frozen(K1, 10))    # equal -> still frozen
            self.assertFalse(s.shaping_frozen(K1, 9))    # strictly past
            self.assertFalse(s.shaping_frozen(K2, 999))  # other objective unaffected

    def test_next_hop_edges_are_run_wide_and_survive_a_stage_change(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._s(d)
            s.mark_next_hop_paid(3, 19, (5, 10), RIGHT)
            self.assertTrue(s.next_hop_already_paid(3, 19, (5, 10), RIGHT))
            # a stage / objective change does NOT re-open it (item 5)
            s.record_distance(K2, 5)          # "new objective"
            self.assertTrue(s.next_hop_already_paid(3, 19, (5, 10), RIGHT))
            s.on_wipe(K2)
            self.assertTrue(s.next_hop_already_paid(3, 19, (5, 10), RIGHT))
            s.save()
            # only a fresh training run clears it
            fresh = self._s(d, run_id=2)
            self.assertFalse(fresh.next_hop_already_paid(3, 19, (5, 10), RIGHT))

    def test_wild_win_and_catch_survive_everything_but_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._s(d)
            self.assertEqual(s.add_wild_win(3, 19), 1)
            self.assertEqual(s.add_wild_win(3, 19), 2)
            self.assertTrue(s.mark_caught(19))
            self.assertFalse(s.mark_caught(19))          # dup
            s.on_wipe(K1)
            s.record_distance(K2, 3)
            s.save()
            same = self._s(d, run_id=1)
            self.assertEqual(same.wild_wins_for(3, 19), 2)
            self.assertTrue(same.already_caught(19))
            fresh = self._s(d, run_id=2)
            self.assertEqual(fresh.wild_wins_for(3, 19), 0)
            self.assertFalse(fresh.already_caught(19))

    def test_global_state_start_fresh_run_is_monotone(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "nav_global.json")
            g = NavGlobalState(p)
            self.assertEqual(g.start_fresh_run(), 1)
            self.assertEqual(NavGlobalState(p).start_fresh_run(), 2)

    def test_two_agents_never_lose_each_others_writes(self):
        # per-agent files are separate -> no shared write path -> no lost update
        with tempfile.TemporaryDirectory() as d:
            a = NavAgentState.load(os.path.join(d, "s", "agent_00.json"), run_id=1)
            b = NavAgentState.load(os.path.join(d, "s", "agent_01.json"), run_id=1)
            for i in range(20):
                a.record_distance(K1, 100 - i)
                b.add_wild_win(3, 19)
                a.mark_caught(i)
                a.save(); b.save()
            a2 = NavAgentState.load(os.path.join(d, "s", "agent_00.json"), run_id=1)
            b2 = NavAgentState.load(os.path.join(d, "s", "agent_01.json"), run_id=1)
            self.assertEqual(a2.best_distance(K1), 81)
            self.assertEqual(len(a2.caught_species), 20)
            self.assertEqual(b2.wild_wins_for(3, 19), 20)   # untouched by A
            self.assertEqual(b2.best_distance(K1), None)     # A's highwater not on B

    def test_wipe_of_one_agent_does_not_touch_another(self):
        with tempfile.TemporaryDirectory() as d:
            a = NavAgentState.load(os.path.join(d, "s", "a.json"), run_id=1)
            b = NavAgentState.load(os.path.join(d, "s", "b.json"), run_id=1)
            a.record_distance(K1, 5)
            b.record_distance(K1, 5)
            a.on_wipe(K1)
            self.assertTrue(a.shaping_frozen(K1, 6))
            self.assertFalse(b.shaping_frozen(K1, 6))       # B keeps earning


if __name__ == "__main__":
    unittest.main()
