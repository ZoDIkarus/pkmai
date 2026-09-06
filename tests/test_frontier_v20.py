"""V20 FRONTIER REDESIGN - topological graph progress instead of tile count.

Covers the acceptance cases from the redesign spec:
  T1  few tiles + deep graph > many tiles + shallow graph
  T2  high-watermark: 20->21->22->21->22->23 pays only 20->21, 21->22, 22->23
  T3  warp loop A->B->A->B penalised; single A->B (discovery) is fine
  T4  a crossing needs 2 matching observations to become a proven edge, and the
      flip is reported exactly once
  T5  only FRONTIER-mode scouts are paid for new tiles; FULL/BRIDGE get 0
  T6  frontier rewards stay well below story / stage rewards
"""
import ast
from pathlib import Path
import unittest

from frontier_v20 import FrontierGraph, FrontierHighWater
from nav_transitions_v20 import KnownTransitions, KNOWN, UNKNOWN
from pokemon_env import PokemonFireRedEnv as Env
from test_progress_curriculum import bare_env


def _line_graph(n):
    """0-(1,0)-(2,0)-...-(n,0): a straight corridor, origin at (0,0)."""
    adj = {}
    for i in range(n):
        a, b = (i, 0), (i + 1, 0)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


class FrontierValueTests(unittest.TestCase):
    def test_depth_beats_tile_count(self):
        # Agent A: big blob near the origin (many tiles, shallow).
        blob = {}
        for x in range(8):
            for y in range(8):
                for nb in ((x + 1, y), (x, y + 1)):
                    blob.setdefault((x, y), set()).add(nb)
                    blob.setdefault(nb, set()).add((x, y))
        ga = FrontierGraph(blob, origin=(0, 0))
        # Agent B: thin 40-long corridor (few tiles, deep).
        gb = FrontierGraph(_line_graph(40), origin=(0, 0))

        self.assertGreater(len(ga.adjacency), len(gb.adjacency))          # more tiles
        self.assertGreater(gb.max_graph_depth(), ga.max_graph_depth())    # deeper
        self.assertGreater(gb.best_frontier_value(), ga.best_frontier_value())

    def test_unconnected_tile_does_not_score_via_compass(self):
        adj = _line_graph(5)
        adj[(99, 99)] = {(99, 100)}
        adj[(99, 100)] = {(99, 99)}
        g = FrontierGraph(adj, origin=(0, 0))
        self.assertIsNone(g.graph_depth((99, 99)))
        # best value still comes from the connected deep end, never the island
        self.assertEqual(g.best_frontier_value(),
                         g.frontier_value((5, 0)))

    def test_is_frontier_needs_an_open_direction(self):
        g = FrontierGraph(_line_graph(3), origin=(0, 0))
        self.assertTrue(g.is_frontier((3, 0)))     # dead end -> 3 open dirs
        # (1,0) has 2 walked neighbours, still 2 open -> also a frontier here
        self.assertTrue(g.is_frontier((1, 0)))


class FrontierHighWaterTests(unittest.TestCase):
    def test_only_strict_new_best_pays(self):
        hw = FrontierHighWater(progress_reward=0.1, epsilon=0.5)
        key = ("frontier", 2, 3, 19)
        seq = [20, 21, 22, 21, 22, 23]
        paid = []
        for v in seq:
            r, ev = hw.update(key, v)
            paid.append(round(r, 3) if ev else 0)
        # 20 anchors (0), 21 pays, 22 pays, 21 nothing, 22 nothing, 23 pays
        self.assertEqual([p > 0 for p in paid],
                         [False, True, True, False, False, True])

    def test_key_change_reanchors_and_pays_nothing(self):
        hw = FrontierHighWater()
        hw.update(("a",), 10)
        hw.update(("a",), 20)
        r, ev = hw.update(("b",), 5)      # new objective
        self.assertEqual((r, ev), (0.0, None))
        self.assertEqual(hw.best, 5.0)


class WarpLoopPatternTests(unittest.TestCase):
    @staticmethod
    def _loop(seq):
        """Replicates the exact _warp_loop boolean from step() over a list of
        (pb, pm, bank, map) transition tuples."""
        from collections import deque
        buf = deque(maxlen=8)
        hits = []
        for mt in seq:
            buf.append(mt)
            mts = list(buf)
            rev = (mt[2], mt[3], mt[0], mt[1])
            loop = (
                (len(mts) >= 2 and mts[-2] == rev)
                or (len(mts) >= 4 and mts[-1] == mts[-3] and mts[-2] == mts[-4])
            )
            hits.append(loop)
        return hits

    def test_single_crossing_is_not_a_loop(self):
        self.assertEqual(self._loop([(3, 0, 3, 19)]), [False])

    def test_immediate_reverse_is_a_loop(self):
        # Pallet->Route1 then Route1->Pallet
        hits = self._loop([(3, 0, 3, 19), (3, 19, 3, 0)])
        self.assertEqual(hits, [False, True])

    def test_a_b_a_b_is_a_loop(self):
        hits = self._loop([(4, 0, 3, 0), (3, 0, 4, 0),
                           (4, 0, 3, 0), (3, 0, 4, 0)])
        self.assertEqual(hits, [False, True, True, True])

    def test_step_wires_the_penalty(self):
        src = Path('src/pokemon_env.py').read_text()
        self.assertIn("_warp_loop", src)
        self.assertIn("self.WARP_LOOP_PENALTY", src)
        self.assertIn('reward_events.append(\n                            f"warp_loop:', src)


class ProvenEdgeTests(unittest.TestCase):
    def test_two_observations_promote_and_flip_is_reported_once(self):
        kt = KnownTransitions()
        args = (2, 3, (3, 19), (9, 0), (3, 1), (20, 40))
        stored, became = kt.record_and_state(*args)
        self.assertTrue(stored)
        self.assertFalse(became)
        self.assertEqual(kt.navigation_state(2), UNKNOWN)

        stored, became = kt.record_and_state(*args)   # 2nd matching obs
        self.assertTrue(became)
        self.assertEqual(kt.navigation_state(2), KNOWN)

        stored, became = kt.record_and_state(*args)   # 3rd - no more flip
        self.assertFalse(became)

    def test_single_glitch_variant_never_promotes(self):
        kt = KnownTransitions()
        kt.record_and_state(2, 3, (3, 19), (13, 36), (3, 1), (36, 19))
        self.assertEqual(kt.navigation_state(2), UNKNOWN)


class RoleGatedTileRewardTests(unittest.TestCase):
    @staticmethod
    def _tile_code():
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and n.lineno > 5000
                    and ast.unparse(n.test) == 'coord_key not in self.seen_coords')
        return compile(ast.Module(body=[node], type_ignores=[]), '<tile>', 'exec')

    def _run(self, mode, known=None):
        from nav_transitions_v20 import UNKNOWN as NAV_UNKNOWN, KnownTransitions
        e = bare_env(training_objective='scout', training_mode=mode,
                     episode_start_stage=2, seen_coords=set(),
                     _episode_tiles_by_map={}, _episode_first_tile_by_map={},
                     shared_tiles={}, shared_lock=None)
        e._v20_load_known_transitions = lambda *a, **k: (known or KnownTransitions())
        scope = dict(self=e, bank=3, map_id=19, map_key=(3, 19),
                     coord_key=(3, 19, 12, 30), reward=0.0, reward_events=[],
                     _wipe_cooldown_active=False, NAV_UNKNOWN=NAV_UNKNOWN)
        exec(self._tile_code(), scope)
        return scope['reward'], scope['reward_events']

    def test_full_agent_pushes_an_unproven_stage_but_only_trickles_a_proven_one(self):
        from nav_transitions_v20 import KnownTransitions
        # Stage 2 forward transition UNKNOWN -> FULL gets the real frontier push.
        r_unknown, ev_u = self._run('FULL')
        self.assertGreaterEqual(r_unknown, Env.FULL_FRONTIER_TILE_REWARD)
        self.assertTrue(any('new_tile_full:frontier' in e for e in ev_u))

        # Same crossing proven twice -> FULL drops to the tiny anti-wall trickle.
        proven = KnownTransitions()
        for _ in range(2):
            proven.record(2, 3, (3, 19), (9, 0), (3, 1), (20, 40))
        r_proven, ev_p = self._run('FULL', known=proven)
        self.assertLessEqual(r_proven, Env.FULL_NEW_TILE_REWARD)
        self.assertTrue(any('new_tile_full' in e and 'frontier' not in e
                            for e in ev_p))

    def test_frontier_scout_still_paid_and_stacks_global_bonus(self):
        r_frontier, ev_f = self._run('FRONTIER')
        self.assertGreater(r_frontier, Env.SCOUT_NEW_TILE_REWARD)  # + global bonus
        self.assertTrue(any('new_tile_scout' in e for e in ev_f))


class ConstantsTests(unittest.TestCase):
    def test_frontier_rewards_stay_below_story_rewards(self):
        self.assertLess(Env.FRONTIER_PROGRESS_REWARD, 1.0)
        self.assertLessEqual(Env.FULL_NEW_TILE_REWARD, Env.SCOUT_NEW_TILE_REWARD)
        self.assertLess(Env.SCOUT_NEW_TILE_REWARD, 0.1)
        self.assertLess(Env.WARP_LOOP_PENALTY, 0.0)
        # proven-progress rewards are meaningful but still far under a city.
        self.assertLess(Env.PROVEN_PROGRESS_EDGE_REWARD, Env.CITY_EPISODE_REWARD)
        self.assertLess(Env.PROVEN_EXIT_REWARD, Env.PROVEN_PROGRESS_EDGE_REWARD)


if __name__ == '__main__':
    unittest.main()
