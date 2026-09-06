"""Exercise actual selection, disk checkpoint loading and per-step reward code."""
import ast
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pokemon_env
from pokemon_env import PokemonFireRedEnv as Env
from curriculum_v20 import CurriculumState
from nav_transitions_v20 import KnownTransitions
from loop_guard import ShortCycleGuard
from test_progress_curriculum import bare_env


class CheckpointIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(pokemon_env, 'SHARED_CURRICULUM_DIR', str(self.root)))
        self.stack.enter_context(patch.object(pokemon_env, 'read_player_party', return_value=[
            {'checksum_ok': True, 'cur_hp': 21, 'max_hp': 21, 'moves': [{'pp': 35}]}]))
        self.state = CurriculumState()
        self.state.record_discovery(2)

    def env(self, rank):
        e = bare_env(rank=rank, n_envs=60, rank_state_dir=str(self.root),
                     total_steps=0, completed_episodes=0, shared_lock=None)
        e._champion_full_starter_ready = lambda: False
        e._v20_load_state = lambda **kw: self.state
        e.memory = b'initial'
        e.env = SimpleNamespace(em=SimpleNamespace(
            get_state=lambda: e.memory,
            set_state=lambda data: setattr(e, 'memory', data)))
        return e

    def write_checkpoint(self, name, stage, map_id):
        data = name.encode()
        with gzip.open(self.root / (name + '.state.gz'), 'wb') as f:
            f.write(data)
        meta = dict(state_validation=1, stage=stage, bank=3, map=map_id,
                    x=12, y=30, has_starter=True,
                    state_sha256=hashlib.sha256(data).hexdigest())
        (self.root / (name + '.meta.json')).write_text(json.dumps(meta))
        return meta

    def test_rank_selection_and_real_loader_for_all_modes(self):
        metas = {name: self.write_checkpoint(name, stage, map_id)
                 for name, stage, map_id in [('stage_1', 1, 0), ('stage_2', 2, 19),
                                            ('stage_frontier_2', 2, 19)]}
        # 60-env layout: FULL 0-20, BRIDGE 21-40, FRONTIER 41-50, RETENTION
        # 51-55, FIGHTER 56-59. FIGHTER reuses the FRONTIER anchor.
        for rank, expected in [(0, 'beginning'), (25, 'stage_1'),
                               (45, 'stage_frontier_2'), (58, 'stage_frontier_2')]:
            e = self.env(rank)
            self.assertEqual(e._choose_episode_start(), expected)
            if expected == 'beginning':
                continue
            meta = metas[expected]
            loc = dict(trusted=True, map_bank=meta['bank'], map_id=meta['map'],
                       x_pos=meta['x'], y_pos=meta['y'])
            with patch.object(pokemon_env, 'read_player_location', return_value=loc):
                self.assertTrue(e._load_curriculum_state(expected))
            self.assertEqual(e.memory, expected.encode())
            self.assertEqual(e.training_objective, 'scout')
        for _ in range(20):
            self.state.record_transition_attempt(1, True, full_chain=True)
        e = self.env(55)
        self.assertEqual(e._choose_episode_start(), 'stage_1')

    def test_frontier_falls_back_to_entry_when_no_frontier_exists(self):
        self.write_checkpoint('stage_2', 2, 19)
        self.assertEqual(self.env(45)._choose_episode_start(), 'stage_2')

    def test_missing_checkpoints_fall_back_to_real_beginning(self):
        for rank in (24, 45, 55):
            self.assertEqual(self.env(rank)._choose_episode_start(), 'beginning')

    def test_corrupt_frontier_is_rejected_without_changing_emulator(self):
        self.write_checkpoint('stage_frontier_2', 2, 19)
        with gzip.open(self.root / 'stage_frontier_2.state.gz', 'wb') as f:
            f.write(b'corrupt')
        e = self.env(45)
        self.assertFalse(e._load_curriculum_state('stage_frontier_2'))
        self.assertEqual(e.memory, b'initial')

    def test_pallet_entry_saves_and_loads_through_actual_disk_methods(self):
        e = self.env(25)  # BRIDGE rank (FULL 0-20, BRIDGE 21-40)
        e.memory = b'pallet state'
        e.route_steps = 10
        e._starter_species = lambda: 7
        loc = dict(trusted=True, map_bank=3, map_id=0, x_pos=16, y_pos=14)
        with patch.object(pokemon_env, 'read_player_location', return_value=loc):
            self.assertTrue(e._save_stage_checkpoint(1, 3, 0, 16, 14))
            self.assertEqual(e._choose_episode_start(), 'stage_1')
            e.memory = b'initial'
            self.assertTrue(e._load_curriculum_state('stage_1'))
            self.assertEqual(e.memory, b'pallet state')


class RewardAndLoopIntegrationTests(unittest.TestCase):
    @staticmethod
    def tile_code():
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and n.lineno > 5000
                    and ast.unparse(n.test) == 'coord_key not in self.seen_coords')
        return compile(ast.Module(body=[node], type_ignores=[]), '<tile reward>', 'exec')

    def test_frontier_explores_spawn_map_but_not_earlier_stage(self):
        # V20 frontier redesign: only FRONTIER-mode scouts are paid for tiles.
        from nav_transitions_v20 import UNKNOWN as NAV_UNKNOWN, KnownTransitions
        code = self.tile_code()
        for map_id, expected in [(19, True), (0, False)]:
            e = bare_env(training_objective='scout', training_mode='FRONTIER',
                         episode_start_stage=2,
                         seen_coords=set(), _episode_tiles_by_map={},
                         _episode_first_tile_by_map={},
                         shared_tiles={}, shared_lock=None)
            e._v20_load_known_transitions = lambda *a, **k: KnownTransitions()
            scope = dict(self=e, bank=3, map_id=map_id, map_key=(3, map_id),
                         coord_key=(3, map_id, 12, 30), reward=0., reward_events=[],
                         _wipe_cooldown_active=False, NAV_UNKNOWN=NAV_UNKNOWN)
            exec(code, scope)
            self.assertEqual(scope['reward'] > 0, expected)
            initial = scope['reward']
            exec(code, scope)
            self.assertEqual(scope['reward'], initial, 'same tile must never repay')
            if expected:
                self.assertEqual(e._episode_tiles_by_map[(3, map_id)], 1)

    def test_cached_positions_do_not_hide_two_or_three_tile_cycles(self):
        for period in (2, 3):
            g = ShortCycleGuard(truncate_after=30)
            results = [g.update((3, 0, (i // 4) % period, 18), (1, 0, 1, 1))
                       for i in range(200)]
            self.assertTrue(any(r['truncate'] for r in results))
            self.assertEqual(g.update((3, 0, 9, 9), (1, 0, 1, 1), active=False)['penalty'], 0)
            self.assertFalse(g.update((3, 19, 12, 39), (2, 0, 1, 1))['cycle'])

    def test_straight_exploration_with_cached_reads_is_not_a_cycle(self):
        g = ShortCycleGuard()
        for i in range(200):
            self.assertFalse(g.update((3, 19, 12, i // 4), 1)['cycle'])

    def test_known_target_reaches_policy_input_unknown_exit_has_no_target(self):
        known = KnownTransitions()
        known.record(1, 2, (3, 0), (13, 0), (3, 19), (13, 39))
        e = bare_env(training_objective='scout', navigation_revision=0,
                     _nav_target_cache=None, total_steps=0, left_house_rewarded=True)
        e._v20_load_known_transitions = lambda: known
        e._target_coords_for_stage = lambda *a: []
        e._progress_targets_for_map = lambda *a: []
        self.assertEqual(e._nav_target(3, 0, 3, 18), (13, 0))
        self.assertIsNone(e._nav_target(3, 19, 12, 39))


class StepWiringTests(unittest.TestCase):
    def test_pallet_entry_capture_requires_three_fresh_safe_reads(self):
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and n.lineno > 5000 and isinstance(n.test, ast.BoolOp)
                    and isinstance(n.test.values[0], ast.Name)
                    and n.test.values[0].id == '_route_roller')
        code = compile(ast.Module(body=[node], type_ignores=[]), '<capture>', 'exec')
        captures = []
        e = bare_env(has_target_starter=True, player_party_cache=[{'cur_hp': 20}],
                     training_mode='BRIDGE', current_reward=0)
        e._save_stage_checkpoint = lambda *a, **kw: captures.append(a) or False
        scope = dict(self=e, _route_roller=True, location_refreshed=False,
                     in_battle=0, _wipe_cooldown_active=False, loc={'valid': True},
                     bank=3, map_id=0, x=16, y=14, reward=0, milestone_saved=None)
        for _ in range(8):
            exec(code, scope)
        self.assertEqual(captures, [])
        scope['location_refreshed'] = True
        exec(code, scope)
        exec(code, scope)
        self.assertEqual(captures, [])
        exec(code, scope)
        self.assertEqual(captures[0][:3], (1, 3, 0))

    def test_reset_bookkeeping_preserves_selected_bridge_bottleneck(self):
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
        begin = next(i for i, n in enumerate(reset.body)
                     if ast.unparse(n) == 'self.episode_start_bottleneck = None')
        end = next(i for i, n in enumerate(reset.body)
                   if ast.unparse(n) == 'self._v20_outcome_recorded = False')
        code = compile(ast.Module(body=reset.body[begin:end + 1], type_ignores=[]), '<reset selection>', 'exec')
        e = bare_env(env=object(), player_party_cache=[])
        def choose():
            e.episode_start_bottleneck = 2
            return 'stage_2'
        e._choose_episode_start = choose
        e._load_curriculum_state = lambda name: True
        e._read_info_with_idle_frame = lambda: {}
        e._set_baseline_from_info = lambda *a, **kw: None
        e._world_stage = lambda: 2
        scope = dict(self=e, party_health=lambda p: {}, read_player_location=lambda *a, **kw: {},
                     BattleState=lambda *a: None, MainBattleReader=lambda: None,
                     read_enemy_party=lambda *a: [])
        exec(code, scope)
        self.assertEqual(e.episode_start_bottleneck, 2)
        self.assertEqual(e.episode_start, 'stage_2')
