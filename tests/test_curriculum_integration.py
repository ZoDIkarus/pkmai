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

    def _frontier_env(self):
        e = self.env(45)  # FRONTIER rank
        e.memory = b'frontier state'
        e.route_steps = 20
        e._starter_species = lambda: 7
        e._stage_at_current_location = lambda *a: 2
        e._current_frontier_value = lambda *a: 50.0
        e._v20_can_create_stage_checkpoint = lambda *a: True
        return e

    def _save(self, e, kind, **frontier_kw):
        loc = dict(trusted=True, map_bank=3, map_id=19, x_pos=10, y_pos=8)
        with patch.object(pokemon_env, 'read_player_location', return_value=loc):
            return e._save_stage_checkpoint(2, 3, 19, 10, 8, kind=kind, **frontier_kw)

    def test_case1_entry_checkpoint_still_rejects_sub_80pct_party(self):
        e = self._frontier_env()
        hurt = [{'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]},
                {'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]}]
        with patch.object(pokemon_env, 'read_player_party', return_value=hurt):
            self.assertFalse(self._save(e, 'entry'))       # 60% -> below strict 80%
        healthy = [{'checksum_ok': True, 'cur_hp': 20, 'max_hp': 20, 'status': 0,
                    'moves': [{'pp': 10}]}]
        with patch.object(pokemon_env, 'read_player_party', return_value=healthy):
            self.assertTrue(self._save(e, 'entry'))

    def test_case2_frontier_checkpoint_accepts_viable_not_ready_party(self):
        e = self._frontier_env()
        hurt = [{'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]},
                {'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]}]
        with patch.object(pokemon_env, 'read_player_party', return_value=hurt):
            self.assertTrue(self._save(e, 'frontier', frontier_score=50))
        meta = e._read_stage_meta('stage_frontier_2')
        self.assertFalse(meta['party_ready'])
        self.assertTrue(meta['frontier_viable'])
        self.assertIn('party_total_hp_ratio', meta)
        # and it is resolvable as a frontier start
        self.assertEqual(e._v20_stage_checkpoint_name(2, 'frontier'),
                         'stage_frontier_2')

    def test_case3_frontier_checkpoint_rejects_unplayable_party(self):
        e = self._frontier_env()
        dead = [{'checksum_ok': True, 'cur_hp': 0, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]},
                {'checksum_ok': True, 'cur_hp': 2, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]}]
        with patch.object(pokemon_env, 'read_player_party', return_value=dead):
            self.assertFalse(self._save(e, 'frontier', frontier_score=99))

    # ---- fixed per-stage healthy safe fallback ---------------------------

    _HEALTHY = [{'checksum_ok': True, 'cur_hp': 20, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]},
                {'checksum_ok': True, 'cur_hp': 20, 'max_hp': 20, 'status': 0,
                 'moves': [{'pp': 10}]}]
    _HURT = [{'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
              'moves': [{'pp': 10}]},
             {'checksum_ok': True, 'cur_hp': 12, 'max_hp': 20, 'status': 0,
              'moves': [{'pp': 10}]}]

    def _safe_files(self):
        return sorted(p.name for p in self.root.glob('stage_frontier_safe_2*'))

    def _downgrade_scenario(self):
        e = self._frontier_env()
        # 1) healthy anchor
        e.memory = b'healthy-frontier-state'
        with patch.object(pokemon_env, 'read_player_party', return_value=self._HEALTHY):
            self.assertTrue(self._save(e, 'frontier', frontier_score=40))
        self.assertEqual(self._safe_files(), [])       # nothing captured yet
        # 2) same anchor, deeper, HURT party -> triggers the one-time capture
        e.memory = b'hurt-frontier-state'
        with patch.object(pokemon_env, 'read_player_party', return_value=self._HURT):
            self.assertTrue(self._save(e, 'frontier', frontier_score=44))
        return e

    def test_safe_fallback_captured_before_first_weak_overwrite(self):
        e = self._downgrade_scenario()
        self.assertEqual(
            self._safe_files(),
            ['stage_frontier_safe_2.meta.json', 'stage_frontier_safe_2.state.gz'])
        with gzip.open(self.root / 'stage_frontier_safe_2.state.gz', 'rb') as f:
            self.assertEqual(f.read(), b'healthy-frontier-state')
        smeta = json.loads((self.root / 'stage_frontier_safe_2.meta.json').read_text())
        self.assertTrue(smeta['party_ready'])
        self.assertEqual(smeta['frontier_score'], 40.0)
        self.assertEqual(smeta['kind'], 'frontier_safe')
        # main anchor is now the hurt state
        m = e._read_stage_meta('stage_frontier_2')
        self.assertFalse(m['party_ready'])
        self.assertTrue(m['frontier_viable'])

    def test_weak_state_never_overwrites_the_safe_fallback(self):
        e = self._downgrade_scenario()
        e.memory = b'even-weaker-state'
        weaker = [{'checksum_ok': True, 'cur_hp': 10, 'max_hp': 20, 'status': 0,
                   'moves': [{'pp': 10}]},
                  {'checksum_ok': True, 'cur_hp': 10, 'max_hp': 20, 'status': 0,
                   'moves': [{'pp': 10}]}]
        with patch.object(pokemon_env, 'read_player_party', return_value=weaker):
            self.assertTrue(self._save(e, 'frontier', frontier_score=48))   # +>=3
        with gzip.open(self.root / 'stage_frontier_safe_2.state.gz', 'rb') as f:
            self.assertEqual(f.read(), b'healthy-frontier-state')  # unchanged

    def test_exactly_one_safe_fallback_file_pair_per_stage(self):
        e = self._downgrade_scenario()
        for score, mem in ((52, b'w1'), (60, b'w2'), (70, b'w3')):
            e.memory = mem
            with patch.object(pokemon_env, 'read_player_party', return_value=self._HURT):
                self._save(e, 'frontier', frontier_score=score)
        self.assertEqual(len(list(self.root.glob('stage_frontier_safe_2.state.gz'))), 1)
        self.assertEqual(len(list(self.root.glob('stage_frontier_safe_*'))), 2)  # state + meta

    def test_episode_start_routes_a_deterministic_minority_to_the_safe_fallback(self):
        self._downgrade_scenario()   # main anchor hurt, safe fallback present
        # FRONTIER ranks are 41-50 at n_envs=60. rank % 3 == 0 -> safe fallback.
        got = {}
        for rank in range(41, 51):
            got[rank] = self.env(rank)._choose_episode_start()
        safe_ranks = [r for r, v in got.items() if v == 'stage_frontier_safe_2']
        main_ranks = [r for r, v in got.items() if v == 'stage_frontier_2']
        self.assertEqual(safe_ranks, [42, 45, 48])
        self.assertEqual(sorted(main_ranks), [41, 43, 44, 46, 47, 49, 50])
        # deterministic: identical result on a repeat
        self.assertEqual([self.env(r)._choose_episode_start() for r in range(41, 51)],
                         list(got.values()))
        # when the main anchor is HEALTHY again, everyone uses the main anchor
        e = self._frontier_env()
        e.memory = b'healed'
        with patch.object(pokemon_env, 'read_player_party', return_value=self._HEALTHY):
            self.assertTrue(self._save(e, 'frontier', frontier_score=44))
        self.assertTrue(all(self.env(r)._choose_episode_start() == 'stage_frontier_2'
                            for r in range(41, 51)))

    # ---- safe fallback: strict-only + FIGHTER isolation -----------------

    def _put_frontier_meta(self, name, *, party_ready, frontier_viable=True,
                           score=40.0, content=None):
        data = (content or name).encode()
        with gzip.open(self.root / (name + '.state.gz'), 'wb') as f:
            f.write(data)
        meta = dict(state_validation=1, stage=2, bank=3, map=19, x=12, y=30,
                    has_starter=True, frontier_score=score,
                    frontier_metric_version=2,
                    state_sha256=hashlib.sha256(data).hexdigest())
        if party_ready is not None:
            meta['party_ready'] = party_ready
        if frontier_viable is not None:
            meta['frontier_viable'] = frontier_viable
        (self.root / (name + '.meta.json')).write_text(json.dumps(meta))

    def test_safe_fallback_meta_without_party_ready_is_rejected_by_name_resolver(self):
        e = self.env(45)
        for missing in ({}, dict(party_ready=False), dict(party_ready=None),
                        dict(party_ready='yes')):
            (self.root / 'stage_frontier_safe_2.meta.json').unlink(missing_ok=True)
            (self.root / 'stage_frontier_safe_2.state.gz').unlink(missing_ok=True)
            data = b's'
            with gzip.open(self.root / 'stage_frontier_safe_2.state.gz', 'wb') as f:
                f.write(data)
            meta = dict(state_validation=1, stage=2, bank=3, map=19, x=12, y=30,
                        has_starter=True,
                        state_sha256=hashlib.sha256(data).hexdigest(), **missing)
            (self.root / 'stage_frontier_safe_2.meta.json').write_text(json.dumps(meta))
            e.saved_milestones = e._discover_saved_milestones()
            self.assertIsNone(e._v20_stage_checkpoint_name(2, 'frontier_safe'), missing)
        # only a strict party_ready True is accepted
        self._put_frontier_meta('stage_frontier_safe_2', party_ready=True)
        e.saved_milestones = e._discover_saved_milestones()
        self.assertEqual(e._v20_stage_checkpoint_name(2, 'frontier_safe'),
                         'stage_frontier_safe_2')

    def test_viable_but_hurt_safe_state_is_rejected_at_actual_load(self):
        e = self.env(45)
        e.memory = b'initial'
        # a "safe" file that is actually only viable/hurt must not load
        self._put_frontier_meta('stage_frontier_safe_2', party_ready=False,
                                content='hurt-safe')
        loc = dict(trusted=True, map_bank=3, map_id=19, x_pos=12, y_pos=30)
        with patch.object(pokemon_env, 'read_player_location', return_value=loc), \
             patch.object(pokemon_env, 'read_player_party', return_value=self._HURT):
            self.assertFalse(e._load_curriculum_state('stage_frontier_safe_2'))
        self.assertEqual(e.memory, b'initial')
        # the same load succeeds once the restored party is party_ready
        with patch.object(pokemon_env, 'read_player_location', return_value=loc), \
             patch.object(pokemon_env, 'read_player_party', return_value=self._HEALTHY):
            self.assertTrue(e._load_curriculum_state('stage_frontier_safe_2'))
        self.assertEqual(e.memory, b'hurt-safe')

    def test_fighter_with_hurt_main_and_healthy_safe_starts_from_safe(self):
        e = self.env(58)   # FIGHTER rank
        self._put_frontier_meta('stage_frontier_2', party_ready=False)
        self._put_frontier_meta('stage_frontier_safe_2', party_ready=True)
        self.write_checkpoint('stage_2', 2, 19)
        e.saved_milestones = e._discover_saved_milestones()
        self.assertEqual(e._choose_episode_start(), 'stage_frontier_safe_2')

    def test_fighter_with_hurt_main_and_no_safe_starts_from_stage_2_entry(self):
        e = self.env(58)
        self._put_frontier_meta('stage_frontier_2', party_ready=False)
        self.write_checkpoint('stage_2', 2, 19)
        e.saved_milestones = e._discover_saved_milestones()
        start = e._choose_episode_start()
        self.assertEqual(start, 'stage_2')
        self.assertNotEqual(start, 'stage_frontier_2')

    def test_fighter_with_healthy_main_still_starts_from_frontier_anchor(self):
        e = self.env(58)
        self._put_frontier_meta('stage_frontier_2', party_ready=True)
        self.write_checkpoint('stage_2', 2, 19)
        e.saved_milestones = e._discover_saved_milestones()
        self.assertEqual(e._choose_episode_start(), 'stage_frontier_2')


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
        # 2026-09-07: entry-checkpoint capture is FRONTIER-only. The outer `if`
        # now tests `getattr(self, "training_mode", "") == "FRONTIER"`.
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and n.lineno > 5000 and isinstance(n.test, ast.BoolOp)
                    and isinstance(n.test.values[0], ast.Compare)
                    and 'FRONTIER' in ast.unparse(n.test.values[0]))
        code = compile(ast.Module(body=[node], type_ignores=[]), '<capture>', 'exec')
        captures = []
        e = bare_env(has_target_starter=True, player_party_cache=[{'cur_hp': 20}],
                     training_mode='FRONTIER', current_reward=0)
        e._save_stage_checkpoint = lambda *a, **kw: captures.append(a) or False
        e._stage_at_current_location = lambda *a: 1
        e._v20_can_create_stage_checkpoint = lambda *a: True
        scope = dict(self=e, location_refreshed=False,
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

    def test_bridge_and_full_never_capture_stage_checkpoints(self):
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and n.lineno > 5000 and isinstance(n.test, ast.BoolOp)
                    and isinstance(n.test.values[0], ast.Compare)
                    and 'FRONTIER' in ast.unparse(n.test.values[0]))
        code = compile(ast.Module(body=[node], type_ignores=[]), '<capture>', 'exec')
        for mode in ('BRIDGE', 'FULL', 'RETENTION', 'FIGHTER'):
            captures = []
            e = bare_env(has_target_starter=True,
                         player_party_cache=[{'cur_hp': 20}],
                         training_mode=mode, current_reward=0)
            e._save_stage_checkpoint = lambda *a, **kw: captures.append(a) or False
            e._stage_at_current_location = lambda *a: 1
            e._v20_can_create_stage_checkpoint = lambda *a: True
            scope = dict(self=e, location_refreshed=True, in_battle=0,
                         _wipe_cooldown_active=False, loc={'valid': True},
                         bank=3, map_id=0, x=16, y=14, reward=0, milestone_saved=None)
            for _ in range(8):
                exec(code, scope)
            self.assertEqual(captures, [], f"{mode} must not capture checkpoints")

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
