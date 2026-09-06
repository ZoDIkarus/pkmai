import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from test_progress_curriculum import bare_env
from pokemon_env import PokemonFireRedEnv
from battle_state import MainBattleReader, BattleState
from loop_guard import LocalLoopGuard
from reward_state import claim_event


class ProgressRegressionTests(unittest.TestCase):
    def test_each_approved_stage_has_slots_scouts_and_pallet_has_none(self):
        slots=PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS
        assigned=[]
        for rank in range(50):
            e=bare_env(rank=rank,n_envs=50)
            e._valid_stage_checkpoints=lambda: {s:f'stage_{s}' for s in range(1,7)}
            assigned.append(e._scout_assigned_stage())
        for s in (2,3,4,5,6): self.assertEqual(assigned.count(s),slots)
        self.assertNotIn(1,assigned)
        self.assertEqual(assigned.count(None),50-5*slots)

    def test_late_checkpoint_never_reassigns_existing_scouts(self):
        slots=PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS
        def assignment(stages):
            result=[]
            for rank in range(50):
                e=bare_env(rank=rank,n_envs=50)
                e._valid_stage_checkpoints=lambda: {s:f'stage_{s}' for s in stages}
                result.append(e._scout_assigned_stage())
            return result
        before=assignment((2,3,5,6))
        after=assignment((2,3,4,5,6))
        for rank,stage in enumerate(before):
            if stage is not None:self.assertEqual(after[rank],stage)
        for stage in (2,3,4,5,6):self.assertEqual(after.count(stage),slots)

    def test_three_reached_maps_allocate_slots_scouts_each(self):
        from collections import Counter
        slots=PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS
        counts=Counter()
        for rank in range(60):
            e=bare_env(rank=rank,n_envs=60)
            e._valid_stage_checkpoints=lambda:{s:f'stage_{s}' for s in (2,3,4)}
            counts[e._scout_assigned_stage()]+=1
        self.assertEqual(counts,Counter({None:60-3*slots,2:slots,3:slots,4:slots}))

    def test_checkpoint_prefers_north_then_reward_even_after_forest(self):
        with tempfile.TemporaryDirectory() as d:
            e=bare_env(rank=0,route_steps=100,shared_lock=None,
                       rank_state_dir=d,saved_milestones=[],visited_maps={(1,0)})
            state=[b'first']
            e.env=SimpleNamespace(em=SimpleNamespace(get_state=lambda:state[0]))
            e._starter_species=lambda:e.TARGET_STARTER_SPECIES
            with patch('pokemon_env.SHARED_CURRICULUM_DIR',d), patch(
                'pokemon_env.read_player_location', return_value={
                    'trusted':True,'map_bank':3,'map_id':19,'x_pos':4,'y_pos':30}
            ) as location:
                self.assertTrue(e._save_stage_checkpoint(2,3,19,4,30,100))
                def save(stage, bank, map_id, x, y, reward):
                    location.return_value = dict(trusted=True,map_bank=bank,map_id=map_id,x_pos=x,y_pos=y)
                    return e._save_stage_checkpoint(stage,bank,map_id,x,y,reward)
                # More points cannot pull the checkpoint south.
                self.assertFalse(save(2,3,19,3,33,9999))
                self.assertFalse(save(2,3,19,3,30,100))
                self.assertFalse(save(2,3,19,3,30,99))
                state[0]=b'better_score'
                self.assertTrue(save(2,3,19,3,30,150))
                state[0]=b'further_north'
                self.assertTrue(save(2,3,19,3,29,50))
                self.assertFalse(save(2,3,19,3,30,9999))
                self.assertFalse(save(5,4,3,6,4,9999))
                with gzip.open(Path(d)/'stage_2.state.gz','rb') as f:
                    self.assertEqual(f.read(),b'further_north')
                meta=e._read_stage_meta('stage_2')
                self.assertEqual((meta['y'],meta['episode_reward']),(29,50))

    def test_stage_load_rejects_wrong_map_and_restores_master(self):
        import json, hashlib
        with tempfile.TemporaryDirectory() as d:
            e=bare_env(shared_lock=None,rank_state_dir=d)
            current=[b'master']
            e.env=SimpleNamespace(em=SimpleNamespace(
                get_state=lambda:current[0],set_state=lambda value:current.__setitem__(0,value)))
            state=b'route1'
            with gzip.open(Path(d)/'stage_3.state.gz','wb') as f:f.write(state)
            (Path(d)/'stage_3.meta.json').write_text(json.dumps(dict(
                state_validation=1,state_sha256=hashlib.sha256(state).hexdigest(),
                stage=3,bank=3,map=1,x=36,y=19)))
            with patch('pokemon_env.SHARED_CURRICULUM_DIR',d), patch(
                'pokemon_env.read_player_location',return_value=dict(
                    trusted=True,map_bank=3,map_id=19,x_pos=2,y_pos=34)):
                self.assertFalse(e._load_curriculum_state('stage_3'))
                self.assertEqual(current[0],b'master')
                self.assertFalse(e._save_stage_checkpoint(3,3,1,36,19,9999))

    def test_stage_unlock_pays_only_after_durable_first_claim(self):
        self.assertEqual(PokemonFireRedEnv.NEW_GLOBAL_DEPTH_REWARD,0)
        self.assertEqual(PokemonFireRedEnv.GLOBAL_STAGE_RECORD_REWARD,1000)
        with tempfile.TemporaryDirectory() as d:
            e=bare_env(shared_progress={},shared_lock=None)
            with patch('pokemon_env.GLOBAL_PROGRESS_FILE',str(Path(d)/'progress.json')):
                self.assertTrue(e._claim_global_depth(2))
                self.assertFalse(e._claim_global_depth(2))
                e.shared_progress={}
                self.assertFalse(e._claim_global_depth(2))
                self.assertTrue(e._claim_global_depth(3))
            with patch('pokemon_env.GLOBAL_PROGRESS_FILE',str(Path(d)/'missing'/'progress.json')):
                self.assertFalse(e._claim_global_depth(4))

    def test_snapshot_publication_keeps_previous_file_on_failed_save(self):
        from train import save_model_atomic
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'resume.zip';path.write_bytes(b'old')
            def broken(target):
                Path(target).write_bytes(b'incomplete')
                raise OSError('save failed')
            with self.assertRaises(OSError):
                save_model_atomic(SimpleNamespace(save=broken),str(path))
            self.assertEqual(path.read_bytes(),b'old')
            save_model_atomic(SimpleNamespace(save=lambda target:Path(target).write_bytes(b'new')),str(path))
            self.assertEqual(path.read_bytes(),b'new')

    def test_wipe_is_charged_once_until_recovery(self):
        e=bare_env(wipe_active=False,run_stats={'party_wipes':0},route_steps=10)
        e._save_run_stats=lambda:None
        events=[];info={}
        self.assertEqual(e._record_party_wipe(events,info),-100)
        e.route_steps=1000
        self.assertEqual(e._record_party_wipe(events,info),0)
        self.assertEqual(e.run_stats['party_wipes'],1)

    def test_global_heal_claim_survives_new_registry(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(claim_event(d,'pokemon_center_ever',{}))
            self.assertFalse(claim_event(d,'pokemon_center_ever',{}))

    def test_per_center_first_heal_claims_are_independent_and_permanent(self):
        # V18: each Pokemon Center pays its +1000 exactly once fleet-wide,
        # keyed per interior map, and the claim survives a trainer restart
        # (fresh in-memory registry) because it is persisted to disk.
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(claim_event(d,'pc_heal_5_4',{}))
            self.assertFalse(claim_event(d,'pc_heal_5_4',{}))
            self.assertTrue(claim_event(d,'pc_heal_6_5',{}))
            self.assertFalse(claim_event(d,'pc_heal_5_4',{}))

    def test_warp_pair_key_is_coarse_and_seeded_pairs_count_as_known(self):
        # V18: der Global-Warp-Bonus laeuft ueber ein grobes Kartenpaar, und
        # jedes Paar, das die Navigations-Historie schon kennt, zaehlt sofort
        # als bekannt - kein new_warp_global beim x-ten Rein/Raus aus Alabastia.
        k1 = PokemonFireRedEnv._warp_pair_key(3, 0, 3, 19)
        k2 = PokemonFireRedEnv._warp_pair_key(3, 19, 3, 0)   # umgekehrte Richtung
        self.assertEqual(k1, k2)
        pairs = PokemonFireRedEnv._derive_warp_pairs([
            (3, 0, 5, 6, 3, 19, 7, 41),      # Alabastia-Rand <-> Route 1 an x5
            (3, 0, 6, 6, 3, 19, 8, 41),      # dieselbe Grenze an x6
            (4, 3, 2, 4, 3, 0, 12, 9),       # Eichs Labor <-> Alabastia
        ])
        self.assertIn(k1, pairs)                       # beide Route-1-Kanten -> ein Paar
        self.assertEqual(len(pairs), 2)
        self.assertIn(PokemonFireRedEnv._warp_pair_key(4, 3, 3, 0), pairs)

    def test_wild_battle_reward_decays_only_after_the_cap_and_only_on_wild_maps(self):
        e = bare_env()
        wild = next(iter(PokemonFireRedEnv.WILD_TRAINING_MAPS))
        e.episode_wild_faints = 0
        self.assertEqual(e._wild_battle_scale(*wild), 1.0)
        e.episode_wild_faints = PokemonFireRedEnv.WILD_BATTLE_DECAY_AFTER - 1
        self.assertEqual(e._wild_battle_scale(*wild), 1.0)
        e.episode_wild_faints = PokemonFireRedEnv.WILD_BATTLE_DECAY_AFTER
        self.assertEqual(e._wild_battle_scale(*wild),
                         PokemonFireRedEnv.WILD_BATTLE_DECAY_FACTOR)
        # A non-wild map (e.g. a city / trainer fight) never decays.
        self.assertEqual(e._wild_battle_scale(3, 1), 1.0)

    def test_v19_brock_rush_milestones_and_flags(self):
        self.assertEqual(PokemonFireRedEnv.BUILD_TAG, "V19_BROCK_RUSH")
        self.assertEqual(PokemonFireRedEnv.STAGE_ADVANCE_REWARD, 250.0)
        self.assertEqual(PokemonFireRedEnv.TARGET_PROGRESS_REWARD, 0.20)
        self.assertEqual(PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS, 3)
        # neue Meilenstein-Konstanten
        self.assertEqual(PokemonFireRedEnv.PEWTER_WITH_PIKACHU_REWARD, 300.0)
        self.assertEqual(PokemonFireRedEnv.PEWTER_GYM_ENTER_REWARD, 200.0)
        self.assertEqual(PokemonFireRedEnv.BROCK_BATTLE_START_REWARD, 500.0)
        self.assertEqual(PokemonFireRedEnv.PEWTER_GYM_TRAINER_REWARD, 300.0)
        # jeder Meilenstein hat ein Episode-Flag, in __init__ UND reset() auf
        # False gesetzt (Anti-Farm: einmal pro Episode).
        import inspect
        src = inspect.getsource(PokemonFireRedEnv)
        for f in ("episode_pewter_reached", "episode_pewter_with_pikachu_rewarded",
                  "episode_pewter_gym_entered", "episode_brock_battle_started",
                  "episode_pewter_gym_trainer_beaten"):
            self.assertGreaterEqual(src.count(f"self.{f} = False"), 2, f)
        # V19: alte Edge-/Warp-Farm-Rewards bleiben aus
        for c in ("NEW_EDGE_REWARD", "EPISODE_EDGE_REWARD", "REPLAY_EDGE_REWARD",
                  "REPLAY_TRANSITION_REWARD", "EPISODE_TRANSITION_REWARD",
                  "CORRIDOR_STEP_REWARD", "NORTH_CORRIDOR_ROW_REWARD"):
            self.assertEqual(getattr(PokemonFireRedEnv, c), 0.0, c)

    def test_v19_forward_targets_point_only_forward(self):
        e = bare_env(best_pokecenter_heal_stage=3, episode_brock_battle_started=False)
        # Route 1 (stage 2) kennt eine Transition nach Viridian (3) und eine
        # zurueck nach Pallet (1).
        e._combined_transitions = lambda: [
            (3, 19, 4, 0, 3, 1, 20, 39),    # Route 1 -> Viridian (vorwaerts)
            (3, 19, 4, 39, 3, 0, 5, 0),     # Route 1 -> Pallet (rueckwaerts)
        ]
        t = e._v19_forward_targets(3, 19)
        self.assertIn((4, 0), t)            # nur der Vorwaerts-Ausgang
        self.assertNotIn((4, 39), t)        # NICHT der Rueckweg
        # Auf der tiefsten bekannten Karte ohne Vorwaerts-Transition: leer.
        e._combined_transitions = lambda: [(3, 2, 1, 1, 3, 1, 2, 2)]  # Pewter->Viridian (rueck)
        self.assertEqual(e._v19_forward_targets(3, 2), [])

    def test_pallet_nav_target_is_only_the_route1_transition(self):
        e = bare_env()
        # gemischte Transitionen: echter Alabastia<->Route1-Uebergang (mehrfach
        # gestuetzt), ein Glitch-Singleton mitten in Alabastia, und Alabastia
        # <->Haus (Bank 4) - darf NICHT als Ziel zaehlen.
        e._combined_transitions = lambda: [
            (3, 0, 13, 0, 3, 19, 13, 39),
            (3, 0, 13, 1, 3, 19, 13, 39),
            (3, 0, 12, 0, 3, 19, 12, 39),
            (3, 19, 13, 38, 3, 0, 13, 0),     # umgekehrt, stuetzt (13,0)
            (3, 0, 6, 8, 3, 19, 8, 33),       # Glitch-Singleton
            (3, 0, 16, 13, 4, 3, 6, 7),       # Alabastia <-> Eichs Labor
        ]
        t = e._pallet_route1_target()
        self.assertIn((13, 0), t)             # gut gestuetzt
        self.assertNotIn((6, 8), t)           # Glitch raus
        self.assertNotIn((16, 13), t)         # Hausausgang ist kein Ziel
        for c in t:
            self.assertLess(c[1], 8)          # nur die echte Nordkante
        # ohne bekannten Uebergang: gar kein Ziel (kein Frontier-Fallback)
        e._combined_transitions = lambda: []
        self.assertEqual(e._pallet_route1_target(), [])
        # immediate-reverse-Guard existiert im step()-Ziel-Block
        import inspect
        src = inspect.getsource(PokemonFireRedEnv.step)
        self.assertIn("_immediate_reverse", src)
        self.assertIn("target_closer_suppressed", src)
        self.assertIn("target_farther", src)   # Strafe NICHT entfernt

    def test_bank4_interiors_never_get_the_500_city_building_reward(self):
        import inspect
        src = inspect.getsource(PokemonFireRedEnv.step)
        # Der Stadt-Gebaeude-Bonus haengt an CITY_BUILDING_BANKS UND explizit
        # `!= 4`, und ein building_<b>_<m>-claim_event entsteht nur fuer
        # Stadt-Gebaeude.
        i = src.index("_is_city_building")
        window = src[i:i + 500]
        self.assertIn("int(bank) in self.CITY_BUILDING_BANKS", window)
        self.assertIn("int(bank) != 4", window)
        self.assertNotIn(4, PokemonFireRedEnv.CITY_BUILDING_BANKS)
        self.assertEqual(PokemonFireRedEnv.CITY_BUILDING_BANKS, {5, 6})
        self.assertGreater(PokemonFireRedEnv.BUILDING_FIRST_GLOBAL_REWARD,
                           PokemonFireRedEnv.EPISODE_NEW_MAP_REWARD)
        # Das +500-Event (`new_building_global`) und der `building_`-Key haengen
        # beide am selben `_is_city_building`-Zweig, der 500-Wert im if-Body.
        self.assertLess(src.index("_is_city_building"),
                        src.index('"new_building_global"'))
        self.assertLess(src.index('"new_building_global"'),
                        src.index('"new_building_seen"'))

    def test_post_wipe_recovery_mode(self):
        self.assertEqual(PokemonFireRedEnv.POST_WIPE_WILD_BATTLE_SCALE, 0.05)
        self.assertEqual(PokemonFireRedEnv.POST_WIPE_TARGET_PROGRESS_REWARD, 0.50)
        self.assertEqual(PokemonFireRedEnv.POST_WIPE_FRONT_RECOVERED_REWARD, 300.0)
        # Der Wipe setzt den Recovery-Modus + merkt sich die Front, ohne
        # Novelty-Memory anzufassen.
        import inspect
        src = inspect.getsource(PokemonFireRedEnv._record_party_wipe)
        self.assertIn("self.post_wipe_recovery = True", src)
        self.assertIn("self.pre_wipe_best_stage", src)
        for forbidden in ("seen_coords = set()", "seen_coords.clear",
                          "visited_maps = set()", "visited_maps.clear",
                          "seen_coords -=", "visited_maps -="):
            self.assertNotIn(forbidden, src)
        self.assertIn("party_wiped:-100.0", src)  # -100-Strafe erhalten
        # Wild-Skala waehrend Recovery, Trainerkaempfe NICHT reduziert.
        e = bare_env(episode_wild_faints=0, post_wipe_recovery=True)
        e.battle_state = SimpleNamespace(raw_flags=0)          # Wildkampf
        wild = next(iter(PokemonFireRedEnv.WILD_TRAINING_MAPS))
        self.assertAlmostEqual(e._battle_reward_scale(*wild),
                               PokemonFireRedEnv.POST_WIPE_WILD_BATTLE_SCALE)
        e.battle_state = SimpleNamespace(raw_flags=0x8)        # Trainerkampf
        self.assertEqual(e._battle_reward_scale(*wild),
                         PokemonFireRedEnv.TRAINER_BATTLE_REWARD_MULT)

    def test_battle_rewards_only_continuous_signals(self):
        # V18: im Kampf zaehlen nur Schaden/Heilung/Level/Fangen - kein
        # pauschaler KO- oder Sieg-Bonus mehr.
        self.assertEqual(PokemonFireRedEnv.ENEMY_FAINT_REWARD, 0.0)
        self.assertEqual(PokemonFireRedEnv.BATTLE_WIN_REWARD, 0.0)
        self.assertGreater(PokemonFireRedEnv.ENEMY_DAMAGE_REWARD_PER_HP, 0.0)
        self.assertGreater(PokemonFireRedEnv.LEVEL_GAIN_REWARD, 0.0)
        self.assertGreater(PokemonFireRedEnv.SPECIES_CAUGHT_FIRST_REWARD, 0.0)

    def test_advancing_pokecenter_heal_is_a_serious_progress_anchor(self):
        # V19: der tiefere Center verschiebt den Wipe-Respawn dauerhaft nach
        # vorn - das ist der wichtigste Anti-Rueckschlag-Anker und darf einen
        # neuen Fang klar schlagen (Fangen ist bis Brock bewusst klein).
        cap = PokemonFireRedEnv.SPECIES_CAUGHT_LEVEL_BONUS_CAP
        best_catch = (PokemonFireRedEnv.SPECIES_CAUGHT_FIRST_REWARD
                      + cap * PokemonFireRedEnv.SPECIES_CAUGHT_LEVEL_BONUS)
        self.assertLess(best_catch, PokemonFireRedEnv.POKECENTER_ADVANCE_HEAL_REWARD)
        self.assertGreaterEqual(PokemonFireRedEnv.POKECENTER_ADVANCE_HEAL_REWARD,
                                PokemonFireRedEnv.CITY_EPISODE_REWARD)
        for m in PokemonFireRedEnv.POKECENTER_HEAL_MAPS:
            self.assertIn(m, PokemonFireRedEnv.POKECENTER_MAPS)

    def test_local_wandering_is_bounded_but_battle_and_progress_are_allowed(self):
        g=LocalLoopGuard(window=10,max_tiles=2)
        for n in range(9):self.assertFalse(g.update((n%2,0),0))
        self.assertTrue(g.update((1,0),0))
        self.assertFalse(g.update((1,0),1))
        for _ in range(20):self.assertFalse(g.update((1,0),1,True))

    def test_wider_door_loop_cannot_evade_idle_limit(self):
        g=LocalLoopGuard(window=10,max_tiles=2,max_idle_steps=20)
        for n in range(19):
            self.assertFalse(g.update((n%9,0),0))
        self.assertTrue(g.update((1,0),0))
        self.assertFalse(g.update((1,0),1))
        for _ in range(30):self.assertFalse(g.update((1,0),1,True))
        self.assertEqual(g.idle_steps,1)

    def test_location_discovery_ignores_ewram_pointer_copies(self):
        import firered_ram as fr
        ram=bytearray(0x48000)
        for slot,base in ((0x100,0x1000),(0x44f58,0x2000)):
            for i,pointer in enumerate((base,base+0x100,base+0x200)):
                ram[slot+i*4:slot+i*4+4]=(0x02000000+pointer).to_bytes(4,'little')
            ram[base:base+6]=bytes((2,0,3,0,3,19))
        self.assertEqual(fr._find_pointer_slot(ram),0x44f58)
        ram[0x2004:0x2006]=bytes((0,57))
        self.assertFalse(fr._valid_location(ram,0x2000))
        fake=SimpleNamespace(get_ram=lambda:ram)
        with patch.object(fr,'_cached_ptr_slot',0x44f58):
            self.assertFalse(fr.read_player_location(fake)['valid'])
            self.assertEqual(fr._cached_ptr_slot,0x44f58)
            ram[0x2004:0x2006]=bytes((3,19))
            self.assertTrue(fr.read_player_location(fake,allow_scan=False)['valid'])

    def test_live_battle_bit_clears_while_stationary_despite_stale_signal(self):
        ram=bytearray(0x48000);base=0x43040
        for offset in (4,12):ram[base+offset:base+offset+4]=(0x08001001).to_bytes(4,'little')
        r=MainBattleReader()
        for counter in (0,14,28):
            ram[base+0x24:base+0x28]=counter.to_bytes(4,'little')
            result=r.read(ram)
        self.assertIs(result,False)
        ram[base+0x439]=2
        self.assertIs(r.read(ram),True)
        b=BattleState();self.assertEqual(b.update([], (3,19,2,32),signal=True,live=True),1)
        ram[base+0x439]=0
        self.assertEqual(b.update([], (3,19,2,32),signal=True,live=r.read(ram)),0)
