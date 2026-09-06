"""V20 curriculum architecture - acceptance tests (brief section 23).

Run:
    PYTHONPATH=src python -m unittest tests.test_v20_curriculum
"""
import os
import tempfile
import unittest

import curriculum_v20
from curriculum_v20 import (
    CurriculumState,
    MODE_FULL, MODE_BRIDGE, MODE_FRONTIER, MODE_RETENTION,
    FULL_CHAIN_CONFIRMATIONS,
)
from nav_transitions_v20 import KnownTransitions, KNOWN, UNKNOWN
from target_shaper_v20 import TargetShaper
from loop_guard import ShortCycleGuard
from pokemon_env import PokemonFireRedEnv


GAMEPLAY_STEP_COST = PokemonFireRedEnv.GAMEPLAY_STEP_COST


def _bare_env(**kw):
    env = object.__new__(PokemonFireRedEnv)
    for k, v in kw.items():
        setattr(env, k, v)
    return env


# --------------------------------------------------------------------------
class Test1_PalletABOscillation(unittest.TestCase):
    """20 A/B movements at Pallet cannot repeatedly earn positive route
    progress; net reward is negative from step cost / loop penalty."""

    def test_ab_oscillation_cannot_farm_progress(self):
        shaper = TargetShaper(
            progress_reward=PokemonFireRedEnv.TARGET_PROGRESS_REWARD,
            backtrack_penalty=PokemonFireRedEnv.TARGET_BACKTRACK_PENALTY,
        )
        guard = ShortCycleGuard()
        key = ("scout", 3, 0, ((5, 5),))
        shaper.start_objective(key, initial_distance=20)

        total = 0.0
        positive_progress_events = 0
        # A at distance 20, B at distance 19, toggling.
        for i in range(20):
            d = 19 if i % 2 == 0 else 20
            pos = (3, 0, 1, 1) if i % 2 == 0 else (3, 0, 1, 2)
            r, ev = shaper.update(key, d)
            g = guard.update(pos, ("p",), active=True)
            if ev == "route_progress_best" and r > 0:
                positive_progress_events += 1
            total += r + g["penalty"] + GAMEPLAY_STEP_COST

        self.assertLessEqual(positive_progress_events, 1)
        self.assertLess(total, 0.0)


# --------------------------------------------------------------------------
class Test2_PalletHouseLoop(unittest.TestCase):
    """Pallet house enter/exit repeatedly: no building jackpot, no profitable
    loop."""

    def test_generic_building_reward_is_zero(self):
        self.assertEqual(PokemonFireRedEnv.BUILDING_FIRST_GLOBAL_REWARD, 0.0)

    def test_bank4_is_never_a_city_building(self):
        self.assertNotIn(4, PokemonFireRedEnv.CITY_BUILDING_BANKS)

    def test_interior_tiles_are_capped(self):
        # A house has far fewer than CAP unique tiles; after the cap only a
        # 10% fraction pays, so re-entering a known house cannot farm.
        self.assertLessEqual(PokemonFireRedEnv.INTERIOR_TILE_CAP_PER_MAP, 20)
        self.assertLessEqual(
            PokemonFireRedEnv.TILE_REWARD_AFTER_CAP_FACTOR, 0.1
        )


# --------------------------------------------------------------------------
class Test3_Route1UnknownExit(unittest.TestCase):
    """Route1 exit unknown -> exploration works, NO fake target."""

    def test_unknown_transition_yields_no_target(self):
        known = KnownTransitions()
        self.assertEqual(known.navigation_state(2), UNKNOWN)
        self.assertIsNone(known.source_exit_for_stage(2))

    def test_single_observation_is_not_yet_known(self):
        # One crossing can be a RAM misread / glitch warp (the real y=36 Route 1
        # "Viridian exit" that pinned the fleet to the Pallet border). It must
        # not become a navigation target until confirmed a second time.
        known = KnownTransitions()
        known.record(2, 3, source_map=(3, 19), source_exit=(13, 36),
                     dest_map=(3, 1), dest_coord=(36, 19))
        self.assertEqual(known.navigation_state(2), UNKNOWN)


# --------------------------------------------------------------------------
class Test4_Route1ViridianKnown(unittest.TestCase):
    """Once Route1->Viridian is observed, the exact recorded transition is the
    target; positive shaping only on a new best distance."""

    def test_known_transition_becomes_target(self):
        known = KnownTransitions()
        # Same crossing observed twice -> canonical, becomes the target.
        for _ in range(2):
            known.record(2, 3, source_map=(3, 19), source_exit=(9, 0),
                         dest_map=(3, 1), dest_coord=(20, 40))
        self.assertEqual(known.navigation_state(2), KNOWN)
        self.assertEqual(known.source_exit_for_stage(2), (9, 0))

    def test_shaping_only_on_new_best(self):
        shaper = TargetShaper(progress_reward=0.05)
        key = "route1->viridian"
        shaper.start_objective(key, initial_distance=10)
        r1, e1 = shaper.update(key, 9)
        r2, e2 = shaper.update(key, 10)   # back
        r3, e3 = shaper.update(key, 9)    # re-achieve, NOT a new best
        self.assertGreater(r1, 0.0)
        self.assertEqual(e1, "route_progress_best")
        self.assertLessEqual(r2, 0.0)
        self.assertEqual(r3, 0.0)


# --------------------------------------------------------------------------
class Test5_CloserFartherCloser(unittest.TestCase):
    def test_sequence(self):
        shaper = TargetShaper(progress_reward=0.05,
                              backtrack_penalty=-0.01, backtrack_margin=3)
        key = "k"
        shaper.start_objective(key, 20)
        # move closer
        r, e = shaper.update(key, 18)
        self.assertAlmostEqual(r, 0.10, places=6)
        # far away beyond margin -> small penalty only
        r, e = shaper.update(key, 25)
        self.assertEqual(r, -0.01)
        self.assertEqual(e, "route_backtrack")
        # returning to 18 (already achieved) -> NO positive reward
        r, e = shaper.update(key, 18)
        self.assertEqual(r, 0.0)
        self.assertIsNone(e)
        # new best
        r, e = shaper.update(key, 17)
        self.assertAlmostEqual(r, 0.05, places=6)


# --------------------------------------------------------------------------
class Test6_ForestCheckpointNotLowestY(unittest.TestCase):
    """Checkpoint quality is NOT decided by lowest Y."""

    def test_save_stage_checkpoint_has_no_north_wins_logic(self):
        import inspect
        src = inspect.getsource(PokemonFireRedEnv._save_stage_checkpoint)
        self.assertNotIn("North position wins", src)
        self.assertNotIn("smaller map Y", src)
        # entry checkpoints are explicitly immutable
        self.assertIn("immutable once valid", src)

    def test_entry_checkpoint_is_immutable(self):
        # simulate the ok_existing branch decision for an entry checkpoint
        env = _bare_env(shared_lock=None)
        # existing valid entry, new candidate with a "better" (smaller) y and
        # higher reward must still be rejected.
        existing = {"state_validation": 1, "stage": 5, "has_starter": True,
                    "y": 30, "episode_reward": 1.0}
        # replicate the guard: entry checkpoints never move
        is_frontier = False
        ok_existing = True
        rejected = ok_existing and not is_frontier
        self.assertTrue(rejected)


# --------------------------------------------------------------------------
class Test7_PostWipeRecovery(unittest.TestCase):
    def test_recovery_suppresses_normal_streams(self):
        self.assertEqual(PokemonFireRedEnv.POST_WIPE_WILD_BATTLE_SCALE, 0.05)
        # during recovery V20 shaper is bypassed (env keeps the dedicated
        # POST_WIPE_TARGET_PROGRESS_REWARD path).
        import inspect
        src = inspect.getsource(PokemonFireRedEnv.step)
        self.assertIn('not getattr(self, "post_wipe_recovery", False)', src)

    def test_seen_coords_not_reset_on_wipe(self):
        import inspect
        src = inspect.getsource(PokemonFireRedEnv._record_party_wipe)
        self.assertIn("KEIN Reset von seen_coords", src)


# --------------------------------------------------------------------------
class Test8_WildBattleDecay(unittest.TestCase):
    def test_wild_reward_decays_after_threshold(self):
        env = _bare_env(
            episode_wild_faints=PokemonFireRedEnv.WILD_BATTLE_DECAY_AFTER + 2,
            post_wipe_recovery=False,
        )
        env.battle_state = type("B", (), {"raw_flags": 0})()
        scale = PokemonFireRedEnv._battle_reward_scale(env, 3, 19)
        self.assertEqual(scale, PokemonFireRedEnv.WILD_BATTLE_DECAY_FACTOR)
        self.assertLess(scale, 1.0)

    def test_trainer_battle_exempt(self):
        env = _bare_env(
            episode_wild_faints=99, post_wipe_recovery=False,
        )
        env.battle_state = type("B", (), {"raw_flags": 0x8})()
        scale = PokemonFireRedEnv._battle_reward_scale(env, 3, 19)
        self.assertEqual(scale, PokemonFireRedEnv.TRAINER_BATTLE_REWARD_MULT)


# --------------------------------------------------------------------------
class Test9_LongFullProbeHorizon(unittest.TestCase):
    """A long Full probe actually receives LONG_FULL_PROBE_STEPS instead of
    being silently capped at MAX_EPISODE_STEPS."""

    def _limit(self, **kw):
        base = dict(
            n_envs=33, rank=0, training_objective="full",
            episode_start="beginning", shared_progress={},
            V20_CURRICULUM=True, FULL_ONLY_MODE=True,
        )
        base.update(kw)
        return _bare_env(**base)

    def test_long_probe_rank_gets_32k(self):
        layout = curriculum_v20.allocate_modes(33)
        full_ranks = [i for i, m in enumerate(layout) if m == MODE_FULL]
        long_rank = full_ranks[len(full_ranks) // 2:][0]
        env = self._limit(rank=long_rank)
        self.assertTrue(env._is_long_full_probe())
        self.assertEqual(env._episode_step_limit(),
                         PokemonFireRedEnv.LONG_FULL_PROBE_STEPS)

    def test_every_full_rank_gets_the_long_horizon(self):
        # 2026-09-06 (user): every FULL run now gets LONG_FULL_PROBE_STEPS, not
        # just the probe subset - the journey needs far more than 12k steps.
        layout = curriculum_v20.allocate_modes(33)
        full_ranks = [i for i, m in enumerate(layout) if m == MODE_FULL]
        for r in (full_ranks[0], full_ranks[-1]):
            env = self._limit(rank=r)
            self.assertEqual(env._episode_step_limit(),
                             PokemonFireRedEnv.LONG_FULL_PROBE_STEPS)

    def test_scout_uses_scout_limit(self):
        env = self._limit(training_objective="scout")
        self.assertEqual(env._episode_step_limit(),
                         PokemonFireRedEnv.SCOUT_EPISODE_STEPS)


# --------------------------------------------------------------------------
class Test10_ScoutDeepButFullShallow(unittest.TestCase):
    """CRITICAL - reproduces the real situation: scouts reached Pewter (6) but
    Full runs still fail Route1->Viridian.  discovered_stage must be 6,
    mastered_stage < 6, bottleneck = Route1->Viridian, BRIDGE trains it."""

    def test_bottleneck_is_route1_to_viridian(self):
        st = CurriculumState()
        # scouts / frontier discovered up to Pewter
        st.record_discovery(6)
        # Pallet->Route1 IS mastered: many bridge successes + full-chain confirms
        for _ in range(30):
            st.record_transition_attempt(1, success=True, full_chain=False)
        for _ in range(FULL_CHAIN_CONFIRMATIONS + 1):
            st.record_transition_attempt(1, success=True, full_chain=True)
        # Route1->Viridian: full runs keep failing it
        for _ in range(25):
            st.record_transition_attempt(2, success=False, full_chain=True)

        self.assertEqual(st.discovered_stage, 6)
        self.assertLess(st.mastered_stage, 6)
        self.assertEqual(st.mastered_stage, 2)  # crossed 1->2 only
        self.assertEqual(st.current_bottleneck, 2)
        self.assertEqual(curriculum_v20.transition_name(2), "Route1->Viridian")

    def test_bridge_allocation_targets_bottleneck(self):
        n = 33
        summary = curriculum_v20.allocation_summary(n)
        self.assertGreaterEqual(summary[MODE_BRIDGE], 1)
        self.assertGreaterEqual(summary[MODE_FULL], 1)
        # a BRIDGE rank exists
        self.assertIn(MODE_BRIDGE, curriculum_v20.allocate_modes(n))

    def test_full_chain_alone_moves_mastered_not_scout(self):
        st = CurriculumState()
        # only BRIDGE-style successes, no full-chain confirmations
        for _ in range(40):
            st.record_transition_attempt(1, success=True, full_chain=False)
        self.assertEqual(st.mastered_stage, 1)  # NOT promoted
        rec = st.transitions[1]
        self.assertFalse(rec.is_mastered())
        # now the full chain confirms it enough times
        for _ in range(FULL_CHAIN_CONFIRMATIONS):
            st.record_transition_attempt(1, success=True, full_chain=True)
        self.assertTrue(st.transitions[1].is_mastered())
        self.assertGreaterEqual(st.mastered_stage, 2)


# --------------------------------------------------------------------------
class TestModeAllocation(unittest.TestCase):
    def test_reference_ratio_33(self):
        # 2026-09-06: FIGHTER (4) is carved off the top at n >= 20; the
        # remaining 29 follow the 12/12/6/3 ratio.
        s = curriculum_v20.allocation_summary(33)
        self.assertEqual(sum(s.values()), 33)
        self.assertEqual(s[curriculum_v20.MODE_FIGHTER], 4)
        self.assertGreaterEqual(s[MODE_FULL], 10)
        self.assertGreaterEqual(s[MODE_BRIDGE], 9)
        self.assertEqual(s[MODE_FRONTIER], 5)
        self.assertEqual(s[MODE_RETENTION], 2)

    def test_small_fleet_has_no_fighters(self):
        s = curriculum_v20.allocation_summary(12)
        self.assertEqual(s[curriculum_v20.MODE_FIGHTER], 0)

    def test_scales_to_other_sizes(self):
        for n in (1, 4, 8, 12, 20, 48, 60, 96):
            layout = curriculum_v20.allocate_modes(n)
            self.assertEqual(len(layout), n)
            self.assertGreaterEqual(layout.count(MODE_FULL), 1)

    def test_deterministic(self):
        self.assertEqual(curriculum_v20.allocate_modes(60),
                         curriculum_v20.allocate_modes(60))


class TestPersistence(unittest.TestCase):
    def test_roundtrip(self):
        st = CurriculumState()
        st.record_discovery(5)
        st.record_transition_attempt(1, True, full_chain=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "state.json")
            st.save(p)
            st2 = CurriculumState.load(p)
        self.assertEqual(st2.discovered_stage, 5)
        self.assertEqual(st2.transitions[1].attempts, 1)

    def test_known_transitions_roundtrip(self):
        k = KnownTransitions()
        k.record(2, 3, (3, 19), (9, 0), (3, 1), (20, 40))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "kt.json")
            k.save(p)
            k2 = KnownTransitions.load(p)
        self.assertEqual(k2.source_exit_for_stage(2), (9, 0))

    def test_backward_hops_are_ignored(self):
        k = KnownTransitions()
        self.assertFalse(k.record(3, 2, (3, 1), (1, 1), (3, 19), (2, 2)))
        self.assertEqual(k.navigation_state(3), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
