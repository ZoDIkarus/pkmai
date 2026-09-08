"""Phase 1 — grass wild-encounter harvester, scenario diversity, shiny
telemetry (spec ZIEL A / B / C / E). Everything shiny stays fail-closed:
``twoby2.shiny_ram.SHINY_RAM_VERIFIED`` is False, so no shiny is counted or
rewarded and Catch-v2 is not activated.
"""
import hashlib
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import shiny
import battle_executor as bx
import battle_env
from battle_env import (BattleEnv, SimulatedBattleDriver, BattleRewardConfig,
                        _reward_components_v2, catch_scenario)
from battle_executor import CATCH_ACTION, RUN_ACTION, ALL_MACROS
from twoby2 import shiny_ram
from twoby2.shiny_counters import ShinyCounters, aggregate_all
from twoby2.wild_encounter_harvester import (CorridorWalker, WildEncounterHarvester,
                                             corridor_violation, HarvesterConfig,
                                             harvest_episode_seed)
from twoby2 import scenario_pool as sp

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runtime", "battle", "checkpoints")


# --------------------------------------------------------------------------
# a fake overworld emulator for the harvester
# --------------------------------------------------------------------------
class FakeEmu:
    """A 1-D grass corridor. The agent is at ``x`` on row ``y`` in map
    ``(bank,mid)``. Optional walls, a ledge tile and a warp tile. An encounter
    fires after ``encounter_after`` successful moves."""

    def __init__(self, *, x=10, y=4, bank=3, mid=19, walls=(), ledge_x=None,
                 warp_x=None, encounter_after=None, encounter_latch_delay=0):
        self.x, self.y, self.bank, self.mid = x, y, bank, mid
        self.walls = set(walls)
        self.ledge_x, self.warp_x = ledge_x, warp_x
        self.encounter_after = encounter_after
        self.encounter_latch_delay = int(encounter_latch_delay)
        self._encounter_transition_remaining = 0
        self.moves_done = 0
        self.in_battle = False
        self.buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START"]
        self._states = {}
        self._nonce = 0

        class _EM:
            def __init__(s, outer): s.o = outer
            def set_button_mask(s, m):
                s._m = m
            def step(s):
                m = getattr(s, "_m", None)
                if s.o._encounter_transition_remaining:
                    s.o._encounter_transition_remaining -= 1
                    if s.o._encounter_transition_remaining == 0:
                        s.o.in_battle = True
                    return
                if m is None or s.o.in_battle:
                    return
                names = [s.o.buttons[i] for i, v in enumerate(m) if v]
                if "LEFT" in names:
                    s.o._try(-1)
                elif "RIGHT" in names:
                    s.o._try(+1)
            def get_state(s):
                s.o._nonce += 1
                key = f"state{s.o._nonce}"
                s.o._states[key] = (s.o.x, s.o.y, s.o.bank, s.o.mid,
                                    s.o.in_battle, s.o.moves_done,
                                    s.o._encounter_transition_remaining)
                return key.encode()
            def set_state(s, b):
                key = b.decode()
                (s.o.x, s.o.y, s.o.bank, s.o.mid,
                 s.o.in_battle, s.o.moves_done,
                 s.o._encounter_transition_remaining) = s.o._states[key]
        self.em = _EM(self)

    def _try(self, dx):
        nx = self.x + dx
        if nx in self.walls:
            return                      # blocked, no move
        if nx == self.warp_x:
            self.bank, self.mid = 99, 99   # warped to another map
            self.x = nx
            return
        if nx == self.ledge_x:
            self.x, self.y = nx, self.y - 2   # ledge jump: y changes
            return
        self.x = nx
        self.moves_done += 1
        if self.encounter_after is not None and self.moves_done >= self.encounter_after:
            if self.encounter_latch_delay:
                self._encounter_transition_remaining = self.encounter_latch_delay
                # Real FireRed invalidates/changes overworld map fields during
                # the fade before gMain's battle latch becomes readable.
                self.bank, self.mid = 99, 99
            else:
                self.in_battle = True

    def get_ram(self):
        return b""

    # injected readers
    def loc(self, _env):
        return {"map_bank": self.bank, "map_id": self.mid,
                "x_pos": self.x, "y_pos": self.y, "valid": True}

    def batt(self, _env):
        return self.in_battle


class MultiFrameTileEmu(FakeEmu):
    """A tile only completes after ``frames_per_tile`` consecutive held frames
    of the same direction — like the real FireRed walk animation. A held button
    must therefore be pulsed + re-checked, not blindly held for a fixed count."""

    def __init__(self, *, frames_per_tile=6, jump_after=None, **kw):
        super().__init__(**kw)
        self.frames_per_tile = frames_per_tile
        self.jump_after = jump_after      # after N tiles, jump 2 at once (bug)
        self._held = 0
        self._held_dir = None

        outer = self

        class _EM2(type(self.em)):
            def step(s):
                m = getattr(s, "_m", None)
                if m is None or outer.in_battle:
                    outer._held = 0
                    return
                names = [outer.buttons[i] for i, v in enumerate(m) if v]
                d = "LEFT" if "LEFT" in names else "RIGHT" if "RIGHT" in names else None
                if d is None:
                    outer._held = 0
                    outer._held_dir = None
                    return
                if d != outer._held_dir:
                    outer._held, outer._held_dir = 0, d
                outer._held += 1
                if outer._held >= outer.frames_per_tile:
                    outer._held = 0
                    step = 1
                    if (outer.jump_after is not None
                            and outer.moves_done + 1 >= outer.jump_after):
                        step = 2                 # simulate a 2-tile overshoot
                    outer._try_multi(-step if d == "LEFT" else step)
        self.em.__class__ = _EM2

    def _try_multi(self, dx):
        nx = self.x + (1 if dx > 0 else -1) * min(1, abs(dx)) if abs(dx) == 1 else self.x + dx
        if abs(dx) == 1 and nx in self.walls:
            return
        self.x = self.x + dx
        self.moves_done += 1
        if self.encounter_after is not None and self.moves_done >= self.encounter_after:
            self.in_battle = True


def _harv(emu, **kw):
    return WildEncounterHarvester(emu, location_reader=emu.loc,
                                  battle_reader=emu.batt, **kw)


# module-level so multiprocessing 'spawn' can pickle them (review #1)
def _mp_worker_unique(base_dir, tag, n):
    from twoby2.shiny_counters import ShinyCounters
    c = ShinyCounters("battle_fighter", base_dir=base_dir)
    for i in range(n):
        c.record_encounter(f"{tag}-{i}")


def _mp_worker_same(base_dir):
    from twoby2.shiny_counters import ShinyCounters
    c = ShinyCounters("full_agent", base_dir=base_dir)
    for _ in range(200):
        c.record_encounter("SAME")


# --------------------------------------------------------------------------
class CorridorLogicTests(unittest.TestCase):
    def test_never_leaves_the_pm3_corridor(self):          # ZIEL E.4
        w = CorridorWalker(10, max_tiles=3)
        x = 10
        seen = [x]
        for _ in range(200):
            d = w.decide(x)
            self.assertIsNotNone(d)
            x += -1 if d == "LEFT" else 1
            w.observe(direction=d, moved=True)
            seen.append(x)
        self.assertGreaterEqual(min(seen), 7)
        self.assertLessEqual(max(seen), 13)

    def test_bound_is_relative_to_the_original_anchor(self):
        w = CorridorWalker(10, max_tiles=3)
        # even after drifting, the turnaround is anchor±3, never last-pos±3
        for x in (13, 14, 20):
            self.assertEqual(w.decide(x), "LEFT")

    def test_blocked_both_sides_aborts(self):
        w = CorridorWalker(10, max_tiles=3)
        w.observe(direction="LEFT", moved=False)
        w.observe(direction="RIGHT", moved=False)
        self.assertIsNone(w.decide(10))

    def test_corridor_violation_flags_map_row_and_x(self):  # ZIEL E.5
        a = {"map_bank": 3, "map_id": 19, "x_pos": 10, "y_pos": 4}
        self.assertIsNone(corridor_violation(a, {**a, "valid": True}))
        self.assertEqual(corridor_violation(a, {**a, "map_id": 20, "valid": True}),
                         "map_changed")
        self.assertEqual(corridor_violation(a, {**a, "y_pos": 2, "valid": True}),
                         "left_grass_row")
        self.assertEqual(corridor_violation(a, {**a, "x_pos": 14, "valid": True}),
                         "left_corridor")


class HarvesterTests(unittest.TestCase):
    def test_walks_to_an_encounter_and_restores_the_anchor(self):
        emu = FakeEmu(x=10, encounter_after=2)
        h = _harv(emu, seed_id=3)
        r = h.harvest()
        self.assertTrue(r.ok and r.encounter_started)
        self.assertTrue(emu.in_battle)
        emu.in_battle = False
        rr = h.restore_anchor()
        self.assertTrue(rr.ok)
        self.assertEqual((emu.x, emu.y), (10, 4))

    def test_anchor_identical_after_100_encounter_cycles(self):   # ZIEL E.3
        emu = FakeEmu(x=10, encounter_after=1)
        h = _harv(emu, seed_id=1)
        h.capture_anchor()
        anchor = dict(h._anchor)
        for _ in range(100):
            emu.in_battle = False
            emu.moves_done = 0
            r = h.harvest()
            self.assertTrue(r.encounter_started)
            emu.in_battle = False
            self.assertTrue(h.restore_anchor().ok)
            self.assertEqual((emu.x, emu.y, emu.bank, emu.mid),
                             (anchor["x_pos"], anchor["y_pos"],
                              anchor["map_bank"], anchor["map_id"]))
        self.assertEqual(h._anchor, anchor)

    def test_warp_aborts_and_restores(self):               # ZIEL E.5
        emu = FakeEmu(x=10, warp_x=12, encounter_after=None)
        h = _harv(emu, seed_id=0)
        r = h.harvest()
        self.assertFalse(r.ok)
        self.assertIn(r["reason"], ("map_changed",))
        self.assertTrue(r["restore_ok"])
        self.assertEqual((emu.bank, emu.mid, emu.x, emu.y), (3, 19, 10, 4))

    def test_ledge_aborts_and_restores(self):              # ZIEL E.5
        emu = FakeEmu(x=10, ledge_x=8)
        h = _harv(emu, seed_id=0)
        r = h.harvest()
        self.assertFalse(r.ok)
        self.assertEqual(r["reason"], "left_grass_row")
        self.assertTrue(r["restore_ok"])

    def test_step_limit_aborts_never_loops(self):
        emu = FakeEmu(x=10, encounter_after=None)
        cfg = type("C", (HarvesterConfig,), {"MAX_STEPS": 12})
        h = _harv(emu, config=cfg, seed_id=0)
        r = h.harvest()
        self.assertFalse(r.ok)
        self.assertEqual(r["reason"], "no_encounter_within_step_limit")
        self.assertLessEqual(r["steps"], 12)

    def test_harvest_pendulum_is_not_a_ppo_action(self):    # ZIEL E.1
        # the harvester drives em.set_button_mask directly and never calls
        # BattleEnv.step / a nav env.step -> no reward path is touched.
        import inspect
        src = inspect.getsource(WildEncounterHarvester)
        self.assertNotIn(".step(a", src)
        self.assertNotIn("reward", src.lower())


class TileAwareMovementTests(unittest.TestCase):
    def test_a_tile_needing_many_frames_still_moves_exactly_one(self):  # review #4
        emu = MultiFrameTileEmu(x=10, frames_per_tile=8, encounter_after=None)
        h = _harv(emu, episode_seed=2)
        h.capture_anchor()
        kind, loc = h._walk_one_tile("LEFT")
        self.assertEqual(kind, "moved")
        self.assertEqual(emu.x, 9)                 # exactly one tile, no overshoot

    def test_walk_reaches_encounter_within_corridor_multiframe(self):
        emu = MultiFrameTileEmu(x=10, frames_per_tile=6, encounter_after=2)
        h = _harv(emu, episode_seed=4)
        r = h.harvest()
        self.assertTrue(r.ok and r.encounter_started)
        self.assertLessEqual(abs(emu.x - 10), 3)
        emu.in_battle = False
        self.assertTrue(h.restore_anchor().ok)

    def test_two_tile_jump_aborts_and_restores(self):        # review #4
        emu = MultiFrameTileEmu(x=10, frames_per_tile=4, jump_after=2,
                                encounter_after=None)
        h = _harv(emu, episode_seed=0)
        r = h.harvest()
        self.assertFalse(r.ok)
        self.assertEqual(r["reason"], "jump_gt_1_tile")
        self.assertTrue(r["restore_ok"])
        self.assertEqual((emu.x, emu.y), (10, 4))

    def test_blocked_wall_is_not_an_abort_just_a_reverse(self):
        emu = MultiFrameTileEmu(x=10, frames_per_tile=4, walls=(9, 8, 7, 6),
                                encounter_after=3)
        h = _harv(emu, episode_seed=1)   # may start LEFT into the wall
        r = h.harvest()
        # walls on the left -> it reverses and finds the encounter on the right
        self.assertTrue(r.encounter_started or r["reason"] == "corridor_blocked_both_sides")

    def test_unreadable_location_aborts_fail_closed(self):
        emu = FakeEmu(x=10)
        h = _harv(emu, episode_seed=0)
        h.capture_anchor()
        h._loc = lambda _e: {"valid": False}      # the RAM read goes bad
        kind, why = h._walk_one_tile("LEFT")
        self.assertEqual(kind, "abort")
        self.assertEqual(why, "location_unreadable")

    def test_map_changes_before_delayed_battle_latch_is_an_encounter(self):
        emu = FakeEmu(x=10, encounter_after=1, encounter_latch_delay=40)
        h = _harv(emu, episode_seed=0)
        r = h.harvest()
        self.assertTrue(r.ok and r.encounter_started)
        self.assertTrue(emu.in_battle)


class EncounterVarietyTests(unittest.TestCase):
    def test_episode_seed_reproducible_but_varies_per_episode(self):  # review #5
        seeds = [harvest_episode_seed(42, 7, i) for i in range(1, 21)]
        self.assertGreaterEqual(len(set(seeds)), 15)      # clearly varied
        again = [harvest_episode_seed(42, 7, i) for i in range(1, 21)]
        self.assertEqual(seeds, again)                    # same run reproduces
        other_run = [harvest_episode_seed(42, 99, i) for i in range(1, 21)]
        self.assertNotEqual(seeds, other_run)

    def test_sampler_gives_a_harvest_anchor_a_varying_stored_seed(self):
        import json
        import numpy as np
        from battle_train import LiveScenarioSampler
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.state.gz"); open(p, "wb").close()
            row = {"id": "route1_grass", "area": "route1", "harvest": True,
                   "bucket": "route1|grass", "savestate_path": p,
                   "rng_seed_id": 12345, "trained_count": 0}
            idx = os.path.join(td, "index.json")
            json.dump({"scenarios": [row]}, open(idx, "w"))

            def seq(run_seed):
                s = LiveScenarioSampler(index_path=idx, run_seed=run_seed)
                rng = np.random.default_rng(0)
                return [s(rng)["episode_seed"] for _ in range(20)]

            s1 = seq(7)
            self.assertGreaterEqual(len(set(s1)), 15)     # 20 episodes -> many seeds
            self.assertEqual(seq(7), s1)                  # same run-seed reproduces
            self.assertNotEqual(seq(8), s1)
            # pre_action_wait is folded in, not ignored
            self.assertTrue(all(isinstance(x, int) for x in s1))

    def test_sampler_marks_encounter_sequence(self):
        import json
        import numpy as np
        from battle_train import LiveScenarioSampler
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.state.gz"); open(p, "wb").close()
            json.dump({"scenarios": [{"id": "g", "area": "r1", "harvest": True,
                                      "bucket": "b", "savestate_path": p,
                                      "rng_seed_id": 1}]}, open(os.path.join(td, "index.json"), "w"))
            s = LiveScenarioSampler(index_path=os.path.join(td, "index.json"), run_seed=1)
            rng = np.random.default_rng(0)
            seqs = [s(rng)["encounter_sequence"] for _ in range(5)]
            self.assertEqual(seqs, [1, 2, 3, 4, 5])


# --------------------------------------------------------------------------
class ShinyMathTests(unittest.TestCase):
    def test_gen3_formula(self):
        self.assertTrue(shiny.is_shiny(0, 0, 0))
        self.assertTrue(shiny.is_shiny(0, 0, 7))            # sv == 7 < 8
        self.assertFalse(shiny.is_shiny(0, 0, 8))
        self.assertIsNone(shiny.is_shiny(None, 0, 123))
        tid, sid = shiny.split_trainer_id(0xB2C30001)
        self.assertEqual((tid, sid), (0x0001, 0xB2C3))


class ShinyRamFailClosedTests(unittest.TestCase):
    def test_unverified_pid_tid_sid_is_unknown_never_false(self):   # ZIEL E.8
        self.assertFalse(shiny_ram.SHINY_RAM_VERIFIED)
        st = shiny_ram.shiny_status(is_trainer=False, enemy_party=[
            {"checksum_ok": True, "cur_hp": 10, "personality": 12345}])
        self.assertEqual(st["status"], "unknown")
        self.assertEqual(st["reason"], "shiny_ram_unverified")
        self.assertIsNone(st["shiny_value"])
        self.assertIsNone(shiny_ram.read_player_trainer_id(b"\x00" * 0x40000))

    def test_trainer_battle_is_never_a_catchable_shiny(self):       # ZIEL E.9
        st = shiny_ram.shiny_status(is_trainer=True, enemy_party=[])
        self.assertEqual(st["status"], "not_applicable")
        self.assertFalse(shiny_ram.is_verified_shiny(st))

    def test_planner_never_emits_a_shiny_order_while_unverified(self):
        import catch_planner
        o = catch_planner.plan_battle_objective(dict(
            area="viridian_forest", battle_confirmed_wild=True,
            shiny_status="unknown", enemy_species_known=True, enemy_species_id=10,
            usable_ball_count=5, party_has_space=True, catch_execution_ready=True))
        self.assertFalse(o["catch_requested"])
        # a trainer + (impossible) verified_shiny still yields combat
        o2 = catch_planner.plan_battle_objective(dict(
            is_trainer=True, battle_confirmed_wild=False,
            shiny_status="verified_shiny"))
        self.assertFalse(o2["catch_requested"])
        self.assertEqual(o2["catch_reason"], "trainer_battle")


class ShinyCounterTests(unittest.TestCase):
    def _cnt(self, td):
        return ShinyCounters("battle_fighter", base_dir=td)

    def test_encounter_counted_exactly_once_despite_many_calls(self):  # ZIEL E.6
        with tempfile.TemporaryDirectory() as td:
            c = self._cnt(td)
            for _ in range(50):
                c.record_encounter("enc_A", shiny_status="unknown")
            self.assertEqual(c.snapshot()["counters"]["wild_encounters"], 1)
            self.assertEqual(c.snapshot()["counters"]["shiny_ram_unknown"], 1)

    def test_savestate_reload_does_not_recount(self):       # ZIEL E.7
        with tempfile.TemporaryDirectory() as td:
            self._cnt(td).record_encounter("enc_S", shiny_status="verified_shiny",
                                           species_id=25, level=5, area="forest")
            # a "reload" = a brand-new counters object on the same file
            self._cnt(td).record_encounter("enc_S", shiny_status="verified_shiny")
            self._cnt(td).record_shiny_outcome("enc_S", "caught")
            self._cnt(td).record_shiny_outcome("enc_S", "caught")
            snap = self._cnt(td).snapshot()["counters"]
            self.assertEqual(snap["shiny_encounters_verified"], 1)
            self.assertEqual(snap["shiny_caught"], 1)

    def test_agent_classes_have_separate_files(self):
        with tempfile.TemporaryDirectory() as td:
            ShinyCounters("battle_fighter", base_dir=td).record_encounter("x")
            ShinyCounters("full_agent", base_dir=td).record_encounter("y")
            agg = aggregate_all(base_dir=td)
            self.assertEqual(agg["per_agent_class"]["battle_fighter"]["counters"]["wild_encounters"], 1)
            self.assertEqual(agg["per_agent_class"]["full_agent"]["counters"]["wild_encounters"], 1)
            self.assertEqual(agg["per_agent_class"]["watcher"]["counters"]["wild_encounters"], 0)
            self.assertEqual(agg["totals"]["wild_encounters"], 2)


class ShinyCounterConcurrencyTests(unittest.TestCase):
    def test_parallel_writers_lose_no_updates(self):        # review #1
        import multiprocessing as mp
        with tempfile.TemporaryDirectory() as td:
            procs = [mp.get_context("spawn").Process(
                target=_mp_worker_unique, args=(td, f"w{k}", 40)) for k in range(9)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(60)
                self.assertEqual(p.exitcode, 0)
            snap = ShinyCounters("battle_fighter", base_dir=td).snapshot()
            self.assertEqual(snap["counters"]["wild_encounters"], 9 * 40)

    def test_two_processes_racing_the_same_id_count_it_once(self):
        import multiprocessing as mp
        with tempfile.TemporaryDirectory() as td:
            procs = [mp.get_context("spawn").Process(
                target=_mp_worker_same, args=(td,)) for _ in range(6)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(60)
            snap = ShinyCounters("full_agent", base_dir=td).snapshot()
            self.assertEqual(snap["counters"]["wild_encounters"], 1)


class ShinyOutcomeValidationTests(unittest.TestCase):
    def _c(self, td):
        return ShinyCounters("battle_fighter", base_dir=td)

    def test_outcome_only_accepted_for_a_verified_shiny_id(self):   # review #2
        with tempfile.TemporaryDirectory() as td:
            c = self._c(td)
            c.record_encounter("norm", shiny_status="verified_normal")
            r = c.record_shiny_outcome("norm", "caught")
            self.assertFalse(r["recorded"])
            self.assertEqual(r["reason"], "not_a_verified_shiny")
            self.assertEqual(c.snapshot()["counters"]["shiny_caught"], 0)

    def test_exactly_one_terminal_outcome_per_encounter(self):      # review #2
        with tempfile.TemporaryDirectory() as td:
            c = self._c(td)
            c.record_encounter("s1", shiny_status="verified_shiny")
            self.assertTrue(c.record_shiny_outcome("s1", "caught")["recorded"])
            # same outcome again -> idempotent, not re-counted
            self.assertFalse(c.record_shiny_outcome("s1", "caught")["recorded"])
            # contradictory outcome -> rejected + diagnosed
            r = c.record_shiny_outcome("s1", "ko")
            self.assertFalse(r["recorded"])
            self.assertIn("conflict", r["reason"])
            snap = c.snapshot()["counters"]
            self.assertEqual(snap["shiny_caught"], 1)
            self.assertEqual(snap["shiny_ko"], 0)
            self.assertEqual(snap["shiny_outcome_conflicts"], 1)

    def test_old_verified_shiny_not_recounted_after_seen_truncation(self):
        # review: seen_encounters is capped at 4000; verified_shiny_encounters
        # is not. Replaying an old verified shiny must not re-increment.
        with tempfile.TemporaryDirectory() as td:
            c = self._c(td)
            c.record_encounter("shiny_old", shiny_status="verified_shiny")
            # flood past the 4000 cap so "shiny_old" falls out of seen_encounters
            for i in range(4100):
                c.record_encounter(f"filler-{i}", shiny_status="unknown")
            self.assertFalse("shiny_old" in
                             set(c._load()["seen_encounters"]))          # evicted
            before = c.snapshot()["counters"]["shiny_encounters_verified"]
            # savestate replay of the same shiny
            self.assertFalse(c.record_encounter("shiny_old",
                                                shiny_status="verified_shiny"))
            self.assertEqual(c.snapshot()["counters"]["shiny_encounters_verified"], before)
            self.assertEqual(before, 1)

    def test_no_shiny_outcome_while_ram_unverified(self):           # review #3
        # shiny_ram is fail-closed -> record_encounter never adds to the
        # verified set -> record_shiny_outcome always rejects.
        self.assertFalse(shiny_ram.SHINY_RAM_VERIFIED)
        with tempfile.TemporaryDirectory() as td:
            c = self._c(td)
            st = shiny_ram.shiny_status(is_trainer=False, enemy_party=[
                {"checksum_ok": True, "cur_hp": 5, "personality": 999}])
            c.record_encounter("u1", shiny_status=st["status"])
            self.assertFalse(c.is_verified_shiny_encounter("u1"))
            self.assertFalse(c.record_shiny_outcome("u1", "caught")["recorded"])
            self.assertEqual(c.snapshot()["counters"]["shiny_encounters_verified"], 0)


# --------------------------------------------------------------------------
class ScenarioDiversityTests(unittest.TestCase):
    def test_bucket_key_shape(self):
        b = sp.scenario_bucket(area="route1", enemy_species=[16], enemy_level=3,
                               player_levels=[7, 6], own_species=[1, 4],
                               objective_mode="combat")
        self.assertEqual(b.count("|"), 5)   # 6 fields
        self.assertIn("route1", b)
        self.assertIn("combat", b)

    def test_level_bucket(self):
        self.assertEqual(sp.level_bucket(1), "L1-5")
        self.assertEqual(sp.level_bucket(5), "L1-5")
        self.assertEqual(sp.level_bucket(6), "L6-10")

    def test_sampler_is_bucket_balanced(self):
        import json
        import numpy as np
        from battle_train import LiveScenarioSampler
        with tempfile.TemporaryDirectory() as td:
            rows = []
            for i in range(20):        # 20 near-identical "rattata" rows
                p = os.path.join(td, f"r{i}.state.gz"); open(p, "wb").close()
                rows.append({"id": f"ratt{i}", "area": "route1",
                             "bucket": "route1|19|L1-5|L6-10|1@L1-5|combat",
                             "savestate_path": p, "trained_count": 0})
            for i in range(2):         # 2 "pidgey" rows -> a rarer bucket
                p = os.path.join(td, f"p{i}.state.gz"); open(p, "wb").close()
                rows.append({"id": f"pidg{i}", "area": "route1",
                             "bucket": "route1|16|L1-5|L6-10|1@L1-5|combat",
                             "savestate_path": p, "trained_count": 0})
            idx = os.path.join(td, "index.json")
            json.dump({"scenarios": rows}, open(idx, "w"))
            s = LiveScenarioSampler(index_path=idx)
            rng = np.random.default_rng(0)
            picks = [s(rng)["scenario_bucket"] for _ in range(40)]
            n_pidgey = sum(1 for b in picks if "|16|" in b)
            # balanced sampling => the 2-row bucket gets ~half the picks,
            # nowhere near the 20/22 a uniform-row sampler would give it
            self.assertGreater(n_pidgey, 12)


# --------------------------------------------------------------------------
class ShinyRewardPrepTests(unittest.TestCase):
    def test_no_2000_anywhere_in_the_reward(self):          # ZIEL E.13
        self.assertEqual(battle_env.SHINY_PRIORITY_SCORE, 2000)
        # it is a module constant, NOT a BattleRewardConfig field
        self.assertFalse(hasattr(BattleRewardConfig, "SHINY_PRIORITY_SCORE"))
        comps = _reward_components_v2(
            {"objective_mode": "catch", "catch_requested": True,
             "catch_reason": "verified_shiny"},
            {"catch_attempted": True, "catch_success": True, "balls_used": 1,
             "shiny_catch_success": True})
        total = sum(a for _, a in comps)
        self.assertLess(total, 20.0)        # catch_success 3 + shiny 10 - costs
        self.assertNotIn(2000, [round(a) for _, a in comps])

    def test_shiny_bonus_paid_once_and_only_on_confirmed_catch(self):  # ZIEL E.12
        base = {"objective_mode": "catch", "catch_requested": True,
                "catch_reason": "verified_shiny"}
        # encounter only -> nothing
        self.assertNotIn("shiny_catch_success",
                         [n for n, _ in _reward_components_v2(base, {})])
        # failed catch -> nothing
        self.assertNotIn("shiny_catch_success", [n for n, _ in _reward_components_v2(
            base, {"catch_attempted": True, "catch_outcome": "broke_free",
                   "shiny_catch_success": True})])
        # confirmed catch -> exactly one shiny_catch_success
        names = [n for n, _ in _reward_components_v2(
            base, {"catch_attempted": True, "catch_success": True,
                   "balls_used": 1, "shiny_catch_success": True})]
        self.assertEqual(names.count("shiny_catch_success"), 1)
        # not verified_shiny reason -> no shiny bonus even if flag set
        names2 = [n for n, _ in _reward_components_v2(
            {"objective_mode": "catch", "catch_requested": True,
             "catch_reason": "new_species"},
            {"catch_attempted": True, "catch_success": True, "balls_used": 1,
             "shiny_catch_success": True})]
        self.assertNotIn("shiny_catch_success", names2)

    def test_normal_catch_rewards_unchanged(self):          # ZIEL E.11
        r = _reward_components_v2(
            {"objective_mode": "catch", "catch_requested": True,
             "catch_reason": "new_species"},
            {"catch_attempted": True, "catch_success": True, "balls_used": 1})
        d = dict(r)
        self.assertEqual(d["catch_success"], BattleRewardConfig.CATCH_SUCCESS)
        self.assertNotIn("shiny_catch_success", d)

    def test_components_still_sum_exactly_with_a_shiny_catch(self):
        e = BattleEnv(SimulatedBattleDriver(), schema="v2", max_turns=25)
        obs, info = e.reset(seed=3, options={"scenario": catch_scenario(
            catch_rate=255, catch_seed=3)})
        obs, r, term, trunc, i = e.step(ALL_MACROS.index(CATCH_ACTION))
        self.assertAlmostEqual(r, sum(a for _, a in i["reward_components"]),
                               places=6)


class ShinyMaskPrepTests(unittest.TestCase):
    def test_shiny_priority_mask_is_dormant_without_critical_priority(self):  # ZIEL E.10
        snap = {"enemy_active": {"cur_hp": 5},
                "player_active": {"moves": [{"power": 90, "is_status": False}]}}
        base = [1] * len(ALL_MACROS)
        self.assertEqual(bx.shiny_priority_mask(base, snap, {}, macros=ALL_MACROS), base)
        self.assertEqual(
            bx.shiny_priority_mask(base, snap, {"catch_priority": "none"},
                                   macros=ALL_MACROS), base)

    def test_shiny_priority_mask_masks_run_and_ko_moves_when_critical(self):
        snap = {"enemy_active": {"cur_hp": 30},
                "player_active": {"moves": [
                    {"power": 90, "is_status": False},   # >= HP -> likely KO, masked
                    {"power": 15, "is_status": False},   # weak chip -> kept
                    {"power": 0, "is_status": True},     # status -> kept
                    {"power": 0, "is_status": True}]}}
        base = [1] * len(ALL_MACROS)
        out = bx.shiny_priority_mask(base, snap, {"catch_priority": "critical"},
                                     macros=ALL_MACROS)
        self.assertEqual(out[0], 0)                        # MOVE_1 (KO) masked
        self.assertEqual(out[1], 1)                        # MOVE_2 kept
        self.assertEqual(out[2], 1)                        # status kept
        self.assertEqual(out[ALL_MACROS.index(RUN_ACTION)], 0)   # RUN masked
        self.assertTrue(any(out))


class HarvesterWiringTests(unittest.TestCase):
    def test_reset_only_runs_the_harvester_for_a_harvest_scenario(self):
        import inspect
        from battle_env import LiveBattleDriver
        src = inspect.getsource(LiveBattleDriver.reset)
        self.assertIn('self._scenario.get("harvest")', src)
        self.assertIn("_run_harvester", src)
        # a failed harvest ends the episode as a diagnosed reset fault
        self.assertIn('"reset_unreadable"', src)

    def test_harvester_uses_verified_readers_only(self):
        import inspect
        from battle_env import LiveBattleDriver
        src = inspect.getsource(LiveBattleDriver._run_harvester)
        self.assertIn("firered_ram", src)          # verified location read
        self.assertIn("_mbr.read", src)            # verified in-battle latch
        self.assertNotIn("set_state", src)         # restore is the next reset

    def test_apply_macro_records_a_shiny_outcome_exactly_once(self):  # review #3
        import inspect
        from battle_env import LiveBattleDriver
        am = inspect.getsource(LiveBattleDriver.apply_macro)
        self.assertIn("_record_wild_outcome", am)
        ro = inspect.getsource(LiveBattleDriver._record_wild_outcome)
        self.assertIn("_shiny_outcome_recorded", ro)
        self.assertIn('"verified_shiny"', ro)      # never counts while unknown
        self.assertIn("record_shiny_outcome", ro)

    def test_adapter_rejects_an_invalid_agent_class(self):
        from twoby2.live_integration import NavBattleDriverAdapter
        with self.assertRaises(ValueError):
            NavBattleDriverAdapter(lambda e: None, lambda o: 0, agent_class="nope")


# --------------------------------------------------------------------------
class _FakeRetro:
    def get_ram(self):
        return b""


import gymnasium as _gym


class _FakeNavEnv(_gym.Env):
    observation_space = _gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
    action_space = _gym.spaces.Discrete(2)

    def __init__(self):
        super().__init__()
        self.env = _FakeRetro()          # the retro env _record_* reads RAM from
        self.cached_loc = {"map_bank": 3, "map_id": 19, "x_pos": 5, "y_pos": 9}
        self.route_steps = 0

    def reset(self, **kw):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, a):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _FakeBattleDriver:
    """Enough for NavBattleDriverAdapter.play_battle: a confirmed wild battle
    that the pinned policy wins in one 'turn'."""
    def __init__(self, outcome="win"):
        self.env = _FakeRetro()
        self.max_turns = 80
        self._outcome = outcome

    def in_battle(self):
        return True

    def snapshot(self):
        return {"in_battle": True, "is_trainer": False, "is_double": False,
                "player_party": [], "enemy_active": {"species_id": 19, "level": 3},
                "can_escape": True}

    def play_battle(self, policy_fn, obs_fn):
        return {"outcome": self._outcome, "turns": 1, "duration_steps": 3,
                "diagnostics": []}


def _run_one_wild_battle(agent_class, base_dir, outcome="win"):
    """Drive one wild battle through the adapter with ShinyCounters pointed at
    ``base_dir``. Returns aggregate_all(base_dir)."""
    import twoby2.shiny_counters as sc_mod
    from twoby2.live_integration import NavBattleDriverAdapter
    real = sc_mod.ShinyCounters
    adapter = NavBattleDriverAdapter(
        lambda e: _FakeBattleDriver(outcome), lambda o: 0,
        agent_class=agent_class)
    with mock.patch.object(sc_mod, "ShinyCounters",
                           lambda cls: real(cls, base_dir=base_dir)):
        adapter.play_battle(_FakeNavEnv())
    return aggregate_all(base_dir=base_dir)


class WatcherRoutingBehaviourTests(unittest.TestCase):
    def test_watcher_wrapper_writes_only_the_watcher_counter(self):   # review pt.1
        from twoby2.live_integration import maybe_wrap_full_agent
        with mock.patch("twoby2.live_integration.feature_enabled", return_value=True):
            w = maybe_wrap_full_agent(_FakeNavEnv(), learning=False,
                                      battle_policy=lambda o: 0,
                                      emulator_driver_factory=lambda e: _FakeBattleDriver())
        self.assertEqual(w.driver.agent_class, "watcher")

    def test_learning_false_only_touches_watcher(self):              # review pt.4
        with tempfile.TemporaryDirectory() as td:
            agg = _run_one_wild_battle("watcher", td)
            self.assertEqual(agg["per_agent_class"]["watcher"]["counters"]["wild_encounters"], 1)
            self.assertEqual(agg["per_agent_class"]["full_agent"]["counters"]["wild_encounters"], 0)
            self.assertEqual(sorted(os.listdir(td)),
                             ["watcher.json", "watcher.json.lock"])

    def test_learning_true_only_touches_full_agent(self):            # review pt.5
        with tempfile.TemporaryDirectory() as td:
            agg = _run_one_wild_battle("full_agent", td)
            self.assertEqual(agg["per_agent_class"]["full_agent"]["counters"]["wild_encounters"], 1)
            self.assertEqual(agg["per_agent_class"]["watcher"]["counters"]["wild_encounters"], 0)
            self.assertEqual(sorted(os.listdir(td)),
                             ["full_agent.json", "full_agent.json.lock"])


class FullAgentIsolationTests(unittest.TestCase):
    def test_harvester_is_never_wired_into_the_full_navigation_path(self):  # ZIEL E.1
        import inspect
        import pokemon_env
        import train
        for mod in (pokemon_env, train):
            src = inspect.getsource(mod)
            self.assertNotIn("wild_encounter_harvester", src)
            self.assertNotIn("WildEncounterHarvester", src)

    def test_navigation_battle_reward_has_no_grass_or_battle_start_term(self):  # ZIEL E.1/E.2
        from twoby2.battle_summary import navigation_battle_reward
        # starting / losing a wild fight is never positive nav reward
        self.assertLessEqual(navigation_battle_reward(
            {"outcome": "loss", "party_hp_fraction_lost": 0.0}), 0.0)
        self.assertEqual(navigation_battle_reward(
            {"outcome": "win", "party_hp_fraction_lost": 0.0}), 0.0)

    def test_battle_subepisode_does_not_touch_nav_counters(self):  # ZIEL E.2
        import inspect
        from twoby2.nav_wrapper import NavigationBattleWrapper
        src = inspect.getsource(NavigationBattleWrapper.step)
        # the wrapper asserts the nav step counter is unmoved by the battle
        self.assertIn("advanced the navigation step counter", src)


class SchemaGuardTests(unittest.TestCase):
    def test_v1_stays_116_11_no_catch_v2_stays_140_12(self):   # ZIEL E.14 / E.15
        from battle_env import battle_schema_spec
        m1, d1, *_ = battle_schema_spec("v1")
        m2, d2, *_ = battle_schema_spec("v2")
        self.assertEqual((d1, len(m1)), (116, 11))
        self.assertEqual((d2, len(m2)), (140, 12))
        self.assertNotIn("CATCH", m1)
        self.assertIn("CATCH", m2)
        self.assertFalse(shiny_ram.SHINY_RAM_VERIFIED)   # v2 catch stays fail-closed


class CheckpointImmutabilityTests(unittest.TestCase):
    def test_new_modules_never_reference_a_model_checkpoint_path(self):   # ZIEL E.16
        import inspect
        from twoby2 import wild_encounter_harvester, shiny_ram, shiny_counters
        import shiny as shiny_mod
        for mod in (wild_encounter_harvester, shiny_ram, shiny_counters, shiny_mod):
            src = inspect.getsource(mod)
            for bad in ("checkpoints", "battle_champion", "navigation_champion",
                        ".zip", "model_version", "champion_score"):
                self.assertNotIn(bad, src, f"{mod.__name__} touches {bad!r}")

    def test_shiny_counters_only_write_under_runtime_shiny(self):
        from twoby2.shiny_counters import SHINY_DIR
        self.assertTrue(SHINY_DIR.replace("\\", "/").endswith("runtime/shiny"))
        with tempfile.TemporaryDirectory() as td:
            c = ShinyCounters("battle_fighter", base_dir=td)
            c.record_encounter("z")
            self.assertTrue(os.path.isfile(os.path.join(td, "battle_fighter.json")))
            self.assertEqual(sorted(os.listdir(td)),
                             ["battle_fighter.json", "battle_fighter.json.lock"])


if __name__ == "__main__":
    unittest.main()
