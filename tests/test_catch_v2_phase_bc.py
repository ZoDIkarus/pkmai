"""Catch-v2 Phase B/C — executor CATCH state machine, fail-closed RAM gate,
live-driver masking, promotion catch gate, v1/v2 migration preview + schema
preflight, telemetry, battle-episode summary and the nav strategic term."""
import os
import tempfile
import unittest

import battle_executor as bx
from battle_executor import (SCREEN_MAIN, SCREEN_BAG, SCREEN_ITEM_USE,
                             SCREEN_MESSAGE, CATCH_ACTION, MENU_BAG,
                             ALL_MACROS_V1, catch_precheck)
import battle_catch


# --------------------------------------------------------------------------
# a catch-capable in-memory battle-menu model
# --------------------------------------------------------------------------
def _mon(slot=0, hp=30, mx=40, sid=0, moves=()):
    return {"slot": slot, "checksum_ok": True, "cur_hp": hp, "max_hp": mx,
            "species_id": sid, "moves": [dict(m) for m in moves]}


def _obj(**kw):
    d = dict(objective_mode="catch", catch_requested=True, target_species_id=19,
             usable_ball_count=4, reserved_ball_count=1, party_has_space=True,
             pc_capture_supported=False)
    d.update(kw)
    return d


def _snap(**kw):
    d = dict(player_active=_mon(0, moves=[{"id": 1, "pp": 10, "mechanics_known": True,
                                           "type": 0, "power": 40}]),
             enemy_active=_mon(0, hp=18, sid=19),
             is_trainer=False, can_escape=True, menu_state="main",
             menu_ready=True, message_pending=False, animation_active=False)
    d["player_party"] = [d["player_active"]]
    d.update(kw)
    return d


class CatchIO(bx.BattleIO):
    def __init__(self, *, pocket=((4, 5),), snapshot=None, no_leave_bag=False):
        self._pocket = [tuple(p) for p in pocket]
        self._snap = snapshot or _snap()
        self.no_leave_bag = no_leave_bag
        self.screen = SCREEN_MAIN
        self.cursor = 0
        self.presses = []
        self.thrown = 0

    def in_battle(self):
        return True

    def snapshot(self):
        return self._snap

    def ball_pocket(self):
        return list(self._pocket)

    def menu_state(self):
        return {"screen": self.screen, "cursor": self.cursor,
                "message_pending": False, "battler_ready": True,
                "pocket": "balls"}

    def wait_frames(self, frames):
        return self.menu_state()

    def press(self, button):
        self.presses.append(button)
        s = self.screen
        if s == SCREEN_MAIN:
            if button == "RIGHT" and self.cursor == 0:
                self.cursor = 1
            elif button == "LEFT" and self.cursor == 1:
                self.cursor = 0
            elif button == "DOWN" and self.cursor < 2:
                self.cursor += 2
            elif button == "UP" and self.cursor >= 2:
                self.cursor -= 2
            elif button == "A" and self.cursor == MENU_BAG:
                self.screen, self.cursor = SCREEN_BAG, 0
        elif s == SCREEN_BAG:
            if button == "A":
                self.screen = SCREEN_ITEM_USE
                self.cursor = 0
            elif button == "B" and not self.no_leave_bag:
                self.screen, self.cursor = SCREEN_MAIN, MENU_BAG
            elif button in ("UP", "DOWN", "LEFT", "RIGHT"):
                pass
        elif s == SCREEN_ITEM_USE:
            if button == "A":
                if self.no_leave_bag:
                    self.screen = SCREEN_BAG
                else:
                    self.thrown = 1
                    self.screen = SCREEN_MESSAGE
            elif button == "B":
                self.screen = SCREEN_BAG
        elif s == SCREEN_MESSAGE:
            self.screen = SCREEN_MAIN
        return self.menu_state()


def _run_catch(io, objective):
    return bx.MacroExecutor(io).execute(CATCH_ACTION, objective=objective,
                                        dry_run=True)


class ExecutorCatchMacroTests(unittest.TestCase):
    def test_catch_macro_reaches_the_bag_and_throws_one_ball(self):
        io = CatchIO()
        r = _run_catch(io, _obj())
        self.assertTrue(r["ok"], r["reason"])
        self.assertEqual(r["balls_used"], 1)
        self.assertIn("A", io.presses)
        self.assertLessEqual(r["presses"], bx.DEFAULT_MAX_PRESSES)

    def test_catch_precheck_rejects_before_any_button(self):
        io = CatchIO(snapshot=_snap(is_trainer=True))
        r = _run_catch(io, _obj())
        self.assertFalse(r["ok"])
        self.assertTrue(r.get("invalid"))
        self.assertEqual(r.get("catch_reject"), "trainer_catch_blocked")
        self.assertEqual(io.presses, [])
        self.assertEqual(r["balls_used"], 0)

    def test_wrong_target_species_is_rejected(self):
        io = CatchIO(snapshot=_snap(enemy_active=_mon(0, hp=18, sid=99)))
        r = _run_catch(io, _obj(target_species_id=19))
        self.assertFalse(r["ok"])
        self.assertEqual(io.presses, [])

    def test_reserve_is_respected_by_ball_pick(self):
        # 1 ball, reserve 1 -> no usable ball -> abort, no throw
        io = CatchIO(pocket=((4, 1),))
        r = _run_catch(io, _obj(usable_ball_count=0, reserved_ball_count=1))
        self.assertFalse(r["ok"])
        self.assertEqual(io.thrown, 0)

    def test_non_ball_in_pocket_is_a_read_fault(self):
        io = CatchIO(pocket=((13, 3),))       # 13 is not a ball id
        r = _run_catch(io, _obj())
        self.assertTrue(r["aborted"])
        self.assertIn("bag_unreadable", r["reason"])

    def test_bag_never_left_aborts_as_bag_unreadable(self):
        io = CatchIO(no_leave_bag=True)
        r = _run_catch(io, _obj())
        self.assertTrue(r["aborted"])
        self.assertIn("bag_unreadable", r["reason"])

    def test_best_usable_ball_prefers_ultra_over_poke(self):
        io = CatchIO(pocket=((4, 5), (2, 2)))   # poke x5, ultra x2
        idx, item = bx.MacroExecutor(io)._best_usable_ball(_obj())
        self.assertEqual(item, battle_catch.ULTRA_BALL)


class CatchPrecheckTests(unittest.TestCase):
    def test_all_unknowns_fail_closed(self):
        for missing in ("catch_requested", "enemy_active", "usable_ball_count"):
            o = _obj()
            s = _snap()
            if missing == "catch_requested":
                o["catch_requested"] = False
            elif missing == "enemy_active":
                s["enemy_active"] = {}
            else:
                o["usable_ball_count"] = 0
            self.assertFalse(catch_precheck(s, o)[0], missing)

    def test_text_or_animation_blocks_the_catch(self):
        self.assertFalse(catch_precheck(_snap(message_pending=True), _obj())[0])
        self.assertFalse(catch_precheck(_snap(animation_active=True), _obj())[0])

    def test_full_party_no_pc_blocks(self):
        self.assertFalse(catch_precheck(
            _snap(), _obj(party_has_space=False, pc_capture_supported=False))[0])
        self.assertTrue(catch_precheck(
            _snap(), _obj(party_has_space=False, pc_capture_supported=True))[0])


class LiveRamGateTests(unittest.TestCase):
    def test_catch_ram_is_fail_closed_for_bprd(self):
        from twoby2 import battle_ram_live as L
        ready, missing = L.catch_ram_ready(None)
        self.assertFalse(ready)
        self.assertTrue(missing)
        self.assertIsNone(L.ball_pocket(b"\x00" * 0x40000))
        self.assertIsNone(L.pokedex_owned(b"\x00" * 0x40000, 19))

    def test_live_driver_never_offers_catch_while_ram_unverified(self):
        from twoby2.emulator_battle_driver import EmulatorBattleDriver
        from battle_executor import ALL_MACROS
        drv = EmulatorBattleDriver.__new__(EmulatorBattleDriver)
        drv._objective = _obj()
        drv.schema = "v2"
        drv._macros = ALL_MACROS
        # a snapshot that would pass catch_precheck but has catch_ram_ready False
        snap = _snap()
        snap.update(catch_ram_ready=False, active_slot=0)
        legal = EmulatorBattleDriver.legal_macros(drv, snap)
        self.assertNotIn(CATCH_ACTION, legal)
        snap2 = dict(snap, catch_ram_ready=True)
        # even "ready" only adds it when the precheck passes; here party ok etc.
        legal2 = EmulatorBattleDriver.legal_macros(drv, snap2)
        self.assertIn(CATCH_ACTION, legal2)
        # a v1 driver has no CATCH slot regardless of the RAM gate
        drv.schema, drv._macros = "v1", ALL_MACROS_V1
        self.assertNotIn(CATCH_ACTION,
                         EmulatorBattleDriver.legal_macros(drv, snap2))


class PromotionCatchGateTests(unittest.TestCase):
    def _good(self):
        return dict(catch_requested_episodes=40,
                    catch_success_rate_when_requested=0.80,
                    per_group_catch_success_rate={"route1": 0.75},
                    target_ko_rate=0.05, balls_per_successful_catch=2.1,
                    unrequested_catch_rate=0.0, trainer_catch_attempts=0)

    def test_clean_catch_eval_passes(self):
        from twoby2.battle_promotion import battle_catch_gate
        ok, reasons = battle_catch_gate(self._good())
        self.assertTrue(ok, reasons)

    def test_low_success_rate_blocks(self):
        from twoby2.battle_promotion import battle_catch_gate
        m = self._good(); m["catch_success_rate_when_requested"] = 0.5
        self.assertFalse(battle_catch_gate(m)[0])

    def test_target_ko_rate_blocks(self):
        from twoby2.battle_promotion import battle_catch_gate
        m = self._good(); m["target_ko_rate"] = 0.2
        self.assertFalse(battle_catch_gate(m)[0])

    def test_trainer_catch_attempt_blocks(self):
        from twoby2.battle_promotion import battle_catch_gate
        m = self._good(); m["trainer_catch_attempts"] = 1
        self.assertFalse(battle_catch_gate(m)[0])

    def test_high_combat_winrate_cannot_promote_a_broken_catch_v2(self):
        from twoby2.battle_promotion import evaluate_battle_promotion
        champ = {"episodes": 200, "win_rate": 0.5, "wild_win_rate": 0.5,
                 "trainer_win_rate": 0.5, "avg_residual_hp_on_win": 0.5}
        cand = {**champ, "win_rate": 0.9, "wild_win_rate": 0.9,
                "trainer_win_rate": 0.9, "avg_residual_hp_on_win": 0.7,
                "n_trainer": 5, "n_wild": 5, "scenario_coverage": 5,
                "episodes": 200}
        bad_catch = self._good(); bad_catch["catch_success_rate_when_requested"] = 0.1
        d = evaluate_battle_promotion(candidate=cand, champion=champ,
                                      catch_eval=bad_catch)
        self.assertFalse(d["promote"])
        self.assertIn("catch gate", d["reason"])


class MigrationPreviewTests(unittest.TestCase):
    def test_preview_applies_nothing_and_lists_both_schemas(self):
        import battle_train as bt
        p = bt.battle_migration_preview()
        self.assertFalse(p["applies_anything"])
        self.assertEqual(p["archived_files"], [])
        self.assertTrue(p["current_v1_champion"]["stays_live"])
        self.assertEqual(p["expected_dims"]["v1"]["obs_dim"], 116)
        self.assertEqual(p["expected_dims"]["v2"]["obs_dim"], 140)
        self.assertEqual(p["expected_dims"]["v2"]["n_actions"], 12)

    def test_schema_preflight_refuses_v1_file_in_v2_trainer(self):
        import zipfile
        import battle_train as bt
        with tempfile.TemporaryDirectory() as d:
            z = os.path.join(d, "battle_champion.zip")
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("data", '{"observation_space":{"vec":{"shape":[116]}},'
                                    '"action_space":{"n":11}}')
            with self.assertRaises(bt.BattleSchemaError):
                bt.battle_schema_preflight(z, expect_obs_dim=140,
                                           expect_n_actions=12, schema="v2")
            # a matching file passes silently
            z2 = os.path.join(d, "v2.zip")
            with zipfile.ZipFile(z2, "w") as zf:
                zf.writestr("data", '{"observation_space":{"vec":{"shape":[140]}},'
                                    '"action_space":{"n":12}}')
            bt.battle_schema_preflight(z2, expect_obs_dim=140,
                                       expect_n_actions=12, schema="v2")


class ScenarioDistributionTests(unittest.TestCase):
    def test_mixed_sampler_holds_the_25_percent_catch_fraction(self):
        import random
        import battle_train as bt
        combat = lambda rng: {"our_party": [], "enemy_party": [], "is_trainer": False}
        s = bt.MixedObjectiveScenarioSampler(combat, catch_fraction=0.25)
        rng = random.Random(0)
        for _ in range(4000):
            s(rng)
        c = s.counts()
        self.assertAlmostEqual(c["catch_fraction_actual"], 0.25, delta=0.03)

    def test_fixed_catch_suite_has_a_trainer_and_a_no_ball_case(self):
        import battle_train as bt
        S = bt.fixed_catch_scenarios()
        self.assertGreaterEqual(len(S), 15)
        self.assertTrue(any(x["is_trainer"] for x in S))
        self.assertTrue(any(x["ball_inventory"] - x["ball_reserve"] <= 0 for x in S))


class BattleEpisodeSummaryTests(unittest.TestCase):
    def test_full_summary_carries_reward_total_but_nav_safe_one_does_not(self):
        from twoby2.battle_summary import (battle_episode_summary, summarize_battle,
                                           assert_navigation_safe)
        raw = {"outcome": "caught", "objective_mode": "catch",
               "catch_requested": True, "caught_species_id": 19,
               "target_species_id": 19, "is_new_species_this_run": True,
               "balls_used": 2, "turns": 4}
        comps = [("turn_cost", -0.04), ("ball_cost", -0.10), ("catch_success", 3.0)]
        full = battle_episode_summary(raw, reward_components=comps)
        self.assertEqual(full["outcome"], "caught")
        self.assertAlmostEqual(full["battle_reward_total"], 2.86, places=2)
        self.assertTrue(full["reward_sum_ok"])
        nav = summarize_battle(raw)
        assert_navigation_safe(nav)              # must not raise
        self.assertNotIn("battle_reward_total", nav)
        self.assertNotIn("reward_components", nav)
        self.assertTrue(nav["catch_requested"])

    def test_nav_reward_pays_the_species_term_only_when_requested(self):
        from twoby2.battle_summary import navigation_battle_reward
        req = {"outcome": "caught", "catch_requested": True, "catch_success": True,
               "caught_species_id": 19, "target_species_id": 19,
               "is_new_species_this_run": True, "party_hp_fraction_lost": 0.0}
        raw_catch = {**req, "catch_requested": False}
        self.assertAlmostEqual(navigation_battle_reward(req), 0.5, places=3)
        self.assertAlmostEqual(navigation_battle_reward(raw_catch), 0.0, places=3)
        dup = {**req, "is_new_species_this_run": False}
        self.assertLess(navigation_battle_reward(dup), 0.0)


class TelemetryCounterTests(unittest.TestCase):
    def test_battle_counters_expose_catch_rolling_section(self):
        from battle_train import BattleCounters
        c = BattleCounters()
        c.recent.append({"outcome": "caught", "reward": 2.8, "turns": 3,
                         "objective_mode": "catch", "catch_requested": True,
                         "catch_success": True})
        c.recent.append({"outcome": "win", "reward": 1.0, "turns": 2,
                         "objective_mode": "combat", "catch_requested": False,
                         "catch_success": False})
        c.catch_successes = 1
        c.balls_used = 2
        roll = c.rolling_dict()
        self.assertIn("catch", roll)
        self.assertEqual(roll["catch"]["catch_success_rate_when_requested"], 1.0)
        self.assertEqual(roll["catch"]["balls_per_successful_catch"], 2.0)
        self.assertIn("catch", roll["catch"]["objective_distribution"])


if __name__ == "__main__":
    unittest.main()
