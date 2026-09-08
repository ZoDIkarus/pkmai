"""Catch-v2 — the spec §17 pflichttests, driven against the SimulatedBattleDriver
(no RAM / emulator needed)."""
import unittest

import battle_types as bt
import battle_catch
import catch_planner
from battle_env import (BattleEnv, SimulatedBattleDriver, BattleRewardConfig,
                        ALL_MACROS, OBS_DIM, OBS_SCHEMA, REWARD_SCHEMA, _mon, _mv,
                        _reward_components_v2)
from battle_executor import (ALL_MACROS_V1, ACTIONS_SCHEMA_V2, catch_precheck,
                             CATCH_ACTION, MacroExecutor)

CI = ALL_MACROS.index(CATCH_ACTION)
MI = 0   # MOVE_1


def _enemy(cur=18, mx=18, catch_rate=45, level=3, status=0, sid=19):
    m = _mon((bt.TYPE_NORMAL,), level=level, cur_hp=cur, max_hp=mx, atk=15, spe=20,
             moves=[_mv(bt.TYPE_NORMAL, 20, mid=9)])
    m["species_id"] = sid
    m["catch_rate"] = catch_rate
    m["status"] = status
    return m


def _party(n=1, hp=55):
    return [_mon((bt.TYPE_WATER,), slot=i, level=18, cur_hp=hp, max_hp=hp,
                 spa=55, spe=60, moves=[_mv(bt.TYPE_WATER, 60, mid=1)])
            for i in range(n)]


def _sc(mode="catch", **kw):
    s = {"our_party": _party(kw.pop("party_n", 1)),
         "enemy_party": [_enemy(**{k: kw.pop(k) for k in
                                   ("cur", "mx", "catch_rate", "level", "status", "sid")
                                   if k in kw})],
         "is_trainer": kw.pop("is_trainer", False),
         "can_escape": kw.pop("can_escape", True),
         "objective_mode": mode, "target_species_id": kw.pop("target_species_id", 19),
         "ball_inventory": kw.pop("ball_inventory", 5),
         "ball_reserve": kw.pop("ball_reserve", 1),
         "party_has_space": kw.pop("party_has_space", True),
         "pc_capture_supported": kw.pop("pc_capture_supported", False),
         "is_new_species_this_run": kw.pop("is_new_species_this_run", True),
         "catch_seed": kw.pop("catch_seed", 1)}
    s.update(kw)
    return s


def _play(scenario, actions, seed=0):
    e = BattleEnv(SimulatedBattleDriver(), max_turns=25)
    obs, info = e.reset(seed=seed, options={"scenario": scenario})
    out = {"obj": info["objective"], "steps": [], "ep_reward": 0.0}
    for a in actions:
        if e._state.get("done"):
            break
        obs, r, term, trunc, i = e.step(a)
        out["ep_reward"] += r
        out["steps"].append({"r": r, "comps": i.get("reward_components"),
                             "outcome": i.get("outcome"), "info": i, "mask": list(obs["action_mask"])})
        if term or trunc:
            break
    return out


class SchemaTests(unittest.TestCase):
    def test_v2_schema_ids_and_dims(self):
        self.assertEqual(OBS_SCHEMA, "battle_obs_v2_catch")
        self.assertEqual(REWARD_SCHEMA, "battle_reward_v2_catch")
        self.assertEqual(ACTIONS_SCHEMA_V2, "battle_actions_v2_catch")
        self.assertEqual(len(ALL_MACROS), len(ALL_MACROS_V1) + 1)
        self.assertEqual(ALL_MACROS[-1], "CATCH")
        self.assertEqual(OBS_DIM, 140)   # v1 116 + 12*2 catch fields


class ObjectiveTests(unittest.TestCase):
    def test_combat_scenario_does_not_request_catch(self):        # §17.1
        r = _play(_sc("combat"), [MI, MI, MI, MI])
        self.assertFalse(r["obj"]["catch_requested"])
        self.assertTrue(all(s["mask"][CI] == 0 for s in r["steps"]))

    def test_catch_scenario_allows_catch(self):                   # §17.2
        r = _play(_sc(catch_rate=255), [MI])
        self.assertTrue(r["obj"]["catch_requested"])
        self.assertEqual(r["steps"][0]["mask"][CI], 1)

    def test_planner_never_requests_on_a_non_allowlisted_map(self):
        o = catch_planner.plan_battle_objective(dict(
            area="viridian_forest", battle_confirmed_wild=True,
            catch_execution_ready=True, enemy_species_known=True,
            enemy_species_id=10, usable_ball_count=5, party_has_space=True,
            caught_species_this_run=[]))
        self.assertFalse(o["catch_requested"])
        self.assertIn("not_allowlisted", o["catch_reason"])

    def test_planner_unknowns_force_combat(self):
        for miss in ("enemy_species_id", "usable_ball_count", "party_has_space"):
            ctx = dict(area="route1", battle_confirmed_wild=True,
                       catch_execution_ready=True, enemy_species_known=True,
                       enemy_species_id=10, usable_ball_count=5,
                       party_has_space=True, caught_species_this_run=[])
            ctx.pop(miss)
            if miss == "enemy_species_id":
                ctx["enemy_species_known"] = False
            o = catch_planner.plan_battle_objective(ctx)
            self.assertFalse(o["catch_requested"], miss)


class PrecheckTests(unittest.TestCase):
    def test_trainer_catch_is_blocked_before_buttons(self):        # §17.3
        r = _play(_sc(catch_rate=255, is_trainer=True, can_escape=False), [CI])
        self.assertEqual(r["steps"][0]["mask"][CI], 0)
        self.assertTrue(r["steps"][0]["info"].get("invalid"))
        self.assertEqual(r["steps"][0]["r"],
                         BattleRewardConfig.TURN_COST + BattleRewardConfig.INVALID_ACTION)

    def test_no_ball_blocks_catch(self):                           # §17.4
        r = _play(_sc(catch_rate=255, ball_inventory=1, ball_reserve=1), [CI])
        self.assertEqual(r["steps"][0]["mask"][CI], 0)

    def test_reserve_ball_is_never_spent(self):                    # §17.5
        # 2 balls, reserve 1 -> exactly one catch attempt possible
        sc = _sc(catch_rate=1, ball_inventory=2, ball_reserve=1, catch_seed=5)
        r = _play(sc, [CI, CI, CI, MI])
        attempts = sum(1 for s in r["steps"] if s["info"].get("catch_attempted"))
        self.assertLessEqual(attempts, 1)

    def test_full_party_without_pc_support_blocks_catch(self):     # §17.6
        r = _play(_sc(catch_rate=255, party_n=6, party_has_space=False,
                      pc_capture_supported=False), [CI])
        self.assertEqual(r["steps"][0]["mask"][CI], 0)

    def test_full_party_with_pc_support_allows_catch(self):
        r = _play(_sc(catch_rate=255, party_n=6, party_has_space=False,
                      pc_capture_supported=True), [CI])
        self.assertEqual(r["steps"][0]["mask"][CI], 1)


class OutcomeTests(unittest.TestCase):
    def test_confirmed_catch_gives_caught_and_no_ko_no_win(self):  # §17.9/12/13
        r = _play(_sc(catch_rate=255, catch_seed=3), [CI])
        s = r["steps"][0]
        self.assertEqual(s["outcome"], "caught")
        names = [n for n, _ in s["comps"]]
        self.assertIn("catch_success", names)
        self.assertNotIn("enemy_ko", names)
        self.assertNotIn("battle_win", names)

    def test_broke_free_continues_the_battle(self):                # §17.8/17
        r = _play(_sc(catch_rate=1, cur=18, mx=18, catch_seed=9), [CI, MI])
        s0 = r["steps"][0]
        self.assertEqual(s0["info"].get("catch_outcome"), "broke_free")
        self.assertFalse(s0["info"].get("outcome"))     # not terminal
        names = [n for n, _ in s0["comps"]]
        self.assertIn("failed_catch", names)
        self.assertIn("ball_cost", names)
        self.assertAlmostEqual(dict(s0["comps"])["failed_catch"],
                               BattleRewardConfig.FAILED_CATCH)

    def test_target_ko_in_catch_mode_is_negative(self):            # §17.14
        # low-HP target, strong mon -> MOVE_1 KOs it
        r = _play(_sc(catch_rate=45, cur=3, mx=45), [MI])
        names = [n for n, _ in r["steps"][0]["comps"]]
        self.assertIn("catch_target_ko", names)
        self.assertNotIn("enemy_ko", names)
        self.assertNotIn("battle_win", names)
        self.assertLess(r["steps"][0]["r"], 0)

    def test_normal_win_pays_no_catch_reward(self):                # §17.15
        r = _play(_sc("combat", cur=3, mx=45), [MI])
        names = [n for n, _ in r["steps"][0]["comps"]]
        self.assertIn("enemy_ko", names)
        self.assertNotIn("catch_success", names)
        self.assertNotIn("catch_target_ko", names)

    def test_catch_in_combat_objective_is_negative(self):          # §17.16
        r = _play(_sc("combat", catch_rate=255), [CI])
        # combat objective -> CATCH not in the mask -> invalid; if the policy
        # still forced it, unrequested_catch would fire. Here it's masked out.
        self.assertEqual(r["steps"][0]["mask"][CI], 0)
        # force it through _reward_components_v2 directly
        comps = _reward_components_v2(
            {"objective_mode": "combat", "catch_requested": False},
            {"catch_attempted": True, "balls_used": 1, "unrequested_catch": True})
        self.assertIn("unrequested_catch", [n for n, _ in comps])
        self.assertLess(sum(a for _, a in comps), 0)

    def test_damage_shaping_stops_below_the_safe_catch_window(self):   # §9B
        # high catch rate -> wide window -> a big hit that drops enemy into the
        # window pays no positive damage reward
        r = _play(_sc(catch_rate=200, cur=18, mx=18, catch_seed=99), [MI])
        names = [n for n, _ in r["steps"][0]["comps"]]
        self.assertNotIn("damage", names)


class RewardConsistencyTests(unittest.TestCase):
    def test_components_sum_exactly_every_step(self):              # §17.18
        r = _play(_sc(catch_rate=1, catch_seed=2), [MI, CI, MI, MI, MI])
        for s in r["steps"]:
            self.assertAlmostEqual(s["r"], sum(a for _, a in s["comps"]), places=6)

    def test_episode_reward_is_the_sum_of_step_rewards(self):      # §17.19
        r = _play(_sc(catch_rate=1, catch_seed=4), [MI, CI, MI, MI])
        self.assertAlmostEqual(r["ep_reward"],
                               sum(s["r"] for s in r["steps"]), places=6)

    def test_catch_success_not_much_higher_than_a_clean_combat_win(self):
        self.assertLessEqual(BattleRewardConfig.CATCH_SUCCESS,
                             BattleRewardConfig.ENEMY_KO
                             + BattleRewardConfig.BATTLE_WIN)


class MigrationTests(unittest.TestCase):
    def test_wrong_obs_dim_is_refused_not_reshaped(self):          # §17.29
        e = BattleEnv(SimulatedBattleDriver())
        e.reset(options={"scenario": _sc()})
        orig = e._obs_dim
        try:
            e._obs_dim = orig + 5           # simulate a mismatched policy space
            with self.assertRaises(RuntimeError):
                e._obs()
        finally:
            e._obs_dim = orig


if __name__ == "__main__":
    unittest.main()
