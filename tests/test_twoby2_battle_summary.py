import math
import unittest

import twoby2.battle_summary as bs


GOOD_WORLD = {"map_group": 3, "map_id": 0, "x": 5, "y": 7,
              "story_flag_count": 12, "badges": 1, "party_hp_total": 40,
              "party_alive": 2, "respawn_map_group": 3, "respawn_map_id": 1}


class SummarizeTests(unittest.TestCase):
    def test_only_allowed_keys_survive(self):
        raw = {
            "outcome": "win", "party_hp_lost": 12, "party_hp_fraction_lost": 0.3,
            "pp_spent": 5, "items_used": 1, "own_faints": 0, "enemy_faints": 2,
            "turns": 6, "fled": False, "battle_kind": "trainer",
            "resulting_world_state": {**GOOD_WORLD, "per_turn_reward": 9,
                                      "macro_history": [1, 2]},
            "macro_actions": ["MOVE_1"], "per_turn_reward": [0.1],
            "battle_obs": [0, 1], "move_damage": 40, "q_values": [1, 2],
        }
        s = bs.summarize_battle(raw)
        bs.assert_navigation_safe(s)
        self.assertEqual(set(s) - bs.ALLOWED_SUMMARY_KEYS, set())
        self.assertEqual(s["outcome"], "win")
        self.assertEqual(set(s["resulting_world_state"]) - set(bs.WORLD_STATE_SCHEMA), set())
        self.assertNotIn("per_turn_reward", s["resulting_world_state"])


class RecursiveLeakTests(unittest.TestCase):
    def test_top_level_micro_combat_key_rejected(self):
        for k in ("per_turn_reward", "macro_action_history", "move_selection",
                  "q_values", "battle_observation"):
            with self.assertRaises(bs.BattleSummaryLeak):
                bs.assert_navigation_safe({"outcome": "win", k: 1})

    def test_nested_per_turn_reward_is_rejected(self):
        with self.assertRaises(bs.BattleSummaryLeak):
            bs.assert_navigation_safe(
                {"outcome": "win", "resulting_world_state": {"per_turn_reward": 999}})

    def test_deeply_nested_leak_is_rejected(self):
        with self.assertRaises(bs.BattleSummaryLeak):
            bs.assert_navigation_safe({
                "outcome": "win",
                "resulting_world_state": {"map_group": {"macro_action": [1, 2]}}})

    def test_world_state_non_schema_key_rejected(self):
        with self.assertRaises(bs.BattleSummaryLeak):
            bs.assert_navigation_safe({"outcome": "win",
                                       "resulting_world_state": {"secret_plan": 1}})

    def test_world_state_container_value_rejected(self):
        with self.assertRaises(bs.BattleSummaryLeak):
            bs.assert_navigation_safe({"outcome": "win",
                                       "resulting_world_state": {"map_group": [1]}})

    def test_non_dict_summary_rejected(self):
        with self.assertRaises(bs.BattleSummaryLeak):
            bs.assert_navigation_safe(["not", "a", "dict"])


class FailClosedNumberTests(unittest.TestCase):
    def test_nan_inf_string_junk_are_clamped_not_crashed(self):
        s = bs.summarize_battle({
            "outcome": "loss",
            "party_hp_lost": "twelve", "turns": float("nan"),
            "party_hp_fraction_lost": float("inf"),
            "resulting_world_state": {"map_group": "x", "x": float("nan"),
                                      "badges": -3},
        })
        self.assertEqual(s["party_hp_lost"], 0)
        self.assertEqual(s["turns"], 0)
        self.assertEqual(s["party_hp_fraction_lost"], 0.0)
        self.assertEqual(s["resulting_world_state"]["map_group"], 0)
        self.assertEqual(s["resulting_world_state"]["x"], 0)
        self.assertEqual(s["resulting_world_state"]["badges"], 0)
        # reward never crashes and is finite
        r = bs.navigation_battle_reward(s)
        self.assertTrue(math.isfinite(r))

    def test_reward_is_coarse_no_win_bonus_no_per_turn(self):
        win = bs.navigation_battle_reward(bs.summarize_battle(
            {"outcome": "win", "party_hp_fraction_lost": 0.0}))
        wipe = bs.navigation_battle_reward(bs.summarize_battle(
            {"outcome": "wipe", "party_hp_fraction_lost": 1.0}))
        self.assertLessEqual(win, 0.0)
        self.assertLess(wipe, win)

    def test_outcome_derived_when_missing(self):
        self.assertEqual(bs.summarize_battle({"fled": True})["outcome"], "fled")
        self.assertEqual(bs.summarize_battle({"party_alive": 0})["outcome"], "wipe")

    def test_wild_win_decay_ladder_caps_at_two(self):
        w = lambda i: bs.navigation_battle_reward(
            bs.summarize_battle({"outcome": "win", "battle_kind": "wild",
                                 "party_hp_fraction_lost": 0.0}),
            wild_win_index=i)
        self.assertAlmostEqual(w(0), 0.5)
        self.assertAlmostEqual(w(1), 0.2)
        self.assertAlmostEqual(w(2), 0.0)
        self.assertAlmostEqual(w(9), 0.0)
        # a trainer win never gets the wild ladder
        t = bs.navigation_battle_reward(
            bs.summarize_battle({"outcome": "win", "battle_kind": "trainer",
                                 "party_hp_fraction_lost": 0.0}),
            wild_win_index=0)
        self.assertAlmostEqual(t, 0.0)
        # the whole repeatable wild reward on a map (0.5+0.2) stays well under a
        # geographic checkpoint (PROVEN_EXIT_REWARD)
        import pokemon_env
        self.assertLess(sum(bs.WILD_WIN_LADDER),
                        pokemon_env.PokemonFireRedEnv.PROVEN_EXIT_REWARD)

    def test_unnecessary_flee_is_negative(self):
        r = bs.navigation_battle_reward(bs.summarize_battle(
            {"outcome": "fled", "party_hp_fraction_lost": 0.0}))
        self.assertLess(r, 0.0)


if __name__ == "__main__":
    unittest.main()
