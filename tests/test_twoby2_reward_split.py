import unittest

import twoby2.reward_split as rs


MASTER_SNAPSHOT = {
    "map_group": 4, "map_id": 3, "x": 6, "y": 4,
    "party_species": [7], "party_levels": [5], "party_total_xp": 135,
    "story_flags": ["got_parcel", "delivered_parcel", "has_pokedex"],
    "badges": 0, "pokedex_owned": [1, 4, 7], "parcel_delivered": True,
    "inventory_item_ids": [13, 13, 4], "money": 3000,
}


class ResetBaselineTests(unittest.TestCase):
    def test_initial_state_pays_zero(self):
        b = rs.ResetBaseline.from_master_load(MASTER_SNAPSHOT)
        # every baseline field is "initial" -> not rewardable
        self.assertTrue(b.is_initial("party_species", [7]))
        self.assertTrue(b.is_initial("badges", 0))
        self.assertTrue(b.is_initial("story_flags",
                                     ["got_parcel", "delivered_parcel", "has_pokedex"]))
        self.assertFalse(b.moved_from_start(MASTER_SNAPSHOT))
        self.assertEqual(b.newly_owned_items([13, 13, 4]), [])
        self.assertEqual(b.newly_set_story_flags(MASTER_SNAPSHOT["story_flags"]), [])
        self.assertEqual(b.gained_badges(0), 0)

    def test_only_post_baseline_changes_count(self):
        b = rs.ResetBaseline.from_master_load(MASTER_SNAPSHOT)
        self.assertEqual(b.newly_owned_items([13, 13, 4, 26]), [26])
        self.assertEqual(b.newly_set_story_flags(
            MASTER_SNAPSHOT["story_flags"] + ["beat_brock"]), ["beat_brock"])
        self.assertEqual(b.gained_badges(1), 1)
        self.assertTrue(b.moved_from_start({**MASTER_SNAPSHOT, "map_id": 19}))

    def test_serialisable(self):
        import json
        json.dumps(rs.ResetBaseline.from_master_load(MASTER_SNAPSHOT).to_dict())


class ChannelSeparationTests(unittest.TestCase):
    def test_navigation_reward_rejects_battle_micro_terms(self):
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"new_coord": 0.3, "damage_per_turn": 1.0})
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"battle_win": 1.0})
        with self.assertRaises(rs.RewardChannelError):
            rs.navigation_reward({"fighter_bonus": 5.0})

    def test_battle_reward_rejects_navigation_terms(self):
        with self.assertRaises(rs.RewardChannelError):
            rs.battle_reward({"enemy_ko": 1.0, "tile": 0.05})
        with self.assertRaises(rs.RewardChannelError):
            rs.battle_reward({"world_stage": 1.0})

    def test_clean_channels_sum(self):
        self.assertAlmostEqual(
            rs.navigation_reward({"new_coord": 0.3, "new_map": 50.0,
                                  "wipe": -3.0}), 47.3)
        self.assertAlmostEqual(
            rs.battle_reward({"damage_dealt": 0.5, "enemy_ko": 1.0,
                              "battle_win": 3.0}), 4.5)

    def test_no_double_count_of_a_battle_win(self):
        with self.assertRaises(rs.RewardChannelError):
            rs.no_double_count({"battle_win": 1.0}, {"battle_win": 3.0})
        # allowed: battle brain gets combat, navigation gets only strategy
        self.assertTrue(rs.no_double_count(
            {"wipe": -3.0, "battle_triggered_story_change": 10.0},
            {"damage_dealt": 0.5, "enemy_ko": 1.0, "wipe": -3.0}))

    def test_schemas_are_versioned_separately(self):
        self.assertNotEqual(rs.NAV_REWARD_SCHEMA, rs.BATTLE_REWARD_SCHEMA)


if __name__ == "__main__":
    unittest.main()
