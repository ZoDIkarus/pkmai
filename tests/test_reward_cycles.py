import unittest

from pokemon_env import (
    PokemonFireRedEnv,
    battle_hp_stagnation_update,
    battle_stagnation_penalty,
    short_cycle_repeats,
    stuck_loop_penalty,
)


class ShortCycleRepeatsTests(unittest.TestCase):
    def test_detects_repeating_backtrack_cycle(self):
        a = (3, 0, 10, 10)
        b = (3, 0, 11, 10)
        self.assertEqual(short_cycle_repeats([a, b, a, b, a, b]), 2)

    def test_ignores_single_necessary_backtrack(self):
        a = (3, 0, 10, 10)
        b = (3, 0, 11, 10)
        self.assertEqual(short_cycle_repeats([a, b, a]), 0)

    def test_ignores_non_repeating_route(self):
        path = [
            (3, 0, 10, 10),
            (3, 0, 11, 10),
            (3, 0, 12, 10),
            (3, 0, 12, 9),
            (3, 0, 13, 9),
            (3, 0, 14, 9),
        ]
        self.assertEqual(short_cycle_repeats(path), 0)

    def test_battle_stagnation_penalizes_only_sustained_unchanged_hp(self):
        self.assertEqual(battle_stagnation_penalty(15), 0.0)
        self.assertEqual(battle_stagnation_penalty(16), -0.5)
        self.assertEqual(battle_stagnation_penalty(17), 0.0)
        self.assertEqual(battle_stagnation_penalty(32), -0.5)

    def test_valid_enemy_hp_snapshots_stagnate_without_a_player_level_gate(self):
        snapshot = ((0, 19, 123, 10),)
        signature, reads, penalty = battle_hp_stagnation_update(
            snapshot,
            15,
            snapshot,
        )

        self.assertEqual(signature, snapshot)
        self.assertEqual(reads, 16)
        self.assertEqual(penalty, -0.5)


class V17RewardTuningTests(unittest.TestCase):
    def test_local_ten_worker_curriculum_revalidates_unmeasured_early_stages(self):
        env = object.__new__(PokemonFireRedEnv)
        env.rank = 8
        env.agent_count = 10
        env.saved_milestones = {
            "intro_complete", "stairs_down", "left_house", "starter", "progress_3",
        }
        env.full_chain_ready = False
        self.assertEqual(PokemonFireRedEnv._agent_role(env)[0], "stairs")

    def test_unmeasured_early_stages_hold_the_frontier_before_progress_resume(self):
        env = object.__new__(PokemonFireRedEnv)
        env.rank = 8
        env.agent_count = 10
        env.saved_milestones = {
            "intro_complete", "stairs_down", "left_house", "starter", "progress_3",
        }
        env._discover_saved_milestones = lambda: list(env.saved_milestones)
        env._champion_full_starter_ready = lambda: False
        self.assertEqual(PokemonFireRedEnv._choose_episode_start(env), "intro_complete")

    def test_frontier_scout_uses_the_shared_full_policy_context(self):
        env = object.__new__(PokemonFireRedEnv)
        env.training_objective = "scout"
        self.assertEqual(PokemonFireRedEnv._policy_objective(env), "full")

    def test_frontier_scout_gets_a_longer_stall_budget(self):
        env = object.__new__(PokemonFireRedEnv)
        env.training_objective = "scout"
        self.assertEqual(
            PokemonFireRedEnv._progress_stall_limit(env),
            PokemonFireRedEnv.SCOUT_STALL_TIMEOUT,
        )

    def test_gameplay_time_and_combat_rewards_favor_progress(self):
        self.assertEqual(PokemonFireRedEnv.GAMEPLAY_STEP_COST, -0.001)
        self.assertEqual(PokemonFireRedEnv.ENEMY_DAMAGE_REWARD_PER_HP, 0.08)
        self.assertEqual(PokemonFireRedEnv.ENEMY_FAINT_REWARD, 0.0)
        self.assertEqual(PokemonFireRedEnv.NEW_TRANSITION_REWARD, 100.0)
        self.assertEqual(PokemonFireRedEnv.LEVEL_GAIN_REWARD, 10.0)

    def test_wipe_cooldown_and_warp_keys_are_safe_and_bidirectional(self):
        env = object.__new__(PokemonFireRedEnv)
        env.wipe_active = False
        env.total_steps = 50
        env.episode_best_stage = 4
        env.best_pokecenter_heal_stage = 3
        env.last_badges = 1
        events = []
        self.assertEqual(PokemonFireRedEnv._record_party_wipe(env, events), -100.0)
        self.assertEqual(env._post_wipe_reward_cooldown_until, 90)
        self.assertEqual(events, ["party_wipe:-100"])
        self.assertTrue(env.post_wipe_recovery)
        self.assertEqual(env.pre_wipe_best_stage, 4)
        self.assertEqual(env.pre_wipe_best_center_stage, 3)
        self.assertEqual(env.pre_wipe_badges, 1)
        self.assertEqual(
            PokemonFireRedEnv._warp_pair_key(3, 0, 5, 4),
            PokemonFireRedEnv._warp_pair_key(5, 4, 3, 0),
        )

    def test_tile_ladder_caps_each_map_but_keeps_a_small_rest_value(self):
        env = object.__new__(PokemonFireRedEnv)
        env._episode_tiles_by_map = {(3, 19): set(range(20))}
        self.assertAlmostEqual(
            PokemonFireRedEnv._tile_reward(env, 3, 19, 20, 1),
            0.6,
        )
        self.assertEqual(
            PokemonFireRedEnv._tile_reward(env, 5, 4, 0, 1),
            2.0,
        )

    def test_scout_backtracking_does_not_pay_below_its_start_stage(self):
        env = object.__new__(PokemonFireRedEnv)
        env.training_objective = "scout"
        env.scout_start_stage = 3
        self.assertFalse(PokemonFireRedEnv._can_reward_exploration(env, 2))
        self.assertTrue(PokemonFireRedEnv._can_reward_exploration(env, 4))

    def test_bank_four_interiors_never_receive_a_city_building_claim(self):
        self.assertEqual(PokemonFireRedEnv.CITY_BUILDING_BANKS, {5, 6})
        self.assertIsNone(PokemonFireRedEnv._city_building_claim_key(4, 3))
        self.assertEqual(
            PokemonFireRedEnv._city_building_claim_key(5, 4), "building_5_4"
        )
        self.assertEqual(
            PokemonFireRedEnv._city_building_claim_key(6, 7), "building_6_7"
        )

    def test_bank_four_interior_tiles_use_the_lower_interior_cap(self):
        env = object.__new__(PokemonFireRedEnv)
        env._episode_tiles_by_map = {(4, 3): set(range(15))}
        self.assertAlmostEqual(
            PokemonFireRedEnv._tile_reward(env, 4, 3, 15, 0), 0.04
        )

    def test_reaching_the_pre_wipe_front_pays_once_and_ends_recovery(self):
        env = object.__new__(PokemonFireRedEnv)
        env.post_wipe_recovery = True
        env.pre_wipe_best_stage = 4
        env.best_pokecenter_heal_stage = 3
        env.pre_wipe_best_center_stage = 3
        env.last_badges = 1
        env.pre_wipe_badges = 1
        events = []
        self.assertEqual(
            PokemonFireRedEnv._complete_post_wipe_recovery(env, 4, events), 300.0
        )
        self.assertFalse(env.post_wipe_recovery)
        self.assertEqual(events, ["post_wipe_front_recovered:+300"])
        self.assertEqual(
            PokemonFireRedEnv._complete_post_wipe_recovery(env, 4, events), 0.0
        )

    def test_geographic_stages_cover_the_route_to_pewter(self):
        self.assertEqual(PokemonFireRedEnv._tile_stage(3, 0), 1)
        self.assertEqual(PokemonFireRedEnv._tile_stage(3, 19), 2)
        self.assertEqual(PokemonFireRedEnv._tile_stage(3, 1), 3)
        self.assertEqual(PokemonFireRedEnv._tile_stage(3, 20), 4)
        self.assertEqual(PokemonFireRedEnv._tile_stage(1, 0), 5)
        self.assertEqual(PokemonFireRedEnv._tile_stage(3, 2), 6)

    def test_pewter_arrival_with_pikachu_is_once_per_episode(self):
        env = object.__new__(PokemonFireRedEnv)
        env.episode_pewter_reached = False
        env.episode_pewter_with_pikachu_rewarded = False
        events = []
        party = [{"species_id": 25}]
        self.assertEqual(
            PokemonFireRedEnv._pewter_arrival_reward(env, 6, party, events), 300.0
        )
        self.assertEqual(events, ["pewter_with_pikachu:+300"])
        self.assertEqual(
            PokemonFireRedEnv._pewter_arrival_reward(env, 6, party, events), 0.0
        )

    def test_first_pewter_trainer_battle_is_a_once_per_episode_milestone(self):
        env = object.__new__(PokemonFireRedEnv)
        env.episode_brock_battle_started = False
        events = []
        self.assertEqual(
            PokemonFireRedEnv._brock_battle_start_reward(env, 6, True, events),
            500.0,
        )
        self.assertEqual(events, ["brock_battle_start:+500"])
        self.assertEqual(
            PokemonFireRedEnv._brock_battle_start_reward(env, 6, True, events),
            0.0,
        )

    def test_stuck_penalty_grows_continuously_without_a_discontinuity(self):
        self.assertEqual(stuck_loop_penalty(59), 0.0)
        self.assertEqual(stuck_loop_penalty(60), -0.001)
        self.assertEqual(stuck_loop_penalty(180), -0.121)
        self.assertEqual(stuck_loop_penalty(400), -0.341)


if __name__ == "__main__":
    unittest.main()
