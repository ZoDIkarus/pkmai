import unittest
import json
import os
import tempfile
from collections import Counter
from unittest.mock import patch

import pokemon_env
from pokemon_env import PokemonFireRedEnv
from train import MilestoneCheckpointCallback


def bare_env(**overrides):
    env = object.__new__(PokemonFireRedEnv)
    env.visited_maps = set()
    env.last_badges = 0
    env.viridian_mart_scene = 0
    env.viridian_old_man_scene = 0
    env.pallet_oaks_lab_scene = 0
    for key, value in overrides.items():
        setattr(env, key, value)
    return env


class WorldStageTests(unittest.TestCase):
    def test_step_cost_is_tiny_relative_to_story_rewards(self):
        self.assertLess(PokemonFireRedEnv.GAMEPLAY_STEP_COST, 0)
        self.assertGreater(PokemonFireRedEnv.GAMEPLAY_STEP_COST, -0.01)
        self.assertGreater(
            PokemonFireRedEnv.STARTER_REWARD,
            abs(PokemonFireRedEnv.GAMEPLAY_STEP_COST) *
            PokemonFireRedEnv.STARTER_SPECIALIST_TIMEOUT * 20,
        )

    def test_arbitrary_interiors_never_increase_stage(self):
        env = bare_env(visited_maps={(3, 0), (4, 0), (4, 1), (5, 0)})
        self.assertEqual(env._world_stage(), 1)

    def test_fire_red_story_chain_is_explicit(self):
        env = bare_env(visited_maps={(3, 0), (3, 19), (3, 1)})
        self.assertEqual(env._world_stage(), 3)
        env.viridian_mart_scene = 1
        self.assertEqual(env._world_stage(), 4)
        env.pallet_oaks_lab_scene = 6
        self.assertEqual(env._world_stage(), 5)
        env.visited_maps.add((3, 20))
        self.assertEqual(env._world_stage(), 6)
        env.visited_maps.add((1, 0))
        self.assertEqual(env._world_stage(), 7)
        env.visited_maps.add((3, 2))
        self.assertEqual(env._world_stage(), 8)
        env.last_badges = 1
        self.assertEqual(env._world_stage(), 9)

    def test_story_checkpoint_requires_matching_map_and_state(self):
        env = bare_env(viridian_mart_scene=1)
        self.assertEqual(env._stage_at_current_location(5, 3), 4)
        self.assertNotEqual(env._stage_at_current_location(5, 0), 4)
        env.pallet_oaks_lab_scene = 6
        self.assertEqual(env._stage_at_current_location(4, 3), 5)

    def test_faster_full_run_wins_only_as_same_quality_tie_breaker(self):
        quality = {
            "max_badges": 0, "max_stage": 5, "badge_episodes": 0,
            "full_starter_permille": 500, "full_exit_permille": 500,
            "full_stairs_permille": 500, "full_intro_permille": 1000,
        }
        slow = dict(quality, full_best_stage_steps=18000)
        fast = dict(quality, full_best_stage_steps=9000)
        deeper = dict(slow, max_stage=6)
        self.assertGreater(
            MilestoneCheckpointCallback._score(fast),
            MilestoneCheckpointCallback._score(slow),
        )
        self.assertGreater(
            MilestoneCheckpointCallback._score(deeper),
            MilestoneCheckpointCallback._score(fast),
        )


class CurriculumResumeFlagTests(unittest.TestCase):
    def resumed(self, start):
        env = bare_env(
            episode_start=start,
            episode_milestone_steps={},
            stairs_down_rewarded=False,
            left_house_rewarded=False,
            left_house_confirmed=False,
            starter_outdoor_rewarded=False,
            outdoor_confirm_reads=0,
        )
        env._apply_curriculum_resume_flags()
        return env

    def test_stage_resume_marks_house_left_so_early_failsafe_is_off(self):
        # stage_4/5 liegen indoor (Vertania-Markt / Eichs Labor) - ohne diese
        # Flags kappt der early-house-Timeout den Resume-Run.
        for start in ("stage_2", "stage_4", "stage_5"):
            env = self.resumed(start)
            self.assertTrue(env.left_house_confirmed, start)
            self.assertTrue(env.stairs_down_rewarded, start)
            self.assertTrue(env.starter_outdoor_rewarded, start)
            self.assertEqual(env.outdoor_confirm_reads, env.OUTDOOR_CONFIRM_READS)

    def test_early_states_do_not_skip_the_house(self):
        for start in ("intro_complete", "stairs_down"):
            env = self.resumed(start)
            self.assertFalse(env.left_house_confirmed, start)

    def test_starter_outdoor_still_suppresses_repeat_reward(self):
        env = self.resumed("starter_outdoor")
        self.assertTrue(env.starter_outdoor_rewarded)

    def test_only_real_german_story_map_pairs_are_confirmed_warps(self):
        self.assertTrue(PokemonFireRedEnv._valid_confirmed_story_warp(
            "stairs", (4, 1, 6, 6, 4, 0, 10, 2)
        ))
        self.assertTrue(PokemonFireRedEnv._valid_confirmed_story_warp(
            "exit", (4, 0, 10, 2, 3, 0, 6, 7)
        ))
        self.assertFalse(PokemonFireRedEnv._valid_confirmed_story_warp(
            "stairs", (3, 1, 22, 14, 5, 3, 4, 7)
        ))


class RoleAllocationTests(unittest.TestCase):
    def roles_for_scores(self, scores, stage=1):
        roles = Counter()
        for rank in range(32):
            env = bare_env(
                rank=rank,
                n_envs=32,
                shared_progress={"max_world_stage": stage},
            )
            env._skill_vault_scores = lambda scores=scores: dict(scores)
            role, _ = env._agent_role()
            roles[role] += 1
        return roles

    def roles_at(self, stage):
        return self.roles_for_scores({
            "intro": 1000,
            "stairs": 1000,
            "exit": 1000,
            "starter": 1000,
            "progress": 1000,
        }, stage=stage)

    def test_starter_bootcamp_uses_most_of_a_32_env_fleet(self):
        roles = self.roles_for_scores({
            "intro": 1000,
            "stairs": 1000,
            "exit": 1000,
            "starter": 0,
            "progress": 0,
        })
        self.assertEqual(roles, Counter({
            "starter": 22,
            "exit": 3,
            "progress": 3,
            "stairs": 2,
            "intro": 1,
            "full": 1,
        }))

    def test_before_parcel_delivery_world_push_dominates(self):
        roles = self.roles_at(3)
        self.assertEqual(roles["progress"], 22)
        self.assertEqual(roles["battle"], 0)

    def test_after_forest_badge_specialists_activate(self):
        roles = self.roles_at(7)
        self.assertEqual(roles["badge"], 4)
        self.assertEqual(sum(roles.values()), 32)

    def test_route_two_prioritizes_world_push_and_full_validation(self):
        roles = self.roles_at(6)
        self.assertEqual(roles, Counter({
            "progress": 14,
            "full": 8,
            "starter": 3,
            "battle": 2,
            "level": 2,
            "intro": 1,
            "stairs": 1,
            "exit": 1,
        }))


class SpecialistStartTests(unittest.TestCase):
    def specialist_start(self, saved):
        env = bare_env(saved_milestones=list(saved))
        env._discover_saved_milestones = lambda: list(saved)
        env._champion_full_starter_ready = lambda: False
        env._agent_role = lambda: ("battle", "Battle")
        return env._choose_episode_start()

    def test_battle_ready_is_preferred_over_indoor_story_front(self):
        self.assertEqual(
            self.specialist_start({"battle_ready", "stage_5", "starter_outdoor"}),
            "battle_ready",
        )

    def test_squirtle_battle_ready_beats_legacy_charmander_state(self):
        self.assertEqual(
            self.specialist_start({
                "squirtle_battle_ready", "battle_ready", "stage_5"
            }),
            "squirtle_battle_ready",
        )

    def test_safe_outdoor_start_beats_indoor_story_front(self):
        self.assertEqual(
            self.specialist_start({"stage_5", "starter_outdoor"}),
            "starter_outdoor",
        )


class TrainerStatusOwnershipTests(unittest.TestCase):
    def test_stale_trainer_process_cannot_hold_new_workers_in_old_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "skill_vault_scores.json"), "w") as f:
                json.dump({"starter": 1000}, f)
            with open(os.path.join(tmp, "trainer_status.json"), "w") as f:
                json.dump({
                    "trainer_pid": 111,
                    "effective_skill_scores": {"starter": 11},
                }, f)
            env = bare_env()
            with (
                patch.object(pokemon_env, "RUNTIME_DIR", tmp),
                patch("pokemon_env.os.getppid", return_value=222),
            ):
                self.assertEqual(env._skill_vault_scores()["starter"], 1000)
            with (
                patch.object(pokemon_env, "RUNTIME_DIR", tmp),
                patch("pokemon_env.os.getppid", return_value=111),
            ):
                self.assertEqual(env._skill_vault_scores()["starter"], 11)


if __name__ == "__main__":
    unittest.main()
