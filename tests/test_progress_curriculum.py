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
    env.has_target_starter = False
    env.viridian_mart_scene = 0
    env.viridian_old_man_scene = 0
    env.pallet_oaks_lab_scene = 0
    env.parcel_obtained_confirmed = False
    env.parcel_delivered_confirmed = False
    env.parcel_obtained_confirm_reads = 0
    env.parcel_delivered_confirm_reads = 0
    for key, value in overrides.items():
        setattr(env, key, value)
    return env


class WorldStageTests(unittest.TestCase):
    def test_time_has_a_tiny_cost_but_new_tiles_reward_exploration(self):
        self.assertEqual(PokemonFireRedEnv.INTRO_STEP_COST, 0.0)
        # V17.3: winzige, aber von Null verschiedene Zeitgebuehr statt
        # strikt neutraler Bewegung.
        self.assertEqual(PokemonFireRedEnv.GAMEPLAY_STEP_COST, -0.005)
        # V17.4: Kanten geben nie mehr Reward (farmbar durch Loop-Ablaufen
        # bekannter Kurzstrecken bei jedem Episodenstart) - der
        # Explorationsanreiz sitzt jetzt auf der einzelnen Kachel.
        self.assertEqual(PokemonFireRedEnv.NEW_EDGE_REWARD, 0.0)
        self.assertEqual(PokemonFireRedEnv.EPISODE_EDGE_REWARD, 0.0)
        self.assertEqual(PokemonFireRedEnv.EPISODE_TILE_REWARD, 0.0)
        # V18: Kachel-Erstfund zahlt PRO LAUF (seen_coords, jede Episode neu),
        # handgesetzte Leiter pro Story-Aussenmap, Innenraeume nach Stadt-Bank,
        # plus ein fleet-weit einmaliger +GLOBAL_NEW_TILE_BONUS obendrauf.
        self.assertEqual(PokemonFireRedEnv.TILE_REWARD_BY_STAGE[1], 2.0)   # Pallet
        self.assertEqual(PokemonFireRedEnv.TILE_REWARD_BY_STAGE[2], 3.0)   # Route 1
        self.assertEqual(PokemonFireRedEnv.TILE_REWARD_BY_STAGE[3], 4.0)   # Viridian
        self.assertEqual(PokemonFireRedEnv.TILE_REWARD_BY_STAGE[6], 6.0)   # Pewter
        self.assertEqual(PokemonFireRedEnv.GLOBAL_NEW_TILE_BONUS, 1.0)
        # Innenraeume: Pallet-Haeuser 1 < Vertania 2 < Marmoria 3, jeweils
        # unter der zugehoerigen Stadt (2/4/6).
        for bank, city_stage in ((4, 1), (5, 3), (6, 6)):
            self.assertLess(PokemonFireRedEnv.INTERIOR_TILE_REWARD_BY_BANK[bank],
                            PokemonFireRedEnv.TILE_REWARD_BY_STAGE[city_stage])
        e = bare_env()
        self.assertEqual(e._current_world_stage(3, 0), 1)   # Pallet
        self.assertEqual(e._current_world_stage(3, 19), 2)  # Route 1
        self.assertEqual(e._current_world_stage(3, 1), 3)   # Viridian
        self.assertEqual(e._current_world_stage(4, 3), 0)   # Oak's lab (interior)

    def test_arbitrary_interiors_never_increase_stage(self):
        env = bare_env(visited_maps={(3, 0), (4, 0), (4, 1), (5, 0)})
        self.assertEqual(env._world_stage(), 1)

    def test_story_flags_do_not_change_geographic_stages(self):
        env = bare_env(visited_maps={(3, 0)}, has_target_starter=True,
                       parcel_obtained_confirmed=True, parcel_delivered_confirmed=True)
        for location, stage in [((3,0),1), ((3,19),2), ((3,1),3),
                                ((3,20),4), ((1,0),5), ((3,2),6)]:
            env.visited_maps.add(location)
            self.assertEqual(env._world_stage(), stage)
            self.assertEqual(env._stage_at_current_location(*location), stage)
        for location in [(4,3), (5,3), (4,0), (4,1)]:
            self.assertEqual(env._stage_at_current_location(*location), 0)
        env.last_badges = 1
        self.assertEqual(env._world_stage(), 6)

    def test_earlier_checkpoint_remains_eligible_after_forest(self):
        env = bare_env(visited_maps={(1,0), (3,19)})
        self.assertEqual(env._world_stage(), 5)
        self.assertEqual(env._stage_at_current_location(3,19), 2)
        self.assertEqual(env._meta_checkpoint_stage({'bank':4, 'map':3, 'stage':5}), 0)

    def test_story_confirmation_requires_map_order_and_three_reads(self):
        env = bare_env(has_target_starter=True)

        wrong_map = {
            "map_bank": 3,
            "map_id": 1,
            "viridian_mart_scene": 1,
            "pallet_oaks_lab_scene": 0,
            "viridian_old_man_scene": 0,
        }
        for _ in range(5):
            env._update_story_state_from_loc(wrong_map)
        self.assertFalse(env.parcel_obtained_confirmed)

        mart = dict(wrong_map, map_bank=5, map_id=3)
        env._update_story_state_from_loc(mart)
        env._update_story_state_from_loc(mart)
        self.assertFalse(env.parcel_obtained_confirmed)
        env._update_story_state_from_loc(mart)
        self.assertTrue(env.parcel_obtained_confirmed)

        lab = dict(
            wrong_map,
            map_bank=4,
            map_id=3,
            viridian_mart_scene=0,
            pallet_oaks_lab_scene=6,
        )
        env._update_story_state_from_loc(lab)
        env._update_story_state_from_loc(lab)
        self.assertFalse(env.parcel_delivered_confirmed)
        env._update_story_state_from_loc(lab)
        self.assertTrue(env.parcel_delivered_confirmed)

    def test_deep_maps_are_independent_of_parcel_flags(self):
        env = bare_env(
            visited_maps={(3, 20), (1, 0), (3, 2)},
            has_target_starter=True,
        )
        self.assertEqual(env._world_stage(), 6)
        env.parcel_delivered_confirmed = True
        self.assertEqual(env._world_stage(), 6)

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
            with patch.object(PokemonFireRedEnv, "FULL_ONLY_MODE", False):
                role, _ = env._agent_role()
            roles[role] += 1
        return roles

    def test_scouts_appear_only_once_a_stage_has_a_valid_checkpoint(self):
        # V17.4: kein fester Scout-Sockel mehr, der schon vor dem ersten
        # echten Checkpoint existiert - "scout" gibt es erst, sobald es
        # ueberhaupt eine validierte Stage-Front zum Bedienen gibt. Jede
        # zusaetzliche validierte Stage bringt FRONTIER_SCOUT_SLOTS weitere
        # Scouts, bestehende werden nicht umgewidmet (siehe naechster Test).
        scouts = PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS
        n = 50

        def role_env(rank, stage_cps):
            env = bare_env(rank=rank, n_envs=n)
            env._valid_stage_checkpoints = lambda: dict(stage_cps)
            return env

        roles = Counter(
            role_env(r, {})._agent_role()[0] for r in range(n)
        )
        self.assertEqual(roles, Counter({"full": n}))

        roles = Counter(
            role_env(r, {2: "stage_2"})._agent_role()[0] for r in range(n)
        )
        self.assertEqual(roles, Counter({"full": n - scouts, "scout": scouts}))

        roles = Counter(
            role_env(r, {2: "stage_2", 6: "stage_6"})._agent_role()[0]
            for r in range(n)
        )
        self.assertEqual(
            roles, Counter({"full": n - 2 * scouts, "scout": 2 * scouts})
        )

    def test_scout_pairs_stay_pinned_to_their_own_stage(self):
        # Frueher wanderten ALLE Scouts sofort zur neuesten Front, sobald sie
        # einen Checkpoint bekam - die alte Front wurde komplett verwaisen
        # gelassen. Jetzt behalten die urspruenglichen (aeussersten) Slots
        # ihre alte, niedrigere Stage; NEUE Slots (naeher am Sockel) bedienen
        # die neu hinzugekommene, hoehere Stage.
        scouts = PokemonFireRedEnv.FRONTIER_SCOUT_SLOTS
        n = 50
        stage_cps = {2: "stage_2", 3: "stage_3"}

        def scout_env(rank):
            env = bare_env(rank=rank, n_envs=n)
            env._valid_stage_checkpoints = lambda: dict(stage_cps)
            env._discover_saved_milestones = lambda: list(stage_cps.values())
            env._champion_full_starter_ready = lambda: True
            return env

        self.assertEqual(scout_env(n - 1)._choose_episode_start(), "stage_2")
        self.assertEqual(
            scout_env(n - scouts)._choose_episode_start(), "stage_2"
        )
        self.assertEqual(
            scout_env(n - scouts - 1)._choose_episode_start(), "stage_3"
        )
        self.assertEqual(
            scout_env(n - 2 * scouts)._choose_episode_start(), "stage_3"
        )
        self.assertEqual(
            scout_env(n - 2 * scouts - 1)._choose_episode_start(), "beginning"
        )

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
        self.assertEqual(sum(roles.values()), 32)
        self.assertGreater(roles["starter"], max(
            roles["full"], roles["progress"], roles["exit"]
        ))

    def test_before_parcel_delivery_world_push_dominates(self):
        roles = self.roles_at(3)
        self.assertEqual(roles["progress"], 22)
        self.assertEqual(roles["battle"], 0)

    def test_after_forest_badge_specialists_activate(self):
        roles = self.roles_at(5)
        self.assertEqual(roles["badge"], 4)
        self.assertEqual(sum(roles.values()), 32)

    def test_route_two_prioritizes_world_push_and_full_validation(self):
        roles = self.roles_at(4)
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
        with patch.object(PokemonFireRedEnv, "FULL_ONLY_MODE", False):
            return env._choose_episode_start()

    def test_v16_full_always_starts_at_beginning(self):
        # rank=0 liegt ausserhalb der FRONTIER_SCOUT_SLOTS am Ende der
        # Flotte (siehe test_scout_slots_resume_from_deepest_checkpoint).
        env = bare_env(rank=0, n_envs=50, saved_milestones=["stage_5"])
        env._discover_saved_milestones = lambda: ["stage_5"]
        env._champion_full_starter_ready = lambda: True
        env._agent_role = lambda: ("full", "Full")
        self.assertEqual(env._choose_episode_start(), "beginning")

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
