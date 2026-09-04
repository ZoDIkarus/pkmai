import os
import tempfile
import unittest
from unittest import mock

import mapper


class MapperIsolationTests(unittest.TestCase):
    def test_mapper_model_is_not_a_main_training_checkpoint(self):
        self.assertNotEqual(mapper.MAPPER_MODEL, mapper.RESUME_MODEL)
        self.assertNotEqual(mapper.MAPPER_MODEL, mapper.BEST_MODEL)

    def test_mapper_navigation_memory_stays_in_mapper_directory(self):
        env = mapper.MapperEnv.__new__(mapper.MapperEnv)
        path = env._exploration_memory_path()
        self.assertEqual(
            os.path.commonpath((mapper.MAPPER_DIR, path)),
            mapper.MAPPER_DIR,
        )

    def test_mapper_cannot_publish_training_checkpoints_or_routes(self):
        env = mapper.MapperEnv.__new__(mapper.MapperEnv)
        self.assertFalse(env._save_curriculum_state("stage_99"))
        self.assertFalse(env._save_stage_checkpoint(99, 3, 99, 1, 1))
        self.assertFalse(env._claim_global_depth(99))
        shared = {}
        self.assertFalse(env._claim_shared(shared, "tile"))
        self.assertEqual(shared, {})
        self.assertIsNone(env._save_confirmed_story_warp("exit", ()))
        self.assertIsNone(env._commit_successful_exit_route())
        self.assertIsNone(env._commit_journey_route())

    def test_atomic_mapper_save_uses_an_actual_zip_temp_path(self):
        class FakeModel:
            saved_path = None

            def save(self, path):
                self.saved_path = path
                with open(path, "wb") as f:
                    f.write(b"mapper")

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "mapper.zip")
            fake = FakeModel()
            with mock.patch.object(mapper, "MAPPER_MODEL", target):
                mapper.save_model_atomic(fake)
            self.assertTrue(fake.saved_path.endswith(".tmp.zip"))
            self.assertTrue(os.path.isfile(target))
            self.assertFalse(os.path.exists(fake.saved_path))

    def test_frontier_brain_tries_each_direction_instead_of_looping(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = mapper.FrontierBrain(os.path.join(directory, "brain.json"))
            loc = {
                "valid": True, "map_bank": 3, "map_id": 0,
                "x_pos": 10, "y_pos": 10,
            }
            self.assertEqual(brain.choose(loc), 3)
            brain.observe(loc, 3, loc)
            brain.observe(loc, 3, loc)
            self.assertEqual(brain.choose(loc), 6)

    def test_frontier_brain_records_only_adjacent_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = mapper.FrontierBrain(os.path.join(directory, "brain.json"))
            before = {
                "valid": True, "map_bank": 3, "map_id": 0,
                "x_pos": 10, "y_pos": 10,
            }
            adjacent = dict(before, x_pos=11)
            brain.observe(before, 6, adjacent)
            self.assertEqual(len(brain.edges), 1)
            jumped = dict(before, x_pos=20)
            brain.observe(before, 6, jumped)
            self.assertEqual(len(brain.edges), 1)

    def test_mapper_battle_escape_is_deterministic_and_repeats(self):
        env = mapper.MapperEnv.__new__(mapper.MapperEnv)
        env.battle_escape_step = 0
        actions = [env._battle_escape_action() for _ in range(7)]
        self.assertEqual(actions, [1, 1, 4, 6, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
