"""Regression coverage for returning upstairs during the exit objective."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from pokemon_env import PokemonFireRedEnv


class NavigationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.env = PokemonFireRedEnv.__new__(PokemonFireRedEnv)
        self.env.stairs_down_rewarded = True
        self.env.left_house_rewarded = False
        self.env.initial_indoor_map = (4, 1)
        self.env.navigation_revision = 0
        self.env.training_objective = "exit"
        self.env.total_steps = 100
        self.env._nav_target_cache = None
        self.env._nav_target_cache_step = -999999
        self.warps = {
            "stairs": {(4, 0, 10, 2, 4, 1, 10, 2)},
            "exit": {(3, 0, 6, 8, 4, 0, 5, 8)},
        }
        self.loader = patch.object(
            self.env, "_load_confirmed_story_warps",
            side_effect=lambda kind: self.warps[kind],
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_exit_objective_recovers_downstairs_without_resetting_milestone(self):
        env = self.env
        self.assertEqual(env._target_coords_for_stage(4, 1), [(10, 2)])
        self.assertEqual(env._nav_target(4, 1, 8, 6), (10, 2))
        self.assertTrue(env.stairs_down_rewarded)
        self.assertFalse(env.left_house_rewarded)
        self.assertEqual(env._target_coords_for_stage(4, 0), [(5, 8)])
        self.assertEqual(env._nav_target(4, 0, 10, 2), (5, 8))

    def test_trusted_downstairs_map_restores_prior_story_flags(self):
        env = PokemonFireRedEnv.__new__(PokemonFireRedEnv)
        env.intro_complete_rewarded = False
        env.stairs_down_rewarded = False
        env.left_house_rewarded = False
        env.left_house_confirmed = False

        env._sync_story_progress_from_location(4, 0)

        self.assertTrue(env.intro_complete_rewarded)
        self.assertTrue(env.stairs_down_rewarded)
        self.assertFalse(env.left_house_confirmed)

    def test_missing_confirmed_stairs_does_not_invent_a_recovery_target(self):
        self.warps["stairs"] = set()
        self.assertEqual(self.env._target_coords_for_stage(4, 1), [])

    def test_exit_specialist_starting_downstairs_uses_actual_exit_map(self):
        self.env.initial_indoor_map = (4, 0)
        self.assertEqual(self.env._target_coords_for_stage(4, 0), [(5, 8)])
        self.assertEqual(self.env._target_coords_for_stage(4, 1), [(10, 2)])

    def test_recovery_ignores_unrelated_stairs_and_handles_both_endpoint_orders(self):
        self.warps["stairs"] = {(4, 1, 10, 2, 4, 0, 10, 2), (5, 0, 1, 1, 5, 1, 2, 2)}
        self.assertEqual(self.env._target_coords_for_stage(4, 1), [(10, 2)])
        self.assertEqual(self.env._target_coords_for_stage(5, 0), [])

    def test_step_history_keeps_coordinates_with_the_previous_map(self):
        # Execute the actual end-of-step history assignments without booting
        # an emulator. A map-only update previously retained reset-time X/Y.
        tree = ast.parse(textwrap.dedent(inspect.getsource(PokemonFireRedEnv.step)))
        blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(stmt, ast.Assign)
                          and any(isinstance(t, ast.Attribute)
                                  and t.attr == "previous_valid_bank" for t in stmt.targets)
                          for stmt in node.body)]
        block = max(blocks, key=lambda node: node.lineno)
        assignments = [stmt for stmt in block.body if isinstance(stmt, ast.Assign)
                       and any(isinstance(t, ast.Attribute)
                               and t.attr.startswith("previous_valid_") for t in stmt.targets)]
        env = self.env
        env.previous_valid_x, env.previous_valid_y = 10, 2
        scope = {"self": env, "bank": 4, "map_id": 0, "x": 5, "y": 8}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "step_history", "exec"), scope)
        self.assertEqual(
            (env.previous_valid_bank, env.previous_valid_map,
             env.previous_valid_x, env.previous_valid_y), (4, 0, 5, 8),
        )


class NavigationEvidenceVersionTests(unittest.TestCase):
    def test_legacy_stale_coordinate_warps_are_not_used_as_confirmed_targets(self):
        with tempfile.TemporaryDirectory() as root, patch("pokemon_env.SHARED_CURRICULUM_DIR", root):
            old = Path(root) / "confirmed_story_warps"
            old.mkdir()
            for rank in range(2):
                (old / f"agent_{rank:03d}_exit.json").write_text(json.dumps(
                    {"transition": [4, 0, 10, 2, 3, 0, 6, 7]}))
            env = PokemonFireRedEnv.__new__(PokemonFireRedEnv)
            self.assertEqual(env._load_confirmed_story_warps("exit"), set())
            for rank in range(2):
                env.rank = rank
                env._save_confirmed_story_warp("exit", (4, 0, 5, 8, 3, 0, 6, 8))
            self.assertEqual(env._load_confirmed_story_warps("exit"), {(4, 0, 5, 8, 3, 0, 6, 8)})
            self.assertTrue((old / "agent_000_exit.json").exists())

    def test_legacy_exit_routes_with_stale_edge_origins_are_quarantined(self):
        with tempfile.TemporaryDirectory() as root, patch("pokemon_env.SHARED_CURRICULUM_DIR", root):
            old = Path(root) / "exit_routes"
            old.mkdir()
            env = PokemonFireRedEnv.__new__(PokemonFireRedEnv)
            for rank in range(env.EXIT_ROUTE_CONFIRM_AGENTS):
                (old / f"agent_{rank:02d}.json").write_text(json.dumps(
                    {"edges": [[4, 0, 10, 2, 4, 0, 5, 8]]}))
            self.assertEqual(env._load_confirmed_exit_route_edges(), set())
            self.assertTrue((old / "agent_00.json").exists())


if __name__ == "__main__":
    unittest.main()
