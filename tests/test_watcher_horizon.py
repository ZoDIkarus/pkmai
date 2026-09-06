import ast
import unittest
from pathlib import Path


class WatcherHorizonRegressionTests(unittest.TestCase):
    def test_dynamic_watcher_marks_its_environment_as_unbounded(self):
        watcher = Path("src/dynamic_watcher.py").read_text(encoding="utf-8")
        self.assertIn("is_watcher=True", watcher)

    def test_watcher_episode_limit_is_unbounded_but_training_is_not(self):
        tree = ast.parse(Path("src/pokemon_env.py").read_text(encoding="utf-8"))
        guards = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "is_watcher"
        ]
        self.assertEqual(len(guards), 1)
        guard = guards[0]
        self.assertIsInstance(guard.body[0], ast.Assign)
        self.assertEqual(ast.unparse(guard.body[0].value), "10 ** 12")
        self.assertTrue(guard.orelse)
        self.assertIn("MAX_EPISODE_STEPS", ast.unparse(guard.orelse[0]))


if __name__ == "__main__":
    unittest.main()
