import unittest

from pokemon_env import reset_emulator_to_episode_start


class WatcherResetTests(unittest.TestCase):
    def test_watcher_restores_its_captured_beginning_snapshot(self):
        class Emulator:
            def __init__(self):
                self.loaded = None
                self.reset_calls = 0

            def set_state(self, value):
                self.loaded = value

            def reset(self):
                self.reset_calls += 1

        environment = type("Environment", (), {"em": Emulator()})()

        restored = reset_emulator_to_episode_start(environment, True, b"beginning")

        self.assertTrue(restored)
        self.assertEqual(environment.em.loaded, b"beginning")
        self.assertEqual(environment.em.reset_calls, 0)

    def test_regular_environments_keep_the_normal_emulator_reset(self):
        class Emulator:
            def __init__(self):
                self.loaded = None
                self.reset_calls = 0

            def set_state(self, value):
                self.loaded = value

            def reset(self):
                self.reset_calls += 1

        class Environment:
            def __init__(self):
                self.em = Emulator()
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

        environment = Environment()

        restored = reset_emulator_to_episode_start(environment, False, b"beginning")

        self.assertFalse(restored)
        self.assertIsNone(environment.em.loaded)
        self.assertEqual(environment.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
