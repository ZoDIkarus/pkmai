import unittest

from loop_guard import LocalLoopGuard


class LocalLoopGuardTests(unittest.TestCase):
    def test_detects_prolonged_wandering_in_a_tiny_area(self):
        guard = LocalLoopGuard(window=6, max_tiles=2, max_idle_steps=100)
        results = [guard.update((3, 19, step % 2, 0), "progress") for step in range(6)]
        self.assertEqual(results[-1], True)

    def test_progress_and_battles_reset_or_pause_the_guard(self):
        guard = LocalLoopGuard(window=6, max_tiles=2, max_idle_steps=100)
        for step in range(5):
            self.assertFalse(guard.update((3, 19, step % 2, 0), "progress"))
        self.assertFalse(guard.update((3, 19, 0, 0), "new-progress"))
        self.assertFalse(guard.update((3, 19, 0, 0), "new-progress", in_battle=True))


if __name__ == "__main__":
    unittest.main()
