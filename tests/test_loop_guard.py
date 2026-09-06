import unittest

from loop_guard import LocalLoopGuard, ShortCycleGuard


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


class ShortCycleGuardTests(unittest.TestCase):
    def test_ab_cycle_escalates_then_resets_on_real_progress(self):
        guard = ShortCycleGuard(history=8, min_repeats=3, max_period=3, escalate_every=1)
        results = [guard.update((3, 0, step % 2, 0), "pallet") for step in range(7)]
        self.assertTrue(results[-1]["cycle"])
        self.assertTrue(results[-1]["suppress_shaping"])
        self.assertLess(results[-1]["penalty"], 0.0)
        reset = guard.update((3, 19, 0, 0), "route1")
        self.assertFalse(reset["cycle"])
        self.assertEqual(reset["penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
