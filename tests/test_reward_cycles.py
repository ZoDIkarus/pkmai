import unittest

from pokemon_env import short_cycle_repeats


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


if __name__ == "__main__":
    unittest.main()
