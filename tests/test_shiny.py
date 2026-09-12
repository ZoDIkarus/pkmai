import unittest

import shiny


class ShinyMathTests(unittest.TestCase):
    def test_gen3_shiny_formula_and_fail_closed_inputs(self):
        self.assertTrue(shiny.is_shiny(0, 0, 0))
        self.assertTrue(shiny.is_shiny(0, 0, 7))
        self.assertFalse(shiny.is_shiny(0, 0, 8))
        self.assertIsNone(shiny.is_shiny(None, 0, 123))

    def test_splits_packed_trainer_and_secret_ids(self):
        self.assertEqual(shiny.split_trainer_id(0xB2C30001), (0x0001, 0xB2C3))
        self.assertEqual(shiny.split_trainer_id(-1), (None, None))


if __name__ == "__main__":
    unittest.main()
