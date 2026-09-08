import unittest
import battle_types as bt


class TypeChartTests(unittest.TestCase):
    def test_water_vs_rock_ground_is_4x(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_WATER,
                                          [bt.TYPE_ROCK, bt.TYPE_GROUND]), 4.0)

    def test_fire_vs_grass_is_2x(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_FIRE, [bt.TYPE_GRASS]), 2.0)

    def test_normal_vs_ghost_is_0x(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_NORMAL, [bt.TYPE_GHOST]), 0.0)
        self.assertEqual(bt.effectiveness(bt.TYPE_NORMAL,
                                          [bt.TYPE_GHOST, bt.TYPE_POISON]), 0.0)

    def test_ground_vs_flying_is_0x_even_with_second_type(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_GROUND,
                                          [bt.TYPE_FLYING, bt.TYPE_STEEL]), 0.0)

    def test_electric_vs_ground_immunity(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_ELECTRIC, [bt.TYPE_GROUND]), 0.0)

    def test_resistant_double_type_is_quarter(self):
        # Fire vs Fire/Water: 0.5 * 0.5 = 0.25
        self.assertEqual(bt.effectiveness(bt.TYPE_FIRE,
                                          [bt.TYPE_FIRE, bt.TYPE_WATER]), 0.25)
        # Grass vs Bug/Steel: 0.5 * 0.5
        self.assertEqual(bt.effectiveness(bt.TYPE_GRASS,
                                          [bt.TYPE_BUG, bt.TYPE_STEEL]), 0.25)

    def test_mono_type_stored_as_duplicate_counts_once(self):
        # Gen III stores mono-types as type1==type2.
        self.assertEqual(bt.effectiveness(bt.TYPE_FIRE,
                                          [bt.TYPE_GRASS, bt.TYPE_GRASS]), 2.0)

    def test_ghost_vs_psychic_is_2x_gen2plus(self):
        self.assertEqual(bt.effectiveness(bt.TYPE_GHOST, [bt.TYPE_PSYCHIC]), 2.0)

    def test_stab_multiplier(self):
        self.assertEqual(bt.stab_multiplier(bt.TYPE_WATER,
                                            [bt.TYPE_WATER, bt.TYPE_WATER]), 1.5)
        self.assertEqual(bt.stab_multiplier(bt.TYPE_WATER, [bt.TYPE_GRASS]), 1.0)
        self.assertEqual(bt.stab_multiplier(bt.TYPE_FLYING,
                                            [bt.TYPE_NORMAL, bt.TYPE_FLYING]), 1.5)

    def test_gen3_physical_special_is_by_type_not_move_category(self):
        # Gen III: physical/special decided by TYPE.
        self.assertTrue(bt.is_physical_type(bt.TYPE_NORMAL))
        self.assertTrue(bt.is_physical_type(bt.TYPE_GHOST))     # physical in Gen III
        self.assertTrue(bt.is_physical_type(bt.TYPE_FLYING))    # physical in Gen III
        self.assertTrue(bt.is_special_type(bt.TYPE_FIRE))
        self.assertTrue(bt.is_special_type(bt.TYPE_DARK))       # special in Gen III
        self.assertTrue(bt.is_special_type(bt.TYPE_ICE))
        # every real type is exactly one of the two
        for t in range(18):
            self.assertNotEqual(bt.is_physical_type(t), bt.is_special_type(t))

    def test_invalid_types_are_neutral_not_crash(self):
        self.assertEqual(bt.effectiveness(99, [bt.TYPE_WATER]), 1.0)
        self.assertEqual(bt.effectiveness(bt.TYPE_WATER, [None, "x"]), 1.0)
        self.assertEqual(bt.effectiveness(bt.TYPE_WATER, []), 1.0)

    def test_full_chart_only_contains_legal_multipliers(self):
        for (a, d), mult in bt._CHART.items():
            self.assertIn(mult, (0.0, 0.5, 2.0))
            self.assertTrue(bt.is_valid_type(a) and bt.is_valid_type(d))
