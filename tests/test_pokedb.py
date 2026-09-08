import unittest
import pokedb
import battle_types as bt


class PokedbTests(unittest.TestCase):
    def test_db_available_and_reproducible_meta(self):
        self.assertTrue(pokedb.is_available(), "run tools/extract_gen3_data.py")
        meta = pokedb.db_meta()
        self.assertEqual(meta["species_meta"]["generation"], 3)
        self.assertEqual(meta["species_meta"]["source_rom_gamecode"], "BPRD")
        self.assertGreaterEqual(meta["species_count"], 386)
        self.assertGreaterEqual(meta["moves_count"], 354)

    def test_species_types_gen3(self):
        self.assertEqual(pokedb.species_types(7), [bt.TYPE_WATER])       # Squirtle
        self.assertEqual(pokedb.species_types(6), [bt.TYPE_FIRE, bt.TYPE_FLYING])  # Charizard
        self.assertEqual(pokedb.species_types(1), [bt.TYPE_GRASS, bt.TYPE_POISON])  # Bulbasaur
        self.assertEqual(pokedb.species_types(95), [bt.TYPE_ROCK, bt.TYPE_GROUND])  # Onix

    def test_base_stats(self):
        s = pokedb.base_stats(150)   # Mewtwo
        self.assertEqual(s["sp_attack"], 154)
        self.assertEqual(s["speed"], 130)

    def test_move_mechanics_gen3_values(self):
        wg = pokedb.move_mechanics(55)   # Water Gun
        self.assertEqual((wg["type"], wg["power"], wg["accuracy"], wg["pp"]),
                         (bt.TYPE_WATER, 40, 100, 25))
        hp = pokedb.move_mechanics(56)   # Hydro Pump - Gen III: 120 BP, 80% acc
        self.assertEqual((hp["power"], hp["accuracy"]), (120, 80))
        qa = pokedb.move_mechanics(98)   # Quick Attack
        self.assertEqual(qa["priority"], 1)
        growl = pokedb.move_mechanics(45)
        self.assertTrue(growl["is_status"])
        self.assertEqual(growl["power"], 0)

    def test_gen3_physical_special_by_move_type(self):
        # Tackle (Normal) -> physical; Ember (Fire) -> special, in Gen III.
        tackle = pokedb.move_mechanics(33)
        ember = pokedb.move_mechanics(52)
        self.assertTrue(bt.is_physical_type(tackle["type"]))
        self.assertTrue(bt.is_special_type(ember["type"]))

    def test_unknown_ids_fail_closed(self):
        self.assertIsNone(pokedb.species_info(99999))
        self.assertIsNone(pokedb.species_types(0))
        self.assertIsNone(pokedb.move_info(99999))
        self.assertIsNone(pokedb.move_mechanics(None))
        self.assertIsNone(pokedb.base_stats("nope"))
