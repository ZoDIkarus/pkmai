import unittest
from unittest.mock import patch

import firered_ram


class _RamEnv:
    def __init__(self, size):
        self.ram = bytearray(size)

    def get_ram(self):
        return self.ram


class PlayerPartyValidationTests(unittest.TestCase):
    def test_rejects_checksum_invalid_but_plausible_party_entry(self):
        env = _RamEnv(
            firered_ram.PLAYER_PARTY_OFFSET
            + firered_ram.POKEMON_STRUCT_SIZE * firered_ram.MAX_PARTY_SIZE
        )
        plausible_invalid = {
            "slot": 0,
            "species_id": 1,
            "level": 5,
            "cur_hp": 10,
            "max_hp": 20,
            "checksum_ok": False,
        }
        valid = {
            "slot": 1,
            "species_id": 7,
            "level": 5,
            "cur_hp": 20,
            "max_hp": 20,
            "checksum_ok": True,
        }
        with patch.object(
            firered_ram,
            "_decode_party_mon",
            side_effect=[plausible_invalid, valid, None, None, None, None],
        ):
            self.assertEqual(firered_ram.read_player_party(env), [valid])


class RamLocationSafetyTests(unittest.TestCase):
    def test_rejects_impossible_group_zero_map(self):
        ram = bytearray(16)
        ram[4] = 0
        ram[5] = 57
        self.assertFalse(firered_ram._valid_location(ram, 0))

    def test_reads_battle_type_flags_from_confirmed_offset(self):
        env = _RamEnv(firered_ram.BATTLE_TYPE_FLAGS_OFFSET + 4)
        env.ram[firered_ram.BATTLE_TYPE_FLAGS_OFFSET:firered_ram.BATTLE_TYPE_FLAGS_OFFSET + 4] = bytes((8, 0, 0, 0))
        self.assertEqual(firered_ram.read_battle_type_flags(env), 8)


if __name__ == "__main__":
    unittest.main()
