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
    def test_reads_source_backed_early_story_scene_variables(self):
        env = _RamEnv(firered_ram.EWRAM_SIZE + 0x8000)
        base = 0x1200
        slot = firered_ram.PTR_SLOT_COMBINED
        for index, pointer in enumerate((base, base + 0x80, base + 0x100)):
            env.ram[slot + index * 4:slot + index * 4 + 4] = (
                firered_ram.EWRAM_BUS_BASE + pointer
            ).to_bytes(4, "little")
        env.ram[base:base + 6] = bytes((10, 0, 20, 0, 3, 19))
        for var_id, value in (
            (0x4055, 3), (0x4051, 2), (0x4057, 1),
        ):
            offset = base + 0x1000 + (var_id - 0x4000) * 2
            env.ram[offset:offset + 2] = value.to_bytes(2, "little")
        with patch.object(firered_ram, "_cached_ptr_slot", None):
            location = firered_ram.read_player_location(env, allow_scan=False)
        self.assertTrue(location["valid"])
        self.assertEqual(location["pallet_oaks_lab_scene"], 3)
        self.assertEqual(location["viridian_old_man_scene"], 2)
        self.assertEqual(location["viridian_mart_scene"], 1)

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
