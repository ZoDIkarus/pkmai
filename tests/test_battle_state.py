import unittest

from battle_state import BattleState, MainBattleReader


def enemy(personality=42, hp=10):
    return [{
        "slot": 0,
        "species_id": 19,
        "personality": personality,
        "cur_hp": hp,
        "moves": [{"id": 33, "pp": 35}],
    }]


class BattleStateTests(unittest.TestCase):
    def test_live_reader_falls_back_cleanly_until_a_main_layout_is_verified(self):
        self.assertIsNone(MainBattleReader().read(bytearray(64)))

    def test_live_reader_keeps_last_verified_battle_state_during_a_transient_layout_gap(self):
        reader = MainBattleReader()
        base = 0x40000
        ram = bytearray(0x48000)
        def u32(offset, value):
            ram[offset:offset + 4] = int(value).to_bytes(4, "little")
        u32(base + 4, 0x08000001); u32(base + 12, 0x08000005); u32(base + 0x24, 14)
        ram[base + 0x439] = 2
        self.assertIsNone(reader.read(ram))
        u32(base + 0x24, 28); self.assertIsNone(reader.read(ram))
        u32(base + 0x24, 42); self.assertTrue(reader.read(ram))
        u32(base + 4, 0)
        self.assertTrue(reader.read(ram))

    def test_enemy_ram_change_detects_wild_battle_with_zero_flags(self):
        tracker = BattleState()
        self.assertEqual(tracker.update(enemy(), (3, 19, 4, 5), flags=0), 1)

    def test_stationary_battle_menu_keeps_battle_active(self):
        tracker = BattleState()
        tracker.update(enemy(), (3, 19, 4, 5), flags=0)
        for _ in range(200):
            self.assertEqual(tracker.update(enemy(), (3, 19, 4, 5), flags=0), 1)

    def test_overworld_movement_ends_stale_battle_state(self):
        tracker = BattleState()
        tracker.update(enemy(), (3, 19, 4, 5), flags=0)
        self.assertEqual(tracker.update(enemy(), (3, 19, 5, 5), flags=8, signal=True), 0)


if __name__ == "__main__":
    unittest.main()
