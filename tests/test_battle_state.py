import unittest

from battle_state import BattleState


def enemy(personality=42, hp=10):
    return [{
        "slot": 0,
        "species_id": 19,
        "personality": personality,
        "cur_hp": hp,
        "moves": [{"id": 33, "pp": 35}],
    }]


class BattleStateTests(unittest.TestCase):
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
