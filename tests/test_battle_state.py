import unittest
from battle_state import BattleState


def enemy(hp=20, personality=42):
    return [{'slot': 0, 'species_id': 16, 'personality': personality,
             'cur_hp': hp, 'moves': [{'id': 33, 'pp': 35}]}]


class BattleStateTests(unittest.TestCase):
    def test_zero_flag_wild_battle_is_detected_before_damage(self):
        tracker = BattleState(position=(3, 19, 4, 5))
        self.assertEqual(tracker.update(enemy(), (3, 19, 4, 5), flags=0), 1)

    def test_menu_does_not_expire_after_96_steps(self):
        tracker = BattleState()
        tracker.update(enemy(), (3, 19, 4, 5))
        for _ in range(200):
            self.assertEqual(tracker.update(enemy(), (3, 19, 4, 5)), 1)

    def test_movement_ends_battle_despite_stale_flags_and_party(self):
        tracker = BattleState()
        tracker.update(enemy(), (3, 19, 4, 5), flags=8, signal=True)
        self.assertEqual(tracker.update(enemy(), (3, 19, 4, 6), flags=8, signal=True), 0)
        self.assertEqual(tracker.update(enemy(), (3, 19, 4, 6), flags=8, signal=True), 0)

    def test_reset_baseline_does_not_start_phantom_battle(self):
        tracker = BattleState(enemy())
        self.assertEqual(tracker.update(enemy(), (3, 19, 4, 5)), 0)

    def test_new_same_species_encounter_detected(self):
        tracker = BattleState(enemy())
        self.assertEqual(tracker.update(enemy(personality=43), (3, 19, 4, 5)), 1)
