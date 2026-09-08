"""Regression tests for the RAM-collector key routing (no emulator / cv2)."""
import importlib.util
import os
import unittest

_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "battle_dump_collect.py")
_spec = importlib.util.spec_from_file_location("battle_dump_collect", _PATH)
bdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdc)


class _Capture:
    def __init__(self):
        self.calls = []

    def __call__(self, label, *, expected_move=None, expected_action=None):
        self.calls.append({"label": label, "expected_move": expected_move,
                           "expected_action": expected_action})


def press(k, st, cap):
    return bdc.route_key(ord(k) if isinstance(k, str) else k, st, cap)


class MoveVsSlotKeyTests(unittest.TestCase):
    def test_key_1_is_move_cursor_0_and_does_not_touch_slot(self):
        st = bdc.new_state()
        st["active_slot"] = 4
        cap = _Capture()
        intent, payload = press("1", st, cap)
        self.assertEqual(intent, "capture")
        self.assertEqual(payload, "move_cursor_0")
        self.assertEqual(cap.calls, [{"label": "move_cursor_0",
                                      "expected_move": 0, "expected_action": None}])
        self.assertEqual(st["active_slot"], 4)          # unchanged

    def test_keys_2_3_4_are_move_cursors_1_2_3(self):
        for key, cur in (("2", 1), ("3", 2), ("4", 3)):
            st, cap = bdc.new_state(), _Capture()
            press(key, st, cap)
            self.assertEqual(cap.calls[0]["expected_move"], cur)
            self.assertEqual(st["active_slot"], 0)

    def test_key_8_sets_active_slot_1_and_makes_no_dump(self):
        st, cap = bdc.new_state(), _Capture()
        intent, payload = press("8", st, cap)
        self.assertEqual(intent, "state")
        self.assertEqual(st["active_slot"], 1)
        self.assertEqual(cap.calls, [])                 # no capture

    def test_slot_key_map_7_8_9_0_dash_equals(self):
        for key, slot in (("7", 0), ("8", 1), ("9", 2), ("0", 3), ("-", 4), ("=", 5)):
            st, cap = bdc.new_state(), _Capture()
            press(key, st, cap)
            self.assertEqual(st["active_slot"], slot)
            self.assertEqual(cap.calls, [])

    def test_number_row_never_appears_in_slot_map(self):
        self.assertEqual(set(bdc.SLOT_KEYS) & set(bdc.MOVE_KEYS), set())
        for n in "1234":
            self.assertIn(ord(n), bdc.MOVE_KEYS)
            self.assertNotIn(ord(n), bdc.SLOT_KEYS)


class OtherKeyTests(unittest.TestCase):
    def test_action_cursor_keys(self):
        for key, cur, name in (("F", 0, "fight"), ("B", 1, "bag"),
                               ("P", 2, "pokemon"), ("R", 3, "run")):
            st, cap = bdc.new_state(), _Capture()
            intent, payload = press(key, st, cap)
            self.assertEqual(payload, f"action_cursor_{name}")
            self.assertEqual(cap.calls[0]["expected_action"], cur)

    def test_space_is_a_generic_capture(self):
        st, cap = bdc.new_state(), _Capture()
        press(" ", st, cap)
        self.assertEqual(cap.calls[0]["label"], "generic")

    def test_context_keys(self):
        st, cap = bdc.new_state(), _Capture()
        press("e", st, cap)
        self.assertEqual(st["encounter"], 1)
        press("t", st, cap)
        self.assertEqual(st["battle_kind"], "trainer")
        press("m", st, cap)
        self.assertEqual((st["menu_state"], st["out_of_battle"]), ("main", False))
        press("v", st, cap)
        self.assertEqual(st["menu_state"], "move")
        press("p", st, cap)
        self.assertEqual(st["menu_state"], "party")
        press("W", st, cap)
        self.assertEqual((st["out_of_battle"], st["menu_state"]), (True, "none"))
        self.assertEqual(cap.calls, [])

    def test_walk_button_reload_quit_intents(self):
        st, cap = bdc.new_state(), _Capture()
        self.assertEqual(press("w", st, cap)[0], "walk")
        self.assertEqual(press("j", st, cap)[0], "button")
        self.assertEqual(press("l", st, cap)[0], "reload")
        self.assertEqual(press("q", st, cap)[0], "quit")

    def test_unknown_key_is_ignored(self):
        st, cap = bdc.new_state(), _Capture()
        self.assertEqual(press("z", st, cap)[0], "ignored")
        self.assertEqual(cap.calls, [])


class AutomaticEncounterTests(unittest.TestCase):
    def test_in_battle_flag_matches_custom_integration(self):
        import json
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "local", "custom_integrations",
            "PokemonFireRed-Gba", "data.json")
        with open(data_path) as f:
            integration = json.load(f)
        self.assertEqual(bdc.IN_BATTLE_FLAG,
                         integration["info"]["in_battle"]["address"])

    def test_battle_start_creates_encounter_without_e(self):
        st = bdc.new_state()
        message = bdc.update_battle_transition(st, False, True)
        self.assertIn("AUTO new encounter enc01", message)
        self.assertEqual(st["encounter"], 1)
        self.assertFalse(st["out_of_battle"])
        self.assertTrue(st["in_battle"])

        # Staying in the same fight must not create another encounter.
        self.assertIsNone(bdc.update_battle_transition(st, True, True))
        self.assertEqual(st["encounter"], 1)

    def test_battle_end_is_marked_automatically(self):
        st = bdc.new_state()
        st["menu_state"] = "main"
        message = bdc.update_battle_transition(st, True, False)
        self.assertEqual(message, "AUTO OUT OF BATTLE")
        self.assertTrue(st["out_of_battle"])
        self.assertEqual(st["menu_state"], "none")
        self.assertFalse(st["in_battle"])

    def test_unknown_gmain_read_holds_previous_presence(self):
        self.assertFalse(bdc.stable_battle_presence(False, None))
        self.assertTrue(bdc.stable_battle_presence(True, None))
        self.assertTrue(bdc.stable_battle_presence(False, True))
        self.assertFalse(bdc.stable_battle_presence(True, False))

    def test_default_state_is_frozen_route1_probe_seed(self):
        self.assertTrue(bdc.DEFAULT_START_STATE.endswith(
            "brain_backups/ram_probe_route1_seed/stage_frontier_2.state.gz"))
        self.assertEqual(bdc.parse_args([]).state, bdc.DEFAULT_START_STATE)

    def test_reload_keeps_encounter_and_file_counters(self):
        st = bdc.new_state()
        st["encounter"] = 1
        st["battle_kind"] = "trainer"
        st["menu_state"] = "move"
        st["count"]["action_cursor_fight"] = 1
        original_counts = st["count"]

        bdc.reset_context_after_reload(st)

        self.assertEqual(st["encounter"], 1)
        self.assertIs(st["count"], original_counts)
        self.assertEqual(st["count"]["action_cursor_fight"], 1)
        self.assertEqual(st["battle_kind"], "wild")
        self.assertEqual(st["menu_state"], "none")


if __name__ == "__main__":
    unittest.main()
