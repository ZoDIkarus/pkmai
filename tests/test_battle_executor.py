import unittest

import battle_executor as bx
from battle_executor import (SCREEN_MAIN, SCREEN_MOVES, SCREEN_PARTY,
                             SCREEN_MESSAGE, MENU_FIGHT, MENU_POKEMON, MENU_RUN)


def mon(slot=0, hp=30, moves=()):
    return {"slot": slot, "checksum_ok": True, "cur_hp": hp, "max_hp": 40,
            "moves": [dict(m) for m in moves]}


def mv(pp=10, mid=1):
    return {"id": mid, "pp": pp, "mechanics_known": True, "type": 0, "power": 40}


def snap(*, our_moves=(mv(), mv(pp=0)), party=None, is_trainer=False,
         can_escape=True):
    active = mon(0, moves=our_moves)
    return {
        "player_active": active,
        "player_party": party if party is not None else [active, mon(1, hp=25)],
        "enemy_active": mon(0, hp=20),
        "is_trainer": is_trainer, "can_escape": can_escape,
        "player_active_source": "authoritative",
    }


class FakeBattleIO(bx.BattleIO):
    """In-memory battle-menu model with a real cursor + screen stack."""

    def __init__(self, snapshot, *, in_battle=True, broken_cursor=False,
                 unexpected_menu=False, move0_zero_pp=False):
        self._snap = snapshot
        self._in_battle = in_battle
        self.broken_cursor = broken_cursor
        self.unexpected_menu = unexpected_menu
        self.move0_zero_pp = move0_zero_pp
        self.screen = SCREEN_MAIN
        self.cursor = 0
        self.messages = 0
        self.presses = []
        self.executed = None

    def in_battle(self):
        return self._in_battle

    def snapshot(self):
        return self._snap

    def menu_state(self):
        return {"screen": self.screen, "cursor": self.cursor,
                "message_pending": self.messages > 0, "battler_ready": True}

    def press(self, button):
        self.presses.append(button)
        if self.messages > 0 and button == "A":
            self.messages -= 1
            if self.messages == 0 and self.screen == SCREEN_MESSAGE:
                self.screen = SCREEN_MAIN          # turn resolved
                self.cursor = MENU_FIGHT
            return self.menu_state()
        if self.unexpected_menu:
            self.screen = "unknown"
            return self.menu_state()
        if self.screen == SCREEN_MAIN:
            if button == "LEFT" and self.cursor % 2:
                self.cursor -= 1
            elif button == "RIGHT" and not self.cursor % 2:
                self.cursor += 1
            elif button == "UP" and self.cursor >= 2:
                self.cursor -= 2
            elif button == "DOWN" and self.cursor < 2:
                self.cursor += 2
            elif button == "A":
                if self.cursor == MENU_FIGHT:
                    self.screen, self.cursor = SCREEN_MOVES, 0
                elif self.cursor == MENU_POKEMON:
                    self.screen, self.cursor = SCREEN_PARTY, 0
                elif self.cursor == MENU_RUN:
                    self.executed = "RUN"
                    self.messages = 1
        elif self.screen == SCREEN_MOVES:
            if button == "LEFT" and self.cursor % 2 and not self.broken_cursor:
                self.cursor -= 1
            elif button == "RIGHT" and not self.cursor % 2 and not self.broken_cursor:
                self.cursor += 1
            elif button == "UP" and self.cursor >= 2 and not self.broken_cursor:
                self.cursor -= 2
            elif button == "DOWN" and self.cursor < 2 and not self.broken_cursor:
                self.cursor += 2
            elif button == "A":
                if self.move0_zero_pp and self.cursor == 0:
                    self.messages = 1        # re-prompt, stay on move list
                else:
                    self.executed = f"MOVE_{self.cursor + 1}"
                    self.screen = SCREEN_MESSAGE
                    self.messages = 1
            elif button == "B":
                self.screen, self.cursor = SCREEN_MAIN, MENU_FIGHT
        elif self.screen == SCREEN_PARTY:
            if button in ("UP", "DOWN"):
                self.cursor = min(5, max(0, self.cursor + (1 if button == "DOWN" else -1)))
            elif button == "A":
                self.executed = f"SWITCH_{self.cursor + 1}"
                self.screen = SCREEN_MESSAGE
                self.messages = 1
            elif button == "B":
                self.screen, self.cursor = SCREEN_MAIN, MENU_POKEMON
        return self.menu_state()


class ActionMaskTests(unittest.TestCase):
    def test_action_mask_none_is_all_zeros(self):
        self.assertEqual(bx.action_mask(None), [0] * len(bx.ALL_MACROS))
        self.assertEqual(bx.action_mask({}), [0] * len(bx.ALL_MACROS))

    def test_zero_pp_and_fainted_and_trainer_run_are_masked(self):
        s = snap(our_moves=(mv(pp=5), mv(pp=0)),
                 party=[mon(0), mon(1, hp=0)], is_trainer=True, can_escape=False)
        legal = bx.legal_macros(s)
        self.assertIn("MOVE_1", legal)
        self.assertNotIn("MOVE_2", legal)     # 0 PP
        self.assertNotIn("SWITCH_2", legal)   # fainted
        self.assertNotIn("RUN", legal)        # trainer battle


class ExecutorTests(unittest.TestCase):
    def test_refuses_outside_battle(self):
        io = FakeBattleIO(snap(), in_battle=False)
        r = MacroExecute(io, "MOVE_1")
        self.assertFalse(r["ok"])
        self.assertIn("not in a confirmed battle", r["reason"])
        self.assertEqual(io.presses, [])

    def test_executes_a_move_with_cursor_verification(self):
        io = FakeBattleIO(snap())
        r = MacroExecute(io, "MOVE_1")
        self.assertTrue(r["ok"], r["reason"])
        self.assertEqual(io.executed, "MOVE_1")
        self.assertIn("A", io.presses)

    def test_move_2_navigates_the_move_cursor(self):
        io = FakeBattleIO(snap(our_moves=(mv(pp=5, mid=1), mv(pp=5, mid=2))))
        r = MacroExecute(io, "MOVE_2")
        self.assertTrue(r["ok"], r["reason"])
        self.assertEqual(io.executed, "MOVE_2")

    def test_switch_navigates_party_cursor(self):
        io = FakeBattleIO(snap(party=[mon(0), mon(1, hp=25), mon(2, hp=30)]))
        r = MacroExecute(io, "SWITCH_3")
        self.assertTrue(r["ok"], r["reason"])
        self.assertEqual(io.executed, "SWITCH_3")

    def test_run_only_in_escapable_wild(self):
        ok_io = FakeBattleIO(snap(can_escape=True, is_trainer=False))
        self.assertTrue(MacroExecute(ok_io, "RUN")["ok"])
        self.assertEqual(ok_io.executed, "RUN")
        bad_io = FakeBattleIO(snap(can_escape=False, is_trainer=True))
        r = MacroExecute(bad_io, "RUN")
        self.assertFalse(r["ok"])          # masked as illegal action

    def test_zero_pp_move_is_masked_not_pressed(self):
        io = FakeBattleIO(snap(our_moves=(mv(pp=0, mid=1), mv(pp=5, mid=2))))
        r = MacroExecute(io, "MOVE_1")
        self.assertFalse(r["ok"])
        self.assertIn("not a legal action", r["reason"])

    def test_move_not_accepted_aborts_safely(self):
        io = FakeBattleIO(snap(our_moves=(mv(pp=5, mid=1),)), move0_zero_pp=True)
        r = MacroExecute(io, "MOVE_1")
        self.assertFalse(r["ok"])
        self.assertTrue(r["aborted"])
        self.assertIn("not accepted", r["reason"])

    def test_unexpected_menu_aborts(self):
        io = FakeBattleIO(snap(), unexpected_menu=True)
        r = MacroExecute(io, "MOVE_1")
        self.assertTrue(r["aborted"])

    def test_broken_cursor_aborts_within_press_budget(self):
        io = FakeBattleIO(snap(our_moves=(mv(pp=5, mid=1), mv(pp=5, mid=2))),
                          broken_cursor=True)
        r = MacroExecute(io, "MOVE_2")
        self.assertTrue(r["aborted"])
        self.assertLessEqual(r["presses"], bx.DEFAULT_MAX_PRESSES)

    def test_message_storm_is_bounded(self):
        io = FakeBattleIO(snap())
        io.messages = 999
        r = MacroExecute(io, "MOVE_1")
        self.assertTrue(r["aborted"])
        self.assertIn("message", r["reason"])


def MacroExecute(io, macro):
    return bx.MacroExecutor(io).execute(macro, dry_run=True)


from battle_executor import MacroExecutor  # noqa: E402  (used above)


if __name__ == "__main__":
    unittest.main()
