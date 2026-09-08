"""EmulatorBattleDriver — read path against real dump RAM + fail-closed logic.

The button-driving path is exercised with a scripted fake env whose RAM is set
from the real 20260907_184350 dumps; a true end-to-end battle canary needs a
live emulator and is covered by the pre-flight (which reports NO-GO until run).
"""
import glob
import gzip
import json
import os
import unittest

import numpy as np

from twoby2 import emulator_battle_driver as D
from twoby2.emulator_battle_driver import build_live_snapshot, EmulatorBattleDriver

DUMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runtime",
                                        "ram_probe", "20260907_184350"))


def _dump(label):
    jf = sorted(glob.glob(os.path.join(DUMP_DIR, f"{label}*.json")))[0]
    with open(jf) as f:
        m = json.load(f)
    with gzip.open(jf[:-5] + ".ram.gz", "rb") as f:
        m["ram"] = f.read()
    return m


class _FakeEm:
    def __init__(self, ram):
        self._ram = ram
        self.steps = 0
        self.masks = []

    def set_button_mask(self, m):
        self.masks.append(bytes(np.asarray(m, dtype=np.uint8)))

    def step(self):
        self.steps += 1

    def get_state(self):
        return b""


class _FakeEnv:
    buttons = ["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]

    def __init__(self, ram):
        self.em = _FakeEm(ram)
        self._ram = ram

    def get_ram(self):
        return self._ram


class _FakeMBR:
    def __init__(self, in_battle):
        self._v = in_battle

    def read(self, ram, frames=None):
        return self._v


HAVE_DUMPS = os.path.isdir(DUMP_DIR) and glob.glob(os.path.join(DUMP_DIR, "*.ram.gz"))


@unittest.skipUnless(HAVE_DUMPS, "verified RAM-probe dataset not present")
class LiveSnapshotTests(unittest.TestCase):
    def test_snapshot_from_real_main_menu_dump(self):
        m = _dump("action_cursor_fight")
        env = _FakeEnv(m["ram"])
        snap = build_live_snapshot(env, main_battle_reader=_FakeMBR(1))
        self.assertEqual(snap["schema"], "battle_snapshot_v2_live")
        self.assertTrue(snap["in_battle"])
        self.assertEqual(snap["menu_state"], "main")
        self.assertEqual(snap["action_cursor"], 0)
        self.assertTrue(snap["active_slot_authoritative"])   # cross-check passed
        self.assertEqual(snap["player_active"]["species_id"], 7)
        self.assertEqual(snap["enemy_active"]["species_id"], 16)
        self.assertTrue(snap["menu_cursor_known"])
        self.assertTrue(snap["stat_stages_known"])
        # A plausible stale trainer-opponent id must not turn this real wild
        # dump into a trainer battle; gBattleTypeFlags is authoritative.
        self.assertFalse(snap["is_trainer"])

    def test_snapshot_after_switch_uses_the_real_active_slot(self):
        # generic_02 = post-switch: active_party_slot 1 (Rattata), gBattleMons[0]
        # must be Rattata, not Squirtle (party[0]).
        for m in (_dump("generic")["ram"],):
            pass
        # find the dump with active_party_slot 1
        jfs = sorted(glob.glob(os.path.join(DUMP_DIR, "generic*.json")))
        target = None
        for jf in jfs:
            d = json.load(open(jf))
            if d.get("active_party_slot") == 1:
                d["ram"] = gzip.open(jf[:-5] + ".ram.gz", "rb").read()
                target = d
        self.assertIsNotNone(target)
        env = _FakeEnv(target["ram"])
        snap = build_live_snapshot(env, main_battle_reader=_FakeMBR(1))
        self.assertEqual(snap["active_slot"], 1)
        self.assertEqual(snap["player_active"]["species_id"], 19)   # Rattata
        self.assertTrue(snap["active_slot_authoritative"])

    def test_out_of_battle_snapshot_is_fail_closed(self):
        jfs = sorted(glob.glob(os.path.join(DUMP_DIR, "*.json")))
        oob = None
        for jf in jfs:
            d = json.load(open(jf))
            if d["battle_kind"] == "out_of_battle":
                d["ram"] = gzip.open(jf[:-5] + ".ram.gz", "rb").read()
                oob = d
        self.assertIsNotNone(oob)
        env = _FakeEnv(oob["ram"])
        snap = build_live_snapshot(env, main_battle_reader=_FakeMBR(0))
        self.assertFalse(snap["in_battle"])
        self.assertFalse(snap["menu_cursor_known"])


@unittest.skipUnless(HAVE_DUMPS, "verified RAM-probe dataset not present")
class DriverFailClosedTests(unittest.TestCase):
    def _driver(self, label, in_battle=1):
        m = _dump(label)
        env = _FakeEnv(m["ram"])
        return EmulatorBattleDriver(env, main_battle_reader=_FakeMBR(in_battle)), m

    def test_action_mask_from_real_snapshot(self):
        drv, _ = self._driver("action_cursor_fight")
        snap = drv.snapshot()
        mask = drv.action_mask(snap)
        self.assertEqual(len(mask), len(D.ALL_MACROS))
        # Squirtle has 4 real moves with PP -> MOVE_1..4 legal
        for i in range(4):
            self.assertEqual(mask[i], 1, D.ALL_MACROS[i])
        # The party-list cursor has not been RAM-verified, so live switching is
        # deliberately fail-closed even though other party members are alive.
        for action in D.SWITCH_ACTIONS:
            self.assertEqual(mask[D.ALL_MACROS.index(action)], 0)

    def test_run_masked_in_a_trainer_battle(self):
        drv, _ = self._driver("action_cursor_fight")
        snap = drv.snapshot()
        snap["is_trainer"] = True
        snap["can_escape"] = False
        self.assertEqual(drv.action_mask(snap)[D.ALL_MACROS.index("RUN")], 0)

    def test_apply_macro_refused_when_not_in_battle(self):
        drv, _ = self._driver("action_cursor_fight", in_battle=0)
        snap, ev = drv.apply_macro("MOVE_1", dry_run=True)
        self.assertTrue(ev["invalid"])
        self.assertIn("not in a confirmed battle", ev["reason"])

    def test_double_battle_is_blocked(self):
        drv, _ = self._driver("action_cursor_fight")
        snap = drv.snapshot()
        snap["is_double"] = True
        with self.assertRaises(D.DoubleBattleBlocked):
            drv._assert_singles(snap)
        snap2, ev = drv.apply_macro("MOVE_1", dry_run=True)
        # apply_macro catches it -> invalid + aborted, no button pressed
        # (snapshot's is_double comes from RAM; here we just confirm the guard
        #  path exists and is reachable via _assert_singles)

    def test_illegal_flee_in_trainer_battle(self):
        drv, _ = self._driver("action_cursor_fight")
        # monkeypatch snapshot to report a trainer battle
        real = drv.snapshot
        drv.snapshot = lambda: {**real(), "is_trainer": True, "can_escape": False,
                                "in_battle": True}
        snap, ev = drv.apply_macro("RUN", dry_run=True)
        self.assertTrue(ev["invalid"])

    def test_resolve_turn_advances_message_then_records_battle_end(self):
        import twoby2.battle_ram_live as L
        drv, _ = self._driver("action_cursor_fight")
        state = {"frame": 0}
        orig = L.menu_state_raw
        self.addCleanup(lambda: setattr(L, "menu_state_raw", orig))
        L.menu_state_raw = lambda ram: 0          # not 18/20/22 -> "message"
        drv._mbr = type("M", (), {"read": staticmethod(
            lambda ram, **kw: state["frame"] < 12)})()
        real_step = drv.env.em.step

        def step():
            state["frame"] += 1
            real_step()
        drv.env.em.step = step

        res = drv._resolve_turn(enemy_hp0=20, enemy_idx0=0)
        self.assertTrue(res["left_battle"])                 # inBattle went false
        a_ix = drv.env.buttons.index("A")
        self.assertTrue(any(m[a_ix] for m in drv.env.em.masks),
                        "A must be pressed to advance the result text")
        # ended with no win/wipe/flee signal -> diagnosed, NOT a silent timeout
        v = D.classify_turn_outcome("MOVE_1", res)
        self.assertTrue(v["terminal_unknown"])
        self.assertFalse(v["battle_won"])


class MbrFrameDeltaTests(unittest.TestCase):
    """Every MainBattleReader.read() inside a block-step loop MUST pass
    frames=_STEP_BLOCK. The default (14) breaks gMain re-discovery at the
    battle-end callback swap (6 != 14) and leaves the reader stale-True -
    which reintroduces the "battle never ends -> 25 invalid turns" bug."""

    def _reads_in(self, func_name):
        import ast
        import inspect
        src = inspect.getsource(getattr(EmulatorBattleDriver, func_name))
        tree = ast.parse(src.strip())
        bad = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "_mbr"):
                kw = {k.arg: k.value for k in node.keywords}
                ok = ("frames" in kw and isinstance(kw["frames"], ast.Name)
                      and kw["frames"].id == "_STEP_BLOCK")
                if not ok:
                    bad.append(ast.unparse(node))
        return bad

    def test_resolve_turn_reads_use_step_block_frames(self):
        self.assertEqual(self._reads_in("_resolve_turn"), [])

    def test_settle_to_readable_menu_reads_use_step_block_frames(self):
        self.assertEqual(self._reads_in("_settle_to_readable_menu"), [])


class ClassifyTurnOutcomeTests(unittest.TestCase):
    def _res(self, **kw):
        base = dict(in_battle=True, left_battle=False, enemy_ko_seen=False,
                    enemy_swapped=False, own_party_wiped=False,
                    own_party_wiped_late=False, menu_stall=False,
                    unexpected_menu=False, a_presses=0, menu_trace=[18],
                    reached_main_menu=False, frames=100, last_menu_raw=18)
        base.update(kw)
        return base

    def test_enemy_ko_then_battle_end_is_a_win(self):
        v = D.classify_turn_outcome("MOVE_1", self._res(
            enemy_ko_seen=True, left_battle=True, in_battle=False))
        self.assertTrue(v["battle_won"])
        self.assertTrue(v["enemy_ko"])          # a win IS a KO (counter credit)
        self.assertFalse(v["terminal_unknown"])

    def test_enemy_ko_with_a_replacement_keeps_the_battle_going(self):
        v = D.classify_turn_outcome("MOVE_1", self._res(
            enemy_ko_seen=True, enemy_swapped=True))
        self.assertTrue(v["enemy_ko"])
        self.assertFalse(v["battle_won"])

    def test_whole_party_at_zero_is_a_wipe(self):
        v = D.classify_turn_outcome("MOVE_2", self._res(own_party_wiped=True))
        self.assertTrue(v["wipe"])

    def test_run_that_ends_the_battle_is_fled(self):
        v = D.classify_turn_outcome("RUN", self._res(left_battle=True, in_battle=False))
        self.assertTrue(v["fled"])
        self.assertFalse(v["terminal_unknown"])

    def test_battle_ended_without_a_signal_is_terminal_unknown_not_timeout(self):
        v = D.classify_turn_outcome("MOVE_1", self._res(
            left_battle=True, in_battle=False, enemy_ko_seen=False))
        self.assertTrue(v["terminal_unknown"])
        self.assertFalse(v["battle_won"])
        self.assertIn("no win/wipe/flee signal", v["reason"])

    def test_stuck_off_the_menu_is_a_menu_stall(self):
        v = D.classify_turn_outcome("MOVE_1", self._res(
            menu_stall=True, in_battle=True, left_battle=False))
        self.assertTrue(v["menu_stall"])
        self.assertTrue(v["aborted"])

    def test_a_reappearing_selection_menu_is_a_menu_stall_not_a_message(self):
        v = D.classify_turn_outcome("MOVE_1", self._res(
            unexpected_menu=True, last_menu_raw=20, menu_trace=[18, 20]))
        self.assertTrue(v["menu_stall"])
        self.assertTrue(v["aborted"])
        self.assertFalse(v["terminal_unknown"])


if __name__ == "__main__":
    unittest.main()
