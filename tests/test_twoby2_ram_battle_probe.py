import struct
import unittest

import twoby2.ram_battle_probe as rp


EW = 0x40000


def _ram():
    return bytearray(EW)


def _put_battlemon(ram, base, battler, species, level, cur_hp):
    b = base + rp.BATTLE_MON_SIZE * battler
    struct.pack_into("<H", ram, b + rp.BATTLE_MON_SPECIES_OFF, species)
    ram[b + rp.BATTLE_MON_LEVEL_OFF] = level
    struct.pack_into("<H", ram, b + rp.BATTLE_MON_HP_OFF, cur_hp)


def party(*mons):
    return [{"slot": i, "species_id": s, "level": lv, "cur_hp": hp}
            for i, (s, lv, hp) in enumerate(mons)]


def dump(ram, *, context, encounter_id, battle_kind="wild", menu_state="main",
         active_party_slot=0, party_list=None, expected_move_cursor=None,
         expected_action_cursor=None):
    return {"ram": bytes(ram), "context": context, "encounter_id": encounter_id,
            "battle_kind": battle_kind, "menu_state": menu_state,
            "active_party_slot": active_party_slot,
            "party": party_list if party_list is not None else party((7, 11, 25)),
            "expected_move_cursor": expected_move_cursor,
            "expected_action_cursor": expected_action_cursor}


class HonestBlockerTests(unittest.TestCase):
    def test_verified_bprd_offsets_are_primary_candidates(self):
        self.assertEqual(rp.CANDIDATE_OFFSETS["gBattlerPartyIndexes"][0], 0x23BCE)
        self.assertEqual(rp.CANDIDATE_OFFSETS["gBattleMons"][0], 0x23BE4)
        self.assertEqual(rp.CANDIDATE_OFFSETS["gActionSelectionCursor"][0], 0x23FF8)
        self.assertEqual(rp.CANDIDATE_OFFSETS["gMoveSelectionCursor"][0], 0x23FFC)
        self.assertEqual(rp.CANDIDATE_OFFSETS["battle_menu_state"][0], 0x22BC4)

    def test_zero_dumps_is_an_honest_blocker(self):
        rep = rp.blocker_report()
        self.assertFalse(rep["verified"])
        self.assertEqual(set(rep["mandatory_blockers"]), set(rp.MANDATORY_FIELDS))
        self.assertFalse(any(rep["mandatory_blockers"].values()))
        # weather is explicitly optional, not a mandatory blocker
        self.assertIn("gBattleWeather", rep["optional_information"])
        self.assertNotIn("gBattleWeather", rp.MANDATORY_FIELDS)

    def test_reference_tables_present_but_nothing_auto_verified(self):
        self.assertIn("gBattleMons", rp.POKEFIRERED_US_REFERENCE)
        self.assertIn("bt_c1", rp.INTEGRATION_GUESSES)
        rep = rp.blocker_report()
        self.assertFalse(any(rep["mandatory_blockers"].values()))


class EncounterIndependenceTests(unittest.TestCase):
    def test_two_dumps_same_encounter_is_not_enough(self):
        ram = _ram()
        d1 = dump(ram, context="a", encounter_id="enc01_wild")
        d2 = dump(ram, context="b", encounter_id="enc01_wild")   # SAME encounter
        r = rp.verify_field("gMoveSelectionCursor", 0x23D2F, [d1, d2])
        self.assertFalse(r["verified"])
        self.assertIn("DISTINCT encounters", r["reason"])

    def test_needs_at_least_two_in_battle_dumps(self):
        r = rp.verify_field("gActionSelectionCursor", 0x22BD2, [])
        self.assertFalse(r["verified"])


class MoveCursorTests(unittest.TestCase):
    def _dumps(self, off, values):
        ram = _ram()
        ds = []
        for i, (enc, ms, exp, val) in enumerate(values):
            r = _ram()
            if val is not None:
                r[off] = val
            ds.append(dump(r, context=f"move_cursor_{exp}", encounter_id=enc,
                           menu_state=ms, expected_move_cursor=exp))
        return ds

    def test_verified_when_cursor_matches_expected_across_encounters_with_negative(self):
        off = 0x23D2F
        ds = [
            dump(self._with(off, 0), context="move_cursor_0", encounter_id="enc01_wild",
                 menu_state="move", expected_move_cursor=0),
            dump(self._with(off, 2), context="move_cursor_2", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move", expected_move_cursor=2),
            dump(self._with(off, 9), context="oob", encounter_id="out_of_battle",
                 battle_kind="out_of_battle", menu_state="none"),
        ]
        r = rp.verify_field("gMoveSelectionCursor", off, ds)
        self.assertTrue(r["verified"], r["reason"])

    def test_wrong_cursor_value_fails(self):
        off = 0x23D2F
        ds = [
            dump(self._with(off, 1), context="move_cursor_0", encounter_id="enc01_wild",
                 menu_state="move", expected_move_cursor=0),      # says 0, reads 1
            dump(self._with(off, 2), context="move_cursor_2", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move", expected_move_cursor=2),
            dump(self._with(off, 0), context="oob", encounter_id="out_of_battle",
                 battle_kind="out_of_battle"),
        ]
        r = rp.verify_field("gMoveSelectionCursor", off, ds)
        self.assertFalse(r["verified"])

    def test_no_negative_context_fails(self):
        off = 0x23D2F
        ds = [
            dump(self._with(off, 0), context="move_cursor_0", encounter_id="enc01_wild",
                 menu_state="move", expected_move_cursor=0),
            dump(self._with(off, 3), context="move_cursor_3", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move", expected_move_cursor=3),
        ]
        r = rp.verify_field("gMoveSelectionCursor", off, ds)
        self.assertFalse(r["verified"])
        self.assertIn("negative", r["reason"])

    @staticmethod
    def _with(off, val):
        r = _ram()
        r[off] = val
        return r


class PartyIndexAndBattleMonsTests(unittest.TestCase):
    def test_battlemons_checked_against_the_switched_slot_not_slot0(self):
        base = 0x23BE4
        # after switching to slot 1, gBattleMons[player] should show slot-1 mon
        ram = _ram()
        _put_battlemon(ram, base, rp.PLAYER_BATTLER, 19, 4, 16)   # = slot 1
        the_party = party((7, 11, 25), (19, 4, 16))
        ds = [
            dump(ram, context="turn1", encounter_id="enc01_wild",
                 active_party_slot=1, party_list=the_party),
            dump(ram, context="after_switch", encounter_id="enc02_trainer",
                 battle_kind="trainer", active_party_slot=1, party_list=the_party),
        ]
        r = rp.verify_field("gBattleMons", base, ds)
        self.assertTrue(r["verified"], r["reason"])

    def test_battlemons_against_wrong_slot_fails(self):
        base = 0x23BE4
        ram = _ram()
        _put_battlemon(ram, base, rp.PLAYER_BATTLER, 19, 4, 16)   # slot-1 mon
        the_party = party((7, 11, 25), (19, 4, 16))
        ds = [
            # sidecar wrongly says slot 0 is active -> cross-check must fail
            dump(ram, context="turn1", encounter_id="enc01_wild",
                 active_party_slot=0, party_list=the_party),
            dump(ram, context="after_switch", encounter_id="enc02_trainer",
                 battle_kind="trainer", active_party_slot=0, party_list=the_party),
        ]
        r = rp.verify_field("gBattleMons", base, ds)
        self.assertFalse(r["verified"])

    def test_party_indexes_must_change_after_a_switch(self):
        off = 0x23BCE
        # value stays 0 even though there is an after_switch dump saying slot 1
        ram = _ram()
        struct.pack_into("<H", ram, off, 0)
        ds = [
            dump(ram, context="turn1", encounter_id="enc01_wild", active_party_slot=0),
            dump(ram, context="after_switch_1", encounter_id="enc02_trainer",
                 battle_kind="trainer", active_party_slot=1),
        ]
        r = rp.verify_field("gBattlerPartyIndexes", off, ds)
        self.assertFalse(r["verified"])

    def test_party_indexes_verified_when_it_tracks_the_switch(self):
        off = 0x23BCE
        r0, r1 = _ram(), _ram()
        struct.pack_into("<H", r0, off, 0)
        struct.pack_into("<H", r1, off, 1)
        ds = [
            dump(r0, context="turn1", encounter_id="enc01_wild", active_party_slot=0),
            dump(r1, context="after_switch_1", encounter_id="enc02_trainer",
                 battle_kind="trainer", active_party_slot=1),
        ]
        r = rp.verify_field("gBattlerPartyIndexes", off, ds)
        self.assertTrue(r["verified"], r["reason"])


class MenuStateTests(unittest.TestCase):
    def test_needs_main_move_party_and_out_of_battle_with_distinct_stable_values(self):
        off = 0x22BD0
        def r(v):
            x = _ram(); x[off] = v; return x
        ds = [
            dump(r(1), context="m", encounter_id="enc01_wild", menu_state="main"),
            dump(r(2), context="v", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move"),
            dump(r(3), context="p", encounter_id="enc01_wild", menu_state="party"),
            dump(r(0), context="oob", encounter_id="out_of_battle",
                 battle_kind="out_of_battle"),
        ]
        res = rp.verify_field("battle_menu_state", off, ds)
        self.assertTrue(res["verified"], res["reason"])

    def test_missing_out_of_battle_context_fails(self):
        off = 0x22BD0
        def r(v):
            x = _ram(); x[off] = v; return x
        ds = [
            dump(r(1), context="m", encounter_id="enc01_wild", menu_state="main"),
            dump(r(2), context="v", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move"),
        ]
        res = rp.verify_field("battle_menu_state", off, ds)
        self.assertFalse(res["verified"])
        self.assertIn("out_of_battle", res["reason"])

    def test_two_menus_sharing_a_value_fails(self):
        off = 0x22BD0
        def r(v):
            x = _ram(); x[off] = v; return x
        ds = [
            dump(r(1), context="m", encounter_id="enc01_wild", menu_state="main"),
            dump(r(1), context="v", encounter_id="enc02_trainer",
                 battle_kind="trainer", menu_state="move"),   # same value as main
            dump(r(3), context="p", encounter_id="enc01_wild", menu_state="party"),
            dump(r(0), context="oob", encounter_id="out_of_battle",
                 battle_kind="out_of_battle"),
        ]
        res = rp.verify_field("battle_menu_state", off, ds)
        self.assertFalse(res["verified"])


class VerifyAllTests(unittest.TestCase):
    def test_scorer_gate_all_mandatory(self):
        results = {f: {"verified": True} for f in rp.MANDATORY_FIELDS}
        self.assertTrue(rp.all_mandatory_verified(results))
        results["gBattleMons"]["verified"] = False
        self.assertFalse(rp.all_mandatory_verified(results))

    def test_verify_all_reports_mandatory_flag(self):
        out = rp.verify_all([])
        self.assertTrue(out["gBattleMons"]["mandatory"])
        self.assertFalse(out["gBattleWeather"]["mandatory"])
        self.assertFalse(rp.all_mandatory_verified(out))


if __name__ == "__main__":
    unittest.main()
