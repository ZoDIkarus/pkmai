"""Verified live in-battle readers, checked against the 20 REAL dumps in
runtime/ram_probe/20260907_184350 (not synthetic)."""
import glob
import gzip
import json
import os
import unittest

from twoby2 import battle_ram_live as L

DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "runtime", "ram_probe",
                        "20260907_184350")
DUMP_DIR = os.path.abspath(DUMP_DIR)


def _dumps():
    out = []
    for jf in sorted(glob.glob(os.path.join(DUMP_DIR, "*.json"))):
        rf = jf[:-5] + ".ram.gz"
        if not os.path.exists(rf):
            continue
        with open(jf) as f:
            m = json.load(f)
        with gzip.open(rf, "rb") as f:
            m["ram"] = f.read()
        out.append(m)
    return out


@unittest.skipUnless(os.path.isdir(DUMP_DIR) and _dumps(),
                     "verified RAM-probe dataset not present")
class LiveReaderRealDumpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dumps = _dumps()

    def test_dataset_covers_wild_trainer_and_out_of_battle(self):
        kinds = {d["battle_kind"] for d in self.dumps}
        self.assertIn("wild", kinds)
        # at least one dump was captured in the trainer fight (enemy L9)
        self.assertTrue(any(L.read_battle_mon(d["ram"], 1)["level"] == 9
                            for d in self.dumps if d["battle_kind"] != "out_of_battle"))

    def test_battle_mons_matches_active_party_slot_never_slot0_after_switch(self):
        switched = [d for d in self.dumps if d.get("active_party_slot") == 1]
        self.assertTrue(switched, "dataset must contain a post-switch dump")
        for d in self.dumps:
            if d["battle_kind"] == "out_of_battle":
                continue
            live = L.read_all(d["ram"])
            ok, detail = L.crosscheck_against_party(live, d["party"])
            self.assertTrue(ok, f"{d['label']}: {detail}")

    def test_action_cursor_matches_every_labelled_position(self):
        seen = set()
        for d in self.dumps:
            exp = d["expected_action_cursor"]
            if exp is None:
                continue
            self.assertEqual(L.action_cursor(d["ram"]), exp, d["label"])
            seen.add(exp)
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_move_cursor_matches_every_labelled_position(self):
        seen = set()
        for d in self.dumps:
            exp = d["expected_move_cursor"]
            if exp is None:
                continue
            self.assertEqual(L.move_cursor(d["ram"]), exp, d["label"])
            seen.add(exp)
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_menu_state_is_stable_and_distinct(self):
        by = {}
        for d in self.dumps:
            if d["battle_kind"] == "out_of_battle":
                self.assertIsNone(L.menu_state(d["ram"]))   # fail-closed
                continue
            ms = L.menu_state(d["ram"])
            if d["menu_state"] in ("main", "move", "party"):
                self.assertEqual(ms, d["menu_state"], d["label"])
                by.setdefault(d["menu_state"], set()).add(L.menu_state_raw(d["ram"]))
        self.assertEqual(by["main"], {L.MENU_CHOOSEACTION})
        self.assertEqual(by["move"], {L.MENU_CHOOSEMOVE})
        self.assertEqual(by["party"], {L.MENU_CHOOSEPOKEMON})

    def test_full_battlemon_struct_is_sane(self):
        d = next(d for d in self.dumps if d["label"] == "action_cursor_fight")
        pm = L.read_battle_mon(d["ram"], 0)
        self.assertEqual(pm["species_id"], 7)         # Squirtle
        self.assertEqual(pm["level"], 11)
        self.assertEqual((pm["cur_hp"], pm["max_hp"]), (25, 31))
        self.assertEqual(len(pm["moves"]), 4)
        self.assertTrue(all(1 <= m["id"] <= 355 for m in pm["moves"]))
        self.assertIn("attack", pm["stat_stages"])
        em = L.read_battle_mon(d["ram"], 1)
        self.assertEqual(em["species_id"], 16)        # Pidgey

    def test_out_of_battle_reads_are_flagged_not_trusted(self):
        oob = [d for d in self.dumps if d["battle_kind"] == "out_of_battle"]
        self.assertTrue(oob)
        for d in oob:
            self.assertIsNone(L.menu_state(d["ram"]))

    def test_weather_is_clear_in_the_dataset(self):
        for d in self.dumps:
            self.assertEqual(L.weather(d["ram"]), 0)


class LiveReaderFailClosedTests(unittest.TestCase):
    def test_none_ram_raises(self):
        with self.assertRaises(L.LiveReadError):
            L.read_all(None)

    def test_short_ram_returns_none_fields(self):
        short = bytes(0x1000)
        self.assertIsNone(L.action_cursor(short))
        self.assertIsNone(L.read_battle_mon(short, 0))

    def test_garbage_species_rejected(self):
        ram = bytearray(0x40000)   # all zero -> species 0 -> invalid
        self.assertIsNone(L.read_battle_mon(bytes(ram), 0))


if __name__ == "__main__":
    unittest.main()
