import unittest
from unittest.mock import patch

import battle_ram
import battle_types as bt


class _FakeEnv:
    def __init__(self, ram=None):
        self._ram = ram
    def get_ram(self):
        return self._ram


def raw_mon(species_id=7, slot=0, cur_hp=30, max_hp=44, level=12, status=0,
            move_ids=(55, 33, 39, 145)):
    return {
        "slot": slot, "species_id": species_id, "id": species_id,
        "name": "x", "level": level, "cur_hp": cur_hp, "max_hp": max_hp,
        "status": status, "checksum_ok": True, "personality": 1, "ot_id": 1,
        "stats": {"attack": 20, "defense": 22, "speed": 18,
                  "sp_attack": 21, "sp_defense": 23},
        "moves": [{"id": m, "name": None, "pp": 10} for m in move_ids if m > 0],
    }


class StatusAndEscapeTests(unittest.TestCase):
    def test_status_flags(self):
        self.assertTrue(battle_ram.status_flags(battle_ram.STATUS_BURN)["burn"])
        self.assertTrue(battle_ram.status_flags(3)["sleep"])   # sleep counter 3
        self.assertFalse(battle_ram.status_flags(0)["paralysis"])
        self.assertEqual(set(battle_ram.status_flags("bad")),
                         {"sleep", "poison", "burn", "freeze", "paralysis", "toxic"})
        self.assertFalse(any(battle_ram.status_flags("bad").values()))

    def test_can_escape_fail_closed(self):
        self.assertTrue(battle_ram.can_escape(0))                       # plain wild
        self.assertFalse(battle_ram.can_escape(battle_ram.BATTLE_TYPE_TRAINER))
        self.assertFalse(battle_ram.can_escape(battle_ram.BATTLE_TYPE_LINK))
        self.assertFalse(battle_ram.can_escape(battle_ram.BATTLE_TYPE_SAFARI))
        self.assertFalse(battle_ram.can_escape(None))                   # unknown
        self.assertFalse(battle_ram.can_escape("x"))
        self.assertTrue(battle_ram.can_escape(battle_ram.BATTLE_TYPE_WILD))

    def test_read_battle_type_flags_checked_distinguishes_zero_from_unknown(self):
        long_buf = b"\x00" * (battle_ram.BATTLE_TYPE_FLAGS_OFFSET + 8)
        flags, known = battle_ram.read_battle_type_flags_checked(_FakeEnv(long_buf))
        self.assertEqual((flags, known), (0, True))     # real wild-battle value
        short = b"\x00" * 16
        self.assertEqual(battle_ram.read_battle_type_flags_checked(_FakeEnv(short)),
                         (None, False))
        self.assertEqual(battle_ram.read_battle_type_flags_checked(_FakeEnv(None)),
                         (None, False))

    def test_escape_allowed_requires_verified_read_and_active_battle(self):
        self.assertTrue(battle_ram.escape_allowed(0, True, True))
        self.assertFalse(battle_ram.escape_allowed(0, False, True))     # unverified
        self.assertFalse(battle_ram.escape_allowed(0, True, None))      # battle not confirmed
        self.assertFalse(battle_ram.escape_allowed(0, True, False))
        self.assertFalse(battle_ram.escape_allowed(
            battle_ram.BATTLE_TYPE_TRAINER, True, True))
        self.assertFalse(battle_ram.escape_allowed(
            battle_ram.BATTLE_TYPE_ROAMER, True, True))
        self.assertFalse(battle_ram.escape_allowed(None, False, True))

    def test_is_trainer_battle(self):
        self.assertTrue(battle_ram.is_trainer_battle(battle_ram.BATTLE_TYPE_TRAINER | 1))
        self.assertFalse(battle_ram.is_trainer_battle(0))
        self.assertFalse(battle_ram.is_trainer_battle(None))


class AnnotateTests(unittest.TestCase):
    def test_annotate_adds_db_types_and_move_mechanics_without_ram(self):
        m = battle_ram._annotate_mon(raw_mon(species_id=7), None,
                                     battle_ram.PLAYER_PARTY_OFFSET)
        self.assertEqual(m["types"], [bt.TYPE_WATER])
        self.assertTrue(m["types_known"])
        self.assertIsNone(m["held_item"])       # no RAM -> unknown, not guessed
        self.assertFalse(m["stat_stages_known"])
        self.assertEqual(m["stat_stages"]["attack"], 0)
        wg = next(mv for mv in m["moves"] if mv["id"] == 55)
        self.assertEqual(wg["type"], bt.TYPE_WATER)
        self.assertEqual(wg["power"], 40)
        self.assertTrue(wg["mechanics_known"])

    def test_annotate_unknown_species_is_fail_closed(self):
        m = battle_ram._annotate_mon(raw_mon(species_id=9999), None,
                                     battle_ram.PLAYER_PARTY_OFFSET)
        self.assertIsNone(m["types"])
        self.assertFalse(m["types_known"])


class SnapshotTests(unittest.TestCase):
    def test_no_ram_snapshot_is_fully_fail_closed(self):
        snap = battle_ram.battle_snapshot(_FakeEnv(None))
        self.assertFalse(snap["ram_ok"])
        self.assertIsNone(snap["in_battle"])
        self.assertFalse(snap["can_escape"])
        self.assertFalse(snap["battle_type_flags_known"])
        self.assertIsNone(snap["battle_type_flags"])
        self.assertIsNone(snap["is_trainer"])
        self.assertFalse(snap["stat_stages_known"])
        self.assertFalse(snap["menu_cursor_known"])
        self.assertEqual(snap["player_party"], [])
        self.assertIn("gBattleMons", snap["unverified"])
        ready, missing = battle_ram.battle_snapshot_ready_for_execution(snap)
        self.assertFalse(ready)
        self.assertTrue(missing)

    def test_snapshot_flags_unknown_when_buffer_too_short(self):
        env = _FakeEnv(b"\x00" * 64)   # far shorter than gBattleTypeFlags offset
        with patch.object(battle_ram, "read_player_party", return_value=[]), \
             patch.object(battle_ram, "read_enemy_party", return_value=[]), \
             patch.object(battle_ram, "read_trainer_battle", return_value=(0, None)), \
             patch.object(battle_ram, "in_battle", return_value=(True, "x")):
            snap = battle_ram.battle_snapshot(env)
        self.assertFalse(snap["battle_type_flags_known"])
        self.assertIsNone(snap["battle_type_flags"])
        self.assertFalse(snap["can_escape"])          # unverified read -> no flee

    def test_ready_for_execution_always_false_in_phase1(self):
        # even a maximally-populated Phase-1 snapshot is not execution-ready
        env = _FakeEnv(b"\x00" * 0x50000)
        with patch.object(battle_ram, "read_player_party",
                          return_value=[raw_mon(7, 0, 30, 44)]), \
             patch.object(battle_ram, "read_enemy_party",
                          return_value=[raw_mon(19, 0, 14, 20)]), \
             patch.object(battle_ram, "read_trainer_battle", return_value=(0, None)), \
             patch.object(battle_ram, "in_battle", return_value=(True, "gMain.inBattle")):
            snap = battle_ram.battle_snapshot(env)
        ready, missing = battle_ram.battle_snapshot_ready_for_execution(snap)
        self.assertFalse(ready)
        self.assertIn("gBattlerPartyIndexes (authoritative active party slot)",
                      missing)
        self.assertTrue(any("gBattleMons" in m for m in missing))

    def test_snapshot_assembly_with_stubbed_reads(self):
        env = _FakeEnv(b"\x00" * 0x50000)
        with patch.object(battle_ram, "read_player_party",
                          return_value=[raw_mon(7, 0, 30, 44),
                                        raw_mon(16, 1, 22, 40)]), \
             patch.object(battle_ram, "read_enemy_party",
                          return_value=[raw_mon(19, 0, 14, 20)]), \
             patch.object(battle_ram, "read_battle_type_flags_checked",
                          return_value=(0, True)), \
             patch.object(battle_ram, "read_trainer_battle", return_value=(0, None)), \
             patch.object(battle_ram, "in_battle", return_value=(True, "gMain.inBattle")):
            snap = battle_ram.battle_snapshot(env)
        self.assertTrue(snap["ram_ok"])
        self.assertTrue(snap["in_battle"])
        self.assertTrue(snap["battle_type_flags_known"])
        self.assertFalse(snap["is_trainer"])
        self.assertTrue(snap["can_escape"])           # wild, flags 0, verified
        self.assertEqual(snap["player_active"]["species_id"], 7)   # first alive
        self.assertEqual(snap["enemy_active"]["species_id"], 19)
        self.assertFalse(snap["active_slot_authoritative"])
        self.assertFalse(snap["stat_stages_known"])

    def test_snapshot_active_skips_fainted(self):
        env = _FakeEnv(b"\x00" * 0x50000)
        with patch.object(battle_ram, "read_player_party",
                          return_value=[raw_mon(7, 0, 0, 44),      # fainted
                                        raw_mon(16, 1, 22, 40)]), \
             patch.object(battle_ram, "read_enemy_party", return_value=[]), \
             patch.object(battle_ram, "read_battle_type_flags_checked",
                          return_value=(battle_ram.BATTLE_TYPE_TRAINER, True)), \
             patch.object(battle_ram, "read_trainer_battle", return_value=(42, None)), \
             patch.object(battle_ram, "in_battle", return_value=(True, "x")):
            snap = battle_ram.battle_snapshot(env)
        self.assertEqual(snap["player_active"]["species_id"], 16)
        self.assertTrue(snap["is_trainer"])
        self.assertFalse(snap["can_escape"])          # trainer battle
        self.assertIsNone(snap["enemy_active"])
