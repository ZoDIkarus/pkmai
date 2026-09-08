import unittest

import twoby2
import twoby2.activation as act
import twoby2.battle_env as tbe


READY_SNAPSHOT = {
    "in_battle": True, "battle_type_flags_known": True, "stat_stages_known": True,
    "active_slot_authoritative": True, "menu_cursor_known": True,
    "player_active": {"slot": 0}, "enemy_active": {"slot": 0},
    "can_escape": True, "is_trainer": False,
}


class ActivationGateTests(unittest.TestCase):
    def test_all_gates_closed_by_default(self):
        self.assertTrue(twoby2.all_gates_closed())

    def test_live_blocked_without_snapshot(self):
        self.assertFalse(act.live_battle_allowed(None)[0])
        with self.assertRaises(act.LiveActivationBlocked):
            act.assert_live_battle_allowed(None)

    def test_live_blocked_by_gate_even_if_snapshot_synthetically_ready(self):
        ok, reasons = act.live_battle_allowed(READY_SNAPSHOT)
        self.assertFalse(ok)
        self.assertTrue(any("battle_executor_live" in r for r in reasons))

    def test_action_specific_readiness(self):
        for action in ("MOVE_1", "SWITCH_2", "RUN"):
            ok, missing = act.snapshot_ready_for_action({}, action)
            self.assertFalse(ok)
            self.assertTrue(missing)

    def test_status_report_is_dynamic_not_hardcoded(self):
        rep = act.status_report()
        self.assertFalse(rep["live_battle_possible"])
        self.assertFalse(rep["snapshot_execution_ready"])
        rep2 = act.status_report(READY_SNAPSHOT)
        # gate still closed -> still not possible, but the snapshot field is real
        self.assertFalse(rep2["live_battle_possible"])
        self.assertTrue(rep2["snapshot_execution_ready"])   # not hardcoded False
        self.assertEqual(set(rep["phase2_ram_blockers"]), set(act.PHASE2_RAM_BLOCKERS))

    def test_router_and_ppo_champion_guards(self):
        with self.assertRaises(act.LiveActivationBlocked):
            act.assert_not_live_router()
        self.assertFalse(act.battle_ppo_may_be_live_champion(
            eval_passed=True, executor_verified=True, migration_done=True))

    def test_battle_env_spec_action_mask_none_is_all_zeros(self):
        mask = tbe.BattleEnvSpec.action_mask(None)
        self.assertEqual(list(mask), [0] * len(tbe.MACRO_ACTIONS))
        mask2 = tbe.BattleEnvSpec.action_mask({})
        self.assertEqual(list(mask2), [0] * len(tbe.MACRO_ACTIONS))


if __name__ == "__main__":
    unittest.main()
