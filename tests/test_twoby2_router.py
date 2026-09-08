import unittest

import twoby2.router as R
from twoby2 import feature_enabled


def V(number=1, sha="a", source="ppo", gen=1, obs="nav_obs_v1"):
    return R.PolicyVersion(number, sha, source, gen, obs)


NOT_READY = (False, ["menu cursors unverified"])
READY = (True, [])


class PolicyVersionTests(unittest.TestCase):
    def test_same_number_different_hash_is_not_the_same(self):
        self.assertFalse(V(3, "aaa").same_as(V(3, "bbb")))
        self.assertTrue(V(3, "aaa").same_as(V(3, "aaa")))
        self.assertFalse(V(3, "a", source="ppo").same_as(V(3, "a", source="rule")))


class ResolveBattlePolicyTests(unittest.TestCase):
    def test_ppo_champion_then_rule_then_blocked(self):
        self.assertEqual(R.resolve_battle_policy(
            battle_champion_valid=True, battle_champion_source="ppo",
            rule_controller_ready=True), R.POLICY_BATTLE_CHAMPION)
        self.assertEqual(R.resolve_battle_policy(
            battle_champion_valid=False, battle_champion_source="ppo",
            rule_controller_ready=True), R.POLICY_RULE_CONTROLLER)
        self.assertEqual(R.resolve_battle_policy(
            battle_champion_valid=False, battle_champion_source="rule",
            rule_controller_ready=False), R.POLICY_BLOCKED)


class ConsumerModeTests(unittest.TestCase):
    def _router(self, mode, **kw):
        return R.BattlePolicyRouter(
            consumer_mode=mode, pinned_battle_champion=V(4),
            battle_champion_valid=kw.get("valid", True),
            battle_champion_source=kw.get("source", "ppo"),
            rule_controller_ready=True)

    def test_nav_training_overworld_uses_learner(self):
        out = self._router(R.NAV_TRAINING).route(in_battle=False, execution_check=READY)
        self.assertEqual(out["policy"], R.POLICY_NAV_LEARNER)

    def test_nav_eval_overworld_uses_model_under_eval(self):
        out = self._router(R.NAV_EVAL).route(in_battle=False, execution_check=READY)
        self.assertEqual(out["policy"], R.POLICY_NAV_UNDER_EVAL)

    def test_watcher_overworld_uses_champion(self):
        out = self._router(R.WATCHER).route(in_battle=False, execution_check=READY)
        self.assertEqual(out["policy"], R.POLICY_NAV_CHAMPION)

    def test_consuming_modes_never_load_battle_learner(self):
        for mode in (R.NAV_TRAINING, R.NAV_EVAL, R.WATCHER):
            out = self._router(mode).route(in_battle=True, execution_check=READY)
            self.assertFalse(out["loads_battle_learner"])
            self.assertIn(out["policy"], (R.POLICY_BATTLE_CHAMPION,
                                          R.POLICY_RULE_CONTROLLER))

    def test_battle_training_uses_learner_only(self):
        out = self._router(R.BATTLE_TRAINING).route(in_battle=True, execution_check=READY)
        self.assertEqual(out["policy"], R.POLICY_BATTLE_LEARNER)
        self.assertTrue(out["loads_battle_learner"])

    def test_not_executable_while_gates_off(self):
        out = self._router(R.WATCHER).route(in_battle=True, execution_check=READY)
        self.assertFalse(feature_enabled("battle_router_live"))
        self.assertFalse(out["executable"])

    def test_raw_boolean_execution_check_is_rejected(self):
        out = self._router(R.WATCHER).route(in_battle=True, execution_check=True)
        self.assertFalse(out["executable"])
        self.assertIn("not a real check result", out["missing"][0])

    def test_pinned_version_is_reported_for_ppo_champion(self):
        out = self._router(R.NAV_TRAINING).route(in_battle=True, execution_check=READY)
        self.assertEqual(out["battle_policy_version"]["number"], 4)


class GenerationPinTests(unittest.TestCase):
    def test_forward_only_and_at_boundary(self):
        pin = R.GenerationPin(7, V(3))
        nxt = pin.next_generation(V(5))
        self.assertEqual((nxt.generation, nxt.battle_champion.number), (8, 5))
        back = nxt.next_generation(V(2))
        self.assertEqual(back.battle_champion.number, 5)


class EvalBatchPinTests(unittest.TestCase):
    def test_candidate_and_champion_share_one_battle_version(self):
        b = R.EvalBatchPin(V(3, "x"))
        self.assertTrue(b.version_for("candidate").same_as(b.version_for("champion")))
        self.assertTrue(b.comparable_with(R.EvalBatchPin(V(3, "x"))))
        self.assertFalse(b.comparable_with(R.EvalBatchPin(V(3, "y"))))


class WatcherBattlePinTests(unittest.TestCase):
    def test_no_swap_mid_battle_swap_next_battle(self):
        w = R.WatcherBattlePin(V(2))
        w.on_battle_start()
        w.observe_available(V(9))
        self.assertTrue(w.newer_ready())
        self.assertEqual(w.on_battle_start().number, 2)   # still battle A
        w.on_battle_end()
        self.assertEqual(w.on_battle_start().number, 9)   # battle B

    def test_forward_only(self):
        w = R.WatcherBattlePin(V(5))
        w.observe_available(V(3))
        w.on_battle_end()
        self.assertEqual(w.on_battle_start().number, 5)

    def test_never_loads_learner(self):
        self.assertFalse(R.WatcherBattlePin().loads_learner())


if __name__ == "__main__":
    unittest.main()
