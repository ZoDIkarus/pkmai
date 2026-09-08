import os
import tempfile
import unittest

import twoby2.scenario_pool as sp


def scen(area="route1", kind="wild", enemy=(16,), enemy_lv=(3,), own=(7,),
         own_lv=(12,), hp=(30,), pp=((15, 15),), status=(0,), ref=None,
         rom="ROMHASH"):
    return sp.BattleScenario(
        area, kind, ref or f"{area}-{enemy}-{hp}-{status}",
        rom_sha256=rom, enemy_species=enemy, enemy_levels=enemy_lv,
        own_species=own, own_levels=own_lv, own_hp=hp, own_pp=pp,
        own_status=status, story_flag_count=5)


def unlock(led, area, *, seeds=("s1", "s2"), runs=60, rate=0.9, ver=3):
    for s in seeds:
        led.record_navigation_evidence(area, beginning_runs=runs, reach_rate=rate,
                                       seed=s, champion_version=ver)


class UnlockLedgerTests(unittest.TestCase):
    def test_needs_50_runs_80pct_two_seeds_and_a_champion(self):
        led = sp.AreaUnlockLedger()
        led.record_navigation_evidence("route1", beginning_runs=40, reach_rate=0.9,
                                       seed="a", champion_version=3)
        self.assertFalse(led.is_trainable("route1"))          # <50 runs
        led.record_navigation_evidence("route1", beginning_runs=60, reach_rate=0.7,
                                       seed="a", champion_version=3)
        self.assertFalse(led.is_trainable("route1"))          # <80% and 1 seed
        led.record_navigation_evidence("route1", beginning_runs=60, reach_rate=0.9,
                                       seed="b", champion_version=3)
        self.assertTrue(led.is_trainable("route1"))

    def test_two_lucky_runs_do_not_unlock(self):
        led = sp.AreaUnlockLedger()
        led.record_navigation_evidence("route1", beginning_runs=2, reach_rate=1.0,
                                       seed="a", champion_version=3)
        led.record_navigation_evidence("route1", beginning_runs=2, reach_rate=1.0,
                                       seed="b", champion_version=3)
        self.assertFalse(led.is_trainable("route1"))

    def test_recent_and_older_split(self):
        led = sp.AreaUnlockLedger()
        for a in ("pallet", "route1", "viridian", "route2", "forest"):
            unlock(led, a)
        recent, older = led.recent_and_older(window=3)
        self.assertEqual(recent, ["viridian", "route2", "forest"])
        self.assertEqual(older, ["pallet", "route1"])

    def test_ledger_roundtrip(self):
        led = sp.AreaUnlockLedger()
        unlock(led, "route1")
        led2 = sp.AreaUnlockLedger.from_dict(led.to_dict())
        self.assertTrue(led2.is_trainable("route1"))


class ScenarioTests(unittest.TestCase):
    def test_material_differences_are_not_deduped(self):
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        self.assertEqual(pool.capture(scen(hp=(30,)))[0], True)
        # different HP -> different scenario
        self.assertEqual(pool.capture(scen(hp=(12,)))[0], True)
        # different status -> different scenario
        self.assertEqual(pool.capture(scen(hp=(30,), status=(0x40,)))[0], True)
        # identical -> dedup
        self.assertEqual(pool.capture(scen(hp=(30,)))[0], False)
        self.assertEqual(len(pool), 3)

    def test_quarantine_on_rom_mismatch_and_bad_data(self):
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        added, reason = pool.capture(scen(rom="WRONG"))
        self.assertFalse(added)
        self.assertIn("quarantined", reason)
        added, reason = pool.capture(sp.BattleScenario("route1", "wild", "ref",
                                     rom_sha256="ROMHASH", enemy_species=(),
                                     own_species=(7,)))
        self.assertFalse(added)

    def test_capture_only_from_validated_battle_and_is_reward_free(self):
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        self.assertEqual(pool.capture(scen(), from_validated_battle=False)[0], False)
        from twoby2.battle_env import capture_is_reward_free
        self.assertTrue(capture_is_reward_free())

    def test_size_cap_keeps_regression_core(self):
        pool = sp.ScenarioPool(max_size=5, rom_sha256="ROMHASH")
        core = scen(enemy=(1,), ref="core")
        pool.capture(core)
        pool.pin_regression_core(core.signature)
        for i in range(30):
            pool.capture(scen(enemy=(i + 100,), ref=f"r{i}"))
        self.assertLessEqual(len(pool), 6)
        self.assertIn(core.signature, pool._by_sig)


class SamplePlanTests(unittest.TestCase):
    def _ready_ledger(self, areas):
        led = sp.AreaUnlockLedger()
        for a in areas:
            unlock(led, a)
        return led

    def test_route1_focus_at_least_5_workers(self):
        led = self._ready_ledger(["route1"])
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        for i in range(4):
            pool.capture(scen(area="route1", enemy=(i + 20,), ref=f"s{i}"))
        out = pool.sample_plan(led, battle_workers=10)
        self.assertEqual(len(out["plan"]), 10)
        self.assertGreaterEqual(out["counts"]["recent"], 5)
        self.assertTrue(out["route1_focus"])

    def test_60_25_15_mix_with_core(self):
        led = self._ready_ledger(["a1", "a2", "a3", "a4", "a5"])
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        for a in ("a1", "a2", "a3", "a4", "a5"):
            pool.capture(scen(area=a, enemy=(hash(a) % 200,), ref=a))
        core = scen(area="a1", enemy=(199,), ref="core")
        pool.capture(core)
        pool.pin_regression_core(core.signature)
        out = pool.sample_plan(led, battle_workers=10)
        self.assertEqual(sum(out["counts"].values()), 10)
        self.assertGreaterEqual(out["counts"]["regression_core"], 1)
        self.assertGreaterEqual(out["counts"]["older"], 1)

    def test_concrete_scenarios_are_assigned_not_just_bucket_names(self):
        led = self._ready_ledger(["route1"])
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        pool.capture(scen(area="route1", ref="one"))
        out = pool.sample_plan(led, battle_workers=3)
        assigned = [p for p in out["plan"] if p["scenario_signature"]]
        self.assertTrue(assigned)
        self.assertTrue(all(p["scenario_area"] == "route1" for p in assigned))

    def test_empty_bucket_redistributes_and_still_returns_worker_count(self):
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        out = pool.sample_plan(sp.AreaUnlockLedger(), battle_workers=10)
        self.assertEqual(len(out["plan"]), 10)

    def test_coverage_metrics_by_area_and_kind(self):
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        pool.capture(scen(area="route1", kind="wild", ref="w"))
        pool.capture(scen(area="route1", kind="trainer", ref="t"))
        cov = pool.coverage()
        self.assertEqual(cov["by_kind"]["wild"], 1)
        self.assertEqual(cov["by_kind"]["trainer"], 1)
        self.assertEqual(cov["by_area"]["route1"], 2)


class RouteGroupPlanTests(unittest.TestCase):
    def _pool_and_ledger(self, routes, scen_per_route=2):
        led = sp.AreaUnlockLedger()
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        for r in routes:
            unlock(led, r)
            for i in range(scen_per_route):
                pool.capture(scen(area=r, enemy=(hash((r, i)) % 300,),
                                  ref=f"{r}-{i}"))
        return pool, led

    def test_only_route1_open_three_dedicated_plus_fill_and_watcher_route1(self):
        pool, led = self._pool_and_ledger(["route1"])
        out = pool.route_group_plan(led, headless_workers=9)
        self.assertEqual(out["active_routes"], ["route1"])
        self.assertEqual(out["headless_count"], 9)
        group0 = [a for a in out["headless_assignments"] if a["group"] == 0]
        self.assertEqual(len(group0), 3)
        self.assertTrue(all(a["route"] == "route1" for a in group0))
        # no invented later route anywhere
        self.assertTrue(all(a["route"] in ("route1", None)
                            for a in out["headless_assignments"]))
        self.assertEqual(out["visible_watcher"]["route"], "route1")
        self.assertEqual(out["visible_watcher"]["switch_policy"],
                         "after_current_battle_only")

    def test_route1_2_3_three_each_watcher_route3(self):
        pool, led = self._pool_and_ledger(["route1", "route2", "route3"])
        out = pool.route_group_plan(led, headless_workers=9)
        self.assertEqual(out["active_routes"], ["route1", "route2", "route3"])
        for gi, r in enumerate(["route1", "route2", "route3"]):
            g = [a for a in out["headless_assignments"] if a["group"] == gi]
            self.assertEqual(len(g), 3)
            self.assertTrue(all(a["route"] == r for a in g))
        self.assertEqual(out["visible_watcher"]["route"], "route3")

    def test_route4_new_active_is_2_3_4_route1_is_core(self):
        pool, led = self._pool_and_ledger(["route1", "route2", "route3", "route4"])
        out = pool.route_group_plan(led, headless_workers=9)
        self.assertEqual(out["active_routes"], ["route2", "route3", "route4"])
        self.assertIn("route1", out["regression_core_routes"])
        self.assertEqual(out["visible_watcher"]["route"], "route4")

    def test_route_reached_but_no_scenario_is_not_activated(self):
        led = sp.AreaUnlockLedger()
        unlock(led, "route1"); unlock(led, "route2")
        pool = sp.ScenarioPool(rom_sha256="ROMHASH")
        pool.capture(scen(area="route1", ref="r1"))     # route2 has NO scenario
        out = pool.route_group_plan(led, headless_workers=9)
        self.assertEqual(out["active_routes"], ["route1"])   # route2 skipped
        self.assertEqual(out["visible_watcher"]["route"], "route1")

    def test_all_nine_headless_get_a_concrete_or_idle_slot(self):
        pool, led = self._pool_and_ledger(["route1"])
        out = pool.route_group_plan(led, headless_workers=9)
        self.assertEqual(len(out["headless_assignments"]), 9)
        # every assignment names a route+signature or is explicitly idle
        for a in out["headless_assignments"]:
            self.assertTrue(a["scenario_signature"] is not None or a["group"] == "idle")


class PoolPersistenceTests(unittest.TestCase):
    def test_atomic_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x", "scenario_pool.json")
            pool = sp.ScenarioPool(rom_sha256="ROMHASH")
            a = scen(ref="a")
            pool.capture(a)
            pool.pin_regression_core(a.signature)
            pool.note_trained(a.signature)
            pool.save_atomic(path)
            self.assertEqual(
                [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f], [])
            back = sp.ScenarioPool.load(path)
            self.assertEqual(len(back), 1)
            self.assertIn(a.signature, back.regression_core)
            self.assertEqual(back.train_counts[a.signature], 1)


if __name__ == "__main__":
    unittest.main()
