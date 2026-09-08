import unittest

import twoby2.isolation as iso


class ModelPathGuardTests(unittest.TestCase):
    def test_battle_cannot_write_navigation_paths(self):
        for p in ("runtime/checkpoints/navigation_learner.zip",
                  "runtime/checkpoints/pokemon_model_champion.zip",
                  "runtime/checkpoints/pokemon_model_resume.zip",
                  "runtime/curriculum_states/stage_3.json",
                  "runtime/exploration_memory/x.json",
                  "runtime/nav_optimizer.pt"):
            self.assertFalse(iso.ModelPathGuard.battle_may_write(p), p)

    def test_battle_may_write_its_own_paths(self):
        for p in ("runtime/checkpoints/battle_learner.zip",
                  "runtime/battle_scenarios/pool.json",
                  "runtime/battle_optimizer.pt"):
            self.assertTrue(iso.ModelPathGuard.battle_may_write(p), p)

    def test_navigation_cannot_write_battle_paths(self):
        self.assertFalse(iso.ModelPathGuard.navigation_may_write(
            "runtime/checkpoints/battle_champion.zip"))
        self.assertTrue(iso.ModelPathGuard.navigation_may_write(
            "runtime/checkpoints/navigation_champion.zip"))


class SystemCounterTests(unittest.TestCase):
    def test_counter_sets_are_disjoint(self):
        nav, bat = iso.navigation_counters(), iso.battle_counters()
        self.assertEqual(set(nav.snapshot()) & set(bat.snapshot()), set())

    def test_battle_counter_rejects_nav_keys(self):
        bat = iso.battle_counters()
        with self.assertRaises(KeyError):
            bat.add("world_stage", 1)
        with self.assertRaises(KeyError):
            bat.add("nav_env_steps", 1)


class RolloutIsolationTests(unittest.TestCase):
    def setUp(self):
        self.nav = iso.IsolatedSystem("navigation", iso.navigation_counters())
        self.bat = iso.IsolatedSystem("battle", iso.battle_counters())

    def test_systems_do_not_share_buffer_or_optimizer(self):
        self.assertTrue(iso.assert_systems_isolated(self.nav, self.bat))

    def test_detects_a_shared_buffer(self):
        self.bat.rollout_buffer = self.nav.rollout_buffer
        with self.assertRaises(AssertionError):
            iso.assert_systems_isolated(self.nav, self.bat)

    def test_battle_step_cannot_mutate_navigation_counters(self):
        self.nav.counters.add("nav_env_steps", 500)
        self.nav.record_rollout({"s": 1})
        with iso.NavigationImmutableDuringBattle(self.nav):
            # a whole battle episode's worth of work
            for _ in range(50):
                self.bat.counters.add("battle_env_steps", 1)
                self.bat.record_rollout({"macro": "MOVE_1"})
            self.bat.counters.add("battle_wins", 1)
            self.bat.counters.add("battle_kos", 3)

    def test_guard_catches_an_accidental_nav_mutation(self):
        with self.assertRaises(AssertionError):
            with iso.NavigationImmutableDuringBattle(self.nav):
                self.nav.counters.add("world_stage", 1)      # forbidden leak

    def test_guard_catches_nav_rollout_append(self):
        with self.assertRaises(AssertionError):
            with iso.NavigationImmutableDuringBattle(self.nav):
                self.nav.rollout_buffer.append({"leaked": True})


if __name__ == "__main__":
    unittest.main()
