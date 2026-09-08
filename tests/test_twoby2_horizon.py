import ast
import json
import os
import pathlib
import tempfile
import unittest

import twoby2.horizon as hz
from twoby2 import feature_enabled


def _advance_kwargs(**over):
    base = dict(evaluated_full_runs=150, stage_reproduction_rate=0.9,
                transition_rates={1: 0.9, 2: 0.85}, retention_passed=True,
                candidate_passed=True)
    base.update(over)
    return base


class HorizonLadderTests(unittest.TestCase):
    def test_ladder_is_the_required_sequence_to_163840(self):
        self.assertEqual(hz.NAV_EPISODE_HORIZONS, [
            2000, 4000, 8000, 12000, 18000, 32768, 49152,
            65536, 98304, 131072, 163840])

    def test_feature_gate_default_off(self):
        self.assertFalse(feature_enabled("adaptive_nav_horizon"))

    def test_fresh_learner_starts_at_2000(self):
        st = hz.NavHorizonState.for_fresh_learner()
        self.assertEqual(st.current_horizon, 2000)
        self.assertEqual(st.current_horizon_index, 0)

    def test_ppo_n_steps_stays_512_and_is_not_a_horizon(self):
        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "train.py"
        tree = ast.parse(src.read_text())
        vals = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)
                and isinstance(n.value, ast.Constant)}
        self.assertEqual(vals.get("PPO_N_STEPS"), 512)
        self.assertNotIn(512, hz.NAV_EPISODE_HORIZONS)


class ChampionHorizonMigrationTests(unittest.TestCase):
    def test_confirmed_champion_not_reset_below_old_live_horizon(self):
        # thin metadata: only "confirmed" -> must still land at the old live
        # effective horizon (32768), NOT 2000 or a coarse stage guess.
        st = hz.NavHorizonState.from_champion({"confirmed": True})
        self.assertGreaterEqual(st.current_horizon, hz.OLD_LIVE_EFFECTIVE_HORIZON)
        self.assertEqual(st.current_horizon, 32768)

    def test_champion_horizon_from_real_signals_takes_the_max(self):
        st = hz.NavHorizonState.from_champion({
            "confirmed": True,
            "episode_horizon_used": 49152,
            "deepest_reliable_stage": 5,
            "median_arrival_steps": 40000,   # 1.5x -> 60000 -> rung >= 65536
        })
        self.assertGreaterEqual(st.current_horizon, 65536)

    def test_unconfirmed_champion_starts_small(self):
        st = hz.NavHorizonState.from_champion({"confirmed": False,
                                               "episode_horizon_used": 49152})
        self.assertEqual(st.current_horizon, 2000)

    def test_fresh_learner_has_no_champion(self):
        self.assertEqual(hz.starting_index_from_champion(None), 0)
        self.assertEqual(hz.starting_index_from_champion({}), 0)


class HorizonWorkerSplitTests(unittest.TestCase):
    def test_80_20_probe_split_every_worker_is_canonical_beginning(self):
        st = hz.NavHorizonState.for_fresh_learner()
        roles = st.assign_worker_horizons(40)
        self.assertEqual(len(roles), 40)
        self.assertTrue(all(r["start_kind"] == "beginning" for r in roles))
        self.assertTrue(all(r["canonical_start"] for r in roles))
        self.assertTrue(all(r["counts_as_beginning_full_run"] for r in roles))
        probes = [r for r in roles if r["probe"]]
        self.assertAlmostEqual(len(probes) / 40, 0.20, delta=0.05)
        self.assertTrue(all(r["horizon"] == 4000 for r in probes))
        self.assertTrue(all(r["horizon"] == 2000 for r in roles if not r["probe"]))

    def test_single_worker_still_probes(self):
        st = hz.NavHorizonState.for_fresh_learner()
        roles = st.assign_worker_horizons(1)
        self.assertTrue(roles[0]["probe"])

    def test_top_rung_has_no_probes(self):
        st = hz.NavHorizonState(len(hz.NAV_EPISODE_HORIZONS) - 1)
        roles = st.assign_worker_horizons(10)
        self.assertFalse(any(r["probe"] for r in roles))
        self.assertTrue(all(r["horizon"] == 163840 for r in roles))

    def test_no_anchor_role_exists(self):
        st = hz.NavHorizonState.for_fresh_learner()
        roles = st.assign_worker_horizons(40)
        self.assertFalse(any(r["start_kind"] == "anchor" for r in roles))
        self.assertFalse(hasattr(st, "horizon_for_worker"))  # anchor helper removed


class HorizonAdvancementTests(unittest.TestCase):
    def test_no_advance_below_min_sample(self):
        st = hz.NavHorizonState.for_fresh_learner()
        res = st.advance_if_ready(**_advance_kwargs(evaluated_full_runs=40))
        self.assertFalse(res["advanced"])
        self.assertEqual(st.current_horizon_index, 0)

    def test_no_advance_on_early_game_regression(self):
        st = hz.NavHorizonState.for_fresh_learner()
        self.assertFalse(st.advance_if_ready(
            **_advance_kwargs(retention_passed=False))["advanced"])

    def test_no_advance_on_weak_transition_or_failed_candidate(self):
        st = hz.NavHorizonState.for_fresh_learner()
        self.assertFalse(st.advance_if_ready(
            **_advance_kwargs(transition_rates={1: 0.9, 2: 0.5}))["advanced"])
        self.assertFalse(st.advance_if_ready(
            **_advance_kwargs(candidate_passed=False))["advanced"])

    def test_advances_exactly_one_rung(self):
        st = hz.NavHorizonState.for_fresh_learner()
        res = st.advance_if_ready(**_advance_kwargs())
        self.assertTrue(res["advanced"])
        self.assertEqual(st.current_horizon, 4000)
        self.assertEqual(st.generation, 1)

    def test_never_auto_descends(self):
        st = hz.NavHorizonState(4)
        st.advance_if_ready(**_advance_kwargs(evaluated_full_runs=0,
                                              stage_reproduction_rate=0.0,
                                              retention_passed=False,
                                              candidate_passed=False))
        self.assertEqual(st.current_horizon_index, 4)


class HorizonPersistenceTests(unittest.TestCase):
    def test_atomic_save_and_reload_keeps_deeper_rung(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "nav_horizon.json")
            st = hz.NavHorizonState(5)
            st.save_atomic(path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(
                [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f], [])
            reloaded = hz.NavHorizonState.load_or_new(path)
            self.assertEqual(reloaded.current_horizon_index, 5)

    def test_reload_never_regresses_to_a_shallower_stale_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nav_horizon.json")
            hz.NavHorizonState(2).save_atomic(path)
            # champion metadata says rung 5 -> must not drop to the file's 2
            st = hz.NavHorizonState.load_or_new(
                path, champion_metrics={"confirmed": True,
                                        "episode_horizon_used": 49152})
            self.assertGreaterEqual(st.current_horizon_index, 6)

    def test_roundtrip_has_all_required_fields(self):
        st = hz.NavHorizonState.for_fresh_learner()
        st.advance_if_ready(**_advance_kwargs(navigation_champion_version=7))
        d = st.to_dict()
        for k in ("current_horizon_index", "current_horizon_steps",
                  "advancement_reason", "evaluated_full_runs", "transition_rates",
                  "next_probe_count", "timestamp", "generation",
                  "navigation_champion_version", "battle_champion_version"):
            self.assertIn(k, d)
        self.assertEqual(hz.NavHorizonState.from_dict(d).navigation_champion_version, 7)


if __name__ == "__main__":
    unittest.main()
