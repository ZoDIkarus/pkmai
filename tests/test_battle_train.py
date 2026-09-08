import warnings
warnings.filterwarnings("ignore")

import json
import os
import tempfile
import unittest

import battle_types as bt
import battle_train as btrain
from battle_env import _mon, _mv


def easy_scenario(area="route1", trainer=False):
    return {
        "area": area, "is_trainer": trainer, "can_escape": not trainer,
        "our_party": [_mon((bt.TYPE_WATER,), level=18, cur_hp=55, max_hp=55,
                           spa=55, spe=50, moves=[_mv(bt.TYPE_WATER, 55, mid=1),
                                                  _mv(bt.TYPE_NORMAL, 40, mid=2)])],
        "enemy_party": [_mon((bt.TYPE_ROCK,), level=5, cur_hp=20, max_hp=20,
                             atk=15, spe=20, moves=[_mv(bt.TYPE_ROCK, 25, mid=9)])],
    }


class BattleTrainerStructureTests(unittest.TestCase):
    def test_counters_have_no_navigation_key(self):
        c = btrain.BattleCounters().to_dict()
        self.assertIn("flees", c)
        self.assertIn("timeouts", c)
        for bad in ("world_stage", "route_steps", "tiles", "story", "nav"):
            self.assertFalse(any(bad in k for k in c))

    def test_diagnostic_counters_exist_and_separate_timeout_from_unknown(self):
        c = btrain.BattleCounters().to_dict()
        for k in ("invalid_actions", "aborted_macros", "menu_stalls",
                  "unreadable_states", "terminal_unknown", "no_damage_turns"):
            self.assertIn(k, c)

        counters = btrain.BattleCounters()
        cb = btrain.BattleCounterCallback(counters).callback
        # 4 dones: a real win, a real clock timeout, an unclassifiable end,
        # a menu stall; plus a mid-battle invalid + no-damage turn.
        cb.locals = {
            "infos": [
                {"outcome": "win", "enemy_ko": True},
                {"outcome": "timeout"},
                {"outcome": "terminal_unknown", "terminal_unknown": True},
                {"outcome": "menu_stall", "menu_stall": True, "aborted": True},
                {"invalid": True, "no_damage": True},
            ],
            "dones": [True, True, True, True, False],
        }
        cb._on_step()
        d = counters.to_dict()
        self.assertEqual(d["wins"], 1)
        self.assertEqual(d["timeouts"], 1)          # ONLY the real clock timeout
        self.assertEqual(d["terminal_unknown"], 2)  # unknown end + menu stall
        self.assertEqual(d["menu_stalls"], 1)
        self.assertEqual(d["invalid_actions"], 1)
        self.assertEqual(d["no_damage_turns"], 1)
        self.assertEqual(d["episodes"], 4)

    def test_rolling_window_and_reward_breakdown_from_callback(self):
        counters = btrain.BattleCounters()
        cb = btrain.BattleCounterCallback(counters).callback
        # env 0: turn 1 deals damage, turn 2 KOs + wins
        cb.locals = {"infos": [{"our_damage_dealt": 5, "enemy_hp_before": 20}],
                     "dones": [False], "rewards": [0.04]}
        cb._on_step()
        cb.locals = {"infos": [{"outcome": "win", "enemy_ko": True,
                                "battle_won": True, "our_damage_dealt": 15,
                                "enemy_hp_before": 15}],
                     "dones": [True], "rewards": [4.14]}
        cb._on_step()
        roll = counters.rolling_dict()
        self.assertEqual(roll["window"], 1)
        self.assertEqual(roll["win_rate_200"], 1.0)
        self.assertAlmostEqual(roll["avg_reward_last_100"], 4.18, places=2)  # 0.04 + 4.14
        self.assertEqual(roll["avg_turns_on_win"], 2)
        comp = roll["reward_components"]
        self.assertGreater(comp["enemy_ko"], 0.9)
        self.assertGreater(comp["battle_win"], 2.9)
        self.assertGreater(comp["damage"], 0)

    def test_aggregate_eval_separates_win_only_hp_and_exposes_safety_metrics(self):
        rec = lambda **k: {"scenario_ix": k.get("ix", 0), "area": k.get("area", "route1"),
                           "battle_kind": k.get("kind", "wild"),
                           "won": k.get("won", False), "wiped": k.get("wiped", False),
                           "fled": k.get("fled", False),
                           "terminal_unknown": k.get("tu", False),
                           "timeout": k.get("to", False), "aborted": k.get("ab", False),
                           "menu_stall": k.get("ms", False), "unreadable": k.get("ur", False),
                           "illegal_flee": k.get("if_", False),
                           "residual_hp_frac": k.get("hp", 0.0), "turns": k.get("t", 3),
                           "invalid_actions": k.get("inv", 0), "switch_loops": 0}
        records = [
            rec(ix=0, kind="wild", won=True, hp=0.8, t=4),
            rec(ix=1, kind="trainer", won=True, hp=0.4, t=8),
            rec(ix=2, kind="wild", fled=True, hp=0.9, t=2, if_=False),
            rec(ix=3, kind="wild", wiped=True, hp=0.0, t=6, ms=True),
        ]
        agg = btrain._aggregate_eval(records)
        self.assertEqual(agg["scenario_coverage"], 4)
        self.assertEqual(agg["n_trainer"], 1)
        self.assertEqual(agg["n_wild"], 3)
        self.assertAlmostEqual(agg["flee_rate"], 0.25)
        # residual HP averaged only over the two WON battles (0.8, 0.4) -> 0.6,
        # not dragged up by the 0.9 flee
        self.assertAlmostEqual(agg["avg_residual_hp_on_win"], 0.6)
        self.assertAlmostEqual(agg["avg_turns_on_win"], 6.0)
        self.assertEqual(agg["menu_stalls"], 1)
        self.assertIn("per_kind_win_rate", agg)

    def test_ppo_n_steps_is_independent_of_train_module(self):
        self.assertEqual(btrain.BATTLE_PPO_N_STEPS, 256)
        import ast
        import pathlib
        t = ast.parse((pathlib.Path(__file__).resolve().parents[1] /
                       "src" / "train.py").read_text())
        vals = {x.id: n.value.value for n in t.body if isinstance(n, ast.Assign)
                for x in n.targets if isinstance(x, ast.Name)
                and isinstance(n.value, ast.Constant)}
        self.assertEqual(vals["PPO_N_STEPS"], 512)   # untouched

    def test_vec_env_has_nine_workers_by_default(self):
        from twoby2.config import BATTLE_WORKERS
        self.assertEqual(BATTLE_WORKERS, 9)
        vec = btrain.make_battle_vec_env(3, seed=0)   # small for the test
        self.assertEqual(vec.num_envs, 3)
        vec.close()

    def test_resume_uses_next_unseen_promotion_boundary(self):
        # fresh run: the early checks fire first
        self.assertEqual(btrain.next_promotion_step(0), 1_000)
        self.assertEqual(btrain.next_promotion_step(1_000), 2_000)
        self.assertEqual(btrain.next_promotion_step(2_000), 5_000)
        self.assertEqual(btrain.next_promotion_step(4_999), 5_000)
        self.assertEqual(btrain.next_promotion_step(6_500), 10_000)
        # then a steady 10k cadence
        self.assertEqual(btrain.next_promotion_step(10_000), 20_000)
        self.assertEqual(btrain.next_promotion_step(49_999), 50_000)
        self.assertEqual(btrain.next_promotion_step(50_000), 60_000)
        self.assertEqual(btrain.next_promotion_step(103_680), 110_000)

    def test_first_live_evaluation_waits_for_real_training_wins(self):
        counters = btrain.BattleCounters()
        counters.wins = btrain.MIN_TRAINING_WINS_BEFORE_FIRST_LIVE_EVAL - 1
        self.assertFalse(btrain.live_eval_ready(counters, has_champion=False))
        counters.wins += 1
        self.assertTrue(btrain.live_eval_ready(counters, has_champion=False))
        counters.wins = 0
        self.assertTrue(btrain.live_eval_ready(counters, has_champion=True))

    def test_live_emulator_env_is_gated(self):
        with self.assertRaises(RuntimeError):
            btrain.make_battle_vec_env(2, simulated=False)


class BattleTrainerLoopTests(unittest.TestCase):
    def test_train_chunk_advances_only_battle_counters_and_saves_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            t = btrain.BattleTrainer(n_workers=2, seed=0,
                                     ckpt_dir=os.path.join(d, "ck"),
                                     stats_path=os.path.join(d, "battle_stats.json"))
            t.setup(scenario_sampler=lambda rng: easy_scenario())
            t.train_chunk(512)
            self.assertGreater(t.counters.env_steps, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "ck", "battle_learner.zip")))
            stats = json.load(open(os.path.join(d, "battle_stats.json")))
            self.assertEqual(stats["n_workers"], 2)
            self.assertIn("env_steps", stats["counters"])
            self.assertNotIn("world_stage", json.dumps(stats))

    def test_promotion_path_runs_and_respects_the_coverage_gate(self):
        with tempfile.TemporaryDirectory() as d:
            t = btrain.BattleTrainer(n_workers=2, seed=0,
                                     ckpt_dir=os.path.join(d, "ck"),
                                     stats_path=os.path.join(d, "s.json"))
            t.setup(scenario_sampler=lambda rng: easy_scenario())
            t.train_chunk(1024)
            # one wild scenario -> the coverage gate must refuse, no champion
            dec = t.try_promote([easy_scenario("route1")])
            self.assertFalse(dec["promote"])
            self.assertIn("scenario", dec["reason"])
            self.assertFalse(os.path.exists(os.path.join(d, "ck", "battle_champion.zip")))

    def test_train_chunk_stops_exactly_on_the_promotion_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            t = btrain.BattleTrainer(n_workers=2, seed=0,
                                     ckpt_dir=os.path.join(d, "ck"),
                                     stats_path=os.path.join(d, "s.json"))
            t.setup(scenario_sampler=lambda rng: easy_scenario())
            start = int(t._model.num_timesteps)
            boundary = start + 300
            t.train_chunk(4096, stop_at=boundary)
            # the callback returns False the instant num_timesteps hits the
            # boundary -> it never carries a full rollout past it
            self.assertGreaterEqual(int(t._model.num_timesteps), boundary)
            self.assertLess(int(t._model.num_timesteps), boundary + t._vec.num_envs + 2)

    def test_promotion_path_can_promote_with_broad_clean_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            t = btrain.BattleTrainer(n_workers=2, seed=0,
                                     ckpt_dir=os.path.join(d, "ck"),
                                     stats_path=os.path.join(d, "s.json"))
            t.setup(scenario_sampler=lambda rng: easy_scenario())
            t.train_chunk(1024)
            evalset = [easy_scenario("route1"), easy_scenario("route22"),
                       easy_scenario("route2"), easy_scenario("route1", trainer=True),
                       easy_scenario("route22", trainer=True)]
            dec = t.try_promote(evalset)
            self.assertIsInstance(dec.get("candidate_episodes"), int)
            self.assertGreaterEqual(dec["candidate_episodes"],
                                    btrain.EVAL_SUITE_MIN_EPISODES)
            # champion file exists iff it promoted (safety + performance gates)
            champ = os.path.join(d, "ck", "battle_champion.zip")
            self.assertEqual(os.path.exists(champ), dec["promote"])

    def test_next_promotion_step_survives_a_bare_stats_write_and_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "battle_stats.json")
            t = btrain.BattleTrainer(n_workers=2, seed=0,
                                     ckpt_dir=os.path.join(d, "ck"), stats_path=sp)
            t.setup(scenario_sampler=lambda rng: easy_scenario())
            t.next_promotion_step = 150_000
            t._write_stats(extra={"phase": "training"})            # the on_progress path
            t.train_chunk(512)                                      # bare _write_stats()
            self.assertEqual(json.load(open(sp))["next_promotion_step"], 150_000)
            t2 = btrain.BattleTrainer(n_workers=2, seed=0,
                                      ckpt_dir=os.path.join(d, "ck2"), stats_path=sp)
            self.assertTrue(t2._restore_stats())
            self.assertEqual(t2.next_promotion_step, 150_000)

    def test_hard_regression_only_touches_battle_learner(self):
        from twoby2.battle_promotion import reset_scope_for_hard_regression
        sc = reset_scope_for_hard_regression()
        self.assertEqual(sc["resets"], ["battle_learner <- battle_champion"])
        for keep in ("navigation_learner", "navigation_champion", "curriculum",
                     "savestates", "exploration_memory"):
            self.assertIn(keep, sc["never_touches"])


if __name__ == "__main__":
    unittest.main()
