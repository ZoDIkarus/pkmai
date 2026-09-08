"""Battle-PPO trainer (Phase 2). Fully separate from ``train.py``.

Own PPO, own rollout buffer, own optimizer, own step / update / episode
counters, own checkpoint paths, own logs. 9 battle workers by default. It
never reads or writes a navigation model, curriculum, savestate or exploration
memory. Promotion and auto-rollback happen ONLY within the battle system.

Live emulator battle roll-outs are gated (``FEATURES['battle_env']``); until the
battle-menu RAM is verified the trainer runs on :class:`battle_env.BattleEnv`
with the deterministic :class:`battle_env.SimulatedBattleDriver`, which is
enough to bring the Battle-PPO up and to exercise every promotion path.
"""
from __future__ import annotations

import collections
import json
import os
import tempfile
import time
import argparse

from twoby2 import feature_enabled
from twoby2.config import BATTLE_WORKERS
from twoby2.battle_promotion import (evaluate_battle_promotion, first_champion_ok,
                                     battle_catch_gate, EVAL_SUITE_MIN_EPISODES)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
BATTLE_DIR = os.path.join(RUNTIME_DIR, "battle")
CKPT_DIR = os.path.join(BATTLE_DIR, "checkpoints")

BATTLE_LEARNER = os.path.join(CKPT_DIR, "battle_learner.zip")
BATTLE_CANDIDATE = os.path.join(CKPT_DIR, "battle_candidate.zip")
BATTLE_CHAMPION = os.path.join(CKPT_DIR, "battle_champion.zip")
BATTLE_RESUME = os.path.join(CKPT_DIR, "battle_resume.zip")
BATTLE_STATS = os.path.join(BATTLE_DIR, "battle_stats.json")
BATTLE_LOG = os.path.join(BATTLE_DIR, "battle_train.log")

# --- Catch-v2 migration (spec §14) --------------------------------------
# v1 (combat-only) artifacts above stay UNTOUCHED and remain the live combat
# fallback. v2 (adds CATCH) trains into a fully SEPARATE set of files; a v1
# model is never loaded with a v2 obs/action space and a v2 model is never
# saved into a v1 file. The manifest records which schema each file carries
# and which one is live.
BATTLE_LEARNER_V2 = os.path.join(CKPT_DIR, "battle_learner_v2.zip")
BATTLE_CANDIDATE_V2 = os.path.join(CKPT_DIR, "battle_candidate_v2.zip")
BATTLE_CHAMPION_V2 = os.path.join(CKPT_DIR, "battle_champion_v2.zip")
BATTLE_RESUME_V2 = os.path.join(CKPT_DIR, "battle_resume_v2.zip")
BATTLE_MODEL_MANIFEST = os.path.join(BATTLE_DIR, "battle_model_manifest.json")

# v1 combat-only dims, for the fail-closed loader check (spec §14: "kein
# v1-Modell mit v2-Shape laden und kein Padding/Truncating").
BATTLE_OBS_DIM_V1 = 116
BATTLE_N_ACTIONS_V1 = 11


class BattleSchemaError(RuntimeError):
    """A model file's observation / action space does not match the schema the
    trainer is running. Raised INSTEAD of an opaque SB3 traceback (spec §14)."""

# PPO hyper-params — deliberately independent of train.PPO_N_STEPS.
BATTLE_PPO_N_STEPS = 256
BATTLE_PPO_BATCH_SIZE = 256
BATTLE_PPO_N_EPOCHS = 4
BATTLE_LEARNING_RATE = 3e-4
BATTLE_GAMMA = 0.99

FIXED_EVAL_SEEDS = tuple(range(1000, 1000 + 40))   # reproducible eval suite
# Check the young learner for a promotable version early and often
# (1k / 2k / 5k / 10k battle-steps), then every 10k. A wild Route-1 fight is
# 1-3 turns, so a good policy shows up fast and we want to catch it. The
# training chunk is split so learn() stops exactly on each boundary.
LIVE_PROMOTION_EARLY_CHECKS = (1_000, 2_000, 5_000, 10_000)
LIVE_PROMOTION_EVERY_STEPS = 10_000
MIN_TRAINING_WINS_BEFORE_FIRST_LIVE_EVAL = 10


def next_promotion_step(current_steps, *, early=LIVE_PROMOTION_EARLY_CHECKS,
                        interval=LIVE_PROMOTION_EVERY_STEPS):
    """First promotion checkpoint strictly after ``current_steps``.

    Fresh runs hit the early checks (1k, 5k, 10k); after that every
    ``interval`` steps. A resumed learner past the early window just follows
    the steady cadence."""
    current_steps = max(0, int(current_steps or 0))
    for boundary in early:
        if current_steps < boundary:
            return int(boundary)
    interval = max(1, int(interval))
    return (current_steps // interval + 1) * interval


def live_eval_ready(counters, *, has_champion):
    """Do not freeze all live workers to evaluate an obviously unready PPO.

    Once a real champion exists, every scheduled regression evaluation remains
    mandatory.  Before the first champion, the learner must first demonstrate
    a small number of real training wins; otherwise 200 emulator episodes only
    pause learning while being unable to clear the 80% first-champion gate.
    """
    return bool(has_champion or int(getattr(counters, "wins", 0) or 0)
                >= MIN_TRAINING_WINS_BEFORE_FIRST_LIVE_EVAL)


def read_battle_model_spaces(path):
    """``(obs_vec_width | None, n_actions | None)`` from an SB3 zip WITHOUT
    building the policy. Decodes the pickled ``observation_space`` /
    ``action_space`` the same way SB3 does (they are base64 pickle under
    ``:serialized:``), so it works on real MaskablePPO checkpoints, not only
    literal-JSON fixtures. Returns ``(None, None)`` on an unreadable zip."""
    import base64
    import json
    import pickle
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data").decode("utf-8", "ignore"))
    except Exception:
        return None, None

    def _space(key):
        entry = data.get(key)
        if isinstance(entry, dict) and entry.get(":serialized:"):
            try:
                return pickle.loads(base64.b64decode(entry[":serialized:"]))
            except Exception:
                return None
        return entry

    obs = _space("observation_space")
    act = _space("action_space")
    vec = None
    try:
        box = obs["vec"] if hasattr(obs, "__getitem__") else obs.spaces["vec"]
        vec = int(box.shape[0])
    except Exception:
        # literal-JSON fixture: {"vec": {"shape": [116]}}
        try:
            vec = int(obs["vec"]["shape"][0])
        except Exception:
            vec = None
    n = None
    try:
        n = int(getattr(act, "n"))
    except Exception:
        try:
            n = int(act["n"])
        except Exception:
            n = None
    return vec, n


def battle_schema_preflight(path, *, expect_obs_dim, expect_n_actions, schema):
    """Fail-closed check BEFORE ``MaskablePPO.load`` (spec §14).

    Compares the checkpoint's stored ``vec`` Box width and ``Discrete`` action
    count against what this trainer expects. On a mismatch it raises
    :class:`BattleSchemaError` with a readable message — a v1 (116 / 11) file is
    never loaded into a v2 (140 / 12) trainer and vice versa, and nothing is
    padded or truncated. An unreadable zip is left for ``MaskablePPO.load`` to
    surface its own error.
    """
    got_obs, got_act = read_battle_model_spaces(path)
    if got_obs is not None and got_obs != int(expect_obs_dim):
        raise BattleSchemaError(
            f"{os.path.basename(path)} has battle obs width {got_obs}, but this "
            f"trainer runs schema {schema} (width {expect_obs_dim}). A v1 model "
            f"is never loaded with a v2 observation space and nothing is padded "
            f"or truncated — start v2 from scratch instead of resuming this file.")
    if got_act is not None and got_act != int(expect_n_actions):
        raise BattleSchemaError(
            f"{os.path.basename(path)} has {got_act} battle actions, but schema "
            f"{schema} has {expect_n_actions} (…, RUN, CATCH). Refusing to load "
            f"a mismatched action space.")


def write_battle_model_manifest(path=BATTLE_MODEL_MANIFEST, *, live_schema="v1",
                                v1=None, v2=None, rom_sha256=""):
    """The Catch-v2 model manifest (spec §14). Records both schema slots, their
    dims and versions, and which one is live. The loader reads this to pick the
    right adapter; a human reads it instead of guessing from filenames."""
    from battle_env import OBS_SCHEMA as _V2_OBS, OBS_DIM as _V2_DIM, ALL_MACROS
    doc = {
        "schema": "battle_model_manifest_v1",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rom_sha256": str(rom_sha256 or ""),
        "live_schema": live_schema,          # "v1" until v2 passes full eval
        "v1": {
            "obs_schema": "battle_obs_v1",
            "actions_schema": "battle_actions_v1",
            "obs_dim": BATTLE_OBS_DIM_V1,
            "n_actions": BATTLE_N_ACTIONS_V1,
            "files": {"learner": "battle_learner.zip",
                      "candidate": "battle_candidate.zip",
                      "champion": "battle_champion.zip",
                      "resume": "battle_resume.zip"},
            **(v1 or {}),
        },
        "v2": {
            "obs_schema": _V2_OBS,
            "actions_schema": "battle_actions_v2_catch",
            "reward_schema": "battle_reward_v2_catch",
            "obs_dim": int(_V2_DIM),
            "n_actions": len(ALL_MACROS),
            "files": {"learner": "battle_learner_v2.zip",
                      "candidate": "battle_candidate_v2.zip",
                      "champion": "battle_champion_v2.zip",
                      "resume": "battle_resume_v2.zip"},
            **(v2 or {}),
        },
    }
    _atomic_json(path, doc)
    return doc


def battle_migration_preview():
    """Spec §14: describe the v1 -> v2 migration WITHOUT touching anything.

    Returns a dict listing the untouched v1 champion, the new v2 files, what is
    archived (nothing — v1 stays), what is untouched, the live version and the
    expected obs / action dims for both schemas. Caller prints it; no file is
    written or moved.
    """
    from battle_env import OBS_SCHEMA as _V2_OBS, OBS_DIM as _V2_DIM, ALL_MACROS
    p = {
        "learner": BATTLE_LEARNER, "candidate": BATTLE_CANDIDATE,
        "champion": BATTLE_CHAMPION, "resume": BATTLE_RESUME,
    }
    v2p = {
        "learner": BATTLE_LEARNER_V2, "candidate": BATTLE_CANDIDATE_V2,
        "champion": BATTLE_CHAMPION_V2, "resume": BATTLE_RESUME_V2,
    }
    return {
        "applies_anything": False,
        "current_v1_champion": {
            "path": BATTLE_CHAMPION,
            "exists": os.path.exists(BATTLE_CHAMPION),
            "stays_live": True,
            "role": "combat fallback during and after v2 training",
        },
        "new_v2_files": {k: {"path": v, "exists": os.path.exists(v)}
                         for k, v in v2p.items()},
        "archived_files": [],          # v1 is NOT archived — it stays live
        "untouched_files": [v for v in p.values()] + [
            BATTLE_STATS,
            os.path.join(BATTLE_DIR, "scenarios", "index.json"),
        ],
        "live_schema_now": "v1",
        "live_schema_after_v2_passes": "v2 (atomic switch, only between battles)",
        "expected_dims": {
            "v1": {"obs_dim": BATTLE_OBS_DIM_V1, "n_actions": BATTLE_N_ACTIONS_V1,
                   "obs_schema": "battle_obs_v1", "actions_schema": "battle_actions_v1"},
            "v2": {"obs_dim": int(_V2_DIM), "n_actions": len(ALL_MACROS),
                   "obs_schema": _V2_OBS, "actions_schema": "battle_actions_v2_catch",
                   "reward_schema": "battle_reward_v2_catch"},
        },
        "manifest_path": BATTLE_MODEL_MANIFEST,
        "notes": [
            "v1 champion file is never overwritten by v2 training",
            "a v1 model is never loaded with the v2 obs/action space (no pad/truncate)",
            "model switch happens only between battles, never mid-battle/animation",
            "v2 promotion is atomic and only after the full combat + catch eval",
        ],
    }


def _atomic_json(path, data):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


class BattleCounters:
    """Battle-only. No navigation key exists here at all.

    ``timeouts`` counts ONLY a real clock timeout (episode hit ``max_turns``).
    A battle that ended but could not be classified is ``terminal_unknown`` -
    a diagnosed driver fault, never a silent timeout.
    """

    _KEYS = ("env_steps", "ppo_updates", "rollout_samples", "episodes",
             "wins", "wipes", "flees", "timeouts", "kos",
             "invalid_actions", "aborted_macros", "menu_stalls",
             "unreadable_states", "terminal_unknown", "no_damage_turns",
             # Catch-v2 telemetry (spec §16)
             "combat_episodes", "catch_episodes", "catch_requested_episodes",
             "catch_orders", "catch_attempts", "catch_successes",
             "catch_failures", "catch_target_kos", "premature_catches",
             "unrequested_catches", "trainer_catch_attempts", "balls_used",
             "reward_sum_mismatch_steps")

    _REWARD_COMPONENTS = ("damage", "enemy_ko", "battle_win", "own_faint",
                          "wipe", "fled", "flee_illegal", "invalid_action",
                          "wasted_turn", "switch_loop", "turn_cost", "other",
                          # Catch-v2 reward components (spec §9)
                          "catch_success", "failed_catch", "catch_target_ko",
                          "ball_cost", "premature_catch_attempt",
                          "unrequested_catch", "illegal_trainer_catch")

    def __init__(self):
        import collections
        for k in self._KEYS:
            setattr(self, k, 0)
        self.reward_sum = 0.0
        self.reward_components = {k: 0.0 for k in self._REWARD_COMPONENTS}
        # last 200 finished battles: {"outcome", "reward", "turns"}
        self.recent = collections.deque(maxlen=200)
        self.action_counts = collections.Counter()   # macro -> count (lifetime)
        # step_reward != sum(components) should never happen; count it if it does
        self.reward_sum_mismatch = 0
        # per-env in-flight accumulators (survive chunk boundaries)
        self._ep_rew = {}
        self._ep_turns = {}

    def to_dict(self):
        return {k: int(getattr(self, k)) for k in self._KEYS}

    @staticmethod
    def _median(vals):
        vals = sorted(vals)
        n = len(vals)
        if not n:
            return 0.0
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    def rolling_dict(self):
        r = list(self.recent)
        last100 = r[-100:]
        wins = [x for x in r if x["outcome"] == "win"]
        rews = [x["reward"] for x in last100]

        def frac(pred, rows):
            return (sum(1 for x in rows if pred(x)) / len(rows)) if rows else 0.0

        outc = collections.Counter(x["outcome"] for x in r)
        act_total = sum(self.action_counts.values()) or 1
        # Catch-v2 rolling telemetry (spec §16)
        catch_req = [x for x in r if x.get("catch_requested")]
        catch_req_ok = [x for x in catch_req if x.get("catch_success")]
        obj_modes = collections.Counter(x.get("objective_mode", "combat") for x in r)
        catch = {
            "objective_distribution": {k: round(v / max(1, len(r)), 3)
                                       for k, v in obj_modes.items()},
            "catch_orders": int(self.catch_orders),
            "catch_attempts": int(self.catch_attempts),
            "catch_successes": int(self.catch_successes),
            "catch_failures": int(self.catch_failures),
            "catch_target_kos": int(self.catch_target_kos),
            "premature_catches": int(self.premature_catches),
            "unrequested_catches": int(self.unrequested_catches),
            "trainer_catch_attempts": int(self.trainer_catch_attempts),
            "balls_used": int(self.balls_used),
            "balls_per_successful_catch": round(
                self.balls_used / self.catch_successes, 2) if self.catch_successes else 0.0,
            "catch_success_rate_when_requested": round(
                len(catch_req_ok) / len(catch_req), 3) if catch_req else 0.0,
            "reward_sum_mismatch_steps": int(self.reward_sum_mismatch_steps),
        }
        return {
            "window": len(r),
            "avg_reward_last_100": (sum(rews) / len(rews)) if rews else 0.0,
            "median_reward_last_100": round(self._median(rews), 3),
            "best_reward": max((x["reward"] for x in r), default=0.0),
            "worst_reward": min((x["reward"] for x in r), default=0.0),
            "win_rate_200": frac(lambda x: x["outcome"] == "win", r),
            "wipe_rate_200": frac(lambda x: x["outcome"] == "wipe", r),
            "flee_rate_200": frac(lambda x: x["outcome"] == "fled", r),
            "avg_turns_last_200": (sum(x["turns"] for x in r) / len(r)) if r else 0.0,
            "avg_turns_on_win": (sum(x["turns"] for x in wins) / len(wins)
                                 if wins else 0.0),
            "outcome_distribution": {k: round(v / max(1, len(r)), 3)
                                     for k, v in outc.items()},
            "action_distribution": {k: round(v / act_total, 3)
                                    for k, v in self.action_counts.most_common()},
            "reward_sum": round(self.reward_sum, 2),
            "reward_sum_mismatch": int(self.reward_sum_mismatch),
            "reward_components": {k: round(v, 2)
                                  for k, v in self.reward_components.items()},
            "catch": catch,
        }


class BattleCounterCallback:
    """SB3 callback that records only BattleEnv terminal/event information.

    ``on_progress()`` (if given) is invoked every ~``progress_every`` env steps
    so the live status file moves DURING a chunk instead of only after the
    whole ``learn()`` call returns.
    """
    def __init__(self, counters, *, on_progress=None, progress_every=400,
                 stop_at=None):
        from stable_baselines3.common.callbacks import BaseCallback

        from battle_env import decompose_battle_reward

        class _Callback(BaseCallback):
            def _on_step(cb_self):
                if stop_at is not None and int(
                        getattr(cb_self.model, "num_timesteps", 0)) >= int(stop_at):
                    return False   # stop learn() exactly on the promotion boundary
                infos = cb_self.locals.get("infos")
                dones = cb_self.locals.get("dones")
                rewards = cb_self.locals.get("rewards")
                infos = [] if infos is None else infos
                dones = [] if dones is None else dones
                rewards = [] if rewards is None else list(rewards)
                ep_rew, ep_turns = counters._ep_rew, counters._ep_turns
                counters.rollout_samples += len(infos)
                for i, info in enumerate(infos):
                    rew = float(rewards[i]) if i < len(rewards) else 0.0
                    counters.reward_sum += rew
                    _comps = decompose_battle_reward(info, rew)
                    for name, amt in _comps:
                        if name in counters.reward_components:
                            counters.reward_components[name] += amt
                    # invariant: step_reward == sum of its components
                    if abs(rew - sum(a for _, a in _comps)) > 1e-3:
                        counters.reward_sum_mismatch += 1
                        counters.reward_sum_mismatch_steps += 1
                    _mac = info.get("macro")
                    if _mac:
                        counters.action_counts[_mac] += 1
                    ep_rew[i] = ep_rew.get(i, 0.0) + rew
                    ep_turns[i] = ep_turns.get(i, 0) + 1
                    # --- Catch-v2 per-step telemetry (spec §16) ---
                    _obj = info.get("objective") or {}
                    if info.get("macro") == "CATCH":
                        counters.catch_orders += 1
                    if info.get("catch_attempted"):
                        counters.catch_attempts += 1
                    if info.get("catch_success"):
                        counters.catch_successes += 1
                    if info.get("catch_outcome") == "broke_free":
                        counters.catch_failures += 1
                    if info.get("premature_catch_attempt"):
                        counters.premature_catches += 1
                    if info.get("unrequested_catch"):
                        counters.unrequested_catches += 1
                    if info.get("illegal_trainer_catch") or info.get("catch_reject") == "trainer_catch_blocked":
                        counters.trainer_catch_attempts += 1
                    if info.get("balls_used"):
                        counters.balls_used += int(info.get("balls_used") or 0)
                    if (_obj.get("catch_requested") and info.get("enemy_ko")
                            and not info.get("catch_success")):
                        counters.catch_target_kos += 1
                    if info.get("enemy_ko"):
                        counters.kos += 1
                    if info.get("invalid"):
                        counters.invalid_actions += 1
                    if info.get("aborted"):
                        counters.aborted_macros += 1
                    if info.get("menu_stall"):
                        counters.menu_stalls += 1
                    if info.get("unreadable"):
                        counters.unreadable_states += 1
                    if info.get("no_damage"):
                        counters.no_damage_turns += 1
                    if i < len(dones) and bool(dones[i]):
                        counters.episodes += 1
                        outcome = info.get("outcome")
                        _is_catch_obj = bool(_obj.get("objective_mode") == "catch")
                        if _is_catch_obj:
                            counters.catch_episodes += 1
                        else:
                            counters.combat_episodes += 1
                        if _obj.get("catch_requested"):
                            counters.catch_requested_episodes += 1
                        if outcome == "win":
                            counters.wins += 1
                        elif outcome == "wipe":
                            counters.wipes += 1
                        elif outcome == "fled":
                            counters.flees += 1
                        elif outcome == "caught":
                            pass                   # own terminal outcome
                        elif outcome == "timeout" or info.get("TimeLimit.truncated"):
                            counters.timeouts += 1
                        else:                      # terminal_unknown / menu_stall
                            counters.terminal_unknown += 1
                        counters.recent.append({
                            "outcome": outcome or "unknown",
                            "reward": round(ep_rew.pop(i, 0.0), 3),
                            "turns": ep_turns.pop(i, 0),
                            "objective_mode": _obj.get("objective_mode", "combat"),
                            "catch_requested": bool(_obj.get("catch_requested")),
                            "catch_success": bool(info.get("catch_success"))
                            or outcome == "caught"})
                if on_progress is not None:
                    n = int(getattr(cb_self.model, "num_timesteps", 0) or 0)
                    if n - getattr(cb_self, "_last_progress", 0) >= progress_every:
                        cb_self._last_progress = n
                        counters.env_steps = n
                        counters.ppo_updates = int(
                            getattr(cb_self.model, "_n_updates", 0) or 0)
                        try:
                            on_progress()
                        except Exception:
                            pass
                return True

        self.callback = _Callback()


def make_battle_vec_env(n_envs=BATTLE_WORKERS, *, scenario_sampler=None,
                        simulated=True, seed=0, schema="v2"):
    """Build the vectorised battle env. ``simulated=True`` uses the
    deterministic driver; the emulator driver is gated off. ``schema`` picks
    the obs/action space: "v1" (combat only, 116 / 11 — never emits CATCH) or
    "v2" (adds CATCH, 140 / 12). A v1 model MUST run on a v1 env."""
    import gymnasium as gym  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from battle_env import BattleEnv, SimulatedBattleDriver
    from twoby2.live_integration import (battle_driver_for,
                                         assert_no_simulated_driver_in_live_path)

    if not simulated and not feature_enabled("battle_env"):
        raise RuntimeError("live emulator battle env is gated off "
                           "(FEATURES['battle_env'] is False)")

    def _factory(rank):
        def _make():
            # gate ON -> real EmulatorBattleDriver on an isolated live env;
            # gate OFF -> SimulatedBattleDriver (offline bring-up / tests only).
            if feature_enabled("battle_env") and not simulated:
                from battle_env import make_live_battle_env
                mirror = (os.path.join(BATTLE_DIR, "battle_worker_live.jpg")
                          if rank == n_envs - 1 else None)
                ranked_sampler = scenario_sampler
                if hasattr(scenario_sampler, "for_worker"):
                    ranked_sampler = lambda rng: scenario_sampler.for_worker(
                        rng, rank=rank, n_workers=n_envs)
                env = make_live_battle_env(scenario_sampler=ranked_sampler,
                                           seed=seed + rank,
                                           mirror_path=mirror, schema=schema)
                assert_no_simulated_driver_in_live_path(getattr(env, "driver", None))
            else:
                env = BattleEnv(SimulatedBattleDriver(),
                                scenario_sampler=scenario_sampler, schema=schema)
            env.reset(seed=seed + rank)
            return env
        return _make

    factories = [_factory(i) for i in range(n_envs)]
    return (SubprocVecEnv(factories, start_method="spawn")
            if feature_enabled("battle_env") and not simulated
            else DummyVecEnv(factories))


def _battle_ppo_cls():
    # MaskablePPO masks its own categorical sampler from BattleEnv.action_masks()
    # -> the policy can never pick a locked switch or a 0-PP move. This replaces
    # the old "sample anything, penalise + fall back" scheme that fed the learner
    # contradictory data.
    from sb3_contrib import MaskablePPO
    return MaskablePPO


def build_battle_ppo(vec_env, *, device="cpu", seed=0):
    return _battle_ppo_cls()(
        "MultiInputPolicy", vec_env,
        n_steps=BATTLE_PPO_N_STEPS, batch_size=BATTLE_PPO_BATCH_SIZE,
        n_epochs=BATTLE_PPO_N_EPOCHS, learning_rate=BATTLE_LEARNING_RATE,
        gamma=BATTLE_GAMMA, seed=seed, device=device, verbose=0,
    )


def atomic_save_model(model, path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp.zip")
    model.save(tmp)
    os.replace(tmp, path)


def evaluate_battle_model(model, scenarios, *, seeds=FIXED_EVAL_SEEDS,
                          deterministic=True, schema="v2"):
    """Run the fixed reproducible eval suite. Returns per-scenario / per-kind /
    per-area metrics for :func:`evaluate_battle_promotion`. ``schema`` MUST
    match the model — a v1 model is scored on a v1 env (116 / 11)."""
    import numpy as np
    from battle_env import BattleEnv, SimulatedBattleDriver

    records = []
    for si, scenario in enumerate(scenarios):
        for seed in seeds:
            env = BattleEnv(SimulatedBattleDriver(), schema=schema)
            obs, _ = env.reset(seed=int(seed), options={"scenario": scenario})
            records.append(_run_one_eval_battle(
                model, env, obs, scenario, si,
                deterministic=deterministic, np=np))
    return _aggregate_eval(records)


def _run_one_eval_battle(model, env, obs, scenario, scenario_ix, *,
                         deterministic, np):
    """One eval battle -> a per-battle record for :func:`_aggregate_eval`.
    Collects the true outcome plus every diagnosed fault the safety gate needs."""
    done = trunc = False
    invalid = switch_loops = turns = 0
    aborted = menu_stall = unreadable = illegal_flee = False
    catch_attempts = balls_used = premature = unrequested = trainer_catch = 0
    caught = target_ko = False
    recent = []
    outcome = None
    info = {}
    while not (done or trunc):
        mask = np.asarray(obs["action_mask"])
        a, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        a = int(a)
        if not mask[a]:
            a = int(np.argmax(mask))
            invalid += 1
        obs, _r, done, trunc, info = env.step(a)
        turns += 1
        if str(info.get("macro", "")).startswith("SWITCH"):
            recent.append(info["macro"])
            if len(recent) >= 3 and recent[-1] == recent[-3]:
                switch_loops += 1
        invalid += 1 if info.get("invalid") else 0
        aborted |= bool(info.get("aborted"))
        menu_stall |= bool(info.get("menu_stall"))
        unreadable |= bool(info.get("unreadable"))
        illegal_flee |= bool(info.get("illegal_flee"))
        # --- Catch-v2 eval facts ---
        if info.get("catch_attempted"):
            catch_attempts += 1
        if info.get("balls_used"):
            balls_used += int(info.get("balls_used") or 0)
        if info.get("premature_catch_attempt"):
            premature += 1
        if info.get("unrequested_catch"):
            unrequested += 1
        if info.get("illegal_trainer_catch") or info.get("catch_reject") == "trainer_catch_blocked":
            trainer_catch += 1
        if info.get("catch_success"):
            caught = True
        _o = info.get("objective") or {}
        if _o.get("catch_requested") and info.get("enemy_ko") and not info.get("catch_success"):
            target_ko = True
        outcome = info.get("outcome") or outcome
    st = env._state or {}
    own_hp = sum(int(m.get("cur_hp", 0)) for m in st.get("our_party", []))
    own_max = sum(int(m.get("max_hp", 1)) for m in st.get("our_party", []))
    is_trainer = bool(scenario.get("is_trainer"))
    obj = getattr(env, "_objective", None) or {}
    catch_requested = bool(obj.get("catch_requested"))
    usable_balls = int(obj.get("usable_ball_count", 0) or 0)
    return {
        "scenario_ix": scenario_ix,
        "area": scenario.get("area", "unknown"),
        "battle_kind": "trainer" if is_trainer else "wild",
        "won": outcome == "win",
        "wiped": outcome == "wipe",
        "fled": outcome == "fled",
        "terminal_unknown": outcome in ("terminal_unknown", "menu_stall",
                                        "unreadable", "reset_unreadable"),
        "timeout": outcome == "timeout" or bool(info.get("TimeLimit.truncated")),
        "aborted": aborted,
        "menu_stall": menu_stall or outcome == "menu_stall",
        "unreadable": unreadable or outcome in ("unreadable", "reset_unreadable"),
        "illegal_flee": illegal_flee or (is_trainer and outcome == "fled"),
        "residual_hp_frac": own_hp / max(1, own_max),
        "turns": turns,
        "invalid_actions": invalid,
        "switch_loops": switch_loops,
        # Catch-v2
        "objective_mode": obj.get("objective_mode", "combat"),
        "catch_requested": catch_requested,
        "catch_possible": catch_requested and usable_balls > 0 and not is_trainer,
        "caught": caught,
        "catch_attempts": catch_attempts,
        "balls_used": balls_used,
        "target_ko_in_catch": target_ko,
        "premature_catches": premature,
        "unrequested_catches": unrequested,
        "trainer_catch_attempts": trainer_catch,
        "catch_group": scenario.get("catch_group", scenario.get("area", "unknown")),
    }


def evaluate_live_battle_model(model, scenario_sampler, *, episodes=200,
                               seed=7000, deterministic=True, schema="v2"):
    """Fixed real-emulator evaluation used for live champion promotion.
    ``schema`` MUST match the model (v1 = 116 / 11)."""
    import numpy as np
    from battle_env import make_live_battle_env
    rng = np.random.default_rng(seed)
    env = make_live_battle_env(scenario_sampler=None, seed=seed, schema=schema)
    records = []
    try:
        for ep in range(int(episodes)):
            scenario = scenario_sampler(rng)
            obs, _ = env.reset(seed=seed + ep, options={"scenario": scenario})
            records.append(_run_one_eval_battle(
                model, env, obs, scenario,
                scenario.get("id", scenario.get("area", "unknown")),
                deterministic=deterministic, np=np))
    finally:
        env.close()
    return _aggregate_eval(records)


def _rule_fallback_metrics():
    """Baseline when no PPO champion exists yet: a beatable zero-win record so
    the first real candidate can promote through the same gate."""
    return {"episodes": EVAL_SUITE_MIN_EPISODES, "win_rate": 0.0,
            "trainer_win_rate": 0.0, "wild_win_rate": 0.0, "wipe_rate": 1.0,
            "flee_rate": 0.0, "avg_residual_hp": 0.0,
            "avg_residual_hp_on_win": 0.0, "avg_turns": 0.0, "avg_turns_on_win": 0.0,
            "invalid_actions": 0, "aborted_macros": 0, "menu_stalls": 0,
            "unreadable_states": 0, "terminal_unknown": 0, "timeouts": 0,
            "trainer_flees": 0, "switch_loops": 0, "needless_flees": 0,
            "per_area_win_rate": {}, "per_kind_win_rate": {},
            "n_trainer": 0, "n_wild": 0, "scenario_coverage": 0}


def _aggregate_eval(records):
    n = len(records) or 1
    wins = [r for r in records if r["won"]]

    def rate(pred, subset=None):
        rows = [r for r in records if (subset is None or subset(r))]
        return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else 0.0

    def avg(vals):
        vals = list(vals)
        return (sum(vals) / len(vals)) if vals else 0.0

    areas = sorted({r["area"] for r in records})
    kinds = sorted({r["battle_kind"] for r in records})
    return {
        "episodes": len(records),
        "win_rate": rate(lambda r: r["won"]),
        "trainer_win_rate": rate(lambda r: r["won"], lambda r: r["battle_kind"] == "trainer"),
        "wild_win_rate": rate(lambda r: r["won"], lambda r: r["battle_kind"] == "wild"),
        "wipe_rate": rate(lambda r: r["wiped"]),
        "flee_rate": rate(lambda r: r["fled"]),
        # residual HP / turns are only meaningful for battles that were WON -
        # a flee keeps HP but is a loss, and must never look "better"
        "avg_residual_hp": avg(r["residual_hp_frac"] for r in records),
        "avg_residual_hp_on_win": avg(r["residual_hp_frac"] for r in wins),
        "avg_turns": avg(r["turns"] for r in records),
        "avg_turns_on_win": avg(r["turns"] for r in wins),
        # hard safety metrics (the promotion gate requires every one == 0)
        "invalid_actions": sum(r["invalid_actions"] for r in records),
        "aborted_macros": sum(1 for r in records if r["aborted"]),
        "menu_stalls": sum(1 for r in records if r["menu_stall"]),
        "unreadable_states": sum(1 for r in records if r["unreadable"]),
        "terminal_unknown": sum(1 for r in records if r["terminal_unknown"]),
        "timeouts": sum(1 for r in records if r["timeout"]),
        "trainer_flees": sum(1 for r in records if r["illegal_flee"]),
        "switch_loops": sum(r["switch_loops"] for r in records),
        "needless_flees": sum(1 for r in records if r["fled"]),
        "per_area_win_rate": {a: rate(lambda r: r["won"], lambda r, a=a: r["area"] == a)
                              for a in areas},
        "per_kind_win_rate": {k: rate(lambda r: r["won"], lambda r, k=k: r["battle_kind"] == k)
                              for k in kinds},
        "n_trainer": sum(1 for r in records if r["battle_kind"] == "trainer"),
        "n_wild": sum(1 for r in records if r["battle_kind"] == "wild"),
        "scenario_coverage": len({r["scenario_ix"] for r in records}),
        # Catch-v2 hard-zero gates also live on the combat aggregate (spec §13)
        "trainer_catch_attempts": sum(r.get("trainer_catch_attempts", 0) for r in records),
        "unrequested_catch_rate": (sum(r.get("unrequested_catches", 0) for r in records)
                                   / (len(records) or 1)),
        "reward_sum_mismatch": 0,
    }


def _aggregate_catch_eval(records):
    """Aggregate the FIXED catch suite into the metrics ``battle_catch_gate``
    reads (spec §13). Mechanically-impossible scenarios (``catch_possible`` is
    False: no usable ball / trainer) are EXCLUDED from the success rate and
    scored separately as correct non-execution."""
    req = [r for r in records if r.get("catch_requested")]
    possible = [r for r in req if r.get("catch_possible")]
    impossible = [r for r in req if not r.get("catch_possible")]
    caught = [r for r in possible if r.get("caught")]

    def _rate(sub, pred):
        return (sum(1 for r in sub if pred(r)) / len(sub)) if sub else 0.0

    groups = sorted({r.get("catch_group", "unknown") for r in possible})
    per_group = {}
    for g in groups:
        gr = [r for r in possible if r.get("catch_group") == g]
        per_group[g] = _rate(gr, lambda r: r.get("caught"))
    balls_on_success = sum(r.get("balls_used", 0) for r in caught)
    return {
        "catch_eval_episodes": len(records),
        "catch_requested_episodes": len(req),
        "catch_possible_episodes": len(possible),
        "catch_impossible_episodes": len(impossible),
        # impossible scenarios pass when the model correctly did NOT throw a ball
        "impossible_handled_correctly": _rate(
            impossible, lambda r: r.get("catch_attempts", 0) == 0),
        "catch_success_rate_when_requested": _rate(possible, lambda r: r.get("caught")),
        "per_group_catch_success_rate": per_group,
        "target_ko_rate": _rate(possible, lambda r: r.get("target_ko_in_catch")),
        "balls_per_successful_catch": (balls_on_success / len(caught)) if caught else 0.0,
        "premature_catch_rate": _rate(possible, lambda r: r.get("premature_catches", 0) > 0),
        "unrequested_catch_rate": (sum(r.get("unrequested_catches", 0) for r in records)
                                   / (len(records) or 1)),
        "trainer_catch_attempts": sum(r.get("trainer_catch_attempts", 0) for r in records),
    }


def evaluate_battle_catch_suite(model, scenarios=None, *, seeds=(0, 1, 2),
                                deterministic=True, schema="v2"):
    """Run the model over the fixed catch suite (spec §13). Returns the metrics
    :func:`twoby2.battle_promotion.battle_catch_gate` reads. Only meaningful for
    a v2 model on a v2 env — a v1 model has no CATCH action."""
    import numpy as np
    from battle_env import BattleEnv, SimulatedBattleDriver
    scenarios = list(scenarios if scenarios is not None else fixed_catch_scenarios())
    records = []
    for si, scenario in enumerate(scenarios):
        for seed in seeds:
            env = BattleEnv(SimulatedBattleDriver(), schema=schema)
            obs, _ = env.reset(seed=int(seed), options={"scenario": scenario})
            records.append(_run_one_eval_battle(
                model, env, obs, scenario, si, deterministic=deterministic, np=np))
    return _aggregate_catch_eval(records)


class BattleTrainer:
    """Thin orchestrator. Real PPO, real env, real checkpoints — the loop is
    small and deterministic so tests can drive it end to end."""

    def __init__(self, *, n_workers=BATTLE_WORKERS, device="cpu", seed=0,
                 ckpt_dir=CKPT_DIR, stats_path=BATTLE_STATS, schema="v1"):
        # schema: "v1" (combat only, 116/11 — the live default, untouched) or
        # "v2" (adds CATCH, 140/12). v2 uses a fully separate file set; a v1
        # file is never loaded with a v2 space and vice versa (spec §14).
        self.schema = str(schema)
        self.n_workers = int(n_workers)
        self.device = device
        self.seed = seed
        self.ckpt_dir = ckpt_dir
        self.stats_path = stats_path
        self.counters = BattleCounters()
        self.champion_version = 0
        self.learner_version = 0
        self.hard_regressions = 0
        self.next_promotion_step = 0
        # last champion check (persisted on every stats write, like next_promotion_step)
        self.last_promotion = None
        self.last_live_eval = None
        self._vec = None
        self._model = None

    def _paths(self):
        sfx = "_v2" if self.schema == "v2" else ""
        return {
            "learner": os.path.join(self.ckpt_dir, f"battle_learner{sfx}.zip"),
            "candidate": os.path.join(self.ckpt_dir, f"battle_candidate{sfx}.zip"),
            "champion": os.path.join(self.ckpt_dir, f"battle_champion{sfx}.zip"),
            "resume": os.path.join(self.ckpt_dir, f"battle_resume{sfx}.zip"),
        }

    def _expected_dims(self):
        from battle_env import battle_schema_spec
        macros, obs_dim, *_ = battle_schema_spec(self.schema)
        return int(obs_dim), len(macros)

    def _load_model(self, path, *, with_env):
        """Load a battle model with a fail-closed schema preflight (spec §14):
        a readable :class:`BattleSchemaError`, never a raw SB3 traceback."""
        obs_dim, n_act = self._expected_dims()
        battle_schema_preflight(path, expect_obs_dim=obs_dim,
                                expect_n_actions=n_act, schema=self.schema)
        kw = {"device": self.device}
        if with_env:
            kw["env"] = self._vec
        return _battle_ppo_cls().load(path, **kw)

    def setup(self, scenario_sampler=None):
        self._vec = make_battle_vec_env(self.n_workers,
                                        scenario_sampler=scenario_sampler,
                                        simulated=not feature_enabled("battle_env"),
                                        seed=self.seed, schema=self.schema)
        p = self._paths()
        if os.path.exists(p["resume"]):
            self._model = self._load_model(p["resume"], with_env=True)
            self._restore_stats()
        else:
            self._model = build_battle_ppo(self._vec, device=self.device, seed=self.seed)
        self.counters.env_steps = int(self._model.num_timesteps)
        self.counters.ppo_updates = int(getattr(self._model, "_n_updates", 0))
        try:
            write_battle_model_manifest(
                os.path.join(os.path.dirname(os.path.abspath(self.stats_path)),
                             "battle_model_manifest.json"),
                live_schema="v1",   # v1 stays live until v2 passes full eval
                v2={"learner_version": self.learner_version,
                    "champion_version": self.champion_version,
                    "training_schema": self.schema})
        except Exception:
            pass
        return self._model

    def _restore_stats(self):
        """Restore cumulative outcome counters which are not stored by SB3."""
        try:
            with open(self.stats_path) as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            return False
        saved = data.get("counters") or {}
        for key in BattleCounters._KEYS:
            if key in ("env_steps", "ppo_updates"):
                continue   # owned by SB3 (model.num_timesteps / _n_updates)
            setattr(self.counters, key, int(saved.get(key, 0) or 0))
        self.champion_version = int(data.get("champion_version", 0) or 0)
        self.learner_version = int(data.get("learner_version", 0) or 0)
        self.hard_regressions = int(data.get("hard_regressions", 0) or 0)
        self.next_promotion_step = int(data.get("next_promotion_step", 0) or 0)
        self.last_promotion = data.get("last_promotion")
        self.last_live_eval = data.get("last_live_eval")
        return True

    def train_chunk(self, timesteps, *, stop_at=None):
        assert self._model is not None, "call setup() first"
        counter_cb = BattleCounterCallback(
            self.counters,
            on_progress=lambda: self._write_stats(extra={"phase": "training"}),
            stop_at=stop_at,
        ).callback
        self._model.learn(total_timesteps=int(timesteps), reset_num_timesteps=False,
                          callback=counter_cb)
        self.counters.env_steps = int(self._model.num_timesteps)
        self.counters.ppo_updates = int(getattr(self._model, "_n_updates", 0))
        self.learner_version += 1
        p = self._paths()
        atomic_save_model(self._model, p["learner"])
        atomic_save_model(self._model, p["resume"])
        self._write_stats()

    def try_promote(self, eval_scenarios, regression_core_scenarios=(), *,
                    catch_scenarios=None):
        """Evaluate the learner as a candidate; promote only on a real,
        regression-free win. For schema v2 a passing FIXED catch suite
        (``catch_scenarios``, default :func:`fixed_catch_scenarios`) is ALSO
        required (spec §13) — a high combat win-rate can never promote a broken
        catch-v2. Returns the decision dict."""
        p = self._paths()
        atomic_save_model(self._model, p["candidate"])
        cand_metrics = evaluate_battle_model(self._model, list(eval_scenarios),
                                             schema=self.schema)
        champ_metrics = self._champion_metrics(eval_scenarios)
        core_cand = (evaluate_battle_model(self._model, list(regression_core_scenarios),
                                           schema=self.schema)
                     if regression_core_scenarios else {})
        core_champ = (self._champion_metrics(regression_core_scenarios)
                      if regression_core_scenarios else {})

        catch_eval = None
        if self.schema == "v2":
            catch_eval = evaluate_battle_catch_suite(
                self._model, catch_scenarios if catch_scenarios is not None
                else fixed_catch_scenarios(), schema=self.schema)

        decision = evaluate_battle_promotion(
            candidate=cand_metrics, champion=champ_metrics,
            candidate_core=core_cand, champion_core=core_champ,
            catch_eval=catch_eval)

        if decision["promote"]:
            atomic_save_model(self._model, p["champion"])
            self.champion_version += 1
            self.hard_regressions = 0
        elif decision["hard_regression"]:
            self.hard_regressions += 1
            self._rollback_learner_to_champion()
        self.last_promotion = dict(decision)
        if catch_eval is not None:
            self.last_promotion["catch_eval"] = catch_eval
        self._write_stats()
        return decision

    def _champion_metrics(self, scenarios):
        p = self._paths()
        if not os.path.exists(p["champion"]):
            return {**_rule_fallback_metrics(), "source": "rule_fallback"}
        champ = self._load_model(p["champion"], with_env=False)
        return evaluate_battle_model(champ, list(scenarios), schema=self.schema)

    def _champion_metrics_live(self, scenario_sampler, *, seed=7000):
        p = self._paths()
        if not os.path.exists(p["champion"]):
            return _rule_fallback_metrics()
        champion = self._load_model(p["champion"], with_env=False)
        return evaluate_live_battle_model(
            champion, scenario_sampler, episodes=EVAL_SUITE_MIN_EPISODES,
            seed=seed, schema=self.schema)

    def _rollback_learner_to_champion(self):
        p = self._paths()
        if os.path.exists(p["champion"]):
            self._model = self._load_model(p["champion"], with_env=True)
            atomic_save_model(self._model, p["learner"])
            atomic_save_model(self._model, p["resume"])

    def _write_stats(self, extra=None):
        data = {
            "schema": "battle_stats_v1",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "counters": self.counters.to_dict(),
            "champion_version": self.champion_version,
            "learner_version": self.learner_version,
            "hard_regressions": self.hard_regressions,
            # Persisted on every write so the status/web view keeps showing the
            # next check / last result between chunks (a bare _write_stats()
            # used to drop anything only passed via extra=).
            "next_promotion_step": int(self.next_promotion_step or 0),
            "last_promotion": self.last_promotion,
            "last_live_eval": self.last_live_eval,
            "rolling": self.counters.rolling_dict(),
            "n_workers": self.n_workers,
        }
        if extra:
            data.update(extra)
        _atomic_json(self.stats_path, data)
        return data


# --- Catch-v2 training / eval scenario distribution (spec §11 / §12) --------
CATCH_TRAINING_FRACTION = 0.25     # start at 75% combat / 25% catch


def fixed_catch_scenarios():
    """The fixed, reproducible catch scenario suite (spec §11). Candidate and
    champion are scored on EXACTLY these, with fixed seeds. Covers: a clean
    weakened-target catch, a catch after prior weakening, a broke-free +
    continuation, no-ball (mechanically impossible -> scored as correct
    non-execution), a full-party + PC catch, a trainer battle (CATCH must never
    be attempted), and a spread of catch rates / levels / HP fractions."""
    from battle_env import catch_scenario
    S = []
    # 1-4: clean catches, ascending difficulty (catch rate down)
    for i, cr in enumerate((255, 150, 90, 45)):
        S.append(catch_scenario(species_id=19 + i, catch_rate=cr, enemy_cur=14,
                                enemy_max=20, catch_seed=100 + i))
    # 5-7: catch after prior weakening (low HP, inside the window)
    for i, cr in enumerate((120, 70, 45)):
        S.append(catch_scenario(species_id=25 + i, catch_rate=cr, enemy_cur=5,
                                enemy_max=30, catch_seed=110 + i))
    # 8-9: status-assisted (sleep = 2.0 bonus)
    S.append(catch_scenario(species_id=41, catch_rate=90, enemy_cur=12,
                            enemy_max=24, status=0x03, catch_seed=120))
    S.append(catch_scenario(species_id=43, catch_rate=45, enemy_cur=8,
                            enemy_max=28, status=0x03, catch_seed=121))
    # 10-11: broke-free then continuation (very low catch rate)
    S.append(catch_scenario(species_id=16, catch_rate=3, enemy_cur=16,
                            enemy_max=22, catch_seed=130))
    S.append(catch_scenario(species_id=21, catch_rate=3, enemy_cur=18,
                            enemy_max=24, catch_seed=131))
    # 12-13: no usable ball (reserve == inventory) -> mechanically impossible
    S.append(catch_scenario(species_id=19, catch_rate=255, ball_inventory=1,
                            ball_reserve=1, catch_seed=140))
    S.append(catch_scenario(species_id=19, catch_rate=255, ball_inventory=0,
                            ball_reserve=1, catch_seed=141))
    # 14: full party, PC supported -> catch still allowed
    S.append(catch_scenario(species_id=27, catch_rate=200, party_n=6,
                            party_has_space=False, pc_capture_supported=True,
                            catch_seed=150))
    # 15: full party, NO PC -> catch must be blocked, fight instead
    S.append(catch_scenario(species_id=27, catch_rate=200, party_n=6,
                            party_has_space=False, pc_capture_supported=False,
                            catch_seed=151))
    # 16: trainer battle with a catch objective set -> CATCH never attempted
    S.append(catch_scenario(species_id=19, catch_rate=255, is_trainer=True,
                            catch_seed=160))
    # 17-18: duplicate species (already caught this run) -> combat, no +0.5
    S.append(catch_scenario(species_id=19, catch_rate=255,
                            is_new_species_this_run=False, catch_seed=170))
    S.append(catch_scenario(species_id=16, catch_rate=200,
                            is_new_species_this_run=False, catch_seed=171))
    return S


class MixedObjectiveScenarioSampler:
    """Wraps a combat scenario sampler and injects catch scenarios at a fixed
    fraction (spec §11: 75/25 at the start). One of the 10 layered protections
    against a catch-only policy (spec §12): the objective distribution is
    controlled here, not by the policy, and is counted + published."""

    def __init__(self, combat_sampler, *, catch_fraction=CATCH_TRAINING_FRACTION,
                 catch_scenarios=None):
        self.combat_sampler = combat_sampler
        self.catch_fraction = float(catch_fraction)
        self.catch_scenarios = list(catch_scenarios or fixed_catch_scenarios())
        self.n_combat = 0
        self.n_catch = 0

    def __call__(self, rng):
        take_catch = (self.catch_scenarios and self._rand(rng) < self.catch_fraction)
        if take_catch:
            self.n_catch += 1
            i = int(self._randint(rng, len(self.catch_scenarios)))
            return dict(self.catch_scenarios[i])
        self.n_combat += 1
        sc = dict(self.combat_sampler(rng))
        sc.setdefault("objective_mode", "combat")
        return sc

    def counts(self):
        tot = self.n_combat + self.n_catch or 1
        return {"combat": self.n_combat, "catch": self.n_catch,
                "catch_fraction_actual": round(self.n_catch / tot, 3),
                "catch_fraction_target": self.catch_fraction}

    @staticmethod
    def _rand(rng):
        return float(rng.random()) if hasattr(rng, "random") else float(rng.uniform(0, 1))

    @staticmethod
    def _randint(rng, n):
        if hasattr(rng, "integers"):
            return rng.integers(0, n)
        return rng.randint(0, n - 1)


class LiveScenarioSampler:
    """Pick from the captured, immutable battle-start scenario index.

    spec ZIEL B: the pick is BUCKET-BALANCED — the least-served diversity
    bucket (``area+species+enemy_lvl+player_lvl+party+objective_mode``) is
    chosen first, so one Rattata / one seed can never dominate the training
    distribution even if it has many near-identical rows.
    """
    def __init__(self, index_path=None, *, run_seed=None):
        self.index_path = index_path or os.path.join(
            BATTLE_DIR, "scenarios", "index.json")
        self._bucket_served = {}          # in-run coverage counters
        # spec ZIEL C.5: reproducible per run. A fixed run_seed replays the same
        # per-episode harvest sequence; consecutive episodes still differ.
        self.run_seed = int(run_seed if run_seed is not None
                            else os.environ.get("PKMAI_HARVEST_RUN_SEED", "0") or 0)
        self._episode_seq = {}           # scenario id -> episodes drawn so far

    def _bucket(self, row):
        return row.get("bucket") or f"{row.get('area','?')}|{row.get('id','?')}"

    def _pick_balanced(self, rows, rng):
        """Least-served bucket, then least-trained row within it, then jitter."""
        buckets = {}
        for r in rows:
            buckets.setdefault(self._bucket(r), []).append(r)
        target = min(buckets, key=lambda b: (self._bucket_served.get(b, 0), b))
        self._bucket_served[target] = self._bucket_served.get(target, 0) + 1
        choices = sorted(buckets[target],
                         key=lambda r: (int(r.get("trained_count", 0) or 0),
                                        r.get("id", "")))
        # keep a little randomness among equally-good rows
        least = int(choices[0].get("trained_count", 0) or 0)
        pool = [r for r in choices if int(r.get("trained_count", 0) or 0) == least]
        row = dict(pool[int(rng.integers(0, len(pool)))])
        row["pre_action_wait"] = int(rng.integers(0, 64))
        row["scenario_bucket"] = target
        if row.get("harvest"):
            from twoby2.wild_encounter_harvester import harvest_episode_seed
            sid = str(row.get("id", target))
            seq = self._episode_seq.get(sid, 0) + 1
            self._episode_seq[sid] = seq
            row["run_seed"] = self.run_seed
            row["encounter_sequence"] = seq
            base = int(row.get("rng_seed_id", 0) or 0) ^ int(row["pre_action_wait"])
            row["episode_seed"] = harvest_episode_seed(base, self.run_seed, seq)
        return row

    def __call__(self, rng):
        rows = self._rows()
        if not rows:
            raise RuntimeError("no captured live battle scenarios")
        return self._pick_balanced(rows, rng)

    def _rows(self):
        with open(self.index_path) as f:
            rows = (json.load(f) or {}).get("scenarios", [])
        return [r for r in rows if os.path.isfile(r.get("savestate_path", ""))]

    def for_worker(self, rng, *, rank, n_workers):
        """Three fixed groups follow the newest three discovered areas.

        With nine workers this is exactly 3+3+3. Worker 8 (the visible mirror)
        is assigned to the newest area. Before three areas exist, groups fold
        onto the available areas, so Route 1 initially receives all workers.
        Within the worker's area the pick is still bucket-balanced.
        """
        rows = self._rows()
        if not rows:
            raise RuntimeError("no captured live battle scenarios")
        newest = {}
        for row in rows:
            area = row.get("area", "unknown")
            newest[area] = max(float(row.get("captured_at", 0) or 0),
                               newest.get(area, 0.0))
        areas = [a for a, _ in sorted(newest.items(), key=lambda kv: kv[1])][-3:]
        area = areas[int(rank) % len(areas)]
        choices = [r for r in rows if r.get("area", "unknown") == area] or rows
        return self._pick_balanced(choices, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--chunk", type=int, default=2304,
                    help="battle decisions per learn() call")
    ap.add_argument("--schema", choices=("v1", "v2"), default="v1",
                    help="v1 = combat only (live default); v2 = adds CATCH, "
                         "trains into a fully separate file set and mixes "
                         f"{int(CATCH_TRAINING_FRACTION * 100)}%% catch scenarios")
    args = ap.parse_args()
    if not feature_enabled("battle_env"):
        raise RuntimeError("PKMAI_TWOBY2_LIVE=1 is required for real battle training")
    sampler = LiveScenarioSampler()
    if args.schema == "v2":
        sampler = MixedObjectiveScenarioSampler(sampler)
    trainer = BattleTrainer(n_workers=args.workers, schema=args.schema)
    trainer.setup(scenario_sampler=sampler)
    print(f"Battle-PPO LIVE [{args.schema}]: {args.workers} Fighter, "
          f"n_steps={BATTLE_PPO_N_STEPS}", flush=True)
    try:
        trainer.next_promotion_step = next_promotion_step(
            trainer._model.num_timesteps)
        trainer._write_stats(extra={"phase": "training"})
        print("Naechste Champion-Pruefung bei Battle-Step "
              f"{trainer.next_promotion_step}", flush=True)
        while True:
            # split the chunk so learn() stops EXACTLY on the next promotion
            # boundary (1k/2k/5k/10k...): the callback returns False the moment
            # num_timesteps reaches it, so the rollout can never carry past it.
            trainer.train_chunk(args.chunk, stop_at=trainer.next_promotion_step)
            print(json.dumps(trainer.counters.to_dict(), sort_keys=True), flush=True)
            if trainer.counters.env_steps >= trainer.next_promotion_step:
                has_champion = os.path.exists(trainer._paths()["champion"])
                if not live_eval_ready(trainer.counters,
                                       has_champion=has_champion):
                    print("Champion-Pruefung uebersprungen: erst "
                          f"{MIN_TRAINING_WINS_BEFORE_FIRST_LIVE_EVAL} echte "
                          "Trainingssiege sammeln; Fighter trainieren weiter.",
                          flush=True)
                    trainer.next_promotion_step = next_promotion_step(
                        trainer.counters.env_steps)
                    trainer._write_stats(extra={
                        "phase": "training",
                        "evaluation_deferred": "insufficient_training_wins",
                    })
                    continue
                trainer._write_stats(extra={
                    "phase": "champion_evaluation",
                    "evaluation_episodes": EVAL_SUITE_MIN_EPISODES,
                })
                print(f"Champion-Pruefung: {EVAL_SUITE_MIN_EPISODES} echte "
                      "Kampfepisoden; Rollouts pausieren waehrenddessen.", flush=True)
                # candidate and champion see the SAME scenarios and seeds
                eval_seed = 7000
                candidate = evaluate_live_battle_model(
                    trainer._model, sampler, episodes=EVAL_SUITE_MIN_EPISODES,
                    seed=eval_seed, schema=trainer.schema)
                catch_eval = None
                if trainer.schema == "v2":
                    catch_eval = evaluate_battle_catch_suite(
                        trainer._model, schema="v2")
                if not has_champion:
                    decision = first_champion_ok(candidate)
                    if catch_eval is not None:
                        _ok, _r = battle_catch_gate(catch_eval)
                        if not _ok:
                            decision = dict(decision, promote=False,
                                            reason="catch gate: " + "; ".join(_r))
                else:
                    decision = evaluate_battle_promotion(
                        candidate=candidate,
                        champion=trainer._champion_metrics_live(sampler, seed=eval_seed),
                        catch_eval=catch_eval)
                if decision["promote"]:
                    atomic_save_model(trainer._model, trainer._paths()["champion"])
                    trainer.champion_version += 1
                    print("BATTLE CHAMPION promoted: " + json.dumps(decision), flush=True)
                else:
                    print("Battle candidate rejected: " + json.dumps(decision), flush=True)
                trainer.next_promotion_step = next_promotion_step(
                    trainer.counters.env_steps)
                trainer.last_promotion = dict(decision)
                trainer.last_live_eval = dict(candidate)
                trainer._write_stats(extra={"phase": "training"})
    except KeyboardInterrupt:
        print("Battle-PPO wird beendet; letzter Resume-Stand ist gespeichert.")
    finally:
        if trainer._vec is not None:
            try:
                trainer._vec.close()
            except (EOFError, BrokenPipeError, OSError):
                # Ctrl+C races the SubprocVecEnv worker shutdown; the workers
                # are already gone. Nothing left to save here (train_chunk
                # persisted resume after every chunk).
                pass


if __name__ == "__main__":
    main()
