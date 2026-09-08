"""The single seam that wires the prepared 2x2 components into the live path.

Nothing here changes default behaviour: every hook is behind a
``twoby2.FEATURES`` gate that ships OFF. When a gate is on:

  * ``maybe_wrap_full_agent`` wraps a ``PokemonFireRedEnv`` with
    :class:`twoby2.nav_wrapper.NavigationBattleWrapper` driven by the REAL
    :class:`twoby2.emulator_battle_driver.EmulatorBattleDriver` (never the
    simulated one) + a pinned battle policy — so out of battle the navigation
    brain acts and only navigation transitions/rewards flow, and in battle the
    battle champion plays the whole fight and only a coarse strategic summary
    reaches navigation;
  * ``battle_driver_for`` gives the battle trainer / battle watcher the real
    emulator driver instead of ``SimulatedBattleDriver``.

Level / XP / KOs never become navigation progress: the wrapper only ever passes
:func:`twoby2.battle_summary.navigation_battle_reward` (coarse, no per-turn / KO
/ win term) and :mod:`twoby2.nav_progress` scrubs promotion metrics.
"""
from __future__ import annotations

import os
import gzip
import hashlib
import json
import time

from twoby2 import feature_enabled
from twoby2.nav_wrapper import NavigationBattleWrapper, BattleDriver as _WrapperDriver
from twoby2.battle_summary import summarize_battle
from battle_executor import ALL_MACROS

SCENARIO_CAP_PER_AREA_KIND = 24


def rule_battle_policy(obs):
    """Pick one legal semantic action from the verified rule controller."""
    from battle_controller import decide
    snap = (obs or {}).get("snap") or {}
    mask = list((obs or {}).get("mask") or [])
    choice = decide(snap).get("action", "MOVE_1")
    try:
        ix = ALL_MACROS.index(choice)
    except ValueError:
        ix = -1
    if 0 <= ix < len(mask) and mask[ix]:
        return ix
    return next((i for i, allowed in enumerate(mask) if allowed), 0)


_BATTLE_POLICY_STATUS = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "runtime", "battle", "battle_policy_status.json")


def _publish_battle_policy_status(**fields):
    """Make the live battle-policy choice visible (spec §1B): PPO vs rule, the
    schema, and — on a fallback — the exact reason. Never swallowed silently."""
    try:
        doc = {"schema": "battle_policy_status_v1",
               "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "pid": os.getpid(), **fields}
        os.makedirs(os.path.dirname(_BATTLE_POLICY_STATUS), exist_ok=True)
        tmp = _BATTLE_POLICY_STATUS + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, _BATTLE_POLICY_STATUS)
    except OSError:
        pass


def _detect_battle_schema(model):
    """Read the loaded model's own spaces and return "v1" / "v2" / None.
    None ⇒ an incompatible checkpoint the live path must NOT use with either env
    (spec §1B: no pad/truncate, no silent rule-fallback without a reason)."""
    try:
        vec = int(model.observation_space["vec"].shape[0])
        n = int(model.action_space.n)
    except Exception:
        return None, ("?", "?")
    if vec == 116 and n == 11:
        return "v1", (vec, n)
    if vec == 140 and n == 12:
        return "v2", (vec, n)
    return None, (vec, n)


class LatestBattleChampionPolicy:
    """Pin the newest valid Battle-PPO once per battle, with rule fallback.

    Every process keeps a small read-only cache. A newly promoted checkpoint is
    noticed before the next battle, never half-way through the current one. The
    obs/action space is built to match the LOADED champion's schema — a v1
    champion (116 / 11) gets a v1 obs vector, a v2 champion (140 / 12) gets the
    catch block. An unrecognised space is refused (rule fallback WITH a
    published reason), never reshaped.
    """
    def __init__(self, path=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.path = path or os.path.join(
            root, "runtime", "battle", "checkpoints", "battle_champion.zip")
        self._signature = None
        self._model = None
        self._schema = None          # "v1" | "v2" | None (incompatible)
        self._dims = ("?", "?")
        self._fallback_reason = "no champion file"

    def _load_current(self):
        try:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._model, self._schema = None, None
            self._fallback_reason = "no battle_champion.zip"
            return None
        if sig != self._signature:
            self._signature = sig
            try:
                # The battle trainer saves a MaskablePPO (sb3-contrib); a plain
                # stable_baselines3.PPO.load cannot reconstruct that policy.
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(self.path, device="cpu")
                schema, dims = _detect_battle_schema(model)
                self._dims = dims
                if schema is None:
                    self._model, self._schema = None, None
                    self._fallback_reason = (
                        f"champion space {dims} is neither v1 (116/11) nor "
                        f"v2 (140/12) — refusing to reshape")
                else:
                    self._model, self._schema = model, schema
                    self._fallback_reason = ""
            except Exception as exc:
                self._model, self._schema = None, None
                self._fallback_reason = f"MaskablePPO.load failed: {exc}"
        return self._model

    def _battle_obs(self, raw):
        from battle_env import BattleEnv, SimulatedBattleDriver
        snap = (raw or {}).get("snap") or {}
        mask = list((raw or {}).get("mask") or [])
        env = BattleEnv(SimulatedBattleDriver(), schema=self._schema or "v1")
        env._state = {
            "our_active": snap.get("player_active") or {},
            "enemy_active": snap.get("enemy_active") or {},
            "our_party": snap.get("player_party") or [],
            "enemy_party": snap.get("enemy_party") or [],
            "is_trainer": bool(snap.get("is_trainer")),
            "can_escape": bool(snap.get("can_escape")),
            "weather": snap.get("weather"),
            "turn": int(snap.get("turn", 0) or 0),
            "done": False,
            "outcome": None,
            "action_mask": mask,
        }
        env._turn = int(snap.get("turn", 0) or 0)
        env._objective = None            # live catch is gated: no catch obj here
        return env._obs()

    def pin(self):
        model = self._load_current()
        if model is None:
            _publish_battle_policy_status(
                battle_policy_source="rule",
                battle_policy_schema=None,
                battle_policy_champion_dims=list(self._dims),
                battle_policy_fallback_reason=self._fallback_reason
                or "no PPO champion")
            return rule_battle_policy

        _published = {"n": 0}

        def act(raw):
            mask = list((raw or {}).get("mask") or [])
            try:
                kw = {"deterministic": True}
                if mask and any(mask):
                    import numpy as np
                    kw["action_masks"] = np.asarray(mask, dtype=bool)
                action, _ = model.predict(self._battle_obs(raw), **kw)
                action = int(action)
                # spec §1B: a v1 champion can never return CATCH (index 11).
                if self._schema == "v1" and action >= 11:
                    raise RuntimeError("v1 champion returned an out-of-range index")
                if 0 <= action < len(mask) and mask[action]:
                    if not _published["n"]:
                        _published["n"] = 1
                        _publish_battle_policy_status(
                            battle_policy_source="ppo",
                            battle_policy_schema=self._schema,
                            battle_policy_champion_dims=list(self._dims),
                            battle_policy_fallback_reason="")
                    return action
                reason = f"predicted index {action} not in the live mask"
            except Exception as exc:
                reason = f"predict error: {exc}"
            _publish_battle_policy_status(
                battle_policy_source="rule",
                battle_policy_schema=self._schema,
                battle_policy_champion_dims=list(self._dims),
                battle_policy_fallback_reason=reason)
            return rule_battle_policy(raw)
        return act


latest_battle_champion_policy = LatestBattleChampionPolicy()


class LiveIntegrationBlocked(RuntimeError):
    pass


# --------------------------------------------------------------------------
# adapter: EmulatorBattleDriver  ->  NavigationBattleWrapper.BattleDriver
# --------------------------------------------------------------------------
class NavBattleDriverAdapter(_WrapperDriver):
    """Plays the in-battle sub-episode with the REAL emulator driver + a pinned
    battle policy. Never a SimulatedBattleDriver."""

    def __init__(self, emulator_driver_factory, battle_policy, *,
                 obs_builder=None, max_turns=80, agent_class="full_agent"):
        if agent_class not in ("full_agent", "watcher"):
            raise ValueError(
                f"NavBattleDriverAdapter agent_class must be 'full_agent' or "
                f"'watcher', got {agent_class!r}")
        self._make_driver = emulator_driver_factory
        self._policy = battle_policy
        self._obs = obs_builder or (lambda snap, mask: {"snap": snap, "mask": mask})
        self.max_turns = int(max_turns)
        # spec: which persistent shiny counter this adapter's wild encounters go
        # to. Set explicitly at construction — NEVER sniffed off ``env`` (the
        # wrapper's _twoby2_learning flag does not reach the underlying env
        # play_battle is called with).
        self.agent_class = agent_class

    def in_battle(self, env):
        drv = self._make_driver(env)
        return bool(drv.in_battle())

    def play_battle(self, env):
        drv = self._make_driver(env)
        drv.max_turns = self.max_turns
        enc_id = None
        agent_class = self.agent_class
        if hasattr(drv, "snapshot"):
            snap0 = drv.snapshot()
            _capture_healthy_full_battle_start(env, drv, snap0)
            enc_id = _record_full_wild_encounter(env, snap0, agent_class=agent_class)
        # pinned policy for the whole battle (no mid-battle swap)
        policy_fn = (self._policy.pin() if hasattr(self._policy, "pin")
                     else self._policy)
        result = drv.play_battle(policy_fn, self._obs)
        _record_full_wild_outcome(enc_id, result.get("outcome"))
        # post-battle overworld observation from the real env
        post_obs = None
        try:
            post_obs = env.unwrapped._observation() if hasattr(env.unwrapped, "_observation") \
                else getattr(env, "_last_obs", None)
        except Exception:
            post_obs = None
        raw = dict(result)
        raw.setdefault("duration_steps", int(result.get("duration_steps", 0)))
        raw["post_battle_obs"] = post_obs
        raw["terminated"] = bool(result.get("outcome") == "wipe")
        return raw


def _record_full_wild_encounter(nav_env, snap, *, agent_class="full_agent"):
    """spec ZIEL C: count a FULL / watcher wild encounter EXACTLY ONCE (its own
    counter file). Telemetry only — the navigation policy still gets ONLY the
    coarse strategic battle reward, nothing from this. Returns the encounter id
    for the outcome hook, or None (nothing to resolve while shiny RAM is
    unverified — the status is always "unknown")."""
    try:
        if snap.get("is_trainer") or snap.get("is_double") or snap.get("in_battle") is not True:
            return None
        import hashlib
        import firered_ram as fr
        from twoby2.shiny_counters import ShinyCounters
        from twoby2 import shiny_ram
        inner = getattr(nav_env, "unwrapped", nav_env)
        retro_env = getattr(inner, "env", inner)

        class _E:
            def get_ram(s):
                return retro_env.get_ram()
        enemy_party = fr.read_enemy_party(_E())
        loc = getattr(inner, "cached_loc", {}) or {}
        enemy = snap.get("enemy_active") or {}
        pid = shiny_ram.read_active_wild_pid(enemy_party)
        eid = hashlib.sha1("|".join(str(x) for x in (
            agent_class, loc.get("map_bank"), loc.get("map_id"),
            loc.get("x_pos"), loc.get("y_pos"),
            enemy.get("species_id"), enemy.get("level"), pid or "no_pid",
        )).encode()).hexdigest()[:16]
        st = shiny_ram.shiny_status(is_trainer=False, enemy_party=enemy_party)
        ShinyCounters(agent_class).record_encounter(
            eid, shiny_status=st["status"], species_id=enemy.get("species_id"),
            level=enemy.get("level"),
            area=f"map_{loc.get('map_bank')}_{loc.get('map_id')}",
            worker=getattr(inner, "instance_id", None))
        return (eid, agent_class) if st["status"] == "verified_shiny" else None
    except Exception:
        return None


def _record_full_wild_outcome(enc, outcome):
    """Record the ONE terminal outcome of a verified-shiny FULL / watcher
    encounter. ``enc`` is the ``(eid, agent_class)`` tuple from the encounter
    hook, or None. Unreachable while shiny RAM is unverified."""
    if not enc:
        return
    try:
        from twoby2.shiny_counters import ShinyCounters
        eid, agent_class = enc
        m = {"win": "ko", "wipe": "wipe", "fled": "fled",
             "caught": "caught"}.get(str(outcome), "unresolved")
        ShinyCounters(agent_class).record_shiny_outcome(eid, m)
    except Exception:
        pass


def _capture_healthy_full_battle_start(nav_env, driver, snap):
    """Publish a healthy real encounter for the isolated fighter pool.

    This is reward-free and never writes the protected navigation master or a
    frontier seed. Concurrent FULL workers serialize only the small JSON index.
    """
    try:
        import fcntl
        from checkpoint_health import party_health
        from twoby2.protected_assets import assert_write_target_ok, ROM_SHA256

        if snap.get("in_battle") is not True or snap.get("is_double"):
            return False
        party = snap.get("player_party") or []
        if party_health(party).get("party_ready") is not True:
            return False
        inner = getattr(nav_env, "unwrapped", nav_env)
        loc = getattr(inner, "cached_loc", {}) or {}
        bank = int(loc.get("map_bank", -1))
        map_id = int(loc.get("map_id", -1))
        if not loc.get("valid") or (bank, map_id) == (3, 0):
            return False
        names = {(3, 19): "route1", (3, 1): "viridian",
                 (3, 20): "route2", (1, 0): "viridian_forest",
                 (3, 2): "pewter"}
        area = names.get((bank, map_id), f"map_{bank}_{map_id}")
        kind = "trainer" if snap.get("is_trainer") else "wild"
        enemy = snap.get("enemy_active") or {}
        # spec ZIEL B: a coarse diversity bucket so no single seed / species
        # dominates the pool. The exact-HP signature still keeps genuinely
        # distinct starts apart.
        from twoby2.scenario_pool import scenario_bucket
        own_species = [m.get("species_id") for m in party]
        own_levels = [m.get("level") for m in party]
        bucket = scenario_bucket(
            area=area, enemy_species=[enemy.get("species_id")],
            enemy_level=enemy.get("level"), player_levels=own_levels,
            own_species=own_species, objective_mode="combat")
        signature_src = json.dumps({
            "area": area, "kind": kind,
            "enemy": [enemy.get("species_id"), enemy.get("level"),
                      enemy.get("cur_hp"), enemy.get("max_hp")],
            "party": [[m.get("species_id"), m.get("level"),
                       round(float(m.get("hp_ratio", 0.0)), 1)] for m in party],
        }, sort_keys=True).encode()
        signature = hashlib.sha1(signature_src).hexdigest()[:16]

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        outdir = os.path.join(root, "runtime", "battle", "scenarios")
        os.makedirs(outdir, exist_ok=True)
        lock_path = os.path.join(outdir, ".capture.lock")
        index_path = os.path.join(outdir, "index.json")
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                with open(index_path) as f:
                    index = json.load(f)
            except Exception:
                index = {"schema": "live_battle_scenarios_v1", "scenarios": []}
            rows = list(index.get("scenarios") or [])
            if any(r.get("signature") == signature for r in rows):
                return False
            total_level = sum(int(m.get("level", 0) or 0) for m in party)
            # spec ZIEL B: cap PER BUCKET, not just per area+kind, so the pool
            # spreads across species / levels / party shapes. A new bucket is
            # always welcome; a full bucket only takes a materially stronger
            # start (replacing its weakest member).
            same_bucket = [r for r in rows if r.get("bucket") == bucket]
            if len(same_bucket) >= 3:
                weakest = min(same_bucket,
                              key=lambda r: int(r.get("party_total_level", 0)))
                if total_level <= int(weakest.get("party_total_level", 0)):
                    return False
                rows.remove(weakest)
            peers = [r for r in rows if r.get("area") == area
                     and r.get("battle_kind") == kind]
            if len(peers) >= SCENARIO_CAP_PER_AREA_KIND:
                weakest = min(peers, key=lambda r: int(r.get("party_total_level", 0)))
                if total_level <= int(weakest.get("party_total_level", 0)):
                    return False
                if weakest in rows:
                    rows.remove(weakest)  # state file remains recoverable on disk

            state_path = os.path.join(
                outdir, f"{area}_{kind}_{signature}_{os.getpid()}.state.gz")
            assert_write_target_ok(state_path, op="write")
            tmp = state_path + ".tmp"
            with gzip.open(tmp, "wb") as f:
                f.write(driver.env.em.get_state())
            os.replace(tmp, state_path)
            h = hashlib.sha256()
            with open(state_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            rows.append({
                "id": f"{area}_{kind}_{signature}", "signature": signature,
                "bucket": bucket,
                "area": area, "map_bank": bank, "map_id": map_id,
                "battle_kind": kind,
                "is_trainer": kind == "trainer", "can_escape": kind == "wild",
                "savestate_path": state_path, "scenario_sha256": h.hexdigest(),
                "rom_sha256": ROM_SHA256, "source": "healthy_full_agent",
                "party_ready": True, "party_total_level": total_level,
                # spec ZIEL B metadata
                "enemy_species": enemy.get("species_id"),
                "enemy_level": enemy.get("level"),
                "enemy_hp": enemy.get("cur_hp"), "enemy_max_hp": enemy.get("max_hp"),
                "player_levels": own_levels, "own_species": own_species,
                "player_hp": [m.get("cur_hp") for m in party],
                "objective_mode": "combat",
                "ram_readable": bool(snap.get("ram_ok", True)),
                "schema_version": snap.get("schema", "battle_snapshot_v2_live"),
                "trained_count": 0,
                "captured_at": time.time(),
            })
            tmp_index = index_path + f".{os.getpid()}.tmp"
            with open(tmp_index, "w") as f:
                json.dump({"schema": "live_battle_scenarios_v1",
                           "scenarios": rows}, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp_index, index_path)
            return True
    except Exception:
        # Scenario collection may never interrupt a FULL run.
        return False


# --------------------------------------------------------------------------
# battle driver selection for battle_train.py / battle_watch.py
# --------------------------------------------------------------------------
def battle_driver_for(live_env=None, *, allow_simulated_fallback=True, schema="v1"):
    """Return the driver the battle trainer / watcher should use.

    * gate ``battle_env`` ON + a live emulator env supplied -> real
      :class:`EmulatorBattleDriver`;
    * otherwise -> :class:`battle_env.SimulatedBattleDriver` (offline bring-up /
      unit tests only) unless ``allow_simulated_fallback`` is False, in which
      case this raises.

    ``schema`` defaults to ``"v1"`` — the live combat path. v2 training passes
    ``"v2"`` explicitly; there is no live v2 until it passes the full eval.
    """
    if feature_enabled("battle_env") and live_env is not None:
        from twoby2.emulator_battle_driver import EmulatorBattleDriver
        return EmulatorBattleDriver(live_env, schema=schema)
    if not allow_simulated_fallback:
        raise LiveIntegrationBlocked(
            "battle_env gate is off or no live env — a SimulatedBattleDriver "
            "must not be used in the live training path")
    from battle_env import SimulatedBattleDriver
    return SimulatedBattleDriver()


def assert_no_simulated_driver_in_live_path(driver):
    from battle_env import SimulatedBattleDriver
    if isinstance(driver, SimulatedBattleDriver) and feature_enabled("battle_env"):
        raise LiveIntegrationBlocked(
            "SimulatedBattleDriver in the live path while battle_env is ON")
    return True


# --------------------------------------------------------------------------
# FULL-agent / FULL-watcher env wrapping
# --------------------------------------------------------------------------
def maybe_wrap_full_agent(env, *, learning=True, battle_policy=None,
                          emulator_driver_factory=None, obs_builder=None,
                          battle_schema="v1"):
    """Wrap ``env`` with the navigation/battle boundary when the
    ``nav_battle_wrapper`` gate is ON. Otherwise return ``env`` unchanged
    (current single-PPO behaviour).

    ``learning`` is False for the FULL-watcher: it uses the exact same wrapper /
    router / champions but its wrapper never contributes rollouts.
    ``battle_schema`` is ``"v1"`` for the live combat path (the only live one).
    """
    if not feature_enabled("nav_battle_wrapper"):
        return env
    if battle_policy is None:
        raise LiveIntegrationBlocked(
            "nav_battle_wrapper is ON but no real battle policy was supplied")
    factory = emulator_driver_factory or (
        lambda e: _default_emulator_driver_factory(e, schema=battle_schema))
    adapter = NavBattleDriverAdapter(
        factory, battle_policy, obs_builder=obs_builder,
        agent_class=("full_agent" if learning else "watcher"))
    wrapped = NavigationBattleWrapper(env, adapter)
    wrapped._twoby2_learning = bool(learning)
    return wrapped


def _default_emulator_driver_factory(env, *, schema="v1"):
    from twoby2.emulator_battle_driver import EmulatorBattleDriver
    inner = getattr(env, "unwrapped", env)
    retro_env = getattr(inner, "env", inner)   # PokemonFireRedEnv.env is the retro env
    return EmulatorBattleDriver(retro_env, schema=schema)


def integration_status():
    """What the preflight / dashboard report."""
    return {
        "nav_battle_wrapper_gate": feature_enabled("nav_battle_wrapper"),
        "battle_env_gate": feature_enabled("battle_env"),
        "wrapper": "twoby2.nav_wrapper.NavigationBattleWrapper",
        "battle_driver": "twoby2.emulator_battle_driver.EmulatorBattleDriver",
        "simulated_driver_in_live_path": False,
        "battle_policy": "latest valid Battle-PPO pinned per battle; rule fallback",
        "nav_reward_from_battle": "coarse summary only (navigation_battle_reward)",
    }
