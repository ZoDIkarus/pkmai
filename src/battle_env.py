"""Isolated Gymnasium battle environment (Phase 2).

NOT a ``pokemon_env`` navigation episode. Its own observation (compact battle
RAM vector + known-masks), its own macro action space, its own anti-farming
reward, its own episode clock. It drives a battle through an injected
``BattleDriver``:

  * :class:`SimulatedBattleDriver` — deterministic in-memory resolver built on
    :mod:`battle_engine`; used for tests and offline Battle-PPO bring-up.
  * ``EmulatorBattleDriver`` (in :mod:`battle_executor` land, gated) — the real
    emulator + macro executor; blocked until the battle-menu RAM is verified.

Isolation: imports ``battle_engine`` / ``battle_types`` / ``battle_executor``
only. No navigation, tile, map, stage or story reward. No navigation counter.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import battle_types as bt
from battle_engine import calc_damage, effective_speed, move_hits_probability
from battle_executor import (ALL_MACROS, ALL_MACROS_V1, MOVE_ACTIONS,
                             SWITCH_ACTIONS, RUN_ACTION, CATCH_ACTION,
                             ACTIONS_SCHEMA_V1, ACTIONS_SCHEMA_V2)
import battle_catch

OBS_SCHEMA_V1 = "battle_obs_v1"
OBS_SCHEMA = "battle_obs_v2_catch"
REWARD_SCHEMA = "battle_reward_v2_catch"
REWARD_SCHEMA_V1 = "battle_reward_v1"

# ordered scalar fields; each real field is followed by a 0/1 "_known" entry
_MON_FIELDS = ("hp_frac", "level", "status", "type1", "type2",
               "atk", "df", "spa", "spd", "spe")
_MOVE_FIELDS = ("power", "type", "pp_frac", "acc", "prio", "stab", "eff",
                "is_status")
_GLOBAL_FIELDS = ("is_trainer", "can_escape", "party_alive", "party_avg_hp",
                  "weather", "turn")
# v2: the strategic catch objective + live catch state. value+known each.
_CATCH_FIELDS = ("objective_is_catch", "target_species_normalized",
                 "enemy_is_target", "species_new_this_run",
                 "usable_ball_count_normalized", "ball_reserve_normalized",
                 "party_has_space", "pc_capture_supported", "catch_valid",
                 "catch_probability_estimate", "last_catch_failed",
                 "catch_attempts_normalized")

_OBS_DIM_V1 = (
    2 * len(_MON_FIELDS) * 2
    + 4 * len(_MOVE_FIELDS) * 2
    + len(_GLOBAL_FIELDS) * 2
)
OBS_DIM = _OBS_DIM_V1 + len(_CATCH_FIELDS) * 2
OBS_DIM_V1 = _OBS_DIM_V1
SPECIES_ID_MAX = 411          # supported Gen-III range
BALL_COUNT_CLIP = 20
CATCH_ATTEMPTS_CLIP = 8

# schema -> (macro tuple, obs width, obs-schema id, actions-schema id, reward id)
_SCHEMAS = {
    "v1": (ALL_MACROS_V1, _OBS_DIM_V1, OBS_SCHEMA_V1, ACTIONS_SCHEMA_V1,
           REWARD_SCHEMA_V1),
    "v2": (ALL_MACROS, OBS_DIM, OBS_SCHEMA, ACTIONS_SCHEMA_V2, REWARD_SCHEMA),
}


def battle_schema_spec(schema):
    """``(macros, obs_dim, obs_schema, actions_schema, reward_schema)`` for
    ``"v1"`` (combat only, 116 / 11) or ``"v2"`` (adds CATCH, 140 / 12)."""
    if schema not in _SCHEMAS:
        raise ValueError(f"unknown battle schema {schema!r}; use 'v1' or 'v2'")
    return _SCHEMAS[schema]


class BattleRewardConfig:
    DAMAGE_PER_HP = 0.01              # capped at the enemy's REAL prior HP
    ENEMY_KO = 1.0                    # once per enemy
    BATTLE_WIN = 3.0                 # once
    OWN_FAINT = -1.0
    WIPE = -3.0
    INVALID_ACTION = -0.3
    WASTED_TURN = -0.05             # legal but no effect (0 PP slipped through)
    SWITCH_LOOP = -0.5             # A->B->A inside the window
    SWITCH_LOOP_WINDOW = 4
    MENU_MOVEMENT = 0.0
    # The goal of an eval / training battle is to WIN. A wild flee used to cost
    # only -0.1, which is far cheaper than a -3.0 wipe, so PPO could learn to
    # bail out of any hard fight. Fleeing must be clearly worse than trying and
    # roughly on par with losing a mon, well short of a full wipe.
    FLEE_WILD_OK = -1.5
    FLEE_TRAINER_ILLEGAL = -1.0
    TURN_COST = -0.01              # mild pressure to end fights

    # --- battle_reward_v2_catch (spec §9) --------------------------------
    CATCH_SUCCESS = 3.0           # own terminal outcome; NEVER stacks with KO/win
    CATCH_TARGET_KO = -1.5        # KOing the mon you were told to catch
    FAILED_CATCH = -0.05
    BALL_COST = -0.05            # per ball actually consumed
    PREMATURE_CATCH = -0.10     # ball thrown while p(catch) < PREMATURE_MIN_PROB
    PREMATURE_MIN_PROB = 0.25
    UNREQUESTED_CATCH = -0.50   # CATCH chosen in a combat objective
    ILLEGAL_TRAINER_CATCH = -0.50
    # catch-mode damage shaping only pays while the target is ABOVE the window
    CATCH_SAFE_WINDOW_PROB = 0.60

    # --- shiny (spec ZIEL D) — PREPARED, dormant until shiny RAM verified ---
    # A verified-shiny catch is a bounded, ONE-OFF bonus on top of catch_success.
    # It never pays for encountering / re-seeing / attempting / fleeing / KOing.
    SHINY_CATCH_SUCCESS = 10.0


# spec ZIEL D.5: a DASHBOARD / evaluation weight only. It is NEVER added to
# step_reward or episode_reward — a "+2000 in the PPO reward" is explicitly
# forbidden. Catch priority for a shiny is enforced via the planner + action
# mask, not via reward magnitude.
SHINY_PRIORITY_SCORE = 2000


def _reward_components_v2(objective, ev):
    """The ONE reward function (spec §9/§10). Returns a list of
    ``(name, amount)`` whose sum IS the step reward - no residual, no
    reconstruction from logs. ``objective`` decides combat vs catch."""
    C = BattleRewardConfig
    ev = ev or {}
    obj = objective or {}
    is_catch = bool(obj.get("catch_requested"))
    bits = []

    if ev.get("invalid"):
        # an invalid action still consumed the turn; a rejected CATCH must NOT
        # have consumed a ball, so nothing else pays.
        bits.append(("turn_cost", C.TURN_COST))
        bits.append(("invalid_action", C.INVALID_ACTION))
        if ev.get("illegal_trainer_catch"):
            bits.append(("illegal_trainer_catch", C.ILLEGAL_TRAINER_CATCH))
        return bits

    bits.append(("turn_cost", C.TURN_COST))
    if ev.get("wasted"):
        bits.append(("wasted_turn", C.WASTED_TURN))

    # ---- catch outcomes (both modes) ----
    if ev.get("unrequested_catch"):
        bits.append(("unrequested_catch", C.UNREQUESTED_CATCH))
    if ev.get("illegal_trainer_catch"):
        bits.append(("illegal_trainer_catch", C.ILLEGAL_TRAINER_CATCH))
    balls = int(ev.get("balls_used", 0) or 0)
    if balls:
        bits.append(("ball_cost", round(C.BALL_COST * balls, 4)))
    if ev.get("premature_catch_attempt"):
        bits.append(("premature_catch_attempt", C.PREMATURE_CATCH))

    caught = bool(ev.get("catch_success"))
    if caught and is_catch:
        bits.append(("catch_success", C.CATCH_SUCCESS))
        # spec ZIEL D: bounded one-off shiny bonus — ONLY on a RAM-confirmed
        # catch of a verified shiny. Never for encounter / repeat / attempt /
        # flee / KO. Dormant until twoby2.shiny_ram is verified.
        if ev.get("shiny_catch_success") and obj.get("catch_reason") == "verified_shiny":
            bits.append(("shiny_catch_success", C.SHINY_CATCH_SUCCESS))
    elif ev.get("catch_attempted") and not caught and ev.get("catch_outcome") == "broke_free":
        bits.append(("failed_catch", C.FAILED_CATCH))

    # ---- damage / KO / win ----
    dmg = max(0, int(ev.get("our_damage_dealt", 0) or 0))
    prior = max(1, int(ev.get("enemy_hp_before", 1) or 1))
    if dmg and not caught:
        pay_dmg = True
        if is_catch:
            # spec §9B: no positive damage reward below the safe catch window
            win_hp = ev.get("catch_safe_window_hp")
            hp_after = ev.get("enemy_hp_after")
            if win_hp is not None and hp_after is not None and hp_after <= win_hp:
                pay_dmg = False
        if pay_dmg:
            bits.append(("damage", round(C.DAMAGE_PER_HP * min(dmg, prior), 4)))

    target_ko = bool(ev.get("enemy_ko"))
    if caught:
        pass                                    # caught -> no KO / no win, ever
    elif target_ko and is_catch:
        bits.append(("catch_target_ko", C.CATCH_TARGET_KO))
    elif target_ko:
        bits.append(("enemy_ko", C.ENEMY_KO))
        if ev.get("battle_won"):
            bits.append(("battle_win", C.BATTLE_WIN))
    elif ev.get("battle_won") and not is_catch:
        bits.append(("battle_win", C.BATTLE_WIN))

    if ev.get("own_faint"):
        bits.append(("own_faint", C.OWN_FAINT))
    if ev.get("wipe"):
        bits.append(("wipe", C.WIPE))
    if ev.get("illegal_flee"):
        bits.append(("flee_illegal", C.FLEE_TRAINER_ILLEGAL))
    if ev.get("fled"):
        bits.append(("fled", C.FLEE_WILD_OK))
    if ev.get("switch_loop"):
        bits.append(("switch_loop", C.SWITCH_LOOP))
    return bits


def decompose_battle_reward(ev, total_reward=None):
    """Back-compat shim: the reward components for one turn. When ``ev`` carries
    an ``objective`` it is used; otherwise a combat objective is assumed. The
    components ALWAYS sum to the step reward exactly (spec §10) - a leftover
    ``other`` term only appears if a caller passed a mismatching total."""
    ev = ev or {}
    bits = _reward_components_v2(ev.get("objective"), ev)
    if total_reward is not None:
        resid = round(float(total_reward) - sum(a for _, a in bits), 4)
        if abs(resid) > 1e-4:
            bits.append(("other", resid))
    return bits


class BattleDriver:
    """Interface the env needs."""

    def reset(self, scenario):
        """Return the initial battle-state dict."""
        raise NotImplementedError

    def state(self):
        raise NotImplementedError

    def apply_macro(self, macro):
        """Advance one full turn (our macro + the opponent's reply). Return
        ``(new_state, event)`` where ``event`` carries the anti-farm facts:
        {"our_damage_dealt", "enemy_hp_before", "enemy_ko", "battle_won",
         "own_faint", "wipe", "invalid", "wasted", "fled", "illegal_flee"}."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# deterministic simulator driver (battle_engine based)
# --------------------------------------------------------------------------
class SimulatedBattleDriver(BattleDriver):
    """Full-information deterministic battle resolver. Uses expected damage
    (mean of the 16 rolls). Enough to train / test the macro policy offline."""

    def __init__(self, *, expected_roll=True):
        self.expected_roll = expected_roll
        self._s = None
        self._objective = None
        self._catch_rng = None

    def set_objective(self, objective):
        self._objective = dict(objective or {})

    def reset(self, scenario):
        import random as _random
        enemy = [dict(m) for m in scenario["enemy_party"]]
        # sim catch inputs come from the scenario, deterministic per catch_seed
        tgt = int(scenario.get("target_species_id", 0) or 0)
        if enemy:
            enemy[0].setdefault("species_id", tgt or enemy[0].get("species_id", 0))
            if "catch_rate" in scenario or "enemy_catch_rate" in scenario:
                enemy[0]["catch_rate"] = int(scenario.get(
                    "catch_rate", scenario.get("enemy_catch_rate", 45)))
        balls = int(scenario.get("ball_inventory", 0) or 0)
        reserve = int(scenario.get("ball_reserve", 0) or 0)
        self._catch_rng = _random.Random(int(scenario.get("catch_seed", 0) or 0))
        self._s = {
            "our_party": [dict(m) for m in scenario["our_party"]],
            "enemy_party": enemy,
            "our_active": 0,
            "enemy_active": 0,
            "is_trainer": bool(scenario.get("is_trainer", False)),
            "can_escape": bool(scenario.get("can_escape", not scenario.get("is_trainer", False))),
            "weather": scenario.get("weather"),
            "turn": 0,
            "enemy_ko_credited": set(),
            "done": False,
            "outcome": None,
            "ball_count": balls,
            "ball_reserve": reserve,
            "ball_id": int(scenario.get("ball_id", battle_catch.POKE_BALL)),
            "best_ball_id": int(scenario.get("ball_id", battle_catch.POKE_BALL)),
            "party_has_space": bool(scenario.get("party_has_space",
                                                len(scenario["our_party"]) < 6)),
            "pc_capture_supported": bool(scenario.get("pc_capture_supported", False)),
            "message_pending": False, "animation_active": False, "menu_ready": True,
        }
        return self.state()

    # -- helpers -----
    def _our(self):
        return self._s["our_party"][self._s["our_active"]]

    def _enemy(self):
        return self._s["enemy_party"][self._s["enemy_active"]]

    def _alive(self, party):
        return [i for i, m in enumerate(party) if int(m.get("cur_hp", 0)) > 0]

    def _dmg(self, atk, dfn, move):
        d = calc_damage(atk, dfn, move,
                        ability_known=True, in_battle_types_authoritative=True,
                        stat_stages_known=True, weather_known=True,
                        screens_known=True, item_known=True)
        if not d.get("is_damaging"):
            return 0
        return int(round(d["expected"])) if self.expected_roll else d["min_damage"]

    def _enemy_choose_move(self):
        e = self._enemy()
        best, bestdmg = None, -1
        for mv in e.get("moves") or []:
            if int(mv.get("pp", 0)) <= 0 or mv.get("power") in (0, None):
                continue
            dm = self._dmg(e, self._our(), mv)
            if dm > bestdmg:
                best, bestdmg = mv, dm
        return best

    def state(self):
        s = self._s
        return {
            "our_active": dict(self._our()),
            "enemy_active": dict(self._enemy()),
            "our_party": [dict(m) for m in s["our_party"]],
            "enemy_party": [dict(m) for m in s["enemy_party"]],
            "is_trainer": s["is_trainer"],
            "can_escape": s["can_escape"],
            "weather": s["weather"],
            "turn": s["turn"],
            "done": s["done"],
            "outcome": s["outcome"],
            "best_ball_id": s.get("best_ball_id", battle_catch.POKE_BALL),
            "usable_balls": max(0, int(s.get("ball_count", 0)) - int(s.get("ball_reserve", 0))),
            "party_has_space": s.get("party_has_space", True),
            "pc_capture_supported": s.get("pc_capture_supported", False),
            "message_pending": False, "animation_active": False, "menu_ready": True,
        }

    def _catch_bonus(self):
        enemy = self._enemy()
        b = battle_catch.ball_bonus(int(self._s.get("ball_id", battle_catch.POKE_BALL)),
                                    target_types=enemy.get("types") or [],
                                    target_level=enemy.get("level"))
        return b if b is not None else 1.0

    def _do_catch(self, ev):
        s = self._s
        enemy = self._enemy()
        obj = self._objective or {}
        # precheck mirror (the executor is authoritative live; here we mirror it)
        if s["is_trainer"]:
            ev.update(invalid=True, catch_outcome="illegal_trainer_catch",
                      illegal_trainer_catch=True)
            return self.state(), ev
        usable = int(s.get("ball_count", 0)) - int(s.get("ball_reserve", 0))
        if usable <= 0:
            ev.update(invalid=True, catch_outcome="no_balls")
            return self.state(), ev
        if not s.get("party_has_space") and not s.get("pc_capture_supported"):
            ev.update(invalid=True, catch_outcome="party_full")
            return self.state(), ev
        if int((enemy or {}).get("species_id", 0) or 0) != int(obj.get("target_species_id", 0) or 0) \
                and obj.get("catch_requested"):
            ev.update(invalid=True, catch_outcome="wrong_target")
            return self.state(), ev

        cr = enemy.get("catch_rate")
        ch, mh = int(enemy.get("cur_hp", 0)), int(enemy.get("max_hp", 1))
        bonus = self._catch_bonus()
        sb = battle_catch.status_bonus(enemy.get("status_name") or enemy.get("status"))
        p = battle_catch.catch_probability(catch_rate=cr, cur_hp=ch, max_hp=mh,
                                           ball_bonus=bonus, status_bonus=sb)
        win_hp = battle_catch.safe_catch_window(
            catch_rate=cr, max_hp=mh, ball_bonus=bonus, status_bonus=sb,
            target_prob=BattleRewardConfig.CATCH_SAFE_WINDOW_PROB)

        s["ball_count"] = int(s.get("ball_count", 0)) - 1
        ev["balls_used"] = 1
        ev["catch_attempted"] = True
        ev["catch_prob"] = p
        ev["enemy_hp_after"] = ch
        ev["catch_safe_window_hp"] = win_hp
        if (p is not None and p < BattleRewardConfig.PREMATURE_MIN_PROB):
            ev["premature_catch_attempt"] = True

        roll = self._catch_rng.random()
        if p is not None and roll < p:
            ev.update(catch_success=True, catch_outcome="caught",
                      caught_species_id=int(enemy.get("species_id", 0) or 0),
                      is_new_species=bool(obj.get("is_new_species_this_run")))
            s["done"] = True
            s["outcome"] = ("sent_to_pc" if not s.get("party_has_space")
                            and s.get("pc_capture_supported") else "caught")
            return self.state(), ev
        ev["catch_outcome"] = "broke_free"
        return self.state(), ev

    def apply_macro(self, macro):
        s = self._s
        ev = {"our_damage_dealt": 0, "enemy_hp_before": int(self._enemy().get("cur_hp", 0)),
              "enemy_ko": False, "battle_won": False, "own_faint": False,
              "wipe": False, "invalid": False, "wasted": False, "fled": False,
              "illegal_flee": False}
        if s["done"]:
            ev["invalid"] = True
            return self.state(), ev
        s["turn"] += 1

        if macro == CATCH_ACTION:
            return self._do_catch(ev)

        our, enemy = self._our(), self._enemy()
        our_move = None

        if macro == RUN_ACTION:
            if s["is_trainer"] or not s["can_escape"]:
                ev["illegal_flee"] = True
            else:
                ev["fled"] = True
                s["done"] = True
                s["outcome"] = "fled"
                return self.state(), ev
        elif macro in SWITCH_ACTIONS:
            slot = SWITCH_ACTIONS.index(macro)
            if (slot < len(s["our_party"]) and slot != s["our_active"]
                    and int(s["our_party"][slot].get("cur_hp", 0)) > 0):
                s["our_active"] = slot
                our = self._our()
            else:
                ev["invalid"] = True
                return self.state(), ev
        elif macro in MOVE_ACTIONS:
            mi = MOVE_ACTIONS.index(macro)
            moves = our.get("moves") or []
            if mi >= len(moves):
                ev["invalid"] = True
                return self.state(), ev
            our_move = moves[mi]
            if int(our_move.get("pp", 0)) <= 0:
                ev["wasted"] = True
                our_move = None
        else:
            ev["invalid"] = True
            return self.state(), ev

        # resolve order (simplified: priority then speed; switch acts first)
        enemy_move = self._enemy_choose_move()
        our_first = macro in SWITCH_ACTIONS or _speed(our) >= _speed(enemy)

        def our_turn():
            if our_move is None:
                return
            our_move["pp"] = int(our_move.get("pp", 0)) - 1
            dm = self._dmg(our, enemy, our_move)
            dm = min(dm, int(enemy.get("cur_hp", 0)))
            enemy["cur_hp"] = int(enemy.get("cur_hp", 0)) - dm
            ev["our_damage_dealt"] = dm
            if enemy["cur_hp"] <= 0 and s["enemy_active"] not in s["enemy_ko_credited"]:
                ev["enemy_ko"] = True
                s["enemy_ko_credited"].add(s["enemy_active"])

        def enemy_turn():
            if enemy_move is None or int(enemy.get("cur_hp", 0)) <= 0:
                return
            enemy_move["pp"] = int(enemy_move.get("pp", 0)) - 1
            dm = self._dmg(enemy, our, enemy_move)
            our["cur_hp"] = max(0, int(our.get("cur_hp", 0)) - dm)

        for fn in ((our_turn, enemy_turn) if our_first else (enemy_turn, our_turn)):
            fn()

        # catch-mode damage shaping (spec §9B): expose the post-hit target HP and
        # the safe catch window so _reward_components_v2 stops paying positive
        # damage once the target has dropped into the window.
        ev["enemy_hp_after"] = max(0, int(enemy.get("cur_hp", 0)))
        _obj = self._objective or {}
        if _obj.get("catch_requested"):
            _win = battle_catch.safe_catch_window(
                catch_rate=enemy.get("catch_rate"),
                max_hp=int(enemy.get("max_hp", 1) or 1),
                ball_bonus=self._catch_bonus(),
                status_bonus=battle_catch.status_bonus(
                    enemy.get("status_name") or enemy.get("status")),
                target_prob=BattleRewardConfig.CATCH_SAFE_WINDOW_PROB)
            if _win is not None:
                ev["catch_safe_window_hp"] = _win

        # faints / switches
        if int(our.get("cur_hp", 0)) <= 0:
            ev["own_faint"] = True
            alive = self._alive(s["our_party"])
            if alive:
                s["our_active"] = alive[0]
            else:
                ev["wipe"] = True
                s["done"] = True
                s["outcome"] = "wipe"
                return self.state(), ev
        if int(enemy.get("cur_hp", 0)) <= 0:
            alive = self._alive(s["enemy_party"])
            if alive:
                s["enemy_active"] = alive[0]
            else:
                ev["battle_won"] = True
                s["done"] = True
                s["outcome"] = "win"
        return self.state(), ev


def _speed(mon):
    sp = effective_speed(mon)
    return sp if sp is not None else int((mon.get("stats") or {}).get("speed", 0) or 0)


# --------------------------------------------------------------------------
# the environment
# --------------------------------------------------------------------------
class BattleEnv(gym.Env):
    metadata = {"render_modes": []}
    obs_schema = OBS_SCHEMA

    def __init__(self, driver=None, *, scenario_sampler=None, max_turns=60,
                 mirror_sidecar=None, schema="v2"):
        super().__init__()
        self.driver = driver or SimulatedBattleDriver()
        self.scenario_sampler = scenario_sampler
        self.max_turns = int(max_turns)
        # schema "v1" = combat only (116 / 11); "v2" = adds CATCH (140 / 12).
        # A v1 env has NO catch obs block and CATCH is never in its action
        # space, so a v1 champion loads + predicts against it unchanged.
        (self._macros, self._obs_dim, self._obs_schema,
         self._actions_schema, self._reward_schema) = battle_schema_spec(schema)
        self.schema = schema
        self.obs_schema = self._obs_schema
        self.action_space = spaces.Discrete(len(self._macros))
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(-1.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(self._macros)),
        })
        self._state = None
        self._recent_switches = []
        self._turn = 0
        # Only the visible mirror worker sets this: a JSON sidecar next to the
        # frame JPEG so `battle_mirror_watch.py` can show the reward stream
        # (like the navigation watcher). Inference/headless workers leave it None.
        self._mirror_sidecar = mirror_sidecar
        self._mirror_events = []
        self._ep_reward = 0.0
        self._episode_no = 0
        self._objective = None          # immutable BattleObjective for this battle
        self._catch_attempts = 0
        self._last_catch_failed = False

    def close(self):
        """Release the isolated emulator owned by a live battle worker."""
        retro_env = getattr(self, "_retro_env", None)
        if retro_env is not None:
            retro_env.close()
            self._retro_env = None
        return super().close()

    # -- gym API -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None and self.scenario_sampler is not None:
            scenario = self.scenario_sampler(self.np_random)
        if scenario is None:
            scenario = default_scenario()
        self._state = self.driver.reset(scenario)
        # the immutable battle objective (spec §2). Isolated training: the
        # scenario is the authority. The Battle-PPO never sees this creation.
        from catch_planner import objective_from_scenario
        self._objective = objective_from_scenario(scenario).frozen()
        if hasattr(self.driver, "set_objective"):
            self.driver.set_objective(self._objective)
        self._recent_switches = []
        self._turn = 0
        self._ep_reward = 0.0
        self._catch_attempts = 0
        self._last_catch_failed = False
        self._episode_no += 1
        self._publish_mirror(None, 0.0, {}, None, reset=True)
        return self._obs(), {"obs_schema": self._obs_schema,
                             "actions_schema": self._actions_schema,
                             "objective": dict(self._objective)}

    def step(self, action):
        macro = (self._macros[int(action)]
                 if 0 <= int(action) < len(self._macros) else None)
        self._turn += 1
        # a driver that reset into an unreadable state ends the episode now
        # (turn 0) rather than burning MAX_TURNS on a dead battle.
        if (self._state or {}).get("done"):
            oc = self._state.get("outcome")
            self._publish_mirror(macro, 0.0, {}, oc)
            return (self._obs(), 0.0, True, False,
                    {"macro": macro, "outcome": oc,
                     "unreadable": oc in ("unreadable", "reset_unreadable")})
        legal = self._legal_mask()
        info = {"macro": macro}
        _inv = {"invalid": True, "objective": dict(self._objective or {})}

        if macro is None or not legal[int(action)]:
            _comps = _reward_components_v2(self._objective, _inv)
            reward = float(sum(a for _, a in _comps))
            # Vanilla SB3 PPO observes the mask but does not apply it to its
            # categorical sampler. In the live emulator, advance with the
            # first verified legal macro so a young policy cannot freeze on
            # an impossible switch. The selected action still receives only
            # the invalid penalty; fallback damage/wins are not rewarded.
            if getattr(self.driver, "advance_on_invalid", False) and any(legal):
                fallback_ix = next(i for i, allowed in enumerate(legal) if allowed)
                fallback = self._macros[fallback_ix]
                state, ev = self.driver.apply_macro(fallback)
                self._state = state
                terminated = bool(state.get("done"))
                truncated = (not terminated) and self._turn >= self.max_turns
                self._publish_mirror(macro, float(reward),
                                     {**ev, "invalid": True}, state.get("outcome"))
                return self._obs(), float(reward), terminated, truncated, {
                    **info, **ev, "invalid": True, "reward_components": _comps,
                    "fallback_macro": fallback,
                    "outcome": state.get("outcome"),
                }
            terminated = False
            truncated = self._turn >= self.max_turns
            self._publish_mirror(macro, float(reward), _inv, None)
            return self._obs(), float(reward), terminated, truncated, {
                **info, "invalid": True, "reward_components": _comps}

        state, ev = self.driver.apply_macro(macro)
        self._state = state
        ev = dict(ev or {})
        ev["objective"] = dict(self._objective or {})
        if macro == CATCH_ACTION:
            self._catch_attempts += 1
            if not self._objective or not self._objective.get("catch_requested"):
                ev["unrequested_catch"] = True
            if ev.get("catch_outcome") == "illegal_trainer_catch":
                ev["illegal_trainer_catch"] = True
            self._last_catch_failed = ev.get("catch_outcome") == "broke_free"
        if macro in SWITCH_ACTIONS:
            self._recent_switches.append(macro)
            self._recent_switches = self._recent_switches[-BattleRewardConfig.SWITCH_LOOP_WINDOW:]
            if _is_switch_loop(self._recent_switches):
                ev["switch_loop"] = True
        # THE reward: one function, components sum exactly to the step reward.
        comps = _reward_components_v2(self._objective, ev)
        reward = float(sum(a for _, a in comps))

        terminated = bool(state.get("done"))
        truncated = (not terminated) and self._turn >= self.max_turns
        outcome = state.get("outcome")
        if truncated and not outcome:
            outcome = "timeout"
        info.update(ev, outcome=outcome, reward_components=comps,
                    catch_attempts=self._catch_attempts)
        self._publish_mirror(macro, float(reward), ev, outcome)
        return self._obs(), float(reward), terminated, truncated, info

    # -- reward ------------------------------------------------
    def _reward_from_event(self, macro, ev):
        r = 0.0
        if ev.get("invalid"):
            return BattleRewardConfig.INVALID_ACTION
        if ev.get("wasted"):
            r += BattleRewardConfig.WASTED_TURN
        # damage capped at the enemy's REAL prior HP (already capped in driver)
        dmg = max(0, int(ev.get("our_damage_dealt", 0)))
        prior = max(1, int(ev.get("enemy_hp_before", 1)))
        r += BattleRewardConfig.DAMAGE_PER_HP * min(dmg, prior)
        if ev.get("enemy_ko"):
            r += BattleRewardConfig.ENEMY_KO      # once per enemy (driver-tracked)
        if ev.get("battle_won"):
            r += BattleRewardConfig.BATTLE_WIN
        if ev.get("own_faint"):
            r += BattleRewardConfig.OWN_FAINT
        if ev.get("wipe"):
            r += BattleRewardConfig.WIPE
        if ev.get("illegal_flee"):
            r += BattleRewardConfig.FLEE_TRAINER_ILLEGAL
        if ev.get("fled"):
            r += BattleRewardConfig.FLEE_WILD_OK
        return r

    # -- visible mirror sidecar (reward stream for battle_mirror_watch) ----
    def _publish_mirror(self, macro, reward, ev, outcome, *, reset=False):
        if not self._mirror_sidecar:
            return
        ev = ev or {}
        if not reset:
            self._ep_reward += float(reward)
            for name, amt in decompose_battle_reward(ev, reward):
                self._mirror_events.append([self._turn, f"{name}:{amt:+.4f}"])
            self._mirror_events = self._mirror_events[-40:]

        s = self._state or {}

        def _hp(mon):
            mon = mon or {}
            return [int(mon.get("cur_hp", 0) or 0), int(mon.get("max_hp", 0) or 0),
                    int(mon.get("level", 0) or 0)]

        payload = {
            "schema": "battle_mirror_v1",
            "updated": time.time(),
            "episode": self._episode_no,
            "turn": self._turn,
            "macro": macro or "",
            "episode_reward": round(self._ep_reward, 3),
            "step_reward": round(float(reward), 4),
            "outcome": outcome,
            "our": _hp(s.get("our_active")),
            "enemy": _hp(s.get("enemy_active")),
            "events": self._mirror_events,
        }
        try:
            tmp = self._mirror_sidecar + f".{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._mirror_sidecar)
        except OSError:
            pass

    def action_masks(self):
        """sb3-contrib MaskablePPO reads this every step. Same source of truth
        as the observation's ``action_mask``."""
        return np.asarray(self._legal_mask(), dtype=np.int8)

    def _catch_snapshot(self):
        """The executor-facing snapshot for :func:`battle_executor.catch_precheck`
        (spec §6). Built from the same state the reward reads."""
        s = self._state or {}
        return {
            "is_trainer": s.get("is_trainer"),
            "can_escape": s.get("can_escape"),
            "enemy_active": s.get("enemy_active") or {},
            "message_pending": bool(s.get("message_pending")),
            "animation_active": bool(s.get("animation_active")),
            "menu_ready": s.get("menu_ready", True),
        }

    # -- observation -----------------------------------------
    def _legal_mask(self):
        s = self._state or {}
        n = len(self._macros)
        supplied = s.get("action_mask")
        if isinstance(supplied, (list, tuple)) and len(supplied) == n:
            mask = [1 if x else 0 for x in supplied]
            return mask if any(mask) else [1] + [0] * (n - 1)
        our = s.get("our_active") or {}
        moves = our.get("moves") or []
        mask = [0] * n
        for i in range(4):
            mv = moves[i] if i < len(moves) else None
            if mv and int(mv.get("pp", 0) or 0) > 0 and mv.get("power") not in (None,):
                mask[i] = 1
        party = s.get("our_party") or []
        for i, mon in enumerate(party[:6]):
            if i != _active_index(s) and int(mon.get("cur_hp", 0) or 0) > 0:
                mask[4 + i] = 1
        if s.get("can_escape") and not s.get("is_trainer"):
            mask[self._macros.index(RUN_ACTION)] = 1
        # CATCH: v2 only, and only when the strategic objective requested it AND
        # the executor precheck passes (spec §4/§6). A v1 env has no CATCH slot.
        if self.schema == "v2" and CATCH_ACTION in self._macros \
                and self._objective is not None:
            from battle_executor import catch_precheck
            if catch_precheck(self._catch_snapshot(), self._objective)[0]:
                mask[self._macros.index(CATCH_ACTION)] = 1
        # spec ZIEL D (dormant): a CRITICAL shiny catch forces the catch —
        # masks RUN + likely-KO moves. Only fires when the planner set
        # catch_priority="critical" for a VERIFIED shiny (impossible today).
        if self.schema == "v2" and self._objective is not None \
                and self._objective.get("catch_priority") == "critical":
            from battle_executor import shiny_priority_mask
            mask = shiny_priority_mask(mask, {"enemy_active": s.get("enemy_active"),
                                              "player_active": s.get("our_active")},
                                       self._objective, macros=self._macros)
        if not any(mask):
            mask[0] = 1  # never a fully-empty mask
        return mask

    def _obs(self):
        s = self._state or {}
        vec = []

        def mon_block(mon):
            mon = mon or {}
            st = mon.get("stats") or {}
            known = 1.0 if mon.get("checksum_ok", True) else 0.0
            hp = _safe(mon.get("cur_hp"), 0) / max(1, _safe(mon.get("max_hp"), 1))
            types = mon.get("types") or []
            row = [
                (hp, known),
                (_safe(mon.get("level")) / 100.0, known),
                (_safe(mon.get("status")) / 255.0, 1.0 if "status" in mon else 0.0),
                ((types[0] if len(types) > 0 else -1) / bt.TYPE_COUNT,
                 1.0 if types else 0.0),
                ((types[1] if len(types) > 1 else (types[0] if types else -1)) / bt.TYPE_COUNT,
                 1.0 if types else 0.0),
                (_safe(st.get("attack")) / 255.0, 1.0 if st.get("attack") else 0.0),
                (_safe(st.get("defense")) / 255.0, 1.0 if st.get("defense") else 0.0),
                (_safe(st.get("sp_attack")) / 255.0, 1.0 if st.get("sp_attack") else 0.0),
                (_safe(st.get("sp_defense")) / 255.0, 1.0 if st.get("sp_defense") else 0.0),
                (_safe(st.get("speed")) / 255.0, 1.0 if st.get("speed") else 0.0),
            ]
            for v, k in row:
                vec.append(float(np.clip(v, -1, 1)))
                vec.append(float(k))

        our = s.get("our_active") or {}
        enemy = s.get("enemy_active") or {}
        mon_block(our)
        mon_block(enemy)

        moves = (our.get("moves") or [])
        a_types = our.get("types") or []
        for i in range(4):
            mv = moves[i] if i < len(moves) else None
            k = 1.0 if (mv and mv.get("mechanics_known")) else 0.0
            if mv:
                d = calc_damage(our, enemy, mv, ability_known=True,
                                in_battle_types_authoritative=True,
                                stat_stages_known=True, weather_known=True,
                                screens_known=True, item_known=True)
                acc = move_hits_probability(mv)
                fields = [
                    (_safe(mv.get("power")) / 200.0, k),
                    (_safe(mv.get("type"), -1) / bt.TYPE_COUNT, k),
                    (_safe(mv.get("pp")) / max(1, _safe(mv.get("max_pp"), 40)), k),
                    ((acc if acc is not None else 1.0), 1.0 if acc is not None else 0.0),
                    (_safe(mv.get("priority")) / 3.0, k),
                    ((d.get("stab") or 1.0) - 1.0, k),
                    ((d.get("effectiveness") if d.get("effectiveness") is not None else 1.0) / 4.0, k),
                    (1.0 if d.get("is_status") else 0.0, k),
                ]
            else:
                fields = [(0.0, 0.0)] * len(_MOVE_FIELDS)
            for v, kk in fields:
                vec.append(float(np.clip(v, -1, 1)))
                vec.append(float(kk))

        party = s.get("our_party") or []
        alive = sum(1 for m in party if int(m.get("cur_hp", 0) or 0) > 0)
        avg_hp = (np.mean([_safe(m.get("cur_hp")) / max(1, _safe(m.get("max_hp"), 1))
                           for m in party]) if party else 0.0)
        globals_ = [
            (1.0 if s.get("is_trainer") else 0.0, 1.0 if "is_trainer" in s else 0.0),
            (1.0 if s.get("can_escape") else 0.0, 1.0 if "can_escape" in s else 0.0),
            (alive / 6.0, 1.0),
            (float(avg_hp), 1.0),
            (_safe(s.get("weather"), 0) / 8.0, 1.0 if s.get("weather") is not None else 0.0),
            (min(1.0, self._turn / self.max_turns), 1.0),
        ]
        for v, k in globals_:
            vec.append(float(np.clip(v, -1, 1)))
            vec.append(float(k))

        # -- v2 catch block (spec §5): 12 fields, value+known each. A v1 env
        # never emits it, so a v1 champion sees exactly its 116-wide vector.
        if self.schema == "v2":
            for v, k in self._catch_obs_fields():
                vec.append(float(np.clip(v, -1, 1)))
                vec.append(float(k))

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != self._obs_dim:           # spec §4: NO pad / truncate
            raise RuntimeError(
                f"battle obs dim {arr.shape[0]} != {self._obs_dim} "
                f"({self._obs_schema}); an incompatible battle policy must be "
                "refused, not reshaped")
        return {"vec": arr,
                "action_mask": np.asarray(self._legal_mask(), dtype=np.int8)}

    def _catch_obs_fields(self):
        """The 12 v2 catch obs fields as ``[(value, known), ...]``. Unknown ->
        (0, 0); a boolean surely-known -> (0/1, 1). Same order in sim + emu."""
        s = self._state or {}
        obj = self._objective or {}
        enemy = s.get("enemy_active") or {}
        esid = enemy.get("species_id")
        tgt = int(obj.get("target_species_id", 0) or 0)
        is_catch = bool(obj.get("catch_requested"))
        usable = int(obj.get("usable_ball_count", 0) or 0)
        reserve = int(obj.get("reserved_ball_count", 0) or 0)

        prob = self._catch_probability_estimate()

        def kv(v, known):
            return (float(v) if known else 0.0, 1.0 if known else 0.0)

        return [
            kv(1.0 if is_catch else 0.0, True),
            kv((tgt / SPECIES_ID_MAX) if tgt else 0.0, tgt > 0),
            kv(1.0 if (esid and int(esid) == tgt and tgt) else 0.0, esid is not None),
            kv(1.0 if obj.get("is_new_species_this_run") else 0.0, "is_new_species_this_run" in obj),
            kv(min(1.0, usable / BALL_COUNT_CLIP), "usable_ball_count" in obj),
            kv(min(1.0, reserve / BALL_COUNT_CLIP), "reserved_ball_count" in obj),
            kv(1.0 if obj.get("party_has_space") else 0.0, "party_has_space" in obj),
            kv(1.0 if obj.get("pc_capture_supported") else 0.0, "pc_capture_supported" in obj),
            kv(1.0 if self._catch_valid_now() else 0.0, self._objective is not None),
            kv(prob if prob is not None else 0.0, prob is not None),
            kv(1.0 if self._last_catch_failed else 0.0, True),
            kv(min(1.0, self._catch_attempts / CATCH_ATTEMPTS_CLIP), True),
        ]

    def _catch_valid_now(self):
        if self._objective is None:
            return False
        from battle_executor import catch_precheck
        return catch_precheck(self._catch_snapshot(), self._objective)[0]

    def _catch_probability_estimate(self):
        """Gen-III p(catch) for the first usable ball, or ``None`` when any
        input (species catch rate, HP, status, ball) is not reliably known."""
        s = self._state or {}
        obj = self._objective or {}
        enemy = s.get("enemy_active") or {}
        if not obj.get("catch_requested"):
            return None
        try:
            import pokedb
            info = pokedb.species_info(int(enemy.get("species_id", 0) or 0))
            cr = info.get("catch_rate") if info else None
        except Exception:
            cr = None
        ch, mh = enemy.get("cur_hp"), enemy.get("max_hp")
        if cr is None or ch is None or mh is None:
            return None
        bonus = battle_catch.ball_bonus(
            int(s.get("best_ball_id", battle_catch.POKE_BALL)),
            target_types=enemy.get("types") or [],
            target_level=enemy.get("level"))
        if bonus is None:
            return None
        sb = battle_catch.status_bonus(enemy.get("status_name") or enemy.get("status"))
        return battle_catch.catch_probability(
            catch_rate=cr, cur_hp=ch, max_hp=mh, ball_bonus=bonus, status_bonus=sb)


def _active_index(state):
    party = state.get("our_party") or []
    act = state.get("our_active") or {}
    for i, m in enumerate(party):
        if m.get("slot") == act.get("slot"):
            return i
    return 0


def _is_switch_loop(recent):
    return len(recent) >= 3 and recent[-1] == recent[-3] and recent[-1] != recent[-2]


def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _mon(species_types, *, slot=0, level=15, cur_hp=45, max_hp=45,
         atk=30, df=30, spa=30, spd=30, spe=30, moves=()):
    return {"slot": slot, "checksum_ok": True, "level": level, "status": 0,
            "cur_hp": cur_hp, "max_hp": max_hp, "types": list(species_types),
            "stats": {"attack": atk, "defense": df, "sp_attack": spa,
                      "sp_defense": spd, "speed": spe},
            "stat_stages": {},
            "moves": [dict(m) for m in moves]}


def _mv(mtype, power, *, pp=15, acc=100, prio=0, is_status=False, mid=1):
    return {"id": mid, "type": mtype, "power": power, "pp": pp, "max_pp": pp,
            "accuracy": acc, "priority": prio, "is_status": is_status,
            "mechanics_known": True}


def default_scenario():
    return {
        "our_party": [
            _mon((bt.TYPE_WATER,), slot=0, level=16, cur_hp=48, max_hp=48, spa=40, spe=42,
                 moves=[_mv(bt.TYPE_WATER, 40, mid=1), _mv(bt.TYPE_NORMAL, 40, mid=2)]),
            _mon((bt.TYPE_GRASS,), slot=1, level=14, cur_hp=40, max_hp=40, spa=35,
                 moves=[_mv(bt.TYPE_GRASS, 45, mid=3)]),
        ],
        "enemy_party": [
            _mon((bt.TYPE_NORMAL,), level=6, cur_hp=22, max_hp=22, atk=18, spe=25,
                 moves=[_mv(bt.TYPE_NORMAL, 35, mid=9)]),
        ],
        "is_trainer": False, "can_escape": True,
    }


def catch_scenario(*, area="route1", species_id=19, catch_rate=255, level=3,
                   enemy_cur=18, enemy_max=18, status=0, ball_inventory=5,
                   ball_reserve=1, party_n=1, party_has_space=True,
                   pc_capture_supported=False, is_new_species_this_run=True,
                   catch_seed=1, is_trainer=False, objective_mode="catch"):
    """A reproducible catch-training / catch-eval scenario (spec §11). The
    scenario IS the authority for the objective — ``catch_planner`` reads these
    fields verbatim, the Battle-PPO only decides HOW."""
    enemy = _mon((bt.TYPE_NORMAL,), level=level, cur_hp=enemy_cur, max_hp=enemy_max,
                 atk=16, spe=22, moves=[_mv(bt.TYPE_NORMAL, 30, mid=9)])
    enemy["species_id"] = int(species_id)
    enemy["catch_rate"] = int(catch_rate)
    enemy["status"] = int(status)
    party = [_mon((bt.TYPE_WATER,), slot=i, level=18, cur_hp=52, max_hp=52,
                  spa=48, spe=50, moves=[_mv(bt.TYPE_WATER, 45, mid=1)])
             for i in range(max(1, party_n))]
    return {
        "area": area,
        "our_party": party,
        "enemy_party": [enemy],
        "is_trainer": bool(is_trainer),
        "can_escape": not is_trainer,
        "objective_mode": objective_mode,
        "target_species_id": int(species_id),
        "ball_inventory": int(ball_inventory),
        "ball_reserve": int(ball_reserve),
        "party_has_space": bool(party_has_space),
        "pc_capture_supported": bool(pc_capture_supported),
        "is_new_species_this_run": bool(is_new_species_this_run),
        "catch_seed": int(catch_seed),
    }


# --------------------------------------------------------------------------
# LIVE battle env — real isolated emulator + EmulatorBattleDriver (Phase 2).
# Requires captured battle-start savestates (scenario pool). Fail-closed when
# none exist: live battle training cannot run on synthetic scenarios.
# --------------------------------------------------------------------------
class LiveBattleDriver(BattleDriver):
    """Adapts :class:`twoby2.emulator_battle_driver.EmulatorBattleDriver` to the
    :class:`BattleDriver` interface ``BattleEnv`` expects."""

    def __init__(self, retro_env, main_battle_reader=None, *, schema="v2"):
        from twoby2.emulator_battle_driver import EmulatorBattleDriver
        self._env = retro_env
        self.schema = schema
        self._emu = EmulatorBattleDriver(retro_env, main_battle_reader=main_battle_reader,
                                         schema=schema)
        self._scenario = None
        self._encounter_id = None
        self._shiny_status = None
        self._shiny_outcome_recorded = False
        self._harvest_result = None
        self._harvest_calls = 0
        # MaskablePPO cannot sample a masked action, so the contradictory
        # "penalise as invalid but advance with MOVE_1" fallback is off: an
        # invalid action (only reachable from a non-masked policy) is a pure
        # penalty with no turn advance.
        self.advance_on_invalid = False

    # macro-event flags that end a battle episode as a diagnosed failure
    # (never a silent timeout).
    _DIAGNOSED_END = ("terminal_unknown", "menu_stall", "unreadable")

    def _translate(self, snap, done_outcome=None):
        pa_ = snap.get("player_active") or {}
        ea = snap.get("enemy_active") or {}
        party = snap.get("player_party") or []
        mask = self._emu.action_mask(snap)
        in_batt = snap.get("in_battle") is True
        # An in-battle state with nothing legal is unreadable RAM, not a real
        # dead end: end the episode as a diagnosed fault, never a silent
        # timeout, and never hand the policy an all-zero mask.
        if in_batt and not any(mask) and done_outcome is None:
            done_outcome = "unreadable"
        done = (not in_batt) or done_outcome is not None
        return {
            "our_active": pa_, "enemy_active": ea,
            "our_party": party, "enemy_party": [ea] if ea else [],
            "is_trainer": bool(snap.get("is_trainer")),
            "can_escape": bool(snap.get("can_escape")),
            "weather": snap.get("weather"),
            "turn": self._emu._turn,
            "done": bool(done),
            "outcome": done_outcome or snap.get("battle_outcome_label"),
            "action_mask": mask,
        }

    def reset(self, scenario):
        import gzip
        self._scenario = scenario or {}
        state_path = self._scenario.get("savestate_path")
        if not state_path or not os.path.isfile(state_path):
            raise RuntimeError("LiveBattleDriver needs a captured battle-start "
                               "savestate ('savestate_path'); none supplied. "
                               "Live battle training is fail-closed until the "
                               "scenario pool holds real battle scenarios.")
        raw = (gzip.open(state_path, "rb").read() if state_path.endswith(".gz")
               else open(state_path, "rb").read())
        # spec ZIEL A: the scenario savestate IS the deterministic anchor
        # restore. Reloading it here puts the emulator back on the exact grass
        # anchor tile before every episode — no anchor drift is possible.
        self._env.em.set_state(raw)
        self._emu.reset_battle_reader()   # stale gMain offset/result must not
        self._emu._turn = 0               # bleed in from the previous battle
        self._emu.diagnostics = []
        self._harvest_result = None
        self._shiny_outcome_recorded = False
        self._shiny_status = None
        self._encounter_id = None

        # spec ZIEL A: a "harvest" scenario is a GRASS OVERWORLD anchor, not a
        # battle start. Walk the fixed ±3-tile corridor until a wild battle
        # begins. The pendulum is not a PPO action and pays no reward.
        if self._scenario.get("harvest"):
            hr = self._run_harvester()
            self._harvest_result = hr
            if not hr.get("ok"):
                self._emu.diagnostics.append({"harvest_failed": dict(hr)})
                st = self._translate(self._emu.snapshot(), "reset_unreadable")
                st["done"] = True
                return st

        live = None
        for _ in range(180):
            # One reader sample per one emulated frame is required while the
            # relocatable gMain reader discovers its address. Calling
            # io.wait_frames() here would perform a second same-frame sample
            # and reset discovery because the VBlank delta is then zero.
            self._env.em.step()
            callback = getattr(self._env, "_pkmai_frame_callback", None)
            if callback is not None:
                callback()
            live = self._emu._mbr.read(self._env.get_ram(), frames=1)
            if live:
                break
        if not live:
            raise RuntimeError("loaded scenario savestate is not in a battle")

        # Captured scenarios may sit at "A wild ... appeared!".  That is a
        # valid in-battle state but not yet an actionable PPO state. Block-step
        # (sparse A) to a readable main menu - reuse the driver's helper instead
        # of a per-frame snapshot()/get_ram() loop (that was ~1400 full RAM
        # parses per reset and the real throughput bottleneck).
        ready = self._emu._settle_to_readable_menu(max_blocks=240)

        # Timing variation is neutral-only and therefore cannot choose a menu
        # item. It also feeds the real visible worker's frame mirror.
        self._emu.io.wait_frames(max(0, int(
            self._scenario.get("pre_action_wait", 0))))
        ready = self._emu.snapshot()
        if ready.get("menu_state") != "main" or not any(self._emu.action_mask(ready)):
            # Do NOT raise (that kills the SubprocVecEnv worker). End this
            # episode at turn 0 as a diagnosed reset fault - visible in the
            # counters, self-healing on the next reset.
            self._emu.diagnostics.append({"reset_unreadable": {
                "menu_state": ready.get("menu_state"),
                "player_active": bool(ready.get("player_active")),
                "enemy_active": bool(ready.get("enemy_active"))}})
            st = self._translate(ready, "reset_unreadable")
            st["done"] = True
            return st
        self._record_wild_encounter(ready)
        return self._translate(ready)

    def _record_wild_encounter(self, snap):
        """spec ZIEL C: count this isolated wild encounter EXACTLY ONCE. The
        id is derived from the (immutable) scenario + the enemy identity, so
        replaying the same captured battle savestate never re-counts it.
        Telemetry only — never blocks a battle, never a reward."""
        try:
            if snap.get("is_trainer") or snap.get("is_double"):
                return
            import hashlib
            import firered_ram as fr
            from twoby2.shiny_counters import ShinyCounters
            from twoby2 import shiny_ram

            class _E:
                def __init__(s, d): s._d = d
                def get_ram(s): return s._d
            enemy_party = fr.read_enemy_party(_E(self._env.get_ram()))
            enemy = snap.get("enemy_active") or {}
            # a replayed captured battle -> one stable encounter id (anti-farm).
            # a HARVEST encounter is genuinely fresh each episode -> fold in the
            # episode nonce so distinct wild battles each count once.
            _nonce = (self._harvest_calls if self._scenario.get("harvest") else "")
            eid = hashlib.sha1("|".join(str(x) for x in (
                self._scenario.get("scenario_sha256", self._scenario.get("id", "?")),
                enemy.get("species_id"), enemy.get("level"),
                shiny_ram.read_active_wild_pid(enemy_party) or "no_pid", _nonce,
            )).encode()).hexdigest()[:16]
            st = shiny_ram.shiny_status(is_trainer=False, enemy_party=enemy_party)
            self._encounter_id = eid
            self._shiny_status = st
            ShinyCounters("battle_fighter").record_encounter(
                eid, shiny_status=st["status"],
                species_id=enemy.get("species_id"), level=enemy.get("level"),
                area=self._scenario.get("area"),
                worker=self._scenario.get("battle_worker_ix"))
        except Exception:
            pass

    def _run_harvester(self):
        """Walk the grass corridor to a wild encounter (spec ZIEL A). Returns
        the :class:`HarvestResult` dict. Fail-closed: any corridor violation or
        limit ends the episode as a diagnosed reset fault."""
        import firered_ram as fr
        from twoby2.wild_encounter_harvester import WildEncounterHarvester

        env = self._env

        class _E:
            def get_ram(s):
                return env.get_ram()

        def _loc(_e):
            return fr.read_player_location(_E())

        def _batt(_e):
            return bool(self._emu._mbr.read(env.get_ram(), frames=1))

        from twoby2.wild_encounter_harvester import harvest_episode_seed
        # spec ZIEL C.5: a reproducible per-episode seed. The scenario carries
        # a stable base (rng_seed_id) + run seed; the sampler bumps
        # encounter_sequence each episode, so the SAME run replays the SAME
        # sequence while consecutive episodes get different jitter + direction.
        # pre_action_wait is folded in too (never ignored).
        sc = self._scenario
        base = int(sc.get("rng_seed_id", 0) or 0) ^ int(sc.get("pre_action_wait", 0) or 0)
        ep_seed = sc.get("episode_seed")
        if ep_seed is None:
            self._harvest_calls += 1
            ep_seed = harvest_episode_seed(base, int(sc.get("run_seed", 0) or 0),
                                           self._harvest_calls)
        h = WildEncounterHarvester(env, location_reader=_loc, battle_reader=_batt,
                                   episode_seed=int(ep_seed))
        return h.harvest()

    def state(self):
        return self._translate(self._emu.snapshot())

    def apply_macro(self, macro):
        snap0 = self._emu.snapshot()
        e0 = (snap0.get("enemy_active") or {}).get("cur_hp")
        snap1, ev = self._emu.apply_macro(macro, dry_run=False)
        outcome = None
        if ev.get("battle_won"):
            outcome = "win"
        elif ev.get("wipe"):
            outcome = "wipe"
        elif ev.get("fled"):
            outcome = "fled"
        elif ev.get("terminal_unknown"):
            outcome = "terminal_unknown"
        elif ev.get("menu_stall"):
            outcome = "menu_stall"
        elif ev.get("unreadable"):
            outcome = "unreadable"
        ev.setdefault("enemy_hp_before", e0)
        state = self._translate(snap1, outcome)
        if outcome in self._DIAGNOSED_END:
            state["done"] = True   # end the episode as a diagnosed failure
        if (outcome or state.get("done")):
            self._record_wild_outcome(outcome or state.get("outcome"), ev)
        return state, ev

    def _record_wild_outcome(self, outcome, ev):
        """spec ZIEL C.3: on a terminal battle outcome, if THIS encounter was a
        RAM-verified shiny, record EXACTLY ONE terminal outcome. While
        SHINY_RAM_VERIFIED is False ``self._shiny_status`` is always "unknown",
        so nothing is ever counted as a shiny (double fail-closed: the counter
        also refuses an id that is not in verified_shiny_encounters)."""
        try:
            if self._shiny_outcome_recorded:
                return
            st = (self._shiny_status or {}).get("status")
            if st != "verified_shiny" or not self._encounter_id:
                return
            from twoby2.shiny_counters import ShinyCounters
            if ev.get("catch_success"):
                mapped = "caught"
            else:
                mapped = {"win": "ko", "wipe": "wipe", "fled": "fled"}.get(
                    outcome, "unresolved")
            ShinyCounters("battle_fighter").record_shiny_outcome(
                self._encounter_id, mapped)
            self._shiny_outcome_recorded = True
        except Exception:
            pass


def make_live_battle_env(*, scenario_sampler=None, seed=0, max_turns=20,
                         mirror_path=None, schema="v2"):
    """Real isolated emulator + :class:`LiveBattleDriver`. Raises if the
    ``battle_env`` feature gate is off. ``schema`` picks the obs/action space
    ("v1" = combat only 116/11, "v2" = adds CATCH 140/12)."""
    from twoby2 import feature_enabled
    if not feature_enabled("battle_env"):
        raise RuntimeError("make_live_battle_env: FEATURES['battle_env'] is OFF")
    import stable_retro as retro
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retro.data.Integrations.add_custom_path(os.path.join(root, "local", "custom_integrations"))
    retro_env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                           inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
    retro_env.reset()
    if mirror_path:
        import cv2
        counter = {"n": 0}

        def publish_frame():
            counter["n"] += 1
            if counter["n"] % 6:
                return
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(retro_env.get_screen(), cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return
            os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
            tmp = mirror_path + f".{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(encoded.tobytes())
            os.replace(tmp, mirror_path)
        retro_env._pkmai_frame_callback = publish_frame
    sidecar = (os.path.splitext(mirror_path)[0] + ".json") if mirror_path else None
    env = BattleEnv(LiveBattleDriver(retro_env, schema=schema),
                    scenario_sampler=scenario_sampler,
                    max_turns=max_turns, mirror_sidecar=sidecar, schema=schema)
    env._retro_env = retro_env
    return env
