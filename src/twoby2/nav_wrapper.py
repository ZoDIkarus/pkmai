"""NavigationBattleWrapper — the hard Navigation/Battle boundary (Phase 3).

The navigation policy picks exactly one overworld action per ``step``. If that
action triggers a battle, the wrapper does NOT feed foreign battle actions into
the navigation transition. Instead a *pinned* battle policy plays the whole
battle as an internal sub-episode; navigation then resumes from the real
resulting overworld state and receives only a coarse strategic consequence.

Semi-MDP handling:
  * the battle sub-episode consumes N emulator steps — those are NOT navigation
    steps and never increment the navigation step counter;
  * navigation's reward for the transition = its own overworld reward for the
    triggering action + a coarse strategic battle term
    (:func:`twoby2.battle_summary.navigation_battle_reward`), discounted for the
    battle duration if a discount is configured;
  * no battle move / KO / per-turn reward ever reaches navigation;
  * ``terminated`` is set if the battle ended in a wipe/blackout that the env
    treats as terminal; otherwise the episode continues from the post-battle
    overworld observation;
  * the observation returned after a battle is the REAL post-battle overworld
    observation (including a blackout respawn), never a stale pre-battle frame.
"""
from __future__ import annotations

import gymnasium as gym

from twoby2.battle_summary import (
    summarize_battle, assert_navigation_safe, navigation_battle_reward,
)
from twoby2.isolation import NavigationImmutableDuringBattle


class BattleDriver:
    """Interface the wrapper needs. A real implementation drives the emulator
    with the pinned battle champion + macro executor; the test double runs a
    deterministic simulator. It must NEVER touch navigation state."""

    def in_battle(self, env):
        raise NotImplementedError

    def play_battle(self, env):
        """Play the current battle to its end. Return a raw result dict for
        :func:`summarize_battle` PLUS ``{"duration_steps": int,
        "post_battle_obs": <obs>, "terminated": bool}``. Must not advance any
        navigation counter."""
        raise NotImplementedError


class NavigationBattleWrapper(gym.Wrapper):
    def __init__(self, env, battle_driver, *, battle_step_discount=1.0,
                 nav_step_counter_attr="route_steps"):
        super().__init__(env)
        self.driver = battle_driver
        self.battle_step_discount = float(battle_step_discount)
        self._nav_step_attr = nav_step_counter_attr
        self.last_battle_summary = None
        self.battles_played = 0
        self.total_battle_steps = 0

    def reset(self, **kw):
        self.last_battle_summary = None
        return self.env.reset(**kw)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        if not terminated and not truncated and self.driver.in_battle(self.env):
            nav_before = self._nav_steps()
            with NavigationImmutableDuringBattle(_NavCounterView(self.env, self._nav_step_attr)):
                raw = self.driver.play_battle(self.env)
            # the wrapper's own guard: navigation step counter unmoved
            assert self._nav_steps() == nav_before, (
                "battle sub-episode advanced the navigation step counter")

            duration = int(raw.get("duration_steps", 0) or 0)
            self.total_battle_steps += duration
            self.battles_played += 1

            summary = summarize_battle(raw)
            assert_navigation_safe(summary)
            summary["duration_steps"] = duration
            self.last_battle_summary = summary

            # wild-win decay (plan I): first couple of wild wins on a map per
            # training run pay a small strategic bonus, then 0. The counter is
            # in the persistent shaping state, so it survives episode resets /
            # wipes / returning to a map - only a fresh run clears it.
            _sk = {}
            if (summary.get("outcome") == "win"
                    and summary.get("battle_kind") != "trainer"):
                try:
                    _env0 = self.env.unwrapped
                    _sh = _env0._load_nav_agent()
                    _loc = _env0.cached_loc or {}
                    _b = int(_loc.get("map_bank", 0))
                    _m = int(_loc.get("map_id", 0))
                    _sk["wild_win_index"] = _sh.wild_wins_for(_b, _m)
                    _sh.add_wild_win(_b, _m)
                except Exception:
                    pass
            strat = navigation_battle_reward(summary, **_sk)
            if self.battle_step_discount != 1.0 and duration > 0:
                strat *= self.battle_step_discount ** duration
            reward = float(reward) + float(strat)

            obs = raw.get("post_battle_obs", obs)   # REAL post-battle overworld obs
            terminated = bool(terminated or raw.get("terminated", False))

            info["battle_summary"] = summary
            info["battle_duration_steps"] = duration
            info["semi_mdp"] = {
                "kind": "battle_subepisode",
                "duration_steps": duration,
                "discount_applied": self.battle_step_discount != 1.0,
                "strategic_reward": strat,
            }
            # FULL battle telemetry (plan K): stash the last finished battle on
            # the env so its inst_XX.json picks it up on the next step. Strategic
            # consequence only - never move/damage/KO micro-reward.
            try:
                _e0 = self.env.unwrapped
                _e0._last_nav_battle = {
                    "outcome": summary.get("outcome"),
                    "battle_kind": summary.get("battle_kind"),
                    "hp_fraction_lost": summary.get("party_hp_fraction_lost"),
                    "pp_spent": summary.get("pp_spent"),
                    "turns": summary.get("turns"),
                    "duration_steps": duration,
                    "wild_win_index": _sk.get("wild_win_index"),
                    "strategic_reward": round(float(strat), 4),
                    "nav_step": int(self._nav_steps()),
                    # Catch-v2 (spec §3): the validated catch summary the env's
                    # species_caught_first block gates the +0.5 on. A raw catch
                    # (no planner request) leaves catch_requested False -> +0.
                    "objective_mode": summary.get("objective_mode"),
                    "catch_requested": bool(summary.get("catch_requested")),
                    "catch_success": bool(summary.get("catch_success")),
                    "caught_species_id": summary.get("caught_species_id"),
                    "target_species_id": summary.get("target_species_id"),
                    "is_new_species_this_run": bool(summary.get("is_new_species_this_run")),
                }
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

    def _nav_steps(self):
        return int(getattr(self.env.unwrapped, self._nav_step_attr, 0) or 0)


class _NavCounterView:
    """Adapts a real nav env to the small surface
    :class:`NavigationImmutableDuringBattle` needs (counters snapshot +
    rollout buffer length)."""

    def __init__(self, env, attr):
        self._env = env
        self._attr = attr

        class _C:
            def snapshot(_self):
                return {"nav_steps": int(getattr(env.unwrapped, attr, 0) or 0)}
        self.counters = _C()
        self.rollout_buffer = []   # the wrapper never appends to nav rollout here
