"""Reset matrix for the 2×2 system (logic; the CLI wrappers live in tools/).

Three scopes, each strictly bounded:

  navigation : navigation learner/candidate + navigation stats ONLY
  battle     : battle learner/candidate + battle stats (+ scenario pool only
               with an explicit extra flag)
  full       : both model systems — requires an extra explicit confirmation

Savestates, curriculum state and exploration memory are ALWAYS preserved unless
an additional, separate opt-in is given (never by default, never by full).
Nothing here deletes a champion; a reset copies champion -> learner.
"""
from __future__ import annotations

import os
import shutil

from twoby2.protected_assets import (assert_not_protected, default_registry,
                                     ProtectedPathError, _op_kind)

# Everything a reset is allowed to touch, by scope.
NAV_MODEL_FILES = ("navigation_learner.zip", "navigation_candidate.zip",
                   "navigation_resume.zip")
NAV_STAT_FILES = ("navigation_stats.json", "nav_horizon.json")
BATTLE_MODEL_FILES = ("battle_learner.zip", "battle_candidate.zip",
                      "battle_resume.zip")
BATTLE_STAT_FILES = ("battle_stats.json",)

# Never touched by ANY reset unless its own dedicated opt-in is passed.
ALWAYS_PRESERVED = (
    "savestates", "curriculum_states", "curriculum_shared", "curriculum_v20",
    "exploration_memory", "scenario_pool", "brain_backups",
    "protected:canonical_post_parcel_master",
    "protected:route1_manual_battle_seed",
)


class ResetPlan(dict):
    @property
    def files_touched(self):
        return list(self.get("touch", []))


def _exists(d, name):
    return os.path.exists(os.path.join(d, name))


# a fresh-obs-schema reset also archives + deletes the champion / version /
# score / shaping-state, because the new NAV_DIM makes every old nav .zip
# structurally incompatible. Savestates / curriculum / exploration memory /
# the directed movement graph are still preserved.
NAV_FRESH_SCHEMA_FILES = (
    "navigation_champion.zip", "navigation_latest.zip",
    "navigation_learner.zip", "navigation_candidate.zip", "navigation_resume.zip",
)
NAV_FRESH_SCHEMA_STATS = (
    "champion_score.json", "model_version.json", "nav_horizon.json",
    "navigation_stats.json", "trainer_status.json", "nav_global.json",
    "shaping_state.json",   # legacy single-file name, delete if present
)
# the per-agent shaping dir is cleared wholesale on a fresh-schema reset
NAV_FRESH_SCHEMA_DIRS = ("shaping",)


def plan_navigation_reset(*, nav_ckpt_dir, nav_stats_dir,
                          champion_name="navigation_champion.zip",
                          fresh_obs_schema=False):
    champ = os.path.join(nav_ckpt_dir, champion_name)
    touch = []
    if fresh_obs_schema:
        # a schema bump: no champion to copy from - start a brand-new PPO
        for f in NAV_FRESH_SCHEMA_FILES:
            touch.append(("delete", os.path.join(nav_ckpt_dir, f)))
        for f in NAV_FRESH_SCHEMA_STATS:
            touch.append(("delete", os.path.join(nav_stats_dir, f)))
        for sub in NAV_FRESH_SCHEMA_DIRS:
            d = os.path.join(nav_stats_dir, sub)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    touch.append(("delete", os.path.join(d, f)))
        note = ("fresh navigation PPO (obs schema bump); champion/learner/score/"
                "horizon + per-agent shaping deleted; savestates + curriculum + "
                "exploration memory + directed movement graph preserved")
    else:
        for f in NAV_MODEL_FILES:
            touch.append(("copy_from_champion" if _exists(nav_ckpt_dir, champion_name)
                          else "delete", os.path.join(nav_ckpt_dir, f)))
        for f in NAV_STAT_FILES:
            touch.append(("reset", os.path.join(nav_stats_dir, f)))
        note = "navigation learner <- navigation champion; battle untouched"
    return ResetPlan(
        scope="navigation", champion=champ, touch=[p for _, p in touch],
        actions=touch, preserves=list(ALWAYS_PRESERVED) + list(BATTLE_MODEL_FILES)
        + ["movement_graph_v1.json"],
        note=note)


def plan_battle_reset(*, battle_ckpt_dir, battle_stats_dir,
                      champion_name="battle_champion.zip", wipe_scenarios=False):
    touch = []
    for f in BATTLE_MODEL_FILES:
        touch.append(("copy_from_champion" if _exists(battle_ckpt_dir, champion_name)
                      else "reset_to_rule_fallback", os.path.join(battle_ckpt_dir, f)))
    for f in BATTLE_STAT_FILES:
        touch.append(("reset", os.path.join(battle_stats_dir, f)))
    if wipe_scenarios:
        touch.append(("delete", os.path.join(battle_stats_dir, "scenario_pool.json")))
    preserves = list(ALWAYS_PRESERVED) + list(NAV_MODEL_FILES)
    if not wipe_scenarios:
        preserves.append("scenario_pool.json")
    return ResetPlan(
        scope="battle", touch=[p for _, p in touch], actions=touch,
        preserves=preserves,
        note="battle learner <- battle champion (or rule fallback); "
             + ("scenario pool WIPED" if wipe_scenarios else "scenario pool kept")
             + "; navigation untouched")


def plan_full_reset(*, nav_ckpt_dir, nav_stats_dir, battle_ckpt_dir,
                    battle_stats_dir, confirm_full=False):
    if not confirm_full:
        return ResetPlan(scope="full", touch=[], actions=[],
                         refused=True,
                         note="full reset needs confirm_full=True (extra "
                              "explicit confirmation)")
    nav = plan_navigation_reset(nav_ckpt_dir=nav_ckpt_dir, nav_stats_dir=nav_stats_dir)
    bat = plan_battle_reset(battle_ckpt_dir=battle_ckpt_dir,
                            battle_stats_dir=battle_stats_dir, wipe_scenarios=False)
    return ResetPlan(
        scope="full", touch=nav["touch"] + bat["touch"],
        actions=nav["actions"] + bat["actions"],
        preserves=list(ALWAYS_PRESERVED),
        note="both model systems reset; savestates/curriculum/exploration/"
             "scenarios all preserved")


def _guard_protected(actions, *, registry=None):
    """Central check: no reset/migration/cleanup action may target a protected
    asset (master or Route-1 seed), through any alias / symlink / ``..`` form.
    Raises :class:`ProtectedPathError`."""
    reg = registry if registry is not None else default_registry()
    for action, path in actions:
        assert_not_protected(path, op=_op_kind(action), registry=reg)
    return True


def apply_plan(plan, *, dry_run=True, registry=None):
    """Execute a plan. ``dry_run=True`` (default) only reports. A real run
    copies champion->learner / resets stats to an empty JSON / deletes only the
    files the plan explicitly lists. Never recursive, never a directory.

    Every action is checked against the protected-asset registry FIRST — the
    master and the Route-1 seed can never be a target, even via ``--apply`` or
    a full reset."""
    if plan.get("refused"):
        return {"applied": False, "dry_run": dry_run, "reason": plan["note"]}
    _guard_protected(plan["actions"], registry=registry)   # raises if protected
    done = []
    for action, path in plan["actions"]:
        if dry_run:
            done.append((action, path, "planned"))
            continue
        champ = plan.get("champion")
        if action == "copy_from_champion" and not champ:
            # A combined full plan contains two independent champion sources.
            # Resolve the source from the explicitly-scoped destination rather
            # than silently skipping the copy.
            base = os.path.dirname(path)
            if os.path.basename(path).startswith("navigation_"):
                champ = os.path.join(base, "navigation_champion.zip")
            elif os.path.basename(path).startswith("battle_"):
                champ = os.path.join(base, "battle_champion.zip")
        if action == "copy_from_champion" and champ and os.path.exists(champ):
            _atomic_copy(champ, path)
            done.append((action, path, "copied"))
        elif action == "reset":
            _atomic_write_text(path, "{}")
            done.append((action, path, "emptied"))
        elif action in ("delete", "reset_to_rule_fallback") and os.path.isfile(path):
            # no PPO champion yet -> there is no model to copy from; the learner
            # must start fresh, so the stale .zip is removed (BattleTrainer.setup
            # then builds a new PPO). The caller backs it up first.
            os.unlink(path)
            done.append((action, path, "deleted"))
        else:
            done.append((action, path, "skipped"))
    return {"applied": not dry_run, "dry_run": dry_run, "results": done,
            "preserves": plan.get("preserves", [])}


def _atomic_copy(src, dst):
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(dst)}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_write_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
