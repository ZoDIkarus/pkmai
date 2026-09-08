"""Strategic catch planner (spec §1/§2).

Decides, ONCE at battle start, whether a wild Pokemon should be caught. The
Battle-PPO never creates or edits this objective — it only reads it from the
observation and decides *how* to execute it.

  * FULL / watcher: :func:`plan_battle_objective` (central, deterministic).
  * isolated Battle-PPO training: the scenario carries the objective verbatim
    (:func:`objective_from_scenario`).

``catch_requested`` is only ``True`` when EVERY condition in the spec holds;
any unknown input (ball count, party space, species, battle kind) forces
``objective_mode="combat"`` with a diagnostic reason.
"""
from __future__ import annotations

import os

import battle_catch

OBJECTIVE_SCHEMA = "battle_objective_v1"

# spec §2: catch objectives only on config-approved, canary-passed wild maps.
# Route 1 may be the first entry AFTER its canary passes. No permanent Route-1
# special-casing — this is a plain allowlist, extended later.
_DEFAULT_ALLOWLIST = ("route1",)


def catch_map_allowlist():
    env = os.environ.get("PKMAI_CATCH_MAP_ALLOWLIST")
    if env:
        return tuple(a.strip() for a in env.split(",") if a.strip())
    return _DEFAULT_ALLOWLIST


# how many balls of the best available type to keep in reserve (never spend)
DEFAULT_BALL_RESERVE = 1
# don't request a catch when the estimated wipe risk is above this
DEFAULT_MAX_WIPE_RISK = 0.35


class BattleObjective(dict):
    """Immutable-by-convention battle order (spec §2)."""

    @property
    def is_catch(self):
        return self.get("objective_mode") == "catch" and self.get("catch_requested")

    def frozen(self):
        return BattleObjective({k: v for k, v in self.items()})


def _combat(reason, **extra):
    o = BattleObjective(
        objective_mode="combat", catch_requested=False,
        target_species_id=0, catch_reason=reason,
        is_new_species_this_run=False, usable_ball_count=0,
        reserved_ball_count=0, party_has_space=False,
        pc_capture_supported=False, schema=OBJECTIVE_SCHEMA)
    o.update(extra)
    return o


def objective_from_scenario(scenario):
    """Isolated Battle-PPO training: the scenario IS the authority."""
    s = scenario or {}
    mode = s.get("objective_mode")
    if mode not in ("combat", "catch"):
        return _combat("scenario_no_objective_mode")
    # spec §2: a trainer / non-escapable battle is ALWAYS combat, whatever the
    # scenario says — CATCH is never a legal order there.
    if s.get("is_trainer") or s.get("can_escape") is False:
        return _combat("trainer_or_forced_battle",
                       target_species_id=int(s.get("target_species_id", 0) or 0))
    if mode == "combat":
        return _combat("scenario_combat", target_species_id=int(s.get("target_species_id", 0) or 0))
    balls = int(s.get("ball_inventory", 0) or 0)
    reserve = int(s.get("ball_reserve", DEFAULT_BALL_RESERVE) or 0)
    return BattleObjective(
        objective_mode="catch",
        catch_requested=bool(balls - reserve > 0),
        target_species_id=int(s.get("target_species_id", 0) or 0),
        catch_reason="scenario_catch" if balls - reserve > 0 else "scenario_catch_no_ball",
        is_new_species_this_run=bool(s.get("is_new_species_this_run", True)),
        usable_ball_count=max(0, balls - reserve),
        reserved_ball_count=reserve,
        party_has_space=bool(s.get("party_has_space", True)),
        pc_capture_supported=bool(s.get("pc_capture_supported", False)),
        catch_seed=int(s.get("catch_seed", 0) or 0),
        schema=OBJECTIVE_SCHEMA)


def plan_battle_objective(ctx):
    """Central FULL/watcher planner. ``ctx`` (all fields optional; missing =>
    unknown => combat):
        area, is_trainer, battle_confirmed_wild,
        enemy_species_id, enemy_species_known,
        caught_species_this_run (set/list), story_target_species (set),
        usable_ball_count, ball_reserve, party_has_space, pc_capture_supported,
        post_wipe_recovery, wipe_risk, catch_execution_ready,
        enemy_types, missing_type_coverage (set of type names)
    """
    c = ctx or {}
    area = str(c.get("area") or "unknown")

    if c.get("is_trainer"):
        return _combat("trainer_battle")
    if not c.get("battle_confirmed_wild"):
        return _combat("wild_not_confirmed")

    # spec ZIEL D: a VERIFIED wild shiny forces a critical catch order — over
    # the map allowlist, the "species not wanted" check and the reserve-ball
    # rule. It does NOT bypass "no ball at all", "party full + no PC" or
    # "catch not execution-ready": those still produce a diagnosed combat
    # objective, never a fake catch. ``shiny_status`` only ever reaches
    # "verified_shiny" once twoby2.shiny_ram.SHINY_RAM_VERIFIED is True, so
    # this branch is dormant until the shiny RAM probe passes.
    _shiny = c.get("shiny_status") == "verified_shiny"

    if c.get("post_wipe_recovery") and not _shiny:
        return _combat("post_wipe_recovery")
    if area not in catch_map_allowlist() and not _shiny:
        return _combat(f"map_not_allowlisted:{area}")
    if not c.get("catch_execution_ready"):
        return _combat("catch_not_execution_ready"
                       + (":shiny_blocked" if _shiny else ""))

    sid = c.get("enemy_species_id")
    if not c.get("enemy_species_known") or not sid:
        return _combat("enemy_species_unknown"
                       + (":shiny_blocked" if _shiny else ""))
    sid = int(sid)

    if _shiny:
        balls = c.get("usable_ball_count")
        reserve = int(c.get("ball_reserve", DEFAULT_BALL_RESERVE) or 0)
        if balls is None:
            return _combat("ball_count_unknown:shiny_blocked", target_species_id=sid)
        # a shiny may spend the reserve ball, but there must be >= 1 ball total
        if int(balls) <= 0:
            return _combat("no_ball_for_shiny", target_species_id=sid)
        has_space = c.get("party_has_space")
        pc_ok = bool(c.get("pc_capture_supported"))
        if has_space is None:
            return _combat("party_space_unknown:shiny_blocked", target_species_id=sid)
        if not has_space and not pc_ok:
            return _combat("party_full_no_pc:shiny_blocked", target_species_id=sid)
        return BattleObjective(
            objective_mode="catch", catch_requested=True,
            target_species_id=sid, catch_reason="verified_shiny",
            catch_priority="critical",
            is_new_species_this_run=bool(
                sid not in set(int(x) for x in (c.get("caught_species_this_run") or []))),
            usable_ball_count=max(1, int(balls)),   # reserve override for a shiny
            reserved_ball_count=0,
            party_has_space=bool(has_space), pc_capture_supported=pc_ok,
            schema=OBJECTIVE_SCHEMA)

    balls = c.get("usable_ball_count")
    reserve = int(c.get("ball_reserve", DEFAULT_BALL_RESERVE) or 0)
    if balls is None:
        return _combat("ball_count_unknown")
    usable_above_reserve = int(balls)          # caller already subtracted reserve? No:
    usable_above_reserve = max(0, int(balls) - reserve)
    if usable_above_reserve <= 0:
        return _combat("no_ball_above_reserve", usable_ball_count=max(0, int(balls) - reserve),
                       reserved_ball_count=reserve, target_species_id=sid)

    has_space = c.get("party_has_space")
    pc_ok = bool(c.get("pc_capture_supported"))
    if has_space is None:
        return _combat("party_space_unknown", target_species_id=sid)
    if not has_space and not pc_ok:
        return _combat("party_full_no_pc", target_species_id=sid)

    wr = c.get("wipe_risk")
    if wr is not None and float(wr) > DEFAULT_MAX_WIPE_RISK:
        return _combat(f"wipe_risk_{float(wr):.2f}", target_species_id=sid)

    caught = set(int(x) for x in (c.get("caught_species_this_run") or []))
    story = set(int(x) for x in (c.get("story_target_species") or []))
    is_new = sid not in caught
    wanted_reason = None
    if is_new:
        wanted_reason = "new_species"
    elif sid in story:
        wanted_reason = "story_target"
    elif c.get("missing_type_coverage") and (
            set(str(t).lower() for t in (c.get("enemy_types") or []))
            & set(str(t).lower() for t in c["missing_type_coverage"])):
        wanted_reason = "type_coverage"
    if wanted_reason is None:
        return _combat("species_not_wanted", target_species_id=sid,
                       is_new_species_this_run=False)

    return BattleObjective(
        objective_mode="catch", catch_requested=True,
        target_species_id=sid, catch_reason=wanted_reason,
        is_new_species_this_run=bool(is_new),
        usable_ball_count=usable_above_reserve,
        reserved_ball_count=reserve,
        party_has_space=bool(has_space), pc_capture_supported=pc_ok,
        schema=OBJECTIVE_SCHEMA)
