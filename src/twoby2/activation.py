"""The single chokepoint for *live* battle activation.

Every seam that could put a battle macro on the real game, wire the router into
the live trainer, or mark a Battle-PPO as the live champion must call
:func:`assert_live_battle_allowed` first. It stays hard-blocked until the
Phase-2 battle-menu RAM work lands and the feature gate is deliberately opened.
"""
from __future__ import annotations

from twoby2 import feature_enabled

# The Phase-2 hard blockers, verbatim, so a report / dashboard can list them.
PHASE2_RAM_BLOCKERS = (
    "gBattleMons",
    "gBattlerPartyIndexes",
    "gActionSelectionCursor",
    "gMoveSelectionCursor",
    "battle_menu_state",
)


class LiveActivationBlocked(RuntimeError):
    pass


def _ready_fns():
    from battle_ram import (battle_snapshot_ready_for_execution,
                            battle_snapshot_ready_for_action)
    return battle_snapshot_ready_for_execution, battle_snapshot_ready_for_action


def snapshot_execution_ready(snapshot):
    full, _ = _ready_fns()
    ready, missing = full(snapshot)
    return bool(ready), list(missing)


def snapshot_ready_for_action(snapshot, action):
    _, per_action = _ready_fns()
    ready, missing = per_action(snapshot, action)
    return bool(ready), list(missing)


def live_battle_allowed(snapshot=None, *, action=None):
    """``(allowed: bool, reasons: list)``. Allowed only when the executor gate
    feature is on AND the concrete snapshot is execution-ready (for the given
    action, if one is named). No snapshot -> never allowed."""
    reasons = []
    if not feature_enabled("battle_executor_live"):
        reasons.append("feature gate 'battle_executor_live' is off")
    if snapshot is None:
        reasons.append("no execution-ready battle snapshot provided")
        return False, reasons
    if action is not None:
        ready, missing = snapshot_ready_for_action(snapshot, action)
        label = f"snapshot not ready for {action}"
    else:
        ready, missing = snapshot_execution_ready(snapshot)
        label = "snapshot not execution-ready"
    if not ready:
        reasons.append(f"{label}; missing: {missing}")
    return (not reasons), reasons


def assert_live_battle_allowed(snapshot=None, *, action=None):
    allowed, reasons = live_battle_allowed(snapshot, action=action)
    if not allowed:
        raise LiveActivationBlocked("; ".join(reasons))
    return True


def assert_not_live_router():
    """Guard for wiring the router into the live trainer / watcher."""
    if feature_enabled("battle_router_live"):
        return True   # deliberately enabled elsewhere; caller takes over
    raise LiveActivationBlocked(
        "battle_router_live gate is off — router must not touch the live path")


def battle_ppo_may_be_live_champion(*, eval_passed, executor_verified,
                                    migration_done):
    """A Battle-PPO can only be marked the *live* champion when it has passed
    its eval suite, the executor is verified against the real ROM, AND the
    one-time migration has run — and the router-live gate is open."""
    return bool(eval_passed and executor_verified and migration_done
                and feature_enabled("battle_router_live"))


def live_battle_possible(sample_snapshot=None):
    """Dynamic — NOT hard-coded. True only if the executor gate is open and a
    real execution-ready snapshot exists. In Phase 1 this is False because the
    menu-cursor RAM is unverified, so no snapshot can be execution-ready."""
    if not feature_enabled("battle_executor_live"):
        return False
    if sample_snapshot is None:
        return False
    ok, _ = snapshot_execution_ready(sample_snapshot)
    return ok


def status_report(sample_snapshot=None):
    """Machine-readable activation status. Reflects the real feature gates and,
    if a snapshot is supplied, its real readiness — nothing hard-coded."""
    gates = {k: feature_enabled(k) for k in (
        "battle_env", "battle_executor_live", "battle_router_live",
        "nav_battle_wrapper", "adaptive_nav_horizon",
        "watcher_battle_champion", "runtime_migration")}
    snap_ready, snap_missing = (snapshot_execution_ready(sample_snapshot)
                                if sample_snapshot is not None else (False, ["no snapshot"]))
    return {
        "features": gates,
        "phase2_ram_blockers": list(PHASE2_RAM_BLOCKERS),
        "snapshot_execution_ready": snap_ready,
        "snapshot_missing": snap_missing,
        "live_battle_possible": live_battle_possible(sample_snapshot),
    }
