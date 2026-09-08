"""Worker / emulator budget for the final 2×2 system. Pure constants + validation.

These describe *process orchestration* for the split trainer. They are NOT PPO
hyper-parameters and have nothing to do with ``train.PPO_N_STEPS`` (rollout
length) — that stays exactly as it is.

Final architecture (no anchor / FRONTIER / BRIDGE / RETENTION / FIGHTER roles):
all 40 navigation workers start at the *identical canonical game start* and
train the one navigation learner. The 80/20 split is only the episode-length
horizon probe (see :mod:`twoby2.horizon`); every worker still starts at the
same state.
"""
from __future__ import annotations

# --- navigation -----------------------------------------------------------
NAV_WORKERS = 40
NAV_BEGINNING_WORKERS = 40   # every nav worker is a canonical-start "beginning" run
NAV_ANCHOR_WORKERS = 0       # the final architecture has NO anchor workers

# --- battle --------------------------------------------------------------
BATTLE_WORKERS = 9            # fixed; never grows as more areas unlock
BATTLE_HEADLESS_WORKERS = 8   # worker 9 publishes frames to the mirror window
BATTLE_VISIBLE_WORKERS = 1    # a real member of the 9-worker learner
#   BATTLE_HEADLESS_WORKERS + BATTLE_VISIBLE_WORKERS == BATTLE_WORKERS
#   the visible worker is the 9th worker, NOT an extra emulator.

# The inference-only FULL watcher owns one emulator but contributes no rollout.
FULL_WATCHER_EMULATORS = 1

# --- global cap ---------------------------------------------------------
MAX_TOTAL_EMULATORS = 50

# The single canonical navigation start. All 40 nav workers load this and only
# this. Resolved via the protected-asset registry (project-relative,
# symlink / alias / ``..`` resistant) — not a bare basename. ``stage_*`` /
# ``stage_frontier_*`` savestates stay DATA (diagnostics, battle-scenario
# seeding, regression, pre-migration recovery), never a navigation start.
CANONICAL_NAV_START = "StartGame.state"   # kept for readability / old tests

NAV_START_KINDS = ("beginning",)


def canonical_nav_start_path(*, registry=None):
    """The absolute, verified path to the canonical navigation start, resolved
    through :mod:`twoby2.protected_assets`. Raises if the master is missing or
    its sha256 does not match the recorded value."""
    from twoby2.protected_assets import (default_registry, canonical_path,
                                         MASTER_REL, verify_master)
    reg = registry if registry is not None else default_registry()
    master = next((a for a in reg.assets
                   if a.asset_type == "immutable_user_master"), None)
    rel = master.files[0] if master and master.files else MASTER_REL
    rep = verify_master()
    if not rep["exists"]:
        raise FileNotFoundError(f"canonical master missing: {rel}")
    if not rep["sha256_ok"]:
        raise ValueError(f"canonical master sha256 mismatch for {rel}: "
                         f"{rep['sha256']} != {rep['expected_sha256']}")
    return canonical_path(rel)


def battle_worker_split():
    if BATTLE_HEADLESS_WORKERS + BATTLE_VISIBLE_WORKERS != BATTLE_WORKERS:
        raise WorkerConfigError(
            f"battle split {BATTLE_HEADLESS_WORKERS}+{BATTLE_VISIBLE_WORKERS} "
            f"!= BATTLE_WORKERS {BATTLE_WORKERS}")
    return {"headless": BATTLE_HEADLESS_WORKERS,
            "visible": BATTLE_VISIBLE_WORKERS,
            "total": BATTLE_WORKERS}


class WorkerConfigError(ValueError):
    pass


def validate_worker_config(*, nav_workers=NAV_WORKERS,
                           nav_beginning=NAV_BEGINNING_WORKERS,
                           nav_anchor=NAV_ANCHOR_WORKERS,
                           battle_workers=BATTLE_WORKERS,
                           max_total=MAX_TOTAL_EMULATORS):
    """Raise :class:`WorkerConfigError` unless the split is internally consistent
    and fits the global emulator cap. Returns a summary dict on success.

    The final architecture requires ``nav_anchor == 0`` and
    ``nav_beginning == nav_workers``.
    """
    if min(nav_workers, nav_beginning, nav_anchor, battle_workers) < 0:
        raise WorkerConfigError("worker counts must be non-negative")
    if nav_anchor != 0:
        raise WorkerConfigError(
            f"the final 2×2 architecture has no anchor workers (got {nav_anchor})")
    if nav_beginning != nav_workers:
        raise WorkerConfigError(
            f"all {nav_workers} nav workers must be 'beginning' (got {nav_beginning})")
    if nav_beginning < 1:
        raise WorkerConfigError("need at least one beginning worker")
    total = nav_workers + battle_workers + FULL_WATCHER_EMULATORS
    if total > max_total:
        raise WorkerConfigError(
            f"nav+battle+full-watcher = {total} exceeds "
            f"MAX_TOTAL_EMULATORS {max_total}")
    if battle_workers < 1:
        raise WorkerConfigError("need at least one battle worker")
    return {
        "nav_workers": nav_workers,
        "nav_beginning": nav_beginning,
        "nav_anchor": nav_anchor,
        "battle_workers": battle_workers,
        "full_watcher_emulators": FULL_WATCHER_EMULATORS,
        "total_emulators": total,
        "headroom": max_total - total,
    }


def worker_roster(*, nav_beginning=NAV_BEGINNING_WORKERS,
                  nav_anchor=NAV_ANCHOR_WORKERS,
                  battle_workers=BATTLE_WORKERS):
    """Deterministic per-worker role list for the orchestrator.

    Every navigation worker: ``system="navigation"``, ``start_kind="beginning"``,
    ``start_state=CANONICAL_NAV_START``. Battle workers are fully separate.
    """
    validate_worker_config(nav_workers=nav_beginning + nav_anchor,
                           nav_beginning=nav_beginning, nav_anchor=nav_anchor,
                           battle_workers=battle_workers)
    roster = []
    for i in range(nav_beginning):
        roster.append({"index": len(roster), "system": "navigation",
                       "start_kind": "beginning",
                       "start_state": CANONICAL_NAV_START,
                       "nav_worker_ix": i})
    headless = min(battle_workers, BATTLE_HEADLESS_WORKERS) \
        if battle_workers == BATTLE_WORKERS else battle_workers
    visible = battle_workers - headless
    for i in range(headless):
        roster.append({"index": len(roster), "system": "battle",
                       "start_kind": "scenario", "render": "headless",
                       "battle_worker_ix": i})
    for i in range(visible):
        roster.append({"index": len(roster), "system": "battle",
                       "start_kind": "scenario", "render": "visible",
                       "window_title": "PKMai – BATTLE",
                       "battle_worker_ix": headless + i})
    return roster


def assert_all_navigation_starts_identical(roster):
    """Guard used by tests + the orchestrator: no nav worker may start anywhere
    but the single canonical start."""
    nav = [w for w in roster if w["system"] == "navigation"]
    starts = {w.get("start_state") for w in nav}
    kinds = {w.get("start_kind") for w in nav}
    if starts != {CANONICAL_NAV_START}:
        raise WorkerConfigError(f"navigation workers start from {starts}, "
                                f"must all be {CANONICAL_NAV_START!r}")
    if kinds != {"beginning"}:
        raise WorkerConfigError(f"navigation start_kinds {kinds}, must be "
                                f"{{'beginning'}}")
    return True
