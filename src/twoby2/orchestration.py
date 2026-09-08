"""Start / stop / status preparation for the 2×2 system (Phase 4).

Nothing here starts a process. It defines the process set, per-process PID /
status field layout, double-start guard and the status-report shape, all behind
feature gates that stay OFF.
"""
from __future__ import annotations

import os

from twoby2 import feature_enabled
from twoby2.config import (NAV_WORKERS, BATTLE_HEADLESS_WORKERS,
                           BATTLE_VISIBLE_WORKERS, BATTLE_WORKERS,
                           FULL_WATCHER_EMULATORS, MAX_TOTAL_EMULATORS)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The process set. Each has its OWN pid file and status field — no shared pid.
PROCESSES = {
    "navigation_trainer": {
        "script": "src/nav_train.py", "pid_file": "runtime/nav_train.pid",
        "workers": NAV_WORKERS, "gate": "nav_battle_wrapper",
        "desc": f"{NAV_WORKERS} navigation workers, canonical-start beginning"},
    "battle_trainer_headless": {
        "script": "src/battle_train.py", "pid_file": "runtime/battle_train.pid",
        "workers": BATTLE_WORKERS, "gate": "battle_env",
        "desc": f"{BATTLE_WORKERS} battle workers; worker 9 publishes "
                "frames to the mirror"},
    "battle_trainer_visible": {
        "script": "tools/battle_mirror_watch.py", "pid_file": "runtime/battle_watch.pid",
        "workers": 0, "gate": "battle_env",
        "desc": "the 'PKMai – BATTLE' frame mirror (no extra emulator)"},
    "full_watcher": {
        "script": "src/watch.py", "pid_file": "runtime/watch.pid",
        "workers": 1, "gate": "nav_battle_wrapper",
        "desc": "inference-only full watcher (shared central components)"},
    "web": {
        "script": "src/web_stream.py", "pid_file": "runtime/web.pid",
        "workers": 1, "gate": None, "desc": "dashboard :8001"},
    "status_monitor": {
        "script": "tools/pkmai_status.py", "pid_file": "runtime/status.pid",
        "workers": 1, "gate": None, "desc": "status monitor"},
}


class DoubleStartError(RuntimeError):
    pass


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def check_no_double_start(name, *, root=PROJECT_ROOT):
    """Raise :class:`DoubleStartError` if the process' pid file exists and its
    pid is alive. (Read-only — does not start anything.)"""
    proc = PROCESSES.get(name)
    if proc is None:
        raise KeyError(name)
    pf = os.path.join(root, proc["pid_file"])
    if os.path.isfile(pf):
        try:
            with open(pf) as f:
                pid = f.read().strip()
        except OSError:
            return True
        if _pid_alive(pid):
            raise DoubleStartError(f"{name} already running (pid {pid}, {pf})")
    return True


def total_emulator_budget():
    total = (NAV_WORKERS + BATTLE_WORKERS + FULL_WATCHER_EMULATORS)
    assert BATTLE_HEADLESS_WORKERS + BATTLE_VISIBLE_WORKERS == BATTLE_WORKERS
    assert total <= MAX_TOTAL_EMULATORS, f"{total} > {MAX_TOTAL_EMULATORS}"
    return {"navigation": NAV_WORKERS,
            "battle_headless": BATTLE_HEADLESS_WORKERS,
            "battle_visible": BATTLE_VISIBLE_WORKERS,
            "battle_total": BATTLE_WORKERS,
            "full_watcher": FULL_WATCHER_EMULATORS,
            "grand_total": total,
            "cap": MAX_TOTAL_EMULATORS}


def status_shape(*, route_group_plan=None, nav_versions=None, battle_versions=None):
    """The status report the dashboard/monitor will render once activated."""
    budget = total_emulator_budget()
    rgp = route_group_plan or {}
    return {
        "schema": "twoby2_status_v1",
        "live": any(feature_enabled(p["gate"]) for p in PROCESSES.values()
                    if p["gate"]),
        "workers": {
            "navigation": budget["navigation"],
            "battle_headless": budget["battle_headless"],
            "battle_visible": budget["battle_visible"],
        },
        "route_groups": rgp.get("active_routes", []),
        "regression_core_routes": rgp.get("regression_core_routes", []),
        "battle_watcher_route": (rgp.get("visible_watcher") or {}).get("route"),
        "battle_watcher_scenario": (rgp.get("visible_watcher") or {}).get("scenario_signature"),
        "navigation_learner_version": (nav_versions or {}).get("learner"),
        "navigation_champion_version": (nav_versions or {}).get("champion"),
        "battle_learner_version": (battle_versions or {}).get("learner"),
        "battle_champion_version": (battle_versions or {}).get("champion"),
        "feature_gates": {k: feature_enabled(k) for k in (
            "battle_env", "battle_executor_live", "battle_router_live",
            "nav_battle_wrapper", "adaptive_nav_horizon",
            "watcher_battle_champion", "runtime_migration")},
        "ram_blockers": ["gBattleMons", "gBattlerPartyIndexes",
                         "gActionSelectionCursor", "gMoveSelectionCursor",
                         "battle_menu_state"],
    }
