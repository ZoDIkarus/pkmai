"""Model-manifest schema + helpers (Phase 4).

The manifest couples the two systems: which navigation champion and which
battle champion are the *active combo*, plus versions/steps/hashes/eval and the
horizon + scenario-pool + migration state. A navigation eval is only comparable
with another when both used the same pinned battle champion **and** the same
observation schema, ROM hash and manifest wiring.

This module never chooses a live path itself — every writer passes an explicit
``path``. The live ``model_manifest.json`` is NOT written by this task.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

SCHEMA = "model_manifest_v2"
BATTLE_SOURCES = ("ppo", "rule")
MIGRATION_STATES = ("not_started", "prepared", "dry_run_ok", "executed")


def _slot(version=0, steps=0, sha256="", eval_metrics=None, source=None,
          extra=None):
    d = {"version": int(version or 0), "steps": int(steps or 0),
         "sha256": str(sha256 or "")}
    if eval_metrics is not None:
        d["eval"] = dict(eval_metrics)
    if source is not None:
        d["source"] = source
    if extra:
        d.update(extra)
    return d


def build_manifest(*, rom_sha256="",
                   nav_obs_schema="nav_obs_v1", battle_obs_schema="battle_obs_v1",
                   navigation_learner=None, navigation_candidate=None,
                   navigation_champion=None,
                   battle_learner=None, battle_candidate=None,
                   battle_champion=None, battle_champion_source="rule",
                   horizon_state=None, scenario_pool_schema="scenario_pool_v1",
                   migration_state="not_started", generation=1,
                   combo_version=1):
    if battle_champion_source not in BATTLE_SOURCES:
        raise ValueError(f"battle_champion_source must be one of {BATTLE_SOURCES}")
    if migration_state not in MIGRATION_STATES:
        raise ValueError(f"migration_state must be one of {MIGRATION_STATES}")
    nav_champ = navigation_champion or _slot()
    bat_champ = {**(battle_champion or _slot()), "source": battle_champion_source}
    return {
        "schema": SCHEMA,
        "updated": _now(),
        "generation": int(generation),
        "rom_sha256": str(rom_sha256 or ""),
        "obs_schemas": {"navigation": nav_obs_schema, "battle": battle_obs_schema},
        "navigation": {
            "learner": navigation_learner or _slot(),
            "candidate": navigation_candidate or _slot(),
            "champion": nav_champ,
        },
        "battle": {
            "learner": battle_learner or _slot(),
            "candidate": battle_candidate or _slot(),
            "champion": bat_champ,
        },
        "horizon_state": dict(horizon_state) if horizon_state else None,
        "scenario_pool_schema": scenario_pool_schema,
        "migration_state": migration_state,
        "watcher_combo": {
            "navigation_champion_version": int(nav_champ.get("version", 0)),
            "navigation_champion_sha256": nav_champ.get("sha256", ""),
            "battle_champion_version": int(bat_champ.get("version", 0)),
            "battle_champion_sha256": bat_champ.get("sha256", ""),
            "battle_champion_source": battle_champion_source,
            "combo_version": int(combo_version),
        },
    }


def validate_manifest(m):
    """Return a list of problems ([] == valid)."""
    problems = []
    if not isinstance(m, dict):
        return ["manifest is not a dict"]
    if m.get("schema") != SCHEMA:
        problems.append(f"schema != {SCHEMA}")
    if not isinstance(m.get("generation"), int):
        problems.append("generation missing/!int")
    if not isinstance(m.get("obs_schemas"), dict) or \
            "navigation" not in m.get("obs_schemas", {}) or \
            "battle" not in m.get("obs_schemas", {}):
        problems.append("obs_schemas incomplete")
    for sys_key in ("navigation", "battle"):
        sysd = m.get(sys_key)
        if not isinstance(sysd, dict):
            problems.append(f"missing {sys_key} block")
            continue
        for role in ("learner", "candidate", "champion"):
            slot = sysd.get(role)
            if not isinstance(slot, dict) or not {"version", "steps", "sha256"} <= set(slot):
                problems.append(f"{sys_key}.{role} missing version/steps/sha256")
    bsrc = (m.get("battle", {}).get("champion", {}) or {}).get("source")
    if bsrc not in BATTLE_SOURCES:
        problems.append(f"battle.champion.source {bsrc!r} invalid")
    if m.get("migration_state") not in MIGRATION_STATES:
        problems.append("migration_state invalid")
    combo = m.get("watcher_combo")
    if not isinstance(combo, dict):
        problems.append("missing watcher_combo")
    else:
        for k in ("navigation_champion_version", "battle_champion_version",
                  "battle_champion_sha256", "combo_version"):
            if k not in combo:
                problems.append(f"watcher_combo.{k} missing")
    return problems


def _combo(m):
    return (m or {}).get("watcher_combo") or {}


def evals_comparable(manifest_a, manifest_b):
    """Two navigation evaluations are comparable only if BOTH manifests are
    valid AND agree on: battle champion version, battle champion sha256, battle
    champion source, navigation obs schema and ROM hash.

    Invalid or empty manifests are never comparable — ``evals_comparable({}, {})``
    is ``False``.
    """
    if validate_manifest(manifest_a) or validate_manifest(manifest_b):
        return False
    ca, cb = _combo(manifest_a), _combo(manifest_b)
    if not ca or not cb:
        return False
    same_battle = (
        ca.get("battle_champion_version") == cb.get("battle_champion_version")
        and ca.get("battle_champion_sha256") == cb.get("battle_champion_sha256")
        and ca.get("battle_champion_source") == cb.get("battle_champion_source"))
    same_obs = (manifest_a.get("obs_schemas", {}).get("navigation")
                == manifest_b.get("obs_schemas", {}).get("navigation"))
    same_rom = manifest_a.get("rom_sha256") == manifest_b.get("rom_sha256")
    return bool(same_battle and same_obs and same_rom)


def needs_rebaseline(old_manifest, new_battle_champion_version,
                     new_battle_champion_sha256=None):
    c = _combo(old_manifest)
    if c.get("battle_champion_version") != int(new_battle_champion_version or 0):
        return True
    if new_battle_champion_sha256 is not None and \
            c.get("battle_champion_sha256") != new_battle_champion_sha256:
        return True
    return False


def pin_for_navigation_generation(manifest):
    c = _combo(manifest)
    return {
        "battle_champion_version": int(c.get("battle_champion_version", 0)),
        "battle_champion_sha256": c.get("battle_champion_sha256", ""),
        "battle_champion_source": c.get("battle_champion_source", "rule"),
        "navigation_champion_version": int(c.get("navigation_champion_version", 0)),
        "combo_version": int(c.get("combo_version", 1)),
        "generation": int((manifest or {}).get("generation", 1)),
    }


def atomic_write(path, manifest):
    """Write the manifest atomically to ``path``. The caller always supplies
    the destination. Crash-safe: a failure before ``os.replace`` leaves the
    existing file untouched and no partial file at ``path``.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(f"refusing to write invalid manifest: {problems}")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, separators=(",", ":"), sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def load(path):
    with open(path) as f:
        return json.load(f)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
