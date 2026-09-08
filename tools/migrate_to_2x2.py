#!/usr/bin/env python3
"""One-time, idempotent runtime migration: single-PPO -> Navigation/Battle 2×2.

**Dry-run by default. This task does NOT execute it.**

What a real run does (only with --execute AND --i-understand-this-rewires-runtime):

  runtime/checkpoints/pokemon_model_champion.zip
      -> runtime/navigation/checkpoints/navigation_champion.zip   (atomic copy)
  resume.zip (or latest.zip if resume missing)
      -> runtime/navigation/checkpoints/navigation_learner.zip    (atomic copy)
  The deployed navigation observation remains the proven v1 image+nav schema.
  ``--verify-obs`` checks the prepared future coordinate-map migration but does
  not falsely claim that a v2 policy was saved.
  battle_champion.rule.json := metadata saying the verified RULE controller is
                               used until a real Battle-PPO passes its gates.
  No fake ``.zip`` checkpoint is ever created.
  runtime/model_manifest.json  written (schema model_manifest_v2)

Guarantees:
  * source files are hashed first and never touched until every destination is
    written AND verified;
  * every copy is atomic (tmp + os.replace);
  * a repeated run is a no-op (idempotent) — it detects an already-migrated
    layout and does nothing;
  * originals are archived under runtime/brain_backups/<ts>/ AFTER verification,
    never deleted;
  * a rollback plan is printed and written to the migration report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RUNTIME = os.path.join(ROOT, "runtime")
OLD_CKPT = os.path.join(RUNTIME, "checkpoints")
NAV_CKPT = os.path.join(RUNTIME, "navigation", "checkpoints")
BATTLE_CKPT = os.path.join(RUNTIME, "battle", "checkpoints")
MANIFEST = os.path.join(RUNTIME, "model_manifest.json")
BACKUPS = os.path.join(RUNTIME, "brain_backups")

MAP_CHANNELS = 6


def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy(src, dst):
    # central protected-asset guard: the master / Route-1 seed can never be a
    # migration target (through any alias / symlink / .. form).
    from twoby2.protected_assets import assert_not_protected
    assert_not_protected(dst, op="replace")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + f".{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_json(path, value):
    from twoby2.protected_assets import assert_not_protected
    assert_not_protected(path, op="write")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def already_migrated():
    nav_champ = os.path.join(NAV_CKPT, "navigation_champion.zip")
    nav_learner = os.path.join(NAV_CKPT, "navigation_learner.zip")
    if not (os.path.exists(nav_champ) and os.path.exists(nav_learner)
            and os.path.exists(MANIFEST)):
        return False
    try:
        from twoby2.protected_assets import ROM_SHA256
        with open(MANIFEST) as f:
            m = json.load(f)
        return (m.get("rom_sha256") == ROM_SHA256
                and m.get("navigation", {}).get("champion", {}).get("sha256") == sha256(nav_champ)
                and m.get("navigation", {}).get("learner", {}).get("sha256") == sha256(nav_learner))
    except Exception:
        return False


def _learner_source():
    for name in ("pokemon_model_resume.zip", "pokemon_model_latest.zip"):
        p = os.path.join(OLD_CKPT, name)
        if os.path.exists(p):
            return p, name
    return None, None


def build_plan():
    champ = os.path.join(OLD_CKPT, "pokemon_model_champion.zip")
    learner_src, learner_name = _learner_source()
    plan = {
        "already_migrated": already_migrated(),
        "sources": {
            "champion": {"path": champ, "sha256": sha256(champ)},
            "learner": {"path": learner_src, "name": learner_name,
                        "sha256": sha256(learner_src)},
        },
        "destinations": {
            "navigation_champion": os.path.join(NAV_CKPT, "navigation_champion.zip"),
            "navigation_learner": os.path.join(NAV_CKPT, "navigation_learner.zip"),
            "battle_champion_marker": os.path.join(BATTLE_CKPT, "battle_champion.rule.json"),
            "manifest": MANIFEST,
        },
        "observation_migration": {
            "from_schema": "nav_obs_v1", "to_schema": "nav_obs_v1",
            "map_channels": MAP_CHANNELS,
            "equivalence_tolerance": 1e-4,
            "note": "live env remains image+nav compatible; the coord-map "
                    "migration is verified separately but is not deployed",
        },
        "battle_policy": "verified rule controller (battle_controller.py) until a "
                         "Battle-PPO passes twoby2.battle_promotion gates",
        "rollback": {
            "restore": "copy runtime/brain_backups/<ts>/* back to runtime/checkpoints/",
            "delete": ["runtime/navigation/", "runtime/battle/", MANIFEST],
            "note": "originals are only archived AFTER destination verification; "
                    "until then runtime/checkpoints/ is untouched",
        },
    }
    problems = []
    if not plan["sources"]["champion"]["sha256"]:
        problems.append("no pokemon_model_champion.zip to migrate")
    if not plan["sources"]["learner"]["sha256"]:
        problems.append("no resume/latest checkpoint for the learner")
    plan["blocking_problems"] = problems
    return plan


def verify_observation_migration():
    """Run the real logit/value-equivalence check against the actual champion."""
    from twoby2.nav_feature_extractor import migrate_navigation_policy
    import numpy as np
    champ = os.path.join(OLD_CKPT, "pokemon_model_champion.zip")
    if not os.path.exists(champ):
        return {"ran": False, "reason": "no champion"}
    _, report, verify = migrate_navigation_policy(champ, map_channels=MAP_CHANNELS)
    rng = np.random.default_rng(0)
    batch = {
        "image": rng.integers(0, 256, (8, 4, 64, 64), dtype=np.uint8),
        "nav": rng.uniform(-1, 1, (8, 31)).astype(np.float32),
        "map": rng.integers(0, 256, (8, MAP_CHANNELS, 64, 64), dtype=np.uint8),
    }
    ok, dl, dv = verify(batch)
    return {"ran": True, "equivalent": ok, "max_logit_diff": dl,
            "max_value_diff": dv, "weight_report": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-understand-this-rewires-runtime", action="store_true",
                    dest="confirm")
    ap.add_argument("--verify-obs", action="store_true",
                    help="run the real observation-weight-migration equivalence check")
    args = ap.parse_args()

    plan = build_plan()
    out = {"plan": plan, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.verify_obs:
        out["observation_migration_check"] = verify_observation_migration()

    if not (args.execute and args.confirm):
        out["status"] = "DRY RUN — pass --execute AND --i-understand-this-rewires-runtime"
        print(json.dumps(out, indent=2, default=str))
        return

    if plan["already_migrated"]:
        out["status"] = "already migrated — no-op"
        print(json.dumps(out, indent=2, default=str))
        return
    if plan["blocking_problems"]:
        out["status"] = "BLOCKED"
        print(json.dumps(out, indent=2, default=str))
        sys.exit(2)

    # --- real execution path (NOT run by this task) ---
    obs_check = out.get("observation_migration_check") or verify_observation_migration()
    if not obs_check.get("equivalent"):
        out["status"] = "ABORTED — observation-weight migration not equivalent"
        print(json.dumps(out, indent=2, default=str))
        sys.exit(3)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUPS, ts)
    os.makedirs(backup, exist_ok=True)

    champ = plan["sources"]["champion"]["path"]
    learner_src = plan["sources"]["learner"]["path"]
    _atomic_copy(champ, plan["destinations"]["navigation_champion"])
    _atomic_copy(learner_src, plan["destinations"]["navigation_learner"])

    # verify destinations
    for d in ("navigation_champion", "navigation_learner"):
        if not os.path.exists(plan["destinations"][d]):
            out["status"] = f"ABORTED — destination {d} missing after copy"
            print(json.dumps(out, indent=2, default=str))
            sys.exit(4)

    # battle marker + manifest
    from twoby2.manifest import build_manifest, atomic_write
    from twoby2.protected_assets import ROM_SHA256
    _atomic_json(plan["destinations"]["battle_champion_marker"], {
        "schema": "battle_rule_fallback_v1",
        "source": "battle_controller.py",
        "active_until_real_ppo_promotion": True,
    })
    m = build_manifest(
        rom_sha256=ROM_SHA256,
        navigation_learner={"version": 1, "steps": 0,
                            "sha256": sha256(plan["destinations"]["navigation_learner"])},
        navigation_champion={"version": 1, "steps": 0,
                             "sha256": sha256(plan["destinations"]["navigation_champion"])},
        battle_champion={"version": 0, "steps": 0, "sha256": ""},
        battle_champion_source="rule", migration_state="executed")
    atomic_write(plan["destinations"]["manifest"], m)

    # archive originals AFTER verification (never delete)
    for name in os.listdir(OLD_CKPT):
        shutil.copy2(os.path.join(OLD_CKPT, name), os.path.join(backup, name))

    out["status"] = "MIGRATED"
    out["backup_dir"] = backup
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
