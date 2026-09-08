#!/usr/bin/env python3
"""Back up and reset one explicitly bounded PKMai runtime component.

The shell entry points in ``scripts/`` provide the interactive confirmations.
This module owns the auditable path lists, protected-asset checks and backups.
It is dry-run unless ``--apply`` is supplied.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
BACKUP_ROOT = os.path.join(ROOT, "backups", "component_resets")
sys.path.insert(0, os.path.join(ROOT, "src"))

from twoby2.protected_assets import assert_not_protected  # noqa: E402


def _p(*parts):
    return os.path.join(RUNTIME, *parts)


NAV_BRAIN = [
    _p("navigation", "checkpoints", "*.zip"),
    _p("navigation", "champion_score.json"),
    _p("navigation", "model_version.json"),
    _p("navigation", "trainer_status.json"),
    _p("navigation", "nav_horizon.json"),
]

MAP_GLOBAL = [
    _p("navigation", "movement_graph_v1.json"),
    _p("navigation", "nav_global.json"),
    _p("navigation", "shaping"),
    _p("curriculum_v20"),
    _p("exploration_memory"),
    _p("training_stats"),
    _p("instances_data"),
    _p("training_history.json"),
    _p("skeleton_map.json"),
    _p("watcher_mapping.json"),
    _p("mapper"),
    _p("room_captures"),
]

BATTLE_BRAIN = [
    _p("battle", "checkpoints", "battle_*.zip"),
    _p("battle", "battle_stats.json"),
    _p("battle", "battle_policy_status.json"),
    _p("battle", "battle_model_manifest.json"),
]

SHINY_TELEMETRY = [
    _p("shiny"),
]

# Legacy pre-2x2 model artifacts are inactive, but a complete reset must not
# leave a second learned brain that could later be loaded accidentally.
LEGACY_BRAINS = [
    _p("checkpoints", "*.zip"),
    _p("champion_score.json"),
    _p("model_version.json"),
    _p("trainer_status.json"),
]

SCOPES = {
    "nav-brain": NAV_BRAIN,
    "map-global": MAP_GLOBAL,
    "battle-brain": BATTLE_BRAIN,
    "full": (NAV_BRAIN + MAP_GLOBAL + BATTLE_BRAIN + SHINY_TELEMETRY
             + LEGACY_BRAINS),
}

PRESERVED = [
    "local/custom_integrations/PokemonFireRed-Gba/StartGame.state",
    "runtime/curriculum_shared (shared savestates)",
    "runtime/curriculum_states (per-agent savestates)",
    "runtime/battle/scenarios (battle scenario seeds)",
    "runtime/navigation/backups and runtime/battle/backups",
    "brain_backups and existing backups",
]


def _within_runtime(path):
    real = os.path.realpath(path)
    return real != os.path.realpath(RUNTIME) and real.startswith(
        os.path.realpath(RUNTIME) + os.sep
    )


def resolve_targets(scope):
    targets = []
    for spec in SCOPES[scope]:
        matches = glob.glob(spec) if glob.has_magic(spec) else [spec]
        for path in matches:
            path = os.path.abspath(path)
            if path not in targets:
                if not _within_runtime(path):
                    raise RuntimeError(f"unsafe reset target outside runtime: {path}")
                targets.append(path)
    return targets


def _iter_files(path):
    if os.path.isfile(path) or os.path.islink(path):
        yield path
    elif os.path.isdir(path):
        for base, _, names in os.walk(path):
            for name in names:
                yield os.path.join(base, name)


def guard_targets(targets):
    for target in targets:
        for path in _iter_files(target):
            assert_not_protected(path, op="delete")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_backup(scope, targets):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = os.path.join(BACKUP_ROOT, f"{scope}_{stamp}")
    os.makedirs(destination, exist_ok=False)
    archive = os.path.join(destination, "runtime_before_reset.tar.gz")

    # Full means exactly that: archive all runtime data, not only targets.
    archive_targets = [RUNTIME] if scope == "full" else [
        path for path in targets if os.path.lexists(path)
    ]
    with tarfile.open(archive, "w:gz") as bundle:
        for path in archive_targets:
            bundle.add(path, arcname=os.path.relpath(path, ROOT), recursive=True)

    archived_files = []
    for path in archive_targets:
        for file_path in _iter_files(path):
            if os.path.isfile(file_path):
                archived_files.append({
                    "path": os.path.relpath(file_path, ROOT),
                    "size": os.path.getsize(file_path),
                    "sha256": _sha256(file_path),
                })
    manifest = {
        "schema": "pkmai_component_reset_backup_v1",
        "scope": scope,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": os.path.basename(archive),
        "archive_sha256": _sha256(archive),
        "files": archived_files,
        "preserved": PRESERVED,
    }
    with open(os.path.join(destination, "BACKUP_MANIFEST.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return destination, manifest


def delete_targets(targets):
    deleted = []
    for path in targets:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
            deleted.append(os.path.relpath(path, ROOT))
        elif os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            deleted.append(os.path.relpath(path, ROOT) + "/ (contents)")
    return deleted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    targets = resolve_targets(args.scope)
    guard_targets(targets)
    present = [p for p in targets if os.path.lexists(p)]
    report = {
        "scope": args.scope,
        "dry_run": not args.apply,
        "targets": [os.path.relpath(p, ROOT) for p in targets],
        "present_targets": [os.path.relpath(p, ROOT) for p in present],
        "preserved": PRESERVED,
    }
    if not args.apply:
        print(json.dumps(report, indent=2))
        print("\nDRY-RUN: nichts geaendert und kein Backup geschrieben.")
        return

    backup_dir, manifest = create_backup(args.scope, targets)
    report["backup_dir"] = os.path.relpath(backup_dir, ROOT)
    report["backup_sha256"] = manifest["archive_sha256"]
    report["deleted"] = delete_targets(targets)
    report["applied"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
