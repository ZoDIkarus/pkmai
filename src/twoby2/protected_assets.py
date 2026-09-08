"""Protected-savestate registry — the master and the manual battle seed.

Some files are the user's own hand-made assets and must NEVER be deleted,
overwritten, normalised or renamed by any reset / migration / eviction /
cleanup path:

  * ``immutable_user_master`` — the canonical post-parcel start
    ``local/custom_integrations/PokemonFireRed-Gba/StartGame.state``
    (bank 4 / map 3 / x6 / y4, in Oak's lab, starter already obtained).
  * ``immutable_user_battle_seed`` — the manually-made Route-1 fighter seed
    (a ``.state.gz`` + ``.meta.json`` pair).

This module builds the registry, resolves protected paths robustly (project-
relative, symlink / ``..`` / alias resistant) and exposes a single
``assert_not_protected(path, op)`` that every mutating path calls first. It
writes **no** real runtime registry here — construction, dry-run and tests only.
"""
from __future__ import annotations

import hashlib
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TYPE_MASTER = "immutable_user_master"
TYPE_BATTLE_SEED = "immutable_user_battle_seed"
TYPE_PROBE_SEED = "immutable_ram_probe_seed"

MUTATING_OPS = ("delete", "replace", "overwrite", "normalize", "rename",
                "truncate", "evict", "move", "publish", "write")


# --------------------------------------------------------------------------
# path resolution — everything is normalised to a real absolute path so that
# ``foo/../StartGame.state``, a relative alias, or a symlink into the file all
# resolve to the same protected identity.
# --------------------------------------------------------------------------
def canonical_path(path, *, root=PROJECT_ROOT):
    """Absolute, symlink-resolved, ``..``-collapsed path. A relative input is
    taken relative to the project root."""
    p = str(path)
    if not os.path.isabs(p):
        p = os.path.join(root, p)
    return os.path.realpath(p)


def project_relative(path, *, root=PROJECT_ROOT):
    try:
        return os.path.relpath(canonical_path(path, root=root), root)
    except ValueError:
        return canonical_path(path, root=root)


def sha256_file(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ProtectedAsset:
    def __init__(self, logical_id, *, asset_type, files, rom_sha256="",
                 map_position=None, story_summary=None, party_summary=None,
                 candidate_ambiguity=False, notes=""):
        self.logical_id = logical_id
        self.asset_type = asset_type
        # files: list of project-relative paths (a seed = state.gz + meta.json)
        self.files = [project_relative(f) for f in files]
        self.rom_sha256 = rom_sha256
        self.map_position = dict(map_position or {})
        self.story_summary = dict(story_summary or {})
        self.party_summary = list(party_summary or [])
        self.candidate_ambiguity = bool(candidate_ambiguity)
        self.notes = notes
        self.protected_from_delete = True
        self.protected_from_replace = True
        self.protected_from_normalize = True

    def abs_files(self, *, root=PROJECT_ROOT):
        return [canonical_path(f, root=root) for f in self.files]

    def covers(self, path, *, root=PROJECT_ROOT):
        return canonical_path(path, root=root) in set(self.abs_files(root=root))

    def to_dict(self, *, root=PROJECT_ROOT):
        d = {
            "logical_id": self.logical_id,
            "type": self.asset_type,
            "files": [
                {"project_relative_path": f,
                 "sha256": sha256_file(canonical_path(f, root=root))}
                for f in self.files
            ],
            "rom_sha256": self.rom_sha256,
            "map_position": self.map_position,
            "story_summary": self.story_summary,
            "party_summary": self.party_summary,
            "candidate_ambiguity": self.candidate_ambiguity,
            "protected_from_delete": True,
            "protected_from_replace": True,
            "protected_from_normalize": True,
            "notes": self.notes,
        }
        return d


class ProtectedRegistry:
    SCHEMA = "protected_savestate_registry_v1"

    def __init__(self, assets=None, *, root=PROJECT_ROOT):
        self.root = root
        self.assets = list(assets or [])

    def add(self, asset):
        self.assets.append(asset)
        return self

    def is_protected(self, path, *, op=None):
        """True if ``path`` is any protected file. ``op`` is advisory — all
        current assets are protected against every mutating op, so the answer
        does not depend on it, but it is echoed in
        :func:`assert_not_protected`."""
        target = canonical_path(path, root=self.root)
        return any(a.covers(target, root=self.root) for a in self.assets)

    def protecting_asset(self, path):
        target = canonical_path(path, root=self.root)
        for a in self.assets:
            if a.covers(target, root=self.root):
                return a
        return None

    def to_dict(self):
        return {"schema": self.SCHEMA,
                "root": self.root,
                "assets": [a.to_dict(root=self.root) for a in self.assets]}

    def dump_dry_run(self, path):
        """Return what WOULD be written; never touches a real runtime file."""
        return {"would_write": path, "content": self.to_dict()}


class ProtectedPathError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# the default registry for this project
# --------------------------------------------------------------------------
MASTER_REL = "local/custom_integrations/PokemonFireRed-Gba/StartGame.state"
MASTER_SHA256 = "0c0e26d9b6e321381ffca4593bf2cc04e8e9d282ddbdd799c07e76b4c66683d0"
ROM_SHA256 = "eed4fb0242bcbb6e0b0b80b197b47e3aec99ef4c5562cc45bae50ae86b970507"

TYPE_LIVE_ANCHOR = "protected_live_anchor"

# The user-confirmed hand-made Route-1 seed. A direct read on 2026-09-07 showed
# that this historical state is hurt, so it remains immutable but must not be
# described or selected as a healthy battle-probe start.
ROUTE1_MANUAL_SEED = {
    "state": "brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.state.gz",
    "meta": "brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.meta.json",
    "state_sha256": "4244fa04db32fcd313ab159930787a613367904d5bd0f9c6ad8a0654ff6b0b48",
    "meta_sha256": "37087431bf762a4cdaf2c885b800d194629f06b5bcd0e782d99c520d6fd57682",
    "note": "user-made historical Route-1 seed; direct RAM read: Squirtle "
            "L11 0/31, Rattata L4 0/16, Pidgey L3 2/15; preserved unchanged",
}

# The OTHER pair is a *mutable* live FRONTIER anchor — NOT a manual seed. It is
# still protected against reset / migration / cleanup, under its own identity.
ROUTE1_LIVE_ANCHOR = {
    "state": "runtime/curriculum_shared/stage_frontier_2.state.gz",
    "meta": "runtime/curriculum_shared/stage_frontier_2.meta.json",
    "state_sha256": "4c22f65e7566ef68b04d5a0ff98af69d858cce95abc73af846ceeef5e48e2563",
    "meta_sha256": "dee031c1f09e146616724722195098af244629783cb3a0dedbbd5ed82433b8ad",
    "note": "live FRONTIER anchor (training-updated, agent 29); protected from "
            "reset/migration/cleanup but NOT a manual seed. Never normalise it "
            "directly — copy first.",
}

RAM_PROBE_ROUTE1_SEED = {
    "state": "brain_backups/ram_probe_route1_seed/stage_frontier_2.state.gz",
    "meta": "brain_backups/ram_probe_route1_seed/stage_frontier_2.meta.json",
    "state_sha256": "4c22f65e7566ef68b04d5a0ff98af69d858cce95abc73af846ceeef5e48e2563",
    "meta_sha256": "dee031c1f09e146616724722195098af244629783cb3a0dedbbd5ed82433b8ad",
    "note": "frozen read-only copy of the healthy live Route-1 anchor for "
            "manual RAM calibration: Squirtle L11 25/31 with four moves, "
            "Rattata 16/16, Pidgey 15/15",
}


def default_registry(*, root=PROJECT_ROOT, include_route1_seed=True):
    """The project's protected assets.

    * ``canonical_post_parcel_master`` — the immutable user master.
    * ``route1_manual_battle_seed`` — the user-confirmed manual Route-1 seed
      (``candidate_ambiguity`` is False).
    * ``protected_live_route1_frontier_anchor`` — the mutable live anchor,
      protected from destructive ops but explicitly NOT a manual seed.
    * ``protected_ram_probe_route1_seed`` — frozen healthy calibration copy.
    """
    reg = ProtectedRegistry(root=root)
    reg.add(ProtectedAsset(
        "canonical_post_parcel_master",
        asset_type=TYPE_MASTER,
        files=[MASTER_REL],
        rom_sha256=ROM_SHA256,
        map_position={"map_bank": 4, "map_id": 3, "x": 6, "y": 4,
                      "location": "Oak's lab, Pallet Town"},
        story_summary={"parcel_delivered": True, "starter_obtained": True,
                       "pokedex": True, "badges": 0,
                       "source": "docs/CURRENT_LOGIC.md + integration StartGame.state"},
        notes="user-created immutable master; never delete/replace/normalize/rename. "
              f"expected sha256 {MASTER_SHA256}"))
    if include_route1_seed:
        reg.add(ProtectedAsset(
            "route1_manual_battle_seed",
            asset_type=TYPE_BATTLE_SEED,
            files=[ROUTE1_MANUAL_SEED["state"], ROUTE1_MANUAL_SEED["meta"]],
            rom_sha256=ROM_SHA256,
            map_position={"map_bank": 3, "map_id": 19, "x": 17, "y": 24,
                          "route": "Route 1"},
            party_summary=[
                {"species": 7, "name": "Squirtle", "level": 11, "hp": "0/31"},
                {"species": 19, "name": "Rattata", "level": 4, "hp": "0/16"},
                {"species": 16, "name": "Pidgey", "level": 3, "hp": "2/15"}],
            candidate_ambiguity=False,          # user-confirmed
            notes="user-confirmed manual Route-1 fighter seed. "
                  f"state sha256 {ROUTE1_MANUAL_SEED['state_sha256']}; "
                  f"meta sha256 {ROUTE1_MANUAL_SEED['meta_sha256']}. "
                  "Battle training uses derived copies only; never write here."))
        reg.add(ProtectedAsset(
            "protected_live_route1_frontier_anchor",
            asset_type=TYPE_LIVE_ANCHOR,
            files=[ROUTE1_LIVE_ANCHOR["state"], ROUTE1_LIVE_ANCHOR["meta"]],
            rom_sha256=ROM_SHA256,
            map_position={"map_bank": 3, "map_id": 19, "route": "Route 1"},
            candidate_ambiguity=False,
            notes="MUTABLE live FRONTIER anchor — NOT a manual seed. Protected "
                  "from reset / migration / cleanup; normalisation must never "
                  "target it directly (copy to a temp working file first)."))
        reg.add(ProtectedAsset(
            "protected_ram_probe_route1_seed",
            asset_type=TYPE_PROBE_SEED,
            files=[RAM_PROBE_ROUTE1_SEED["state"], RAM_PROBE_ROUTE1_SEED["meta"]],
            rom_sha256=ROM_SHA256,
            map_position={"map_bank": 3, "map_id": 19, "x": 10, "y": 6,
                          "route": "Route 1"},
            party_summary=[
                {"species": 7, "name": "Squirtle", "level": 11, "hp": "25/31",
                 "moves": 4},
                {"species": 19, "name": "Rattata", "level": 4, "hp": "16/16"},
                {"species": 16, "name": "Pidgey", "level": 3, "hp": "15/15"}],
            candidate_ambiguity=False,
            notes=RAM_PROBE_ROUTE1_SEED["note"] + "; read-only; never overwrite"))
    return reg


def assert_not_protected(path, *, op="mutate", registry=None, root=PROJECT_ROOT):
    """Raise :class:`ProtectedPathError` if ``path`` (any alias / symlink /
    ``..`` form) is a protected asset and ``op`` mutates it. Call this on every
    real WRITE / PUBLISH target — never on a pure read source.

    Read ops (``op="read"`` / ``op="copy_source"``) are always allowed.
    """
    if op in ("read", "copy_source"):
        return True
    reg = registry if registry is not None else default_registry(root=root)
    asset = reg.protecting_asset(path)
    if asset is None:
        return True
    if op in MUTATING_OPS or op == "mutate":
        raise ProtectedPathError(
            f"refusing to {op} protected asset {asset.logical_id!r} "
            f"({project_relative(path, root=root)})")
    return True


def assert_write_target_ok(path, *, op="write", registry=None, root=PROJECT_ROOT):
    """Strict check for an actual write / publish destination. The caller here
    is definitely about to write, so ANY protected target raises regardless of
    the ``op`` label."""
    if op in ("read", "copy_source"):
        raise ValueError("assert_write_target_ok is for write targets only")
    reg = registry if registry is not None else default_registry(root=root)
    asset = reg.protecting_asset(path)
    if asset is not None:
        raise ProtectedPathError(
            f"refusing to {op} protected asset {asset.logical_id!r} "
            f"({project_relative(path, root=root)})")
    return True


class Sha256Mismatch(RuntimeError):
    pass


def read_only_copy(protected_source, dest, *, registry=None, root=PROJECT_ROOT):
    """Copy a (possibly protected) SOURCE byte-for-byte into an UNPROTECTED
    ``dest``.

    * the source is only ever read;
    * ``dest`` must not be a protected path (raises otherwise);
    * the source sha256 is captured before AND re-checked after the copy — any
      change to the original is a hard error;
    * returns ``{"source", "dest", "sha256", "bytes"}``.
    """
    reg = registry if registry is not None else default_registry(root=root)
    src = canonical_path(protected_source, root=root)
    dst = canonical_path(dest, root=root)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source not found: {protected_source}")
    # dest must be OUTSIDE every protected path
    if reg.protecting_asset(dst) is not None:
        raise ProtectedPathError(
            f"copy destination {project_relative(dst, root=root)} is itself "
            f"a protected asset")
    before = sha256_file(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        data = fi.read()
        fo.write(data)
    after = sha256_file(src)
    if before != after:
        raise Sha256Mismatch(
            f"protected source {project_relative(src, root=root)} changed during "
            f"a read-only copy: {before} -> {after}")
    dst_sha = sha256_file(dst)
    if dst_sha != before:
        raise Sha256Mismatch(
            f"copy is not byte-identical: src {before} != dst {dst_sha}")
    return {"source": project_relative(src, root=root),
            "dest": dst, "sha256": before, "bytes": len(data)}


def guarded_actions(actions, *, registry=None, root=PROJECT_ROOT):
    """Filter a list of ``(op, path)`` tuples, raising on any protected target.
    Returns the (unchanged) list when clean — used by the reset planners."""
    for op, path in actions:
        assert_not_protected(path, op=_op_kind(op), registry=registry, root=root)
    return actions


def _op_kind(action_name):
    a = str(action_name).lower()
    if "delete" in a:
        return "delete"
    if "normal" in a:
        return "normalize"
    if "copy_from" in a or "reset" in a or "overwrite" in a or "replace" in a:
        return "replace"
    if "evict" in a:
        return "evict"
    return "mutate"


def verify_master(*, root=PROJECT_ROOT):
    """Read-only check of the master file: exists + sha256 matches the recorded
    value. Returns a report; never modifies anything."""
    p = canonical_path(MASTER_REL, root=root)
    actual = sha256_file(p)
    return {
        "path": project_relative(p, root=root),
        "exists": os.path.isfile(p),
        "sha256": actual,
        "expected_sha256": MASTER_SHA256,
        "sha256_ok": actual == MASTER_SHA256,
        "is_symlink": os.path.islink(canonical_path(MASTER_REL, root=root))
        or os.path.islink(os.path.join(root, MASTER_REL)),
    }
