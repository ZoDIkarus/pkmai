"""Atomic, dependency-light rollout spool shared by master and brain."""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path
from typing import Iterator

import numpy as np

_WORKER_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_worker_id(worker_id: str) -> str:
    value = _WORKER_ID.sub("-", worker_id.strip()).strip(".-")
    if not value:
        raise ValueError("worker_id required")
    return value


def encode_rollout(batch: dict[str, np.ndarray]) -> bytes:
    required = {"images", "nav", "actions", "rewards", "dones", "log_probs", "values"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"rollout fields missing: {sorted(missing)}")
    sizes = {len(np.asarray(batch[name])) for name in required}
    if len(sizes) != 1 or not sizes.pop():
        raise ValueError("rollout fields must have the same non-zero length")
    output = io.BytesIO()
    np.savez_compressed(output, **batch)
    return output.getvalue()


def decode_rollout(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_rollout(inbox: Path, worker_id: str, batch: dict[str, np.ndarray]) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_worker_id(worker_id)
    name = f"{time.time_ns()}-{safe_id}.npz"
    target = inbox / name
    temporary = inbox / f".{name}.tmp"
    temporary.write_bytes(encode_rollout(batch))
    os.replace(temporary, target)
    return target


def consume_rollouts(inbox: Path, limit: int) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    if limit < 1 or not inbox.exists():
        return
    for path in sorted(inbox.glob("*.npz"))[:limit]:
        claimed = path.with_suffix(".processing")
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            continue
        try:
            worker_id = claimed.name.split("-", 1)[1].rsplit(".processing", 1)[0]
            yield worker_id, decode_rollout(claimed.read_bytes())
        finally:
            claimed.unlink(missing_ok=True)
