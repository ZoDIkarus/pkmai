"""Atomic JPEG publication for visible policy watchers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


WATCHER_STREAM_QUALITY = 85


def write_watcher_stream_frame(frame: np.ndarray, target: str | Path) -> bool:
    """Encode and atomically replace a watcher JPEG without exposing partial files."""
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, WATCHER_STREAM_QUALITY]
    )
    if not ok:
        return False
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=target_path.parent, prefix=".watcher-", suffix=".jpg"
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded.tobytes())
        os.replace(temporary_path, target_path)
        return True
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
