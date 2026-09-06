"""Restart-safe claims for rewards that may pay only once globally."""

import json
import os
from contextlib import nullcontext
from pathlib import Path


def claim_event(directory, key, lock=None):
    """Persist and claim ``key`` atomically; failure never pays a bonus."""
    path = Path(directory) / "reward_events.json"
    context = lock if lock is not None else nullcontext()
    with context:
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
            if key in data:
                return False
            data[key] = True
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(data, sort_keys=True))
            os.replace(temporary, path)
            return True
        except (OSError, ValueError, TypeError):
            return False
