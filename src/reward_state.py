"""Persistent one-time reward events, scoped to a training/evaluation directory."""
import json
import os
from contextlib import nullcontext
from pathlib import Path


def claim_event(directory, key, registry=None, lock=None):
    path = Path(directory) / 'reward_events.json'
    with lock if lock is not None else nullcontext():
        if registry is not None and key in registry:
            return False
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
            if key in data:
                if registry is not None:
                    registry[key] = 1
                return False
            data[key] = True
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            if registry is not None:
                registry[key] = 1
            return True
        except (OSError, ValueError):
            # Do not pay an allegedly once-only bonus if it cannot be recorded.
            return False
