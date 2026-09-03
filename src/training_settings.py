"""Portable, local-overridable trainer settings."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import os
from pathlib import Path


SETTINGS_FILE_ENV = "PKMAI_SETTINGS_FILE"
LOCAL_SETTINGS_RELATIVE_PATH = Path("local") / "training_settings.json"


@dataclass(frozen=True)
class TrainingSettings:
    num_envs: int = 120
    train_forever: bool = True
    train_chunk_timesteps: int = 1_000_000
    total_timesteps: int = 100_000_000
    save_every_timesteps: int = 25_000
    learning_rate: float = 2.5e-05
    ppo_n_steps: int = 64
    ppo_batch_size: int = 256
    ppo_n_epochs: int = 4
    ppo_gamma: float = 0.995
    ppo_ent_coef: float = 0.008
    device: str = "auto"


def _settings_path(project_root: str | os.PathLike[str]) -> Path:
    override = os.environ.get(SETTINGS_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path(project_root) / LOCAL_SETTINGS_RELATIVE_PATH


def _validate(settings: TrainingSettings) -> None:
    positive_ints = (
        "num_envs",
        "train_chunk_timesteps",
        "total_timesteps",
        "save_every_timesteps",
        "ppo_n_steps",
        "ppo_batch_size",
        "ppo_n_epochs",
    )
    for name in positive_ints:
        if not isinstance(getattr(settings, name), int) or isinstance(
            getattr(settings, name), bool
        ) or getattr(settings, name) <= 0:
            raise ValueError(f"{name} muss eine positive ganze Zahl sein.")

    if settings.device not in {"auto", "cpu", "mps"}:
        raise ValueError("device muss auto, cpu oder mps sein.")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate muss größer als 0 sein.")
    if not 0 < settings.ppo_gamma <= 1:
        raise ValueError("ppo_gamma muss zwischen 0 und 1 liegen.")
    if settings.ppo_ent_coef < 0:
        raise ValueError("ppo_ent_coef darf nicht negativ sein.")


def load_training_settings(project_root: str | os.PathLike[str]) -> tuple[TrainingSettings, str]:
    """Load optional local JSON overrides without making them part of Git."""
    path = _settings_path(project_root)
    if not path.exists():
        settings = TrainingSettings()
        _validate(settings)
        return settings, str(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ungültiges JSON in {path}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path} muss ein JSON-Objekt enthalten.")

    allowed = {field.name for field in fields(TrainingSettings)}
    settings = TrainingSettings(**{key: value for key, value in raw.items() if key in allowed})
    _validate(settings)
    return settings, str(path)
