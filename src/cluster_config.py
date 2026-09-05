"""Shared, dependency-light cluster compatibility rules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping


class ClusterCompatibilityError(ValueError):
    """Raised when a remote worker cannot safely join the active brain."""


@dataclass(frozen=True)
class ClusterSettings:
    environment_signature: str
    version_window: int = 2


@dataclass(frozen=True)
class ClusterRuntimeSettings:
    env_runners: int = 64
    agents_per_runner: int = 1


def load_cluster_runtime_settings(
    environment: Mapping[str, str],
) -> ClusterRuntimeSettings:
    """Read the brain's rollout topology from its explicit deployment settings."""
    return ClusterRuntimeSettings(
        env_runners=max(1, int(environment.get("PKMAI_CLUSTER_ENV_RUNNERS", "64"))),
        agents_per_runner=max(
            1, int(environment.get("PKMAI_CLUSTER_AGENTS_PER_RUNNER", "1"))
        ),
    )


@dataclass(frozen=True)
class RegistrationDecision:
    accepted: bool
    reason: str
    policy_version: int


def build_environment_signature(
    *, observation_shape: tuple[int, int, int], nav_features: int, action_count: int
) -> str:
    payload = {
        "action_count": action_count,
        "nav_features": nav_features,
        "observation_shape": list(observation_shape),
        "protocol": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_worker_registration(
    settings: ClusterSettings,
    *,
    worker_signature: str,
    worker_policy_version: int,
    master_policy_version: int,
) -> RegistrationDecision:
    if worker_signature != settings.environment_signature:
        raise ClusterCompatibilityError("environment signature mismatch")
    if worker_policy_version < master_policy_version - settings.version_window:
        return RegistrationDecision(False, "policy_reload_required", master_policy_version)
    return RegistrationDecision(True, "accepted", master_policy_version)
