"""Battle-policy routing + version pinning (Phases 3 & 4).

Four consumer modes, each with its own overworld / battle policy rule:

  NAV_TRAINING    overworld -> navigation LEARNER
                  battle    -> pinned battle CHAMPION (or verified rule fallback)
  NAV_EVAL        overworld -> the model under eval (candidate or champion)
                  battle    -> ONE pinned battle champion for the whole batch
  WATCHER         overworld -> navigation CHAMPION
                  battle    -> battle CHAMPION (or verified rule fallback)
  BATTLE_TRAINING battle    -> battle LEARNER, in its own scenario emulators only

Hard rules:
  * the battle LEARNER is only ever used in BATTLE_TRAINING;
  * no hot-swap of the in-battle policy — a version is pinned for the whole
    battle / eval batch / generation;
  * "same version" means same number AND same file hash AND same source AND
    same manifest generation — a re-used number with different bytes is NOT the
    same policy.
"""
from __future__ import annotations

from twoby2 import feature_enabled

SYSTEM_OVERWORLD = "overworld"
SYSTEM_BATTLE = "battle"

POLICY_NAV_CHAMPION = "navigation_champion"
POLICY_NAV_LEARNER = "navigation_learner"
POLICY_NAV_UNDER_EVAL = "navigation_under_eval"
POLICY_BATTLE_CHAMPION = "battle_champion"
POLICY_BATTLE_LEARNER = "battle_learner"
POLICY_RULE_CONTROLLER = "rule_controller"
POLICY_BLOCKED = "blocked"

NAV_TRAINING = "nav_training"
NAV_EVAL = "nav_eval"
WATCHER = "watcher"
BATTLE_TRAINING = "battle_training"
CONSUMER_MODES = (NAV_TRAINING, NAV_EVAL, WATCHER, BATTLE_TRAINING)


class PolicyVersion:
    """Identity of a published policy. Two are 'the same' only if every field
    matches."""

    __slots__ = ("number", "sha256", "source", "manifest_generation", "obs_schema")

    def __init__(self, number=0, sha256="", source="rule", manifest_generation=0,
                 obs_schema="nav_obs_v1"):
        self.number = int(number or 0)
        self.sha256 = str(sha256 or "")
        self.source = source
        self.manifest_generation = int(manifest_generation or 0)
        self.obs_schema = str(obs_schema or "")

    def same_as(self, other):
        return (isinstance(other, PolicyVersion)
                and self.number == other.number
                and self.sha256 == other.sha256
                and self.source == other.source
                and self.manifest_generation == other.manifest_generation
                and self.obs_schema == other.obs_schema)

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((self.number, self.sha256, self.source,
                     self.manifest_generation, self.obs_schema))

    def to_dict(self):
        return {"number": self.number, "sha256": self.sha256,
                "source": self.source,
                "manifest_generation": self.manifest_generation,
                "obs_schema": self.obs_schema}


def resolve_battle_policy(*, battle_champion_valid, battle_champion_source,
                          rule_controller_ready):
    """The in-battle policy for a *consuming* system (not the battle trainer),
    ignoring live gating. ``battle_champion_source`` must be "ppo" for the
    champion to count as a learned policy; "rule" means the published champion
    IS the rule fallback."""
    if battle_champion_valid and battle_champion_source == "ppo":
        return POLICY_BATTLE_CHAMPION
    if rule_controller_ready:
        return POLICY_RULE_CONTROLLER
    return POLICY_BLOCKED


class BattlePolicyRouter:
    """Routes a per-step decision to the right system + a *pinned* battle policy
    according to the consumer mode.

    ``execution_ready`` MUST come from the central snapshot/executor check
    (:func:`twoby2.activation.snapshot_execution_ready`), never from a raw
    caller boolean — :meth:`route` re-verifies it is a real check result.
    """

    def __init__(self, *, consumer_mode, pinned_battle_champion,
                 battle_champion_valid=False, battle_champion_source="rule",
                 rule_controller_ready=True):
        if consumer_mode not in CONSUMER_MODES:
            raise ValueError(f"consumer_mode must be one of {CONSUMER_MODES}")
        self.consumer_mode = consumer_mode
        self.pinned_battle_champion = pinned_battle_champion  # PolicyVersion
        self.battle_champion_valid = bool(battle_champion_valid)
        self.battle_champion_source = battle_champion_source
        self.rule_controller_ready = bool(rule_controller_ready)

    # -- overworld ------------------------------------------------
    def _overworld_policy(self):
        return {
            NAV_TRAINING: POLICY_NAV_LEARNER,
            NAV_EVAL: POLICY_NAV_UNDER_EVAL,
            WATCHER: POLICY_NAV_CHAMPION,
            BATTLE_TRAINING: POLICY_BLOCKED,   # battle trainer has no overworld
        }[self.consumer_mode]

    def route(self, *, in_battle, execution_check):
        """``execution_check`` is the ``(ready: bool, missing: list)`` tuple
        from the central check. Anything else -> not executable."""
        try:
            ready = bool(execution_check[0])
            missing = list(execution_check[1])
        except (TypeError, IndexError, KeyError):
            ready, missing = False, ["execution_check was not a real check result"]

        if not in_battle:
            pol = self._overworld_policy()
            return {"system": SYSTEM_OVERWORLD, "policy": pol,
                    "battle_policy_version": None,
                    "executable": pol != POLICY_BLOCKED and self.consumer_mode != BATTLE_TRAINING,
                    "loads_battle_learner": False, "missing": []}

        if self.consumer_mode == BATTLE_TRAINING:
            # the battle trainer runs the LEARNER, but only inside its own
            # scenario emulators and only when the snapshot is really ready.
            live_ok = feature_enabled("battle_env") and ready
            return {"system": SYSTEM_BATTLE, "policy": POLICY_BATTLE_LEARNER,
                    "battle_policy_version": None,
                    "executable": bool(live_ok),
                    "loads_battle_learner": True, "missing": missing}

        policy = resolve_battle_policy(
            battle_champion_valid=self.battle_champion_valid,
            battle_champion_source=self.battle_champion_source,
            rule_controller_ready=self.rule_controller_ready)
        live_ok = (feature_enabled("battle_router_live") and ready
                   and policy != POLICY_BLOCKED)
        return {
            "system": SYSTEM_BATTLE,
            "policy": policy,
            "battle_policy_version": (
                self.pinned_battle_champion.to_dict()
                if (policy == POLICY_BATTLE_CHAMPION and self.pinned_battle_champion)
                else None),
            "executable": bool(live_ok),
            "loads_battle_learner": False,
            "missing": missing,
        }


class GenerationPin:
    """One navigation generation's frozen view of the battle champion. Adopting
    a newer champion happens ONLY at a generation boundary and only forward."""

    def __init__(self, generation, battle_champion):
        self.generation = int(generation)
        self.battle_champion = battle_champion  # PolicyVersion

    def next_generation(self, available_battle_champion):
        cur = self.battle_champion
        nxt = available_battle_champion
        take = nxt if (nxt and (not cur or nxt.number >= cur.number)) else cur
        return GenerationPin(self.generation + 1, take)

    def to_dict(self):
        return {"generation": self.generation,
                "battle_champion": self.battle_champion.to_dict()
                if self.battle_champion else None}


class EvalBatchPin:
    """One navigation-eval batch pins a single battle champion for the WHOLE
    batch — candidate and champion are scored against the same battle policy."""

    def __init__(self, battle_champion):
        self.battle_champion = battle_champion
        self._locked = True

    def version_for(self, _which):        # 'candidate' or 'champion' — same pin
        return self.battle_champion

    def comparable_with(self, other_batch):
        return (isinstance(other_batch, EvalBatchPin)
                and self.battle_champion is not None
                and self.battle_champion.same_as(other_batch.battle_champion))


class WatcherBattlePin:
    """Battle-champion version the watcher uses, swapped only BETWEEN battles.
    ``observe_available`` records what's newest; ``on_battle_start`` is the only
    place the pin can move, and only when no battle is running."""

    def __init__(self, version=None):
        self.active = version                     # PolicyVersion or None
        self._available = version
        self.in_battle = False

    def observe_available(self, version):
        if version is None:
            return
        if self._available is None or version.number >= self._available.number:
            self._available = version

    def newer_ready(self):
        if self._available is None:
            return False
        if self.active is None:
            return True
        return not self._available.same_as(self.active)

    def on_battle_start(self):
        if self.in_battle:
            return self.active          # guard: mid-battle, never swap
        self.in_battle = True
        if self._available is not None and (
                self.active is None
                or self._available.number >= self.active.number):
            self.active = self._available
        return self.active

    def on_battle_end(self):
        self.in_battle = False

    def loads_learner(self):
        return False
