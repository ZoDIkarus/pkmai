"""
V20 curriculum architecture.

Design goals (see project brief V20):

  * Separate ``discovered_stage`` (deepest any agent ever reached) from
    ``mastered_stage`` (deepest transition the shared PPO policy reproduces
    reliably AND that Full-from-start runs confirm).
  * Track every world-stage transition (1->2 .. N-1->N) with rolling
    statistics, so a single lucky scout can never mark a stage "learned".
  * Expose the earliest transition the Full policy cannot reproduce as
    ``current_bottleneck`` and concentrate BRIDGE training there.
  * Four mutually exclusive training modes on the SAME PPO policy:
        FULL      - from the real StartGame, chains everything
        BRIDGE    - from the bottleneck stage entry, learns the next hop
        FRONTIER  - from the deepest discovered frontier, discovers the next hop
        RETENTION - rotates through already mastered transitions
    plus the existing POST_WIPE_RECOVERY handled inside the env.
  * A generic story-objective representation that extends to the whole game
    (reach_map / reach_transition / enter_required_building / heal_center /
    win_trainer / win_gym / obtain_badge / obtain_item / trigger_story_flag)
    without hardcoding the FireRed storyline now.

This module is pure Python and fully unit tested; the env and trainer only
read/write the JSON state file it manages.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import deque

# --------------------------------------------------------------------------
# Tunables (brief section 2 / 3)
# --------------------------------------------------------------------------
TRANSITION_MASTERY_WINDOW = 50
TRANSITION_MASTERY_MIN_ATTEMPTS = 20
TRANSITION_MASTERY_RATE = 0.80
FULL_CHAIN_CONFIRMATIONS = 5

# Training modes.  POST_WIPE_RECOVERY is intentionally NOT allocated here - it
# is entered dynamically by the env when a party wipe happens and overrides the
# rank's normal mode for the duration of the recovery.
MODE_FULL = "FULL"
MODE_BRIDGE = "BRIDGE"
MODE_FRONTIER = "FRONTIER"
MODE_RETENTION = "RETENTION"
MODE_POST_WIPE_RECOVERY = "POST_WIPE_RECOVERY"

ALL_MODES = (MODE_FULL, MODE_BRIDGE, MODE_FRONTIER, MODE_RETENTION)

# Reference allocation from the brief for ~33 envs.  Applied as ratios so it
# scales to any fleet size.
_ALLOC_RATIO = {
    MODE_FULL: 12.0 / 33.0,
    MODE_BRIDGE: 12.0 / 33.0,
    MODE_FRONTIER: 6.0 / 33.0,
    MODE_RETENTION: 3.0 / 33.0,
}

# --------------------------------------------------------------------------
# Geography <-> stage.  Kept in lockstep with PokemonFireRedEnv.WORLD_STAGE_BY_MAP
# but owned here so future stages only need one edit.
# --------------------------------------------------------------------------
WORLD_STAGES = {
    1: "Pallet",
    2: "Route1",
    3: "Viridian",
    4: "Route2",
    5: "Forest",
    6: "Pewter",
}
MAX_KNOWN_STAGE = max(WORLD_STAGES)


def transition_name(src_stage: int) -> str:
    a = WORLD_STAGES.get(int(src_stage), f"S{int(src_stage)}")
    b = WORLD_STAGES.get(int(src_stage) + 1, f"S{int(src_stage) + 1}")
    return f"{a}->{b}"


# --------------------------------------------------------------------------
# Story objective representation (brief section 19).
# --------------------------------------------------------------------------
STORY_OBJECTIVE_KINDS = frozenset({
    "reach_map",
    "reach_transition",
    "enter_required_building",
    "heal_center",
    "win_trainer",
    "win_gym",
    "obtain_badge",
    "obtain_item",
    "trigger_story_flag",
})


class Objective:
    """A single, checkable story objective. ``world_stage`` (geography) and the
    objective itself (story progress) are deliberately independent."""

    __slots__ = ("kind", "world_stage", "params", "name")

    def __init__(self, kind, world_stage, name=None, **params):
        if kind not in STORY_OBJECTIVE_KINDS:
            raise ValueError(f"unknown objective kind: {kind!r}")
        self.kind = kind
        self.world_stage = int(world_stage)
        self.params = dict(params)
        self.name = name or f"{kind}@{self.world_stage}"

    def to_dict(self):
        return {
            "kind": self.kind,
            "world_stage": self.world_stage,
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["world_stage"], name=d.get("name"),
                   **(d.get("params") or {}))

    def __repr__(self):
        return f"Objective({self.name!r}, stage={self.world_stage})"


# The current immediate chain, expressed generically.  Extend this list (not the
# code) as scouts confirm later maps.
STORY_OBJECTIVES = [
    Objective("reach_transition", 1, name="reach_route1", to_stage=2),
    Objective("reach_transition", 2, name="reach_viridian", to_stage=3),
    Objective("heal_center", 3, name="heal_viridian", city="Viridian"),
    Objective("reach_transition", 3, name="reach_route2", to_stage=4),
    Objective("reach_transition", 4, name="reach_forest", to_stage=5),
    Objective("reach_transition", 5, name="reach_pewter", to_stage=6),
    Objective("heal_center", 6, name="heal_pewter", city="Pewter"),
    Objective("enter_required_building", 6, name="enter_pewter_gym",
              building="pewter_gym"),
    Objective("win_gym", 6, name="beat_brock", gym="pewter"),
    Objective("obtain_badge", 6, name="boulder_badge", badge=1),
]


def objectives_for_stage(stage):
    return [o for o in STORY_OBJECTIVES if o.world_stage == int(stage)]


# --------------------------------------------------------------------------
# Transition statistics
# --------------------------------------------------------------------------
class TransitionRecord:
    """Rolling success statistics for one world-stage transition."""

    def __init__(self, src_stage, window=TRANSITION_MASTERY_WINDOW):
        self.src_stage = int(src_stage)
        self.window = int(window)
        self.attempts = 0
        self.successes = 0
        self.full_chain_confirmations = 0
        self.recent_results = deque(maxlen=self.window)

    # -- recording -------------------------------------------------------
    def record(self, success, full_chain=False):
        success = bool(success)
        self.attempts += 1
        if success:
            self.successes += 1
        self.recent_results.append(1 if success else 0)
        if full_chain and success:
            self.full_chain_confirmations += 1

    # -- queries --------------------------------------------------------
    @property
    def window_attempts(self):
        return len(self.recent_results)

    @property
    def success_rate(self):
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def lifetime_rate(self):
        if self.attempts <= 0:
            return 0.0
        return self.successes / self.attempts

    def is_mastered(self):
        return (
            self.window_attempts >= TRANSITION_MASTERY_MIN_ATTEMPTS
            and self.success_rate >= TRANSITION_MASTERY_RATE
            and self.full_chain_confirmations >= FULL_CHAIN_CONFIRMATIONS
        )

    # -- serialization ------------------------------------------------
    def to_dict(self):
        return {
            "src_stage": self.src_stage,
            "attempts": self.attempts,
            "successes": self.successes,
            "full_chain_confirmations": self.full_chain_confirmations,
            "recent_results": list(self.recent_results),
        }

    @classmethod
    def from_dict(cls, d):
        rec = cls(d.get("src_stage", 0))
        rec.attempts = int(d.get("attempts", 0))
        rec.successes = int(d.get("successes", 0))
        rec.full_chain_confirmations = int(d.get("full_chain_confirmations", 0))
        for r in d.get("recent_results", [])[-rec.window:]:
            rec.recent_results.append(1 if r else 0)
        return rec


# --------------------------------------------------------------------------
# Curriculum state
# --------------------------------------------------------------------------
class CurriculumState:
    """Owns discovered/mastered stage tracking + transition statistics.

    Persisted as a small JSON file so every SubprocVecEnv worker and the
    trainer callback share one view.
    """

    def __init__(self):
        self.discovered_stage = 1
        self.transitions = {}  # src_stage -> TransitionRecord
        # rotating pointer used by RETENTION allocation
        self.retention_cursor = 0

    # -- transition access -------------------------------------------
    def _rec(self, src_stage):
        src_stage = int(src_stage)
        rec = self.transitions.get(src_stage)
        if rec is None:
            rec = TransitionRecord(src_stage)
            self.transitions[src_stage] = rec
        return rec

    # -- recording --------------------------------------------------
    def record_discovery(self, stage):
        stage = int(stage)
        if stage > self.discovered_stage:
            self.discovered_stage = stage

    def record_transition_attempt(self, src_stage, success, full_chain=False):
        """One BRIDGE/FULL attempt at transition src_stage -> src_stage+1."""
        self._rec(src_stage).record(success, full_chain=full_chain)
        if success:
            self.record_discovery(int(src_stage) + 1)

    def record_full_chain_result(self, reached_stage):
        """A Full-from-start episode ended having reached ``reached_stage``.

        Every transition below it counts as an in-chain success; the first
        transition it failed to cross counts as an in-chain failure.  This is
        the ONLY path that increments ``full_chain_confirmations`` in bulk and
        therefore the only path that can move ``mastered_stage``.
        """
        reached_stage = int(reached_stage)
        self.record_discovery(reached_stage)
        for src in range(1, reached_stage):
            self._rec(src).record(True, full_chain=True)
        # the hop it could not make (if it is a known part of the world)
        if reached_stage < MAX_KNOWN_STAGE and reached_stage >= 1:
            self._rec(reached_stage).record(False, full_chain=True)

    # -- derived ----------------------------------------------------
    @property
    def mastered_stage(self):
        """Deepest stage reachable by a contiguous run of mastered transitions
        from stage 1."""
        stage = 1
        while stage < MAX_KNOWN_STAGE:
            rec = self.transitions.get(stage)
            if rec is None or not rec.is_mastered():
                break
            stage += 1
        return stage

    @property
    def current_bottleneck(self):
        """Earliest transition (its source stage) that the Full policy does not
        yet reproduce reliably.  Restricted to transitions whose destination has
        actually been discovered - discovering the next hop is FRONTIER's job,
        not BRIDGE's.  Returns ``None`` when everything discovered is mastered.
        """
        limit = max(1, self.discovered_stage)
        for src in range(1, limit):
            rec = self.transitions.get(src)
            if rec is None or not rec.is_mastered():
                return src
        return None

    def frontier_stage(self):
        """Stage FRONTIER agents work from: the deepest discovered stage, but
        never racing more than one hop past what the chain can reach."""
        return max(self.mastered_stage, min(self.discovered_stage,
                                            self.mastered_stage + 2))

    def mastered_transitions(self):
        return [s for s in range(1, MAX_KNOWN_STAGE)
                if (self.transitions.get(s) is not None
                    and self.transitions[s].is_mastered())]

    # -- serialization --------------------------------------------
    def to_dict(self):
        return {
            "schema": "curriculum_v20",
            "discovered_stage": self.discovered_stage,
            "mastered_stage": self.mastered_stage,   # derived, stored for readers
            "current_bottleneck": self.current_bottleneck,
            "retention_cursor": self.retention_cursor,
            "transitions": {str(k): v.to_dict()
                            for k, v in sorted(self.transitions.items())},
        }

    @classmethod
    def from_dict(cls, d):
        st = cls()
        st.discovered_stage = int(d.get("discovered_stage", 1) or 1)
        st.retention_cursor = int(d.get("retention_cursor", 0) or 0)
        for k, v in (d.get("transitions") or {}).items():
            st.transitions[int(k)] = TransitionRecord.from_dict(v)
        return st

    # -- disk -----------------------------------------------------
    def save(self, path):
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   prefix=".curr_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        try:
            with open(str(path)) as f:
                return cls.from_dict(json.load(f) or {})
        except (OSError, ValueError):
            return cls()


# --------------------------------------------------------------------------
# Mode allocation (brief section 3 / 17)
# --------------------------------------------------------------------------
def allocate_modes(n_envs):
    """Deterministic rank -> mode assignment, scaled from the 12/12/6/3 ratio.

    Guarantees: at least one FULL always; BRIDGE/FRONTIER/RETENTION each get at
    least one slot once the fleet is large enough (>= 4 / 8 / 12 envs); any
    rounding remainder goes to FULL (the champion-measured objective).
    """
    n = max(1, int(n_envs))
    if n == 1:
        return [MODE_FULL]

    counts = {m: int(n * _ALLOC_RATIO[m]) for m in ALL_MODES}

    # Minimum viable coverage.
    counts[MODE_BRIDGE] = max(counts[MODE_BRIDGE], 1 if n >= 4 else 0)
    counts[MODE_FRONTIER] = max(counts[MODE_FRONTIER], 1 if n >= 8 else 0)
    counts[MODE_RETENTION] = max(counts[MODE_RETENTION], 1 if n >= 12 else 0)
    counts[MODE_FULL] = max(counts[MODE_FULL], 1)

    # Trim if we over-allocated small fleets (take from RETENTION, then FRONTIER,
    # then BRIDGE - never drop FULL below 1).
    overflow = sum(counts.values()) - n
    for m in (MODE_RETENTION, MODE_FRONTIER, MODE_BRIDGE, MODE_FULL):
        while overflow > 0 and counts[m] > (1 if m == MODE_FULL else 0):
            counts[m] -= 1
            overflow -= 1

    # Remainder -> FULL.
    counts[MODE_FULL] += max(0, n - sum(counts.values()))

    layout = []
    for m in (MODE_FULL, MODE_BRIDGE, MODE_FRONTIER, MODE_RETENTION):
        layout.extend([m] * counts[m])
    layout = layout[:n]
    while len(layout) < n:
        layout.append(MODE_FULL)
    return layout


def mode_for_rank(rank, n_envs):
    layout = allocate_modes(n_envs)
    return layout[int(rank) % len(layout)]


def allocation_summary(n_envs):
    layout = allocate_modes(n_envs)
    return {m: layout.count(m) for m in ALL_MODES}
