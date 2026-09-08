"""Adaptive navigation episode-length curriculum.

A monotonically growing *navigation horizon* (max route-steps per navigation
episode). This is NOT ``train.PPO_N_STEPS`` — that stays the PPO rollout length
(512). The horizon only bounds how long a single navigation episode may run
before it is truncated.

Final architecture: every navigation worker starts at the identical canonical
game start. The 80/20 split here is purely episode length — ~80% of workers use
the current rung, ~20% probe the next rung — and probe runs still count as real
beginning-full runs.

Rules:
  * only a genuinely fresh learner with no confirmed champion starts at rung 0
    (2000);
  * an existing trained champion is NEVER forced below its proven horizon on
    migration — the start rung is derived from real data (episode horizon
    actually used, deepest reliably-reached stage, arrival-step distribution,
    existing full-probe metrics); if metadata is thin, fall back to the *old
    live system's effective horizon*, not a coarse stage table;
  * the horizon rises by exactly one rung at a time, never falls automatically;
  * advancing needs >=100 evaluable beginning-full runs, >=80% reproduction of
    every already-mastered transition, passing early-game retention and a
    passing candidate/champion safety check;
  * the horizon state is persisted atomically and never regresses on reload.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

NAV_EPISODE_HORIZONS = [
    2000, 4000, 8000, 12000, 18000, 32768, 49152,
    65536, 98304, 131072, 163840,
]

# The effective FULL-episode horizon the *old* live single-PPO system uses
# (``pokemon_env.PokemonFireRedEnv.LONG_FULL_PROBE_STEPS``). Used as the
# migration floor when a champion's own metadata is too thin to place it
# precisely — we never migrate a proven champion below what it already ran.
OLD_LIVE_EFFECTIVE_HORIZON = 32768

# Advancement thresholds.
MIN_EVALUATED_FULL_RUNS = 100
STAGE_REPRODUCTION_RATE = 0.80
TRANSITION_RATE_FLOOR = 0.80

# Fraction of workers that probe the next rung.
PROBE_FRACTION = 0.20


def _clamp_index(i):
    return max(0, min(len(NAV_EPISODE_HORIZONS) - 1, int(i)))


def horizon_for_index(i):
    return NAV_EPISODE_HORIZONS[_clamp_index(i)]


def index_for_horizon(h):
    """Smallest rung index whose horizon is >= ``h``."""
    for i, v in enumerate(NAV_EPISODE_HORIZONS):
        if v >= int(h or 0):
            return i
    return len(NAV_EPISODE_HORIZONS) - 1


def index_at_or_below_horizon(h):
    """Largest rung index whose horizon is <= ``h`` (>=0)."""
    idx = 0
    for i, v in enumerate(NAV_EPISODE_HORIZONS):
        if v <= int(h or 0):
            idx = i
    return idx


def _arrival_headroom_index(median_arrival_steps):
    """Pick a rung that gives real headroom over the champion's median arrival
    time at its deepest transition: horizon >= ~1.5x median arrival."""
    if not median_arrival_steps or median_arrival_steps <= 0:
        return None
    want = int(median_arrival_steps * 1.5)
    return index_for_horizon(want)


def starting_index_from_champion(champion_metrics):
    """Derive the migration start rung for an EXISTING champion from real data.

    ``champion_metrics`` keys (all optional, best-effort):
      confirmed              : bool  — is this a real trained champion?
      episode_horizon_used   : int   — the route-step budget its episodes ran on
      effective_live_horizon : int   — the live system's FULL horizon (fallback)
      deepest_reliable_stage : int
      median_arrival_steps   : int   — median steps to its deepest transition
      full_probe_horizon     : int   — any long-probe horizon it was evaluated at
      horizon_index_used     : int   — explicit prior rung, if we stored one

    A fresh / unconfirmed learner -> rung 0. A confirmed champion -> the MAX of
    every signal we have, but never below the old live effective horizon.
    """
    m = champion_metrics or {}
    if not m.get("confirmed"):
        return 0

    candidates = []
    if isinstance(m.get("horizon_index_used"), int):
        candidates.append(_clamp_index(m["horizon_index_used"]))
    for key in ("episode_horizon_used", "full_probe_horizon"):
        if m.get(key):
            candidates.append(index_at_or_below_horizon(m[key]))
    ah = _arrival_headroom_index(m.get("median_arrival_steps"))
    if ah is not None:
        candidates.append(ah)

    # Fallback fl0or: the horizon the old live system actually uses.
    live_floor = index_at_or_below_horizon(
        m.get("effective_live_horizon") or OLD_LIVE_EFFECTIVE_HORIZON)
    candidates.append(live_floor)

    return _clamp_index(max(candidates)) if candidates else live_floor


class NavHorizonState:
    """Owned by the navigation system only. Persisted atomically as JSON."""

    SCHEMA = "nav_horizon_v2"

    def __init__(self, index=0):
        self.current_horizon_index = _clamp_index(index)
        self.current_horizon_steps = 0
        self.advancement_reason = "init"
        self.evaluated_full_runs = 0
        self.transition_rates = {}          # {src_stage: rate}
        self.next_probe_count = 0
        self.timestamp = _now()
        self.navigation_champion_version = None
        self.battle_champion_version = None
        self.generation = 0
        self._history = []

    # -- construction ---------------------------------------------------
    @classmethod
    def for_fresh_learner(cls):
        st = cls(0)
        st.advancement_reason = "fresh_learner_smallest_rung"
        return st

    @classmethod
    def from_champion(cls, champion_metrics):
        st = cls(starting_index_from_champion(champion_metrics))
        m = champion_metrics or {}
        st.advancement_reason = (
            f"champion_derived_rung_{st.current_horizon_index}"
            f"_h{st.current_horizon}" if m.get("confirmed")
            else "unconfirmed_champion_smallest_rung")
        st.navigation_champion_version = m.get("version")
        return st

    # -- queries ------------------------------------------------------
    @property
    def current_horizon(self):
        return horizon_for_index(self.current_horizon_index)

    @property
    def next_horizon(self):
        return horizon_for_index(self.current_horizon_index + 1)

    @property
    def at_top(self):
        return self.current_horizon_index >= len(NAV_EPISODE_HORIZONS) - 1

    # -- worker split -----------------------------------------------
    def assign_worker_horizons(self, worker_count):
        """One role dict per navigation worker. ~80% current rung, ~20% probe
        the next rung, always >= 1 probe unless at the top. EVERY role is a
        canonical-start beginning-full run."""
        n = max(1, int(worker_count))
        if self.at_top:
            probes = 0
        elif n == 1:
            probes = 1
        else:
            probes = max(1, round(n * PROBE_FRACTION))
            probes = min(probes, n - 1)
        self.next_probe_count = probes
        roles = []
        for i in range(n):
            probe = i < probes
            roles.append({
                "start_kind": "beginning",           # never anything else
                "canonical_start": True,
                "probe": probe,
                "counts_as_beginning_full_run": True,
                "horizon": self.next_horizon if (probe and not self.at_top)
                else self.current_horizon,
                "rung": "next" if (probe and not self.at_top) else "current",
            })
        return roles

    # -- advancement ------------------------------------------------
    def evaluate_advancement(self, *, evaluated_full_runs, stage_reproduction_rate,
                             transition_rates, retention_passed,
                             candidate_passed):
        """Pure check — does NOT mutate. Returns (ok, reasons)."""
        reasons = []
        if self.at_top:
            return False, ["already at the deepest horizon rung"]
        if int(evaluated_full_runs or 0) < MIN_EVALUATED_FULL_RUNS:
            reasons.append(
                f"{int(evaluated_full_runs or 0)} evaluated full runs "
                f"< {MIN_EVALUATED_FULL_RUNS}")
        if float(stage_reproduction_rate or 0) < STAGE_REPRODUCTION_RATE:
            reasons.append(
                f"stage reproduction {float(stage_reproduction_rate or 0):.0%} "
                f"< {STAGE_REPRODUCTION_RATE:.0%}")
        weak = sorted(
            (str(s), round(float(r), 3))
            for s, r in (transition_rates or {}).items()
            if float(r) < TRANSITION_RATE_FLOOR)
        if weak:
            reasons.append(f"transitions below {TRANSITION_RATE_FLOOR:.0%}: {weak}")
        if not retention_passed:
            reasons.append("early-game retention regressed")
        if not candidate_passed:
            reasons.append("candidate failed the champion safety check")
        return (not reasons), reasons

    def advance_if_ready(self, *, evaluated_full_runs, stage_reproduction_rate,
                         transition_rates, retention_passed, candidate_passed,
                         navigation_champion_version=None,
                         battle_champion_version=None):
        """Advance exactly one rung when every gate passes. Never lowers the
        rung. Returns a result dict."""
        ok, reasons = self.evaluate_advancement(
            evaluated_full_runs=evaluated_full_runs,
            stage_reproduction_rate=stage_reproduction_rate,
            transition_rates=transition_rates,
            retention_passed=retention_passed,
            candidate_passed=candidate_passed)
        self.evaluated_full_runs = int(evaluated_full_runs or 0)
        self.transition_rates = {str(k): round(float(v), 4)
                                 for k, v in (transition_rates or {}).items()}
        if navigation_champion_version is not None:
            self.navigation_champion_version = navigation_champion_version
        if battle_champion_version is not None:
            self.battle_champion_version = battle_champion_version
        self.timestamp = _now()
        if not ok:
            return {"advanced": False, "reasons": reasons,
                    "horizon_index": self.current_horizon_index,
                    "horizon": self.current_horizon}
        prev = self.current_horizon_index
        self.current_horizon_index = _clamp_index(prev + 1)
        self.current_horizon_steps = 0
        self.generation += 1
        self.advancement_reason = (
            f"advanced {NAV_EPISODE_HORIZONS[prev]}->{self.current_horizon} "
            f"(runs={self.evaluated_full_runs}, "
            f"repro={float(stage_reproduction_rate):.0%})")
        self._history.append({
            "from_index": prev, "to_index": self.current_horizon_index,
            "reason": self.advancement_reason, "timestamp": self.timestamp,
            "generation": self.generation,
        })
        return {"advanced": True, "reasons": [],
                "horizon_index": self.current_horizon_index,
                "horizon": self.current_horizon,
                "advancement_reason": self.advancement_reason}

    def add_steps(self, n):
        self.current_horizon_steps += max(0, int(n))

    # -- serialization --------------------------------------------
    def to_dict(self):
        return {
            "schema": self.SCHEMA,
            "current_horizon_index": self.current_horizon_index,
            "current_horizon": self.current_horizon,
            "current_horizon_steps": self.current_horizon_steps,
            "advancement_reason": self.advancement_reason,
            "evaluated_full_runs": self.evaluated_full_runs,
            "transition_rates": dict(self.transition_rates),
            "next_probe_count": self.next_probe_count,
            "timestamp": self.timestamp,
            "generation": self.generation,
            "navigation_champion_version": self.navigation_champion_version,
            "battle_champion_version": self.battle_champion_version,
            "history": list(self._history),
        }

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        st = cls(int(d.get("current_horizon_index", 0) or 0))
        st.current_horizon_steps = int(d.get("current_horizon_steps", 0) or 0)
        st.advancement_reason = d.get("advancement_reason", "loaded")
        st.evaluated_full_runs = int(d.get("evaluated_full_runs", 0) or 0)
        st.transition_rates = dict(d.get("transition_rates", {}) or {})
        st.next_probe_count = int(d.get("next_probe_count", 0) or 0)
        st.timestamp = d.get("timestamp", _now())
        st.generation = int(d.get("generation", 0) or 0)
        st.navigation_champion_version = d.get("navigation_champion_version")
        st.battle_champion_version = d.get("battle_champion_version")
        st._history = list(d.get("history", []) or [])
        return st

    def merge_persisted(self, other_dict):
        """When re-loading, keep the DEEPER rung — the horizon must never
        regress just because a stale file was read."""
        other = NavHorizonState.from_dict(other_dict)
        if other.current_horizon_index > self.current_horizon_index:
            self.current_horizon_index = other.current_horizon_index
            self.advancement_reason = "merged_persisted_deeper_rung"
            self.generation = max(self.generation, other.generation)
        return self

    # -- atomic persistence --------------------------------------
    def save_atomic(self, path):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp.json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"), sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load_or_new(cls, path, *, champion_metrics=None):
        """Load a persisted state; if absent, derive from champion metrics (or
        a fresh learner). A loaded state that is somehow *shallower* than the
        champion-derived rung is bumped up — never down."""
        derived = (cls.from_champion(champion_metrics) if champion_metrics
                   else cls.for_fresh_learner())
        if os.path.exists(path):
            try:
                with open(path) as f:
                    derived.merge_persisted(json.load(f) or {})
            except (OSError, ValueError):
                pass
        return derived


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
