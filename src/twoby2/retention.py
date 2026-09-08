"""Early-game retention gates (2×2, post-parcel).

Every navigation evaluation in the final architecture starts at the canonical
**post-parcel master** (`StartGame.state`, in Oak's lab, starter already in the
party). Anything that happens *before* that state — the title cutscene, the
stairs, leaving the house, obtaining the starter — is `precanonical` and can
NEVER block a promotion; its old metrics are kept for history only.

The active, ordered retention chain begins at leaving Oak's lab:

  1. leave_oak_lab   — Oak's lab -> Pallet Town overworld (post-parcel)
  2. pallet_route1   — Pallet Town -> Route 1            (stage 1 -> 2)
  3. route1_viridian — Route 1 -> Viridian City          (stage 2 -> 3)
  4. ... then every further reliably-mastered transition
"""
from __future__ import annotations

# precanonical — before the master state; NEVER a promotion gate.
PRECANONICAL_TRANSITIONS = ("intro", "stairs_down", "left_house", "starter")

# the active, ordered retention chain (post-parcel).
ACTIVE_EARLY_TRANSITIONS = (
    "leave_oak_lab",      # Oak's lab -> Pallet overworld (first post-master step)
    "pallet_route1",      # Pallet Town -> Route 1        (stage 1 -> 2)
    "route1_viridian",    # Route 1 -> Viridian City      (stage 2 -> 3)
)

# Back-compat name. Old callers importing EARLY_TRANSITIONS now get the ACTIVE
# chain only (precanonical entries are gone from the gate list on purpose).
EARLY_TRANSITIONS = ACTIVE_EARLY_TRANSITIONS

# transition key -> curriculum_v20 source stage (None = no numbered stage).
TRANSITION_SOURCE_STAGE = {
    "leave_oak_lab": None, "pallet_route1": 1, "route1_viridian": 2,
    # precanonical, kept for lookup:
    "intro": None, "stairs_down": None, "left_house": None, "starter": None,
}


def classify_transition(key):
    """'precanonical' | 'active' | 'later' — 'precanonical' never gates."""
    if key in PRECANONICAL_TRANSITIONS:
        return "precanonical"
    if key in ACTIVE_EARLY_TRANSITIONS:
        return "active"
    return "later"

# Defaults. A candidate rate may dip by at most ABS below the champion, and may
# not retain less than REL of the champion's rate, before it counts as a
# regression on that gate.
DEFAULT_MIN_SAMPLE = 100
DEFAULT_ABS_DROP = 0.10       # 10 percentage points (rates are 0..1)
DEFAULT_REL_RETENTION = 0.85  # keep >= 85% of the champion's rate
# how many individual gate regressions make a "hard" retention failure
HARD_REGRESSION_COUNT = 2


class RetentionGate:
    __slots__ = ("key", "min_sample", "abs_drop", "rel_retention")

    def __init__(self, key, *, min_sample=DEFAULT_MIN_SAMPLE,
                 abs_drop=DEFAULT_ABS_DROP, rel_retention=DEFAULT_REL_RETENTION):
        self.key = key
        self.min_sample = int(min_sample)
        self.abs_drop = float(abs_drop)
        self.rel_retention = float(rel_retention)

    def evaluate(self, champion_rate, candidate_rate, candidate_sample):
        """Return a per-gate result dict.

        Fail-closed: an under-powered candidate sample is a regression (we
        cannot prove the skill is retained).
        """
        cr = _rate(champion_rate)
        dr = _rate(candidate_rate)
        n = int(candidate_sample or 0)
        under_sampled = n < self.min_sample
        abs_regressed = dr < cr - self.abs_drop
        rel_regressed = cr > 0 and dr < cr * self.rel_retention
        regressed = under_sampled or abs_regressed or rel_regressed
        reasons = []
        if under_sampled:
            reasons.append(f"sample {n} < {self.min_sample}")
        if abs_regressed:
            reasons.append(f"rate {dr:.3f} < champ {cr:.3f} - {self.abs_drop:.2f}")
        if rel_regressed:
            reasons.append(f"rate {dr:.3f} < {self.rel_retention:.0%} of champ {cr:.3f}")
        return {
            "gate": self.key, "champion_rate": cr, "candidate_rate": dr,
            "candidate_sample": n, "regressed": regressed,
            "reasons": reasons,
        }


def _rate(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:                 # tolerate permille inputs
        f = f / 1000.0
    return max(0.0, min(1.0, f))


def default_gates(mastered_keys=ACTIVE_EARLY_TRANSITIONS):
    """One :class:`RetentionGate` per mastered ACTIVE transition, in order.

    Precanonical keys are dropped: they are before the master state and can
    never gate a promotion. ``leave_oak_lab`` / ``pallet_route1`` /
    ``route1_viridian`` come first, then any further mastered transition in the
    order given.
    """
    seen = []
    for k in list(ACTIVE_EARLY_TRANSITIONS) + [k for k in mastered_keys
                                               if k not in ACTIVE_EARLY_TRANSITIONS]:
        if k in mastered_keys and k not in PRECANONICAL_TRANSITIONS and k not in seen:
            seen.append(k)
    return [RetentionGate(k) for k in seen]


def evaluate_retention(champion_rates, candidate_rates, candidate_samples,
                       mastered_keys=ACTIVE_EARLY_TRANSITIONS, gates=None):
    """Aggregate retention decision.

    Precanonical transitions never appear as a gate, so missing intro / starter
    metrics cannot block a promotion. ``*_rates`` / ``candidate_samples`` are
    ``{transition_key: value}``.
    """
    gates = gates if gates is not None else default_gates(mastered_keys)
    details = []
    for g in gates:
        details.append(g.evaluate(
            (champion_rates or {}).get(g.key),
            (candidate_rates or {}).get(g.key),
            (candidate_samples or {}).get(g.key)))
    regressions = [d for d in details if d["regressed"]]
    precanonical_ignored = [k for k in (champion_rates or {})
                            if k in PRECANONICAL_TRANSITIONS]
    return {
        "passed": not regressions,
        "hard_failure": len(regressions) >= HARD_REGRESSION_COUNT,
        "regression_count": len(regressions),
        "regressions": [d["gate"] for d in regressions],
        "precanonical_ignored": sorted(precanonical_ignored),
        "details": details,
    }
