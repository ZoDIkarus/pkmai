"""Navigation champion promotion policy (2×2, final architecture).

Every evaluation run starts at the identical canonical game start — there are
NO anchor runs in the final architecture. All promotion-eligible runs are
``start_kind == "beginning"``. Any ``anchor_runs`` passed in are rejected and
counted, never mixed into the metrics. A single lucky deep run is not evidence;
a deeper max stage does not beat a candidate that has forgotten the early game
(retention, :mod:`twoby2.retention`, is a veto). Candidate and champion must
have been evaluated against the same battle-champion version (or both
re-baselined) to be comparable.
"""
from __future__ import annotations

from twoby2.retention import evaluate_retention, EARLY_TRANSITIONS

# A candidate needs at least this many completed, evaluable beginning-full runs
# on the fixed reproducible suite before it can be judged at all.
MIN_BEGINNING_FULL_RUNS = 100
# "reliably reproduces the current stage" threshold (fraction of beginning runs
# that reach the champion's established max stage).
STAGE_REPRO_RATE = 0.80
# a relevant onward transition must be crossed at least this often
TRANSITION_MIN_RATE = 0.80
# depth improvement that, on its own, is treated as "one lucky run" unless the
# reproduction rate also clears STAGE_REPRO_RATE
LUCKY_RUN_GUARD_RATE = 0.10


class PromotionDecision(dict):
    @property
    def promote(self):
        return bool(self.get("promote"))


def _beginning_only(runs):
    return [r for r in runs or [] if r.get("start_kind") == "beginning"
            and r.get("completed") and r.get("evaluable", True)]


def _reach_rate(runs, stage):
    if not runs:
        return 0.0
    return sum(1 for r in runs if int(r.get("max_stage", 0)) >= stage) / len(runs)


def _battle_versions_comparable(champion, candidate):
    """Candidate & champion navigation evals are only comparable if they used
    the same battle-champion version (number + sha), or both were re-baselined
    together (same ``rebaselined_batch`` token)."""
    cv = champion.get("battle_champion_version")
    dv = candidate.get("battle_champion_version")
    cs = champion.get("battle_champion_sha256")
    ds = candidate.get("battle_champion_sha256")
    if champion.get("rebaselined_batch") and \
            champion.get("rebaselined_batch") == candidate.get("rebaselined_batch"):
        return True
    if cv is None and dv is None:
        return True                       # neither pinned (e.g. rule fallback only)
    return cv == dv and cs == ds


def nav_hard_regression_reset_scope():
    """A navigation hard regression may reset ONLY the navigation learner."""
    return {
        "resets": ["navigation_learner <- navigation_champion"],
        "never_touches": ["battle_learner", "battle_champion", "curriculum",
                          "savestates", "exploration_memory", "scenario_pool"],
    }


# NOT WIRED INTO THE LIVE TRAINER. The single navigation-promotion authority is
# ``train.Trainer._score`` / ``_evaluate`` (geographic depth + reproducible
# story chain; never level / XP / KO / battle reward). This module is retained
# only as a reference design and for its unit tests - importing it into the
# live path is a bug (test_nav_route1_signal asserts train.py does not).
_LIVE_AUTHORITY = "train._score"


def evaluate_promotion(*, champion, candidate, mastered_early_keys=EARLY_TRANSITIONS):
    """Decide whether ``candidate`` may replace ``champion`` as nav champion.

    NOTE: this function is NOT called by the live trainer - see the module note.

    Both args are dicts:
      champion  = {"max_stage": int,
                   "early_rates": {key: rate},        # baseline
                   "version": int}
      candidate = {"beginning_runs": [ {start_kind, completed, evaluable,
                                        max_stage, transitions:{src:rate}}... ],
                   "anchor_runs":    [ ... ],          # informational only
                   "early_rates": {key: rate},
                   "early_samples": {key: n},
                   "version": int}
    """
    from twoby2.nav_progress import strip_level_from_promotion_metrics
    # navigation promotion NEVER sees level / XP / KO / battle-win signals.
    champion = strip_level_from_promotion_metrics(champion or {})
    candidate = {**strip_level_from_promotion_metrics(candidate or {}),
                 # keep the run lists (they are geographic) but scrub each run
                 "beginning_runs": [
                     strip_level_from_promotion_metrics(r)
                     for r in (candidate or {}).get("beginning_runs") or []],
                 "anchor_runs": (candidate or {}).get("anchor_runs") or []}
    reasons = []

    begin = _beginning_only(candidate.get("beginning_runs"))
    anchor_runs = candidate.get("anchor_runs") or []
    n_begin = len(begin)

    # 1) anchor runs do not exist in the final architecture — reject + count,
    #    never fold into the metrics.
    used_anchor_for_eval = False
    if anchor_runs:
        reasons.append(f"rejected {len(anchor_runs)} anchor runs "
                       f"(not part of the final architecture)")

    # 1b) comparability: same battle-champion version, or both re-baselined
    if not _battle_versions_comparable(champion, candidate):
        return PromotionDecision(
            promote=False,
            reason="candidate and champion used different battle-champion "
                   "versions and were not re-baselined together",
            beginning_full_runs=n_begin, anchor_runs=len(anchor_runs),
            used_anchor_for_eval=False, retention=None,
            comparable=False)

    if n_begin < MIN_BEGINNING_FULL_RUNS:
        reasons.append(
            f"only {n_begin} evaluable beginning-full runs "
            f"(need {MIN_BEGINNING_FULL_RUNS})")
        return PromotionDecision(
            promote=False, reason="; ".join(reasons),
            beginning_full_runs=n_begin, anchor_runs=len(anchor_runs),
            used_anchor_for_eval=used_anchor_for_eval, retention=None)

    champ_stage = int(champion.get("max_stage", 0))
    cand_stage = max((int(r.get("max_stage", 0)) for r in begin), default=0)

    # 2) reproduction of the *champion's* established stage (or deeper)
    repro_stage = max(1, champ_stage)
    repro_rate = _reach_rate(begin, repro_stage)
    reproduces_current = repro_rate >= STAGE_REPRO_RATE
    if not reproduces_current:
        reasons.append(
            f"reproduces stage {repro_stage} only {repro_rate:.0%} "
            f"(need {STAGE_REPRO_RATE:.0%})")

    # 3) single-lucky-deep-run guard
    deeper = cand_stage > champ_stage
    if deeper:
        deep_rate = _reach_rate(begin, cand_stage)
        if deep_rate < LUCKY_RUN_GUARD_RATE:
            reasons.append(
                f"deepest stage {cand_stage} reached only {deep_rate:.0%} "
                f"of runs — treated as noise, not progress")
            deeper = False  # do not credit the depth

    # 4) onward transitions from beginning runs
    trans_acc = {}
    trans_cnt = {}
    for r in begin:
        for src, rate in (r.get("transitions") or {}).items():
            trans_acc[str(src)] = trans_acc.get(str(src), 0.0) + _f(rate)
            trans_cnt[str(src)] = trans_cnt.get(str(src), 0) + 1
    weak_transitions = []
    for src, total in trans_acc.items():
        mean = total / max(1, trans_cnt[src])
        if mean < TRANSITION_MIN_RATE:
            weak_transitions.append((src, round(mean, 3)))
    if weak_transitions:
        reasons.append(f"weak onward transitions: {sorted(weak_transitions)}")

    # 5) retention veto (never mix in anchor runs here)
    retention = evaluate_retention(
        champion.get("early_rates"), candidate.get("early_rates"),
        candidate.get("early_samples"), mastered_keys=mastered_early_keys)
    if not retention["passed"]:
        reasons.append(f"early-game retention regressions: {retention['regressions']}")

    promote = (reproduces_current and not weak_transitions
               and retention["passed"]
               and (deeper or _score_better(candidate, champion, begin)))

    if promote and not deeper and not reasons:
        reasons.append("equal depth, retained early game, improved reach/speed")

    return PromotionDecision(
        promote=bool(promote),
        reason="; ".join(reasons) or "meets all promotion gates",
        beginning_full_runs=n_begin, anchor_runs=len(anchor_runs),
        used_anchor_for_eval=used_anchor_for_eval,
        candidate_max_stage=cand_stage, champion_max_stage=champ_stage,
        reproduces_current_stage=reproduces_current,
        repro_rate=round(repro_rate, 3),
        retention=retention)


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f / 1000.0 if f > 1.0 else f


def _score_better(candidate, champion, begin_runs):
    """Tie-break at equal depth: better reach rate at the shared stage, or a
    faster median arrival. Conservative — defaults to False."""
    stage = max(1, int(champion.get("max_stage", 0)))
    cand_reach = _reach_rate(begin_runs, stage)
    champ_reach = _f(champion.get("early_rates", {}).get("stage_reach", 0)) or \
        champion.get("stage_reach_rate", 0.0)
    try:
        champ_reach = float(champ_reach)
    except (TypeError, ValueError):
        champ_reach = 0.0
    return cand_reach > champ_reach + 0.02
