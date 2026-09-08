"""Battle champion promotion policy (2×2).

The battle candidate is scored against a fixed, reproducible eval suite (the
SAME scenarios and seeds the champion is scored on). It only promotes on a
real, sufficiently-sampled improvement in true battle outcomes with:

  * a hard safety gate at zero (no unclassified ends, driver aborts, menu
    stalls, unreadable states, real clock timeouts, illegal trainer flees,
    invalid actions) — reward can never buy past this;
  * enough coverage (>= 200 battles across several wild AND trainer scenarios);
  * no wild flees in the eval (a flee is "did not demonstrate a win");
  * a measured performance win over the champion, per area and per battle kind.

A rising training reward is a learning signal only — it never authorises a
champion. Several confirmed hard regressions authorise ONLY
``battle_learner <- battle_champion`` — never navigation, curriculum,
savestates, exploration memory or the scenario pool.
"""
from __future__ import annotations

EVAL_SUITE_MIN_EPISODES = 200

# A promotable champion must be exercised broadly, not just on one Route-1 wild.
MIN_DISTINCT_SCENARIOS = 3
REQUIRE_TRAINER_AND_WILD = True

# Any of these > 0 in the candidate eval hard-blocks promotion. This is the gate
# a rising reward can never override.
SAFETY_ZERO_METRICS = (
    "invalid_actions", "aborted_macros", "menu_stalls", "unreadable_states",
    "terminal_unknown", "timeouts", "trainer_flees",
    # Catch-v2 (spec §13): a v2 candidate additionally may not have produced a
    # single trainer-catch attempt, an unrequested catch, or a step where the
    # reward components did not sum to the step reward.
    "trainer_catch_attempts", "unrequested_catch_rate", "reward_sum_mismatch",
)
# A wild flee in an eval battle means the candidate did not demonstrate a win.
MAX_FLEE_RATE = 0.0

# --- Catch-v2 promotion gate (spec §13) -----------------------------------
CATCH_SUCCESS_RATE_WHEN_REQUESTED_MIN = 0.70
CATCH_GROUP_SUCCESS_RATE_MIN = 0.50      # per normal wild group
CATCH_TARGET_KO_RATE_MAX = 0.10
BALLS_PER_SUCCESSFUL_CATCH_MAX = 3.0

# regression thresholds: a candidate must not drop a key metric below
# (champion - abs) or (champion * rel)
_ABS = 0.08
_REL = 0.90
_CORE_ABS = 0.05
_CORE_REL = 0.95

# how many separate hard regressions before a learner<-champion reset
HARD_REGRESSION_TRIGGER = 2

# residual HP is only meaningful on WON battles (a flee keeps HP but is a loss)
_KEY_METRICS = ("win_rate", "trainer_win_rate", "wild_win_rate",
                "avg_residual_hp_on_win")
_BAD_METRICS = ("wipe_rate",)   # lower is better


class BattlePromotionDecision(dict):
    @property
    def promote(self):
        return bool(self.get("promote"))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def battle_safety_report(metrics):
    """``(clean, [reasons])`` — the hard gate. Every listed counter must be 0
    and no wild flees. Reward can never override this."""
    m = metrics or {}
    reasons = []
    for k in SAFETY_ZERO_METRICS:
        v = _f(m.get(k))
        if v > 0:
            reasons.append(f"{k}={int(v)} (must be 0)")
    fr = _f(m.get("flee_rate"))
    if fr > MAX_FLEE_RATE:
        reasons.append(f"flee_rate={fr:.3f} (eval battles must be won, not fled)")
    return (not reasons, reasons)


def battle_coverage_report(metrics):
    """``(ok, [reasons])`` — enough battles across enough scenario variety."""
    m = metrics or {}
    reasons = []
    n = int(_f(m.get("episodes")))
    if n < EVAL_SUITE_MIN_EPISODES:
        reasons.append(f"only {n} eval episodes (need {EVAL_SUITE_MIN_EPISODES})")
    cov = int(_f(m.get("scenario_coverage")))
    if cov < MIN_DISTINCT_SCENARIOS:
        reasons.append(
            f"only {cov} distinct eval scenarios (need {MIN_DISTINCT_SCENARIOS}; "
            "capture more wild + trainer battle starts at varied level / HP)")
    if REQUIRE_TRAINER_AND_WILD:
        if int(_f(m.get("n_trainer"))) < 1:
            reasons.append("no trainer battle in the eval suite")
        if int(_f(m.get("n_wild"))) < 1:
            reasons.append("no wild battle in the eval suite")
    return (not reasons, reasons)


def battle_catch_gate(catch_metrics):
    """``(ok, [reasons])`` — the Catch-v2 eval gate (spec §13).

    Scored on a FIXED catch suite (candidate + champion identical scenarios +
    seeds). Total / average reward is NEVER a criterion. Mechanically-impossible
    scenarios (``no_balls``) are excluded from the success rate and scored
    separately as correct non-execution.
    """
    m = catch_metrics or {}
    reasons = []
    req = int(_f(m.get("catch_requested_episodes")))
    if req < 1:
        reasons.append("no catch-requested episodes in the catch eval suite")
    sr = _f(m.get("catch_success_rate_when_requested"))
    if sr < CATCH_SUCCESS_RATE_WHEN_REQUESTED_MIN:
        reasons.append(
            f"catch_success_rate_when_requested {sr:.3f} < "
            f"{CATCH_SUCCESS_RATE_WHEN_REQUESTED_MIN}")
    for grp, gsr in (m.get("per_group_catch_success_rate") or {}).items():
        if _f(gsr) < CATCH_GROUP_SUCCESS_RATE_MIN:
            reasons.append(
                f"group {grp!r} catch_success_rate {_f(gsr):.3f} < "
                f"{CATCH_GROUP_SUCCESS_RATE_MIN}")
    ko = _f(m.get("target_ko_rate"))
    if ko > CATCH_TARGET_KO_RATE_MAX:
        reasons.append(f"target_ko_rate {ko:.3f} > {CATCH_TARGET_KO_RATE_MAX}")
    bpc = _f(m.get("balls_per_successful_catch"))
    if bpc > BALLS_PER_SUCCESSFUL_CATCH_MAX:
        reasons.append(
            f"balls_per_successful_catch {bpc:.2f} > {BALLS_PER_SUCCESSFUL_CATCH_MAX}")
    if _f(m.get("unrequested_catch_rate")) > 0:
        reasons.append("unrequested_catch_rate must be 0")
    if _f(m.get("trainer_catch_attempts")) > 0:
        reasons.append("trainer_catch_attempts must be 0")
    return (not reasons, reasons)


def first_champion_ok(candidate):
    """The very first PPO champion replaces the verified rule brain: same hard
    safety + coverage gate as any promotion, plus an absolute win bar (there is
    no champion to beat yet)."""
    candidate = candidate or {}
    clean, safety = battle_safety_report(candidate)
    covered, cov = battle_coverage_report(candidate)
    reasons = list(safety) + list(cov)
    wr = _f(candidate.get("win_rate"))
    wpr = _f(candidate.get("wipe_rate"))
    if wr < 0.80:
        reasons.append(f"win_rate {wr:.3f} < 0.80")
    if wpr > 0.20:
        reasons.append(f"wipe_rate {wpr:.3f} > 0.20")
    return BattlePromotionDecision(
        promote=not reasons, hard_regression=False,
        reason=("; ".join(reasons)
                or "first live PPO clears the safety + coverage + 80% win gate"),
        safety_reasons=safety, coverage_reasons=cov,
        candidate_episodes=int(_f(candidate.get("episodes"))))


def _regressions(champ, cand, *, abs_drop, rel_keep):
    out = []
    for k in _KEY_METRICS:
        c, d = _f(champ.get(k)), _f(cand.get(k))
        if d < c - abs_drop or (c > 0 and d < c * rel_keep):
            out.append((k, round(c, 3), round(d, 3)))
    for k in _BAD_METRICS:
        c, d = _f(champ.get(k)), _f(cand.get(k))
        if d > c + abs_drop or (c < 1 and d > 1 - (1 - c) * rel_keep):
            out.append((k, round(c, 3), round(d, 3)))
    # per-area AND per-battle-kind: nothing may lose > abs_drop win rate
    for grp in ("per_area_win_rate", "per_kind_win_rate"):
        for key, cw in (champ.get(grp) or {}).items():
            dw = _f((cand.get(grp) or {}).get(key))
            if dw < _f(cw) - abs_drop:
                out.append((f"{grp}:{key}", round(_f(cw), 3), round(dw, 3)))
    return out


def _improved(champ, cand):
    """The performance ladder: a real win over the champion, not a coin flip."""
    win_gain = _f(cand.get("win_rate")) - _f(champ.get("win_rate"))
    hp_gain = (_f(cand.get("avg_residual_hp_on_win"))
               - _f(champ.get("avg_residual_hp_on_win")))
    pt = _f(champ.get("avg_turns_on_win"))
    ct = _f(cand.get("avg_turns_on_win"))
    turn_cut = (pt - ct) / pt if pt > 0 else 0.0
    same_wr = abs(win_gain) < 0.01
    same_hp = abs(hp_gain) < 0.01
    ok = (win_gain >= 0.03
          or (same_wr and hp_gain >= 0.05)
          or (same_wr and same_hp and turn_cut >= 0.10))
    return ok, {"win_gain": round(win_gain, 3),
                "residual_hp_on_win_gain": round(hp_gain, 3),
                "turn_cut_on_win": round(turn_cut, 3)}


def evaluate_battle_promotion(*, candidate, champion,
                              candidate_core=None, champion_core=None,
                              catch_eval=None):
    """``catch_eval`` (spec §13): the v2 candidate's FIXED catch-suite metrics.
    When present, ``battle_catch_gate`` must also pass — a high combat win-rate
    can never promote a broken catch-v2, and a high catch rate can never mask a
    combat regression (both gates are ANDed)."""
    candidate = candidate or {}
    champion = champion or {}
    reasons = []

    covered, cov_reasons = battle_coverage_report(candidate)
    if not covered:
        return BattlePromotionDecision(
            promote=False, hard_regression=False,
            reason="; ".join(cov_reasons),
            coverage_reasons=cov_reasons, safety_reasons=[],
            regressions=[], core_regressions=[],
            candidate_episodes=int(_f(candidate.get("episodes"))))

    clean, safety_reasons = battle_safety_report(candidate)
    if not clean:
        return BattlePromotionDecision(
            promote=False, hard_regression=False,
            reason="safety gate: " + "; ".join(safety_reasons),
            safety_reasons=safety_reasons, coverage_reasons=[],
            regressions=[], core_regressions=[],
            candidate_episodes=int(_f(candidate.get("episodes"))))

    catch_ok, catch_reasons = (True, [])
    if catch_eval is not None:
        catch_ok, catch_reasons = battle_catch_gate(catch_eval)
        if not catch_ok:
            return BattlePromotionDecision(
                promote=False, hard_regression=False,
                reason="catch gate: " + "; ".join(catch_reasons),
                safety_reasons=[], coverage_reasons=[], catch_reasons=catch_reasons,
                regressions=[], core_regressions=[],
                candidate_episodes=int(_f(candidate.get("episodes"))))

    improved, gains = _improved(champion, candidate)
    regressions = _regressions(champion, candidate, abs_drop=_ABS, rel_keep=_REL)
    core_regressions = []
    if candidate_core and champion_core:
        core_regressions = _regressions(champion_core, candidate_core,
                                        abs_drop=_CORE_ABS, rel_keep=_CORE_REL)
    if _f(candidate.get("switch_loops")) > max(3, 2 * _f(champion.get("switch_loops", 0))):
        regressions.append(("switch_loops", champion.get("switch_loops", 0),
                            candidate.get("switch_loops")))

    if not improved:
        reasons.append(f"no meaningful improvement over champion ({gains})")
    if regressions:
        reasons.append(f"regressions: {regressions}")
    if core_regressions:
        reasons.append(f"regression-core regressions: {core_regressions}")

    promote = improved and not regressions and not core_regressions
    hard = len(core_regressions) >= 1 and len(regressions) >= HARD_REGRESSION_TRIGGER

    return BattlePromotionDecision(
        promote=bool(promote),
        hard_regression=bool(hard),
        reason="; ".join(reasons) or "passes all battle-promotion gates",
        safety_reasons=[], coverage_reasons=[], catch_reasons=[],
        regressions=regressions,
        core_regressions=core_regressions,
        candidate_episodes=int(_f(candidate.get("episodes"))),
        **gains)


def reset_scope_for_hard_regression():
    """What a battle hard-regression is allowed to touch — nothing else."""
    return {
        "resets": ["battle_learner <- battle_champion"],
        "never_touches": ["navigation_learner", "navigation_champion",
                          "curriculum", "savestates", "exploration_memory",
                          "nav_horizon_state", "scenario_pool"],
    }
