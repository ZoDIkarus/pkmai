"""Health requirements for reusable exploration and battle starts.

Two distinct bars:

* ``party_ready`` (strict) - EVERY valid party member >= MIN_CHECKPOINT_HP_RATIO
  HP, no status, usable PP. Required for ENTRY checkpoints (``stage_<n>``) and
  FIGHTER anchors: BRIDGE/RETENTION resume these, a weak party here strands them.

* ``frontier_viable`` (2026-09-07) - a hurt but still playable party that is
  allowed to anchor a ``stage_frontier_<n>`` checkpoint only. This lets an agent
  that fought its way deeper into a no-heal route (e.g. Route 1) actually SAVE
  its spatial progress instead of losing it on the next wipe. A strict
  ``party_ready`` party is always ``frontier_viable`` too.
"""
MIN_CHECKPOINT_HP_RATIO = 0.80

# Frontier viability - see module docstring. Deliberately separate constants so
# this never silently loosens the strict entry bar above.
FRONTIER_MIN_ALIVE = 2
FRONTIER_MIN_TOTAL_HP_RATIO = 0.50
# A viable-but-not-``party_ready`` state may only replace a healthy frontier
# anchor if it advances the topological frontier score by at least this much.
FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN = 3.0


def _valid_members(party):
    return [m for m in party if m.get('checksum_ok') and m.get('max_hp', 0) > 0]


def frontier_viable(party):
    """True if this party may anchor a ``stage_frontier_<n>`` checkpoint.

    Requirements (all of):
      * at least ``FRONTIER_MIN_ALIVE`` members alive (cur_hp > 0),
      * total current HP / total max HP over the valid party
        >= ``FRONTIER_MIN_TOTAL_HP_RATIO``,
      * at least one alive member with a move that still has PP
        (the party can still take a turn).
    A strict ``party_ready`` party trivially satisfies this.
    """
    valid = _valid_members(party)
    if not valid:
        return False
    alive = [m for m in valid if m.get('cur_hp', 0) > 0]
    total_max = sum(m['max_hp'] for m in valid)
    total_cur = sum(m['cur_hp'] for m in valid)
    total_ratio = (total_cur / total_max) if total_max > 0 else 0.0
    can_act = any(
        m.get('cur_hp', 0) > 0
        and any(mv.get('pp', 0) > 0 for mv in m.get('moves', []))
        for m in valid
    )
    return (
        len(alive) >= FRONTIER_MIN_ALIVE
        and total_ratio >= FRONTIER_MIN_TOTAL_HP_RATIO
        and can_act
    )


def party_health(party):
    valid = _valid_members(party)
    ready = bool(valid) and len(valid) == len(party)
    ratios = [m['cur_hp'] / m['max_hp'] for m in valid]
    ready = ready and all(
        ratio >= MIN_CHECKPOINT_HP_RATIO and not m.get('status', 0)
        and any(move.get('pp', 0) > 0 for move in m.get('moves', []))
        for m, ratio in zip(valid, ratios)
    )
    alive = [m for m in valid if m.get('cur_hp', 0) > 0]
    total_max = sum(m['max_hp'] for m in valid)
    total_cur = sum(m['cur_hp'] for m in valid)
    total_ratio = round((total_cur / total_max) if total_max > 0 else 0.0, 4)
    viable = bool(ready) or frontier_viable(party)
    return {
        'party_ready': bool(ready),
        'party_min_hp_ratio': round(min(ratios, default=0.0), 4),
        'party_hp': total_cur,
        'party_max_hp': total_max,
        'party_size': len(valid),
        # Additive frontier-viability fields (2026-09-07). Older meta files that
        # predate these keep working: every reader uses ``.get`` with a default.
        'party_alive': len(alive),
        'party_total_hp_ratio': total_ratio,
        'frontier_viable': viable,
    }


def may_replace_frontier(existing, score, health, metric_version):
    """Whether a candidate frontier state may overwrite ``existing``.

    Hard invariant: **no replacement path may ever reduce ``frontier_score``.**
    ``score`` is graph depth from the stage origin; a smaller value is a
    spatially worse anchor and is never accepted, no matter how healthy the
    candidate is.

    Rules:
      * a candidate that is neither ``party_ready`` nor ``frontier_viable`` may
        anchor nothing;
      * ``party_ready`` candidate vs ``party_ready`` anchor: strictly deeper
        score, or an HP refresh (>= +5 pp minimum HP) at the same score;
      * ``party_ready`` candidate vs hurt anchor (viable or legacy): reclaim it
        as long as ``score >= old_score`` - health may restore the party but
        must not walk the anchor backwards;
      * merely-viable (hurt) candidate vs ANY anchor (healthy or hurt): only on
        a real forward jump of >= FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN.
    """
    if not (health.get('party_ready') or health.get('frontier_viable')):
        return False

    new_ready = bool(health.get('party_ready', False))
    old_ready = bool(existing.get('party_ready', False))
    old_score = (float(existing.get('frontier_score', 0))
                 if existing.get('frontier_metric_version', 0) >= metric_version
                 else 0.0)

    if new_ready and old_ready:
        # healthy -> healthy: strictly deeper, or an HP refresh at equal depth.
        return (score > old_score or
                (score >= old_score and health['party_min_hp_ratio'] >=
                 float(existing.get('party_min_hp_ratio', 0)) + 0.05))

    if new_ready:
        # healthy candidate vs a hurt / legacy / poisoned anchor: reclaim it
        # only if it does NOT give up spatial progress.
        return score >= old_score

    # candidate is only viable (hurt): replacing anything needs a real jump.
    return score >= old_score + FRONTIER_WEAK_ANCHOR_MIN_SCORE_GAIN
