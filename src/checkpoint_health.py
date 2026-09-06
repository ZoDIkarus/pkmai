"""Health requirements for reusable exploration and battle starts."""
MIN_CHECKPOINT_HP_RATIO = 0.80


def party_health(party):
    valid = [m for m in party if m.get('checksum_ok') and m.get('max_hp', 0) > 0]
    ready = bool(valid) and len(valid) == len(party)
    ratios = [m['cur_hp'] / m['max_hp'] for m in valid]
    ready = ready and all(
        ratio >= MIN_CHECKPOINT_HP_RATIO and not m.get('status', 0)
        and any(move.get('pp', 0) > 0 for move in m.get('moves', []))
        for m, ratio in zip(valid, ratios)
    )
    return {
        'party_ready': bool(ready),
        'party_min_hp_ratio': round(min(ratios, default=0.0), 4),
        'party_hp': sum(m['cur_hp'] for m in valid),
        'party_max_hp': sum(m['max_hp'] for m in valid),
        'party_size': len(valid),
    }


def may_replace_frontier(existing, score, health, metric_version):
    if not health['party_ready']:
        return False
    if not existing.get('party_ready', False):
        # Health beats a poisoned/legacy anchor, even without a distance record.
        return True
    old_score = (float(existing.get('frontier_score', 0))
                 if existing.get('frontier_metric_version', 0) >= metric_version else 0)
    return (score > old_score or
            (score >= old_score and health['party_min_hp_ratio'] >=
             float(existing.get('party_min_hp_ratio', 0)) + 0.05))
