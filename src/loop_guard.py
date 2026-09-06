"""Bound prolonged local wandering without interrupting battles or real progress."""

from collections import deque


class LocalLoopGuard:
    def __init__(self, window=900, max_tiles=8, max_idle_steps=1800):
        self.positions = deque(maxlen=window)
        self.max_tiles = max_tiles
        self.progress = None
        self.idle_steps = 0
        self.max_idle_steps = max_idle_steps

    def update(self, position, progress, in_battle=False):
        if progress != self.progress:
            self.positions.clear()
            self.progress = progress
            self.idle_steps = 0
        if in_battle or position is None:
            return False
        self.idle_steps += 1
        self.positions.append(position)
        return (
            self.idle_steps >= self.max_idle_steps
            or (
                len(self.positions) == self.positions.maxlen
                and len(set(self.positions)) <= self.max_tiles
            )
        )


# Detect sustained A-B-A-B / A-B-C loops before they become long local stalls.
LOCAL_LOOP_PENALTY_START = -0.05
LOCAL_LOOP_PENALTY_MAX = -0.25


class ShortCycleGuard:
    def __init__(self, history=18, min_repeats=3, max_period=3,
                 penalty_start=LOCAL_LOOP_PENALTY_START,
                 penalty_max=LOCAL_LOOP_PENALTY_MAX, escalate_every=12,
                 truncate_after=600):
        self.positions = deque(maxlen=int(history))
        self.min_repeats = int(min_repeats)
        self.max_period = int(max_period)
        self.penalty_start = float(penalty_start)
        self.penalty_max = float(penalty_max)
        self.escalate_every = int(escalate_every)
        self.truncate_after = int(truncate_after)
        self.progress = None
        self.cycle_steps = 0
        self.in_cycle = False

    def _detect_period(self):
        sequence = list(self.positions)
        for period in range(1, self.max_period + 1):
            needed = period * self.min_repeats
            if len(sequence) < needed:
                continue
            tail = sequence[-needed:]
            base = tail[:period]
            if (all(tail[i] == base[i % period] for i in range(needed))
                    and len(set(base)) >= 2):
                return period
        return 0

    def update(self, position, progress, active=True):
        if progress != self.progress:
            self.progress = progress
            self.positions.clear()
            self.cycle_steps = 0
            self.in_cycle = False
        if not active or position is None:
            return {"penalty": 0.0, "suppress_shaping": self.in_cycle,
                    "truncate": False, "cycle": self.in_cycle, "period": 0}
        self.positions.append(position)
        period = self._detect_period()
        if not period:
            if self.in_cycle and len(set(list(self.positions)[-4:])) >= 4:
                self.in_cycle = False
                self.cycle_steps = 0
            return {"penalty": 0.0, "suppress_shaping": False,
                    "truncate": False, "cycle": False, "period": 0}
        self.in_cycle = True
        self.cycle_steps += 1
        steps_over = max(0, self.cycle_steps - period * self.min_repeats)
        penalty = self.penalty_start * (1 + steps_over // max(1, self.escalate_every))
        penalty = max(self.penalty_max, penalty)
        return {"penalty": penalty, "suppress_shaping": True,
                "truncate": self.cycle_steps >= self.truncate_after,
                "cycle": True, "period": period}
