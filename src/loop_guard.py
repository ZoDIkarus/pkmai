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
        return (self.idle_steps >= self.max_idle_steps or
                (len(self.positions) == self.positions.maxlen
                 and len(set(self.positions)) <= self.max_tiles))
