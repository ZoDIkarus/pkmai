"""PWhiddy-style local coordinate map for the navigation observation.

Built **only** from RAM / map-graph data — never from screenshot similarity.
Produces a fixed ``(C, 64, 64)`` uint8 tensor centred on the player, with one
channel per kind of spatial knowledge. Separate from the existing 4×64×64
screen stack.

Channels (index -> meaning):
  0  discovered/walkable structure of the current map (global knowledge)
  1  tiles visited in THIS episode          (per-episode exploration)
  2  tiles known globally (persistent memory)
  3  player position                        (single bright pixel at centre)
  4  known exits / warps on this map
  5  current frontier / target direction signal

Per-episode rule (PWhiddy): channel 1 starts empty every episode and a tile
pays exploration reward at most once per episode; after a reset the early path
is "new" again so the whole route keeps training. The persistent channels 0/2/4
carry global knowledge for context but do not suppress that.
"""
from __future__ import annotations

import numpy as np

MAP_SIZE = 64
NUM_CHANNELS = 6
CH_STRUCTURE = 0
CH_EPISODE_VISITS = 1
CH_GLOBAL_VISITS = 2
CH_PLAYER = 3
CH_WARPS = 4
CH_FRONTIER = 5

_HALF = MAP_SIZE // 2


class NavCoordMap:
    """Rolling per-episode local map. Feed it RAM-derived facts; read a tensor.

    Coordinates are ``(map_key, x, y)`` where ``map_key`` is any hashable id of
    the current map (e.g. ``(bank, map_id)``). The window is centred on the
    player so the policy always sees itself at the centre.
    """

    def __init__(self, *, map_size=MAP_SIZE):
        self.map_size = int(map_size)
        self._half = self.map_size // 2
        self.episode_visited = set()        # {(map_key, x, y)} this episode
        self.global_visited = set()         # persistent (loaded from memory)
        self.structure = set()              # known walkable tiles (global)
        self.warps = set()                  # {(map_key, x, y)} known exits
        self._frontier_vec = (0.0, 0.0)     # unit-ish direction toward target
        self._player = None                 # (map_key, x, y)

    # -- lifecycle -------------------------------------------------
    def reset_episode(self, *, keep_global=True):
        """PWhiddy per-episode reset: clear this-episode visits so early tiles
        are rewarding again. Global knowledge is retained for context only."""
        self.episode_visited = set()
        if not keep_global:
            self.global_visited = set()
        self._player = None

    def load_global(self, *, visited=(), structure=(), warps=()):
        self.global_visited |= {tuple(t) for t in visited}
        self.structure |= {tuple(t) for t in structure}
        self.warps |= {tuple(t) for t in warps}

    # -- per-step updates ---------------------------------------
    def observe(self, map_key, x, y, *, walkable=True, is_warp=False,
                frontier_vec=None):
        """Record the player at ``(map_key, x, y)``. Returns True the FIRST time
        this tile is seen this episode (the caller pays exploration reward)."""
        key = (map_key, int(x), int(y))
        self._player = key
        first = key not in self.episode_visited
        self.episode_visited.add(key)
        self.global_visited.add(key)
        if walkable:
            self.structure.add(key)
        if is_warp:
            self.warps.add(key)
        if frontier_vec is not None:
            fx, fy = float(frontier_vec[0]), float(frontier_vec[1])
            n = (fx * fx + fy * fy) ** 0.5 or 1.0
            self._frontier_vec = (fx / n, fy / n)
        return first

    def episode_tile_is_new(self, map_key, x, y):
        return (map_key, int(x), int(y)) not in self.episode_visited

    # -- render -------------------------------------------------
    def tensor(self):
        """``(NUM_CHANNELS, map_size, map_size)`` uint8, centred on the player."""
        out = np.zeros((NUM_CHANNELS, self.map_size, self.map_size), dtype=np.uint8)
        if self._player is None:
            return out
        pk, px, py = self._player

        def plot(ch, coords, value=255):
            for (mk, tx, ty) in coords:
                if mk != pk:
                    continue
                gx = tx - px + self._half
                gy = ty - py + self._half
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    out[ch, gy, gx] = value

        plot(CH_STRUCTURE, self.structure, 180)
        plot(CH_EPISODE_VISITS, self.episode_visited, 255)
        plot(CH_GLOBAL_VISITS, self.global_visited, 120)
        plot(CH_WARPS, self.warps, 255)
        out[CH_PLAYER, self._half, self._half] = 255
        # frontier channel: a short ray from the centre toward the target
        fx, fy = self._frontier_vec
        for r in range(1, self._half):
            gx = int(round(self._half + fx * r))
            gy = int(round(self._half + fy * r))
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                out[CH_FRONTIER, gy, gx] = max(60, 255 - r * 6)
        return out

    # -- serialization (global part only) ---------------------
    def global_state(self):
        return {
            "visited": [list(t) if not isinstance(t[0], (list, tuple))
                        else [list(t[0])] + list(t[1:]) for t in self.global_visited],
        }
