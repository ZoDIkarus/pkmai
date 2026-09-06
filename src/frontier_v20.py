"""
V20 topological frontier progress (replaces "count explored tiles").

The old ``frontier_score`` was essentially ``len(tiles explored on this stage)``.
An agent that thoroughly grazes the southern half of Route 1 (167 tiles) beat an
agent that pushed 40 tiles toward the unknown northern end. That is backwards:
progress must mean **real topological extension of the known reachable graph**,
direction-independent (works in caves, buildings, cities), and not farmable by
walking back and forth.

Design:

* A *frontier tile* is a known, walked position that still has at least one
  never-walked cardinal neighbour direction.
* Every frontier tile gets a ``graph_depth`` = shortest known walkable BFS
  distance from the stage origin to that tile.  Pure graph distance - never
  ``-y``, ``north_progress``, Manhattan-to-north or any compass heuristic.
* ``frontier_value`` blends depth, how open the tile still is, and how heavily
  the local area has already been re-walked.
* The reward is a strict **high-watermark**: only a new best ``frontier_value``
  within the episode pays.  ``depth 21 -> 22 -> 21 -> 22`` pays once, at the
  first 22.

Nothing here builds a parallel map structure - callers pass in the adjacency
dict the navigation code already maintains (``_adjacency_for_map``).
"""
from __future__ import annotations

from collections import deque

# -- weights / knobs (kept here so pokemon_env imports one source) -----------
FRONTIER_PROGRESS_REWARD = 0.15       # per genuine new graph-depth high-watermark
FRONTIER_PROGRESS_EPSILON = 0.5       # min value gain to count as a new watermark
FRONTIER_GRAPH_DEPTH_WEIGHT = 1.0
FRONTIER_UNKNOWN_WEIGHT = 0.25
FRONTIER_REVISIT_WEIGHT = 0.15

EVENT_FRONTIER_PROGRESS = "frontier_highwater"


def _cardinal_neighbours(tile):
    x, y = tile
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


class FrontierGraph:
    """Frontier metrics for ONE map, derived from the known walkable adjacency.

    ``adjacency``: ``{(x, y): set((x, y), ...)}`` - the walked-edge graph of this
    map (exactly what ``_adjacency_for_map`` returns).
    ``origin``: the stage-entry coordinate on this map (BFS root).  ``None`` ->
    depth is measured from the lowest-id node so the metric still works, just
    less meaningfully; callers should pass a real origin when they have one.
    """

    def __init__(self, adjacency, origin):
        self.adjacency = adjacency or {}
        self.origin = tuple(origin) if origin is not None else None
        self._depth = None  # lazy BFS field
        self._best_cache = {}  # revisit_density -> best value (graph is immutable)

    # -- depth ------------------------------------------------------------
    def _largest_component(self):
        """Nodes of the biggest connected component. Isolated RAM-glitch
        islands (a few stray warp coordinates) are excluded so depth is
        measured on the real walkable map."""
        seen = set()
        best = set()
        for start in self.adjacency:
            if start in seen:
                continue
            comp = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp.add(n)
                seen.add(n)
                stack.extend(self.adjacency.get(n, ()))
            if len(comp) > len(best):
                best = comp
        return best

    def _depth_field(self):
        if self._depth is not None:
            return self._depth
        field = {}
        comp = self._largest_component()
        roots = []
        if self.origin is not None and self.origin in comp:
            roots = [self.origin]
        elif self.origin is not None and comp:
            # The exact stage-origin tile has no walked edge yet (fresh episode
            # / stale shared snapshot). Root at the known node closest to it,
            # within the main component, so depth still means "graph distance
            # from where the stage begins" - never a compass heuristic.
            ox, oy = self.origin
            roots = [min(
                comp,
                key=lambda t: abs(t[0] - ox) + abs(t[1] - oy),
            )]
        elif comp:
            roots = [min(comp)]
        for r in roots:
            field[r] = 0
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            nd = field[node] + 1
            for nxt in self.adjacency.get(node, ()):
                if nxt in field:
                    continue
                field[nxt] = nd
                queue.append(nxt)
        self._depth = field
        return field

    def graph_depth(self, tile):
        """Known walkable BFS distance origin -> tile, or ``None`` if the tile
        is not connected to the origin on the known graph (never a compass
        fallback - an unconnected tile simply does not score)."""
        return self._depth_field().get(tuple(tile))

    # -- openness -------------------------------------------------------
    def unknown_neighbour_count(self, tile):
        walked = self.adjacency.get(tuple(tile), ())
        return sum(1 for n in _cardinal_neighbours(tuple(tile)) if n not in walked)

    def is_frontier(self, tile):
        tile = tuple(tile)
        return tile in self.adjacency and self.unknown_neighbour_count(tile) > 0

    # -- value ---------------------------------------------------------
    def frontier_value(self, tile, revisit_density=0.0,
                       branch_novelty=0.0, loop_risk=0.0):
        """Topological value AT ``tile``: how deep on the known graph (BFS from
        the stage origin) and how much still-unexplored neighbourhood it has.
        ``None`` if the tile is not connected to the origin on the known graph
        - never a compass fallback."""
        depth = self.graph_depth(tile)
        if depth is None:
            return None
        return (
            FRONTIER_GRAPH_DEPTH_WEIGHT * float(depth)
            + FRONTIER_UNKNOWN_WEIGHT * float(self.unknown_neighbour_count(tile))
            + branch_novelty
            - FRONTIER_REVISIT_WEIGHT * float(revisit_density)
            - loop_risk
        )

    def best_frontier_value(self, revisit_density=0.0):
        """Max ``frontier_value`` over all current frontier tiles.  ``0.0`` when
        there is no frontier (safe neutral fallback).  Cached per
        ``revisit_density`` - the graph itself is immutable for this instance."""
        rd = round(float(revisit_density), 3)
        if rd in self._best_cache:
            return self._best_cache[rd]
        best = 0.0
        seen_any = False
        for tile in self.adjacency:
            if self.unknown_neighbour_count(tile) <= 0:
                continue
            v = self.frontier_value(tile, revisit_density=revisit_density)
            if v is None:
                continue
            seen_any = True
            if v > best:
                best = v
        result = best if seen_any else 0.0
        self._best_cache[rd] = result
        return result

    def max_graph_depth(self):
        field = self._depth_field()
        return max(field.values()) if field else 0

    def frontier_tile_count(self):
        return sum(1 for t in self.adjacency
                   if self.unknown_neighbour_count(t) > 0)


class FrontierHighWater:
    """Strict high-watermark shaper for an INCREASING value (mirror of
    ``TargetShaper``, which tracks a decreasing distance).

    Positive reward is paid ONLY on a new best value within the current key.
    Returning to a value already reached pays nothing, so oscillating between
    two already-visited areas can never repeatedly earn progress.
    """

    def __init__(self, progress_reward=FRONTIER_PROGRESS_REWARD,
                 epsilon=FRONTIER_PROGRESS_EPSILON):
        self.progress_reward = float(progress_reward)
        self.epsilon = float(epsilon)
        self.best = 0.0
        self._key = None

    def start(self, key, initial=0.0):
        self._key = key
        self.best = float(initial)

    def reset(self):
        self.best = 0.0
        self._key = None

    def update(self, key, value):
        """Return ``(reward, event_or_None)`` for this step."""
        if value is None:
            return 0.0, None
        value = float(value)
        if key != self._key:
            # Objective changed (new episode / stage / map) - re-anchor, pay 0.
            self.start(key, value)
            return 0.0, None
        if value > self.best + self.epsilon:
            gain = value - self.best
            self.best = value
            return self.progress_reward * gain, EVENT_FRONTIER_PROGRESS
        return 0.0, None
