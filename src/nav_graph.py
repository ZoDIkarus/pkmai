"""Directed movement graph for overworld navigation.

The old navigation graph was **undirected** (`_edge_key` sorted its endpoints,
`_adjacency_for_map` added every edge both ways) and it silently dropped any
move that changed position by more than one tile — so a one-way Route-1 ledge
looked exactly like a normal two-way path. Directed BFS and the reward shaping
could never tell "right, around the wall" from "left, off the ledge, back to
the wall".

This module records a *direction-aware* graph:

    (map, from_xy, action) -> MovementEdge(to_xy, movement_kind, confidence)

`movement_kind`:
  * ``walk``            - single-tile move in the action's direction
  * ``jump_or_ledge``   - the action moved > 1 tile (one-way; the reverse is
                          NEVER synthesised)
  * ``warp``            - a map change (handled by the caller's warp branch)
  * ``blocked_static``  - the action reliably produces no movement here
  * ``blocked_dynamic`` - blocked once but later cleared (NPC / event); open
                          again after a re-check TTL
  * ``unknown``         - imported from the legacy undirected store, direction
                          not yet confirmed by a real observation

Pure Python, no torch / stable_retro, so the env and the tests can both use it.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import deque

SCHEMA = "directed_nav_graph_v1"

# action indices from PokemonFireRedEnv.action_map
UP, DOWN, LEFT, RIGHT = 3, 4, 5, 6
MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT)
DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
REVERSE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
_ACTION_NAME = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

WALK = "walk"
JUMP = "jump_or_ledge"
WARP = "warp"
BLOCKED_STATIC = "blocked_static"
BLOCKED_DYNAMIC = "blocked_dynamic"
UNKNOWN = "unknown"

# a `blocked_static` needs this many *globally distinct* (agent, encounter)
# confirmations - the env feeds a rank-prefixed visit_id so a fleet union
# cannot forge a wall out of one mid-animation sample per worker.
BLOCKED_CONFIRM_DISTINCT_VISITS = 3
# A `blocked_dynamic` (blocked once, not yet multiply confirmed) is ADVISORY
# ONLY: it is never used to cut a route. A `blocked_static` is a real wall and
# is permanent until a successful ``observe_move`` clears it - it has NO
# step-based TTL, because the env's ``now_step`` resets every episode while a
# persisted block's ``last_confirmed_step`` does not (comparing the two made a
# block read as active for a 6000-step window of EVERY later episode).
BLOCKED_TTL_STEPS = 6000            # kept only for backwards-compat imports

# The overworld location is sampled every few agent steps, not every frame, so
# a stretch of continuous walking shows up as a multi-tile delta. Such a delta
# is NOT a ledge and NOT a real single edge - it is discarded. A real forced
# ledge hop can only be learned by a dedicated per-frame observer (not wired).
MAX_TRUSTED_MOVE_DIST = 1


def action_name(a):
    return _ACTION_NAME.get(int(a), str(a))


class MovementEdge:
    __slots__ = ("to", "kind", "confidence", "last_seen_step", "legacy",
                 "both_dirs_observed")

    def __init__(self, to, kind, *, confidence=1, last_seen_step=0,
                 legacy=False):
        self.to = (int(to[0]), int(to[1]))
        self.kind = kind
        self.confidence = int(confidence)
        self.last_seen_step = int(last_seen_step)
        self.legacy = bool(legacy)
        # a plain walk edge only counts as bidirectional once BOTH
        # (A,act->B) and (B,rev->A) have been really observed
        self.both_dirs_observed = False

    def to_dict(self):
        return {"to": list(self.to), "kind": self.kind,
                "confidence": self.confidence,
                "last_seen_step": self.last_seen_step,
                "legacy": self.legacy,
                "both_dirs_observed": self.both_dirs_observed}

    @classmethod
    def from_dict(cls, d):
        e = cls(d["to"], d["kind"], confidence=d.get("confidence", 1),
                last_seen_step=d.get("last_seen_step", 0),
                legacy=d.get("legacy", False))
        e.both_dirs_observed = bool(d.get("both_dirs_observed", False))
        return e


class _Block:
    __slots__ = ("visit_ids", "last_confirmed_step", "kind", "generation")

    def __init__(self):
        self.visit_ids = set()
        self.last_confirmed_step = 0
        self.kind = BLOCKED_DYNAMIC     # promoted to static after enough visits
        self.generation = 0

    def to_dict(self):
        return {"visit_ids": sorted(self.visit_ids)[:16],
                "last_confirmed_step": self.last_confirmed_step,
                "kind": self.kind, "generation": self.generation}

    @classmethod
    def from_dict(cls, d):
        b = cls()
        b.visit_ids = set(int(v) for v in d.get("visit_ids", []))
        b.last_confirmed_step = int(d.get("last_confirmed_step", 0))
        b.kind = d.get("kind", BLOCKED_DYNAMIC)
        b.generation = int(d.get("generation", 0))
        return b


class DirectedNavGraph:
    def __init__(self):
        # {map: {(from_xy, action): MovementEdge}}   map = (bank, map_id)
        self._edges = {}
        # {map: {(from_xy, action): _Block}}
        self._blocks = {}
        self.generation = 0

    # -- recording -----------------------------------------------------
    def observe_move(self, mp, frm, action, to, *, step=0):
        """One observed position change.

        The stored edge action is DERIVED FROM THE GEOMETRY of the move, not
        from the button that was pressed: the overworld location is sampled
        every few agent steps, so the "last pressed action" is often not what
        produced the net displacement. A self-consistent edge
        ``(frm, a) -> to`` where ``to == frm + DELTA[a]`` is the only kind that
        can safely drive ``next_hop_action``.

        Returns ``WALK`` when a clean single-tile cardinal edge was recorded,
        ``None`` for no move, ``"ambiguous_multi_tile"`` for a >1-tile delta,
        ``"ambiguous_diagonal"`` for a 1+1 diagonal delta (both discarded).
        """
        mp = (int(mp[0]), int(mp[1]))
        frm = (int(frm[0]), int(frm[1]))
        to = (int(to[0]), int(to[1]))
        dx, dy = to[0] - frm[0], to[1] - frm[1]
        dist = abs(dx) + abs(dy)

        if dist == 0:
            return None
        if dist > MAX_TRUSTED_MOVE_DIST:
            # continuous walking between samples / a missed warp / a battle
            # transition - never a trustworthy single edge (this is what stored
            # ~1400 fake `jump_or_ledge` edges).
            return "ambiguous_multi_tile"
        geo_action = next((a for a, v in DELTA.items() if v == (dx, dy)), None)
        if geo_action is None:
            return "ambiguous_diagonal"
        action = geo_action                      # geometry wins over the button
        kind = WALK

        # a successful move clears any block on this (tile, action)
        self.clear_block(mp, frm, action, step=step)

        emap = self._edges.setdefault(mp, {})
        key = (frm, action)
        e = emap.get(key)
        if e is None:
            emap[key] = e = MovementEdge(to, kind, confidence=1,
                                         last_seen_step=step)
        else:
            # the latest real observation is authoritative for destination and
            # kind (the old value may have been a legacy guess, or a stale walk
            # where the map actually has a ledge). Confidence still accumulates.
            e.confidence = (e.confidence + 1) if not e.legacy else 1
            e.last_seen_step = step
            e.kind = kind
            e.to = to
            e.legacy = False
            if e.to != to or kind != WALK:
                e.both_dirs_observed = False

        # mark bidirectionality only when the reverse walk is also real
        if kind == WALK and action in REVERSE:
            rev = self._edges.get(mp, {}).get((to, REVERSE[action]))
            if rev is not None and rev.kind == WALK and not rev.legacy:
                e.both_dirs_observed = True
                rev.both_dirs_observed = True
        return kind

    def observe_block(self, mp, frm, action, *, step=0, visit_id=0):
        """The action produced no movement at a trusted position. ``visit_id``
        must change between *separate* encounters with this tile (leave and
        return) so an NPC blocking for a few frames cannot forge a wall."""
        mp = (int(mp[0]), int(mp[1]))
        frm = (int(frm[0]), int(frm[1]))
        action = int(action)
        if action not in MOVE_ACTIONS:
            return
        bmap = self._blocks.setdefault(mp, {})
        b = bmap.get((frm, action))
        if b is None:
            bmap[(frm, action)] = b = _Block()
        b.visit_ids.add(int(visit_id))
        b.last_confirmed_step = int(step)
        b.generation = self.generation
        if len(b.visit_ids) >= BLOCKED_CONFIRM_DISTINCT_VISITS:
            b.kind = BLOCKED_STATIC
        else:
            b.kind = BLOCKED_DYNAMIC

    def clear_block(self, mp, frm, action, *, step=0):
        mp = (int(mp[0]), int(mp[1]))
        b = self._blocks.get(mp, {}).pop(((int(frm[0]), int(frm[1])), int(action)), None)
        return b is not None

    def is_blocked(self, mp, frm, action, *, now_step=0):
        """Only a multiply-confirmed ``blocked_static`` wall cuts a route.

        A ``blocked_static`` is PERMANENT (a wall does not move; only a
        successful :meth:`observe_move` clears it). No step-based TTL: the env's
        ``now_step`` resets each episode, so any TTL against a persisted
        ``last_confirmed_step`` is meaningless. A ``blocked_dynamic`` (blocked
        once, not yet multiply confirmed) is advisory only and never routes.
        """
        b = self._blocks.get((int(mp[0]), int(mp[1])), {}).get(
            ((int(frm[0]), int(frm[1])), int(action)))
        return b is not None and b.kind == BLOCKED_STATIC

    def blocked_actions(self, mp, xy, *, now_step=0):
        return sorted(a for a in MOVE_ACTIONS
                      if self.is_blocked(mp, xy, a, now_step=now_step))

    # -- cleanup --------------------------------------------------
    def sanitize(self):
        """Remove only PROVABLY-WRONG block / ledge entries (spec point 2).
        Returns a before/after report. Does NOT touch ``walk`` edges or legacy
        ``unknown`` edges (still a valid weak-fallback + display hint)."""
        rep = {"edges_before": self.edge_count(),
               "ledge_edges_removed": 0,
               "walk_edges_mislabelled_dropped": 0,
               "walk_edges_rekeyed": 0,
               "blocks_before": sum(len(v) for v in self._blocks.values()),
               "blocked_static_before": 0, "blocked_static_after": 0,
               "blocks_removed_walk_contradiction": 0,
               "blocks_removed_reverse_walk": 0,
               "blocks_removed_legacy_adjacency": 0,
               "blocks_removed_under_confirmed": 0,
               "blocks_downgraded_to_dynamic": 0}
        for bmap in self._blocks.values():
            rep["blocked_static_before"] += sum(
                1 for b in bmap.values() if b.kind == BLOCKED_STATIC)

        # 1) drop every jump_or_ledge edge - the sampled observer cannot tell a
        #    real forced hop from continuous walking, so all of them are suspect.
        for mp, emap in list(self._edges.items()):
            for k in [k for k, e in emap.items() if e.kind == JUMP]:
                emap.pop(k, None)
                rep["ledge_edges_removed"] += 1

        # 1b) an old `walk` edge whose stored ``to`` is NOT ``frm + DELTA[action]``
        #     was mis-keyed (4-step sampling attached the wrong last-pressed
        #     button). Re-key it to the geometric direction, or drop it if the
        #     delta is not a single cardinal step or the slot is taken.
        for mp, emap in list(self._edges.items()):
            for (frm, action), e in list(emap.items()):
                if e.kind != WALK or e.legacy:
                    continue
                dx, dy = e.to[0] - frm[0], e.to[1] - frm[1]
                if (dx, dy) == DELTA.get(action):
                    continue
                geo = next((a for a, v in DELTA.items() if v == (dx, dy)), None)
                emap.pop((frm, action), None)
                if geo is not None and (frm, geo) not in emap:
                    e_new = MovementEdge(e.to, WALK, confidence=e.confidence,
                                         last_seen_step=e.last_seen_step)
                    emap[(frm, geo)] = e_new
                    rep["walk_edges_rekeyed"] += 1
                else:
                    rep["walk_edges_mislabelled_dropped"] += 1

        # 2) clean blocks
        for mp, bmap in list(self._blocks.items()):
            emap = self._edges.get(mp, {})
            for k in list(bmap):
                b = bmap[k]
                frm, action = k
                walk_edge = emap.get((frm, action))
                # 2a) a real (non-legacy) walk edge for the same (tile, action)
                #     proves the direction is passable -> the block is wrong.
                if (walk_edge is not None and walk_edge.kind == WALK
                        and not walk_edge.legacy):
                    bmap.pop(k, None)
                    rep["blocks_removed_walk_contradiction"] += 1
                    continue
                # 2c) the destination tile has a confirmed REVERSE walk edge
                #     back to `frm` -> the two tiles are connected by a real
                #     bidirectional walk (no ledges survive sanitize), so this
                #     block is a false positive (animation / warp-fade / battle
                #     frame). This is what walls off the Route-1 corridor.
                dxy = DELTA.get(action)
                if dxy is not None:
                    to = (frm[0] + dxy[0], frm[1] + dxy[1])
                    rev = emap.get((to, REVERSE[action]))
                    if (rev is not None and rev.kind == WALK and not rev.legacy
                            and rev.to == frm):
                        bmap.pop(k, None)
                        rep["blocks_removed_reverse_walk"] += 1
                        continue
                    # 2d) the legacy undirected store (thousands of real
                    #     single-tile moves from the old system) recorded this
                    #     adjacency in either direction -> the direction is
                    #     passable, the block is a false positive.
                    fwd_leg = emap.get((frm, action))
                    rev_leg = emap.get((to, REVERSE[action]))
                    if ((fwd_leg is not None and fwd_leg.legacy
                         and fwd_leg.to == to)
                            or (rev_leg is not None and rev_leg.legacy
                                and rev_leg.to == frm)):
                        bmap.pop(k, None)
                        rep["blocks_removed_legacy_adjacency"] += 1
                        continue
                # 2b) a `blocked_static` that never actually reached the
                #     distinct-confirmation bar (promoted only by the fleet
                #     visit_id union bug) -> downgrade to advisory dynamic.
                if (b.kind == BLOCKED_STATIC
                        and len(b.visit_ids) < BLOCKED_CONFIRM_DISTINCT_VISITS):
                    b.kind = BLOCKED_DYNAMIC
                    rep["blocks_downgraded_to_dynamic"] += 1
            if not bmap:
                self._blocks.pop(mp, None)

        for bmap in self._blocks.values():
            rep["blocked_static_after"] += sum(
                1 for b in bmap.values() if b.kind == BLOCKED_STATIC)
        rep["edges_after"] = self.edge_count()
        rep["blocks_after"] = sum(len(v) for v in self._blocks.values())
        return rep

    # -- legacy import ----------------------------------------------
    def import_legacy_edges(self, undirected_edges, *, step=0):
        """``[(bank, map_id, x1, y1, x2, y2), ...]`` from the old undirected
        store. Each becomes two ``unknown`` directed edges, flagged ``legacy``:
        usable as a weak routing fallback and for display, but a real
        ``observe_move`` is what upgrades a specific direction to ``walk``."""
        n = 0
        for e in undirected_edges:
            if len(e) != 6:
                continue
            b, m, x1, y1, x2, y2 = (int(v) for v in e)
            mp = (b, m)
            a, c = (x1, y1), (x2, y2)
            for frm, to in ((a, c), (c, a)):
                dx, dy = to[0] - frm[0], to[1] - frm[1]
                act = next((k for k, v in DELTA.items() if v == (dx, dy)), None)
                if act is None:
                    continue
                emap = self._edges.setdefault(mp, {})
                if (frm, act) in emap:
                    continue
                emap[(frm, act)] = MovementEdge(to, UNKNOWN, confidence=0,
                                                last_seen_step=step, legacy=True)
                n += 1
        return n

    # -- routing --------------------------------------------------
    def neighbors(self, mp, xy, *, now_step=0, allow_legacy=True):
        """Traversable neighbours of ``xy``.

        A confirmed ``walk`` edge ``(A,a)->B`` is also traversable in reverse
        (``B --REVERSE[a]--> A``) UNLESS the reverse is a ``blocked_static``
        wall. All ledges are removed in :meth:`sanitize`, so a real walked
        corridor is bidirectional; without this, a corridor only ever walked
        northbound has no southbound edges and BFS from the south fails. The
        synthesised reverse hop is routing-only - it is never written to disk.
        """
        mp = (int(mp[0]), int(mp[1]))
        xy = (int(xy[0]), int(xy[1]))
        emap = self._edges.get(mp, {})
        seen_dirs = set()
        for (frm, action), e in emap.items():
            if frm != xy:
                continue
            if e.kind in (BLOCKED_STATIC, BLOCKED_DYNAMIC):
                continue
            if e.legacy and not allow_legacy:
                continue
            if self.is_blocked(mp, frm, action, now_step=now_step):
                continue
            seen_dirs.add(action)
            yield e.to, action, e.kind
        # synthesised reverse of a confirmed walk: an edge (A,a)->xy means xy is
        # one tile from A in direction REVERSE[a]; absent an explicit wall that
        # reverse walk is real (all ledges are removed in sanitize).
        for (frm, action), e in emap.items():
            if e.kind != WALK or e.legacy or e.to != xy:
                continue
            rev = REVERSE.get(action)
            if rev is None or rev in seen_dirs:
                continue
            if (xy, rev) in emap:
                continue                        # xy already has a real edge that way
            if self.is_blocked(mp, xy, rev, now_step=now_step):
                continue                        # explicit wall - respect it
            seen_dirs.add(rev)
            yield frm, rev, WALK

    def directed_bfs(self, mp, start, targets, *, now_step=0, max_nodes=6000,
                     allow_legacy=True):
        """Forward BFS along directed edges. Returns
        ``{"distance", "next_action", "next_dxy", "path_len", "confirmed"}`` or
        ``None``. ``confirmed`` is True only when the whole route is made of
        REAL observed directed edges (no ``legacy_unverified`` hop). A route
        that needs a legacy edge is display-only: it must NOT drive reward,
        target_valid, a timeout, or a reverse ledge.  ``allow_legacy=False``
        returns only a confirmed route (or ``None``)."""
        targets = {(int(t[0]), int(t[1])) for t in targets}
        if not targets:
            return None
        start = (int(start[0]), int(start[1]))
        strict = self._bfs(mp, start, targets, now_step, max_nodes, False)
        if strict is not None:
            strict["confirmed"] = True
            return strict
        if not allow_legacy:
            return None
        loose = self._bfs(mp, start, targets, now_step, max_nodes, True)
        if loose is not None:
            loose["confirmed"] = False
        return loose

    def _bfs(self, mp, start, targets, now_step, max_nodes, allow_legacy):
        if start in targets:
            return {"distance": 0, "next_action": None, "next_dxy": (0, 0),
                    "path_len": 0}
        seen = {start: (None, None)}       # node -> (prev, first_action)
        q = deque([start])
        nodes = 0
        while q and nodes < max_nodes:
            cur = q.popleft()
            nodes += 1
            for to, action, _kind in self.neighbors(
                    mp, cur, now_step=now_step, allow_legacy=allow_legacy):
                if to in seen:
                    continue
                first = seen[cur][1] if seen[cur][1] is not None else action
                seen[to] = (cur, first)
                if to in targets:
                    dist = 0
                    n = to
                    while seen[n][0] is not None:
                        n = seen[n][0]
                        dist += 1
                    return {"distance": dist, "next_action": first,
                            "next_dxy": DELTA.get(first, (0, 0)),
                            "path_len": dist}
                q.append(to)
        return None

    # -- persistence ---------------------------------------------
    def to_dict(self):
        return {
            "schema": SCHEMA,
            "generation": self.generation,
            "edges": {
                f"{b},{m}": [
                    [list(frm), action] + [e.to_dict()]
                    for (frm, action), e in emap.items()
                ]
                for (b, m), emap in self._edges.items()
            },
            "blocks": {
                f"{b},{m}": [
                    [list(frm), action, blk.to_dict()]
                    for (frm, action), blk in bmap.items()
                ]
                for (b, m), bmap in self._blocks.items() if bmap
            },
        }

    @classmethod
    def from_dict(cls, d):
        g = cls()
        g.generation = int((d or {}).get("generation", 0))
        for mk, rows in ((d or {}).get("edges", {}) or {}).items():
            b, m = (int(v) for v in mk.split(","))
            emap = g._edges.setdefault((b, m), {})
            for row in rows:
                frm = (int(row[0][0]), int(row[0][1]))
                action = int(row[1])
                emap[(frm, action)] = MovementEdge.from_dict(row[2])
        for mk, rows in ((d or {}).get("blocks", {}) or {}).items():
            b, m = (int(v) for v in mk.split(","))
            bmap = g._blocks.setdefault((b, m), {})
            for row in rows:
                frm = (int(row[0][0]), int(row[0][1]))
                action = int(row[1])
                bmap[(frm, action)] = _Block.from_dict(row[2])
        return g

    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".navgraph_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (OSError, ValueError):
            return cls()

    # -- introspection for telemetry -----------------------------
    def edge_at(self, mp, frm, action):
        return self._edges.get((int(mp[0]), int(mp[1])), {}).get(
            ((int(frm[0]), int(frm[1])), int(action)))

    def edge_count(self, mp=None):
        if mp is not None:
            return len(self._edges.get((int(mp[0]), int(mp[1])), {}))
        return sum(len(v) for v in self._edges.values())

    def directed_edges_for_map(self, mp):
        """For the web map: ``[(from_xy, to_xy, action, kind, legacy)]``."""
        out = []
        for (frm, action), e in self._edges.get((int(mp[0]), int(mp[1])), {}).items():
            out.append((frm, e.to, action, e.kind, e.legacy))
        return out
