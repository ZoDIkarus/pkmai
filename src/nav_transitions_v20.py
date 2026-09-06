"""
V20 known-vs-unknown story transition memory (brief sections 4, 5, 21).

A world-stage transition is either:

    UNKNOWN_NEXT_TRANSITION  - never observed. Exploration is enabled, target
                               shaping is disabled. No fake target, no generic
                               graph frontier, no wall/house/dead-end target.
    KNOWN_NEXT_TRANSITION    - a real ``stage N -> stage N+1`` crossing has been
                               observed. We persist the exact source map, source
                               exit coordinate, destination map and destination
                               coordinate. That exact transition becomes the
                               navigation objective; exploration reward on the
                               solved stage is strongly reduced.

Navigation targets NEVER come from north-most points, map edges, nearest
frontiers, random houses or dead ends - only from real recorded crossings.
"""
from __future__ import annotations

import json
import os
import tempfile

UNKNOWN = "UNKNOWN_NEXT_TRANSITION"
KNOWN = "KNOWN_NEXT_TRANSITION"

# A single observation can be a RAM misread or a glitch warp (e.g. a Pallet
# coordinate bleeding into the Route 1 frame right at the shared border). Just
# like ``_pallet_route1_target`` discards vote-1 outliers, a crossing only
# becomes KNOWN once its canonical variant has been seen at least this often.
MIN_CANONICAL_OBSERVATIONS = 2


def _coord(v):
    return [int(v[0]), int(v[1])]


class KnownTransitions:
    def __init__(self):
        # src_stage -> record dict
        self._by_src = {}

    # -- recording -----------------------------------------------------
    def record(self, src_stage, dst_stage, source_map, source_exit,
               dest_map, dest_coord):
        """Record a genuine crossing. Only ``dst == src + 1`` forward hops are
        stored; anything else is ignored (backtracking, glitch warps)."""
        src_stage = int(src_stage)
        dst_stage = int(dst_stage)
        if dst_stage != src_stage + 1 or src_stage < 1:
            return False

        rec = self._by_src.get(src_stage)
        key = (tuple(_coord(source_map)), tuple(_coord(source_exit)),
               tuple(_coord(dest_map)), tuple(_coord(dest_coord)))
        if rec is None:
            rec = {
                "src_stage": src_stage,
                "dst_stage": dst_stage,
                "source_map": _coord(source_map),
                "source_exit": _coord(source_exit),
                "dest_map": _coord(dest_map),
                "dest_coord": _coord(dest_coord),
                "observations": 1,
                "variants": {repr(key): 1},
            }
            self._by_src[src_stage] = rec
            return True

        rec["observations"] += 1
        variants = rec.setdefault("variants", {})
        variants[repr(key)] = variants.get(repr(key), 0) + 1
        # Promote the most-observed variant as the canonical crossing so a
        # single RAM misread cannot pin the objective to a bad tile.
        best = max(variants.items(), key=lambda kv: kv[1])[0]
        if best == repr(key):
            rec["source_map"] = _coord(source_map)
            rec["source_exit"] = _coord(source_exit)
            rec["dest_map"] = _coord(dest_map)
            rec["dest_coord"] = _coord(dest_coord)
        return True

    def record_and_state(self, src_stage, dst_stage, source_map, source_exit,
                         dest_map, dest_coord):
        """Like ``record`` but returns ``(stored, became_known)``.

        ``became_known`` is ``True`` only on the exact observation that flips
        this crossing from UNKNOWN to KNOWN (its canonical variant reaching
        ``MIN_CANONICAL_OBSERVATIONS``) - used once to pay a scout for
        confirming a real progress edge.
        """
        before = self.navigation_state(int(src_stage))
        stored = self.record(src_stage, dst_stage, source_map, source_exit,
                             dest_map, dest_coord)
        after = self.navigation_state(int(src_stage))
        return stored, (before == UNKNOWN and after == KNOWN)

    # -- queries -----------------------------------------------------
    def navigation_state(self, src_stage):
        rec = self._by_src.get(int(src_stage))
        if not rec:
            return UNKNOWN
        variants = rec.get("variants") or {}
        canonical = max(variants.values()) if variants else int(
            rec.get("observations", 0)
        )
        if int(canonical) >= MIN_CANONICAL_OBSERVATIONS:
            return KNOWN
        return UNKNOWN

    def is_known(self, src_stage):
        return self.navigation_state(src_stage) == KNOWN

    def source_exit_for_stage(self, src_stage):
        """The exact source-side exit coordinate to steer toward, or ``None``.

        ``None`` means: do NOT shape a target here (the transition is unknown).
        """
        rec = self._by_src.get(int(src_stage))
        if not rec:
            return None
        return (int(rec["source_exit"][0]), int(rec["source_exit"][1]))

    def source_map_for_stage(self, src_stage):
        rec = self._by_src.get(int(src_stage))
        if not rec:
            return None
        return (int(rec["source_map"][0]), int(rec["source_map"][1]))

    def record_for_stage(self, src_stage):
        return self._by_src.get(int(src_stage))

    # -- serialization --------------------------------------------
    def to_dict(self):
        return {"schema": "nav_transitions_v20",
                "transitions": {str(k): v
                                for k, v in sorted(self._by_src.items())}}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k, v in (d.get("transitions") or {}).items():
            obj._by_src[int(k)] = dict(v)
        return obj

    def save(self, path):
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                   prefix=".navt_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        try:
            with open(str(path)) as f:
                return cls.from_dict(json.load(f) or {})
        except (OSError, ValueError):
            return cls()
