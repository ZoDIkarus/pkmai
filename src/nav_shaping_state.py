"""Navigation shaping state — split into a race-safe global file and one
private file per worker.

Every FULL worker had its own cached ``NavShapingState`` and ``_flush`` just
wrote that cached full copy back under the lock, so worker B silently clobbered
worker A's updates (an atomic write only prevents *corrupt* JSON, not lost
updates).

Now:

* ``NavGlobalState``  (``runtime/navigation/nav_global.json``) — the ONLY shared
  mutable field is ``training_run_id`` (bumped by ``train.py`` when a brand-new
  PPO is created). Every mutation is a locked read-modify-write of the current
  disk version. The confirmed directed movement graph and the fleet-first
  topology claims live in ``nav_graph`` / ``shared_tiles`` / ``shared_edges``
  and are not duplicated here.

* ``NavAgentState``  (``runtime/navigation/shaping/agent_<rank>.json``) —
  private to one worker, so no lock is needed. Holds everything that must
  survive an episode reset AND a wipe but is per-agent-journey:
    - ``objectives[key]``          -> ``{"best", "pre_wipe"}`` per objective key
                                      (run_id, world_stage, target_source,
                                      normalized_targets) so distance 0 at the
                                      old exit never blocks the next objective;
    - ``rewarded_next_hop_edges``  -> RUN-WIDE set: a physical directed edge
                                      that already paid a next-hop reward this
                                      run never pays again (stage change / wipe
                                      / episode reset / recovery included);
    - ``wild_wins[map]``           -> FULL wild-win decay counter;
    - ``caught_species``           -> persistent catch dedup (kills the
                                      +54/+56/+58 re-farm over resets).
  A ``training_run_id`` change wipes the whole agent file.
"""
from __future__ import annotations

import json
import os
import tempfile

GLOBAL_SCHEMA = "nav_global_v1"
AGENT_SCHEMA = "nav_agent_shaping_v1"


def _atomic_write(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".navshape_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def objective_key(run_id, world_stage, target_source, targets):
    """Stable key for one navigation objective. Changing world stage / target
    source / target set is a real objective change; anything else is not."""
    norm = sorted((int(t[0]), int(t[1])) for t in (targets or []))
    return "|".join([str(int(run_id)), str(int(world_stage)),
                     str(target_source or "none"),
                     ";".join(f"{x},{y}" for x, y in norm)])


# ---------------------------------------------------------------------------
class NavGlobalState:
    """Shared. Only ``training_run_id`` is mutable; every write is a locked
    read-modify-write of the CURRENT disk version."""

    def __init__(self, path):
        self.path = path
        self.training_run_id = 0

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f) or {}
        except (OSError, ValueError, TypeError):
            return {}

    def load(self):
        self.training_run_id = int(self._read().get("training_run_id", 0))
        return self

    def start_fresh_run(self, lock=None):
        """Bump the run id (a genuinely fresh PPO / brain reset). Locked RMW."""
        from contextlib import nullcontext
        with (lock or nullcontext()):
            d = self._read()
            d["schema"] = GLOBAL_SCHEMA
            d["training_run_id"] = int(d.get("training_run_id", 0)) + 1
            _atomic_write(self.path, d)
            self.training_run_id = d["training_run_id"]
        return self.training_run_id

    def current_run_id(self):
        return int(self._read().get("training_run_id", 0))


# ---------------------------------------------------------------------------
class NavAgentState:
    """Private to one worker (``agent_<rank>.json``) — no lock needed."""

    def __init__(self, path, *, run_id=0):
        self.path = path
        self.run_id = int(run_id)
        self.objectives = {}          # key -> {"best": int|None, "pre_wipe": int|None}
        self._edges = set()           # (map_str, x, y, action) run-wide
        self.wild_wins = {}           # "bank,map" -> int
        self.caught_species = set()   # species_id
        self.stages_advanced = set()  # "from->to" run-wide (the +250 magnet)
        self._dirty = False

    # -- lifecycle ---------------------------------------------------
    @classmethod
    def load(cls, path, *, run_id=0):
        s = cls(path, run_id=run_id)
        try:
            with open(path) as f:
                d = json.load(f) or {}
        except (OSError, ValueError, TypeError):
            d = {}
        if int(d.get("run_id", -1)) != int(run_id):
            # stale (a fresh training run) -> start clean, keep nothing
            s._dirty = True
            return s
        s.objectives = {k: dict(v) for k, v in (d.get("objectives") or {}).items()}
        s._edges = {tuple(e) for e in (d.get("rewarded_next_hop_edges") or [])}
        s.wild_wins = dict(d.get("wild_wins") or {})
        s.caught_species = {int(x) for x in (d.get("caught_species") or [])}
        s.stages_advanced = {str(x) for x in (d.get("stages_advanced") or [])}
        return s

    def save(self):
        _atomic_write(self.path, {
            "schema": AGENT_SCHEMA,
            "run_id": self.run_id,
            "objectives": self.objectives,
            "rewarded_next_hop_edges": [list(e) for e in sorted(self._edges)],
            "wild_wins": self.wild_wins,
            "caught_species": sorted(self.caught_species),
            "stages_advanced": sorted(self.stages_advanced),
        })
        self._dirty = False

    def save_if_dirty(self):
        if self._dirty:
            self.save()

    # -- per-objective distance highwater --------------------------
    def _obj(self, key):
        return self.objectives.setdefault(key, {"best": None, "pre_wipe": None})

    def best_distance(self, key):
        return self._obj(key).get("best")

    def record_distance(self, key, dist):
        """True iff ``dist`` is a strict new highwater for this objective."""
        if dist is None:
            return False
        o = self._obj(key)
        if o["best"] is None or int(dist) < int(o["best"]):
            o["best"] = int(dist)
            self._dirty = True
            return True
        return False

    # -- wipe / recovery, per objective ---------------------------
    def on_wipe(self, key):
        o = self._obj(key)
        o["pre_wipe"] = o["best"]
        self._dirty = True

    def pre_wipe_highwater(self, key):
        return self._obj(key).get("pre_wipe")

    def clear_pre_wipe(self, key):
        if self._obj(key).get("pre_wipe") is not None:
            self._obj(key)["pre_wipe"] = None
            self._dirty = True

    def shaping_frozen(self, key, current_distance):
        pw = self.pre_wipe_highwater(key)
        if pw is None or current_distance is None:
            return False
        return int(current_distance) >= int(pw)

    # -- run-wide next-hop edge dedup -----------------------------
    @staticmethod
    def _edge(bank, map_id, frm, action):
        return (f"{int(bank)},{int(map_id)}", int(frm[0]), int(frm[1]), int(action))

    def next_hop_already_paid(self, bank, map_id, frm, action):
        return self._edge(bank, map_id, frm, action) in self._edges

    def mark_next_hop_paid(self, bank, map_id, frm, action):
        self._edges.add(self._edge(bank, map_id, frm, action))
        self._dirty = True

    # -- run-wide world-stage advance dedup --------------------
    # The big one-off world-stage bonus (e.g. +250 Pallet->Route1) is a
    # learning MAGNET. Paying it every episode teaches "sprint to Route 1", not
    # "then push north". It is paid at most ONCE per (from->to) per training
    # run - and this set only clears on a genuine fresh run (stale run_id),
    # NOT on episode reset, wipe or savestate reload.
    @staticmethod
    def _stage_key(from_stage, to_stage):
        return f"{int(from_stage)}->{int(to_stage)}"

    def stage_advance_already_paid(self, from_stage, to_stage):
        return self._stage_key(from_stage, to_stage) in self.stages_advanced

    def mark_stage_advance_paid(self, from_stage, to_stage):
        self.stages_advanced.add(self._stage_key(from_stage, to_stage))
        self._dirty = True

    # -- wild-win decay -----------------------------------------
    def wild_wins_for(self, bank, map_id):
        return int(self.wild_wins.get(f"{int(bank)},{int(map_id)}", 0))

    def add_wild_win(self, bank, map_id):
        k = f"{int(bank)},{int(map_id)}"
        self.wild_wins[k] = int(self.wild_wins.get(k, 0)) + 1
        self._dirty = True
        return self.wild_wins[k]

    # -- persistent catch dedup --------------------------------
    def already_caught(self, species_id):
        return int(species_id) in self.caught_species

    def mark_caught(self, species_id):
        if int(species_id) not in self.caught_species:
            self.caught_species.add(int(species_id))
            self._dirty = True
            return True
        return False
