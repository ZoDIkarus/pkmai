"""Persistent, process-safe wild-encounter + shiny counters (spec ZIEL C).

Separate files per agent CLASS so the isolated Battle fighters, the FULL
navigation agents and the Watcher never share a counter:

    runtime/shiny/<agent_class>.json      (+ a sibling .lock file)

Concurrency: 9 battle fighters and 40 FULL agents can write the same file at
once. The **entire** load -> check -> modify -> atomic-write cycle runs under an
exclusive ``fcntl.flock`` on a separate lock file, so no update is ever lost.

Every counter is monotonic. Guarantees:
  * one wild encounter is counted EXACTLY ONCE (keyed on ``encounter_id``);
  * a verified-shiny encounter gets EXACTLY ONE terminal outcome — a second or
    contradictory outcome is rejected and diagnosed;
  * ``record_shiny_outcome`` only accepts an id that is in the persistent
    ``verified_shiny_encounters`` set;
  * while shiny RAM is unverified nothing is ever counted as a shiny.

Nothing here is a reward. The dashboard reads these files directly.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

try:
    import fcntl
except ImportError:                       # pragma: no cover - POSIX only
    fcntl = None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHINY_DIR = os.path.join(_ROOT, "runtime", "shiny")

AGENT_CLASSES = ("battle_fighter", "full_agent", "watcher")

SCHEMA = "shiny_counters_v2"
_MAX_RECENT = 20
_TERMINAL_OUTCOMES = ("caught", "ko", "fled", "wipe", "unresolved")
_OUTCOME_COUNTER = {"caught": "shiny_caught", "ko": "shiny_ko",
                    "fled": "shiny_fled", "wipe": "shiny_lost_wipe",
                    "unresolved": "shiny_unresolved"}

_COUNTERS = (
    "wild_encounters",
    "shiny_encounters_verified",
    "shiny_caught",
    "shiny_ko",
    "shiny_fled",
    "shiny_lost_wipe",
    "shiny_unresolved",
    "shiny_ram_unknown",
    "shiny_outcome_conflicts",       # diagnostic: rejected 2nd / contradictory
)


def _empty(agent_class):
    return {
        "schema": SCHEMA, "agent_class": agent_class,
        "counters": {k: 0 for k in _COUNTERS},
        "seen_encounters": [],
        "verified_shiny_encounters": [],
        "resolved_shiny_encounters": {},     # eid -> the ONE terminal outcome
        "recent_shinies": [],
        "updated": None,
    }


def _normalise(raw, agent_class):
    d = _empty(agent_class)
    raw = raw or {}
    d["counters"].update({k: int((raw.get("counters") or {}).get(k, 0) or 0)
                          for k in _COUNTERS})
    d["seen_encounters"] = list(raw.get("seen_encounters") or [])
    d["verified_shiny_encounters"] = list(raw.get("verified_shiny_encounters") or [])
    d["resolved_shiny_encounters"] = dict(raw.get("resolved_shiny_encounters") or {})
    d["recent_shinies"] = list(raw.get("recent_shinies") or [])
    d["updated"] = raw.get("updated")
    return d


def _atomic_write(path, doc):
    dd = os.path.dirname(os.path.abspath(path))
    os.makedirs(dd, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dd, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, separators=(",", ":"), sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


class ShinyCounters:
    def __init__(self, agent_class, *, base_dir=SHINY_DIR):
        if agent_class not in AGENT_CLASSES:
            raise ValueError(f"agent_class must be one of {AGENT_CLASSES}")
        self.agent_class = agent_class
        self.path = os.path.join(base_dir, f"{agent_class}.json")
        self.lock_path = self.path + ".lock"

    # -- process-safe critical section --------------------------
    def _update(self, mutate):
        """Run ``mutate(doc) -> (changed, result)`` with the whole
        load->modify->write cycle under an exclusive flock."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        lf = open(self.lock_path, "a+")
        try:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                with open(self.path) as f:
                    raw = json.load(f)
            except (OSError, ValueError):
                raw = None
            doc = _normalise(raw, self.agent_class)
            changed, result = mutate(doc)
            if changed:
                doc["seen_encounters"] = doc["seen_encounters"][-4000:]
                doc["recent_shinies"] = doc["recent_shinies"][-_MAX_RECENT:]
                doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _atomic_write(self.path, doc)
            return result
        finally:
            if fcntl is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()

    def _load(self):
        try:
            with open(self.path) as f:
                return _normalise(json.load(f), self.agent_class)
        except (OSError, ValueError):
            return _empty(self.agent_class)

    def snapshot(self):
        d = self._load()
        for k in ("seen_encounters", "verified_shiny_encounters",
                  "resolved_shiny_encounters"):
            d.pop(k, None)
        return d

    # -- record ------------------------------------------------
    def record_encounter(self, encounter_id, *, shiny_status="unknown",
                         species_id=None, level=None, area=None, worker=None):
        """Record ONE wild encounter (idempotent per ``encounter_id``).

        ``shiny_status`` (from :mod:`twoby2.shiny_ram`):
          * ``verified_shiny``  -> ``shiny_encounters_verified`` + added to the
            persistent ``verified_shiny_encounters`` set + a pending recent row;
          * ``verified_normal`` -> only ``wild_encounters``;
          * anything else -> ``shiny_ram_unknown`` (NEVER a shiny).
        Returns ``True`` if it was newly counted.
        """
        eid = str(encounter_id)

        def mut(d):
            if eid in set(d["seen_encounters"]):
                return False, False
            # ``seen_encounters`` is capped at 4000; ``verified_shiny_encounters``
            # is NOT. Always re-check the un-truncated shiny set so a savestate
            # replay of an OLD verified shiny can never re-count anything.
            if eid in set(d["verified_shiny_encounters"]):
                return False, False
            d["seen_encounters"].append(eid)
            d["counters"]["wild_encounters"] += 1
            if shiny_status == "verified_shiny":
                d["counters"]["shiny_encounters_verified"] += 1
                d["verified_shiny_encounters"].append(eid)
                d["recent_shinies"].append({
                    "encounter_id": eid, "species_id": species_id, "level": level,
                    "area": area, "worker": worker, "agent_class": self.agent_class,
                    "ts": time.time(), "outcome": "pending"})
            elif shiny_status != "verified_normal":
                d["counters"]["shiny_ram_unknown"] += 1
            return True, True

        return self._update(mut)

    def record_shiny_outcome(self, encounter_id, outcome):
        """Record the ONE terminal outcome of a VERIFIED shiny encounter.

        Returns ``{"recorded": bool, "reason": str}``. Rejected (and diagnosed
        via ``shiny_outcome_conflicts``) when:
          * ``encounter_id`` is not in ``verified_shiny_encounters``;
          * the encounter already has a terminal outcome (idempotent for the
            same outcome, conflict-counted for a different one).
        """
        eid = str(encounter_id)
        outcome = str(outcome)

        def mut(d):
            if outcome not in _TERMINAL_OUTCOMES:
                return False, {"recorded": False, "reason": f"bad_outcome:{outcome}"}
            if eid not in set(d["verified_shiny_encounters"]):
                return False, {"recorded": False, "reason": "not_a_verified_shiny"}
            prev = d["resolved_shiny_encounters"].get(eid)
            if prev is not None:
                if prev == outcome:
                    return False, {"recorded": False, "reason": "already_resolved"}
                d["counters"]["shiny_outcome_conflicts"] += 1
                return True, {"recorded": False,
                              "reason": f"conflict:{prev}!={outcome}"}
            d["resolved_shiny_encounters"][eid] = outcome
            d["counters"][_OUTCOME_COUNTER[outcome]] += 1
            for row in d["recent_shinies"]:
                if row.get("encounter_id") == eid and row.get("outcome") == "pending":
                    row["outcome"] = outcome
                    break
            return True, {"recorded": True, "reason": outcome}

        return self._update(mut)

    def is_verified_shiny_encounter(self, encounter_id):
        return str(encounter_id) in set(self._load()["verified_shiny_encounters"])


def aggregate_all(*, base_dir=SHINY_DIR):
    """Combined dashboard view over all three agent classes (spec ZIEL C.9)."""
    per_class, totals = {}, {k: 0 for k in _COUNTERS}
    recent = []
    for cls in AGENT_CLASSES:
        snap = ShinyCounters(cls, base_dir=base_dir).snapshot()
        per_class[cls] = snap
        for k in _COUNTERS:
            totals[k] += int(snap["counters"].get(k, 0) or 0)
        recent.extend(snap.get("recent_shinies") or [])
    recent.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {"schema": SCHEMA, "per_agent_class": per_class,
            "totals": totals, "recent_shinies": recent[:_MAX_RECENT]}
