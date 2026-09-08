"""Battle-scenario pool + route-based sampling (Phase 2).

Navigation / Watcher runs may *capture* a validated battle start-state into the
pool. Capture is a pure data transfer — it grants no reward to any policy. The
battle trainer works only on **copies** of these savestates in its own
emulators.

An area is "reliably unlocked" only after a real navigation gate:
  * >= 50 completed beginning-full runs,
  * area/transition reach rate >= 80%,
  * confirmed on >= 2 distinct seeds,
  * a confirmed navigation-champion version.

Sampling for 10 battle workers:
  * once Route 1 is reliably open: >= 5 workers on Route-1 scenarios;
  * general mix: 60% newest-3 areas / 25% older areas / 15% fixed regression
    core; empty buckets redistribute; exactly 10 workers; no growth per route.
Within a bucket, concrete scenarios are drawn by weight (coverage-balancing).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

MAX_POOL_SIZE = 4000
SAMPLE_MIX = {"recent": 0.60, "older": 0.25, "regression_core": 0.15}
RECENT_AREA_WINDOW = 3
ROUTE1_MIN_WORKERS = 5

# area-unlock evidence thresholds
UNLOCK_MIN_BEGINNING_RUNS = 50
UNLOCK_MIN_REACH_RATE = 0.80
UNLOCK_MIN_SEEDS = 2

SCHEMA = "scenario_pool_v1"


def _digest(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


# --- scenario diversity buckets (spec ZIEL B) ----------------------------
LEVEL_BUCKET_SIZE = 5          # 1-5, 6-10, ...


def level_bucket(level):
    try:
        lv = max(1, int(level or 0))
    except (TypeError, ValueError):
        return "L?"
    return f"L{((lv - 1) // LEVEL_BUCKET_SIZE) * LEVEL_BUCKET_SIZE + 1}-" \
           f"{((lv - 1) // LEVEL_BUCKET_SIZE + 1) * LEVEL_BUCKET_SIZE}"


def party_signature(own_species, own_levels):
    """A coarse, order-independent party fingerprint: sorted species each with
    its level bucket. Two functionally-similar parties collapse to one bucket
    so a single seed cannot dominate the training distribution."""
    pairs = sorted((int(s or 0), level_bucket(lv))
                   for s, lv in zip(own_species or (), own_levels or ()))
    return "+".join(f"{s}@{b}" for s, b in pairs) or "empty"


def scenario_bucket(*, area, enemy_species, enemy_level, player_levels,
                    own_species, objective_mode="combat"):
    """The dedup / balancing key (spec ZIEL B):

        area + species + enemy_level_bucket + player_level_bucket
             + party_signature + objective_mode
    """
    es = tuple(sorted(int(s or 0) for s in (enemy_species or ()))) or (int(enemy_species or 0),)
    plev = level_bucket(max((int(x or 0) for x in (player_levels or [0])), default=0))
    return "|".join((
        str(area), ",".join(str(s) for s in es), level_bucket(enemy_level),
        plev, party_signature(own_species, player_levels), str(objective_mode)))


class BattleScenario:
    """A materially-distinct battle start. The signature includes HP / PP /
    status / enemy / party so two genuinely different fights are never merged.
    """

    FIELDS = ("area", "battle_kind", "savestate_ref", "rom_sha256",
              "enemy_species", "enemy_levels", "own_species", "own_levels",
              "own_hp", "own_pp", "own_status", "own_items", "story_flag_count",
              "rng_seed", "readiness_schema")

    def __init__(self, area, battle_kind, savestate_ref, *, rom_sha256="",
                 enemy_species=(), enemy_levels=(), own_species=(), own_levels=(),
                 own_hp=(), own_pp=(), own_status=(), own_items=(),
                 story_flag_count=0, rng_seed=None, readiness_schema="battle_snapshot_v2",
                 meta=None):
        self.area = str(area)
        self.battle_kind = battle_kind if battle_kind in ("wild", "trainer") else "wild"
        self.savestate_ref = str(savestate_ref)
        self.rom_sha256 = str(rom_sha256 or "")
        self.enemy_species = tuple(enemy_species)
        self.enemy_levels = tuple(enemy_levels)
        self.own_species = tuple(own_species)
        self.own_levels = tuple(own_levels)
        self.own_hp = tuple(own_hp)
        self.own_pp = tuple(tuple(x) if isinstance(x, (list, tuple)) else (x,)
                            for x in own_pp)
        self.own_status = tuple(own_status)
        self.own_items = tuple(own_items)
        self.story_flag_count = int(story_flag_count or 0)
        self.rng_seed = rng_seed
        self.readiness_schema = str(readiness_schema)
        self.meta = dict(meta or {})
        self.quarantined = False
        self.quarantine_reason = ""
        self.signature = self._signature()

    def _signature(self):
        return _digest(
            self.area, self.battle_kind, self.rom_sha256,
            self.enemy_species, self.enemy_levels,
            self.own_species, self.own_levels, self.own_hp, self.own_pp,
            self.own_status, self.own_items, self.story_flag_count,
            self.readiness_schema)[:20]

    def validate(self, *, expected_rom_sha256=None):
        """Mark corrupt/incompatible scenarios for quarantine. Returns
        (ok, reason)."""
        if not self.savestate_ref:
            return self._quarantine("no savestate ref")
        if not self.own_species or not self.enemy_species:
            return self._quarantine("missing party/enemy species")
        if expected_rom_sha256 and self.rom_sha256 and \
                self.rom_sha256 != expected_rom_sha256:
            return self._quarantine("ROM hash mismatch")
        if self.readiness_schema != "battle_snapshot_v2":
            return self._quarantine(f"incompatible readiness schema "
                                    f"{self.readiness_schema}")
        self.quarantined = False
        self.quarantine_reason = ""
        return True, ""

    def _quarantine(self, reason):
        self.quarantined = True
        self.quarantine_reason = reason
        return False, reason

    def to_dict(self):
        d = {f: getattr(self, f) for f in self.FIELDS}
        d.update(signature=self.signature, quarantined=self.quarantined,
                 quarantine_reason=self.quarantine_reason, meta=self.meta)
        # JSON-safe tuples -> lists
        return json.loads(json.dumps(d, default=list))

    @classmethod
    def from_dict(cls, d):
        s = cls(d["area"], d["battle_kind"], d["savestate_ref"],
                rom_sha256=d.get("rom_sha256", ""),
                enemy_species=d.get("enemy_species", ()),
                enemy_levels=d.get("enemy_levels", ()),
                own_species=d.get("own_species", ()),
                own_levels=d.get("own_levels", ()),
                own_hp=d.get("own_hp", ()), own_pp=d.get("own_pp", ()),
                own_status=d.get("own_status", ()),
                own_items=d.get("own_items", ()),
                story_flag_count=d.get("story_flag_count", 0),
                rng_seed=d.get("rng_seed"),
                readiness_schema=d.get("readiness_schema", "battle_snapshot_v2"),
                meta=d.get("meta"))
        s.quarantined = bool(d.get("quarantined"))
        s.quarantine_reason = d.get("quarantine_reason", "")
        return s


class AreaUnlockLedger:
    """Tracks which areas are *reliably* navigation-unlocked, with real
    evidence, not just two tokens."""

    def __init__(self):
        self._evidence = {}   # area -> {"runs":int,"reach_rate":float,"seeds":set,"champion_version":int}
        self._order = []      # unlock order

    def record_navigation_evidence(self, area, *, beginning_runs, reach_rate,
                                   seed, champion_version):
        area = str(area)
        e = self._evidence.setdefault(
            area, {"runs": 0, "reach_rate": 0.0, "seeds": set(),
                   "champion_version": 0})
        was = self.is_trainable(area)
        e["runs"] = max(e["runs"], int(beginning_runs or 0))
        e["reach_rate"] = max(e["reach_rate"], float(reach_rate or 0.0))
        if seed is not None:
            e["seeds"].add(str(seed))
        e["champion_version"] = max(e["champion_version"], int(champion_version or 0))
        if not was and self.is_trainable(area):
            self._order.append(area)
        return self.is_trainable(area)

    def is_trainable(self, area):
        e = self._evidence.get(str(area))
        if not e:
            return False
        return (e["runs"] >= UNLOCK_MIN_BEGINNING_RUNS
                and e["reach_rate"] >= UNLOCK_MIN_REACH_RATE
                and len(e["seeds"]) >= UNLOCK_MIN_SEEDS
                and e["champion_version"] >= 1)

    def trainable_areas(self):
        return [a for a in self._order if self.is_trainable(a)]

    def recent_and_older(self, window=RECENT_AREA_WINDOW):
        t = self.trainable_areas()
        older = t[:-window] if len(t) > window else []
        return t[-window:], older

    def to_dict(self):
        return {a: {**e, "seeds": sorted(e["seeds"])}
                for a, e in self._evidence.items()}

    @classmethod
    def from_dict(cls, d):
        led = cls()
        for area, e in (d or {}).items():
            led._evidence[area] = {
                "runs": int(e.get("runs", 0)),
                "reach_rate": float(e.get("reach_rate", 0.0)),
                "seeds": set(e.get("seeds", [])),
                "champion_version": int(e.get("champion_version", 0))}
            if led.is_trainable(area):
                led._order.append(area)
        return led


class ScenarioPool:
    def __init__(self, max_size=MAX_POOL_SIZE, rom_sha256=""):
        self.max_size = int(max_size)
        self.rom_sha256 = str(rom_sha256 or "")
        self._by_sig = {}
        self._order = []
        self._quarantine = {}
        self.regression_core = set()
        # coverage counters: how often each signature was actually trained on
        self.train_counts = {}

    def __len__(self):
        return len(self._by_sig)

    # -- capture ----------------------------------------------------
    def capture(self, scenario, *, from_validated_battle=True):
        """Add a scenario. Returns (added: bool, reason: str). **No reward.**"""
        if not from_validated_battle:
            return False, "capture only from a validated navigation/watcher battle"
        ok, reason = scenario.validate(expected_rom_sha256=self.rom_sha256 or None)
        if not ok:
            self._quarantine[scenario.signature] = scenario
            return False, f"quarantined: {reason}"
        if scenario.signature in self._by_sig:
            return False, "duplicate signature (materially identical start)"
        self._by_sig[scenario.signature] = scenario
        self._order.append(scenario.signature)
        self.train_counts.setdefault(scenario.signature, 0)
        self._evict()
        return True, "captured"

    def pin_regression_core(self, signature):
        if signature in self._by_sig:
            self.regression_core.add(signature)

    def _evict(self):
        i = 0
        while len(self._order) > self.max_size and i < len(self._order):
            sig = self._order[i]
            if sig in self.regression_core:
                i += 1
                continue
            self._order.pop(i)
            self._by_sig.pop(sig, None)
            self.train_counts.pop(sig, None)

    # -- queries --------------------------------------------------
    def scenarios_for_area(self, area):
        return [s for s in self._by_sig.values() if s.area == str(area)
                and not s.quarantined]

    def note_trained(self, signature):
        if signature in self.train_counts:
            self.train_counts[signature] += 1

    # -- sampling ------------------------------------------------
    def _weighted_pick(self, scenarios):
        """Least-trained-first weighting so coverage evens out. Deterministic
        given equal counts (stable by signature)."""
        if not scenarios:
            return None
        return min(scenarios,
                   key=lambda s: (self.train_counts.get(s.signature, 0), s.signature))

    def sample_plan(self, ledger, battle_workers=10, mix=SAMPLE_MIX,
                    window=RECENT_AREA_WINDOW):
        """Concrete per-worker scenario assignment for one round of episodes."""
        n = int(battle_workers)
        recent, older = ledger.recent_and_older(window)
        route1_open = "route1" in ledger.trainable_areas() or \
                      "Route1" in ledger.trainable_areas()
        core_sigs = [s for s in self.regression_core if s in self._by_sig]
        has_core = bool(core_sigs)

        counts = {
            "recent": round(n * mix["recent"]) if recent else 0,
            "older": round(n * mix["older"]) if older else 0,
            "regression_core": round(n * mix["regression_core"]) if has_core else 0,
        }
        assigned = sum(counts.values())
        # redistribute remainder into the first non-empty bucket
        for b in ("recent", "older", "regression_core"):
            avail = {"recent": bool(recent), "older": bool(older),
                     "regression_core": has_core}[b]
            if avail:
                counts[b] += n - assigned
                break
        # guarantee >= 1 core worker when a core exists
        if has_core and counts["regression_core"] < 1 and n >= 2:
            counts["regression_core"] = 1
            biggest = max(("recent", "older"), key=lambda k: counts[k])
            counts[biggest] = max(0, counts[biggest] - 1)
        # Route-1 focus
        if route1_open and n >= 2:
            focus = min(ROUTE1_MIN_WORKERS, n - (1 if has_core else 0))
            if counts["recent"] < focus:
                take = focus - counts["recent"]
                counts["recent"] += take
                counts["older"] = max(0, counts["older"] - take)

        bucket_areas = {
            "recent": recent, "older": older,
            "regression_core": None,  # handled via core_sigs
        }
        seq = (["recent"] * counts["recent"] + ["older"] * counts["older"]
               + ["regression_core"] * counts["regression_core"])
        # pad to exactly n with whatever bucket is available
        pad_bucket = ("recent" if recent else "older" if older else
                      "regression_core" if has_core else "recent")
        seq = (seq + [pad_bucket] * n)[:n]

        plan = []
        core_cursor = 0
        for i in range(n):
            b = seq[i]
            scenario = None
            if b == "regression_core" and core_sigs:
                sig = core_sigs[core_cursor % len(core_sigs)]
                core_cursor += 1
                scenario = self._by_sig.get(sig)
            else:
                areas = bucket_areas.get(b) or []
                pool = [s for a in areas for s in self.scenarios_for_area(a)]
                scenario = self._weighted_pick(pool)
            plan.append({
                "battle_worker_ix": i, "bucket": b,
                "scenario_signature": scenario.signature if scenario else None,
                "scenario_area": scenario.area if scenario else None,
                "battle_kind": scenario.battle_kind if scenario else None,
            })
            if scenario:
                self.note_trained(scenario.signature)
        return {"counts": counts, "plan": plan,
                "recent_areas": recent, "older_areas": older,
                "route1_focus": route1_open}

    # -- three-groups-of-three route plan (final user decision) --------
    def route_group_plan(self, ledger, *, headless_workers=9, groups=3,
                         per_group=3, window=3):
        """Fixed groups of ``per_group`` headless workers per active route.

        Active routes = the newest ``groups`` reliably-unlocked routes that ALSO
        have >= 1 validated scenario. Older routes go to the regression core.
        Leftover headless workers fill with available variants / regression
        scenarios of the active routes — never an invented later route. Also
        returns the single VISIBLE watcher scenario (always the newest active
        route).
        """
        trainable = ledger.trainable_areas()
        active = [r for r in trainable[-window:]
                  if self.scenarios_for_area(r)][-groups:]
        older = [r for r in trainable if r not in active]
        core_sigs = [s for s in self.regression_core if s in self._by_sig]

        assignments = []          # list of dicts, one per headless worker
        # 1) fixed groups
        for gi, route in enumerate(active):
            for _ in range(per_group):
                sc = self._weighted_pick(self.scenarios_for_area(route))
                assignments.append({"group": gi, "route": route,
                                    "scenario_signature": sc.signature if sc else None,
                                    "battle_kind": sc.battle_kind if sc else None})
                if sc:
                    self.note_trained(sc.signature)
        # 2) leftover workers -> variants of active routes / regression core
        leftover = headless_workers - len(assignments)
        fill_pool = ([s for r in active for s in self.scenarios_for_area(r)]
                     or [self._by_sig[s] for s in core_sigs])
        core_cursor = 0
        for _ in range(max(0, leftover)):
            if fill_pool:
                sc = self._weighted_pick(fill_pool)
                assignments.append({"group": "fill", "route": sc.area,
                                    "scenario_signature": sc.signature,
                                    "battle_kind": sc.battle_kind})
                self.note_trained(sc.signature)
            elif core_sigs:
                sig = core_sigs[core_cursor % len(core_sigs)]
                core_cursor += 1
                sc = self._by_sig[sig]
                assignments.append({"group": "regression_core", "route": sc.area,
                                    "scenario_signature": sig,
                                    "battle_kind": sc.battle_kind})
            else:
                assignments.append({"group": "idle", "route": None,
                                    "scenario_signature": None, "battle_kind": None})
        assignments = assignments[:headless_workers]

        # 3) visible watcher -> newest active route (or the deepest trainable
        #    route that has a scenario; never invented)
        watcher_route = active[-1] if active else None
        watcher_sc = (self._weighted_pick(self.scenarios_for_area(watcher_route))
                      if watcher_route else None)
        return {
            "active_routes": active,
            "regression_core_routes": older,
            "headless_assignments": assignments,
            "headless_count": len(assignments),
            "visible_watcher": {
                "route": watcher_route,
                "scenario_signature": watcher_sc.signature if watcher_sc else None,
                "battle_kind": watcher_sc.battle_kind if watcher_sc else None,
                # the watcher only adopts a newer route at the END of its
                # current battle (see twoby2.orchestration / battle_watch)
                "switch_policy": "after_current_battle_only",
            },
        }

    # -- coverage metrics --------------------------------------
    def coverage(self):
        by_area, by_kind = {}, {"wild": 0, "trainer": 0}
        for s in self._by_sig.values():
            by_area[s.area] = by_area.get(s.area, 0) + 1
            by_kind[s.battle_kind] = by_kind.get(s.battle_kind, 0) + 1
        return {"total": len(self._by_sig), "by_area": by_area,
                "by_kind": by_kind, "quarantined": len(self._quarantine),
                "regression_core": len(self.regression_core)}

    # -- persistence -------------------------------------------
    def to_dict(self):
        return {
            "schema": SCHEMA, "rom_sha256": self.rom_sha256,
            "max_size": self.max_size,
            "order": list(self._order),
            "scenarios": {sig: s.to_dict() for sig, s in self._by_sig.items()},
            "quarantine": {sig: s.to_dict() for sig, s in self._quarantine.items()},
            "regression_core": sorted(self.regression_core),
            "train_counts": dict(self.train_counts),
        }

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        p = cls(max_size=d.get("max_size", MAX_POOL_SIZE),
                rom_sha256=d.get("rom_sha256", ""))
        for sig, sd in (d.get("scenarios") or {}).items():
            p._by_sig[sig] = BattleScenario.from_dict(sd)
        p._order = [s for s in (d.get("order") or []) if s in p._by_sig]
        for sig in p._by_sig:
            if sig not in p._order:
                p._order.append(sig)
        for sig, sd in (d.get("quarantine") or {}).items():
            p._quarantine[sig] = BattleScenario.from_dict(sd)
        p.regression_core = set(s for s in (d.get("regression_core") or [])
                                if s in p._by_sig)
        p.train_counts = {k: int(v) for k, v in (d.get("train_counts") or {}).items()}
        for sig in p._by_sig:
            p.train_counts.setdefault(sig, 0)
        return p

    def save_atomic(self, path):
        dd = os.path.dirname(os.path.abspath(path))
        os.makedirs(dd, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dd, suffix=".tmp.json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"), sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
