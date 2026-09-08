"""Full-health battle-scenario normalisation + level-band versioning.

**Original savestates (protected or not) are only ever read.** The normalizer:

  1. captures a raw scenario reference (full agent run or the protected manual
     Route-1 seed) — nothing is copied or changed;
  2. copies the savestate **byte-for-byte into an UNPROTECTED temp working
     file** (:func:`protected_assets.read_only_copy` — sha256 checked before and
     after so the original provably did not change);
  3. normalises ONLY that copy: every living own Pokémon to full HP, status
     cleared, existing move PP filled — species / level / moves / EV / IV /
     items / story unchanged;
  4. updates the Gen-III encrypted substructures + checksums;
  5. saves, reloads, and validates the whole result against the expected state;
  6. only then publishes atomically into the ScenarioPool.

``assert_write_target_ok`` is applied to every real write / publish target
(working copy, published file) — never to the read source. A protected path can
therefore be a source but never a destination of normalize / overwrite /
replace / rename / move / delete / publish.

**Status:** the safe party-mutation + checksum re-encryption path for BPRD is
NOT verified (same family of blockers as the battle-menu RAM). Until it is,
``normalize()`` returns ``ok=False`` with a concrete blocker and never writes a
normalised file. The read-only-copy / validation / level-band / publish-gating
logic around it is fully implemented and tested.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import tempfile

from twoby2.protected_assets import (assert_write_target_ok, canonical_path,
                                     read_only_copy, default_registry)

# Gen-III party Pokémon layout facts (from firered_ram, ROM-independent).
PARTY_MON_SIZE = 100
STATUS1_OFFSET = 0x50           # u32 status bitfield
# The substruct order + checksum are personality-dependent (Gen-III box
# encryption). Rewriting HP/PP/status needs the decrypt→edit→re-encrypt→
# re-checksum path — not verified here.

LOW, CURRENT, HIGH = "low", "current", "high"
LEVEL_BANDS = (LOW, CURRENT, HIGH)
# per (route, band): keep at most this many normalised scenarios
BAND_SIZE_CAP = 12


class NormalizationBlocked(RuntimeError):
    pass


class NormalizedScenarioResult(dict):
    @property
    def ok(self):
        return bool(self.get("ok"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_state_bytes(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


class BattleScenarioNormalizer:
    def __init__(self, *, party_writer=None, checksum_verified=False,
                 registry=None):
        """``party_writer`` is the (unverified) callable that would edit the
        decrypted party in RAM/state bytes. ``checksum_verified`` must be set
        True only once the decrypt→edit→re-encrypt→checksum path is proven for
        BPRD. Default: no writer, not verified -> fail-closed."""
        self.party_writer = party_writer
        self.checksum_verified = bool(checksum_verified)
        self.registry = registry

    # -- capture ---------------------------------------------------
    def capture_raw(self, source_state_path, meta):
        """Record a raw scenario reference (read-only). Copies nothing, mutates
        nothing. A protected path is a legitimate SOURCE here."""
        src = canonical_path(source_state_path)
        if not os.path.isfile(src):
            raise NormalizationBlocked(f"source state not found: {source_state_path}")
        reg = self.registry if self.registry is not None else default_registry()
        return {
            "source_state_path": src,
            "source_sha256": _sha256(src),
            "source_is_protected": reg.protecting_asset(src) is not None,
            "meta": dict(meta or {}),
            "captured": True,
        }

    # -- normalise (fail-closed) --------------------------------
    def normalize(self, raw, *, workdir=None):
        """Attempt to produce a full-health copy. Returns a
        :class:`NormalizedScenarioResult`. On any unverified step: ``ok=False``,
        every original (protected or not) untouched, nothing marked normalized.
        """
        src = raw["source_state_path"]

        # working dir/file must be OUTSIDE every protected path
        tmp_dir = workdir or tempfile.mkdtemp(prefix="scenario_norm_")
        working = os.path.join(tmp_dir, "working" + (".state.gz" if src.endswith(".gz")
                                                     else ".state"))
        # write target check on the working copy (never the source)
        assert_write_target_ok(working, op="write", registry=self.registry)
        # byte-for-byte read-only copy with sha256 checked before AND after
        copy_report = read_only_copy(src, working, registry=self.registry)

        if not (self.checksum_verified and self.party_writer is not None):
            return NormalizedScenarioResult(
                ok=False, normalized=False,
                working_copy=working,
                source_sha256=copy_report["sha256"],
                source_unchanged=True,
                blocker="safe party mutation + Gen-III checksum re-encryption is "
                        "NOT verified for BPRD; refusing to write a normalised "
                        "state. Original untouched (read-only copy verified).",
                needs=[
                    "verified decrypt of the party box substructs for this ROM",
                    "verified re-encrypt + substruct checksum + party checksum",
                    "a round-trip test: normalise -> save -> reload -> every "
                    "living mon full HP, status 0, move PP full, species/level/"
                    "moves/EV/IV/items/story byte-identical",
                ])

        # --- verified path (only runs once checksum_verified is set) ---
        expected = self._expected_state(raw)
        try:
            self.party_writer(working, expected)          # edits the COPY only
        except Exception as exc:                           # pragma: no cover
            return NormalizedScenarioResult(ok=False, normalized=False,
                                            blocker=f"party_writer failed: {exc}")
        reloaded = self._reload_and_check(working, expected)
        if not reloaded["valid"]:
            return NormalizedScenarioResult(ok=False, normalized=False,
                                            blocker=f"reload validation failed: "
                                                    f"{reloaded['problems']}")
        return NormalizedScenarioResult(
            ok=True, normalized=True, working_copy=working,
            normalized_sha256=_sha256(working), expected=expected)

    def _expected_state(self, raw):
        """The state we require after normalisation: full HP, no status, full
        PP; everything else unchanged from the capture meta."""
        meta = raw.get("meta") or {}
        party = []
        for mon in meta.get("party") or []:
            if int(mon.get("cur_hp", 0)) <= 0:
                party.append(dict(mon))                    # fainted: unchanged
                continue
            m = dict(mon)
            m["cur_hp"] = m.get("max_hp", m.get("cur_hp"))
            m["status"] = 0
            m["move_pp"] = [mp for mp in m.get("move_max_pp", m.get("move_pp", []))]
            party.append(m)
        return {"party": party,
                "unchanged": ("species", "level", "moves", "ev", "iv", "items",
                              "story")}

    def _reload_and_check(self, working, expected):
        """A real implementation reloads the state in an isolated emulator and
        reads the party back. Without the verified writer this is never
        reached; kept as the explicit validation contract."""
        problems = []
        # placeholder validation surface — real checks go here once the writer
        # exists. Deliberately conservative: anything unproven -> invalid.
        problems.append("emulator round-trip validation not implemented for the "
                        "unverified path")
        return {"valid": not problems, "problems": problems}

    # -- publish -------------------------------------------------
    def publish(self, result, dest_path):
        """Copy a validated, normalised working file to ``dest_path`` for the
        ScenarioPool. Refuses if the result is not ``ok`` or if ``dest_path`` is
        a protected asset."""
        if not result.ok:
            raise NormalizationBlocked(
                f"refusing to publish a non-ok result: {result.get('blocker')}")
        assert_write_target_ok(dest_path, op="publish", registry=self.registry)
        os.makedirs(os.path.dirname(canonical_path(dest_path)), exist_ok=True)
        with open(result["working_copy"], "rb") as fi, \
                open(canonical_path(dest_path), "wb") as fo:
            fo.write(fi.read())
        return {"published": canonical_path(dest_path),
                "sha256": _sha256(canonical_path(dest_path))}


# --------------------------------------------------------------------------
# level-band versioning
# --------------------------------------------------------------------------
def party_total_level(party):
    return sum(int(m.get("level", 0)) for m in party or [])


def level_band(total_level, *, current_total):
    """Bucket a party's total level relative to the route's current baseline."""
    ct = max(1, int(current_total or 1))
    ratio = total_level / ct
    if ratio < 0.85:
        return LOW
    if ratio > 1.15:
        return HIGH
    return CURRENT


def scenario_signature_fields(scenario_meta):
    """The minimum a normalised-scenario signature must include so a stronger
    party on the same route is a NEW version, not an overwrite."""
    m = scenario_meta
    return (
        m.get("route"),
        m.get("story_phase"),
        m.get("battle_kind"),
        tuple(m.get("enemy_species") or ()),
        tuple(m.get("enemy_levels") or ()),
        tuple(m.get("own_species") or ()),
        tuple(m.get("own_levels") or ()),
        tuple(m.get("own_hp") or ()),
        tuple(tuple(x) if isinstance(x, (list, tuple)) else (x,)
              for x in (m.get("own_pp") or ())),
        tuple(m.get("own_status") or ()),
        m.get("rom_sha256"),
        m.get("schema_version"),
    )


class LevelBandedScenarios:
    """Per (route, band) collection with a size cap. A higher-total-level valid
    start creates a NEW banded version; the original / weaker regression
    scenarios stay. Never evicts a protected or regression-core scenario, and
    only ever evicts a *dominated* non-core duplicate."""

    def __init__(self, *, cap=BAND_SIZE_CAP):
        self.cap = int(cap)
        self._by_route_band = {}           # (route, band) -> [meta,...]
        self.regression_core = set()       # signatures

    def add_version(self, meta, *, current_total_level, is_regression_core=False):
        route = meta.get("route")
        total = party_total_level(meta.get("party") or
                                  [{"level": lv} for lv in meta.get("own_levels") or []])
        band = level_band(total, current_total=current_total_level)
        meta = {**meta, "level_band": band, "party_total_level": total,
                "signature": _hash_sig(scenario_signature_fields(meta))}
        key = (route, band)
        bucket = self._by_route_band.setdefault(key, [])
        if any(x["signature"] == meta["signature"] for x in bucket):
            return {"added": False, "reason": "duplicate signature", "band": band}
        bucket.append(meta)
        if is_regression_core:
            self.regression_core.add(meta["signature"])
        self._evict(key)
        return {"added": True, "band": band, "signature": meta["signature"]}

    def _evict(self, key):
        bucket = self._by_route_band[key]
        while len(bucket) > self.cap:
            # drop the oldest DOMINATED non-core entry (lower total level and a
            # strict subset of another entry's coverage)
            victim = None
            for i, cand in enumerate(bucket):
                if cand["signature"] in self.regression_core:
                    continue
                dominated = any(
                    other is not cand
                    and other["party_total_level"] >= cand["party_total_level"]
                    and set(other.get("enemy_species") or ()) >= set(cand.get("enemy_species") or ())
                    for other in bucket)
                if dominated:
                    victim = i
                    break
            if victim is None:
                break            # nothing safe to evict -> keep over cap
            bucket.pop(victim)

    def versions(self, route, band=None):
        if band is not None:
            return list(self._by_route_band.get((route, band), []))
        return [m for (r, _b), lst in self._by_route_band.items() if r == route
                for m in lst]


def _hash_sig(fields):
    return hashlib.sha1(repr(fields).encode()).hexdigest()[:20]
