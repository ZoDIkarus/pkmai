"""Dedicated grass wild-encounter harvester (isolated Battle system only).

Before an isolated wild training battle, the fighter's emulator is standing on a
grass ANCHOR tile. This harvester walks a fixed ±3-tile horizontal corridor
around that anchor until a wild battle starts, then hands control back. After
the battle it deterministically restores the anchor.

Hard rules (spec ZIEL A / ZIEL E):
  * the corridor bound is always relative to the ORIGINAL anchor x, never the
    last position reached;
  * never leave the grass row: a map change, a y change (ledge / stairs), or an
    x jump > 3 aborts immediately and restores the anchor;
  * no warps / doors / ledges / jumps are ever pressed (only LEFT / RIGHT);
  * hard step + wall-clock limits — on breach: diagnosed abort, never a loop;
  * the pendulum walk is NOT a PPO action and produces NO reward;
  * this runs ONLY for isolated battle scenarios. The FULL / navigation agent
    never touches this module.

RNG: the caller supplies a reproducible ``seed_id`` (stored in the scenario
metadata). This module adds only neutral frame jitter (a small number of
no-input frames before the walk) — never an unverified RAM write.
"""
from __future__ import annotations

import time

# only horizontal moves — a vertical press could take a ledge / stairs
_LEFT, _RIGHT = "LEFT", "RIGHT"
_OPPOSITE = {_LEFT: _RIGHT, _RIGHT: _LEFT}
_DIR_SIGN = {_LEFT: -1, _RIGHT: +1}


class HarvesterConfig:
    MAX_TILES_FROM_ANCHOR = 3          # ±3 -> a fixed 7-tile corridor
    MAX_STEPS = 120                    # tile walks before "no_encounter"
    MAX_WALL_CLOCK_S = 20.0
    # tile-aware movement: hold the direction and re-read the position EVERY
    # frame; the moment exactly one tile of travel is confirmed, neutralise.
    MAX_FRAMES_PER_TILE = 24           # a FireRed grid step settles in ~16 frames
    SETTLE_FRAMES = 4                  # neutral frames after a confirmed tile
    # A random encounter freezes overworld movement before gMain's battle latch
    # becomes readable.  Do not misclassify that transition as a wall.
    BATTLE_TRANSITION_WAIT_FRAMES = 240
    JITTER_MAX_FRAMES = 24             # neutral pre-walk frames (RNG variety)


def harvest_episode_seed(base_seed_id, run_seed, sequence):
    """A reproducible per-episode seed: deterministic given
    ``(base_seed_id, run_seed, sequence)`` so the SAME run replays the SAME
    sequence, but consecutive episodes (``sequence`` +1) get different jitter
    and a different start direction. Pure, 31-bit."""
    a = int(base_seed_id or 0) & 0xFFFFFFFF
    b = int(run_seed or 0) & 0xFFFFFFFF
    c = int(sequence or 0) & 0xFFFFFFFF
    h = (a * 0x9E3779B1) ^ (b * 0x85EBCA77) ^ (c * 0xC2B2AE3D)
    h ^= (h >> 15)
    h = (h * 0x2545F491) & 0xFFFFFFFF
    h ^= (h >> 13)
    return h & 0x7FFFFFFF


class HarvestResult(dict):
    @property
    def ok(self):
        return bool(self.get("ok"))

    @property
    def encounter_started(self):
        return bool(self.get("encounter_started"))


# --------------------------------------------------------------------------
# pure corridor logic (no emulator — unit-testable)
# --------------------------------------------------------------------------
class CorridorWalker:
    """Decides the next horizontal direction. Never returns a vertical move.
    ``decide`` returns ``None`` when the corridor is blocked on both sides
    (the agent is trapped) — the caller then aborts."""

    def __init__(self, anchor_x, *, max_tiles=HarvesterConfig.MAX_TILES_FROM_ANCHOR,
                 start_dir=_LEFT):
        self.anchor_x = int(anchor_x)
        self.max_tiles = int(max_tiles)
        self.last_dir = start_dir if start_dir in (_LEFT, _RIGHT) else _LEFT
        self.consecutive_blocks = 0

    def decide(self, cur_x):
        cur_x = int(cur_x)
        lo = self.anchor_x - self.max_tiles
        hi = self.anchor_x + self.max_tiles
        if self.consecutive_blocks >= 2:
            return None                       # trapped -> abort
        d = self.last_dir
        # bound is relative to the ORIGINAL anchor, not the last position
        if d == _LEFT and cur_x <= lo:
            d = _RIGHT
        elif d == _RIGHT and cur_x >= hi:
            d = _LEFT
        return d

    def observe(self, *, direction, moved):
        """Record the outcome of the last press."""
        if moved:
            self.consecutive_blocks = 0
            self.last_dir = direction
        else:
            self.consecutive_blocks += 1
            # a blocked press -> try the other way next
            self.last_dir = _OPPOSITE.get(direction, direction)


def corridor_violation(anchor, loc, *, max_tiles=HarvesterConfig.MAX_TILES_FROM_ANCHOR):
    """``None`` if ``loc`` is still a valid corridor tile, else a diagnostic
    string. Checks map identity, the grass row (y) and the ±max_tiles x bound
    against the ORIGINAL anchor."""
    if not loc or loc.get("valid") is False:
        return "location_unreadable"
    if (int(loc.get("map_bank", -1)), int(loc.get("map_id", -1))) != \
            (int(anchor["map_bank"]), int(anchor["map_id"])):
        return "map_changed"
    if int(loc.get("y_pos", -999)) != int(anchor["y_pos"]):
        return "left_grass_row"          # ledge / stairs / vertical move
    if abs(int(loc.get("x_pos", -999)) - int(anchor["x_pos"])) > int(max_tiles):
        return "left_corridor"           # x jump past the ±3 bound
    return None


# --------------------------------------------------------------------------
# emulator-driving harvester
# --------------------------------------------------------------------------
class WildEncounterHarvester:
    """``env`` is a stable_retro env (``env.em``, ``env.buttons``).
    ``location_reader(env) -> {"map_bank","map_id","x_pos","y_pos","valid"}``,
    ``battle_reader(env) -> bool``. Both are injected so tests use fakes and the
    live wiring passes ``firered_ram.read_player_location`` +
    ``battle_state.MainBattleReader``.
    """

    def __init__(self, env, *, location_reader, battle_reader,
                 config=HarvesterConfig, seed_id=None, episode_seed=None):
        self.env = env
        self._loc = location_reader
        self._in_battle = battle_reader
        self.cfg = config
        # episode_seed (spec ZIEL C.5) varies per episode; seed_id is the older
        # single-value fallback. Either seeds the pre-walk jitter + start dir.
        self.episode_seed = int(episode_seed) if episode_seed is not None else (
            int(seed_id) if seed_id is not None else None)
        self._anchor = None
        self._anchor_state = None

    # -- low-level ------------------------------------------------
    def _mask(self, name):
        import numpy as np
        m = [0] * len(self.env.buttons)
        if name in self.env.buttons:
            m[self.env.buttons.index(name)] = 1
        return np.asarray(m, dtype=np.uint8)

    def _one_frame(self, mask):
        self.env.em.set_button_mask(mask)
        self.env.em.step()
        cb = getattr(self.env, "_pkmai_frame_callback", None)
        if cb is not None:
            cb()

    def _step_frames(self, mask, frames):
        import numpy as np
        neutral = np.zeros(len(self.env.buttons), dtype=np.uint8)
        for i in range(int(frames)):
            self._one_frame(mask if i == 0 else neutral)

    def _read_loc(self):
        try:
            return dict(self._loc(self.env) or {})
        except Exception:
            return {"valid": False}

    def _walk_one_tile(self, direction):
        """Hold ``direction`` and re-read the position EVERY frame. Returns:
          ("moved", loc)      -> exactly one tile of travel confirmed
          ("blocked", loc)    -> held the button, no tile change (a wall)
          ("encounter", loc)  -> a wild battle started mid-step
          ("abort", reason)   -> map change / y change / jump > 1 tile /
                                 wrong-direction move / unreadable location
        The button is neutralised the instant a tile change or a battle is seen,
        so the walk never overshoots to a second tile (spec ZIEL A.4).
        """
        start = self._read_loc()
        if start.get("valid") is False or "x_pos" not in start:
            return "abort", "location_unreadable"
        sx, sy = int(start["x_pos"]), int(start["y_pos"])
        sbank, smid = int(start["map_bank"]), int(start["map_id"])
        want = _DIR_SIGN[direction]
        press = self._mask(direction)
        neutral = self._mask("_none_")

        def await_battle_transition(fallback_reason=None):
            """Poll gMain with neutral input while the encounter fade runs.

            Overworld location fields can become invalid or change map before
            the verified battle latch appears.  Such a violation is deferred,
            never discarded: if no battle appears inside the bounded window,
            the original fail-closed reason is returned.
            """
            for _ in range(self.cfg.BATTLE_TRANSITION_WAIT_FRAMES):
                self._one_frame(neutral)
                if self._in_battle(self.env):
                    return "encounter", self._read_loc()
            if fallback_reason is not None:
                return "abort", fallback_reason
            return None

        for _ in range(self.cfg.MAX_FRAMES_PER_TILE):
            self._one_frame(press)
            if self._in_battle(self.env):
                self._one_frame(neutral)
                return "encounter", self._read_loc()
            loc = self._read_loc()
            if loc.get("valid") is False:
                return await_battle_transition("location_unreadable")
            if (int(loc.get("map_bank", -1)), int(loc.get("map_id", -1))) != (sbank, smid):
                return await_battle_transition("map_changed")
            if int(loc.get("y_pos", sy)) != sy:
                return await_battle_transition("left_grass_row")
            dx = int(loc.get("x_pos", sx)) - sx
            if dx != 0:
                self._one_frame(neutral)                   # stop before tile #2
                if abs(dx) > 1:
                    return "abort", "jump_gt_1_tile"
                if (dx > 0) != (want > 0):
                    return "abort", "wrong_direction_move"
                self._step_frames(neutral, self.cfg.SETTLE_FRAMES)
                return "moved", self._read_loc()
        # No tile movement yet can also mean that a random encounter has frozen
        # the overworld while the battle scene is fading in.  Poll the verified
        # latch with neutral input before declaring a real collision.
        self._one_frame(neutral)
        transition = await_battle_transition()
        if transition is not None:
            return transition
        loc = self._read_loc()
        if loc.get("valid") is False:
            return "abort", "location_unreadable"
        if (int(loc.get("map_bank", -1)), int(loc.get("map_id", -1))) != (sbank, smid):
            return "abort", "map_changed"
        if int(loc.get("y_pos", sy)) != sy:
            return "abort", "left_grass_row"
        dx = int(loc.get("x_pos", sx)) - sx
        if dx:
            if abs(dx) > 1:
                return "abort", "jump_gt_1_tile"
            if (dx > 0) != (want > 0):
                return "abort", "wrong_direction_move"
            return "moved", loc
        # Held the direction and observed no battle transition or movement.
        return "blocked", self._read_loc()

    # -- public --------------------------------------------------
    def capture_anchor(self):
        """Snapshot the current tile as the anchor. Fail-closed if the location
        cannot be read or the tile is not the intended grass anchor."""
        loc = self._read_loc()
        if loc.get("valid") is False or "x_pos" not in loc:
            return HarvestResult(ok=False, reason="anchor_location_unreadable")
        self._anchor = {k: int(loc[k]) for k in
                        ("map_bank", "map_id", "x_pos", "y_pos")}
        try:
            self._anchor_state = self.env.em.get_state()
        except Exception:
            return HarvestResult(ok=False, reason="anchor_state_unavailable")
        return HarvestResult(ok=True, anchor=dict(self._anchor))

    def harvest(self):
        """Walk the corridor until a wild battle starts. Returns a
        :class:`HarvestResult`. Does NOT play the battle — the caller's battle
        driver takes over when ``encounter_started`` is True."""
        if self._anchor is None:
            r = self.capture_anchor()
            if not r.ok:
                return r
        if self._in_battle(self.env):
            return HarvestResult(ok=False, reason="already_in_battle",
                                 anchor=dict(self._anchor))

        # spec ZIEL C.5: per-episode jitter + start direction from episode_seed.
        jitter, start_dir = 0, _LEFT
        if self.episode_seed is not None:
            if self.cfg.JITTER_MAX_FRAMES > 0:
                jitter = self.episode_seed % (self.cfg.JITTER_MAX_FRAMES + 1)
                self._step_frames(self._mask("_none_"), jitter)
            start_dir = _LEFT if (self.episode_seed >> 5) % 2 == 0 else _RIGHT

        walker = CorridorWalker(self._anchor["x_pos"],
                                max_tiles=self.cfg.MAX_TILES_FROM_ANCHOR,
                                start_dir=start_dir)
        deadline = time.time() + self.cfg.MAX_WALL_CLOCK_S
        move_trace = []
        self._movement_attempts = []
        for steps in range(1, self.cfg.MAX_STEPS + 1):
            if time.time() > deadline:
                return self._abort("wall_clock_exceeded", steps, jitter, move_trace)
            loc = self._read_loc()
            v = corridor_violation(self._anchor, loc,
                                   max_tiles=self.cfg.MAX_TILES_FROM_ANCHOR)
            if v is not None:
                return self._abort(v, steps, jitter, move_trace)

            direction = walker.decide(int(loc["x_pos"]))
            if direction is None:
                return self._abort("corridor_blocked_both_sides", steps, jitter,
                                   move_trace)

            kind, payload = self._walk_one_tile(direction)
            attempt = {
                "direction": direction,
                "kind": kind,
                "from_x": int(loc.get("x_pos", -1)),
                "from_y": int(loc.get("y_pos", -1)),
            }
            if isinstance(payload, dict):
                attempt["to_x"] = int(payload.get("x_pos", -1))
                attempt["to_y"] = int(payload.get("y_pos", -1))
            else:
                attempt["detail"] = str(payload)
            self._movement_attempts.append(attempt)
            moved_symbol = "<" if direction == _LEFT else ">"
            blocked_symbol = "l" if direction == _LEFT else "r"
            move_trace.append(moved_symbol if kind == "moved"
                              else blocked_symbol if kind == "blocked" else kind)
            if kind == "abort":
                return self._abort(payload, steps, jitter, move_trace)
            if kind == "encounter":
                enc = payload if isinstance(payload, dict) else self._read_loc()
                return HarvestResult(
                    ok=True, encounter_started=True, steps=steps, jitter=jitter,
                    move_trace="".join(move_trace), episode_seed=self.episode_seed,
                    movement_attempts=list(self._movement_attempts),
                    anchor=dict(self._anchor),
                    encounter_tile={
                        "map_bank": self._anchor["map_bank"],
                        "map_id": self._anchor["map_id"],
                        "x_pos": int(enc.get("x_pos", self._anchor["x_pos"])),
                        "y_pos": int(enc.get("y_pos", self._anchor["y_pos"]))})

            after = payload if isinstance(payload, dict) else {}
            v2 = corridor_violation(self._anchor, after,
                                    max_tiles=self.cfg.MAX_TILES_FROM_ANCHOR)
            if v2 is not None:
                return self._abort(v2, steps, jitter, move_trace)
            walker.observe(direction=direction, moved=(kind == "moved"))

        return self._abort("no_encounter_within_step_limit",
                           self.cfg.MAX_STEPS, jitter, move_trace)

    def restore_anchor(self):
        """Deterministically put the emulator back on the anchor tile. Prefers
        the unchanged anchor savestate; verifies ``(bank,map,x,y)`` afterwards
        and is fail-closed on any drift (spec ZIEL A.3)."""
        if self._anchor is None or self._anchor_state is None:
            return HarvestResult(ok=False, reason="no_anchor_to_restore")
        try:
            self.env.em.set_state(self._anchor_state)
            self.env.em.step()
        except Exception as exc:
            return HarvestResult(ok=False, reason=f"set_state_failed:{exc}")
        loc = self._read_loc()
        drift = corridor_violation(self._anchor, loc, max_tiles=0)
        if drift is not None or (
                int(loc.get("x_pos", -1)) != self._anchor["x_pos"]
                or int(loc.get("y_pos", -1)) != self._anchor["y_pos"]):
            return HarvestResult(ok=False, reason="anchor_restore_drift",
                                 expected=dict(self._anchor),
                                 got={k: loc.get(k) for k in
                                      ("map_bank", "map_id", "x_pos", "y_pos")})
        return HarvestResult(ok=True, restored=True, anchor=dict(self._anchor))

    def _abort(self, reason, steps, jitter, move_trace=None):
        restore = self.restore_anchor() if self._anchor_state is not None else \
            HarvestResult(ok=False, reason="no_anchor")
        return HarvestResult(ok=False, encounter_started=False, reason=reason,
                             steps=steps, jitter=jitter,
                             move_trace="".join(move_trace or []),
                             episode_seed=self.episode_seed,
                             movement_attempts=list(getattr(
                                 self, "_movement_attempts", [])),
                             anchor=dict(self._anchor) if self._anchor else None,
                             restore_ok=bool(restore.get("ok")),
                             restore_reason=restore.get("reason"))
