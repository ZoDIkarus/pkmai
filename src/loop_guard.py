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


# V20 (brief section 10 / 22): explicit short-cycle detection.
#
# Detects  A B A B A B  (period 2) and  A B C A B C  (period 3) style loops from
# recent overworld path history.  Battle movement, menus and legitimate one-time
# map transitions are excluded by the caller passing ``active=False`` for those
# steps.  Behaviour on a sustained short cycle:
#   * suppress positive navigation shaping (caller checks ``suppress_shaping``)
#   * apply a small escalating penalty (LOCAL_LOOP_PENALTY_START .. _MAX)
#   * after a long time in the same tiny cycle, ask the caller to truncate.
LOCAL_LOOP_PENALTY_START = -0.05
LOCAL_LOOP_PENALTY_MAX = -0.25


class ShortCycleGuard:
    def __init__(self,
                 history=18,
                 min_repeats=3,
                 max_period=3,
                 penalty_start=LOCAL_LOOP_PENALTY_START,
                 penalty_max=LOCAL_LOOP_PENALTY_MAX,
                 escalate_every=12,
                 truncate_after=600):
        self.history = int(history)
        self.min_repeats = int(min_repeats)
        self.max_period = int(max_period)
        self.penalty_start = float(penalty_start)
        self.penalty_max = float(penalty_max)
        self.escalate_every = int(escalate_every)
        self.truncate_after = int(truncate_after)

        self.positions = deque(maxlen=self.history)
        self.progress = None
        self.cycle_steps = 0
        self.in_cycle = False

    def _detect_period(self):
        seq = list(self.positions)
        for period in range(1, self.max_period + 1):
            need = period * self.min_repeats
            if len(seq) < need:
                continue
            tail = seq[-need:]
            base = tail[:period]
            if all(tail[i] == base[i % period] for i in range(need)) \
                    and len(set(base)) >= 2:
                return period
        return 0

    def update(self, position, progress, active=True):
        """Feed one step. Returns a dict:
            {"penalty": float, "suppress_shaping": bool, "truncate": bool,
             "cycle": bool, "period": int}
        """
        if progress != self.progress:
            self.progress = progress
            self.positions.clear()
            self.cycle_steps = 0
            self.in_cycle = False

        if not active or position is None:
            # Do not let battles / warps break the running counter hard, but
            # stop feeding positions so the pattern check stays clean.
            return {"penalty": 0.0, "suppress_shaping": self.in_cycle,
                    "truncate": False, "cycle": self.in_cycle, "period": 0}

        # RAM positions are cached for several agent actions. Detect cycles
        # in actual moves, while counting penalties/time in agent actions.
        if not self.positions or self.positions[-1] != position:
            self.positions.append(position)
        period = self._detect_period()

        if period == 0:
            # One clean non-cycling step is enough to release.
            if self.in_cycle and len(set(list(self.positions)[-4:])) >= 4:
                self.in_cycle = False
                self.cycle_steps = 0
            return {"penalty": 0.0, "suppress_shaping": False,
                    "truncate": False, "cycle": False, "period": 0}

        self.in_cycle = True
        self.cycle_steps += 1

        steps_over = max(0, self.cycle_steps - period * self.min_repeats)
        escalation = steps_over // max(1, self.escalate_every)
        penalty = self.penalty_start + escalation * self.penalty_start
        penalty = max(self.penalty_max, penalty)  # both negative; _MAX is floor

        truncate = self.cycle_steps >= self.truncate_after
        return {"penalty": penalty, "suppress_shaping": True,
                "truncate": truncate, "cycle": True, "period": period}


# ---------------------------------------------------------------------------
# Larger local round-trips (ledge loops).
#
# `ShortCycleGuard` only catches strict period<=3 tile cycles. A Route-1 ledge
# loop is bigger and edge-sequence shaped: `rechts hoch -> Absatz runter ->
# rechts hoch` visits ~6-12 tiles and repeats a directed-edge sequence while
# making no real progress. `RegionLoopGuard` measures stagnation since the last
# genuine advance:
#
#   * feed `progress_event()` on {global-new topology, new directed-distance
#     highwater, stage/story progress} -> the stagnation window resets;
#   * feed `update(position, action, distance)` every navigation step.
#
# A region loop is flagged when, since the last progress event:
#   * enough steps elapsed, AND
#   * few distinct tiles were visited, AND
#   * a directed-edge subsequence repeated, AND
#   * the agent returned to a prior (position, directed_distance) state.
#
# A legitimate detour around an obstacle keeps lowering the directed distance,
# which fires `progress_event()` and clears the guard - so honest detours are
# never penalised.
REGION_LOOP_MIN_STEPS = 260
REGION_LOOP_MAX_TILES = 16
REGION_LOOP_TRUNCATE_STEPS = 900
REGION_LOOP_PENALTY_START = -0.03
REGION_LOOP_PENALTY_MAX = -0.30


class RegionLoopGuard:
    def __init__(self,
                 min_steps=REGION_LOOP_MIN_STEPS,
                 max_tiles=REGION_LOOP_MAX_TILES,
                 truncate_steps=REGION_LOOP_TRUNCATE_STEPS,
                 penalty_start=REGION_LOOP_PENALTY_START,
                 penalty_max=REGION_LOOP_PENALTY_MAX,
                 edge_history=40,
                 escalate_every=60):
        self.min_steps = int(min_steps)
        self.max_tiles = int(max_tiles)
        self.truncate_steps = int(truncate_steps)
        self.penalty_start = float(penalty_start)
        self.penalty_max = float(penalty_max)
        self.escalate_every = int(escalate_every)
        self._edge_hist = deque(maxlen=int(edge_history))
        self.reset_window()

    def reset_window(self):
        self._steps = 0
        self._tiles = set()
        self._states = set()          # (position, directed_distance)
        self._edge_hist.clear()
        self._in_loop = False
        self._loop_steps = 0

    # a real advance clears the stagnation window
    def progress_event(self):
        self.reset_window()

    def _edge_subsequence_repeats(self):
        seq = list(self._edge_hist)
        for period in range(2, 9):
            if len(seq) < period * 3:
                continue
            tail = seq[-period * 3:]
            base = tail[:period]
            if all(tail[i] == base[i % period] for i in range(period * 3)) \
                    and len(set(base)) >= 2:
                return period
        return 0

    def update(self, position, action, directed_distance, *, active=True):
        """Returns ``{"loop", "kind", "suppress_shaping", "penalty",
        "truncate", "period"}``."""
        idle = {"loop": self._in_loop, "kind": None,
                "suppress_shaping": self._in_loop, "penalty": 0.0,
                "truncate": False, "period": 0}
        if not active or position is None:
            return idle

        self._steps += 1
        self._tiles.add(tuple(position))
        if action is not None:
            self._edge_hist.append((tuple(position), int(action)))
        state = (tuple(position), directed_distance)
        revisited_state = state in self._states
        self._states.add(state)

        period = self._edge_subsequence_repeats()
        region = (self._steps >= self.min_steps
                  and len(self._tiles) <= self.max_tiles
                  and (period >= 2 or revisited_state))

        if not region:
            if self._in_loop and len(self._tiles) > self.max_tiles:
                # broke out into new ground
                self.reset_window()
            return idle

        self._in_loop = True
        self._loop_steps += 1
        kind = "ledge_loop" if period >= 2 else "region_loop"
        escalation = self._loop_steps // max(1, self.escalate_every)
        penalty = max(self.penalty_max,
                      self.penalty_start * (1 + escalation))
        return {"loop": True, "kind": kind, "suppress_shaping": True,
                "penalty": penalty,
                "truncate": self._loop_steps >= self.truncate_steps,
                "period": period}
