"""
V20 non-farmable target shaping (brief section 6, 22).

The old ``closer +0.20 / farther -0.20`` per-step shaping produced A<->B
oscillation loops.  The V20 rule:

  * POSITIVE shaping is paid ONLY for a NEW BEST distance within the current
    objective/episode.  Returning to a distance already achieved pays nothing.
    Therefore  A -> B -> A -> B -> A  can never repeatedly earn positive
    progress.
  * NEGATIVE shaping is tiny and only fires when the agent moves meaningfully
    farther from the best route than a small margin.  Day-to-day pressure comes
    from GAMEPLAY_STEP_COST and the loop guard, not a dense +/- signal storm.

Concept:

    if new_d < best_target_distance:
        improvement = best_target_distance - new_d
        reward += TARGET_PROGRESS_REWARD * improvement
        best_target_distance = new_d
"""
from __future__ import annotations

TARGET_PROGRESS_REWARD = 0.05
TARGET_BACKTRACK_PENALTY = -0.01
# Only penalise once the agent is more than this many tiles worse than its best.
TARGET_BACKTRACK_MARGIN = 3

EVENT_PROGRESS = "route_progress_best"
EVENT_BACKTRACK = "route_backtrack"


class TargetShaper:
    def __init__(self,
                 progress_reward=TARGET_PROGRESS_REWARD,
                 backtrack_penalty=TARGET_BACKTRACK_PENALTY,
                 backtrack_margin=TARGET_BACKTRACK_MARGIN):
        self.progress_reward = float(progress_reward)
        self.backtrack_penalty = float(backtrack_penalty)
        self.backtrack_margin = int(backtrack_margin)
        self.best_target_distance = None
        self._objective_key = None
        self._best_by_objective = {}

    # -- lifecycle ---------------------------------------------------
    def start_objective(self, key, initial_distance=None):
        """Begin a fresh objective (new episode, new bottleneck, new target).

        ``key`` identifies the objective; calling ``update`` with a different
        key auto-starts a new objective so callers cannot accidentally farm
        across target changes.
        """
        self._objective_key = key
        self.best_target_distance = (
            None if initial_distance is None else int(initial_distance)
        )

    def reset(self):
        self.best_target_distance = None
        self._objective_key = None
        self._best_by_objective = {}

    # -- per step --------------------------------------------------
    def update(self, key, new_distance):
        """Return ``(reward, event_or_None)`` for this step.

        ``new_distance`` is the graph/Manhattan distance to the known target.
        A positive number is only ever returned on a strict new best.
        """
        if new_distance is None:
            return 0.0, None
        new_distance = int(new_distance)

        if key != self._objective_key:
            # Save the previous objective before switching maps. A return to
            # an already visited objective retains its high-watermark.
            if self._objective_key is not None:
                self._best_by_objective[self._objective_key] = self.best_target_distance
            previous_best = self._best_by_objective.get(key)
            self.start_objective(key, previous_best if previous_best is not None else new_distance)
            return 0.0, None

        if self.best_target_distance is None:
            self.best_target_distance = new_distance
            return 0.0, None

        if new_distance < self.best_target_distance:
            improvement = self.best_target_distance - new_distance
            self.best_target_distance = new_distance
            return self.progress_reward * improvement, EVENT_PROGRESS

        if new_distance > self.best_target_distance + self.backtrack_margin:
            return self.backtrack_penalty, EVENT_BACKTRACK

        # Between best and best+margin: neutral. This is what kills the loop.
        return 0.0, None
