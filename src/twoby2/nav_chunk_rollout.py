"""Navigation PPO updates decoupled from episode end.

The navigation episode horizon may grow to 163 840 route-steps, but PPO still
updates every ``PPO_N_STEPS`` (512) real navigation decisions. A chunk boundary
is NOT a terminal and NOT a truncation: the env keeps running, GAE bootstraps
from the last value, and only a *real* ``terminated`` zeroes the bootstrap.

This module owns the accounting so the seam in the live env / trainer is a
thin call. It never counts:
  * battle steps (the wrapper plays the battle internally),
  * the internal button presses of a battle macro,
  * watcher / rendering steps.
"""
from __future__ import annotations

NAV_PPO_N_STEPS = 512   # mirrors train.PPO_N_STEPS — kept independent of horizon


class NavChunkedRollout:
    """Feed it one *navigation decision* at a time. It tells you when a rollout
    chunk is full and how to bootstrap it."""

    def __init__(self, *, n_steps=NAV_PPO_N_STEPS):
        self.n_steps = int(n_steps)
        self.decisions_in_chunk = 0
        self.total_nav_decisions = 0
        self.chunks_emitted = 0
        self.updates = 0
        self.episode_route_steps = 0

    # -- feed -----------------------------------------------------
    def on_navigation_decision(self):
        """Exactly one overworld PPO action was taken. Returns True when the
        chunk is now full and should be processed."""
        self.decisions_in_chunk += 1
        self.total_nav_decisions += 1
        self.episode_route_steps += 1
        return self.decisions_in_chunk >= self.n_steps

    def on_battle_subepisode(self, battle_decisions, battle_button_presses):
        """A battle happened. NOTHING here touches the navigation decision
        counter or the chunk — battle decisions and their button presses are
        the battle system's, not navigation's."""
        return {"nav_decisions_added": 0, "chunk_advanced": False,
                "battle_decisions": int(battle_decisions),
                "battle_button_presses": int(battle_button_presses)}

    def on_watcher_step(self):
        """Watcher / rendering — never increments a learner counter."""
        return {"nav_decisions_added": 0, "updates_added": 0}

    # -- chunk boundary ----------------------------------------
    def close_chunk(self, *, episode_terminated, episode_truncated,
                    last_value, terminal_value=None):
        """Process a full (or forced) chunk.

        Returns the bootstrap decision:
          bootstrap_value : the value to bootstrap the advantage of the last
                            step with.
          reset_env       : whether the ENV should be reset (only on a real
                            terminal).
          is_chunk_boundary_only : True when the episode simply continues.
        """
        real_terminal = bool(episode_terminated)
        time_limit = bool(episode_truncated) and not real_terminal
        chunk_only = not real_terminal and not time_limit

        if real_terminal:
            bootstrap = 0.0
        elif time_limit:
            # real TimeLimit truncation: bootstrap from the terminal/final obs
            bootstrap = float(terminal_value if terminal_value is not None
                              else last_value)
        else:
            # plain chunk boundary mid-episode: bootstrap from the current value,
            # episode keeps running, env is NOT reset
            bootstrap = float(last_value)

        self.chunks_emitted += 1
        self.updates += 1
        self.decisions_in_chunk = 0
        if real_terminal or time_limit:
            self.episode_route_steps = 0

        return {
            "bootstrap_value": bootstrap,
            "reset_env": real_terminal,
            "is_chunk_boundary_only": chunk_only,
            "is_real_terminal": real_terminal,
            "is_time_limit_truncation": time_limit,
            "updates": self.updates,
            "total_nav_decisions": self.total_nav_decisions,
        }

    # -- horizon -------------------------------------------------
    def episode_truncated_by_horizon(self, horizon):
        """True once the running episode has used its full navigation horizon.
        A horizon truncation IS a real TimeLimit truncation (bootstrap from the
        final obs), NOT a chunk boundary."""
        return self.episode_route_steps >= int(horizon)
