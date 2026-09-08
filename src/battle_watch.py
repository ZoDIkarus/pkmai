"""Visible battle training worker — the 9th battle worker (`PKMai – BATTLE`).

This is not an extra worker: worker 9 belongs to the shared 9-worker learner.
It runs the **same** `BattleEnv` / RAM observation
/ action mask / `MacroExecutor` / reward schema as a headless worker and its
rollouts go to the **same central Battle learner** — it has no own optimizer
and no own model.

Rendering:
  * smooth visible ~60 FPS output;
  * render FPS produces NO extra actions (the env only steps on policy
    decisions — the renderer just displays the current frame);
  * rendering never changes state, seed, actions or reward.

Policy pinning:
  * the acting policy (Battle learner snapshot) is pinned for at least a full
    battle / rollout; no swap mid-battle.

Fail-closed:
  * while the real `EmulatorBattleDriver` is blocked (battle-menu RAM
    unverified), this worker refuses to publish anything as real training.
    `run()` raises unless the emulator driver is available AND the feature gate
    is on. A simulated battle is never mislabelled as real training output.
"""
from __future__ import annotations

from twoby2 import feature_enabled

WINDOW_TITLE = "PKMai – BATTLE"
TARGET_FPS = 60


class VisibleBattleWorker:
    """Config + guarded runner for the visible battle worker."""

    def __init__(self, *, learner_rollout_sink, policy_provider,
                 scenario_provider, driver=None, fps=TARGET_FPS):
        # learner_rollout_sink: callable(list_of_transitions) -> None
        #   (the CENTRAL battle learner's rollout buffer; NOT a local one)
        self.rollout_sink = learner_rollout_sink
        # policy_provider: callable() -> a pinned policy snapshot (learner)
        self.policy_provider = policy_provider
        # scenario_provider: callable() -> the visible watcher scenario
        self.scenario_provider = scenario_provider
        # gate ON + a live env -> real EmulatorBattleDriver (same as headless);
        # otherwise the fail-closed guards below keep this worker inert.
        if driver is None:
            from twoby2.live_integration import battle_driver_for
            driver = battle_driver_for(getattr(self, "_live_env", None))
        self.driver = driver
        self.fps = int(fps)
        self._pinned_policy = None
        self._own_optimizer = None   # deliberately always None
        self._own_model = None       # deliberately always None

    # -- guards --------------------------------------------------
    def is_real_training_ready(self):
        """Only real when the emulator battle driver exists AND the gate is on.
        Otherwise this worker must not contribute training data."""
        from battle_env import SimulatedBattleDriver
        real_driver = self.driver is not None and not isinstance(
            self.driver, SimulatedBattleDriver)
        return bool(real_driver and feature_enabled("battle_env"))

    def render_is_side_effect_free(self):
        """Contract marker: the render loop reads the current frame buffer and
        blits it; it never calls ``env.step`` and never touches RNG."""
        return True

    def uses_central_learner_only(self):
        return self._own_optimizer is None and self._own_model is None

    # -- lifecycle ---------------------------------------------
    def pin_policy_for_battle(self):
        """Snapshot the learner policy for the whole upcoming battle."""
        self._pinned_policy = self.policy_provider()
        return self._pinned_policy

    def collect_battle(self):
        """Play ONE battle with the pinned policy and return its transitions
        for the central learner. Raises while fail-closed."""
        if not self.is_real_training_ready():
            raise RuntimeError(
                "visible battle worker is fail-closed: the real EmulatorBattleDriver "
                "is unavailable / gated. A simulated battle must not be published "
                "as real training. Blocked on battle-menu RAM verification.")
        if self._pinned_policy is None:
            self.pin_policy_for_battle()
        # real implementation: run BattleEnv with self.driver + MacroExecutor,
        # acting with self._pinned_policy, rendering at self.fps, collecting
        # (obs, action, reward, done, value, logp) transitions.
        raise NotImplementedError(
            "runnable visible battle loop is gated until the emulator driver "
            "is verified")

    def run(self):
        while True:                                    # pragma: no cover
            self.pin_policy_for_battle()
            transitions = self.collect_battle()        # raises while gated
            self.rollout_sink(transitions)             # -> CENTRAL learner


def worker_descriptor():
    """What start/stop/status shows for this worker."""
    return {
        "system": "battle",
        "render": "visible",
        "window_title": WINDOW_TITLE,
        "fps": TARGET_FPS,
        "uses": "central battle learner (no own optimizer / model)",
        "policy_pin": "per battle (no mid-battle swap)",
        "counts_toward": "BATTLE_WORKERS (the 9th worker, not an extra)",
        "fail_closed": True,
    }
