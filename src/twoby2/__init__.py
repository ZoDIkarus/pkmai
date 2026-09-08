"""2×2 Navigation / Battle architecture — Phase 2+ preparation.

Everything in this package is **isolated and feature-gated**. Nothing here is
imported by ``pokemon_env.py`` / ``train.py`` / ``watch.py`` /
``watcher_runtime.py`` / ``web_stream.py``. It builds the state machines and
policy logic the later phases need, all as pure functions over plain dicts —
no torch, no stable_retro, no emulator.

Live activation stays blocked until:
  * ``gBattleMons`` / ``gBattlerPartyIndexes`` / ``gActionSelectionCursor`` /
    ``gMoveSelectionCursor`` are reproducibly verified for the BPRD ROM
    (so ``battle_ram.battle_snapshot_ready_for_execution`` can return True),
  * the macro executor + isolated battle env + one-time migration exist,
  * isolated emulator smoke tests pass.

Until then every gate in :data:`FEATURES` is ``False`` and
:func:`twoby2.activation.assert_not_live` guards the seams.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Feature gates — ALL default OFF. Flipping one is a deliberate, reviewed act;
# the code paths behind them must already be tested first.
# ---------------------------------------------------------------------------
FEATURES = {
    # Phase 2: run the isolated Battle env / Battle-PPO learner at all.
    "battle_env": False,
    # Phase 2: let the macro executor press real buttons in a live battle.
    "battle_executor_live": False,
    # Phase 3: route Overworld->nav / Battle->battle-champion in the live path.
    "battle_router_live": False,
    # Phase 3: navigation-only wrapper plays the battle sub-episode internally.
    "nav_battle_wrapper": False,
    # Phase 1.5: adaptive navigation horizon curriculum drives episode length.
    "adaptive_nav_horizon": False,
    # Phase 4: watcher consults the battle champion for in-battle actions.
    "watcher_battle_champion": False,
    # Phase 5: one-time single-PPO -> 2×2 runtime migration is allowed to run.
    "runtime_migration": False,
}

_LIVE_GATES = frozenset({
    "battle_env", "battle_executor_live", "battle_router_live",
    "nav_battle_wrapper", "adaptive_nav_horizon",
    "watcher_battle_champion",
})
_FROZEN_TRUE = (_LIVE_GATES if os.environ.get("PKMAI_TWOBY2_LIVE") == "1"
                else frozenset())


def feature_enabled(name):
    """Read a gate. Unknown name -> False (fail-closed)."""
    if name in _FROZEN_TRUE:
        return True
    return bool(FEATURES.get(name, False))


def all_gates_closed():
    return not any(FEATURES.values())


__all__ = ["FEATURES", "feature_enabled", "all_gates_closed"]
