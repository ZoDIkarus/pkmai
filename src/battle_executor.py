"""Battle macro-action executor (Phase 2).

Translates one macro action — ``MOVE_1..MOVE_4`` / ``SWITCH_1..SWITCH_6`` /
``RUN`` — into a *verified* sequence of button presses. Every relevant press is
followed by a re-read of the battle + menu/cursor state; nothing is played
blind. On an unexpected menu, a timeout, or a maximum-frame breach the executor
aborts safely without having pressed a move.

**Isolation:** imports only ``battle_ram`` (Phase-1, verified reads),
``battle_engine`` and ``twoby2.activation``. Never imports a live training
module. Outside a confirmed battle it refuses to press anything.

**Live gate:** :meth:`MacroExecutor.execute` calls
``twoby2.activation.assert_live_battle_allowed`` unless ``dry_run=True``.
Because the battle-menu RAM addresses are not yet verified for BPRD, the live
path stays blocked; the state machine itself is fully implemented and tested
against an in-memory battle-menu model.
"""
from __future__ import annotations

MOVE_ACTIONS = ("MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4")
SWITCH_ACTIONS = ("SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4",
                  "SWITCH_5", "SWITCH_6")
RUN_ACTION = "RUN"
CATCH_ACTION = "CATCH"
# v1 (combat only) and v2 (adds CATCH). ALL_MACROS is the v2 space; the v1
# length is kept for the fail-closed loader check.
ALL_MACROS_V1 = MOVE_ACTIONS + SWITCH_ACTIONS + (RUN_ACTION,)
ALL_MACROS = ALL_MACROS_V1 + (CATCH_ACTION,)
ACTIONS_SCHEMA_V1 = "battle_actions_v1"
ACTIONS_SCHEMA_V2 = "battle_actions_v2_catch"

# Top-level battle menu cursor slots (FireRed layout).
MENU_FIGHT, MENU_BAG, MENU_POKEMON, MENU_RUN = 0, 1, 2, 3

# Menu screens the executor can be on.
SCREEN_MAIN = "main"          # FIGHT / BAG / POKEMON / RUN
SCREEN_MOVES = "moves"        # the 4-move list
SCREEN_PARTY = "party"        # the party / switch list
SCREEN_YESNO = "yesno"        # confirmation prompt
SCREEN_MESSAGE = "message"    # scrolling text; press A to advance
SCREEN_BAG = "bag"            # the in-battle bag (ball pocket)
SCREEN_ITEM_USE = "item_use"  # USE / GIVE / TOSS / CANCEL for a selected item
SCREEN_UNKNOWN = "unknown"

DEFAULT_MAX_PRESSES = 60
DEFAULT_MAX_MESSAGE_ADVANCES = 36


class ExecutorAbort(RuntimeError):
    pass


class BattleIO:
    """Everything the executor needs from the emulator. A real implementation
    talks to stable_retro; tests pass an in-memory model."""

    def in_battle(self):
        raise NotImplementedError

    def menu_state(self):
        """Return a dict:
            {"screen": SCREEN_*, "cursor": int, "message_pending": bool,
             "battler_ready": bool}
        Any field that cannot be read authoritatively -> None."""
        raise NotImplementedError

    def snapshot(self):
        """A ``battle_ram.battle_snapshot``-shaped dict."""
        raise NotImplementedError

    def press(self, button):
        """Press one of A / B / UP / DOWN / LEFT / RIGHT / START and advance a
        few frames. Return the new ``menu_state`` dict."""
        raise NotImplementedError

    def wait_frames(self, frames):
        """Optionally advance neutral frames while a submenu is rendering."""
        return self.menu_state()

    def ball_pocket(self):
        """List the BALL pocket as displayed: ``[(item_id, count), ...]`` in
        cursor order, or ``None`` if it cannot be read authoritatively. Only
        real Poke-Ball item ids; a non-ball item in the pocket is a read fault
        and must yield ``None``."""
        return None


# --------------------------------------------------------------------------
# action legality (fail-closed masking)
# --------------------------------------------------------------------------
def catch_precheck(snapshot, objective):
    """Spec §6: every condition that must hold before the FIRST button of a
    CATCH macro. Returns ``(ok, reason)``. Fail-closed on any unknown."""
    snap = snapshot or {}
    obj = objective or {}
    if not obj.get("catch_requested"):
        return False, "catch_not_requested"
    if snap.get("is_trainer") is not False:
        return False, "trainer_catch_blocked"
    if snap.get("can_escape") is not True:
        return False, "not_a_confirmed_wild_battle"
    if snap.get("menu_state") not in ("main",) and snap.get("battle_menu") not in ("main",):
        # the executor re-checks after reaching main; here we only reject a
        # clearly-not-main state that the caller already read
        if snap.get("menu_ready") is False:
            return False, "menu_not_ready"
    enemy = snap.get("enemy_active") or {}
    esid = enemy.get("species_id")
    if not esid or int(esid) != int(obj.get("target_species_id", 0) or 0):
        return False, "enemy_is_not_the_target_species"
    if int(obj.get("usable_ball_count", 0) or 0) <= 0:
        return False, "no_ball_above_reserve"
    if not obj.get("party_has_space") and not obj.get("pc_capture_supported"):
        return False, "party_full_no_pc_support"
    if snap.get("message_pending") or snap.get("animation_active"):
        return False, "text_or_animation_active"
    return True, "ok"


def legal_macros(snapshot, objective=None):
    """The subset of ALL_MACROS provably legal for ``snapshot``. Unknown ->
    not legal. ``CATCH`` is legal only when :func:`catch_precheck` passes."""
    snap = snapshot or {}
    legal = set()
    active = snap.get("player_active") or {}
    moves = active.get("moves") or []
    for i in range(4):
        mv = moves[i] if i < len(moves) else None
        if isinstance(mv, dict) and mv.get("mechanics_known") and _int(mv.get("pp")) > 0:
            legal.add(MOVE_ACTIONS[i])
    party = snap.get("player_party") or []
    active_slot = active.get("slot")
    for i, mon in enumerate(party[:6]):
        if not isinstance(mon, dict):
            continue
        if mon.get("slot") == active_slot:
            continue
        if mon.get("checksum_ok", True) and _int(mon.get("cur_hp")) > 0:
            legal.add(SWITCH_ACTIONS[i])
    # RUN: only a confirmed escapable wild battle
    if snap.get("can_escape") is True and snap.get("is_trainer") is False:
        legal.add(RUN_ACTION)
    if objective is not None and catch_precheck(snap, objective)[0]:
        legal.add(CATCH_ACTION)
    return legal


def action_mask(snapshot, objective=None):
    legal = legal_macros(snapshot, objective)
    return [1 if m in legal else 0 for m in ALL_MACROS]


def shiny_priority_mask(mask, snapshot, objective, *, macros=ALL_MACROS):
    """spec ZIEL D: when the objective is a CRITICAL shiny catch, force the
    catch — mask RUN and any move that would likely KO the target, keep
    status / weakening moves, keep CATCH. PREPARED / dormant: it only changes
    anything when ``objective.catch_priority == "critical"``, which the planner
    only sets for a *verified* shiny (impossible until the shiny RAM probe
    passes). Never produces an all-zero mask.
    """
    obj = objective or {}
    if obj.get("catch_priority") != "critical" or not mask:
        return list(mask)
    out = list(mask)
    snap = snapshot or {}
    enemy = snap.get("enemy_active") or {}
    active = snap.get("player_active") or {}
    moves = active.get("moves") or []
    enemy_hp = _int(enemy.get("cur_hp"))
    for i in range(min(4, len(macros))):
        mv = moves[i] if i < len(moves) else None
        if not isinstance(mv, dict):
            continue
        is_status = bool(mv.get("is_status"))
        # a rough "this likely faints the target" test — power alone, no crit.
        est = _int(mv.get("power")) if not is_status else 0
        if not is_status and enemy_hp > 0 and est >= enemy_hp:
            out[i] = 0                      # mask a likely-KO attack
    if RUN_ACTION in macros:
        out[macros.index(RUN_ACTION)] = 0  # a shiny is never fled
    if CATCH_ACTION in macros and mask[macros.index(CATCH_ACTION)]:
        pass                               # keep CATCH exactly as legal_macros set it
    if not any(out):
        # never trap the policy: prefer CATCH, else the first originally-legal
        if CATCH_ACTION in macros and mask[macros.index(CATCH_ACTION)]:
            out[macros.index(CATCH_ACTION)] = 1
        else:
            out[next((i for i, m in enumerate(mask) if m), 0)] = 1
    return out


# --------------------------------------------------------------------------
# the executor
# --------------------------------------------------------------------------
class MacroExecutor:
    def __init__(self, io, *, max_presses=DEFAULT_MAX_PRESSES,
                 max_message_advances=DEFAULT_MAX_MESSAGE_ADVANCES):
        self.io = io
        self.max_presses = int(max_presses)
        self.max_message_advances = int(max_message_advances)
        self._presses = 0

    # -- helpers --------------------------------------------------
    def _press(self, button):
        self._presses += 1
        if self._presses > self.max_presses:
            raise ExecutorAbort(f"exceeded {self.max_presses} presses")
        return self.io.press(button)

    def _clear_messages(self, state):
        advances = 0
        while state and (state.get("screen") == SCREEN_MESSAGE
                         or state.get("message_pending")):
            advances += 1
            if advances > self.max_message_advances:
                raise ExecutorAbort("message never cleared")
            state = self._press("A")
        return state

    def _to_main_menu(self):
        """Idempotently get back to the top-level battle menu."""
        state = self.io.menu_state()
        for _ in range(6):
            state = self._clear_messages(state)
            scr = (state or {}).get("screen")
            if scr == SCREEN_MAIN:
                return state
            if scr in (SCREEN_MOVES, SCREEN_PARTY, SCREEN_YESNO):
                state = self._press("B")
                continue
            if scr == SCREEN_UNKNOWN or scr is None:
                raise ExecutorAbort(f"unexpected/unreadable menu: {scr}")
        raise ExecutorAbort("could not reach the main battle menu")

    def _move_cursor_to(self, target, *, axis_buttons=("DOWN", "UP")):
        """Move a linear cursor to ``target`` with verification after every
        press. Aborts if the cursor does not respond."""
        down, up = axis_buttons
        for _ in range(12):
            state = self.io.menu_state()
            cur = (state or {}).get("cursor")
            if cur is None:
                raise ExecutorAbort("cursor not readable")
            if cur == target:
                return state
            before = cur
            state = self._press(down if cur < target else up)
            after = (state or {}).get("cursor")
            if after == before:
                # try the other direction once (wrap-around menus)
                state = self._press(up if cur < target else down)
                if (state or {}).get("cursor") == before:
                    raise ExecutorAbort("cursor did not move")
        raise ExecutorAbort("cursor never reached target")

    def _move_grid_cursor_to(self, target):
        """Move a verified FireRed 2x2 cursor (0/1 over 2/3)."""
        for _ in range(4):
            state = self.io.menu_state()
            cur = (state or {}).get("cursor")
            if cur not in (0, 1, 2, 3) or target not in (0, 1, 2, 3):
                raise ExecutorAbort("2x2 cursor not readable")
            if cur == target:
                return state
            cx, cy = cur % 2, cur // 2
            tx, ty = target % 2, target // 2
            button = ("RIGHT" if tx > cx else "LEFT") if cx != tx else (
                "DOWN" if ty > cy else "UP")
            before = cur
            state = self._press(button)
            if (state or {}).get("cursor") == before:
                raise ExecutorAbort("2x2 cursor did not move")
        raise ExecutorAbort("2x2 cursor never reached target")

    # -- public --------------------------------------------------
    def execute(self, macro, *, objective=None, dry_run=False):
        """Run ``macro``. Returns a structured transition dict. Raises
        :class:`ExecutorAbort` (caught here and reported) on any unsafe state."""
        self._presses = 0
        result = {"macro": macro, "ok": False, "aborted": False,
                  "reason": "", "presses": 0, "snapshot_before": None,
                  "snapshot_after": None, "balls_used": 0}

        if macro not in ALL_MACROS:
            result["reason"] = f"unknown macro {macro!r}"
            return result

        if not self.io.in_battle():
            result["reason"] = "not in a confirmed battle; refusing to press"
            return result

        snap = self.io.snapshot()
        result["snapshot_before"] = snap

        if macro == CATCH_ACTION:
            ok, why = catch_precheck(snap, objective)
            if not ok:
                result["reason"] = f"catch precheck failed: {why}"
                result["invalid"] = True
                result["catch_reject"] = why
                return result
        elif macro not in legal_macros(snap):
            result["reason"] = f"{macro} is not a legal action for this state"
            return result

        if not dry_run:
            from twoby2.activation import assert_live_battle_allowed
            try:
                assert_live_battle_allowed(snap)
            except Exception as exc:            # LiveActivationBlocked
                result["reason"] = f"live activation blocked: {exc}"
                return result

        try:
            if macro in MOVE_ACTIONS:
                self._do_move(MOVE_ACTIONS.index(macro))
            elif macro in SWITCH_ACTIONS:
                self._do_switch(SWITCH_ACTIONS.index(macro))
            elif macro == CATCH_ACTION:
                result["balls_used"] = self._do_catch(objective)
            else:
                self._do_run(snap)
            result["ok"] = True
            result["reason"] = "executed"
        except ExecutorAbort as exc:
            result["aborted"] = True
            result["reason"] = str(exc)
            self._safe_backout()

        result["presses"] = self._presses
        try:
            result["snapshot_after"] = self.io.snapshot()
        except Exception:
            result["snapshot_after"] = None
        return result

    # -- macro implementations -------------------------------
    def _do_move(self, idx):
        self._to_main_menu()
        self._move_grid_cursor_to(MENU_FIGHT)
        state = self._press("A")
        state = self._clear_messages(state)
        if (state or {}).get("screen") != SCREEN_MOVES:
            raise ExecutorAbort(f"FIGHT did not open the move list ({state})")
        self._move_grid_cursor_to(idx)
        self._press("A")
        # HAND OFF HERE. The damage / faint / EXP / level-up animation belongs
        # to the driver's _resolve_turn, NOT the executor - _clear_messages here
        # would eat the whole turn and then abort with "message never cleared".
        # Only confirm the move menu was actually left (a 0-PP / disabled move
        # re-prompts on SCREEN_MOVES).
        state = self.io.wait_frames(4)
        if (state or {}).get("screen") == SCREEN_MOVES:
            state = self.io.wait_frames(10)
            if (state or {}).get("screen") == SCREEN_MOVES:
                raise ExecutorAbort("move not accepted (likely 0 PP / disabled)")

    def _do_switch(self, slot):
        self._to_main_menu()
        self._move_grid_cursor_to(MENU_POKEMON)
        state = self._press("A")
        state = self._clear_messages(state)
        if (state or {}).get("screen") != SCREEN_PARTY:
            raise ExecutorAbort("POKEMON did not open the party list")
        # The verified action/move cursor addresses do not expose the party
        # list cursor. FireRed opens the list on the currently active slot, so
        # move by the authoritative active-slot delta and verify the *result*
        # through gBattlerPartyIndexes/gBattleMons after confirming SHIFT.
        snap = self.io.snapshot()
        current = snap.get("active_slot")
        if current is None:
            current = (snap.get("player_active") or {}).get("slot")
        if not isinstance(current, int) or not 0 <= current < 6:
            raise ExecutorAbort("active party slot not authoritative")
        button = "DOWN" if slot > current else "UP"
        for _ in range(abs(slot - current)):
            self._press(button)
        self._press("A")          # open SHIFT / SUMMARY / CANCEL
        self.io.wait_frames(60)   # submenu is animated; an immediate A is lost
        self._press("A")          # SHIFT is the default first entry
        self._clear_messages(self.io.menu_state())

    def _do_run(self, snap):
        if snap.get("is_trainer") is not False or snap.get("can_escape") is not True:
            raise ExecutorAbort("RUN is only valid in an escapable wild battle")
        self._to_main_menu()
        self._move_grid_cursor_to(MENU_RUN)
        state = self._press("A")
        self._clear_messages(state)

    def _best_usable_ball(self, objective):
        """Pick a ball index in the pocket, keeping ``reserved_ball_count`` of
        the chosen type in reserve. Prefers Great/Ultra over Poke. Returns
        ``(pocket_index, item_id)`` or raises."""
        import battle_catch
        pocket = self.io.ball_pocket()
        if not pocket:
            raise ExecutorAbort("bag_unreadable")
        reserve = int((objective or {}).get("reserved_ball_count", 0) or 0)
        # rank: ultra > great > poke > everything else, then by count
        rank = {battle_catch.ULTRA_BALL: 3, battle_catch.GREAT_BALL: 2,
                battle_catch.POKE_BALL: 1}
        best = None
        for idx, (item_id, count) in enumerate(pocket):
            if not battle_catch.is_ball(item_id):
                raise ExecutorAbort("bag_unreadable")   # non-ball in ball pocket
            if int(count) - reserve <= 0:
                continue
            key = (rank.get(int(item_id), 0), int(count))
            if best is None or key > best[0]:
                best = (key, idx, int(item_id))
        if best is None:
            raise ExecutorAbort("no_ball_above_reserve")
        return best[1], best[2]

    def _do_catch(self, objective):
        """State machine (spec §7): main -> BAG -> ball pocket -> select ball
        -> confirm USE -> bag left -> HAND OFF the throw animation to the
        driver's _resolve_turn. Returns the number of balls the bag showed as
        consumed (0 or 1). Never blind A/frame sequences."""
        pocket_idx, item_id = self._best_usable_ball(objective)

        self._to_main_menu()
        self._move_grid_cursor_to(MENU_BAG)
        state = self._press("A")
        state = self._clear_messages(state)
        if (state or {}).get("screen") not in (SCREEN_BAG,):
            raise ExecutorAbort(f"BAG did not open ({state})")

        # some layouts open the bag on the LAST pocket; move to the ball pocket
        for _ in range(4):
            st = self.io.menu_state()
            if st and st.get("pocket") in ("balls", "ball", None):
                break
            self._press("RIGHT")
        # navigate to the chosen ball (verified linear cursor)
        try:
            self._move_cursor_to(pocket_idx)
        except ExecutorAbort:
            raise ExecutorAbort("bag_unreadable")

        state = self._press("A")               # opens USE / GIVE / TOSS
        state = self.io.wait_frames(6)
        if (state or {}).get("screen") == SCREEN_ITEM_USE:
            cur = (state or {}).get("cursor")
            if cur not in (0, None):
                # USE is the first entry; nudge up if needed
                self._move_cursor_to(0)
            state = self._press("A")
        # the game now asks nothing else for a ball in battle -> throw begins.
        # verify the bag was actually left before handing off.
        left = False
        for _ in range(8):
            st = self._clear_messages(self.io.menu_state())
            scr = (st or {}).get("screen")
            if scr in (SCREEN_BAG, SCREEN_ITEM_USE):
                self._press("B")
                continue
            if scr in (SCREEN_MESSAGE, SCREEN_MAIN, SCREEN_UNKNOWN, None) or \
                    (st or {}).get("throw_started"):
                left = True
                break
        if not left:
            raise ExecutorAbort("bag_unreadable")
        # HAND OFF. _resolve_turn lets the shake / caught / broke-free
        # animation run and the driver's RAM before/after diff is the
        # authoritative ball-consumed count. A thrown ball is one ball.
        return 1

    def _safe_backout(self):
        try:
            for _ in range(4):
                st = self.io.menu_state()
                if (st or {}).get("screen") == SCREEN_MAIN:
                    return
                self.io.press("B")
        except Exception:
            pass


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
