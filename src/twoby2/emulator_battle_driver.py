"""Real emulator battle driver for the BPRD ROM (Phase 2).

Replaces :class:`battle_env.SimulatedBattleDriver` for LIVE battle training /
FULL-worker in-battle control. Uses:

  * :class:`battle_state.MainBattleReader` for ``gMain.inBattle`` (the ONLY
    trusted battle-active latch; the pulsing 0x23BC8 integration byte is never
    used as the persistent state);
  * :mod:`twoby2.battle_ram_live` for the VERIFIED in-battle addresses
    (``gBattleMons`` full struct, ``gBattlerPartyIndexes``, action/move cursor,
    ``battle_menu_state``, weather);
  * :mod:`firered_ram` for the verified party read (cross-check + switch targets);
  * :class:`battle_executor.MacroExecutor` for the actual, cursor-verified
    button sequences.

Fail-closed:
  * unknown ``battle_menu_state`` -> no button is sent;
  * invalid species / HP / move id in ``gBattleMons`` -> no action;
  * a trainer battle is never handled with RUN;
  * a double battle (``gBattleTypeFlags`` DOUBLE, or >1 live own battler) is
    NOT driven live — the action routing for two active battlers is not
    separately verified; ``make_env`` / the router keep it blocked;
  * bounded presses / message-advances / per-macro timeout, with a clear
    diagnostic reason on abort;
  * the acting policy is pinned for a whole battle (no mid-battle model swap);
    the next Battle-Champion version is loaded only at the next battle.

Live use is still gated: :func:`twoby2.activation.assert_live_battle_allowed`
must pass, which requires ``FEATURES['battle_executor_live']`` on AND an
execution-ready snapshot.
"""
from __future__ import annotations

import time

import battle_types as bt
import battle_catch
from battle_state import MainBattleReader
from battle_executor import (
    MacroExecutor, BattleIO, ALL_MACROS, ALL_MACROS_V1, MOVE_ACTIONS,
    SWITCH_ACTIONS, RUN_ACTION, CATCH_ACTION, SCREEN_MAIN, SCREEN_MOVES,
    SCREEN_PARTY, SCREEN_MESSAGE, SCREEN_UNKNOWN, MENU_FIGHT, MENU_POKEMON,
    MENU_RUN, catch_precheck,
)
from twoby2 import battle_ram_live as L
from twoby2 import feature_enabled

BATTLE_TYPE_DOUBLE = 0x0001
BATTLE_TYPE_TRAINER = 0x0008

# The RAM probes verified the battle/action/move cursors, but not the party-list
# cursor. Live switching therefore remains masked until that cursor is verified.
LIVE_SWITCH_CURSOR_VERIFIED = False

_PRESS_FRAMES = 8          # frames advanced per button press
_SETTLE_FRAMES = 4        # neutral frames after a press
# Wall-clock cap per macro - EMERGENCY BRAKE ONLY. A turn normally exits on
# `reached_main_menu` / `left_battle` long before this; it exists so a single
# wedged worker cannot hang the SubprocVecEnv fleet forever.
_MAX_TURN_SECONDS = 8.0

# Turn-resolution window (the whole damage / faint / EXP / level-up text that
# plays out AFTER our move-select A, before the next main menu / OUT_OF_BATTLE).
# The executor hands off IMMEDIATELY after the move-select A now, so this owns
# the entire animation. Real trace: main menu can reappear as late as frame
# ~710, then needs stable samples.
_RESOLVE_MAX_FRAMES = 1200
_STEP_BLOCK = 6              # frames stepped between RAM reads (get_ram() rebuilds
                             # the whole GBA memory view - do it ~6x less often)
_A_PRESS_EVERY_BLOCKS = 4    # A on a message screen every 4th block (~24 frames)
_LEFT_BATTLE_STABLE = 3      # consecutive BLOCKS reading "not in battle" -> ended
_MAIN_MENU_STABLE = 3        # consecutive BLOCKS at a *readable* CHOOSEACTION menu
_MAX_RESOLVE_A_PRESSES = 40  # cap on A presses; past it, keep stepping NEUTRAL
_PARTY_POLL_EVERY = 5        # party-wipe check every Nth block (box decrypt is dear)


class DoubleBattleBlocked(RuntimeError):
    pass


class EmulatorBattleIO(BattleIO):
    """Adapts the retro env to what :class:`MacroExecutor` needs, using the
    verified live readers."""

    def __init__(self, env, *, buttons=None, main_battle_reader=None):
        self.env = env
        self.buttons = list(buttons if buttons is not None else env.buttons)
        self._mbr = main_battle_reader or MainBattleReader()
        self._n = len(self.buttons)

    # -- low level ---------------------------------------------------
    def _ram(self):
        return self.env.get_ram()

    def _mask(self, *names):
        m = [0] * self._n
        for name in names:
            if name in self.buttons:
                m[self.buttons.index(name)] = 1
        return m

    def _step(self, mask, frames):
        import numpy as np
        arr = np.asarray(mask, dtype=np.uint8)
        neutral = np.zeros(self._n, dtype=np.uint8)
        for i in range(frames):
            self.env.em.set_button_mask(arr if i == 0 else neutral)
            self.env.em.step()
            callback = getattr(self.env, "_pkmai_frame_callback", None)
            if callback is not None:
                callback()

    # -- BattleIO API ---------------------------------------------
    def in_battle(self):
        val = self._mbr.read(self._ram())
        return bool(val) if val is not None else False

    def menu_state(self):
        ram = self._ram()
        raw = L.menu_state_raw(ram)
        screen = {L.MENU_CHOOSEACTION: SCREEN_MAIN,
                  L.MENU_CHOOSEMOVE: SCREEN_MOVES,
                  L.MENU_CHOOSEPOKEMON: SCREEN_PARTY}.get(raw)
        if screen == SCREEN_MAIN:
            cursor = L.action_cursor(ram)
        elif screen == SCREEN_MOVES:
            cursor = L.move_cursor(ram)
        elif screen == SCREEN_PARTY:
            cursor = L.action_cursor(ram)   # party cursor tracked elsewhere; see note
        else:
            screen = SCREEN_MESSAGE if self.in_battle() else SCREEN_UNKNOWN
            cursor = None
        return {"screen": screen, "cursor": cursor,
                "message_pending": screen is None,
                "battler_ready": self.in_battle()}

    def snapshot(self):
        return build_live_snapshot(self.env, main_battle_reader=self._mbr)

    def press(self, button):
        self._step(self._mask(button), _PRESS_FRAMES)
        self._step(self._mask(), _SETTLE_FRAMES)
        return self.menu_state()

    def wait_frames(self, frames):
        self._step(self._mask(), max(0, int(frames)))
        return self.menu_state()


# --------------------------------------------------------------------------
# live snapshot
# --------------------------------------------------------------------------
def build_live_snapshot(env, *, main_battle_reader=None):
    """A ``battle_ram.battle_snapshot``-shaped dict built from the VERIFIED
    addresses. ``None`` for anything not authoritatively readable."""
    import firered_ram as fr
    mbr = main_battle_reader or MainBattleReader()
    ram = env.get_ram()
    in_batt = mbr.read(ram)

    class _E:
        def get_ram(self_inner):
            return ram
    e = _E()
    try:
        flags = fr.read_battle_type_flags(e)
    except Exception:
        flags = 0
    try:
        tid, outcome = fr.read_trainer_battle(e)
    except Exception:
        tid, outcome = 0, None
    # gTrainerBattleOpponent_A is not cleared after every trainer encounter and
    # therefore may contain a plausible stale id during a later wild battle.
    # The live battle-kind authority is gBattleTypeFlags.
    is_trainer = bool(int(flags) & BATTLE_TYPE_TRAINER)
    is_double = bool(int(flags) & BATTLE_TYPE_DOUBLE)

    live = L.read_all(ram)
    player_party = fr.read_player_party(e)
    slot = live["party_index_player"]
    pmon = live["battle_mon_player"]
    emon = live["battle_mon_enemy"]

    xc_ok, xc = L.crosscheck_against_party(live, player_party)

    def enrich(mon):
        if mon is None:
            return None
        import pokedb
        m = dict(mon)
        for mv in m.get("moves", []):
            mech = pokedb.move_mechanics(mv["id"]) if pokedb.is_available() else None
            if mech:
                mv.update(type=mech["type"], power=mech["power"],
                          accuracy=mech["accuracy"], priority=mech["priority"],
                          is_status=mech["is_status"], max_pp=mech["pp"],
                          mechanics_known=True)
            else:
                mv["mechanics_known"] = False
        m["checksum_ok"] = True
        m["types_known"] = bool(m.get("types"))
        m["stat_stages_known"] = True
        return m

    # -- Catch-v2 (spec §6/§8): everything the catch precheck reads. The bag /
    # ball-pocket / dex RAM is UNVERIFIED for BPRD, so usable_ball_count and
    # pc_capture_supported are fail-closed and catch_ram_ready() is False —
    # the live CATCH macro stays masked until tools/catch_ram_probe.py passes.
    _pp = [m for m in (player_party or []) if m and m.get("checksum_ok", True)]
    _ball_pocket = L.ball_pocket(ram)
    _catch_ready, _catch_missing = L.catch_ram_ready(ram)

    menu = live["menu_state"]
    snap = {
        "schema": "battle_snapshot_v2_live",
        "ram_ok": True,
        "in_battle": (bool(in_batt) if in_batt is not None else None),
        "in_battle_source": "gMain.inBattle",
        "battle_type_flags": int(flags),
        "battle_type_flags_known": True,
        "is_trainer": is_trainer,
        "is_double": is_double,
        "is_wild_flagged": bool(int(flags) & 0x0004),
        "can_escape": (in_batt is True and not is_trainer and not is_double),
        "trainer_id": int(tid),
        "battle_outcome": outcome,
        "player_active": enrich(pmon),
        "enemy_active": enrich(emon),
        "player_active_source": "gBattlerPartyIndexes",
        "player_party": player_party,
        "enemy_party": [],
        "active_slot": slot,
        "active_slot_enemy": live["party_index_enemy"],
        "active_slot_authoritative": (slot is not None and xc_ok),
        "active_slot_crosscheck": xc,
        "stat_stages_known": pmon is not None,
        "menu_cursor_known": menu in ("main", "move", "party"),
        "menu_state": menu,
        "action_cursor": live["action_cursor"],
        "move_cursor": live["move_cursor"],
        "weather": live["weather"],
        "weather_known": live["weather"] is not None,
        "db_available": _pokedb_ok(),
        # --- catch precheck inputs (fail-closed) -----------------------
        "party_slot_count": len(_pp),
        "party_has_space": len(_pp) < 6,
        "ball_pocket": _ball_pocket,
        "usable_ball_count": (None if _ball_pocket is None
                              else sum(int(c) for _, c in _ball_pocket)),
        "pc_capture_supported": False,       # unverified for BPRD
        "catch_ram_ready": bool(_catch_ready),
        "catch_ram_missing": list(_catch_missing),
        "message_pending": menu is None,
        "animation_active": menu is None,
    }
    return snap


def _pokedb_ok():
    try:
        import pokedb
        return pokedb.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------
class EmulatorBattleDriver:
    """Drives ONE live battle to the end for the battle env / FULL wrapper.

    ``policy_fn(obs) -> macro_index`` is pinned for the whole battle by the
    caller (battle env / wrapper); this driver never swaps it.
    """

    def __init__(self, env, *, main_battle_reader=None, max_turns=80, schema="v2"):
        self.env = env
        self._mbr = main_battle_reader or MainBattleReader()
        self.io = EmulatorBattleIO(env, main_battle_reader=self._mbr)
        self.executor = MacroExecutor(self.io)
        self.max_turns = int(max_turns)
        self._turn = 0
        self._recent_switches = []
        self.diagnostics = []
        self._objective = None          # immutable BattleObjective for this battle
        # "v1" = combat only (11 macros, no CATCH); "v2" = adds CATCH (12).
        self.schema = schema
        self._macros = ALL_MACROS_V1 if schema == "v1" else ALL_MACROS

    def set_objective(self, objective):
        """The immutable battle order (spec §2). Set once at battle start by the
        env / FULL wrapper; the driver only reads it. A live CATCH is still
        gated on ``battle_ram_live.catch_ram_ready`` regardless of the order."""
        self._objective = dict(objective or {}) or None

    # -- state ------------------------------------------------------
    def reset_battle_reader(self):
        """Fresh gMain discovery. MUST be called after em.set_state(): the
        located offset + last inBattle result would otherwise carry over from
        the previous battle and report a stale 'out of battle' for the first
        samples of the new one."""
        self._mbr = MainBattleReader()
        self.io._mbr = self._mbr

    def in_battle(self):
        return self.io.in_battle()

    def snapshot(self):
        return self.io.snapshot()

    def _catch_ram_snapshot(self, snap):
        """The before/after dict :func:`battle_catch.classify_catch_outcome`
        needs. Built from VERIFIED reads (battle-active latch, gBattleMons enemy
        HP, verified party) plus FAIL-CLOSED bag / dex fields. While
        ``catch_ram_ready`` is False the ball-pocket / dex entries are ``None``
        and the classifier correctly returns ``unreadable`` — nothing is
        guessed."""
        snap = snap or {}
        enemy = snap.get("enemy_active") or {}
        party = snap.get("player_party") or []
        pocket = snap.get("ball_pocket")
        return {
            "battle_active": snap.get("in_battle"),
            "enemy_hp": int(enemy.get("cur_hp", 0) or 0),
            "ball_counts": ({"total": sum(int(c) for _, c in pocket)}
                            if pocket is not None else None),
            "usable_balls": snap.get("usable_ball_count"),
            "party_size": len([m for m in party
                               if m and m.get("checksum_ok", True)]),
            "target_identity": (int(enemy.get("species_id")) if enemy.get("species_id")
                                else None),
            "new_party_member": None,          # dex/party identity read unverified
            "target_dex_owned_before": None,
            "target_dex_owned_now": None,
            "pc_transfer_confirmed": None,
        }

    def _assert_singles(self, snap):
        if snap.get("is_double"):
            raise DoubleBattleBlocked(
                "double battle: live action routing for two battlers is not "
                "verified — driver stays fail-closed")

    # -- action masking -----------------------------------------
    def legal_macros(self, snap):
        snap = snap or {}
        legal = set()
        active = snap.get("player_active") or {}
        for i, mv in enumerate((active.get("moves") or [])[:4]):
            if mv.get("mechanics_known") and int(mv.get("pp", 0) or 0) > 0:
                legal.add(MOVE_ACTIONS[i])
        if LIVE_SWITCH_CURSOR_VERIFIED:
            party = snap.get("player_party") or []
            cur = snap.get("active_slot")
            for i, mon in enumerate(party[:6]):
                if i == cur:
                    continue
                if mon.get("checksum_ok", True) and int(mon.get("cur_hp", 0) or 0) > 0:
                    legal.add(SWITCH_ACTIONS[i])
        if snap.get("can_escape") is True and snap.get("is_trainer") is False:
            legal.add(RUN_ACTION)
        # CATCH (spec §6): v2 driver only, and only when the objective asked for
        # it, the precheck passes AND the catch RAM is verified for this ROM.
        # The RAM gate is currently always closed
        # (battle_ram_live.catch_ram_ready -> False), so a live CATCH stays
        # masked. A v1 driver has no CATCH slot at all.
        if (self.schema == "v2" and CATCH_ACTION in self._macros
                and self._objective and snap.get("catch_ram_ready") is True
                and catch_precheck(snap, self._objective)[0]):
            legal.add(CATCH_ACTION)
        return legal

    def action_mask(self, snap):
        legal = self.legal_macros(snap)
        return [1 if m in legal else 0 for m in self._macros]

    # -- turn -----------------------------------------------------
    def apply_macro(self, macro, *, dry_run=False):
        """Execute one macro (our action + resolve the whole turn / result
        animation). Returns ``(new_snapshot, event)``. ``event`` carries the
        anti-farm + diagnostic facts the battle reward and counters need."""
        self._turn += 1
        ev = {"our_damage_dealt": 0, "enemy_hp_before": None, "enemy_ko": False,
              "battle_won": False, "own_faint": False, "wipe": False,
              "invalid": False, "wasted": False, "fled": False,
              "illegal_flee": False, "switch_loop": False, "aborted": False,
              "menu_stall": False, "terminal_unknown": False, "no_damage": False,
              "unreadable": False, "reason": ""}

        snap0 = self.snapshot()
        if snap0.get("in_battle") is not True:
            ev.update(invalid=True, reason="not in a confirmed battle")
            return snap0, ev
        try:
            self._assert_singles(snap0)
        except DoubleBattleBlocked as exc:
            ev.update(invalid=True, aborted=True, reason=str(exc))
            return snap0, ev

        # A party list up when we didn't ask to switch = a FORCED switch after
        # our active fainted. Live switching is masked (party cursor unverified),
        # so we cannot pick a replacement safely - end the episode as a
        # diagnosed, bounded failure instead of looping aborts to max_turns.
        if (snap0.get("menu_state") == "party" and macro not in SWITCH_ACTIONS):
            ev.update(aborted=True, own_faint=True, unreadable=True,
                      reason="forced switch after a faint (live SWITCH masked)")
            self.diagnostics.append({"forced_switch_unhandled": ev["reason"]})
            return snap0, ev

        # If nothing is legal, the battle menu is almost always just not
        # rendered yet (mid-message / mid-animation). Settle to a readable main
        # menu FIRST - it is a driver-side read problem, not a bad policy pick.
        if not self.legal_macros(snap0):
            snap0 = self._settle_to_readable_menu()
        legal_now = self.legal_macros(snap0)
        if not legal_now:
            ev.update(aborted=True, unreadable=True,
                      reason="battle state not readable at action time "
                             f"(menu_state={snap0.get('menu_state')}, "
                             f"player_active={'yes' if snap0.get('player_active') else 'no'})")
            self.diagnostics.append({"unreadable_action_state": ev["reason"]})
            return snap0, ev

        if macro not in self._macros or macro not in legal_now:
            # MaskablePPO cannot reach this; a non-masked caller gets a clean
            # invalid with no turn advance (no contradictory "penalise but run
            # MOVE_1" data).
            ev.update(invalid=True, reason=f"{macro} not legal for this state")
            return snap0, ev
        if macro == RUN_ACTION and snap0.get("is_trainer"):
            ev.update(illegal_flee=True, invalid=True,
                      reason="RUN in a trainer battle")
            return snap0, ev
        if macro == CATCH_ACTION:
            # spec §6/§7: a trainer CATCH never presses a button.
            if snap0.get("is_trainer") is not False:
                ev.update(invalid=True, illegal_trainer_catch=True,
                          catch_reject="trainer_catch_blocked",
                          reason="CATCH in a trainer battle")
                return snap0, ev
            # spec §8: no live catch until the bag / ball-pocket / dex / result
            # RAM is verified for this ROM. No button is pressed.
            if snap0.get("catch_ram_ready") is not True:
                ev.update(invalid=True, catch_reject="catch_ram_not_ready",
                          reason="catch RAM unverified for BPRD: "
                                 + "; ".join(snap0.get("catch_ram_missing") or []))
                self.diagnostics.append({"catch_ram_not_ready": ev["reason"]})
                return snap0, ev
            ok, why = catch_precheck(snap0, self._objective)
            if not ok:
                ev.update(invalid=True, catch_reject=why,
                          reason=f"catch precheck failed: {why}")
                return snap0, ev

        enemy_hp0 = (snap0.get("enemy_active") or {}).get("cur_hp")
        enemy_idx0 = snap0.get("active_slot_enemy")
        ev["enemy_hp_before"] = enemy_hp0
        _catch_before = self._catch_ram_snapshot(snap0) if macro == CATCH_ACTION else None

        result = self.executor.execute(macro, objective=self._objective,
                                       dry_run=dry_run)
        if not result["ok"]:
            self.diagnostics.append(result)
            if result["aborted"] and int(result.get("presses", 0)) > 0:
                # buttons were sent, then the executor bailed mid-macro - the
                # battle state is now dirty. Let the animation settle, then end
                # the episode as a diagnosed fault (NOT a silent invalid loop).
                self._resolve_turn(enemy_hp0=enemy_hp0, enemy_idx0=enemy_idx0)
                snap = self._settle_to_readable_menu()
                ev.update(aborted=True, unreadable=True,
                          reason=f"executor aborted after {result['presses']} "
                                 f"presses: {result['reason']}")
                return snap, ev
            # no buttons sent -> a clean pre-check failure (bad policy pick /
            # not readable). Not an episode-ender.
            ev.update(invalid=not result["aborted"], aborted=result["aborted"],
                      reason=result["reason"])
            return self.snapshot(), ev

        if macro in SWITCH_ACTIONS:
            self._recent_switches.append(macro)
            self._recent_switches = self._recent_switches[-4:]
            if _switch_loop(self._recent_switches):
                ev["switch_loop"] = True

        # Watch the WHOLE result animation, not just the final snapshot: an
        # enemy that dropped to 0 HP then had its struct cleared as the battle
        # ended is still a win.
        res = self._resolve_turn(enemy_hp0=enemy_hp0, enemy_idx0=enemy_idx0)
        # gMain changes callbacks as a battle closes.  The previously located
        # struct can remain superficially valid with a stale inBattle=True bit,
        # so a capped resolver must re-discover gMain instead of merely adding
        # more neutral frames against the stale offset.
        if res["menu_stall"]:
            self.reset_battle_reader()
            snap1 = self._settle_to_readable_menu(max_blocks=120)
            recovered = (snap1.get("in_battle") is False or (
                snap1.get("in_battle") is True
                and snap1.get("menu_state") == "main"
                and bool(self.legal_macros(snap1))))
            if recovered:
                res["menu_stall"] = False
                res["reached_main_menu"] = snap1.get("menu_state") == "main"
            else:
                self.diagnostics.append({
                    "resolve_turn": "menu_stall",
                    "menu_trace": list(res.get("menu_trace") or []),
                    "a_presses": res.get("a_presses", 0),
                    "frames": res.get("frames", 0)})
        else:
            snap1 = self.snapshot()
        # The NEXT turn needs a real, cursor-known main menu (the executor's
        # activation gate rejects any snapshot whose battle_menu_state is not
        # main/move/party). If the battle continues but we did not land cleanly
        # on the main menu, settle there now.
        if (snap1.get("in_battle") is True and not res["left_battle"]
                and snap1.get("menu_state") != "main"):
            snap1 = self._settle_to_readable_menu()
            if snap1.get("in_battle") is False:
                res["left_battle"] = True
                res["in_battle"] = False
            elif snap1.get("in_battle") is not True:
                ev.update(aborted=True, unreadable=True,
                          reason="battle-active latch unreadable after turn settle")
                self.diagnostics.append({"turn_no_settle": ev["reason"]})
                return snap1, ev
            elif snap1.get("menu_state") != "main" or not self.legal_macros(snap1):
                # could not get back to an actionable menu - end the episode as
                # a diagnosed fault rather than feeding the next turn a state
                # the executor's activation gate will reject forever.
                ev.update(aborted=True, unreadable=True,
                          reason=f"turn did not settle to the main menu "
                                 f"(menu_state={snap1.get('menu_state')}, "
                                 f"trace={res.get('menu_trace')})")
                self.diagnostics.append({"turn_no_settle": ev["reason"]})
                return snap1, ev
        # The post-resolution MainBattleReader read is authoritative: if the
        # battle is over now, `left_battle` is true regardless of whether the
        # frame-by-frame accumulator caught 24 stable off-battle frames.
        if snap1.get("in_battle") is False:
            res["left_battle"] = True
            res["in_battle"] = False

        if macro in SWITCH_ACTIONS:
            expected_slot = SWITCH_ACTIONS.index(macro)
            if (res["in_battle"] and not res["left_battle"]
                    and snap1.get("active_slot") != expected_slot):
                ev.update(aborted=True,
                          reason=(f"switch not confirmed: expected slot "
                                  f"{expected_slot}, got {snap1.get('active_slot')}"))

        e1 = (snap1.get("enemy_active") or {})
        enemy_hp1 = e1.get("cur_hp")
        if res["enemy_ko_seen"]:
            ev["our_damage_dealt"] = int(enemy_hp0 or 0)
        elif enemy_hp0 is not None and enemy_hp1 is not None and not res["enemy_swapped"]:
            ev["our_damage_dealt"] = max(0, int(enemy_hp0) - int(enemy_hp1))

        p1 = (snap1.get("player_active") or {})
        if int(p1.get("cur_hp", 1) or 0) <= 0:
            ev["own_faint"] = True

        # -- classify the turn from the observed animation ---------------
        verdict = classify_turn_outcome(macro, res)
        ev.update({k: v for k, v in verdict.items() if k != "reason"})
        if verdict.get("reason"):
            ev["reason"] = verdict["reason"]
        if ev["terminal_unknown"]:
            self.diagnostics.append({"terminal_unknown": ev["reason"]})
        if ev["menu_stall"]:
            self.diagnostics.append({"menu_stall": ev["reason"]})

        # -- CATCH outcome from a VERIFIED RAM before/after diff (spec §8) --
        # A catch is ``caught`` ONLY with stable RAM evidence — never from
        # "battle ended" / "party grew" / a log line. This path is only
        # reachable once catch_ram_ready() is True (checked above).
        if macro == CATCH_ACTION:
            ev["catch_attempted"] = True
            ev["balls_used"] = int(result.get("balls_used", 0) or 0)
            after = self._catch_ram_snapshot(snap1)
            outcome = battle_catch.classify_catch_outcome(
                _catch_before, after,
                is_trainer=bool(snap0.get("is_trainer")),
                ball_thrown=ev["balls_used"] > 0,
                pc_capture_supported=bool(snap0.get("pc_capture_supported")))
            ev["catch_outcome"] = outcome
            if outcome == "caught":
                ev.update(catch_success=True, battle_won=False, enemy_ko=False,
                          caught_species_id=int((snap0.get("enemy_active") or {})
                                                .get("species_id", 0) or 0))
            elif outcome == "sent_to_pc":
                ev.update(catch_success=True, battle_won=False, enemy_ko=False,
                          sent_to_pc=True,
                          caught_species_id=int((snap0.get("enemy_active") or {})
                                                .get("species_id", 0) or 0))
            elif outcome == "broke_free":
                ev["failed_catch"] = True
            elif outcome in ("bag_unreadable", "menu_stall", "executor_abort",
                             "unreadable"):
                ev.update(aborted=True, unreadable=True,
                          reason=f"catch outcome unreadable: {outcome}")
                self.diagnostics.append({"catch_unreadable": outcome})

        if (macro in MOVE_ACTIONS and not ev["enemy_ko"] and not ev["battle_won"]
                and int(ev["our_damage_dealt"]) == 0
                and not ev["aborted"] and not ev["terminal_unknown"]):
            ev["no_damage"] = True
        return snap1, ev

    def _step_block(self, *, press_first=None):
        """Step _STEP_BLOCK emulator frames (optionally a button on the first
        frame), feeding the frame mirror, then return a single fresh RAM view."""
        import numpy as np
        neutral = np.zeros(len(self.io.buttons), dtype=np.uint8)
        first = self.io._mask(press_first) if press_first else neutral
        cb = getattr(self.env, "_pkmai_frame_callback", None)
        for i in range(_STEP_BLOCK):
            self.env.em.set_button_mask(first if i == 0 else neutral)
            self.env.em.step()
            if cb is not None:
                cb()
        return self.env.get_ram()

    def _settle_to_readable_menu(self, *, max_blocks=48):
        """Advance until the battle main menu is up AND gBattleMons is readable
        (2 stable samples), or give up. Returns the snapshot at that point."""
        deadline = time.time() + _MAX_TURN_SECONDS
        # step one block first so the VBlank delta seen by _mbr.read is always
        # exactly _STEP_BLOCK - a wrong `frames` breaks gMain re-discovery at
        # the battle-end callback swap and leaves the reader stale-True.
        ram = self._step_block()
        stable = 0
        for _ in range(max_blocks):
            if time.time() > deadline:
                break
            live = self._mbr.read(ram, frames=_STEP_BLOCK)
            if live is False:
                break
            if live is None:
                ram = self._step_block()
                continue
            menu_raw = L.menu_state_raw(ram)
            if menu_raw == L.MENU_CHOOSEACTION and L.read_battle_mon(ram, L.PLAYER_BATTLER):
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            is_message = menu_raw not in (L.MENU_CHOOSEACTION, L.MENU_CHOOSEMOVE,
                                          L.MENU_CHOOSEPOKEMON)
            ram = self._step_block(press_first="A" if is_message else None)
        return self.snapshot()

    def _resolve_turn(self, *, enemy_hp0, enemy_idx0):
        """Block-step through the post-press animation. Records what actually
        happened rather than trusting one stale end snapshot. RAM is read once
        per _STEP_BLOCK frames - get_ram() rebuilds the whole memory view and
        was the single biggest per-turn cost at 9 parallel emulators."""
        import firered_ram as fr

        class _E:
            def __init__(self, d):
                self._d = d

            def get_ram(self):
                return self._d

        deadline = time.time() + _MAX_TURN_SECONDS
        _KNOWN_MENUS = (L.MENU_CHOOSEACTION, L.MENU_CHOOSEMOVE, L.MENU_CHOOSEPOKEMON)
        max_blocks = _RESOLVE_MAX_FRAMES // _STEP_BLOCK

        res = {
            "in_battle": True, "left_battle": False,
            "enemy_ko_seen": False, "enemy_swapped": False,
            "own_party_wiped": False, "own_party_wiped_late": False,
            "menu_stall": False, "reached_main_menu": False,
            "unexpected_menu": False, "a_presses": 0,
            "frames": 0, "last_menu_raw": None, "menu_trace": [],
        }
        off_battle_run = 0
        main_menu_run = 0
        # step one block first so every _mbr.read below sees a VBlank delta of
        # exactly _STEP_BLOCK - the default (14) would fail gMain re-discovery
        # at the battle-end callback swap and hold the reader stale-True.
        ram = self._step_block()

        for block in range(max_blocks):
            res["frames"] = block * _STEP_BLOCK
            if time.time() > deadline:
                self.diagnostics.append({"warn": "turn wall-clock cap hit"})
                break

            menu_raw = L.menu_state_raw(ram)
            res["last_menu_raw"] = menu_raw
            if not res["menu_trace"] or res["menu_trace"][-1] != menu_raw:
                res["menu_trace"].append(menu_raw)
            in_batt = self._mbr.read(ram, frames=_STEP_BLOCK)

            # a CHOOSEMOVE / CHOOSEPOKEMON re-appearing here is NOT result text.
            if in_batt and menu_raw in (L.MENU_CHOOSEMOVE, L.MENU_CHOOSEPOKEMON):
                res["unexpected_menu"] = True

            emon = L.read_battle_mon(ram, L.ENEMY_BATTLER)
            if emon is not None and int(emon.get("cur_hp", 1) or 0) <= 0:
                res["enemy_ko_seen"] = True
            eidx = L.battler_party_index(ram, L.ENEMY_BATTLER)
            if eidx is not None and enemy_idx0 is not None and eidx != enemy_idx0:
                res["enemy_swapped"] = True
                res["enemy_ko_seen"] = True   # a swap only follows a KO here

            if block % _PARTY_POLL_EVERY == 0:
                try:
                    party = fr.read_player_party(_E(ram))
                    valid = [m for m in party if m and m.get("checksum_ok", True)]
                    if valid and all(int(m.get("cur_hp", 1) or 0) <= 0 for m in valid):
                        res["own_party_wiped" if in_batt else "own_party_wiped_late"] = True
                except Exception:
                    pass

            if in_batt is not True:
                off_battle_run += 1
                if off_battle_run >= _LEFT_BATTLE_STABLE:
                    res["in_battle"] = False
                    res["left_battle"] = True
                    break
            else:
                off_battle_run = 0

            # the main menu only counts once gBattleMons[player] is actually
            # readable - a menu that is drawn but whose battler struct has not
            # settled would store an empty action mask for the next turn.
            if (in_batt and menu_raw == L.MENU_CHOOSEACTION
                    and L.read_battle_mon(ram, L.PLAYER_BATTLER)):
                main_menu_run += 1
                if main_menu_run >= _MAIN_MENU_STABLE:
                    res["reached_main_menu"] = True
                    break
            else:
                main_menu_run = 0

            # advance a confirmed non-menu (message) screen with a sparse A
            # (~every 24 frames); never blind-press a real selection menu.
            # Past the A budget keep stepping NEUTRAL - the text usually still
            # auto-advances - rather than declaring a stall.
            press = None
            if (in_batt and menu_raw not in _KNOWN_MENUS
                    and block % _A_PRESS_EVERY_BLOCKS == 0
                    and res["a_presses"] < _MAX_RESOLVE_A_PRESSES):
                press = "A"
                res["a_presses"] += 1
            ram = self._step_block(press_first=press)
        else:
            if res["in_battle"] and not res["reached_main_menu"]:
                res["menu_stall"] = True

        # A max-frame result gets one bounded gMain re-discovery in
        # apply_macro(); only record menu_stall if that recovery also fails.
        if res["unexpected_menu"]:
            self.diagnostics.append({
                "resolve_turn": "unexpected_menu",
                "menu_trace": list(res["menu_trace"]),
                "a_presses": res["a_presses"], "frames": res["frames"]})
        return res

    def play_battle(self, policy_fn, obs_fn):
        """Play the whole battle with a PINNED ``policy_fn``. Returns a raw
        result dict for ``twoby2.battle_summary.summarize_battle`` +
        ``{'duration_steps', 'terminated', 'post_battle_obs'}``."""
        steps = 0
        outcome = "loss"
        while self.in_battle() and self._turn < self.max_turns:
            snap = self.snapshot()
            mask = self.action_mask(snap)
            try:
                obs = obs_fn(snap, mask)
                a = int(policy_fn(obs))
            except Exception as exc:
                self.diagnostics.append({"warn": f"policy error {exc}; WAIT"})
                a = 0
            if not (0 <= a < len(self._macros)) or not mask[a]:
                a = next((i for i, m in enumerate(mask) if m), 0)
            snap1, ev = self.apply_macro(self._macros[a])
            steps += 1
            if ev.get("fled"):
                outcome = "fled"; break
            if ev.get("wipe"):
                outcome = "wipe"; break
            if ev.get("catch_success"):
                outcome = "caught"; break
            if ev.get("battle_won"):
                outcome = "win"; break
            if ev.get("terminal_unknown"):
                outcome = "terminal_unknown"; break
            if ev.get("menu_stall"):
                outcome = "menu_stall"; break
            if snap1.get("in_battle") is not True:
                break
        else:
            if self._turn >= self.max_turns and self.in_battle():
                outcome = "timeout"
        return {
            "outcome": outcome,
            "turns": self._turn,
            "duration_steps": steps,
            "diagnostics": self.diagnostics,
        }


def classify_turn_outcome(macro, res):
    """Pure classification of one resolved battle turn from the observed
    animation (:meth:`EmulatorBattleDriver._resolve_turn` output). Returns a
    dict with exactly the outcome flags that changed. A battle that ended
    without a clear win/wipe/flee signal is ``terminal_unknown`` - a diagnosed
    fault, never a silent timeout.
    """
    out = {"fled": False, "wipe": False, "battle_won": False, "enemy_ko": False,
           "terminal_unknown": False, "menu_stall": False, "aborted": False,
           "reason": ""}
    left = bool(res.get("left_battle"))
    ko_seen = bool(res.get("enemy_ko_seen"))

    # "an enemy fainted this turn" - independent of whether the battle then
    # ended (a win) or a replacement came in (battle continues).
    out["enemy_ko"] = ko_seen

    if macro == RUN_ACTION and left:
        out["fled"] = True
    elif res.get("own_party_wiped") or (left and res.get("own_party_wiped_late")):
        out["wipe"] = True
        out["enemy_ko"] = False
    elif ko_seen and left:
        out["battle_won"] = True
    elif ko_seen:
        pass                       # enemy_ko already set; battle continues
    elif left:
        out["terminal_unknown"] = True
        out["reason"] = ("battle ended but no win/wipe/flee signal was observed "
                         f"(enemy_ko_seen={ko_seen}, frames={res.get('frames')})")
    elif res.get("menu_stall") or res.get("unexpected_menu"):
        out.update(menu_stall=True, aborted=True,
                   reason=(f"stuck / unexpected menu after {res.get('frames')} frames "
                           f"(trace={res.get('menu_trace')}, "
                           f"a_presses={res.get('a_presses')})"))
    return out


def _switch_loop(recent):
    return len(recent) >= 3 and recent[-1] == recent[-3] and recent[-1] != recent[-2]
