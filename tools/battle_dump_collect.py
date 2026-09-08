#!/usr/bin/env python3
"""Collect controlled in-battle RAM dumps for BPRD battle-address verification.

Runs its OWN isolated emulator (the running trainer / watcher / web are NOT
touched). You play in the window; each capture writes a uniquely-named RAM dump
plus a JSON sidecar with the full verified party, the encounter id, the battle
kind, the labelled menu state and the expected cursor position, so
``twoby2.ram_battle_probe`` can score them without ever producing a false
VERIFIED.

Window keys
-----------
  w a s d  walk      j = A      k = B      n = START      l = reload start state
  q        quit

  Encounter / context:
    t  toggle battle kind for THIS encounter: wild  <->  trainer
    New encounters and leaving battle are detected automatically.
    W remains an out-of-battle fallback. ``e`` is a developer-only encounter
    counter override and does NOT replace the verified in-battle signal.

  Menu state you are currently on (press to set the label):
    m  main menu (2x2: FIGHT top-left, BAG top-right, POKEMON bottom-left, RUN bottom-right)
    v  move menu (2x2 list of the four moves)
    p  party menu (switch list)

  "I just switched to party slot N" (party-slot keys, NOT the number row):
    7 = slot 0   8 = slot 1   9 = slot 2   0 = slot 3   - = slot 4   = = slot 5

  Capture (writes a dump with all the current labels):
    SPACE          generic capture
    1 2 3 4        capture AND set expected move-cursor = 0 / 1 / 2 / 3
    F B P R        capture AND set expected action-cursor = FIGHT(0)/BAG(1)/POKEMON(2)/RUN(3)

Do the whole thing for a WILD fight and again for a TRAINER fight (press `t`
once when the trainer fight begins) — a field is only VERIFIED with >= 2
DISTINCT encounters.
Dumps: runtime/ram_probe/<timestamp>/
"""
from __future__ import annotations

import datetime
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from battle_state import MainBattleReader

DEFAULT_START_STATE = os.path.join(
    ROOT, "brain_backups", "ram_probe_route1_seed", "stage_frontier_2.state.gz")
# Backwards-compatible name for callers/tests that imported it.
START_STATE = DEFAULT_START_STATE
# Verified by the existing custom integration (data.json: 146376 decimal).
# Keep the hexadecimal spelling here to make accidental 0x23B48/0x23BC8
# transpositions visible in review.
IN_BATTLE_FLAG = 0x23BC8

# key -> expected move cursor (0..3). The number row is ONLY move-cursor.
MOVE_KEYS = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3}
# key -> expected action cursor (0..3)
ACTION_KEYS = {ord("F"): 0, ord("B"): 1, ord("P"): 2, ord("R"): 3}
ACTION_NAMES = {0: "fight", 1: "bag", 2: "pokemon", 3: "run"}
# key -> party slot (0..5). NOT the number row.
SLOT_KEYS = {ord("7"): 0, ord("8"): 1, ord("9"): 2,
             ord("0"): 3, ord("-"): 4, ord("="): 5}

WALK_KEYS = (ord("w"), ord("s"), ord("a"), ord("d"))
BUTTON_KEYS = (ord("j"), ord("k"), ord("n"))

HUD_LINE = ("SPACE / 1-4 / F B P R = capture   m v p = menu   "
            "7 8 9 0 - = slot   t trainer   l reload   q quit")


def new_state():
    return {"encounter": 0, "battle_kind": "wild", "menu_state": "none",
            "active_slot": 0, "out_of_battle": False, "in_battle": False,
            "count": {}}


def stable_battle_presence(previous, live_read):
    """Use gMain.inBattle and hold the last value while it is being located.

    The integration byte is retained only as diagnostic data because it can
    legitimately toggle during different phases of one battle.
    """
    return bool(previous) if live_read is None else bool(live_read)


def update_battle_transition(st, was_in_battle, is_in_battle):
    """Update labels when the verified in-battle flag changes.

    A fresh 0 -> nonzero transition creates the encounter automatically, so
    the user does not have to press ``e`` before a fight.  The legacy ``e``
    key remains available only as a manual fallback.
    """
    if is_in_battle and not was_in_battle:
        st["encounter"] += 1
        st["out_of_battle"] = False
        st["menu_state"] = "none"
        st["active_slot"] = 0
        st["in_battle"] = True
        return f"AUTO new encounter enc{st['encounter']:02d} ({st['battle_kind']})"
    if was_in_battle and not is_in_battle:
        st["out_of_battle"] = True
        st["menu_state"] = "none"
        st["in_battle"] = False
        return "AUTO OUT OF BATTLE"
    return None


def reset_context_after_reload(st):
    """Reset game-context labels but retain this probe session's dumps.

    File counters and the encounter number must survive a savestate reload;
    otherwise the next fight could reuse names such as
    ``action_cursor_fight_01`` and overwrite an earlier wild-fight dump.
    """
    encounter = st["encounter"]
    counts = st["count"]
    st.update(new_state())
    st["encounter"] = encounter
    st["count"] = counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", default=DEFAULT_START_STATE,
        help=".state.gz to load read-only (default: frozen Route-1 RAM-probe seed)")
    return parser.parse_args(argv)


def resolve_state_path(path):
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return os.path.realpath(path)


def route_key(k, st, capture_fn):
    """Pure key dispatch — no emulator, separately testable.

    ``capture_fn(label, *, expected_move=None, expected_action=None)`` is called
    for capture keys. ``st`` is mutated in place for context/state keys.
    Returns a ``(kind, payload)`` intent for the caller to act on:
      ("walk", key) ("button", key) ("reload", None) ("quit", None)
      ("state", str) ("capture", label) ("ignored", key)

    Order matters: MOVE_KEYS (number row) are checked BEFORE SLOT_KEYS so
    pressing 1-4 always captures a move cursor and never changes the party slot.
    """
    if k == ord("q"):
        return ("quit", None)
    if k in WALK_KEYS:
        return ("walk", k)
    if k in BUTTON_KEYS:
        return ("button", k)
    if k == ord("l"):
        return ("reload", None)

    # --- capture keys (checked before slot keys) ---
    if k == ord(" "):
        capture_fn("generic")
        return ("capture", "generic")
    if k in MOVE_KEYS:
        cur = MOVE_KEYS[k]
        capture_fn(f"move_cursor_{cur}", expected_move=cur)
        return ("capture", f"move_cursor_{cur}")
    if k in ACTION_KEYS:
        cur = ACTION_KEYS[k]
        capture_fn(f"action_cursor_{ACTION_NAMES[cur]}", expected_action=cur)
        return ("capture", f"action_cursor_{ACTION_NAMES[cur]}")

    # --- context / label keys ---
    if k == ord("e"):
        st["encounter"] += 1
        st["out_of_battle"] = False
        st["active_slot"] = 0
        return ("state", f"new encounter enc{st['encounter']:02d} ({st['battle_kind']})")
    if k == ord("t"):
        st["battle_kind"] = "trainer" if st["battle_kind"] == "wild" else "wild"
        return ("state", f"battle kind: {st['battle_kind']}")
    if k == ord("W"):
        st["out_of_battle"] = True
        st["menu_state"] = "none"
        return ("state", "OUT OF BATTLE")
    if k == ord("m"):
        st["menu_state"], st["out_of_battle"] = "main", False
        return ("state", "menu: main")
    if k == ord("v"):
        st["menu_state"], st["out_of_battle"] = "move", False
        return ("state", "menu: move")
    if k == ord("p"):
        st["menu_state"], st["out_of_battle"] = "party", False
        return ("state", "menu: party")
    if k in SLOT_KEYS:
        st["active_slot"] = SLOT_KEYS[k]
        return ("state", f"active party slot now {st['active_slot']}")

    return ("ignored", k)


def make_capture_fn(env, fr, outdir, st):
    def capture(label, *, expected_move=None, expected_action=None):
        ram = bytes(env.get_ram())
        in_battle = ram[IN_BATTLE_FLAG] if len(ram) > IN_BATTLE_FLAG else None
        loc = fr.read_player_location(_Env(env))
        party = fr.read_player_party(_Env(env))
        kind = ("out_of_battle" if st["out_of_battle"] or not st["in_battle"]
                else st["battle_kind"])
        enc = (f"enc{st['encounter']:02d}_{st['battle_kind']}"
               if kind != "out_of_battle" else "out_of_battle")
        st["count"][label] = st["count"].get(label, 0) + 1
        base = os.path.join(outdir, f"{label}_{st['count'][label]:02d}")
        with gzip.open(base + ".ram.gz", "wb") as f:
            f.write(ram)
        sidecar = {
            "label": label, "context": label, "encounter_id": enc,
            "battle_kind": kind,
            "menu_state": ("none" if kind == "out_of_battle" else st["menu_state"]),
            "in_battle_flag_0x23BC8": int(in_battle) if in_battle is not None else None,
            "active_party_slot": None if kind == "out_of_battle" else st["active_slot"],
            "expected_move_cursor": expected_move,
            "expected_action_cursor": expected_action,
            "party": [
                {"slot": m.get("slot", i), "species_id": m.get("species_id"),
                 "level": m.get("level"), "cur_hp": m.get("cur_hp"),
                 "max_hp": m.get("max_hp"), "fainted": m.get("fainted")}
                for i, m in enumerate(party)],
            "location": {kk: loc.get(kk) for kk in ("map_bank", "map_id",
                                                    "x_pos", "y_pos")},
            "ram_len": len(ram),
        }
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"  captured {label} #{st['count'][label]:02d}  "
              f"enc={enc} kind={kind} menu={sidecar['menu_state']} "
              f"slot={sidecar['active_party_slot']} "
              f"exp_move={expected_move} exp_act={expected_action} "
              f"in_battle={in_battle} party_slots={len(party)}")
    return capture


class _Env:
    def __init__(self, env):
        self._env = env

    def get_ram(self):
        return self._env.get_ram()


def main(argv=None):
    args = parse_args(argv)
    start_state = resolve_state_path(args.state)
    if not os.path.isfile(start_state):
        print(f"start state not found: {start_state}")
        sys.exit(2)
    try:
        import cv2
        import numpy as np
        import stable_retro as retro
    except Exception as exc:                       # pragma: no cover
        print(f"needs cv2 + numpy + stable_retro: {exc}")
        sys.exit(1)

    import firered_ram as fr
    retro.data.Integrations.add_custom_path(os.path.join(ROOT, "local", "custom_integrations"))
    env = retro.make(game="PokemonFireRed-Gba", state=retro.State.NONE,
                     inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None)
    env.reset()
    with gzip.open(start_state, "rb") as f:
        env.em.set_state(f.read())
    env.em.step()

    reader_env = _Env(env)
    party = fr.read_player_party(reader_env)
    print(f"Loaded read-only state: {start_state}")
    for mon in party:
        moves = ", ".join(m.get("name", "?") for m in mon.get("moves", []))
        print(f"  slot {mon.get('slot', '?')}: {mon.get('name', '?')} L{mon.get('level', '?')} "
              f"HP {mon.get('cur_hp', '?')}/{mon.get('max_hp', '?')} | {moves}")

    outdir = os.path.join(ROOT, "runtime", "ram_probe",
                          datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    print(__doc__)
    print(f"dumps -> {outdir}\n")

    btns = list(env.buttons)

    def mask(*names):
        m = [0] * len(btns)
        for n in names:
            if n in btns:
                m[btns.index(n)] = 1
        return np.array(m, dtype=np.uint8)

    WALK = {ord("w"): mask("UP"), ord("s"): mask("DOWN"),
            ord("a"): mask("LEFT"), ord("d"): mask("RIGHT")}
    BTN = {ord("j"): mask("A"), ord("k"): mask("B"), ord("n"): mask("START")}
    NOOP = np.zeros(len(btns), dtype=np.uint8)

    st = new_state()
    capture = make_capture_fn(env, fr, outdir, st)
    action, hold = NOOP, 0
    battle_reader = MainBattleReader()
    ram = bytes(env.get_ram())
    was_in_battle = False

    cv2.namedWindow("battle_dump_collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("battle_dump_collect", 720, 540)

    while True:
        env.em.set_button_mask(action if hold > 0 else NOOP)
        env.em.step()
        hold = max(0, hold - 1)

        ram = bytes(env.get_ram())
        live_read = battle_reader.read(ram, frames=1)
        is_in_battle = stable_battle_presence(was_in_battle, live_read)
        transition = update_battle_transition(st, was_in_battle, is_in_battle)
        if transition:
            print(f"-- {transition} --")
        was_in_battle = is_in_battle

        frame = cv2.cvtColor(env.get_screen(), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (720, 540), interpolation=cv2.INTER_NEAREST)
        hud = (f"enc{st['encounter']:02d}/{st['battle_kind']}  menu={st['menu_state']}  "
               f"slot={st['active_slot']}  oob={st['out_of_battle']}")
        cv2.putText(frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 1)
        cv2.putText(frame, HUD_LINE, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (0, 230, 118), 1)
        cv2.imshow("battle_dump_collect", frame)
        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            continue

        intent, payload = route_key(k, st, capture)
        if intent == "quit":
            break
        if intent == "walk":
            action, hold = WALK[payload], 16
        elif intent == "button":
            action, hold = BTN[payload], 6
        elif intent == "reload":
            with gzip.open(start_state, "rb") as f:
                env.em.set_state(f.read())
            env.em.step()
            reset_context_after_reload(st)
            ram = bytes(env.get_ram())
            live_read = battle_reader.read(ram, frames=1)
            was_in_battle = stable_battle_presence(False, live_read)
            st["in_battle"] = was_in_battle
            print(f"reloaded read-only state: {start_state}")
        elif intent == "state":
            print(f"-- {payload} --")

    cv2.destroyAllWindows()
    env.close()
    total = sum(st["count"].values())
    print(f"\nDone. {total} dumps in {outdir}")
    print(f"Next:  python tools/battle_dump_score.py {outdir}")


if __name__ == "__main__":
    main()
