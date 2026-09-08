#!/usr/bin/env python3
"""Display the ninth real Battle-PPO worker + its live reward stream.

The worker itself owns the emulator and writes two files:
  * ``runtime/battle/battle_worker_live.jpg``  - the current frame
  * ``runtime/battle/battle_worker_live.json`` - episode / turn / outcome and
    the decomposed per-turn reward events (written by ``BattleEnv`` only for
    the mirror worker).

This tool just composites them - it never steps an emulator, never learns.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "runtime", "battle", "battle_worker_live.jpg")
SIDECAR = os.path.join(ROOT, "runtime", "battle", "battle_worker_live.json")
WINDOW = "PKMai - BATTLE (lernt mit)"

_W, _H = 1000, 500
_GX, _GY, _GW, _GH = 8, 44, 660, 440          # game frame box (GBA 3:2)
_PX0, _PX1 = 680, 992                          # right panel x-range
_PY0 = 44                                      # panels start below the header strip

_POS = (110, 215, 110)   # BGR green
_NEG = (90, 90, 235)     # BGR red
_NEU = (150, 150, 150)
_GOLD = (100, 190, 225)


def _parse(ev):
    try:
        name, amt = str(ev).rsplit(":", 1)
        return name, float(amt)
    except ValueError:
        return str(ev), None


def _col(a):
    if a is None or abs(a) < 1e-9:
        return _NEU
    return _POS if a > 0 else _NEG


def _read_sidecar():
    try:
        with open(SIDECAR) as f:
            d = json.load(f)
        if time.time() - float(d.get("updated", 0)) > 25:
            d["_stale"] = True
        return d
    except (OSError, ValueError):
        return None


def render(frame, data):
    c = np.full((_H, _W, 3), 24, np.uint8)

    def label(t, x, y, color=(205, 205, 205), s=.44, w=1):
        cv2.putText(c, str(t), (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, w, cv2.LINE_AA)

    def rlabel(t, xr, y, color, s=.4, w=1):
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, w)
        label(t, xr - tw, y, color, s, w)

    def panel(x0, y0, x1, y1, fill=(44, 44, 46), border=(64, 64, 70)):
        cv2.rectangle(c, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(c, (x0, y0), (x1, y1), border, 1)

    def bar(x0, y, w, frac, color):
        cv2.rectangle(c, (x0, y), (x0 + w, y + 8), (54, 54, 60), -1)
        cv2.rectangle(c, (x0, y), (x0 + int(w * max(0.0, min(1.0, frac))), y + 8),
                      color, -1)

    # game frame
    if frame is not None:
        c[_GY:_GY + _GH, _GX:_GX + _GW] = cv2.resize(frame, (_GW, _GH),
                                                     interpolation=cv2.INTER_NEAREST)
    else:
        label("warte auf battle_worker_live.jpg ...", _GX + 20, _GY + _GH // 2,
              (120, 125, 135), .5)

    d = data or {}
    stale = bool(d.get("_stale")) or not data
    ep = int(d.get("episode", 0) or 0)
    turn = int(d.get("turn", 0) or 0)
    epr = float(d.get("episode_reward", 0.0) or 0.0)
    events = list(d.get("events", []) or [])

    # header
    label("PKMAI - BATTLE  (lernt mit)", 16, 30, (255, 190, 150), .62)
    (tw, _), _ = cv2.getTextSize("PKMAI - BATTLE  (lernt mit)",
                                 cv2.FONT_HERSHEY_SIMPLEX, .62, 1)
    sub = f"Episode {ep}   |   Turn {turn}   |   Reward {epr:+.2f}"
    if stale:
        sub += "     (kein Live-Update)"
    label(sub, 16 + tw + 24, 29, (150, 205, 240) if not stale else (120, 120, 135), .46)

    # -- right panel: OVERVIEW ----------------------------------------
    x0, x1 = _PX0, _PX1
    y = _PY0
    panel(x0, y, x1, y + 176)
    cv2.rectangle(c, (x0, y), (x1, y + 22), (52, 44, 40), -1)
    label("OVERVIEW", x0 + 12, y + 16, (232, 210, 196), .44)

    oc = d.get("outcome")
    badge = {"win": ("WIN", _POS), "battle_won": ("WIN", _POS),
             "wipe": ("WIPE", _NEG), "fled": ("FLED", _GOLD),
             "timeout": ("TIMEOUT", _NEG), "terminal_unknown": ("UNKNOWN", _NEG),
             "menu_stall": ("MENU-STALL", _NEG), "unreadable": ("UNREADABLE", _NEG)}
    txt, bcol = badge.get(str(oc), ("IN BATTLE", (120, 150, 90)))
    cv2.rectangle(c, (x0 + 12, y + 30), (x0 + 132, y + 50), bcol, -1)
    label(txt, x0 + 19, y + 45, (20, 20, 20), .42, 1)
    label(f"macro {d.get('macro') or '-'}", x0 + 144, y + 45, (200, 205, 214), .38)

    our = d.get("our") or [0, 0, 0]
    enemy = d.get("enemy") or [0, 0, 0]
    label(f"OWN   L{our[2]}", x0 + 12, y + 74, (200, 210, 220), .36)
    bar(x0 + 108, y + 66, 150, (our[0] / our[1]) if our[1] else 0, _POS)
    rlabel(f"{our[0]}/{our[1]}", x1 - 12, y + 74, (170, 180, 192), .34)
    label(f"ENEMY L{enemy[2]}", x0 + 12, y + 96, (200, 210, 220), .36)
    bar(x0 + 108, y + 88, 150, (enemy[0] / enemy[1]) if enemy[1] else 0, _NEG)
    rlabel(f"{enemy[0]}/{enemy[1]}", x1 - 12, y + 96, (170, 180, 192), .34)

    rc = _POS if epr >= 0 else _NEG
    label("EPISODE REWARD", x0 + 12, y + 120, (140, 143, 152), .32)
    label(f"{epr:+.3f}", x0 + 120, y + 124, rc, .52)

    amounts = [(n, a) for n, a in (_parse(e[1]) for e in events) if a is not None]
    if amounts:
        best = max(amounts, key=lambda t: t[1])
        worst = min(amounts, key=lambda t: t[1])
        label(f"hoch  {best[1]:+.2f}  {best[0][:14]}", x0 + 12, y + 146, _POS, .32)
        label(f"tief  {worst[1]:+.2f}  {worst[0][:14]}", x0 + 12, y + 164, _NEG, .32)

    # -- right panel: REWARD EVENTS ---------------------------------
    ty0, ty1 = y + 186, _H - 8
    panel(x0, ty0, x1, ty1, fill=(33, 33, 36))
    cv2.rectangle(c, (x0, ty0), (x1, ty0 + 22), (36, 50, 40), -1)
    label("REWARD EVENTS", x0 + 12, ty0 + 16, (170, 225, 170), .42)
    label("neueste zuerst", x1 - 96, ty0 + 16, (110, 130, 115), .3)

    row_h = 22
    vis = list(reversed(events))[:16]
    if not vis:
        label("noch keine Events.", x0 + 16, ty0 + 46, (120, 125, 135), .38)
    for i, (t, ev) in enumerate(vis):
        ry = ty0 + 30 + i * row_h
        if ry + row_h > ty1:
            break
        if i % 2:
            cv2.rectangle(c, (x0 + 1, ry - 12), (x1 - 1, ry + row_h - 12), (39, 39, 42), -1)
        name, amt = _parse(ev)
        col = _col(amt)
        cv2.circle(c, (x0 + 12, ry - 4), 4, col, -1, cv2.LINE_AA)
        label(f"T{t}", x0 + 22, ry, (110, 115, 125), .3)
        label(name[:18], x0 + 52, ry, (205, 208, 215), .36)
        if amt is not None:
            rlabel(f"{amt:+.3f}", x1 - 10, ry, col, .36, 1)

    return c


def main():
    print("Zeigt den 9. echten Battle-Fighter + seinen Reward-Stream.")
    print("Ctrl-C im Terminal oder q im Bildfenster beendet nur die Anzeige.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, _W, _H)
    last_frame = None
    try:
        while True:
            if os.path.isfile(FRAME):
                f = cv2.imread(FRAME)
                if f is not None:
                    last_frame = f
            canvas = render(last_frame, _read_sidecar())
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
