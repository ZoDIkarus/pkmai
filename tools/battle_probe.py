#!/usr/bin/env python3
"""RAM-Adressen-Finder fuer die DEUTSCHE Pokemon FireRed ROM.

Interaktiv: du spielst selbst, machst Snapshots vor/im Kampf, das Tool
gibt die Byte-Offsets aus die sich aendern (0 -> !=0 = idealer "im Kampf"-Flag).

Steuerung im Fenster (Buchstaben, KEINE Pfeiltasten):
  w/a/s/d = laufen   j = A   k = B   n = START
  l = Savestate laden (naeher am Kampf)
  p = Snapshot "NICHT im Kampf"
  o = Snapshot "IM Kampf"   (dann Diff-Ausgabe)
  q = beenden
Das Fenster muss aktiv/fokussiert sein damit Tasten ankommen.

Aufruf:  python tools/battle_probe.py
"""
import gzip
import os
import sys

import cv2
import numpy as np
import stable_retro as retro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
retro.data.Integrations.add_custom_path(
    os.path.join(ROOT, "local", "custom_integrations")
)
import glob as _glob
_cur = os.path.join(ROOT, "runtime", "curriculum_shared")
# tiefster outdoor_N (auf Route 1 -> Gras -> wilde Kaempfe leicht ausloesbar),
# sonst starter.state
_outs = sorted(_glob.glob(os.path.join(_cur, "outdoor_*.state.gz")))
STATE = _outs[-1] if _outs else os.path.join(_cur, "starter.state.gz")
print(f"Lade-Savestate ('l' im Fenster): {os.path.basename(STATE)}")

env = retro.make(
    game="PokemonFireRed-Gba",
    state=retro.State.NONE,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
    render_mode=None,
)
env.reset()
btns = list(env.buttons)


def mask(*names):
    m = [0] * len(btns)
    for n in names:
        if n in btns:
            m[btns.index(n)] = 1
    return m


# Buchstaben statt Pfeiltasten (cv2-Pfeilcodes sind auf macOS unzuverlaessig).
KEYMAP = {
    ord("w"): mask("UP"), ord("s"): mask("DOWN"),
    ord("a"): mask("LEFT"), ord("d"): mask("RIGHT"),
    ord("j"): mask("A"), ord("k"): mask("B"), ord("n"): mask("START"),
    # Pfeiltasten falls sie doch gehen:
    82: mask("UP"), 84: mask("DOWN"), 81: mask("LEFT"), 83: mask("RIGHT"),
    0: mask("UP"), 1: mask("DOWN"), 2: mask("LEFT"), 3: mask("RIGHT"),
}
NOOP = [0] * len(btns)

snap_out = None
snap_in = None
cv2.namedWindow("battle_probe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("battle_probe", 720, 480)
print(__doc__)

MOVE_KEYS = {ord("w"), ord("s"), ord("a"), ord("d"), 0, 1, 2, 3, 81, 82, 83, 84}
action = NOOP
hold = 0          # verbleibende Frames fuer die aktuelle Bewegung
while True:
    if hold > 0:
        env.step(action)
        hold -= 1
    else:
        env.step(NOOP)
        action = NOOP

    screen = cv2.cvtColor(env.get_screen(), cv2.COLOR_RGB2BGR)
    screen = cv2.resize(screen, (720, 480), interpolation=cv2.INTER_NEAREST)
    txt = f"w/a/s/d laufen  j=A k=B  |  p=out({'OK' if snap_out is not None else '-'})  o=in  l=load  q=quit"
    cv2.putText(screen, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
               (0, 230, 118), 1, cv2.LINE_AA)
    cv2.imshow("battle_probe", screen)

    k = cv2.waitKey(16) & 0xFF
    if k == 255:
        continue
    if k == ord("q"):
        break
    if k in KEYMAP:
        action = KEYMAP[k]
        hold = 16 if k in MOVE_KEYS else 6
    elif k == ord("l") and os.path.exists(STATE):
        with gzip.open(STATE, "rb") as f:
            env.em.set_state(f.read())
        print(f"{os.path.basename(STATE)} geladen")
    elif k == ord("p"):
        snap_out = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
        print(f"Snapshot NICHT im Kampf: {snap_out.nbytes} bytes")
    elif k == ord("o"):
        snap_in = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
        if snap_out is None:
            print("erst 'p' druecken (Snapshot ausserhalb Kampf, VORHER 10 Schritte laufen)")
            continue
        # Battle-Struct-Region: EWRAM 0x02022000 - 0x02024400
        LO, HI = 0x22000, 0x24400
        z2n = [o for o in np.where((snap_out == 0) & (snap_in != 0))[0]
               if LO <= o <= HI and 1 <= snap_in[o] <= 12]
        print(f"\n=== Kandidaten in Battle-Region, kleiner stabiler Wert (0 -> 1..12) ===")
        for o in z2n[:40]:
            print(f"  Offset {o:>7} (dez)   EWRAM 0x{0x02000000+o:08X}"
                  f"   {snap_out[o]} -> {snap_in[o]}")
        print(f"\n{len(z2n)} Kandidaten. Jetzt Kampf beenden (fliehen/gewinnen),"
              f" im Overworld 'u' druecken -> zeigt welche wieder auf 0 gehen.")

    elif k == ord("u"):
        if snap_out is None or snap_in is None:
            print("erst p (Overworld) und o (Kampf) machen")
            continue
        snap_after = np.frombuffer(env.get_ram(), dtype=np.uint8).copy()
        LO, HI = 0x22000, 0x24400
        toggles = [o for o in range(LO, HI)
                   if snap_out[o] == 0 and snap_in[o] != 0 and snap_after[o] == 0]
        print(f"\n=== ECHTE Battle-Toggles: 0 (overworld) -> N (kampf) -> 0 (danach) ===")
        for o in toggles[:20]:
            print(f"  in_battle-ADRESSE:  {o}  (dez)   EWRAM 0x{0x02000000+o:08X}"
                  f"   Wert im Kampf: {snap_in[o]}")
        if toggles:
            print(f"\n>>> Nimm die erste: address {toggles[0]}, type |u1 <<<")
        else:
            print("keine sauberen Toggles - schick mir die 'o'-Kandidatenliste")

cv2.destroyAllWindows()
env.close()
