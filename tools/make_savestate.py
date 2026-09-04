import stable_retro as retro
import cv2
import numpy as np
import os
import gzip

# Muss auf denselben Ordner zeigen wie das eigentliche Training
# (src/pokemon_env.py: CUSTOM_DIR = LOCAL_DIR/custom_integrations). Der alte
# Pfad ohne "local/" existierte nicht -> Speichern crashte mit FileNotFoundError.
custom_dir = os.path.expanduser("~/pokemon_ai_project/local/custom_integrations")
retro.data.Integrations.add_custom_path(custom_dir)

env = retro.make(
    game="PokemonFireRed-Gba",
    state=retro.State.NONE,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
    # Sonst rendert env.step() bei jedem Aufruf zusaetzlich in ein eigenes
    # Pyglet/Cocoa-Fenster (retro_env.py: render_mode="human" per Default) -
    # das crasht auf diesem Mac. Wir zeigen das Bild selbst via cv2 an.
    render_mode=None,
)
env.reset()

# env.buttons ist NICHT die einfache 10-Knopf-GBA-Reihenfolge, sondern das
# generische 12-Slot-Layout (SNES-Erbe): ['B', None, 'SELECT', 'START', 'UP',
# 'DOWN', 'LEFT', 'RIGHT', 'A', None, 'L', 'R'] - A liegt auf Index 8, nicht 1!
# Eine fest verdrahtete Liste war hier tagelang falsch (A traf einen None-Slot
# und tat schlicht nichts). Immer per .index() dynamisch nachschlagen, so wie
# es tools/create_savestate.py von Anfang an richtig gemacht hat.
buttons = list(env.buttons)
num_buttons = len(buttons)

def get_mask(name):
    m = [0] * num_buttons
    if name in buttons:
        m[buttons.index(name)] = 1
    return m

btn_none = get_mask(None) if False else [0] * num_buttons

print("\n--- STEUERUNG ---")
print("Pfeiltasten ODER i/j/k/l : Bewegen (Hoch/Links/Runter/Rechts)")
print("Taste 'a'   : A-Knopf (Bestaetigen / Text weiter)")
print("Taste 's'   : B-Knopf (Abbrechen / Rennen)")
print("Taste 'd'   : Start-Knopf")
print("Taste 'f'   : Select-Knopf")
print("Taste 'q'   : Beenden & State speichern")
print("-----------------\n")
print("Hinweis: falls die Pfeiltasten in diesem Fenster nicht reagieren")
print("(macOS/OpenCV ist da unzuverlaessig), nutze i/j/k/l - die sind")
print("bewusst getrennt von a/s/d/f (Knoepfe) belegt, keine Ueberschneidung.\n")
print("WICHTIG: erst auf das Spiel-Fenster klicken, damit es Tastatur-Fokus hat!\n")

cv2.namedWindow("Pokemon FireRed - Setup State", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pokemon FireRed - Setup State", 480, 320)

# FireRed registriert einen Tastendruck erst nach mehreren gehaltenen Frames
# (siehe src/pokemon_env.py ACTION_HOLD_FRAMES/ACTION_RELEASE_FRAMES - dort
# steht auch: "4 Frames drehen nur die Spielfigur, ein echter Schritt braucht
# ~16"). Gleiche Werte wie Training/Watcher benutzen.
ACTION_HOLD_FRAMES = 12
ACTION_RELEASE_FRAMES = 6

BTN = {
    "UP": get_mask("UP"), "DOWN": get_mask("DOWN"),
    "LEFT": get_mask("LEFT"), "RIGHT": get_mask("RIGHT"),
    "A": get_mask("A"), "B": get_mask("B"),
    "START": get_mask("START"), "SELECT": get_mask("SELECT"),
}

# Bekannte Sonderwerte einiger OpenCV/macOS-Backends fuer Pfeiltasten VOR jeder
# Maskierung. key & 0xFF darauf anzuwenden wuerde sie auf einen anderen Wert
# projizieren als die 0/1/2/3-Kurzform unten - deshalb hier ungemaskiert pruefen.
ARROW_RAW = {
    2490368: "UP", 63232: "UP", 65362: "UP",
    2621440: "DOWN", 63233: "DOWN", 65364: "DOWN",
    2424832: "LEFT", 63234: "LEFT", 65361: "LEFT",
    2555904: "RIGHT", 63235: "RIGHT", 65363: "RIGHT",
}

# Statt pro Tastendruck 18x env.step() blockierend hintereinander auszufuehren
# (Bild aktualisiert sich dann nur alle 18 Frames -> ruckelig, Events stauen
# sich), wird jetzt jeden einzelnen Frame genau 1x env.step() aufgerufen und
# ueber hold_counter nachverfolgt, in welcher Phase (Halten/Loslassen) einer
# laufenden 18-Frame-Aktion wir gerade sind. Neue Tasten werden dabei staendig
# mit 1ms Poll-Intervall erkannt statt nur alle 20ms.
current_action = btn_none
hold_counter = 0
total_frames = 0

while True:
    raw_key = cv2.waitKey(1)
    key = raw_key & 0xFF
    if ord('A') <= key <= ord('Z'):
        key += 32  # Caps Lock/Shift-Grossbuchstaben auf Kleinbuchstaben normalisieren

    if raw_key != -1:
        print(f"[DEBUG] Taste erkannt: raw={raw_key}  maskiert={key}")

        if key == ord('q'):
            print("Speichere State...")
            state_data = env.em.get_state()
            state_dir = os.path.join(custom_dir, "PokemonFireRed-Gba")
            os.makedirs(state_dir, exist_ok=True)
            state_path = os.path.join(state_dir, "StartGame.state")
            # StartGame.state ist der bestaetigte, verifizierte Spielstand -
            # wird nie stillschweigend ueberschrieben. Ein neuer Lauf landet
            # stattdessen unter einem durchnummerierten Namen.
            if os.path.exists(state_path):
                n = 1
                while os.path.exists(os.path.join(state_dir, f"StartGame_new{n}.state")):
                    n += 1
                state_path = os.path.join(state_dir, f"StartGame_new{n}.state")
                print(f"StartGame.state existiert schon und wird NICHT ueberschrieben.")
            with gzip.open(state_path, "wb") as f:
                f.write(state_data)
            print(f"State gespeichert unter:\n{state_path}")
            break

        # Neue Eingabe nur uebernehmen, wenn gerade keine Aktion laeuft
        if hold_counter == 0:
            name = None
            if raw_key in ARROW_RAW:
                name = ARROW_RAW[raw_key]
            elif key in (0, 1, 2, 3):
                name = ["UP", "DOWN", "LEFT", "RIGHT"][key]
            elif key == ord('i'): name = "UP"
            elif key == ord('k'): name = "DOWN"
            elif key == ord('j'): name = "LEFT"
            elif key == ord('l'): name = "RIGHT"
            elif key == ord('a'): name = "A"
            elif key == ord('s'): name = "B"
            elif key == ord('d'): name = "START"
            elif key == ord('f'): name = "SELECT"

            if name is not None:
                current_action = BTN[name]
                hold_counter = ACTION_HOLD_FRAMES + ACTION_RELEASE_FRAMES

    # Schrittweise ausfuehren statt harter Blockade-Schleife
    step_btn = btn_none
    if hold_counter > ACTION_RELEASE_FRAMES:
        step_btn = current_action  # Taste gedrueckt halten
        hold_counter -= 1
    elif hold_counter > 0:
        step_btn = btn_none        # Taste loslassen (Release-Phase)
        hold_counter -= 1

    obs, reward, terminated, truncated, info = env.step(step_btn)
    total_frames += 1

    cv2.imshow("Pokemon FireRed - Setup State", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    if total_frames % 120 == 0:
        print(f"[STATUS] Frame {total_frames}  Helligkeit={np.mean(obs):.1f}")
        debug_dir = os.path.expanduser("~/pokemon_ai_project/runtime/savestate_debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "latest.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

env.close()
cv2.destroyAllWindows()
