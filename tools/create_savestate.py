import stable_retro as retro
import numpy as np
import cv2
import os
import time

CUSTOM_DIR = os.path.expanduser("~/pokemon_ai_project/custom_integrations")
retro.data.Integrations.add_custom_path(CUSTOM_DIR)

print("🎮 Starte Retro-Instanz für Savestate-Erstellung...")
env = retro.make(
    game="PokemonFireRed-Gba",
    state=retro.State.NONE,
    inttype=retro.data.Integrations.CUSTOM_ONLY
)

env.reset()

btn_list = list(env.buttons)
num_buttons = len(btn_list)

def get_mask(name):
    mask = [0] * num_buttons
    if name in btn_list:
        mask[btn_list.index(name)] = 1
    return mask

btn_none  = [0] * num_buttons
btn_a     = get_mask("A")
btn_b     = get_mask("B")
btn_start = get_mask("START")
btn_down  = get_mask("DOWN")

def is_in_reds_room(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([100, 70, 70]), np.array([130, 255, 255]))
    wood_mask = cv2.inRange(hsv, np.array([10, 50, 70]), np.array([25, 200, 220]))
    has_carpet = np.sum(blue_mask > 0) > 150
    wood_ratio = np.sum(wood_mask > 0) / (bgr.shape[0] * bgr.shape[1])
    return has_carpet and (wood_ratio > 0.15)

print("⚡ Navigiere durch Intro & Namenswahl...")

in_room_count = 0
max_steps = 25000

for step in range(max_steps):
    # Intelligentes Makro: Hämmere A, drücke ab und zu START und bewege den Cursor bei Namenswahl nach unten auf OK
    if step % 200 in [0, 50, 100, 150]:
        action = btn_start
    elif step % 15 == 0:
        action = btn_down
    elif step % 2 == 0:
        action = btn_a
    else:
        action = btn_none

    env.step(action)
    screen = env.get_screen()
    bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    if step % 1000 == 0:
        mean_val = np.mean(bgr)
        print(f"⏳ Step {step} läuft... (Helligkeit: {mean_val:.1f})")

    if is_in_reds_room(bgr):
        in_room_count += 1
        if in_room_count >= 10:
            print(f"✅ Red's Zimmer (2F) erreicht bei Step {step}!")
            
            game_dir = os.path.join(CUSTOM_DIR, "PokemonFireRed-Gba")
            state_path = os.path.join(game_dir, "RedsRoom2F.state")
            state_data = env.em.get_state()
            with open(state_path, "wb") as f:
                f.write(state_data)
            print(f"💾 Savestate erfolgreich gesichert: {state_path}")
            
            capture_dir = os.path.expanduser("~/pokemon_ai_project/room_captures")
            os.makedirs(capture_dir, exist_ok=True)
            cv2.imwrite(os.path.join(capture_dir, "Reds_Zimmer_2F.png"), bgr)
            break
    else:
        in_room_count = 0

env.close()
