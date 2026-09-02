import stable_retro as retro
import cv2
import os
import gzip

custom_dir = os.path.expanduser("~/pokemon_ai_project/custom_integrations")
retro.data.Integrations.add_custom_path(custom_dir)

env = retro.make(
    game="PokemonFireRed-Gba",
    state=retro.State.NONE,
    inttype=retro.data.Integrations.CUSTOM_ONLY
)
env.reset()

print("\n--- STEUERUNG ---")
print("Pfeiltasten : Bewegen (Hoch, Runter, Links, Rechts)")
print("Taste 'a'   : A-Knopf (Bestaetigen / Text weiter)")
print("Taste 's'   : B-Knopf (Abbrechen / Rennen)")
print("Taste 'd'   : Start-Knopf")
print("Taste 'f'   : Select-Knopf")
print("Taste 'q'   : Beenden & State speichern")
print("-----------------\n")

cv2.namedWindow("Pokemon FireRed - Setup State", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pokemon FireRed - Setup State", 480, 320)

buttons = ['B', 'A', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']

while True:
    action = [0] * len(buttons)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        print("Speichere State...")
        state_data = env.em.get_state()
        state_path = os.path.expanduser("~/pokemon_ai_project/custom_integrations/PokemonFireRed-Gba/StartGame.state")
        with gzip.open(state_path, "wb") as f:
            f.write(state_data)
        print(f"State erfolgreich gespeichert unter:\n{state_path}")
        break

    if key == 0 or key == 2490368: action[4] = 1   # UP
    elif key == 1 or key == 2621440: action[5] = 1 # DOWN
    elif key == 2 or key == 2424832: action[6] = 1 # LEFT
    elif key == 3 or key == 2555904: action[7] = 1 # RIGHT
    elif key == ord('a'): action[1] = 1            # A
    elif key == ord('s'): action[0] = 1            # B
    elif key == ord('d'): action[3] = 1            # START
    elif key == ord('f'): action[2] = 1            # SELECT

    obs, reward, terminated, truncated, info = env.step(action)
    cv2.imshow("Pokemon FireRed - Setup State", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

env.close()
cv2.destroyAllWindows()
