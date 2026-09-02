import stable_retro as retro
import cv2
import os

custom_dir = os.path.expanduser("~/pokemon_ai_project/custom_integrations")
retro.data.Integrations.add_custom_path(custom_dir)

print("Initialisiere Pokémon Feuerrot...")
env = retro.make(
    game="PokemonFireRed-Gba",
    state=retro.State.NONE,
    inttype=retro.data.Integrations.CUSTOM_ONLY
)

obs, info = env.reset()
print(f"Erfolgreich geladen! Bildauflösung (Shape): {obs.shape}")

for _ in range(60):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

cv2.imwrite("test_frame.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
print("Test erfolgreich! 'test_frame.png' wurde erstellt.")

env.close()
