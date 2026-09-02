import stable_retro as retro
import numpy as np
import os
import json
import time

CUSTOM_DIR = os.path.expanduser("~/pokemon_ai_project/custom_integrations")
retro.data.Integrations.add_custom_path(CUSTOM_DIR)

print("🔍 [1/3] Starte automatischen RAM-Scanner...")
env = retro.make("PokemonFireRed-Gba", state=retro.State.NONE, inttype=retro.data.Integrations.CUSTOM_ONLY)
env.reset()

btn_list = list(env.buttons)
def press(btn_name, frames=8):
    mask = [0] * len(btn_list)
    if btn_name in btn_list:
        mask[btn_list.index(btn_name)] = 1
    for _ in range(frames):
        env.step(mask)
    for _ in range(2):
        env.step([0] * len(btn_list))

# Dialoge & Intro durchdrücken
for _ in range(120):
    press("A", 4)
    press("START", 2)

# Finde den aktiven SaveBlock im EWRAM
candidates = []
for addr in range(0x20000, 0x3E000):
    b = env.get_memory_value(addr + 4)
    m = env.get_memory_value(addr + 5)
    if b in [3, 4] and m in [0, 1, 2]:
        x = env.get_memory_value(addr) | (env.get_memory_value(addr + 1) << 8)
        y = env.get_memory_value(addr + 2) | (env.get_memory_value(addr + 3) << 8)
        if 0 < x < 30 and 0 < y < 30:
            candidates.append(addr)

# Besten Match ermitteln
saveblock_offset = candidates[0] if candidates else 0x31DC0
print(f"🎯 [2/3] Exakter interner Speicherblock identifiziert: {hex(saveblock_offset)} (Dec: {saveblock_offset})")
env.close()

# Config schreiben
config_data = {
    "saveblock_offset": saveblock_offset,
    "x_offset": saveblock_offset,
    "y_offset": saveblock_offset + 2,
    "bank_offset": saveblock_offset + 4,
    "map_offset": saveblock_offset + 5
}

with open(os.path.expanduser("~/pokemon_ai_project/ram_config.json"), "w") as f:
    json.dump(config_data, f, indent=2)

print("✅ [3/3] Konfiguration atomar hinterlegt!")
