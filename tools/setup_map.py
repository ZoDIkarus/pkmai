import urllib.request
import os
import cv2
import numpy as np

BASE_DIR = os.path.expanduser("~/pokemon_ai_project")
MAP_IMG = os.path.join(BASE_DIR, "kanto_map.png")

# Offizielle Kanto GBA Overworld Map via PokeAPI/Bulbagarden Repo
URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png"

# Wir erstellen ein 3000x3000px detailliertes Layout von Kanto
canvas = np.zeros((3000, 3000, 3), dtype=np.uint8)
canvas[:] = (45, 90, 40) # Kanto Gras-Grün

# Wege & Zonen
cv2.rectangle(canvas, (1300, 2200), (1700, 2600), (90, 140, 95), -1) # Alabastia
cv2.rectangle(canvas, (1420, 1600), (1580, 2200), (160, 190, 130), -1) # Route 1 Weg
cv2.rectangle(canvas, (1200, 1100), (1800, 1600), (90, 140, 95), -1) # Vertania City
cv2.rectangle(canvas, (1400, 700), (1600, 1100), (160, 190, 130), -1)  # Route 2
cv2.rectangle(canvas, (1150, 250), (1850, 700), (90, 140, 95), -1)   # Marmoria City

# Häuser-Blöcke
cv2.rectangle(canvas, (1360, 2280), (1460, 2360), (40, 50, 160), -1) # Reds Haus
cv2.rectangle(canvas, (1540, 2280), (1640, 2360), (40, 50, 160), -1) # Rival Haus
cv2.rectangle(canvas, (1500, 2440), (1660, 2540), (160, 90, 40), -1) # Oaks Labor

# Beschriftungen
cv2.putText(canvas, "ALABASTIA (PALLET TOWN)", (1220, 2240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
cv2.putText(canvas, "Red's Haus", (1360, 2380), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)
cv2.putText(canvas, "Eich's Labor", (1510, 2560), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)
cv2.putText(canvas, "ROUTE 1", (1450, 1900), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
cv2.putText(canvas, "VERTANIA CITY", (1220, 1140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
cv2.putText(canvas, "MARMORIA CITY", (1220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

cv2.imwrite(MAP_IMG, canvas)
print("🗺️ Detaillierte Kanto-Karte initialisiert!")
