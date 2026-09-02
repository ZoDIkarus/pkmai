import cv2
import numpy as np
import os

MAP_IMG = os.path.expanduser("~/pokemon_ai_project/kanto_map.png")
canvas = np.zeros((3000, 3000, 3), dtype=np.uint8)
canvas[:] = (22, 26, 36)

# Raster
for i in range(0, 3000, 100):
    cv2.line(canvas, (i, 0), (i, 3000), (32, 38, 52), 1)
    cv2.line(canvas, (0, i), (3000, i), (32, 38, 52), 1)

# Wege / Routen-Zonen andeuten
cv2.rectangle(canvas, (1350, 2200), (1650, 2500), (28, 34, 46), -1) # Pallet
cv2.rectangle(canvas, (1420, 1600), (1580, 2200), (30, 36, 50), -1) # Route 1
cv2.rectangle(canvas, (1300, 1150), (1700, 1600), (28, 34, 46), -1) # Viridian
cv2.rectangle(canvas, (1420, 750), (1580, 1150), (30, 36, 50), -1)  # Route 2 / Forest
cv2.rectangle(canvas, (1250, 350), (1750, 750), (28, 34, 46), -1)   # Pewter

# Beschriftungen
cv2.putText(canvas, "ALABASTIA (PALLET TOWN)", (1250, 2380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
cv2.putText(canvas, "ROUTE 1", (1440, 1900), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (41, 121, 255), 2)
cv2.putText(canvas, "VERTANIA CITY (VIRIDIAN CITY)", (1200, 1380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
cv2.putText(canvas, "ROUTE 2 & VERTANIA WALD", (1260, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (41, 121, 255), 2)
cv2.putText(canvas, "MARMORIA CITY (PEWTER CITY)", (1200, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)

cv2.imwrite(MAP_IMG, canvas)
print("🗺️ Kanto-Karten-Canvas (3000x3000) erfolgreich erstellt!")
