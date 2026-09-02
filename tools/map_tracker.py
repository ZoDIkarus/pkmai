import json
import os
import cv2
import numpy as np

HISTORY_FILE = "path_history.json"
MAP_OUTPUT = "world_map_progress.png"
BG_IMAGE = "kanto_map.png"

# Offset-Tabelle: (Bank, Map_ID) -> (Global_Tile_X, Global_Tile_Y)
MAP_OFFSETS = {
    # Outdoor Maps
    (3, 0): (140, 260),   # Alabastia
    (3, 19): (140, 210),  # Route 1
    (3, 1): (140, 160),   # Vertania City
    (3, 20): (140, 110),  # Route 2
    (3, 2): (140, 60),    # Marmoria City
    
    # Indoor (Projektion auf Alabastia)
    (4, 0): (145, 266),   # Spieler-Zimmer 1. OG
    (4, 1): (145, 268),   # Haus EG
    (4, 2): (156, 270),   # Eichs Labor
}

def get_global_coords(bank, map_id, local_x, local_y):
    offset_x, offset_y = MAP_OFFSETS.get((bank, map_id), (140, 260))
    tile_x = offset_x + local_x
    tile_y = offset_y + local_y
    pixel_x = int(tile_x * 16)
    pixel_y = int(tile_y * 16)
    return pixel_x, pixel_y

def save_run_path(path_coordinates):
    if not path_coordinates:
        return
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append(path_coordinates)
    history = history[-20:]
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)
        
    render_global_map(history)

def render_global_map(history):
    if os.path.exists(BG_IMAGE):
        canvas = cv2.imread(BG_IMAGE)
    else:
        canvas = np.zeros((4000, 4000, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (255, 128, 0), (0, 165, 255), (128, 0, 255)
    ]

    for run_idx, run in enumerate(history):
        color = colors[run_idx % len(colors)]
        pts = []
        for step in run:
            bank, mid, x, y = step
            px, py = get_global_coords(bank, mid, x, y)
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                pts.append((px, py))
                
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 4, cv2.LINE_AA)
            
        if pts:
            cv2.circle(canvas, pts[-1], 8, (0, 255, 0), -1)

    resized_preview = cv2.resize(canvas, (1000, 1000), interpolation=cv2.INTER_AREA)
    cv2.imwrite(MAP_OUTPUT, resized_preview)

if __name__ == '__main__':
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            h = json.load(f)
        render_global_map(h)
        print("--> Weltkarte aus Historie neu gerendert!")
    else:
        render_global_map([])
        print("--> Initiales Kartenbild erzeugt.")
