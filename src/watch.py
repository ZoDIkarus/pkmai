import stable_retro as retro
import numpy as np
import cv2
import os
import json
import time
import glob

from stable_baselines3 import PPO
from firered_ram import read_player_location, read_player_party


# ================================================================
# USER CONFIG / WATCHER TUNING
# ================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
BASE_DIR = PROJECT_ROOT

# Geschwindigkeit / Anzeige
TARGET_FPS = 300.0
GUI_EVERY_FRAMES = 6
TELEMETRY_INTERVAL = 0.5
FPS_TITLE_INTERVAL = 0.5

# Action-Block: Taste halten + neutrale Frames
ACTION_HOLD_FRAMES = 5
ACTION_RELEASE_FRAMES = 3

# Reload / RAM
MODEL_CHECK_INTERVAL = 1.0
RAM_DISCOVERY_INTERVAL = 0.75
WATCHER_DEVICE = "cpu"

# Live Map Tiling (nur RAM-Koordinaten; keine Screenshots)
MAPPING_GRID = True
MAPPING_EDGE_REWARD = 0.35
MAPPING_MAP_REWARD = 20.0
MAPPING_TRANSITION_REWARD = 30.0
WATCHER_MAPPING_FILE = os.path.join(RUNTIME_DIR, "watcher_mapping.json")

# Fenster
GAME_PANEL_W = 720
GAME_PANEL_H = 480
MAP_PANEL_W = 620
TOP_H = 84
BOTTOM_H = 74

# ================================================================
# INTERNAL PATHS - normalerweise nicht aendern
# ================================================================
MODEL_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
LATEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_latest.zip")
VERSION_FILE = os.path.join(RUNTIME_DIR, "model_version.json")
CUSTOM_DIR = os.path.join(LOCAL_DIR, "custom_integrations")
INSTANCES_DIR = os.path.join(RUNTIME_DIR, "instances_data")

os.makedirs(INSTANCES_DIR, exist_ok=True)
WATCHER_BATTLE_FILE = os.path.join(RUNTIME_DIR, "watcher_battle_stats.json")

def load_watcher_battle_stats():
    try:
        with open(WATCHER_BATTLE_FILE, "r") as f:
            d=json.load(f)
        return {"started":int(d.get("started",0)), "completed":int(d.get("completed",0))}
    except Exception:
        return {"started":0,"completed":0}

def save_watcher_battle_stats(d):
    try:
        tmp=WATCHER_BATTLE_FILE+".tmp"
        with open(tmp,"w") as f: json.dump(d,f,separators=(",",":"))
        os.replace(tmp,WATCHER_BATTLE_FILE)
    except Exception:
        pass



def get_latest_version():
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                return int(json.load(f).get("version", 0))
        except Exception:
            return 0
    return 0


def process_image(screen):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    )
    return np.expand_dims(resized, axis=0).astype(np.uint8)


def load_global_navigation_memory():
    edges = set()
    maps = set()
    transitions = set()

    exploration_dir = os.path.join(RUNTIME_DIR, "exploration_memory")
    for path in glob.glob(os.path.join(exploration_dir, "agent_*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            edges.update(
                tuple(x) for x in data.get("edges", [])
                if isinstance(x, list) and len(x) == 6
            )
            maps.update(
                tuple(x) for x in data.get("maps", [])
                if isinstance(x, list) and len(x) == 2
            )
            transitions.update(
                tuple(x) for x in data.get("transitions", [])
                if isinstance(x, list) and len(x) == 8
            )
        except Exception:
            pass

    return edges, maps, transitions


def watcher_nav_target(
    bank, map_id, x, y,
    known_edges, known_maps, known_transitions
):
    # Indoor: bekannte Tuer/Treppe bevorzugen.
    candidates = []
    for t in known_transitions:
        if len(t) != 8:
            continue
        a = tuple(map(int, t[:4]))
        b = tuple(map(int, t[4:]))
        for here, other in ((a, b), (b, a)):
            if (here[0], here[1]) != (bank, map_id):
                continue
            if bank != 3:
                candidates.append((here[2], here[3]))
            elif (other[0], other[1]) not in known_maps:
                candidates.append((here[2], here[3]))

    if candidates:
        return min(
            candidates,
            key=lambda p: abs(p[0] - x) + abs(p[1] - y)
        )

    # Generischer Frontier aus selbst entdeckten Kanten.
    nodes = set()
    adjacency = {}
    for e in known_edges:
        if len(e) != 6:
            continue
        eb, em, x1, y1, x2, y2 = map(int, e)
        if (eb, em) != (bank, map_id):
            continue
        a = (x1, y1)
        b = (x2, y2)
        nodes.add(a)
        nodes.add(b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    frontiers = [
        p for p in nodes
        if len(adjacency.get(p, ())) < 4
    ]
    if not frontiers:
        return None

    return min(
        frontiers,
        key=lambda p: abs(p[0] - x) + abs(p[1] - y)
    )


def build_v7_obs(
    screen,
    loc,
    info,
    known_edges,
    known_maps,
    known_transitions,
):
    valid = bool(
        loc.get("valid", False)
        and loc.get("trusted", False)
    )
    bank = int(loc.get("map_bank", 0)) if valid else 0
    map_id = int(loc.get("map_id", 0)) if valid else 0
    x = int(loc.get("x_pos", 0)) if valid else 0
    y = int(loc.get("y_pos", 0)) if valid else 0

    # Watcher fuehrt die Policy als Full-Chain-Agent aus.
    vec = [0.0, 0.0, 0.0, 0.0, 1.0]

    in_battle = int(info.get("in_battle", 0))
    p_lvl = int(info.get("p1_level", 0))
    badges_raw = int(info.get("badges", 0))
    badges = (
        bin(badges_raw).count("1")
        if badges_raw > 8 else badges_raw
    )

    gameplay_ready = bool(valid and 0 <= x < 512 and 0 <= y < 512)
    has_starter = p_lvl >= 5

    # Watcher kennt Storyflags nur konservativ aus aktuellem Ort.
    stairs_done = 1.0 if gameplay_ready and bank != 4 else 0.0
    house_left = 1.0 if gameplay_ready and bank == 3 else 0.0

    vec.extend([
        1.0 if gameplay_ready else 0.0,
        1.0 if in_battle else 0.0,
        stairs_done,
        house_left,
        1.0 if has_starter else 0.0,
    ])

    if gameplay_ready:
        vec.extend([
            float(np.clip(bank / 31.0, 0.0, 1.0)),
            float(np.clip(map_id / 255.0, 0.0, 1.0)),
            float(np.clip(x / 511.0, 0.0, 1.0)),
            float(np.clip(y / 511.0, 0.0, 1.0)),
        ])
    else:
        vec.extend([0.0, 0.0, 0.0, 0.0])

    target = (
        watcher_nav_target(
            bank, map_id, x, y,
            known_edges, known_maps, known_transitions
        )
        if gameplay_ready else None
    )

    if target is None:
        vec.extend([0.0, 0.0, 0.0, 0.0])
    else:
        dx = int(target[0]) - x
        dy = int(target[1]) - y
        dist = abs(dx) + abs(dy)
        vec.extend([
            1.0,
            float(np.clip(dx / 32.0, -1.0, 1.0)),
            float(np.clip(dy / 32.0, -1.0, 1.0)),
            float(np.clip(dist / 64.0, 0.0, 1.0)),
        ])

    vec.extend([
        float(np.clip(p_lvl / 100.0, 0.0, 1.0)),
        float(np.clip(badges / 8.0, 0.0, 1.0)),
    ])

    nav = np.asarray(vec, dtype=np.float32)
    if nav.shape != (20,):
        raise RuntimeError(f"Watcher V7 nav shape invalid: {nav.shape}")

    return {
        "image": process_image(screen),
        "nav": nav,
    }



def _edge_key(bank, map_id, x1, y1, x2, y2):
    a = (int(x1), int(y1))
    b = (int(x2), int(y2))
    if b < a:
        a, b = b, a
    return (int(bank), int(map_id), a[0], a[1], b[0], b[1])


def _transition_key(
    from_bank, from_map, from_x, from_y,
    to_bank, to_map, to_x, to_y
):
    a = (
        int(from_bank), int(from_map),
        int(from_x), int(from_y)
    )
    b = (
        int(to_bank), int(to_map),
        int(to_x), int(to_y)
    )
    if b < a:
        a, b = b, a
    return a + b


def load_watcher_mapping():
    known_tiles = set()
    known_edges = set()
    known_maps = set()
    known_transitions = set()

    if not os.path.exists(WATCHER_MAPPING_FILE):
        return known_tiles, known_edges, known_maps, known_transitions

    try:
        with open(WATCHER_MAPPING_FILE, "r") as f:
            data = json.load(f)

        known_tiles = {
            tuple(x) for x in data.get("tiles", [])
            if isinstance(x, list) and len(x) == 4
        }
        known_edges = {
            tuple(x) for x in data.get("edges", [])
            if isinstance(x, list) and len(x) == 6
        }
        known_maps = {
            tuple(x) for x in data.get("maps", [])
            if isinstance(x, list) and len(x) == 2
        }
        known_transitions = {
            tuple(x) for x in data.get("transitions", [])
            if isinstance(x, list) and len(x) == 8
        }
    except Exception:
        pass

    return known_tiles, known_edges, known_maps, known_transitions


def save_watcher_mapping(
    known_tiles, known_edges, known_maps, known_transitions
):
    tmp = WATCHER_MAPPING_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(
                {
                    "schema": 1,
                    "tiles": [list(x) for x in known_tiles],
                    "edges": [list(x) for x in known_edges],
                    "maps": [list(x) for x in known_maps],
                    "transitions": [
                        list(x) for x in known_transitions
                    ],
                },
                f,
                separators=(",", ":")
            )
        os.replace(tmp, WATCHER_MAPPING_FILE)
    except Exception:
        pass


def build_mapping_preview(
    known_tiles, known_edges, known_transitions,
    bank, map_id, x, y, width, height
):
    """Persistente Live-Map aus eindeutig bekannten RAM-Tiles/Kanten."""
    view = np.zeros((height, width, 3), dtype=np.uint8)
    view[:] = (12, 15, 22)

    map_tiles = [
        (tx, ty)
        for tb, tm, tx, ty in known_tiles
        if tb == bank and tm == map_id
    ]

    if 0 <= x < 512 and 0 <= y < 512:
        map_tiles.append((x, y))

    if not map_tiles:
        cv2.putText(
            view, "Live Map Tiling wartet auf Bewegung ...", (22, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (190, 195, 210), 1, cv2.LINE_AA
        )
        return view, {
            "tiles": 0,
            "edges": 0,
            "transitions": len(known_transitions),
            "width_tiles": 0,
            "height_tiles": 0,
        }

    xs = [p[0] for p in map_tiles]
    ys = [p[1] for p in map_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad_tiles = 2
    min_x -= pad_tiles
    min_y -= pad_tiles
    max_x += pad_tiles
    max_y += pad_tiles

    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)
    margin = 34
    sx = (width - margin * 2) / span_x
    sy = (height - margin * 2) / span_y
    scale = max(3.0, min(28.0, min(sx, sy)))

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    def screen_pos(px, py):
        vx = int(width / 2 + (px - cx) * scale)
        vy = int(height / 2 + (py - cy) * scale)
        return vx, vy

    if MAPPING_GRID and scale >= 8:
        grid_color = (31, 36, 49)
        for gx in range(int(min_x), int(max_x) + 2):
            cv2.line(
                view,
                screen_pos(gx, min_y),
                screen_pos(gx, max_y + 1),
                grid_color, 1
            )
        for gy in range(int(min_y), int(max_y) + 2):
            cv2.line(
                view,
                screen_pos(min_x, gy),
                screen_pos(max_x + 1, gy),
                grid_color, 1
            )

    # Bereits erkundete Tiles als grüne Felder.
    tile_size = max(3, int(scale * 0.72))
    half = max(1, tile_size // 2)
    for tx, ty in set(map_tiles):
        px, py = screen_pos(tx, ty)
        cv2.rectangle(
            view,
            (px - half, py - half),
            (px + half, py + half),
            (35, 105, 45),
            -1
        )

    # Jede bekannte Kante wird EXAKT EINMAL gezeichnet.
    map_edges = 0
    for e in known_edges:
        if len(e) != 6:
            continue
        eb, em, x1, y1, x2, y2 = e
        if eb != bank or em != map_id:
            continue
        map_edges += 1
        cv2.line(
            view,
            screen_pos(x1, y1),
            screen_pos(x2, y2),
            (255, 180, 40),
            1,
            cv2.LINE_AA
        )

    player = screen_pos(x, y)
    cv2.circle(view, player, 8, (0, 230, 118), -1, cv2.LINE_AA)
    cv2.circle(view, player, 11, (230, 255, 240), 1, cv2.LINE_AA)

    cv2.putText(
        view, f"Bank {bank} / Map {map_id}", (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )

    return view, {
        "tiles": len({
            (tx, ty)
            for tb, tm, tx, ty in known_tiles
            if tb == bank and tm == map_id
        }),
        "edges": map_edges,
        "transitions": len(known_transitions),
        "width_tiles": max(1, max(xs) - min(xs) + 1),
        "height_tiles": max(1, max(ys) - min(ys) + 1),
    }

def main():
    retro.data.Integrations.add_custom_path(CUSTOM_DIR)

    env = retro.make(
        game="PokemonFireRed-Gba",
        state=retro.State.NONE,
        inttype=retro.data.Integrations.CUSTOM_ONLY
    )


    btn_list = list(env.buttons)
    num_buttons = len(btn_list)

    def get_btn_mask(name):
        mask = [0] * num_buttons
        if name in btn_list:
            mask[btn_list.index(name)] = 1
        return mask

    no_action = [0] * num_buttons

    action_map = [
        get_btn_mask("A"),
        get_btn_mask("B"),
        get_btn_mask("START"),
        get_btn_mask("UP"),
        get_btn_mask("DOWN"),
        get_btn_mask("LEFT"),
        get_btn_mask("RIGHT"),
    ]
    action_names = ["A", "B", "START", "UP", "DOWN", "LEFT", "RIGHT"]
    current_action_name = "NONE"
    action_history = []
    watcher_gameplay_ready = False
    watcher_in_battle = 0
    watcher_loc = {
        "valid": False,
        "trusted": False,
        "map_bank": 0,
        "map_id": 0,
        "x_pos": 0,
        "y_pos": 0,
    }
    watcher_info = {}
    watcher_start_spam_count = 0
    watcher_last_start_decision = -999999

    # Links: FireRed standardmaessig 3x hochskaliert.
    CONTENT_H = GAME_PANEL_H

    CANVAS_W = GAME_PANEL_W + MAP_PANEL_W
    CANVAS_H = TOP_H + CONTENT_H + BOTTOM_H

    WINDOW = "Pokemon Firered AI by Alex - Watcher + Live Map"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)
    # Auf macOS sicher sichtbar platzieren.
    try:
        cv2.moveWindow(WINDOW, 40, 40)
    except Exception:
        pass

    model = None
    loaded_version = -1

    env.reset()

    FRAME_TIME = 1.0 / TARGET_FPS

    current_action = no_action
    frame_counter = 0
    total_steps = 0

    bank = 0
    map_id = 0
    x = 0
    y = 0
    in_battle = 0
    watcher_battle_stats = load_watcher_battle_stats()
    p1_level = 0
    watcher_party = []
    last_party_read_step = -999999
    previous_party_exp = {}
    watcher_exp_gained = 0
    watcher_last_exp_gain = 0
    badge_count = 0

    last_coord = None
    recent_path = []

    (
        watcher_known_tiles,
        watcher_known_edges,
        watcher_known_maps,
        watcher_known_transitions,
    ) = load_watcher_mapping()

    (
        global_nav_edges,
        global_nav_maps,
        global_nav_transitions,
    ) = load_global_navigation_memory()
    last_global_nav_reload = 0.0

    watcher_mapping_last_coord = None
    watcher_mapping_reward = 0.0
    watcher_mapping_event = "-"

    # Screenshot-Tile-Mapping ist deaktiviert. Der Watcher zeichnet nur noch
    # die robuste RAM-Skeleton-Route.
    map_new_tiles = 0
    total_new_tiles = 0

    # Live-FPS fuer den Fenstertitel.
    fps_value = 0.0
    fps_frames = 0
    fps_window_start = time.perf_counter()
    fps_title_interval = FPS_TITLE_INTERVAL

    # Tuning-Werte stehen oben im USER CONFIG Block.
    last_ram_discovery_time = 0.0
    last_telemetry_time = 0.0
    last_model_check_time = 0.0

    # Mapping erst freigeben, wenn wir echte Overworld-Bewegung sehen.
    # Dadurch werden Intro, Namenseingabe und sonstige Menues nicht als Map gebaut.
    mapping_unlocked = False
    stable_loc_frames = 0
    gate_last_loc = None
    gate_last_map = None

    # Performance caches: expensive RAM/map work must not run every frame.
    loc = {
        "valid": False,
        "source": "init",
        "map_bank": 0,
        "map_id": 0,
        "x_pos": 0,
        "y_pos": 0,
    }

    # Watcher-side live reward mirror. This does not train the model; it only
    # visualises the same style of exploration/progress reward live.
    watcher_step_reward = 0.0
    watcher_episode_reward = 0.0
    watcher_seen_coords = set()
    watcher_visited_maps = set()
    watcher_last_level = 0
    watcher_last_badges = 0
    watcher_has_starter = False
    watcher_stuck_counter = 0
    watcher_last_progress_signature = None

    watcher_intro_seen_states = set()
    watcher_intro_last_thumb = None
    watcher_intro_same_screen_steps = 0
    watcher_intro_novelty_reward_total = 0.0
    watcher_intro_complete_rewarded = False

    watcher_left_house_rewarded = False
    watcher_north_grass_rewarded = False
    watcher_next_outdoor_map_rewarded = False
    watcher_first_outdoor_map = None
    watcher_outdoor_entry_y = None
    watcher_previous_valid_bank = None
    watcher_previous_valid_map = None

    last_raw_screen = env.get_screen()

    while True:
        frame_start = time.perf_counter()

        # Eine Agentenentscheidung alle 8 Emulatorframes.
        action_cycle_frames = ACTION_HOLD_FRAMES + ACTION_RELEASE_FRAMES
        if frame_counter % action_cycle_frames == 0:
            total_steps += 1

            now_model = time.perf_counter()
            if now_model - last_model_check_time >= MODEL_CHECK_INTERVAL:
                last_model_check_time = now_model
                current_version = get_latest_version()

                if (
                    current_version != loaded_version and
                    os.path.exists(LATEST_MODEL)
                ):
                    try:
                        model = PPO.load(LATEST_MODEL, device=WATCHER_DEVICE)
                        loaded_version = current_version
                        print(
                            f"Watcher loaded brain: "
                            f"PKMAI v{loaded_version:06d}"
                        )
                    except Exception as e:
                        print("Model load failed:", e)

            now_nav = time.time()
            if now_nav - last_global_nav_reload >= 2.0:
                (
                    global_nav_edges,
                    global_nav_maps,
                    global_nav_transitions,
                ) = load_global_navigation_memory()
                # Watcher-eigene Funde sofort mitverwenden.
                global_nav_edges |= watcher_known_edges
                global_nav_maps |= watcher_known_maps
                global_nav_transitions |= watcher_known_transitions
                last_global_nav_reload = now_nav

            obs = build_v7_obs(
                last_raw_screen,
                watcher_loc,
                watcher_info,
                global_nav_edges,
                global_nav_maps,
                global_nav_transitions,
            )

            if model is not None:
                action, _ = model.predict(
                    obs,
                    deterministic=False
                )
                action_idx = int(action)
                current_action_name = action_names[action_idx]
                current_action = action_map[action_idx]
                action_history.append(current_action_name)
                action_history = action_history[-10:]

                if action_idx == 2:
                    if total_steps - watcher_last_start_decision <= 6:
                        watcher_start_spam_count += 1
                    else:
                        watcher_start_spam_count = 1
                    watcher_last_start_decision = total_steps
                elif total_steps - watcher_last_start_decision > 6:
                    watcher_start_spam_count = 0
            else:
                action_idx = int(np.random.randint(0, len(action_map)))
                current_action = action_map[action_idx]
                current_action_name = action_names[action_idx]
                action_history.append(current_action_name)
                action_history = action_history[-10:]

        # 5 aktive + 3 Ruheframes.
        # Am letzten Ruheframe ist das Bild fuer Mapping stabiler.
        phase = frame_counter % action_cycle_frames
        step_action = current_action if phase < 5 else no_action

        step_res = env.step(step_action)

        info = (
            step_res[4]
            if len(step_res) == 5
            else step_res[3]
        )
        watcher_info = info if isinstance(info, dict) else {}

        if not isinstance(info, dict):
            info = {}

        raw_screen = step_res[0]
        last_raw_screen = raw_screen

        # Dynamic FireRed RAM:
        # Solange noch kein Pointer gefunden ist, nicht bei jedem 8-Frame-Block
        # den kompletten RAM scannen. Bekannte Slots bleiben billig aktiv.
        if phase == 0:
            now_ram = time.perf_counter()
            allow_full_scan = False

            if not loc.get("valid", False):
                if (
                    now_ram - last_ram_discovery_time
                    >= RAM_DISCOVERY_INTERVAL
                ):
                    allow_full_scan = True
                    last_ram_discovery_time = now_ram

            loc = read_player_location(
                env,
                allow_scan=allow_full_scan
            )

            if loc.get("valid", False):
                bank = int(loc["map_bank"])
                map_id = int(loc["map_id"])
                x = int(loc["x_pos"])
                y = int(loc["y_pos"])
            else:
                bank = 0
                map_id = 0
                x = 0
                y = 0

        previous_battle_state = int(in_battle)
        in_battle = int(info.get("in_battle", in_battle))
        if previous_battle_state == 0 and in_battle == 1:
            watcher_battle_stats["started"] += 1
            save_watcher_battle_stats(watcher_battle_stats)
        elif previous_battle_state == 1 and in_battle == 0:
            watcher_battle_stats["completed"] += 1
            save_watcher_battle_stats(watcher_battle_stats)

        # -------------------------------------------------------------
        # GAMEPLAY GATE
        # -------------------------------------------------------------
        # Ein RAM-Fallback kann waehrend des Intros zufaellig plausible Werte
        # finden. Deshalb erst Mapping erlauben, wenn:
        # 1) dieselbe Bank/Map mehrere Frames stabil bleibt UND
        # 2) wir eine echte Nachbarbewegung von genau 1 Tile sehen.
        trusted_loc = bool(
            loc.get("valid", False) and
            loc.get("trusted", False)
        )

        if phase == 0 and trusted_loc:
            current_map_key = (bank, map_id)

            if current_map_key == gate_last_map:
                stable_loc_frames += 1
            else:
                stable_loc_frames = 1
                gate_last_map = current_map_key
                gate_last_loc = None

            if gate_last_loc is not None:
                old_bank, old_map, old_x, old_y = gate_last_loc

                same_map = (
                    old_bank == bank and
                    old_map == map_id
                )

                tile_move = abs(x - old_x) + abs(y - old_y)

                if (
                    same_map and
                    1 <= tile_move <= 3 and
                    stable_loc_frames >= 3 and
                    in_battle == 0
                ):
                    mapping_unlocked = True

            # A trusted SaveBlock pointer plus a stable nonzero room/map is
            # already sufficient to begin drawing. This lets the player's
            # upstairs bedroom appear even before much movement happens.
            if (
                stable_loc_frames >= 4 and
                (bank, map_id, x, y) != (0, 0, 0, 0) and
                in_battle == 0
            ):
                mapping_unlocked = True

            gate_last_loc = (bank, map_id, x, y)

        p1_level = int(info.get("p1_level", p1_level))
        if total_steps - last_party_read_step >= 32:
            try:
                decoded_party = read_player_party(env)
                watcher_last_exp_gain = 0

                if decoded_party:
                    for mon in decoded_party:
                        slot = int(mon.get("slot", -1))
                        exp_now = int(mon.get("experience", 0))
                        exp_before = previous_party_exp.get(slot)

                        exp_delta = 0
                        if (
                            exp_before is not None
                            and exp_now >= exp_before
                        ):
                            exp_delta = exp_now - exp_before

                        mon["exp_delta"] = int(exp_delta)

                        if exp_delta > 0:
                            watcher_exp_gained += int(exp_delta)
                            watcher_last_exp_gain += int(exp_delta)

                        previous_party_exp[slot] = exp_now

                    watcher_party = decoded_party

                elif p1_level <= 0:
                    watcher_party = []
                    previous_party_exp = {}

            except Exception:
                pass

            last_party_read_step = total_steps

        badges_raw = int(info.get("badges", 0))
        badge_count = (
            bin(badges_raw).count("1")
            if badges_raw > 0
            else 0
        )

        coord = (bank, map_id, x, y)

        if (
            coord != last_coord and
            coord != (0, 0, 0, 0)
        ):
            recent_path.append(
                [bank, map_id, x, y]
            )
            recent_path = recent_path[-300:]
            last_coord = coord

        # -------------------------------------------------------------
        # LIVE MAP TILING / MAPPING EVENT
        # -------------------------------------------------------------
        watcher_mapping_reward = 0.0
        watcher_mapping_event = "-"

        if (
            trusted_loc
            and coord != (0, 0, 0, 0)
            and in_battle == 0
        ):
            tile_key = (bank, map_id, x, y)
            map_key = (bank, map_id)

            if tile_key not in watcher_known_tiles:
                watcher_known_tiles.add(tile_key)

            if map_key not in watcher_known_maps:
                watcher_known_maps.add(map_key)
                watcher_mapping_reward += MAPPING_MAP_REWARD
                watcher_mapping_event = (
                    f"NEW MAP +{MAPPING_MAP_REWARD:.2f}"
                )

            if watcher_mapping_last_coord is not None:
                pb, pm, px, py = watcher_mapping_last_coord

                if (pb, pm) == (bank, map_id):
                    if abs(x - px) + abs(y - py) == 1:
                        edge_key = _edge_key(
                            bank, map_id, px, py, x, y
                        )
                        if edge_key not in watcher_known_edges:
                            watcher_known_edges.add(edge_key)
                            watcher_mapping_reward += MAPPING_EDGE_REWARD
                            watcher_mapping_event = (
                                f"NEW EDGE +{MAPPING_EDGE_REWARD:.2f}"
                            )
                else:
                    transition_key = _transition_key(
                        pb, pm, px, py,
                        bank, map_id, x, y
                    )
                    if (
                        transition_key
                        not in watcher_known_transitions
                    ):
                        watcher_known_transitions.add(
                            transition_key
                        )
                        watcher_mapping_reward += (
                            MAPPING_TRANSITION_REWARD
                        )
                        watcher_mapping_event = (
                            "NEW TRANSITION "
                            f"+{MAPPING_TRANSITION_REWARD:.2f}"
                        )

            watcher_mapping_last_coord = coord

            if watcher_mapping_reward > 0.0:
                save_watcher_mapping(
                    watcher_known_tiles,
                    watcher_known_edges,
                    watcher_known_maps,
                    watcher_known_transitions,
                )

        # -------------------------------------------------------------
        # LIVE REWARD (Watcher-Anzeige)
        # Ein Reward-Schritt entspricht einem kompletten 8-Frame-Aktionsblock.
        # -------------------------------------------------------------
        if phase == 7:
            gameplay_ready = bool(
                trusted_loc and coord != (0, 0, 0, 0)
            )
            watcher_gameplay_ready = gameplay_ready
            watcher_in_battle = in_battle
            watcher_step_reward = 0.01 if gameplay_ready else 0.0

            # Intro-/Namensvergabe-Rewards nur fuer die Live-Anzeige.
            if not gameplay_ready:
                gray_intro = cv2.cvtColor(
                    raw_screen,
                    cv2.COLOR_RGB2GRAY
                )
                thumb_intro = cv2.resize(
                    gray_intro,
                    (12, 8),
                    interpolation=cv2.INTER_AREA
                ).astype(np.int16)

                if watcher_intro_last_thumb is None:
                    watcher_intro_last_thumb = thumb_intro
                    q = (thumb_intro // 32).astype(np.uint8)
                    watcher_intro_seen_states.add(q.tobytes())
                else:
                    intro_diff = float(np.mean(np.abs(
                        thumb_intro - watcher_intro_last_thumb
                    )))
                    if intro_diff < 4.0:
                        watcher_intro_same_screen_steps += 1
                    else:
                        watcher_intro_same_screen_steps = 0

                    q = (thumb_intro // 32).astype(np.uint8)
                    intro_key = q.tobytes()
                    if (
                        intro_diff >= 10.0
                        and intro_key not in watcher_intro_seen_states
                        and watcher_intro_novelty_reward_total < 25.0
                    ):
                        intro_bonus = 1.0 if intro_diff >= 28.0 else 0.5
                        intro_bonus = min(
                            intro_bonus,
                            25.0 - watcher_intro_novelty_reward_total
                        )
                        watcher_step_reward += intro_bonus
                        watcher_intro_novelty_reward_total += intro_bonus
                        watcher_intro_seen_states.add(intro_key)

                    watcher_intro_last_thumb = thumb_intro

                    if watcher_intro_same_screen_steps >= 120:
                        watcher_step_reward -= 0.01
                    if watcher_intro_same_screen_steps >= 300:
                        watcher_step_reward -= 0.03

            elif not watcher_intro_complete_rewarded:
                watcher_intro_complete_rewarded = True
                watcher_step_reward += 35.0

            if gameplay_ready and in_battle == 0:
                # Haus verlassen: Indoor -> FireRed Overworld Bank 3.
                if (
                    not watcher_left_house_rewarded
                    and bank == 3
                    and watcher_previous_valid_bank is not None
                    and watcher_previous_valid_bank != 3
                ):
                    watcher_left_house_rewarded = True
                    watcher_first_outdoor_map = map_id
                    watcher_outdoor_entry_y = y
                    watcher_step_reward += 52.0

                if bank == 3 and watcher_first_outdoor_map is None:
                    watcher_first_outdoor_map = map_id
                    watcher_outdoor_entry_y = y

                # Richtung Gras / deutlich nach Norden.
                if (
                    watcher_left_house_rewarded
                    and not watcher_north_grass_rewarded
                    and bank == 3
                    and map_id == watcher_first_outdoor_map
                    and watcher_outdoor_entry_y is not None
                    and y <= watcher_outdoor_entry_y - 5
                ):
                    watcher_north_grass_rewarded = True
                    watcher_step_reward += 75.0

                # Erste neue Aussenwelt-Map.
                if (
                    watcher_first_outdoor_map is not None
                    and not watcher_next_outdoor_map_rewarded
                    and bank == 3
                    and map_id != watcher_first_outdoor_map
                ):
                    watcher_next_outdoor_map_rewarded = True
                    watcher_step_reward += 150.0

                map_key = (bank, map_id)
                if map_key not in watcher_visited_maps:
                    watcher_visited_maps.add(map_key)
                    watcher_step_reward += 10.0

                if coord not in watcher_seen_coords:
                    watcher_seen_coords.add(coord)
                    watcher_step_reward += 0.25

            # Erstes Pokemon exakt +100, danach normale Levelups.
            if not watcher_has_starter and p1_level >= 5:
                watcher_has_starter = True
                watcher_step_reward += 100.0
                watcher_last_level = p1_level
            elif watcher_has_starter and p1_level > watcher_last_level:
                watcher_step_reward += (
                    p1_level - watcher_last_level
                ) * 25.0
                watcher_last_level = p1_level
            elif watcher_last_level == 0 and p1_level > 0:
                watcher_last_level = p1_level

            if badge_count > watcher_last_badges:
                watcher_step_reward += (
                    badge_count - watcher_last_badges
                ) * 500.0
                watcher_last_badges = badge_count

            if gameplay_ready:
                watcher_progress_signature = (
                    bank,
                    map_id,
                    x,
                    y,
                    p1_level,
                    badge_count,
                    in_battle
                )

                if watcher_progress_signature != watcher_last_progress_signature:
                    watcher_stuck_counter = 0
                    watcher_last_progress_signature = watcher_progress_signature
                else:
                    watcher_stuck_counter += 1

                if in_battle == 0 and watcher_stuck_counter >= 60:
                    watcher_step_reward -= 0.03
                if in_battle == 0 and watcher_stuck_counter >= 180:
                    watcher_step_reward -= 0.12
                if in_battle == 0 and watcher_stuck_counter >= 400:
                    watcher_step_reward -= 0.40

                watcher_previous_valid_bank = bank
                watcher_previous_valid_map = map_id
            else:
                watcher_stuck_counter = 0
                watcher_last_progress_signature = None

            watcher_episode_reward += watcher_step_reward

            # Watcher-Anti-Loop-Reset:
            # Neue Modellversionen uebernehmen sonst denselben festgefahrenen
            # Emulatorzustand (z.B. PC -> Itemfach -> raus -> rein).
            if gameplay_ready and in_battle == 0 and watcher_stuck_counter >= 900:
                print(
                    f"🔄 Watcher Anti-Loop Reset nach "
                    f"{watcher_stuck_counter} stagnierenden Aktionsschritten "
                    f"(Episode {watcher_episode_reward:.2f})"
                )

                env.reset()

                watcher_step_reward = 0.0
                watcher_episode_reward = 0.0
                watcher_seen_coords.clear()
                watcher_visited_maps.clear()
                watcher_last_level = 0
                watcher_last_badges = 0
                watcher_has_starter = False
                watcher_stuck_counter = 0

                watcher_intro_seen_states = set()
                watcher_intro_last_thumb = None
                watcher_intro_same_screen_steps = 0
                watcher_intro_novelty_reward_total = 0.0
                watcher_intro_complete_rewarded = False
                watcher_last_progress_signature = None

                watcher_left_house_rewarded = False
                watcher_north_grass_rewarded = False
                watcher_next_outdoor_map_rewarded = False
                watcher_first_outdoor_map = None
                watcher_outdoor_entry_y = None
                watcher_previous_valid_bank = None
                watcher_previous_valid_map = None

                mapping_unlocked = False
                stable_loc_frames = 0
                gate_last_loc = None
                recent_path.clear()

                loc = {
                    "valid": False,
                    "trusted": False,
                    "source": "watcher_reset",
                    "map_bank": 0,
                    "map_id": 0,
                    "x_pos": 0,
                    "y_pos": 0,
                }
                bank = map_id = x = y = 0
                coord = (0, 0, 0, 0)
                trusted_loc = False
                phase = 0
                current_action = no_action
                current_action_name = "NONE"
                action_history.clear()
                last_raw_screen = env.get_screen()
                continue

        # Screenshot-Tile-Mapper entfernt. Die Karte basiert nur auf
        # den bereits gelesenen RAM-Koordinaten in recent_path.
        map_new_tiles = 0
        map_work_ms = 0.0

        # Telemetrie fuer Webstream / andere Tools: 2x pro Sekunde.
        telemetry_now = time.perf_counter()
        if telemetry_now - last_telemetry_time >= TELEMETRY_INTERVAL:
            last_telemetry_time = telemetry_now
            inst_file = os.path.join(
                INSTANCES_DIR,
                "inst_99.json"
            )
            tmp_file = os.path.join(
                INSTANCES_DIR,
                "tmp_99.json"
            )

            try:
                data = {
                    "id": 99,
                    "name": "Alex (Watcher)",
                    "bank": bank,
                    "map": map_id,
                    "x": x,
                    "y": y,
                    "path": recent_path,
                    "room": f"Bank {bank} / Map {map_id}",
                    "steps": total_steps,
                    "reward": round(watcher_episode_reward, 2),
                    "step_reward": round(watcher_step_reward, 4),
                    "stuck_counter": watcher_stuck_counter,
                    "level": p1_level,
                    "party": watcher_party,
                    "exp_stats": {
                        "gained_total": int(watcher_exp_gained),
                        "last_gain": int(watcher_last_exp_gain),
                    },
                    "badges": badge_count,
                    "in_battle": in_battle,
                    "battle_stats": {
                        "started": int(watcher_battle_stats.get("started", 0)),
                        "completed": int(watcher_battle_stats.get("completed", 0)),
                    },
                    "new_visible_tiles": map_new_tiles,
                    "total_mapped_tiles": total_new_tiles,
                    "ram_source": loc.get("source", "unknown"),
                    "ram_trusted": trusted_loc,
                    "input": current_action_name,
                    "input_history": list(action_history)
                }

                with open(tmp_file, "w") as f:
                    json.dump(data, f)

                os.replace(tmp_file, inst_file)

            except Exception:
                pass

        # Emulator-loop FPS messen (nicht die GUI-Refresh-Rate).
        fps_frames += 1
        fps_now = time.perf_counter()
        fps_elapsed = fps_now - fps_window_start
        if fps_elapsed >= fps_title_interval:
            fps_value = fps_frames / fps_elapsed
            fps_frames = 0
            fps_window_start = fps_now

        # -------------------------------------------------------------
        # GUI
        # -------------------------------------------------------------
        if frame_counter % GUI_EVERY_FRAMES == 0:
            canvas = np.zeros(
                (CANVAS_H, CANVAS_W, 3),
                dtype=np.uint8
            )
            canvas[:] = (10, 13, 20)

            game_bgr = cv2.cvtColor(
                raw_screen,
                cv2.COLOR_RGB2BGR
            )

            # Exaktes Integer-Scaling: keine Verzerrung, keine weichen Pixel.
            game_view = cv2.resize(
                game_bgr,
                (GAME_PANEL_W, GAME_PANEL_H),
                interpolation=cv2.INTER_NEAREST
            )

            canvas[
                TOP_H:TOP_H + GAME_PANEL_H,
                0:GAME_PANEL_W
            ] = game_view

            # Persistentes Live Map Tiling: bekannte Tiles/Kanten nur einmal.
            map_view, map_meta = build_mapping_preview(
                watcher_known_tiles,
                watcher_known_edges,
                watcher_known_transitions,
                bank, map_id, x, y,
                MAP_PANEL_W, CONTENT_H
            )

            canvas[
                TOP_H:TOP_H + CONTENT_H,
                GAME_PANEL_W:CANVAS_W
            ] = map_view

            # Trennlinie
            cv2.line(
                canvas,
                (GAME_PANEL_W, TOP_H),
                (GAME_PANEL_W, TOP_H + CONTENT_H),
                (70, 78, 95),
                2
            )

            brain = (
                f"PKMAI v{loaded_version:06d}"
                if loaded_version >= 0
                else "Random Policy"
            )

            cv2.putText(
                canvas,
                f"{brain} | Pokemon FireRed AI by Alex",
                (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 230, 118),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                canvas,
                f"RAM Bank {bank} Map {map_id} | X {x} Y {y} | Battle {in_battle} | Reward {watcher_step_reward:+.2f} | START spam x{watcher_start_spam_count}",
                (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 190, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                canvas,
                "LIVE MAP TILING",
                (GAME_PANEL_W + 18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if map_meta:
                map_text = (
                    f"{map_meta.get('tiles', 0)} Felder | "
                    f"{map_meta.get('edges', 0)} Kanten | "
                    f"{map_meta.get('transitions', 0)} Warps | "
                    f"{watcher_mapping_event}"
                )
            else:
                map_text = "Live Map Tiling wartet ..."

            cv2.putText(
                canvas,
                map_text,
                (GAME_PANEL_W + 18, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (210, 215, 225),
                1,
                cv2.LINE_AA
            )

            footer_y = TOP_H + CONTENT_H

            cv2.putText(
                canvas,
                (
                    f"Steps {total_steps}   Level {p1_level}   "
                    f"Badges {badge_count}/8   "
                    f"Mapping {len(watcher_known_tiles)} Felder / {len(watcher_known_edges)} Kanten"
                ),
                (16, footer_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 220, 0),
                1,
                cv2.LINE_AA
            )

            history_text = "  ".join(action_history[-8:]) if action_history else "-"
            cv2.putText(
                canvas,
                f"INPUT: {current_action_name}    LAST: {history_text}",
                (16, footer_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 230, 118),
                1,
                cv2.LINE_AA
            )

            version_text = (
                f"PKMAI v{loaded_version:06d}"
                if loaded_version >= 0
                else "kein Modell"
            )
            try:
                cv2.setWindowTitle(
                    WINDOW,
                    f"Pokemon Firered AI by Alex - Watcher + Live Map | "
                    f"{version_text} | {fps_value:.1f} Emulator-FPS | ~50 GUI-FPS"
                )
            except Exception:
                pass

            cv2.imshow(WINDOW, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_counter += 1

        elapsed = time.perf_counter() - frame_start
        to_sleep = FRAME_TIME - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    save_watcher_mapping(
        watcher_known_tiles,
        watcher_known_edges,
        watcher_known_maps,
        watcher_known_transitions,
    )
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
