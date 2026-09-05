import stable_retro as retro
import numpy as np
import cv2
import os
import json
import time
import glob
import signal
import urllib.request

from stable_baselines3 import PPO
from firered_ram import (
    read_battle_type_flags,
    read_player_location,
    read_player_party,
    read_enemy_party,
)

# The visible watcher executes the trainer environment from StartGame.state.
# It performs inference only and stores evaluation data separately.


# ================================================================
# USER CONFIG / WATCHER TUNING
# ================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
BASE_DIR = PROJECT_ROOT

# Geschwindigkeit / Anzeige
TARGET_FPS = 90.0
GUI_EVERY_FRAMES = 6
TELEMETRY_INTERVAL = 0.5
FPS_TITLE_INTERVAL = 0.5

# Action-Block: Taste halten + neutrale Frames
# Exakt dieselbe Aktionsausfuehrung wie PokemonFireRedEnv.step().
# MUSS mit PokemonFireRedEnv.ACTION_HOLD_FRAMES/ACTION_RELEASE_FRAMES identisch
# sein. 16 Halte-Frames = echter Kachel-Schritt statt nur Drehung.
# A/B-getestet 2026-09-05 (siehe pokemon_env.py) - 9/5 genauso zuverlaessig
# wie 12/6, ~22% weniger Emulator-Frames pro Entscheidung.
ACTION_HOLD_FRAMES = 9
ACTION_RELEASE_FRAMES = 5

# V15.3 BRAIN-MODUS: der sichtbare Watcher benutzt EIN vollstaendiges Netz
# end-to-end, ohne Skill-Snapshots oder hartkodierte Skill-Umschaltung. Er zeigt
# den bestaetigten Champion: der rohe Learner kann waehrend eines PPO-Blocks
# bereits gelernte Intro-Faehigkeiten zeitweise vergessen. Nach jeder echten
# Champion-Befoerderung wird das neue Gesamt-Brain automatisch nachgeladen.
WATCHER_BRAIN_MODE = True

# Reload / RAM
MODEL_CHECK_INTERVAL = 1.0
RAM_DISCOVERY_INTERVAL = 0.75
WATCHER_DEVICE = "cpu"

# Voruebergehendes, gut lesbares Reward-Protokoll fuer den sichtbaren Watcher.
# Bonus-/Strafereignisse werden sofort ausgegeben; reine Zeitkosten nur als
# gelegentliches Lebenszeichen, damit das Terminal nicht mit 300 FPS volllaeuft.
WATCHER_REWARD_DEBUG = os.environ.get(
    "PKMAI_WATCHER_REWARD_DEBUG", "1"
).strip().lower() not in {"0", "false", "no", "off"}
WATCHER_REWARD_IDLE_LOG_INTERVAL = 100

TARGET_STARTER_SPECIES = 7
STARTER_SPECIES = {1, 4, 7}
INTERACTION_SPAM_PENALTY_AFTER = 24
INTERACTION_SPAM_RESET_AT = 64
INTERACTION_SPAM_PENALTY = -0.5


def detect_starter_species(party):
    """Return only a validated Kanto starter species from party telemetry."""
    for mon in party or []:
        species = int(mon.get("species_id", 0) or 0)
        if (
            species in STARTER_SPECIES
            and int(mon.get("level", 0) or 0) >= 5
            and int(mon.get("max_hp", 0) or 0) > 0
        ):
            return species
    return 0


def watcher_starter_reward(species):
    if int(species) == TARGET_STARTER_SPECIES:
        return 1000.0
    if int(species) in STARTER_SPECIES:
        return -500.0
    return 0.0

# Live Map Tiling zeichnet ausschliesslich. Entdeckungen veraendern weder den
# Watcher-Reward noch den PPO-Reward.
MAPPING_GRID = True
WATCHER_MAPPING_FILE = os.path.join(RUNTIME_DIR, "watcher_mapping.json")
WATCHER_FRAME_FILE = os.path.join(RUNTIME_DIR, "watcher.jpg")

# Fenster
GAME_PANEL_W = 720
GAME_PANEL_H = 480
TEAM_PANEL_W = 118   # V11.5: Team-Spalte zwischen Emu und Live-Map
MAP_PANEL_W = 620
TOP_H = 84
BOTTOM_H = 74

# ================================================================
# INTERNAL PATHS - normalerweise nicht aendern
# ================================================================
MODEL_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
LATEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_latest.zip")
BEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_best.zip")
RESUME_MODEL = os.path.join(MODEL_DIR, "pokemon_model_resume.zip")
VERSION_FILE = os.path.join(RUNTIME_DIR, "model_version.json")
CHAMPION_FILE = os.path.join(RUNTIME_DIR, "champion_score.json")
SKILL_MODELS = {
    "intro": os.path.join(MODEL_DIR, "pokemon_skill_intro_best.zip"),
    "stairs": os.path.join(MODEL_DIR, "pokemon_skill_stairs_best.zip"),
    "exit": os.path.join(MODEL_DIR, "pokemon_skill_exit_best.zip"),
    "starter": os.path.join(MODEL_DIR, "pokemon_skill_squirtle_best.zip"),
    "progress": os.path.join(MODEL_DIR, "pokemon_skill_progress_best.zip"),
}

# Stage-Routing: Jede vorhandene Skill-Vault-Policy bedient ihren Abschnitt.
# Solange fuer "progress" noch kein Vault existiert, nutzt der Watcher den
# laufend gespeicherten Learner statt des frischen, weltunerfahrenen Champions.
STAGE_SKILLS_USING_VAULT = {"intro", "stairs", "exit", "starter", "progress"}

def get_watcher_model_path(skill=None):
    if WATCHER_BRAIN_MODE:
        # Ein einziges, vollstaendiges Champion-Netz fuer den gesamten Lauf.
        if os.path.exists(BEST_MODEL):
            return BEST_MODEL
        if os.path.exists(RESUME_MODEL):
            return RESUME_MODEL
        return LATEST_MODEL

    if skill in STAGE_SKILLS_USING_VAULT:
        # Der sichtbare Lauf benutzt pro Abschnitt den besten bestaetigten
        # Skill-Snapshot. Der rohe Resume-Learner wird alle 50k Steps
        # ueberschrieben und kann zwischendurch fruehe Faehigkeiten vergessen.
        vault_path = SKILL_MODELS.get(skill)
        if vault_path and os.path.exists(vault_path):
            return vault_path
        if os.path.exists(RESUME_MODEL):
            return RESUME_MODEL

    if os.path.exists(BEST_MODEL):
        return BEST_MODEL
    return LATEST_MODEL

def get_model_signature(path):
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
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


def publish_watcher_frame(canvas):
    """Atomically publish the visible watcher canvas as a JPEG snapshot."""
    try:
        ok, encoded = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not ok:
            return
        tmp = WATCHER_FRAME_FILE + ".tmp"
        with open(tmp, "wb") as f:
            f.write(encoded.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, WATCHER_FRAME_FILE)
    except Exception:
        pass



def get_latest_version():
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                return int(json.load(f).get("version", 0))
        except Exception:
            pass
    # Bei einem frischen Lauf existiert model_version.json erst nach der
    # ersten echten Champion-Befoerderung. Die geschuetzte Baseline hat ihre
    # Version bereits im Champion-Sidecar und soll nicht als v000000 erscheinen.
    if os.path.exists(CHAMPION_FILE):
        try:
            with open(CHAMPION_FILE, "r") as f:
                return int(json.load(f).get("version", 0))
        except Exception:
            pass
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


def load_confirmed_story_warps(kind, min_agents=2):
    base = os.path.join(
        RUNTIME_DIR,
        "curriculum_shared",
        "confirmed_story_warps"
    )
    votes = {}
    try:
        names = os.listdir(base)
    except Exception:
        return set()

    suffix = f"_{kind}.json"
    for name in names:
        if not name.startswith("agent_") or not name.endswith(suffix):
            continue
        try:
            with open(os.path.join(base, name), "r") as f:
                data = json.load(f)
            raw = data.get("transition", [])
            maps = (
                frozenset(((int(raw[0]), int(raw[1])),
                           (int(raw[4]), int(raw[5]))))
                if isinstance(raw, list) and len(raw) == 8
                else frozenset()
            )
            expected = {
                "stairs": frozenset(((4, 1), (4, 0))),
                "exit": frozenset(((4, 0), (3, 0))),
            }
            if maps == expected.get(kind):
                t = tuple(int(v) for v in raw)
                votes[t] = votes.get(t, 0) + 1
        except Exception:
            pass

    return {
        t for t, count in votes.items()
        if count >= min_agents
    }


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


def watcher_objective(loc, info):
    valid = bool(
        loc.get("valid", False)
        and loc.get("trusted", False)
    )

    if not valid:
        return "intro"

    bank = int(loc.get("map_bank", 0))
    in_battle = int(info.get("in_battle", 0))
    p_lvl = int(info.get("p1_level", 0))

    badges_raw = int(info.get("badges", 0))
    badges = bin(badges_raw).count("1") if badges_raw > 0 else 0

    if in_battle:
        return "battle"
    if bank == 4:
        return "stairs"
    if bank != 3:
        return "exit"
    if p_lvl < 5:
        return "starter"
    if p_lvl < 7:
        return "level"
    if badges < 1:
        return "badge"
    return "progress"


def build_v7_obs(
    screen,
    loc,
    info,
    known_edges,
    known_maps,
    known_transitions,
    party=None,
    story_flags=None,
    image_frames=None,
):
    valid = bool(
        loc.get("valid", False)
        and loc.get("trusted", False)
    )
    bank = int(loc.get("map_bank", 0)) if valid else 0
    map_id = int(loc.get("map_id", 0)) if valid else 0
    x = int(loc.get("x_pos", 0)) if valid else 0
    y = int(loc.get("y_pos", 0)) if valid else 0

    # V10 unified policy: Watcher uses exactly the same policy objective
    # as every training agent. Stage specialization now lives only in
    # curriculum/reward control, not in the PPO observation.
    active_objective = "full"
    vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    in_battle = int(info.get("in_battle", 0))
    p_lvl = int(info.get("p1_level", 0))
    badges_raw = int(info.get("badges", 0))
    badges = bin(badges_raw).count("1") if badges_raw > 0 else 0

    gameplay_ready = bool(valid and 0 <= x < 512 and 0 <= y < 512)
    party = party or []
    party_has_starter = any(
        int(mon.get("level", 0) or 0) >= 5
        and int(mon.get("max_hp", 0) or 0) > 0
        for mon in party
    )
    has_starter = p_lvl >= 5 or party_has_starter

    # Persistente Flags muessen exakt dieselbe Bedeutung wie im Training haben.
    story_flags = story_flags or {}
    stairs_done = 1.0 if story_flags.get("stairs_done", False) else 0.0
    house_left = 1.0 if story_flags.get("house_left", False) else 0.0

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

    # V10.4.1 WATCHER PARITY:
    # Policy objective remains FULL, but navigation stage is independent.
    if gameplay_ready and bool(story_flags.get("house_left", False)):
        nav_stage = "overworld"
        nav_transitions = known_transitions
    elif gameplay_ready and not bool(story_flags.get("stairs_done", False)):
        nav_stage = "stairs"
        nav_transitions = load_confirmed_story_warps("stairs")
    elif gameplay_ready:
        nav_stage = "exit"
        nav_transitions = load_confirmed_story_warps("exit")
    else:
        nav_stage = "intro"
        nav_transitions = set()

    target = (
        watcher_nav_target(
            bank, map_id, x, y,
            known_edges, known_maps, nav_transitions
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
        float(np.clip(int(loc.get("viridian_mart_scene", 0) or 0) / 2.0, 0.0, 1.0)),
        float(np.clip(int(loc.get("pallet_oaks_lab_scene", 0) or 0) / 6.0, 0.0, 1.0)),
        float(np.clip(int(loc.get("viridian_old_man_scene", 0) or 0) / 2.0, 0.0, 1.0)),
    ])

    levels = [int(m.get("level", 0)) for m in party if int(m.get("level", 0)) > 0]
    hp_values = [
        float(m.get("hp_ratio", 0.0))
        for m in party
        if int(m.get("max_hp", 0)) > 0
    ]
    party_size = len(levels)
    party_max = max(levels) if levels else 0
    party_avg = sum(levels) / len(levels) if levels else 0.0
    party_hp = sum(hp_values) / len(hp_values) if hp_values else 0.0
    vec.extend([
        float(np.clip(party_size / 6.0, 0.0, 1.0)),
        float(np.clip(party_max / 100.0, 0.0, 1.0)),
        float(np.clip(party_avg / 100.0, 0.0, 1.0)),
        float(np.clip(party_hp, 0.0, 1.0)),
    ])

    nav = np.asarray(vec, dtype=np.float32)
    if nav.shape != (31,):
        raise RuntimeError(f"Watcher V8 nav shape invalid: {nav.shape}")

    frame = process_image(screen)
    if image_frames is None:
        stacked_image = np.concatenate([frame] * 4, axis=0)
    else:
        if not image_frames:
            image_frames.extend(frame.copy() for _ in range(4))
        else:
            image_frames.append(frame)
            del image_frames[:-4]
        stacked_image = np.concatenate(image_frames, axis=0)

    return {
        "image": stacked_image,
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

    # V15.3: stabile Kamera statt Auto-Fit. Vorher wurden cx/cy/scale JEDEN
    # Frame neu aus der Bounding-Box ALLER bekannten Tiles berechnet - sobald
    # der Agent eine noch unbekannte Kachel betrat, aenderte sich die Box und
    # die GESAMTE Karte sprang (neues Zentrum, neuer Zoom). Jede gruene Kachel
    # "wanderte" bei jedem Schritt durch unerkundetes Gebiet. Jetzt: feste
    # Kachelgroesse, Kamera zentriert immer auf die aktuelle Spielerposition -
    # wie ein mitlaufender Minimap-Ausschnitt, kein Springen mehr.
    scale = 20.0
    cx, cy = float(x), float(y)

    def screen_pos(px, py):
        vx = int(width / 2 + (px - cx) * scale)
        vy = int(height / 2 + (py - cy) * scale)
        return vx, vy

    half_tiles_x = int(width / 2 / scale) + 1
    half_tiles_y = int(height / 2 / scale) + 1
    min_x, max_x = int(cx) - half_tiles_x, int(cx) + half_tiles_x
    min_y, max_y = int(cy) - half_tiles_y, int(cy) + half_tiles_y

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

def _hp_color(ratio, fainted):
    if fainted:
        return (90, 90, 90)
    if ratio <= 0.25:
        return (60, 60, 240)   # rot (BGR)
    if ratio <= 0.55:
        return (60, 210, 240)  # gelb
    return (120, 230, 90)      # gruen


SPRITE_DIR = os.path.join(LOCAL_DIR, "..", "assets", "sprites", "pokemon")
SPRITE_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/"
    "pokemon/versions/generation-iii/firered-leafgreen/{}.png"
)
_sprite_cache = {}


def _get_pokemon_sprite(species_id, size=32):
    """Laedt/cached das FireRed/LeafGreen-Sprite fuer eine Pokemon-Spezies.
    Dieselbe oeffentliche PokeAPI-Sprite-Quelle wie im Web-Dashboard
    (web_stream.py). Lokal unter assets/sprites/pokemon/<id>.png gecached,
    damit nicht bei jedem Frame neu heruntergeladen wird. Gibt bei Fehlern
    None zurueck - der Team-Panel-Text funktioniert dann weiter wie bisher."""
    species_id = int(species_id or 0)
    if species_id <= 0:
        return None
    cache_key = (species_id, size)
    if cache_key in _sprite_cache:
        return _sprite_cache[cache_key]

    os.makedirs(SPRITE_DIR, exist_ok=True)
    local_path = os.path.join(SPRITE_DIR, f"{species_id}.png")
    if not os.path.exists(local_path):
        try:
            urllib.request.urlretrieve(
                SPRITE_URL.format(species_id), local_path
            )
        except Exception:
            _sprite_cache[cache_key] = None
            return None

    try:
        img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            _sprite_cache[cache_key] = None
            return None
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    except Exception:
        img = None
    _sprite_cache[cache_key] = img
    return img


def _blit_rgba(canvas, sprite, x, y):
    """Zeichnet ein BGRA-Sprite alphaueberblendet auf den Canvas an (x, y)."""
    if sprite is None:
        return
    sh, sw = sprite.shape[:2]
    ch, cw = canvas.shape[:2]
    if x < 0 or y < 0 or x + sw > cw or y + sh > ch:
        return
    region = canvas[y:y + sh, x:x + sw]
    alpha = (sprite[:, :, 3:4].astype(np.float32)) / 255.0
    blended = (
        sprite[:, :, :3].astype(np.float32) * alpha
        + region.astype(np.float32) * (1.0 - alpha)
    )
    canvas[y:y + sh, x:x + sw] = blended.astype(np.uint8)


# National-Dex-IDs der Starter (Bisasam/Glumanda/Schiggy) - einzige Spezies,
# deren Wachstumsrate ("medium slow") hier sicher bekannt ist.
_MEDIUM_SLOW_SPECIES = {1, 4, 7}


def _exp_for_level(level, species_id):
    """Naeherungsweise Gesamt-EXP fuer ein gegebenes Level. Medium-Slow-Formel
    fuer die Starter, sonst Medium-Fast (n^3) als verbreitete Naeherung fuer
    fruehe Gen-1-Wildpokemon (nicht exakt fuer jede Wachstumsrate-Gruppe)."""
    n = max(1, int(level))
    if int(species_id or 0) in _MEDIUM_SLOW_SPECIES:
        return max(0, int(1.2 * n**3 - 15 * n**2 + 100 * n - 140))
    return int(n**3)


def draw_team_overlay(canvas, party, ui, x0, y0, w, h):
    """Team-SPALTE zwischen Emu und Live-Map. 6 Pokemon untereinander.
    Klick auf eines -> Stats/Moves-Popup, nochmal klicken -> weg."""
    party = [m for m in (party or []) if int(m.get("max_hp", 0)) > 0][:6]
    ui["slots"] = []
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (14, 18, 26), -1)
    cv2.putText(canvas, "TEAM", (x0 + 10, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 130, 150), 1, cv2.LINE_AA)
    if not party:
        return

    top = y0 + 24
    sh = (h - 28) // 6
    for i, mon in enumerate(party):
        sy = top + i * sh
        ratio = float(mon.get("hp_ratio", 0.0))
        fainted = bool(mon.get("fainted", False))
        if ui.get("expanded") == i:
            cv2.rectangle(canvas, (x0 + 2, sy + 1), (x0 + w - 2, sy + sh - 2),
                          (0, 230, 118), 1)
        sprite = _get_pokemon_sprite(mon.get("species_id", 0), size=28)
        text_x0 = x0 + 8
        if sprite is not None:
            _blit_rgba(canvas, sprite, x0 + 4, sy + 2)
            text_x0 = x0 + 36
        name = str(mon.get("name", "?"))[:11]
        cv2.putText(canvas, name, (text_x0, sy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 240, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Lv{mon.get('level', 0)}", (text_x0, sy + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 160, 180), 1, cv2.LINE_AA)
        bx0, bx1 = x0 + 8, x0 + w - 8
        by = sy + 42
        cv2.rectangle(canvas, (bx0, by), (bx1, by + 7), (40, 44, 54), -1)
        fillw = int((bx1 - bx0) * max(0.0, min(1.0, ratio)))
        if fillw > 0:
            cv2.rectangle(canvas, (bx0, by), (bx0 + fillw, by + 7),
                          _hp_color(ratio, fainted), -1)
        cv2.putText(canvas, f"{mon.get('cur_hp',0)}/{mon.get('max_hp',0)}",
                    (bx0, by + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (140, 150, 168), 1, cv2.LINE_AA)
        ui["slots"].append((x0, sy, x0 + w, sy + sh, i))

    exp = ui.get("expanded")
    if exp is not None and exp < len(party):
        mon = party[exp]
        pw, ph = 290, 208
        px = x0 - pw - 6
        py = y0 + 10
        ov = canvas.copy()
        cv2.rectangle(ov, (px, py), (px + pw, py + ph), (18, 22, 32), -1)
        cv2.addWeighted(ov, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 230, 118), 1)
        big_sprite = _get_pokemon_sprite(mon.get("species_id", 0), size=64)
        if big_sprite is not None:
            _blit_rgba(canvas, big_sprite, px + pw - 72, py + 8)
        cv2.putText(canvas, f"{mon.get('name','?')}  Lv{mon.get('level',0)}",
                    (px + 12, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 230, 118), 1, cv2.LINE_AA)
        st = mon.get("stats", {})
        lines = [
            f"HP  {mon.get('cur_hp',0)}/{mon.get('max_hp',0)}",
            f"ATK {st.get('attack',0)}   DEF {st.get('defense',0)}",
            f"SPA {st.get('sp_attack',0)}   SPD {st.get('sp_defense',0)}   SPE {st.get('speed',0)}",
        ]
        for j, ln in enumerate(lines):
            cv2.putText(canvas, ln, (px + 12, py + 48 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 222, 234), 1, cv2.LINE_AA)
        moves = mon.get("moves", []) or []
        cv2.putText(canvas, "Moves:", (px + 12, py + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 160, 180), 1, cv2.LINE_AA)
        for j, mv in enumerate(moves[:4]):
            cv2.putText(canvas, f"- {mv.get('name', mv.get('id','?'))}",
                        (px + 20, py + 136 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 208, 220), 1, cv2.LINE_AA)

        # EXP-Balken: Fortschritt zum naechsten Level. Gen-3-Wachstumsrate ist
        # aus der Party-Struktur nicht direkt dekodiert - Starter sind bekannt
        # "medium slow", fuer alles andere "medium fast" (n^3) als verbreitete
        # Naeherung fuer fruehe Gen-1-Wildpokemon. Nicht exakt fuer jede
        # Spezies, aber deutlich naeher als gar keine Anzeige.
        level = int(mon.get("level", 0))
        exp_cur = int(mon.get("experience", 0))
        exp_lo = _exp_for_level(level, mon.get("species_id", 0))
        exp_hi = _exp_for_level(level + 1, mon.get("species_id", 0))
        exp_ratio = 0.0
        if exp_hi > exp_lo:
            exp_ratio = max(0.0, min(1.0, (exp_cur - exp_lo) / (exp_hi - exp_lo)))
        ey = py + ph - 22
        cv2.putText(canvas, "EXP", (px + 12, ey - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 160, 180), 1, cv2.LINE_AA)
        ebx0, ebx1 = px + 46, px + pw - 12
        cv2.rectangle(canvas, (ebx0, ey - 10), (ebx1, ey - 2), (40, 44, 54), -1)
        efw = int((ebx1 - ebx0) * exp_ratio)
        if efw > 0:
            cv2.rectangle(canvas, (ebx0, ey - 10), (ebx0 + efw, ey - 2),
                          (120, 230, 90), -1)


def make_team_click_handler(ui):
    def _on(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (sx0, sy0, sx1, sy1, idx) in ui.get("slots", []):
            if sx0 <= mx <= sx1 and sy0 <= my <= sy1:
                ui["expanded"] = None if ui.get("expanded") == idx else idx
                return
        ui["expanded"] = None
    return _on


def main():
    # Single source of truth: run the actual trainer environment, never the
    # removed historical approximate reward mirror.
    import sys
    from watcher_runtime import run
    run(sys.modules[__name__])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Watcher beendet.")
