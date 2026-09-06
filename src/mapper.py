#!/usr/bin/env python3
"""Sichtbarer Karten-Spezialist mit persistentem Frontier-Graphen."""

import json
import os
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pokemon_env import PokemonFireRedEnv  # noqa: E402
from firered_ram import read_player_location  # noqa: E402
from tools.tile_map_builder import TileMapBuilder  # noqa: E402

RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
CHECKPOINT_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
MAPPER_DIR = os.path.join(RUNTIME_DIR, "mapper")
MAPPER_MODEL = os.path.join(CHECKPOINT_DIR, "pokemon_mapper_latest.zip")
RESUME_MODEL = os.path.join(CHECKPOINT_DIR, "pokemon_model_resume.zip")
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "pokemon_model_champion.zip")
MAPPER_FRAME = os.path.join(RUNTIME_DIR, "mapper.jpg")
MAPPER_STATUS = os.path.join(RUNTIME_DIR, "mapper_status.json")
MAPPER_MEMORY = os.path.join(MAPPER_DIR, "exploration_memory.json")
FRONTIER_MEMORY = os.path.join(MAPPER_DIR, "frontier_brain.json")

ACTION_INTERVAL = 1.5
DISPLAY_FPS = 60.0
CAPTURE_SETTLE_SECONDS = 0.75
WINDOW = "PKMAI Mapper - 60 FPS HD Map Capture"

DIRECTION_ACTIONS = {
    3: (0, -1),   # UP
    4: (0, 1),    # DOWN
    5: (-1, 0),   # LEFT
    6: (1, 0),    # RIGHT
}

# B bringt Dialoge/Untermenues zurueck, danach wird im 2x2-Kampfmenue
# reproduzierbar RUN (unten rechts) gewaehlt. Die Folge wiederholt sich, falls
# ein Dialog noch nicht fertig war oder ein Fluchtversuch fehlschlaegt.
BATTLE_ESCAPE_SEQUENCE = (1, 1, 4, 6, 0)


class FrontierBrain:
    """Deterministischer Graph-Explorer: jedes erreichbare Feld systematisch."""

    def __init__(self, path=FRONTIER_MEMORY):
        self.path = path
        self.edges = set()
        self.tried = set()
        self.failures = {}
        self.dirty = False
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f) or {}
            self.edges = {
                tuple(int(v) for v in item)
                for item in data.get("edges", [])
                if isinstance(item, list) and len(item) == 6
            }
            self.tried = {
                tuple(int(v) for v in item)
                for item in data.get("tried", [])
                if isinstance(item, list) and len(item) == 5
            }
            self.failures = {
                tuple(int(v) for v in item[:5]): int(item[5])
                for item in data.get("failures", [])
                if isinstance(item, list) and len(item) == 6
            }
        except Exception:
            self.edges = set()
            self.tried = set()
            self.failures = {}

    def save(self):
        if not self.dirty:
            return
        atomic_json(self.path, {
            "schema": 1,
            "edges": [list(v) for v in sorted(self.edges)],
            "tried": [list(v) for v in sorted(self.tried)],
            "failures": [
                list(key) + [count]
                for key, count in sorted(self.failures.items())
            ],
            "updated_at": time.time(),
        })
        self.dirty = False

    @staticmethod
    def _position(loc):
        if not loc or not loc.get("valid", False):
            return None
        return (
            int(loc.get("map_bank", 0)), int(loc.get("map_id", 0)),
            int(loc.get("x_pos", 0)), int(loc.get("y_pos", 0)),
        )

    @staticmethod
    def _edge(a, b):
        if b < a:
            a, b = b, a
        return a + b[2:]

    def observe(self, before_loc, action, after_loc):
        before = self._position(before_loc)
        after = self._position(after_loc)
        if before is None or after is None or action not in DIRECTION_ACTIONS:
            return
        attempt = before + (int(action),)
        if before[:2] == after[:2] and (
            abs(before[2] - after[2]) + abs(before[3] - after[3]) == 1
        ):
            self.edges.add(self._edge(before, after))
            self.tried.add(attempt)
            dx = before[2] - after[2]
            dy = before[3] - after[3]
            reverse = next(
                key for key, delta in DIRECTION_ACTIONS.items()
                if delta == (dx, dy)
            )
            self.tried.add(after + (reverse,))
            self.failures.pop(attempt, None)
            self.dirty = True
        elif before == after:
            count = self.failures.get(attempt, 0) + 1
            self.failures[attempt] = count
            if count >= 2:
                self.tried.add(attempt)
            self.dirty = True
        elif before[:2] != after[:2]:
            # Echte Tuer/Mapkante: als ausprobiert merken, aber niemals als
            # lange geometrische Linie in dieselbe Tilemap schreiben.
            self.tried.add(attempt)
            self.dirty = True
        self.save()

    def _has_frontier(self, pos):
        return any(pos + (action,) not in self.tried for action in DIRECTION_ACTIONS)

    def choose(self, loc):
        start = self._position(loc)
        if start is None:
            return None

        # Erst direkt am aktuellen Feld jede noch unbekannte Richtung testen.
        for action in (3, 6, 4, 5):
            if start + (action,) not in self.tried:
                return action

        adjacency = {}
        for edge in self.edges:
            a = edge[:4]
            b = (edge[0], edge[1], edge[4], edge[5])
            if a[:2] != start[:2]:
                continue
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        queue = deque([start])
        parent = {start: None}
        target = None
        while queue:
            node = queue.popleft()
            if node != start and self._has_frontier(node):
                target = node
                break
            for neighbor in sorted(adjacency.get(node, ())):
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)

        if target is not None:
            while parent[target] != start:
                target = parent[target]
            dx = target[2] - start[2]
            dy = target[3] - start[3]
            for action, delta in DIRECTION_ACTIONS.items():
                if delta == (dx, dy):
                    return action

        # Komplett abgeschlossener/temporär blockierter Bereich: alte Blockade
        # lokal erneut prüfen (NPCs können inzwischen weitergelaufen sein).
        for action in (4, 5, 6, 3):
            key = start + (action,)
            self.tried.discard(key)
            self.failures.pop(key, None)
        self.dirty = True
        self.save()
        return 4


def atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def atomic_jpeg(path, image):
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not ok:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(encoded.tobytes())
    os.replace(tmp, path)


class MapperEnv(PokemonFireRedEnv):
    """Progress-Umgebung mit einem ausschliesslich kartografischen Zusatzreward."""

    # ------------------------------------------------------------------
    # Harte Schreib-Isolation
    # ------------------------------------------------------------------
    # PokemonFireRedEnv wird nur fuer Emulator, Observation und robuste
    # RAM-Erkennung wiederverwendet. Alles, was das Training beeinflussen
    # koennte, wird in runtime/mapper umgeleitet oder ganz deaktiviert.

    def _exploration_memory_path(self):
        return os.path.join(MAPPER_DIR, "navigation_memory.json")

    def _load_run_stats(self):
        self.stats_file = os.path.join(MAPPER_DIR, "training_stats.json")
        return super()._load_run_stats()

    def _save_run_stats(self):
        self.stats_file = os.path.join(MAPPER_DIR, "training_stats.json")
        return super()._save_run_stats()

    def _save_curriculum_state(self, _milestone_name):
        return False

    def _save_stage_checkpoint(self, _stage, _bank, _map_id, _x, _y):
        return False

    def _claim_global_depth(self, _stage):
        return False

    def _claim_shared(self, _registry, _key):
        # Selbst falls der Mapper spaeter versehentlich Shared-Container
        # erhaelt, darf er sie weder ergaenzen noch Rewards darin claimen.
        return False

    def _save_confirmed_story_warp(self, _kind, _transition):
        return None

    def _commit_successful_exit_route(self):
        return None

    def _commit_journey_route(self):
        return None

    def __init__(self):
        self.map_builder = TileMapBuilder(MAPPER_DIR, save_interval=1.0)
        self.frontier_brain = FrontierBrain()
        self.mapper_seen_positions, self.mapper_seen_maps = self._load_mapper_memory()
        self.mapper_episode_positions = set()
        self.mapper_total_reward = 0.0
        self.mapper_new_positions = len(self.mapper_seen_positions)
        self.mapper_new_visual_tiles = 0
        self.mapper_last_event = "Start"
        self.mapper_started_at = time.time()
        self.stop_requested = False
        self.battle_escape_step = 0
        super().__init__(rank=121, n_envs=32)
        self.rank_state_dir = os.path.join(MAPPER_DIR, "curriculum_readonly")
        os.makedirs(self.rank_state_dir, exist_ok=True)
        self.saved_milestones = self._discover_saved_milestones()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        try:
            cv2.moveWindow(WINDOW, 80, 70)
        except Exception:
            pass

    def _agent_role(self):
        # Eigene Instanz/Telemetrie, aber dieselbe robuste Progress-Observation.
        return "progress", "Mapper"

    def _choose_episode_start(self):
        self.saved_milestones = self._discover_saved_milestones()
        self.training_objective = "progress"
        # Idealer Mapper-Start: Starter vorhanden, gesund und bereits draussen
        # in einer Wildgras-Zone. Das ist genau die Bedeutung von battle_ready.
        if "battle_ready" in set(self.saved_milestones):
            return "battle_ready"
        if "starter_outdoor" in set(self.saved_milestones):
            return "starter_outdoor"
        return self._best_progress_milestone()

    @staticmethod
    def _load_mapper_memory():
        try:
            with open(MAPPER_MEMORY, "r") as f:
                data = json.load(f)
            positions = {
                tuple(int(v) for v in item)
                for item in data.get("positions", [])
                if isinstance(item, list) and len(item) == 4
            }
            maps = {
                tuple(int(v) for v in item)
                for item in data.get("maps", [])
                if isinstance(item, list) and len(item) == 2
            }
            return positions, maps
        except Exception:
            return set(), set()

    def _save_mapper_memory(self):
        atomic_json(MAPPER_MEMORY, {
            "positions": [list(v) for v in sorted(self.mapper_seen_positions)],
            "maps": [list(v) for v in sorted(self.mapper_seen_maps)],
            "updated_at": time.time(),
        })

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Der Savestate-Sprung ist eine neue Aufnahme-Session und darf weder
        # Kameraversatz noch einen gezeichneten Pfad erzeugen.
        self.map_builder.reset_tracking()
        self.recent_path = []
        self.mapper_episode_positions.clear()
        self.battle_escape_step = 0
        return obs, info

    def _battle_escape_action(self):
        action = BATTLE_ESCAPE_SEQUENCE[
            self.battle_escape_step % len(BATTLE_ESCAPE_SEQUENCE)
        ]
        self.battle_escape_step += 1
        return action

    def _canvas(self, screen, status):
        game = cv2.cvtColor(screen[:160, :240], cv2.COLOR_RGB2BGR)
        game = cv2.resize(game, (960, 640), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((720, 960, 3), dtype=np.uint8)
        canvas[:640] = game
        canvas[640:] = (13, 17, 25)
        line = (
            f"MAPPER | {status['map']}  ({status['x']},{status['y']}) | "
            f"neue Felder {status['new_positions']} | "
            f"Bild-Tiles {status['new_visual_tiles']} | "
            f"Reward {status['reward']:.1f}"
        )
        cv2.putText(
            canvas, line, (18, 676), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (92, 235, 170), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "Frontier-Tilemap - 1 Aktion/1.5s - Q beendet sauber",
            (18, 704), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (190, 199, 215), 1, cv2.LINE_AA
        )
        return canvas

    def _run_idle_frames(self, deadline, status):
        """Emuliert neutrale Frames in Echtzeit und zeigt jedes davon an."""
        frame_time = 1.0 / DISPLAY_FPS
        next_frame_at = time.perf_counter()
        screen = self.env.get_screen()
        last_info = {}
        idle_terminated = False
        idle_truncated = False

        while not self.stop_requested:
            now = time.perf_counter()
            if now >= deadline:
                break

            if now < next_frame_at:
                time.sleep(min(next_frame_at - now, frame_time))
                continue

            result = self.env.step(self.btn_none)
            screen = result[0]
            if len(result) == 5:
                idle_terminated = idle_terminated or bool(result[2])
                idle_truncated = idle_truncated or bool(result[3])
                raw_info = result[4]
            else:
                idle_terminated = idle_terminated or bool(result[2])
                raw_info = result[3]
            if isinstance(raw_info, dict):
                last_info = raw_info

            cv2.imshow(WINDOW, self._canvas(screen, status))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                self.stop_requested = True
                break

            next_frame_at += frame_time
            # Bei einem kurzen GUI-Haenger nicht dutzende Frames ohne Pause
            # nachholen, sondern wieder sauber in den 60-Hz-Takt einsteigen.
            if next_frame_at < time.perf_counter() - frame_time:
                next_frame_at = time.perf_counter()

        return screen, last_info, idle_terminated, idle_truncated

    def step(self, action):
        started = time.perf_counter()
        before_loc = dict(getattr(self, "cached_loc", {}) or {})
        was_in_battle = int(getattr(self, "last_in_battle", 0) or 0)
        planned_action = None
        if was_in_battle:
            effective_mapper_action = self._battle_escape_action()
            controller = "battle_escape"
        else:
            self.battle_escape_step = 0
            planned_action = self.frontier_brain.choose(before_loc)
            effective_mapper_action = (
                int(planned_action) if planned_action is not None else int(action)
            )
            controller = (
                "frontier" if planned_action is not None else "ppo_fallback"
            )
        obs, _base_reward, terminated, truncated, info = super().step(
            effective_mapper_action
        )

        # Die Bewegung ist nach den 4+4 Action-Frames zwar steuerungsseitig
        # beendet, Scroll-/Spriteanimationen koennen aber noch sichtbar sein.
        # Zuerst 750 ms echte neutrale Emulatorframes laufen lassen; erst das
        # danach stabile Bild darf in die Karte oder ins Web.
        preliminary_loc = dict(getattr(self, "cached_loc", {}) or {})
        preliminary_status = {
            "map": (
                f"Bank {int(preliminary_loc.get('map_bank', 0))} / "
                f"Map {int(preliminary_loc.get('map_id', 0))}"
            ),
            "x": int(preliminary_loc.get("x_pos", 0)),
            "y": int(preliminary_loc.get("y_pos", 0)),
            "new_positions": self.mapper_new_positions,
            "new_visual_tiles": self.mapper_new_visual_tiles,
            "reward": self.mapper_total_reward,
        }
        settle_deadline = max(
            time.perf_counter(), started + CAPTURE_SETTLE_SECONDS
        )
        screen, idle_info, idle_term, idle_trunc = self._run_idle_frames(
            settle_deadline, preliminary_status
        )
        terminated = bool(terminated or idle_term)
        truncated = bool(truncated or idle_trunc)
        if idle_info:
            info = dict(info)
            info.update(idle_info)

        loc = dict(getattr(self, "cached_loc", {}) or {})
        try:
            settled_loc = read_player_location(self.env, allow_scan=False)
            if settled_loc and settled_loc.get("valid", False):
                loc = dict(settled_loc)
                self.cached_loc = dict(settled_loc)
        except Exception:
            pass
        bank = int(loc.get("map_bank", 0))
        map_id = int(loc.get("map_id", 0))
        x = int(loc.get("x_pos", 0))
        y = int(loc.get("y_pos", 0))
        trusted = bool(loc.get("valid", False) and loc.get("trusted", False))
        # Kampfmenue-Richtungen sind keine Weltbewegungen und duerfen niemals
        # als blockierte oder begehbare Kartenkante im Frontier-Graph landen.
        if not was_in_battle:
            self.frontier_brain.observe(
                before_loc, effective_mapper_action, loc
            )

        # Drei zeitlich getrennte, ruhige Beobachtungen desselben Standpunkts.
        # Dadurch gewinnt langfristig ein echter Animationsframe per Mehrheit,
        # statt dass Wasser/Blumen oder ein vorbeilaufender NPC Geometrie
        # verschieben. Bewegt sich die RAM-Position dazwischen, wird die
        # zusaetzliche Aufnahme verworfen.
        stable_screens = [screen]
        anchor = (bank, map_id, x, y)
        for sample_offset in (1.05, 1.35):
            sample_screen, sample_info, sample_term, sample_trunc = (
                self._run_idle_frames(
                    max(time.perf_counter(), started + sample_offset),
                    preliminary_status,
                )
            )
            terminated = bool(terminated or sample_term)
            truncated = bool(truncated or sample_trunc)
            if sample_info:
                info.update(sample_info)
            try:
                sample_loc = read_player_location(self.env, allow_scan=False)
            except Exception:
                sample_loc = None
            sample_pos = (
                int(sample_loc.get("map_bank", -1)),
                int(sample_loc.get("map_id", -1)),
                int(sample_loc.get("x_pos", -1)),
                int(sample_loc.get("y_pos", -1)),
            ) if sample_loc and sample_loc.get("valid", False) else None
            sample_battle = int(sample_info.get("in_battle", 0) or 0)
            if sample_pos == anchor and sample_battle == 0:
                stable_screens.append(sample_screen)
            screen = sample_screen

        new_visual = 0
        map_reward = -0.02
        event = "bekanntes Feld"
        in_battle = int(
            info.get("in_battle", getattr(self, "last_in_battle", 0)) or 0
        )
        if trusted and in_battle == 0:
            for stable_screen in stable_screens:
                new_visual += self.map_builder.add_frame(
                    stable_screen, bank, map_id, x, y, in_battle=0
                )
            pos = (bank, map_id, x, y)
            map_key = (bank, map_id)
            if map_key not in self.mapper_seen_maps:
                self.mapper_seen_maps.add(map_key)
                map_reward += 20.0
                event = "neue Map"
            if pos not in self.mapper_episode_positions:
                self.mapper_episode_positions.add(pos)
                map_reward += 1.2
                event = "neues Episoden-Feld"
            if pos not in self.mapper_seen_positions:
                self.mapper_seen_positions.add(pos)
                self.mapper_new_positions += 1
                map_reward += 4.0
                event = "weltweit neues Feld"
                self._save_mapper_memory()
            if new_visual:
                self.mapper_new_visual_tiles += int(new_visual)
                map_reward += min(3.0, float(new_visual) * 0.03)

        # Ausschliesslich Kartografie. Kein Story-, Kampf-, Level- oder
        # Haupt-Learner-Reward fliesst in dieses getrennte Gehirn ein.
        reward = map_reward
        self.mapper_total_reward += reward
        self.mapper_last_event = event

        status = {
            "running": not self.stop_requested,
            "id": 121,
            "name": "Mapper",
            "map": f"Bank {bank} / Map {map_id}",
            "bank": bank,
            "map_id": map_id,
            "x": x,
            "y": y,
            "new_positions": self.mapper_new_positions,
            "new_visual_tiles": self.mapper_new_visual_tiles,
            "known_maps": len(self.mapper_seen_maps),
            "reward": self.mapper_total_reward,
            "last_event": event,
            "actions": int(self.total_steps),
            "action_interval_seconds": ACTION_INTERVAL,
            "display_fps": DISPLAY_FPS,
            "capture_settle_ms": int(CAPTURE_SETTLE_SECONDS * 1000),
            "controller": controller,
            "effective_action": int(effective_mapper_action),
            "frontier_edges": len(self.frontier_brain.edges),
            "updated_at": time.time(),
        }
        canvas = self._canvas(screen, status)
        atomic_json(MAPPER_STATUS, status)
        # Im Kampf niemals ein Bild fuer Web oder Stitcher persistieren. Das
        # Dashboard behaelt stattdessen den letzten sauberen Welt-Screenshot.
        if in_battle == 0:
            atomic_jpeg(MAPPER_FRAME, canvas)

        final_deadline = max(time.perf_counter(), started + ACTION_INTERVAL)
        final_screen, final_info, idle_term, idle_trunc = self._run_idle_frames(
            final_deadline, status
        )
        terminated = bool(terminated or idle_term)
        truncated = bool(truncated or idle_trunc)
        if final_info:
            info.update(final_info)

        # Die naechste Entscheidung muss die wirklich zuletzt dargestellten
        # Frames sehen, nicht das Bild direkt nach der Bewegung.
        obs = self._make_obs(final_screen, loc=loc, info=info)
        if self.stop_requested:
            truncated = True
        info = dict(info)
        info["mapper_reward"] = reward
        info["mapper_new_visual_tiles"] = int(new_visual)
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.frontier_brain.save()
            self._save_mapper_memory()
            self.map_builder.save_all()
        finally:
            try:
                cv2.destroyWindow(WINDOW)
            except Exception:
                pass
            super().close()


def save_model_atomic(model):
    # Der temporaere Pfad muss selbst auf .zip enden. SB3 haengt die Endung
    # nur bei suffixlosen Pfaden an; ein Pfad auf `.tmp` wurde daher als
    # exakt diese Datei geschrieben und der anschliessende Replace suchte
    # irrtuemlich nach `.tmp.zip`.
    tmp_path = MAPPER_MODEL[:-4] + ".tmp.zip"
    model.save(tmp_path)
    os.replace(tmp_path, MAPPER_MODEL)


def main():
    os.makedirs(MAPPER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = MapperEnv()

    def request_stop(_signum, _frame):
        env.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    source = next(
        (p for p in (MAPPER_MODEL, RESUME_MODEL, BEST_MODEL) if os.path.exists(p)),
        None,
    )
    if source is None:
        raise FileNotFoundError("Kein Mapper-, Resume- oder Champion-Modell gefunden")

    print(f"🗺️ Mapper-Gehirn: {os.path.basename(source)}")
    print("   Frontier-Steuerung | 1 Aktion/1.5s | 60 FPS | 3 ruhige Aufnahmen")
    model = PPO.load(source, env=env, device="cpu")
    # Beim ersten Lauf sofort vom Haupt-Learner abzweigen. Danach wird nur noch
    # dieser eigene Checkpoint geladen und fortgeschrieben.
    if os.path.abspath(source) != os.path.abspath(MAPPER_MODEL):
        save_model_atomic(model)
    obs, _ = env.reset()
    try:
        while not env.stop_requested:
            # Nur in Kaempfen/Menues wird diese Policy-Aktion tatsaechlich
            # benutzt. In der Welt steuert der systematische Frontier-Graph.
            action, _ = model.predict(obs, deterministic=False)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                if env.stop_requested:
                    break
                obs, _ = env.reset()
    except KeyboardInterrupt:
        env.stop_requested = True
    finally:
        print("💾 Speichere Frontier-Gehirn und Bildkarten ...")
        atomic_json(MAPPER_STATUS, {
            "running": False,
            "name": "Mapper",
            "updated_at": time.time(),
        })
        env.close()


if __name__ == "__main__":
    main()
