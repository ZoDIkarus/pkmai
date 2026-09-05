"""Read-only policy evaluation using the trainer's actual environment.

Only the module's runtime paths are redirected. Reward/observation/action/reset
methods are not overridden. Novelty registries belong to this evaluation run;
they never claim bonuses from the training fleet.
"""
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import signal
import time

import cv2
import numpy as np
from stable_baselines3 import PPO


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':')))
    tmp.replace(path)


def evaluation_module(root):
    spec = importlib.util.spec_from_file_location(
        'watcher_evaluation_env', Path(__file__).with_name('pokemon_env.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, folder in {
        'RUNTIME_DIR': '', 'INSTANCES_DIR': 'instances_data',
        'EXPLORATION_MEMORY_DIR': 'exploration_memory',
        'CURRICULUM_DIR': 'curriculum_states',
        'SHARED_CURRICULUM_DIR': 'curriculum_shared', 'STATS_DIR': 'training_stats',
    }.items():
        target = Path(root) / folder
        target.mkdir(parents=True, exist_ok=True)
        setattr(module, name, str(target))
    module.GLOBAL_PROGRESS_FILE = str(Path(root) / 'exploration_memory/global_progress.json')
    return module


def make_evaluation_env(root, navigation=((), (), ()), n_envs=60):
    module = evaluation_module(root)
    edges, maps, transitions = navigation
    # V17.4: shared_tiles darf NICHT leer starten wie shared_species - sonst
    # meldet der isolierte Watcher fuer JEDE Kachel, die er selbst seit
    # Prozessstart zum ersten Mal betritt, "new_tile_global", obwohl die
    # echte Flotte dort laengst war (Meldung im Stream: "bekommt staendig
    # tile global +2"). Es gibt keine persistierte Tile-Historie, aber jede
    # bekannte Kante verbindet zwei tatsaechlich betretene Kacheln - aus den
    # ohnehin schon geladenen Kanten (load_global_navigation_memory) lassen
    # sich beide Endpunkte ableiten und ergeben eine realistische Naeherung
    # der von der Flotte bereits gesehenen Kacheln.
    tiles = set()
    for e in edges:
        if len(e) == 6:
            bank, map_id, x1, y1, x2, y2 = e
            tiles.add((bank, map_id, x1, y1))
            tiles.add((bank, map_id, x2, y2))
    return module.PokemonFireRedEnv(
        rank=0, n_envs=n_envs,
        shared_edges=dict.fromkeys(edges, 1), shared_maps=dict.fromkeys(maps, 1),
        shared_transitions=dict.fromkeys(transitions, 1),
        shared_progress={'max_world_stage': 0}, shared_species={},
        shared_tiles=dict.fromkeys(tiles, 1),
    )


def telemetry(env, info, reward, episode, model_name, version):
    loc = env.cached_loc
    bank, map_id = int(loc.get('map_bank', 0)), int(loc.get('map_id', 0))
    return {
        'id': 120, 'name': 'Alex · Watcher', 'bank': bank, 'map': map_id,
        'x': int(loc.get('x_pos', 0)), 'y': int(loc.get('y_pos', 0)),
        'room': f'Bank {bank} / Map {map_id}', 'path': env.recent_path,
        'steps': env.route_steps, 'route_steps': env.route_steps,
        'battle_steps': env.battle_steps, 'ppo_episode_steps': env.total_steps,
        'reward': env.current_reward, 'step_reward': float(reward),
        'reward_events': env.recent_reward_events,
        'reward_engine': 'PokemonFireRedEnv.step', 'reward_scope': 'isolated_evaluation',
        'episode': episode, 'model_name': model_name, 'model_version': version,
        'level': env.last_level, 'badges': env.last_badges,
        'party': env.player_party_cache, 'enemy_party': env.enemy_party_cache,
        'has_starter': env.has_starter, 'has_target_starter': env.has_target_starter,
        'in_battle': env.last_in_battle, 'battle_detection': info.get('battle_detection'), 'world_stage': int(info.get('world_stage', 0)),
        'story_stage': info.get('story_stage', 'OUTDOOR'),
        'training_objective': 'watcher', 'training_role': 'watcher',
        'episode_start': env.episode_start, 'visited_maps': len(env.visited_maps),
        'explored_tiles': len(env.seen_coords), 'stuck_counter': env.stuck_counter,
        'last_stage_timeout': env.last_stage_timeout,
        'persistent_exploration': {'known_edges': len(env.persistent_known_edges),
                                 'known_maps': len(env.persistent_known_maps)},
        'battle_stats': {'started': env.run_stats.get('battles_started', 0),
                        'completed': env.run_stats.get('battles_completed', 0),
                        'enemy_faints': env.episode_enemy_faints},
        'updated_at': time.time(),
    }


_POS = (110, 215, 110)   # BGR green - positive reward
_NEG = (90, 90, 235)      # BGR red - negative reward
_NEU = (150, 150, 150)    # BGR gray - zero / unparsed
_GOLD = (100, 190, 225)   # BGR amber - neutral stat highlight


def _parse_event(ev):
    """Split 'name:+12.34' into ('name', 12.34); returns (ev, None) if unparsed."""
    m = re.match(r'^(.*):([+-]?\d+(?:\.\d+)?)$', str(ev))
    if not m:
        return str(ev), None
    return m.group(1), float(m.group(2))


def _event_color(amount):
    if amount is None or abs(amount) < 1e-9:
        return _NEU
    return _POS if amount > 0 else _NEG


def render_console(screen, data, events):
    """Neutral charcoal UI; game pixels retain their original colors."""
    # V17.3: 680 -> 570 - die Reward-Tabelle rechts braucht seit dem
    # 4-Zeilen-Kuerzen keine 680px Hoehe mehr, das Fenster wird dadurch
    # insgesamt kompakter fuers Streaming.
    # V17.4: 1200 -> 980 - die rechte Box (Status + Reward Events) auf die
    # Haelfte ihrer Breite gebracht (464 -> 232px), damit das Fenster
    # schmaler auf dem Screen sitzt. Statuswerte stehen dafuer einspaltig
    # untereinander statt in einem 3-Spalten-Raster - bleibt bei jeder
    # Zahlenlaenge (z.B. "Reward +1234.56") sauber lesbar, ohne Ueberlappung.
    canvas = np.full((570, 980, 3), 24, dtype=np.uint8)
    canvas[64:544, :720] = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR),
                                    (720, 480), interpolation=cv2.INTER_NEAREST)

    def label(text, x, y, color=(205, 205, 205), scale=.48, weight=1):
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, weight, cv2.LINE_AA)

    def right_label(text, x_right, y, color, scale=.4, weight=1):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        label(text, x_right - w, y, color, scale, weight)

    def panel(x0, y0, x1, y1, fill=(44, 44, 46), border=(64, 64, 70)):
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 1)

    def dot(x, y, color, r=4):
        cv2.circle(canvas, (x, y), r, color, -1, cv2.LINE_AA)

    label('PKMAI - LIVE WATCHER', 20, 24, (210, 225, 210), .65)
    label(
        f"Champion v{data.get('champion_version', 0)}  |  "
        f"Learner {data.get('learner_steps', 0):,} steps total  |  "
        f"FPS {data.get('fps', 0):.1f} / {data.get('target_fps', 300):.0f} target",
        20, 52, (150, 200, 235), .46
    )

    # -- status card: model + episode/reward/stage at a glance -----------
    # V17.4: rechte Box auf halbe Breite (464 -> 232px) - fuer ein
    # 3-Spalten-Raster reicht die Breite nicht mehr (z.B. "Reward +1234.56"
    # kann leicht ueber 100px breit werden). Einspaltig untereinander bleibt
    # bei jeder Zahlenlaenge garantiert ueberlappungsfrei und lesbar.
    rx0, rx1 = 728, 960
    panel(rx0, 8, rx1, 234)
    battle = bool(data['in_battle'])
    badge_color = _NEG if battle else _POS
    cv2.rectangle(canvas, (rx0 + 12, 18), (rx0 + 128, 40), badge_color, -1)
    label('BATTLE' if battle else 'OVERWORLD', rx0 + 20, 34, (20, 20, 20), .44, 1)
    label(f"{data['model_name']} v{data['model_version']}"[:30],
          rx0 + 12, 55, (185, 190, 202), .36)
    label(str(data.get('battle_detection', ''))[:34], rx0 + 12, 70,
          (135, 140, 152), .32)

    reward_col = _POS if data['reward'] >= 0 else _NEG
    step_col = _POS if data['step_reward'] >= 0 else _NEG
    rows = (
        (_POS, f"Episode {data['episode']}", (220, 220, 220)),
        (reward_col, f"Reward {data['reward']:+.2f}", reward_col),
        (_GOLD, f"Stage {data['world_stage']}", (220, 220, 220)),
        (_GOLD, f"Route {data['route_steps']}", (220, 220, 220)),
        (_NEG, f"Battle {data['battle_steps']}", (220, 220, 220)),
        (step_col, f"Step {data['step_reward']:+.4f}", step_col),
    )
    for i, (dot_color, text, text_color) in enumerate(rows):
        y = 92 + i * 19
        dot(rx0 + 12, y, dot_color, 5)
        label(text, rx0 + 26, y + 5, text_color, .42)

    # Kleingedrucktes bewusst winzig - nur Kontext, kein Statuswert, damit
    # die Karte kompakt bleibt und das Fenster sich leichter fuers Streamen
    # zurechtstutzen laesst.
    label('Same rules as training, inference only, no learning.',
          rx0 + 12, 210, (110, 114, 122), .28)
    label('Isolated evaluation - one-time bonuses stay private.',
          rx0 + 12, 222, (110, 114, 122), .28)

    # -- reward events table ---------------------------------------------
    # V17.3: 4 Zeilen weniger (16 -> 12), damit die Tabelle und damit das
    # ganze Fenster kompakter fuers Streaming wird.
    # V17.4: Zeilen-Layout an die halbierte Breite angepasst (kuerzerer
    # Event-Name, engerer Step-Prefix) - die Zeilenzahl sinkt leicht (12 ->
    # 10), weil die Statuskarte darueber jetzt einspaltig mehr Hoehe braucht.
    tx0, ty0, tx1, ty1 = rx0, 242, rx1, 542
    panel(tx0, ty0, tx1, ty1, fill=(33, 33, 36))
    cv2.rectangle(canvas, (tx0, ty0), (tx1, ty0 + 26), (36, 50, 40), -1)
    label('REWARD EVENTS', tx0 + 12, ty0 + 18, (170, 225, 170), .44)
    label('newest first', tx1 - 78, ty0 + 18, (110, 130, 115), .3)

    row_h = 27
    visible = list(reversed(events[-12:]))
    if not visible:
        label('No reward events yet.', tx0 + 16, ty0 + 48,
              (120, 125, 135), .38)
    for i, (step, ev) in enumerate(visible):
        y = ty0 + 26 + i * row_h
        if y + row_h > ty1:
            break
        if i % 2 == 1:
            cv2.rectangle(canvas, (tx0 + 1, y), (tx1 - 1, y + row_h), (39, 39, 42), -1)
        name, amount = _parse_event(ev)
        col = _event_color(amount)
        dot(tx0 + 10, y + row_h // 2 + 3, col, 4)
        label(f"S{step}", tx0 + 18, y + 18, (110, 115, 125), .3)
        label(name[:16], tx0 + 52, y + 18, (205, 208, 215), .36)
        if amount is not None:
            right_label(f"{amount:+.2f}", tx1 - 8, y + 18, col, .38, 1)

    label('Q / Esc to close.', 20, 556, (150, 150, 150), .38)
    label(f"Last reset: {data.get('last_reset', '-')}", 220, 556, (150, 150, 150), .38)
    return canvas


def run(api):
    root = Path(api.RUNTIME_DIR)
    # Persistent private memory, never included in fleet statistics or curriculum.
    env = make_evaluation_env(root / 'watcher_evaluation', api.load_global_navigation_memory())
    obs, info = env.reset()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    window = 'PKMAI / Alex Watcher'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 980, 570)
    model = None
    signature = None
    last_model_check = last_publish = 0
    fps_started = time.monotonic()
    fps_frames = 0
    measured_fps = 0.0
    episode = 1
    events = []
    last_reset = '-'
    learner_steps, champion_version = api.get_trainer_progress()
    consecutive_errors = 0
    log_path = root / 'watcher_rewards.jsonl'
    print('WATCHER: PokemonFireRedEnv.step | inference only | private reward memory', flush=True)
    print('Same reward rules; global first-discovery bonuses use a separate evaluation registry.', flush=True)
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    logger = logging.getLogger('watcher.reward.audit')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
            while not stop:
                start = time.monotonic()
                if start - last_model_check >= 1:
                    path = api.get_watcher_model_path()
                    candidate = api.get_model_signature(path)
                    if candidate != signature or model is None:
                        try:
                            loaded = PPO.load(path, device='cpu')
                            model, signature = loaded, candidate
                            print('MODEL:', Path(path).name, flush=True)
                        except Exception as exc:
                            print('MODEL LOAD:', exc, flush=True)
                    learner_steps, champion_version = api.get_trainer_progress()
                    last_model_check = start
                if model is None:
                    cv2.waitKey(100)
                    continue
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                except Exception as exc:
                    # Streaming-Ziel: der Watcher soll nie manuell neu
                    # gestartet werden muessen. Ein einzelner Emulator-/RAM-
                    # Ausrutscher darf das Fenster nicht beenden - Episode
                    # verwerfen und mit einem frischen Reset weitermachen.
                    consecutive_errors += 1
                    print(f"STEP ERROR ({consecutive_errors}):", exc, flush=True)
                    time.sleep(min(5.0, consecutive_errors * 0.5))
                    try:
                        obs, info = env.reset()
                        episode += 1
                        last_reset = f"recovered after error: {exc}"[:80]
                    except Exception as reset_exc:
                        print('RESET AFTER ERROR FAILED:', reset_exc, flush=True)
                    continue
                consecutive_errors = 0
                fps_frames += env.ACTION_HOLD_FRAMES + env.ACTION_RELEASE_FRAMES
                fps_now = time.monotonic()
                if fps_now - fps_started >= api.FPS_TITLE_INTERVAL:
                    measured_fps = fps_frames / (fps_now - fps_started)
                    fps_frames = 0
                    fps_started = fps_now
                    cv2.setWindowTitle(window, f"PKMAI / Alex Watcher | {measured_fps:.1f} FPS")
                step_events = list(info.get('reward_events', []))
                record = {'time': time.time(), 'episode': episode, 'step': env.total_steps,
                          'action': int(action), 'reward': float(reward),
                          'episode_reward': float(env.current_reward), 'events': step_events,
                          'terminated': terminated, 'truncated': truncated,
                          'reason': info.get('last_stage_timeout'),
                          'in_battle': env.last_in_battle, 'battle_detection': info.get('battle_detection'),
                          'battle_type_flags': info.get('battle_type_flags'),
                          'engine': 'PokemonFireRedEnv.step', 'scope': 'isolated_evaluation'}
                logger.info(json.dumps(record))
                if step_events or terminated or truncated:
                    line = f"E{episode} S{env.total_steps} {reward:+.4f} | " + ', '.join(step_events)
                    events.extend((env.total_steps, ev) for ev in step_events)
                    events = events[-60:]
                    print(line, flush=True)
                elif env.total_steps % 100 == 0:
                    print(f"E{episode} S{env.total_steps} | reward={reward:+.4f} total={env.current_reward:+.2f}", flush=True)
                if start - last_publish >= .1 or terminated or truncated:
                    data = telemetry(env, info, reward, episode, Path(path).stem, api.get_latest_version())
                    data['last_reset'] = last_reset
                    data['fps'] = round(measured_fps, 1)
                    data['target_fps'] = api.TARGET_FPS
                    data['learner_steps'] = learner_steps
                    data['champion_version'] = champion_version
                    atomic_json(root / 'instances_data/inst_120.json', data)
                    screen = env.env.get_screen()
                    canvas = render_console(screen, data, events)
                    api.publish_watcher_frame(canvas)
                    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(screen, cv2.COLOR_RGB2BGR),
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if ok:
                        tmp = root / 'watcher_emulator.tmp'
                        tmp.write_bytes(jpeg.tobytes())
                        tmp.replace(root / 'watcher_emulator.jpg')
                    tiles = {tuple(e[:2]) + tuple(e[2:4]) for e in env.persistent_known_edges}
                    tiles |= {tuple(e[:2]) + tuple(e[4:6]) for e in env.persistent_known_edges}
                    api.save_watcher_mapping(tiles, env.persistent_known_edges,
                                             env.persistent_known_maps, env.persistent_known_transitions)
                    cv2.imshow(window, canvas)
                    last_publish = start
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if terminated or truncated:
                    last_reset = info.get('last_stage_timeout') or 'episode limit / objective complete'
                    print('RESET:', last_reset, flush=True)
                    obs, info = env.reset()
                    episode += 1
                # Similar emulator speed to the old 300-frame/sec watcher.
                remaining = (env.ACTION_HOLD_FRAMES + env.ACTION_RELEASE_FRAMES) / api.TARGET_FPS - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        logger.removeHandler(handler)
        handler.close()
        env.close()
        cv2.destroyAllWindows()
