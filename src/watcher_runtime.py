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
from watcher_display import FramePacer, FramePublisher
from watcher_models import load_valid_policy


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


def make_evaluation_env(root, navigation=((), (), ()), n_envs=46):
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
    env = module.PokemonFireRedEnv(
        rank=0, n_envs=n_envs,
        shared_edges=dict.fromkeys(edges, 1), shared_maps=dict.fromkeys(maps, 1),
        shared_transitions=dict.fromkeys(transitions, 1),
        shared_progress={'max_world_stage': max(
            (module.PokemonFireRedEnv.WORLD_STAGE_BY_MAP.get(tuple(m), 0) for m in maps),
            default=1)}, shared_species={},
        shared_tiles=dict.fromkeys(tiles, 1),
    )
    # The watcher always plays a full run from the master savegame, never a
    # curriculum checkpoint (rank 0 must not inherit FIGHTER / any resume mode).
    env.is_watcher = True
    # FULL-watcher parity: same NavigationBattleWrapper / router / champions as a
    # FULL worker, but learning=False (no rollouts, no optimizer, no counters).
    try:
        from twoby2.live_integration import (maybe_wrap_full_agent,
                                             latest_battle_champion_policy)
        env = maybe_wrap_full_agent(env, learning=False,
                                    battle_policy=latest_battle_champion_policy)
    except Exception:
        pass
    return env


def telemetry(env, info, reward, episode, model_name, version):
    loc = env.cached_loc
    bank, map_id = int(loc.get('map_bank', 0)), int(loc.get('map_id', 0))
    return {
        'id': 120, 'name': 'Alex · Watcher', 'bank': bank, 'map': map_id,
        'x': int(loc.get('x_pos', 0)), 'y': int(loc.get('y_pos', 0)),
        'room': f'Bank {bank} / Map {map_id}', 'path': env.recent_path,
        'steps': env.route_steps, 'route_steps': env.route_steps,
        'battle_steps': env.battle_steps, 'ppo_episode_steps': env.total_steps,
        'episode_step_limit': int(info.get('episode_step_limit', 0) or 0),
        'episode_steps_remaining': int(info.get('episode_steps_remaining', 0) or 0),
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

# Canvas / game-frame geometry - kept here so run() and render_console agree.
# 2026-09-08 (user): header + subtitle on one row, no bottom bar, tighter
# right card -> ~46px shorter, more screen room for the livestream.
_WCANVAS_W, _WCANVAS_H = 980, 524
_GAME_X0, _GAME_Y0, _GAME_W, _GAME_H = 0, 44, 720, 480


def _parse_event(ev):
    """Split 'name:+12.34' into ('name', 12.34); returns (ev, None) if unparsed."""
    m = re.match(r'^(.*):([+-]?\d+(?:\.\d+)?)$', str(ev))
    if not m:
        return str(ev), None
    return m.group(1), float(m.group(2))


def _abbrev_steps(n):
    """3,464,600 -> '3.465 Mil'; below a million stays fully written."""
    n = int(n or 0)
    return f"{n / 1e6:.3f} Mil" if n >= 1_000_000 else f"{n:,}"


def _event_color(amount):
    if amount is None or abs(amount) < 1e-9:
        return _NEU
    return _POS if amount > 0 else _NEG


def render_console(screen, data, events):
    """Neutral charcoal UI; game pixels retain their original colors."""
    # 2026-09-08 (user): the header keeps a big title with a readable
    # "Champion vN <steps>" subtitle beside it. The OVERVIEW card carries the
    # OVERWORLD/BATTLE badge, a BRAIN block (heading + two pills: NAV vN and
    # BATTLE vN/rule, the steering one lit), a Learner/Fights line, then a
    # 2x3 grid of metric tiles (Episode | Reward / Steps | Stage /
    # Battle | Step-rwd). REWARD EVENTS below gains a best/worst summary line
    # with a green up- / red down-triangle, like the web dashboard.
    canvas = np.full((_WCANVAS_H, _WCANVAS_W, 3), 24, dtype=np.uint8)
    canvas[_GAME_Y0:_GAME_Y0 + _GAME_H, _GAME_X0:_GAME_X0 + _GAME_W] = cv2.resize(
        cv2.cvtColor(screen, cv2.COLOR_RGB2BGR), (_GAME_W, _GAME_H),
        interpolation=cv2.INTER_NEAREST)

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

    # -- header: title + a readable champion subtitle on one baseline ---
    title = 'PKMAI - LIVE WATCHER'
    label(title, 20, 33, (210, 225, 210), .62)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, .62, 1)
    label(
        f"Champion v{data.get('champion_version', 0)}  "
        f"{_abbrev_steps(data.get('champion_steps', 0))}"
        f"    |    Battle  {int(data.get('battle_fights', 0)):,} Fights",
        20 + tw + 26, 32, (150, 205, 240), .50
    )

    # -- OVERVIEW card ------------------------------------------------
    rx0, rx1 = 728, 960
    cy0, cy1 = 8, 228
    panel(rx0, cy0, rx1, cy1)
    cv2.rectangle(canvas, (rx0, cy0), (rx1, cy0 + 22), (44, 52, 60), -1)
    label('OVERVIEW', rx0 + 12, cy0 + 16, (200, 216, 232), .44)

    battle = bool(data['in_battle'])
    badge_color = _NEG if battle else _POS
    cv2.rectangle(canvas, (rx0 + 12, cy0 + 30), (rx0 + 130, cy0 + 50), badge_color, -1)
    label('BATTLE' if battle else 'OVERWORLD', rx0 + 20, cy0 + 45, (20, 20, 20), .42, 1)

    # BRAIN: heading + two pills; the pill that is steering right now lights up.
    label('BRAIN', rx0 + 12, cy0 + 68, (150, 190, 225), .38)
    ncv = int(data.get('champion_version', 0) or 0)
    bcv = int(data.get('battle_champion_version', 0) or 0)
    pills = (
        (f"NAV v{ncv}", not battle, (46, 104, 46), (120, 190, 120)),
        (f"BATTLE {'v' + str(bcv) if bcv else 'rule'}", battle,
         (52, 52, 128), (120, 120, 220)),
    )
    px = rx0 + 12
    for text, active, on_fill, on_border in pills:
        pw = 102
        cv2.rectangle(canvas, (px, cy0 + 74), (px + pw, cy0 + 94),
                      on_fill if active else (46, 46, 50), -1)
        cv2.rectangle(canvas, (px, cy0 + 74), (px + pw, cy0 + 94),
                      on_border if active else (70, 70, 76), 1)
        label(text[:13], px + 8, cy0 + 88,
              (232, 236, 232) if active else (168, 170, 178), .34)
        px += pw + 6

    label(
        f"Learner  {_abbrev_steps(data.get('learner_steps', 0))} steps",
        rx0 + 12, cy0 + 108, (150, 152, 162), .32
    )

    # 2x3 metric tiles
    reward_col = _POS if data['reward'] >= 0 else _NEG
    step_col = _POS if data['step_reward'] >= 0 else _NEG
    metric_tiles = (
        ("EPISODE", f"{data['episode']}", (222, 222, 222)),
        ("REWARD", f"{data['reward']:+.2f}", reward_col),
        ("STEPS", f"{data['route_steps']}", _GOLD),
        ("STAGE", f"{data['world_stage']}", _GOLD),
        ("BATTLE", f"{data['battle_steps']}", (222, 222, 222)),
        ("STEP RWD", f"{data['step_reward']:+.4f}", step_col),
    )
    gx = (rx0 + 12, rx0 + 122)
    gy0, tile_w, tile_h = cy0 + 116, 104, 30
    for i, (name, value, vcol) in enumerate(metric_tiles):
        x0 = gx[i % 2]
        y0 = gy0 + (i // 2) * (tile_h + 4)
        panel(x0, y0, x0 + tile_w, y0 + tile_h, fill=(38, 38, 42))
        label(name, x0 + 7, y0 + 12, (140, 143, 152), .30)
        label(value, x0 + 7, y0 + 26, vcol, .42)

    # -- reward events table (with a best / worst summary line) --------
    tx0, ty0, tx1, ty1 = rx0, 236, rx1, _WCANVAS_H - 6
    panel(tx0, ty0, tx1, ty1, fill=(33, 33, 36))
    cv2.rectangle(canvas, (tx0, ty0), (tx1, ty0 + 22), (36, 50, 40), -1)
    label('REWARD EVENTS', tx0 + 12, ty0 + 16, (170, 225, 170), .42)

    def _tri(x, y, up, color):
        pts = (np.array([[x, y - 4], [x - 4, y + 3], [x + 4, y + 3]], np.int32)
               if up else
               np.array([[x, y + 4], [x - 4, y - 3], [x + 4, y - 3]], np.int32))
        cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)

    amounts = [(n, a) for n, a in (_parse_event(ev) for _, ev in events)
               if a is not None]
    if amounts:
        best = max(amounts, key=lambda t: t[1])
        worst = min(amounts, key=lambda t: t[1])
        _tri(tx0 + 14, ty0 + 34, True, _POS)
        label(f"{best[1]:+.2f}  {best[0][:16]}", tx0 + 24, ty0 + 38, _POS, .34)
        _tri(tx0 + 14, ty0 + 52, False, _NEG)
        label(f"{worst[1]:+.2f}  {worst[0][:16]}", tx0 + 24, ty0 + 56, _NEG, .34)
    else:
        label('best / worst: no events yet', tx0 + 14, ty0 + 46,
              (110, 115, 125), .32)
    cv2.line(canvas, (tx0 + 6, ty0 + 64), (tx1 - 6, ty0 + 64), (52, 52, 58), 1)

    row_h = 24
    visible = list(reversed(events[-12:]))
    if not visible:
        label('No reward events yet.', tx0 + 16, ty0 + 90, (120, 125, 135), .38)
    for i, (step, ev) in enumerate(visible):
        y = ty0 + 70 + i * row_h
        if y + row_h > ty1:
            break
        if i % 2 == 1:
            cv2.rectangle(canvas, (tx0 + 1, y), (tx1 - 1, y + row_h), (39, 39, 42), -1)
        name, amount = _parse_event(ev)
        col = _event_color(amount)
        dot(tx0 + 10, y + row_h // 2 + 3, col, 4)
        label(f"S{step}", tx0 + 18, y + 16, (110, 115, 125), .3)
        label(name[:16], tx0 + 52, y + 16, (205, 208, 215), .36)
        if amount is not None:
            right_label(f"{amount:+.2f}", tx1 - 8, y + 16, col, .38, 1)

    return canvas


def run(api):
    root = Path(api.RUNTIME_DIR)
    # Persistent private memory, never included in fleet statistics or curriculum.
    env = make_evaluation_env(root / 'watcher_evaluation', api.load_global_navigation_memory())
    core = env.unwrapped
    obs, info = env.reset()
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    window = 'PKMAI / Alex Watcher'   # cv2 handle id - stays stable
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, _WCANVAS_W, _WCANVAS_H)
    cv2.setWindowTitle(window, "PKMai Watcher / FPS --")
    model = None
    signature = None
    rejected_models = {}
    active_model_path = None
    last_model_check = last_publish = 0
    last_navigation_refresh = time.monotonic()
    fps_started = time.monotonic()
    fps_frames = 0
    measured_fps = 0.0
    episode = 1
    events = []
    last_reset = '-'
    learner_steps, champion_version, champion_steps = api.get_trainer_progress()
    battle_prog = api.get_battle_progress()
    consecutive_errors = 0
    canvas = np.full((_WCANVAS_H, _WCANVAS_W, 3), 24, dtype=np.uint8)
    pacer = FramePacer(api.TARGET_FPS)
    publisher = FramePublisher(root)
    last_stream_frame = 0.0

    def display_frame():
        nonlocal stop, last_stream_frame, fps_frames
        pacer.wait()
        screen = core.env.get_screen()
        canvas[_GAME_Y0:_GAME_Y0 + _GAME_H, _GAME_X0:_GAME_X0 + _GAME_W] = cv2.resize(
            cv2.cvtColor(screen, cv2.COLOR_RGB2BGR), (_GAME_W, _GAME_H),
            interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, canvas)
        fps_frames += 1
        if cv2.waitKey(1) & 0xff in (27, ord('q')):
            stop = True
        now = time.monotonic()
        if now - last_stream_frame >= 1.0 / 30:
            publisher.submit(canvas, screen)
            last_stream_frame = now

    core.frame_callback = display_frame
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
                # V17.4-Fix: die isolierte Navigation (shared_edges/maps/
                # transitions/tiles) wurde bisher nur EINMAL bei Prozessstart
                # aus den Agenten-Dateien geseedet - lief der Watcher lange
                # ohne Neustart (z.B. waehrend eines Livestreams, der bewusst
                # nicht unterbrochen werden soll), driftete dieser Schnappschuss
                # immer weiter von der echten Flotte weg. Ergebnis: laengst
                # von der Flotte bekannte Kanten/Kacheln/Warps (z.B. die
                # Haustuer nach einem Party-Wipe-Teleport) wurden im Watcher
                # faelschlich wieder als "global neu" gemeldet. Alle 5 Minuten
                # neu aus den (weiter wachsenden) Agenten-Dateien nachladen und
                # in die bestehenden Dicts MERGEN (nicht ersetzen) - reine
                # Lesenoperation, kein Einfluss auf das echte Training.
                if start - last_navigation_refresh >= 300:
                    try:
                        edges, maps, transitions = api.load_global_navigation_memory()
                        core.shared_edges.update(dict.fromkeys(edges, 1))
                        core.shared_maps.update(dict.fromkeys(maps, 1))
                        core.shared_transitions.update(dict.fromkeys(transitions, 1))
                        tiles = {}
                        for e in edges:
                            if len(e) == 6:
                                b, m, ex1, ey1, ex2, ey2 = e
                                tiles[(b, m, ex1, ey1)] = 1
                                tiles[(b, m, ex2, ey2)] = 1
                        core.shared_tiles.update(tiles)
                    except Exception as exc:
                        print('NAV REFRESH ERROR:', exc, flush=True)
                    last_navigation_refresh = start
                if start - last_model_check >= 1:
                    preferred = api.get_watcher_model_path()
                    candidates = [preferred]
                    if model is None:
                        candidates += [api.BEST_MODEL, api.LATEST_MODEL,
                                       str(root / 'checkpoints/watcher_recovery.zip')]
                    for candidate_path in dict.fromkeys(candidates):
                        candidate = api.get_model_signature(candidate_path)
                        if candidate is None or (candidate_path, candidate) == signature:
                            continue
                        if rejected_models.get(candidate_path) == candidate:
                            continue
                        try:
                            loaded = load_valid_policy(candidate_path, obs)
                            model = loaded
                            signature = (candidate_path, candidate)
                            active_model_path = candidate_path
                            print('MODEL:', Path(candidate_path).name, flush=True)
                            break
                        except Exception as exc:
                            rejected_models[candidate_path] = candidate
                            print('MODEL REJECTED:', Path(candidate_path).name, exc, flush=True)
                    learner_steps, champion_version, champion_steps = api.get_trainer_progress()
                    battle_prog = api.get_battle_progress()
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
                fps_now = time.monotonic()
                if fps_now - fps_started >= api.FPS_TITLE_INTERVAL:
                    measured_fps = fps_frames / (fps_now - fps_started)
                    fps_frames = 0
                    fps_started = fps_now
                    cv2.setWindowTitle(window, f"PKMai Watcher / FPS {measured_fps:.1f}")
                step_events = list(info.get('reward_events', []))
                record = {'time': time.time(), 'episode': episode, 'step': core.total_steps,
                          'action': int(action), 'reward': float(reward),
                          'episode_reward': float(core.current_reward), 'events': step_events,
                          'terminated': terminated, 'truncated': truncated,
                          'reason': info.get('last_stage_timeout'),
                          'in_battle': core.last_in_battle, 'battle_detection': info.get('battle_detection'),
                          'battle_type_flags': info.get('battle_type_flags'),
                          'engine': 'PokemonFireRedEnv.step', 'scope': 'isolated_evaluation'}
                logger.info(json.dumps(record))
                if step_events or terminated or truncated:
                    line = f"E{episode} S{core.total_steps} {reward:+.4f} | " + ', '.join(step_events)
                    events.extend((core.total_steps, ev) for ev in step_events)
                    events = events[-60:]
                    print(line, flush=True)
                elif core.total_steps % 100 == 0:
                    print(f"E{episode} S{core.total_steps} | reward={reward:+.4f} total={core.current_reward:+.2f}", flush=True)
                if start - last_publish >= api.TELEMETRY_INTERVAL or terminated or truncated:
                    data = telemetry(core, info, reward, episode, Path(active_model_path).stem, api.get_latest_version())
                    data['last_reset'] = last_reset
                    data['fps'] = round(measured_fps, 1)
                    data['target_fps'] = api.TARGET_FPS
                    data['learner_steps'] = learner_steps
                    data['champion_version'] = champion_version
                    data['champion_steps'] = champion_steps
                    data['battle_fights'] = battle_prog['fights']
                    data['battle_champion_version'] = battle_prog['champion_version']
                    data['battle_learner_version'] = battle_prog['learner_version']
                    atomic_json(root / 'instances_data/inst_120.json', data)
                    screen = core.env.get_screen()
                    canvas = render_console(screen, data, events)
                    tiles = {tuple(e[:2]) + tuple(e[2:4]) for e in core.persistent_known_edges}
                    tiles |= {tuple(e[:2]) + tuple(e[4:6]) for e in core.persistent_known_edges}
                    api.save_watcher_mapping(tiles, core.persistent_known_edges,
                                             core.persistent_known_maps, core.persistent_known_transitions)
                    last_publish = start
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if terminated or truncated:
                    last_reset = info.get('last_stage_timeout') or 'episode limit / objective complete'
                    print('RESET:', last_reset, flush=True)
                    obs, info = env.reset()
                    episode += 1
    finally:
        core.frame_callback = None
        publisher.close()
        logger.removeHandler(handler)
        handler.close()
        env.close()
        cv2.destroyAllWindows()
