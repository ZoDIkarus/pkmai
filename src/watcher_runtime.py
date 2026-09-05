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
    return module.PokemonFireRedEnv(
        rank=0, n_envs=n_envs,
        shared_edges=dict.fromkeys(edges, 1), shared_maps=dict.fromkeys(maps, 1),
        shared_transitions=dict.fromkeys(transitions, 1),
        shared_progress={'max_world_stage': 0}, shared_species={},
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


def render_console(screen, data, events):
    """Neutral charcoal UI; game pixels retain their original colors."""
    canvas = np.full((680, 1200, 3), 24, dtype=np.uint8)
    canvas[64:544, :720] = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR),
                                    (720, 480), interpolation=cv2.INTER_NEAREST)
    def label(text, x, y, color=(205, 205, 205), scale=.48):
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    label('ALEX / LIVE WATCHER', 20, 28, (210, 225, 210), .65)
    label(f"FPS {data.get('fps', 0):.1f} / {data.get('target_fps', 300):.0f} target", 420, 28, (205, 205, 205), .55)
    label(f"{data['model_name']} v{data['model_version']}  |  INFERENCE ONLY", 740, 28)
    label('TRAINER REWARD ENGINE / isolated evaluation', 740, 64)
    label(f"Episode {data['episode']}    Reward {data['reward']:+.2f}", 740, 96)
    label(f"Step {data['step_reward']:+.4f}    Stage {data['world_stage']}", 740, 122)
    label(f"Route {data['route_steps']}    Battle {data['battle_steps']}", 740, 148)
    label(('BATTLE' if data['in_battle'] else 'OVERWORLD') + ' / ' + str(data.get('battle_detection', '')), 740, 176, scale=.4)
    label('REWARD EVENTS', 740, 198, (170, 210, 170))
    for i, event in enumerate(events[-15:]):
        label(event[:62], 740, 214 + i * 26, scale=.4)
    label('Same actions, observations, rewards and episode rules as training.', 20, 578)
    label('No learning. No fleet bonus claims. Q / Esc to close.', 20, 608)
    label(f"Last reset: {data.get('last_reset', '-')}", 20, 638)
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
    cv2.resizeWindow(window, 1200, 680)
    model = None
    signature = None
    last_model_check = last_publish = 0
    fps_started = time.monotonic()
    fps_frames = 0
    measured_fps = 0.0
    episode = 1
    events = []
    last_reset = '-'
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
                    last_model_check = start
                if model is None:
                    cv2.waitKey(100)
                    continue
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(int(action))
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
                    events.append(line)
                    events = events[-40:]
                    print(line, flush=True)
                elif env.total_steps % 100 == 0:
                    print(f"E{episode} S{env.total_steps} | reward={reward:+.4f} total={env.current_reward:+.2f}", flush=True)
                if start - last_publish >= .1 or terminated or truncated:
                    data = telemetry(env, info, reward, episode, Path(path).stem, api.get_latest_version())
                    data['last_reset'] = last_reset
                    data['fps'] = round(measured_fps, 1)
                    data['target_fps'] = api.TARGET_FPS
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
