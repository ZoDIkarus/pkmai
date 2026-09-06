"""Frozen, paired policy evaluation. No optimizer calls or training-state writes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

SCHEMA = 'paired_arrival_v1'
EPISODES = 32
MAX_STEPS = 12000


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_signature():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'src'/name for name in (
        'pokemon_env.py', 'firered_ram.py', 'battle_state.py', 'loop_guard.py',
        'reward_state.py', 'champion_evaluation.py', 'watcher_runtime.py')]
    paths.append(root/'local/custom_integrations/PokemonFireRed-Gba/StartGame.state')
    return hashlib.sha256(''.join(digest(p) for p in paths).encode()).hexdigest()


def metrics(records):
    """Prefer reliably reached geography, success frequency and actual arrival time."""
    n = len(records)
    needed = max(1, math.ceil(n * .12))
    stages = sorted({int(r['stage']) for r in records}, reverse=True)
    stage = next((s for s in stages if sum(r['stage'] >= s for r in records) >= needed), 0)
    arrivals = [int(r['arrivals'][str(stage)]) for r in records
                if str(stage) in r.get('arrivals', {})]
    rates = {str(s): round(1000 * sum(str(s) in r.get('arrivals', {}) for r in records) / n)
             if n else 0 for s in range(2, 7)}
    return dict(episodes=n, full_episodes=n, max_stage=stage,
                max_badges=max((r.get('badges', 0) for r in records), default=0),
                stage_reach_permille=rates,
                stage_success_permille=round(1000 * sum(r['stage'] >= stage for r in records)/n) if n else 0,
                median_arrival_steps=int(statistics.median(arrivals)) if arrivals else MAX_STEPS+1,
                max_level=max((r.get('level', 0) for r in records), default=0),
                max_maps=max((r.get('maps', 0) for r in records), default=0),
                full_starter_permille=round(1000*sum(r.get('starter', 0) for r in records)/n) if n else 0,
                full_intro_permille=1000 if n else 0,
                full_stairs_permille=1000 if n else 0,
                full_exit_permille=round(1000*sum(r.get('starter', 0) for r in records)/n) if n else 0,
                evaluation_schema=SCHEMA)


def score(m):
    return (int(m.get('max_stage', 0)),
            *(int(m.get('stage_reach_permille', {}).get(str(s), 0)) for s in range(6, 1, -1)),
            int(m.get('stage_success_permille', 0)),
            -int(m.get('median_arrival_steps', MAX_STEPS+1)),
            int(m.get('max_badges', 0)), int(m.get('max_level', 0)))


def evaluate_trial(job_dir, label, seed):
    # Separate OS processes and runtime dirs keep both policy and novelty state isolated.
    import random
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from watcher_runtime import make_evaluation_env
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    job = Path(job_dir)
    result_file = job/f'{label}_{seed}.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    root = job/'trials'/f'{label}_{seed}'
    if root.exists():
        shutil.rmtree(root)  # Only this evaluator's incomplete private trial.
    env = make_evaluation_env(root)
    try:
        model = PPO.load(job/f'{label}.zip', device='cpu')
        obs, info = env.reset(seed=seed)
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        result = dict(label=label, seed=seed, stage=int(info.get('world_stage', 1)),
                      arrivals=dict(info.get('stage_arrival_steps', {})),
                      badges=int(info.get('badges_count', 0)), level=int(info.get('level', 0)),
                      maps=int(info.get('visited_maps', 0)),
                      starter=int(bool(info.get('has_target_starter')) and env.starter_outdoor_rewarded),
                      steps=step+1)
        atomic_json(result_file, result)
        return result
    finally:
        env.close()


def run_job(job_dir, workers=4):
    job = Path(job_dir)
    manifest = json.loads((job/'manifest.json').read_text())
    signature = environment_signature()
    if manifest['environment'] != signature:
        raise RuntimeError('Environment changed; do not mix evaluation conditions')
    for label in ('champion', 'candidate'):
        if digest(job/f'{label}.zip') != manifest[label+'_sha']:
            raise RuntimeError('Frozen model changed')
    atomic_json(job/'progress.json', dict(status='running',completed=0,total=2*EPISODES,pid=os.getpid()))
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate_trial,str(job),label,seed)
                     for seed in range(EPISODES) for label in ('champion','candidate')]
            for future in as_completed(futures):
                records.append(future.result())
                atomic_json(job/'progress.json',dict(status='running',completed=len(records),total=2*EPISODES,pid=os.getpid()))
        if environment_signature() != signature:
            raise RuntimeError('Environment changed during evaluation')
        report=dict(manifest=manifest,
                    champion=metrics([r for r in records if r['label']=='champion']),
                    candidate=metrics([r for r in records if r['label']=='candidate']))
        report['promote']=score(report['candidate']) > score(report['champion'])
        atomic_json(job/'result.json',report)
        atomic_json(job/'progress.json',dict(status='complete',completed=len(records),total=2*EPISODES))
    except Exception as exc:
        atomic_json(job/'progress.json',dict(status='failed',error=str(exc),completed=len(records),total=2*EPISODES))
        raise


class EvaluationQueue:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.pointer=self.root/'active.json'
        self.process=None

    def active(self):
        if not self.pointer.exists():return None
        return self.root/json.loads(self.pointer.read_text())['job']

    def status(self):
        job=self.active()
        if job is None:return dict(status='idle')
        path=job/'progress.json'
        return json.loads(path.read_text()) if path.exists() else dict(status='starting')

    def launch(self, job):
        log=open(job/'runner.log','a')
        try:
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',str(job)],
                                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:log.close()
        atomic_json(job/'progress.json',dict(status='starting',pid=self.process.pid,completed=0,total=64))

    def ensure_running(self):
        job=self.active()
        if job is None or (job/'result.json').exists():return
        status=self.status()
        if status.get('status')=='failed':return
        pid=status.get('pid')
        if pid:
            try:os.kill(int(pid),0);return
            except ProcessLookupError:pass
        self.launch(job)

    def schedule(self, model, champion_path, steps):
        if self.active() is not None:
            self.ensure_running();return
        job=self.root/f'pair_{int(steps)}_{time.time_ns()}'
        job.mkdir()
        model.save(job/'candidate.zip')
        shutil.copy2(champion_path,job/'champion.zip')
        atomic_json(job/'manifest.json',dict(schema=SCHEMA,steps=int(steps),environment=environment_signature(),
                                           champion_sha=digest(job/'champion.zip'),candidate_sha=digest(job/'candidate.zip'),
                                           seeds=list(range(EPISODES)),max_steps=MAX_STEPS))
        atomic_json(self.pointer,dict(job=job.name))
        self.launch(job)

    def result(self):
        job=self.active()
        if job and (job/'result.json').exists():return job,json.loads((job/'result.json').read_text())
        return None

    def acknowledge(self):
        self.pointer.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();run_job(args.job)
