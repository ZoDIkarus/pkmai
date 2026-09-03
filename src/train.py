import os
import json
import torch
import multiprocessing as mp
import shutil
import resource
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from pokemon_env import PokemonFireRedEnv


# ================================================================
# USER CONFIG / TRAINING TUNING
# ================================================================
# Die wichtigsten Werte stehen absichtlich hier oben.

# 120 parallele Envs (Mac). Fuer kleinere Rechner runtersetzen - dann aber
# auch die "% 120" Slot-Logik in pokemon_env._agent_role anpassen.
NUM_ENVS = 120

# Endlos-Training: laeuft in Bloecken weiter, bis du Ctrl+C drueckst.
# TRAIN_CHUNK_TIMESTEPS ist nur die Groesse eines learn()-Blocks.
TRAIN_FOREVER = True
TRAIN_CHUNK_TIMESTEPS = 1_000_000

# Nur benutzt, wenn TRAIN_FOREVER = False.
TOTAL_TIMESTEPS = 100_000_000

SAVE_EVERY_TIMESTEPS = 25_000

# PPO
LEARNING_RATE = 7.5e-05
PPO_N_STEPS = 32
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_GAMMA = 0.995
# V10.28: Nach 19 Mio. Steps ist max_episode_maps global immer noch bei 5 -
# die Policy ist zu deterministisch geworden, um zufaellig neues Terrain zu
# entdecken, obwohl echte neue Tiefe stark belohnt wird (NEW_GLOBAL_DEPTH_
# REWARD=300, einmalig, nicht farmbar). Entropie temporaer angehoben, um
# wieder mehr Aktionsvariation/Exploration zuzulassen ("Entropy Re-Heat").
# Sobald neue Depth-Records in exploration_memory/global_progress.json
# auftauchen, kann der Wert wieder Richtung 0.008-0.012 zurückgefahren werden.
# V11: kraeftiges Entropy-Re-Heat. Die alte Policy ist nach 25 Mio Steps
# auf "am Alabastia-Rand kreisen" festgefahren. Zusammen mit der neuen
# Reward-Logik (Exploration gratis + dominant, keine Straf-Suppe) soll die
# hohe Entropie sie da rausschiessen, OHNE die fruehen Skills zu verlernen
# (deren Navigations-Ziel bleibt ja gleich). Spaeter wieder Richtung 0.015.
PPO_ENT_COEF = 0.05

# "auto" = MPS auf Apple Silicon wenn verfuegbar, sonst CPU.
# Alternativ: "cpu" oder "mps"
TRAIN_DEVICE = "auto"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
BASE_DIR = PROJECT_ROOT
EXPLORATION_MEMORY_DIR = os.path.join(RUNTIME_DIR, "exploration_memory")
CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_states")
SHARED_CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_shared")

# ================================================================
# INTERNAL PATHS - normalerweise nicht aendern
# ================================================================
MODEL_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
LATEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_latest.zip")
BEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_best.zip")
CANDIDATE_MODEL = os.path.join(MODEL_DIR, "pokemon_model_candidate.zip")
RESUME_MODEL = os.path.join(MODEL_DIR, "pokemon_model_resume.zip")
CHAMPION_FILE = os.path.join(RUNTIME_DIR, "champion_score.json")
TRAINER_STATUS_FILE = os.path.join(RUNTIME_DIR, "trainer_status.json")
VERSION_FILE = os.path.join(RUNTIME_DIR, "model_version.json")
SKILL_SCORE_FILE = os.path.join(RUNTIME_DIR, "skill_vault_scores.json")
SKILL_MODELS = {
    "intro": os.path.join(MODEL_DIR, "pokemon_skill_intro_best.zip"),
    "stairs": os.path.join(MODEL_DIR, "pokemon_skill_stairs_best.zip"),
    "exit": os.path.join(MODEL_DIR, "pokemon_skill_exit_best.zip"),
    "starter": os.path.join(MODEL_DIR, "pokemon_skill_starter_best.zip"),
    "progress": os.path.join(MODEL_DIR, "pokemon_skill_progress_best.zip"),
}

os.makedirs(MODEL_DIR, exist_ok=True)


def seed_shared_curriculum():
    os.makedirs(SHARED_CURRICULUM_DIR, exist_ok=True)

    seeded = []

    if not os.path.isdir(CURRICULUM_DIR):
        return seeded

    # Fruehe Story-States zuerst; spaetere States duerfen ebenfalls geteilt
    # werden, sobald sie existieren.
    candidates = {}

    for root, _, files in os.walk(CURRICULUM_DIR):
        if os.path.abspath(root).startswith(
            os.path.abspath(SHARED_CURRICULUM_DIR)
        ):
            continue

        for name in files:
            if not name.endswith(".state.gz"):
                continue
            milestone = name[:-9]
            candidates.setdefault(
                milestone,
                os.path.join(root, name)
            )

    for milestone, source in sorted(candidates.items()):
        target = os.path.join(
            SHARED_CURRICULUM_DIR,
            f"{milestone}.state.gz"
        )
        if os.path.exists(target):
            continue
        try:
            shutil.copy2(source, target)
            seeded.append(milestone)
        except Exception:
            pass

    return seeded


def load_global_exploration():
    edges = {}
    maps = {}
    transitions = {}

    if not os.path.isdir(EXPLORATION_MEMORY_DIR):
        return edges, maps, transitions

    for name in os.listdir(EXPLORATION_MEMORY_DIR):
        if not name.startswith("agent_") or not name.endswith(".json"):
            continue

        path = os.path.join(EXPLORATION_MEMORY_DIR, name)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for x in data.get("edges", []):
                if isinstance(x, list) and len(x) == 6:
                    edges[tuple(x)] = 1

            for x in data.get("maps", []):
                if isinstance(x, list) and len(x) == 2:
                    maps[tuple(x)] = 1

            for x in data.get("transitions", []):
                if isinstance(x, list) and len(x) == 8:
                    transitions[tuple(x)] = 1

        except Exception:
            pass

    return edges, maps, transitions


def make_env(
    rank,
    shared_edges,
    shared_maps,
    shared_transitions,
    shared_progress,
    shared_lock,
):
    def _init():
        return PokemonFireRedEnv(
            rank=rank,
            shared_edges=shared_edges,
            shared_maps=shared_maps,
            shared_transitions=shared_transitions,
            shared_progress=shared_progress,
            shared_lock=shared_lock,
        )
    return _init


class MilestoneCheckpointCallback(BaseCallback):
    """V10.27: sequential protected-skill bootcamp."""

    def __init__(self, check_freq=15_000, verbose=1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.last_check_step = 0
        self.last_resume_save_step = 0
        self.resume_save_freq = 50_000
        self.version = 1
        self.recent = deque(maxlen=600)
        self.recent_full = deque(maxlen=256)
        self.full_live = {}
        self.min_eval_episodes = 24
        # 12 abgeschlossene Beginning-Full-Runs pro Eval: genug Signal fuer die
        # Regressions-Pruefung, ohne die Eval-Kadenz unnoetig zu verlangsamen.
        self.min_full_episodes = 12
        self.champion_score = None
        self.champion_metrics = {}
        self.rollback_count = 0
        self.regression_strikes = 0
        self.last_eval_metrics = {}
        self.skill_scores = {}
        # Nur Telemetrie. V10.28.1: der zeitbasierte Stale-Champion-Fallback
        # wurde entfernt - er konnte einen funktionierenden Champion durch
        # eine fruehgame-vergessliche Policy ersetzen, nur weil Zeit verging.
        self.steps_since_champion_update = 0
        self.champion_published_at_step = 0

        # V10.9:
        # Letzten bekannten Episode-Zustand jedes VecEnv-Slots
        # behalten. Dadurch verlieren wir beim SB3 Auto-Reset
        # keine Full-Episode.
        self.last_episode_info = {}

        if os.path.exists(VERSION_FILE):
            try:
                with open(VERSION_FILE, "r") as f:
                    self.version = int(json.load(f).get("version", 0)) + 1
            except Exception:
                self.version = 1

        if os.path.exists(CHAMPION_FILE):
            try:
                with open(CHAMPION_FILE, "r") as f:
                    data = json.load(f) or {}
                raw = data.get("score")
                if isinstance(raw, list):
                    self.champion_score = tuple(int(x) for x in raw)
                self.champion_metrics = dict(data.get("metrics") or {})
                self.champion_published_at_step = int(
                    data.get("timesteps", 0) or 0
                )
            except Exception:
                pass

        if self.champion_score is None:
            self._seed_baseline_from_training_stats()

        self._load_skill_scores()

    def _seed_baseline_from_training_stats(self):
        stats_dir = os.path.join(RUNTIME_DIR, "training_stats")
        totals = {}
        try:
            names = os.listdir(stats_dir)
        except Exception:
            names = []

        for name in names:
            if not name.startswith("agent_") or not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(stats_dir, name), "r") as f:
                    d = json.load(f) or {}
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        totals[k] = totals.get(k, 0) + v
            except Exception:
                pass

        def permille(success, episodes):
            e = int(totals.get(episodes, 0))
            if e <= 0:
                return 0
            return round(1000 * int(totals.get(success, 0)) / e)

        full_ep = int(totals.get("v2_full_episodes", 0))
        def full_rate(key):
            return round(1000 * int(totals.get(key, 0)) / full_ep) if full_ep else 0

        self.champion_metrics = {
            "episodes": int(totals.get("all_episodes", 0)),
            "full_episodes": full_ep,
            "full_intro_permille": full_rate("v2_full_intro"),
            "full_stairs_permille": full_rate("v2_full_stairs"),
            "full_exit_permille": full_rate("v2_full_left_house"),
            "full_starter_permille": full_rate("v2_full_starter"),
            "stairs_skill_permille": permille("v2_stairs_success", "v2_stairs_episodes"),
            "exit_skill_permille": permille("v2_exit_success", "v2_exit_episodes"),
            "starter_skill_permille": permille("v8_starter_success", "v8_starter_episodes"),
            "badge_skill_permille": permille("v8_badge_success", "v8_badge_episodes"),
            "badge_episodes": int(totals.get("v7_full_badge1", 0)),
            "max_badges": 1 if int(totals.get("v7_full_badge1", 0)) > 0 else 0,
            "max_maps": 0,
        }
        self.champion_score = self._score(self.champion_metrics)
        self._write_champion_score(self.champion_score, self.champion_metrics)

        print(
            "🛡️ Champion-Baseline: "
            f"Full Intro={self.champion_metrics['full_intro_permille']/10:.1f}% | "
            f"Full Treppe={self.champion_metrics['full_stairs_permille']/10:.1f}% | "
            f"Full Exit={self.champion_metrics['full_exit_permille']/10:.1f}%"
        )

    def _load_skill_scores(self):
        champion = self.champion_metrics or {}
        defaults = {
            "intro": int(champion.get("full_intro_permille", 0)),
            "stairs": int(champion.get("full_stairs_permille", 0)),
            "exit": int(champion.get("full_exit_permille", 0)),
            "starter": int(champion.get("full_starter_permille", 0)),
            # Progress-Runs beginnen aus einem Curriculum-State; ihre lokale
            # Map-Tiefe ist nicht mit der alten Full-Mapzahl vergleichbar.
            "progress": 0,
        }
        try:
            with open(SKILL_SCORE_FILE, "r") as f:
                loaded = json.load(f) or {}
            for key in defaults:
                if key in loaded:
                    defaults[key] = int(loaded[key])
        except Exception:
            pass
        self.skill_scores = defaults
        self._save_skill_scores()

    def _save_skill_scores(self):
        try:
            tmp = SKILL_SCORE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.skill_scores, f, separators=(",", ":"))
            os.replace(tmp, SKILL_SCORE_FILE)
        except Exception:
            pass

    def _update_skill_vault(self):
        """Keep the best whole-policy snapshot for each routed story stage."""
        rows = list(self.recent)
        candidates = {}

        for role, key in (
            ("intro", "intro"),
            ("stairs", "stairs"),
            ("exit", "exit"),
            ("starter", "starter"),
        ):
            group = [r for r in rows if r["role"] == role]
            if len(group) < 6:
                continue
            candidates[role] = round(
                1000 * sum(int(r[key]) for r in group) / len(group)
            )

        progress = [r for r in rows if r["role"] == "progress"]
        if len(progress) >= 6:
            candidates["progress"] = max(
                int(r.get("badges", 0)) * 10000
                + int(r.get("maps", 0)) * 100
                + int(r.get("level", 0))
                for r in progress
            )

        changed = False
        for skill, score in candidates.items():
            old = int(self.skill_scores.get(skill, 0))
            if score <= old:
                continue
            target = SKILL_MODELS[skill]
            temp_base = target[:-4] + "_tmp"
            temp_zip = temp_base + ".zip"
            self.model.save(temp_base)
            os.replace(temp_zip, target)
            self.skill_scores[skill] = int(score)
            changed = True
            print(
                f"🧰 SKILL VAULT: {skill} {old} -> {score} "
                f"bei {int(self.num_timesteps):,} Steps"
            )

        if changed:
            self._save_skill_scores()

    @staticmethod
    def _badge_count(info):
        raw = int(info.get("badges", 0))
        return bin(raw).count("1") if raw > 0 else 0

    def _full_stage_rank(self, info):
        if str(info.get("training_objective", "")) != "full":
            return 0
        if str(info.get("episode_start", "")) != "beginning":
            return 0

        badges = int(info.get("badges_count", 0) or 0)
        level = int(info.get("level", info.get("p1_level", 0)) or 0)
        stage = str(info.get("story_stage", "INTRO"))

        if badges >= 1:
            return 4
        if level >= 5 or bool(info.get("has_starter", False)):
            return 3
        if stage == "OUTDOOR":
            return 2
        if stage == "F1_TO_EXIT":
            return 1
        return 0

    def _champion_full_stage_rank(self):
        m = self.champion_metrics or {}
        if int(m.get("max_badges", 0)) >= 1:
            return 4
        if int(m.get("full_starter_permille", 0)) > 0:
            return 3
        if int(m.get("full_exit_permille", 0)) > 0:
            return 2
        if int(m.get("full_stairs_permille", 0)) > 0:
            return 1
        return 0

    def _episode_record(self, info):
        role = str(info.get("training_objective", ""))
        stage = str(info.get("story_stage", "INTRO"))
        start = str(info.get("episode_start", ""))
        level = int(
            info.get(
                "level",
                info.get("p1_level", 0)
            ) or 0
        )
        badges = int(
            info.get(
                "badges_count",
                self._badge_count(info)
            ) or 0
        )
        return {
            "role": role,
            "start": start,
            "intro": int(stage != "INTRO"),
            "stairs": int(stage in ("F1_TO_EXIT", "OUTDOOR")),
            "exit": int(stage == "OUTDOOR"),
            "starter": int(
                level >= 5 or bool(info.get("has_starter", False))
            ),
            "badge": int(badges >= 1),
            "badges": badges,
            "maps": int(info.get("visited_maps", 0)),
            "level": level,
        }

    @staticmethod
    def _score(m):
        return (
            int(m.get("max_badges", 0)),
            int(m.get("badge_episodes", 0)),
            int(m.get("full_starter_permille", 0)),
            int(m.get("full_exit_permille", 0)),
            int(m.get("full_stairs_permille", 0)),
            int(m.get("full_intro_permille", 0)),
            int(m.get("starter_skill_permille", 0)),
            int(m.get("exit_skill_permille", 0)),
            int(m.get("stairs_skill_permille", 0)),
            int(m.get("max_maps", 0)),
        )

    def _metrics(self):
        rows = list(self.recent)
        # full_live enthaelt nur momentane Zwischenstaende aktiver Episoden.
        # Das erzeugte bisher falsche 0%-Regressionen. Vergleichbar mit dem
        # Champion sind nur abgeschlossene Full-Runs ab "beginning".
        full = list(self.recent_full)

        def group(role):
            return [r for r in rows if r["role"] == role]

        def rate(g, key):
            if not g:
                return 0
            return round(1000 * sum(r[key] for r in g) / len(g))

        return {
            "episodes": len(rows),
            "full_episodes": len(full),
            "full_intro_permille": rate(full, "intro"),
            "full_stairs_permille": rate(full, "stairs"),
            "full_exit_permille": rate(full, "exit"),
            "full_starter_permille": rate(full, "starter"),
            "stairs_skill_permille": rate(group("stairs"), "stairs"),
            "exit_skill_permille": rate(group("exit"), "exit"),
            "starter_skill_permille": rate(group("starter"), "starter"),
            "badge_skill_permille": rate(group("badge"), "badge"),
            "badge_episodes": sum(r["badge"] for r in rows),
            "max_badges": max((r["badges"] for r in rows), default=0),
            "max_maps": max((r["maps"] for r in rows), default=0),
        }

    def _metrics_floor(self, metrics):
        """Direkt nach Neustart ist recent_full leer -> alle full_*_permille
        aus _metrics() sind 0. Ein Frontier-/Milestone-Publish in diesem
        Moment wuerde sonst die bekannten guten Champion-Raten mit 0
        ueberschreiben und den Regressions-Schutz aushebeln. Deshalb die
        alten Champion-Werte als Untergrenze behalten, solange noch keine
        echten Beginning-Full-Runs abgeschlossen sind."""
        if int(metrics.get("full_episodes", 0)) > 0:
            return metrics
        old = self.champion_metrics or {}
        for k in ("full_intro_permille", "full_stairs_permille",
                  "full_exit_permille", "full_starter_permille"):
            metrics[k] = max(int(metrics.get(k, 0)), int(old.get(k, 0)))
        return metrics

    def _write_trainer_status(self):
        try:
            champion_steps = 0
            champion_version = 0
            if os.path.exists(CHAMPION_FILE):
                try:
                    with open(CHAMPION_FILE, "r") as f:
                        c = json.load(f) or {}
                    champion_steps = int(c.get("timesteps", 0) or 0)
                    champion_version = int(c.get("version", 0) or 0)
                except Exception:
                    pass
            payload = {
                "learner_steps": int(self.num_timesteps),
                "champion_steps": champion_steps,
                "champion_version": champion_version,
                "delta_steps": int(self.num_timesteps) - champion_steps,
                "recent_full_done": len(getattr(self, "recent_full", [])),
                "mode": "v11_clean_explore",
                "rollback_count": int(self.rollback_count),
                "regression_strikes": int(self.regression_strikes),
                "skill_scores": dict(self.skill_scores),
                "last_eval_metrics": dict(self.last_eval_metrics),
                "training_phase": (
                    "1_intro" if int(self.skill_scores.get("intro", 1000)) < 880
                    else "2_stairs" if int(self.skill_scores.get("stairs", 1000)) < 880
                    else "3_exit" if int(self.skill_scores.get("exit", 1000)) < 880
                    else "4_starter" if int(self.skill_scores.get("starter", 1000)) < 880
                    else "5_world_explore"
                ),
            }
            tmp = TRAINER_STATUS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp, TRAINER_STATUS_FILE)
        except Exception:
            pass

    def _write_champion_score(self, score, metrics):
        tmp = CHAMPION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "score": list(score),
                "metrics": metrics,
                "timesteps": int(self.num_timesteps),
                "version": int(self.version),
            }, f)
        os.replace(tmp, CHAMPION_FILE)

    def _publish_champion(self, score, metrics, reason):
        self.model.save(BEST_MODEL)
        self.model.save(LATEST_MODEL)
        self.champion_score = tuple(score)
        self.champion_metrics = dict(metrics)
        self._write_champion_score(score, metrics)
        self.steps_since_champion_update = 0
        self.champion_published_at_step = int(self.num_timesteps)

        tmp = VERSION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "version": self.version,
                "timesteps": int(self.num_timesteps),
                "champion": True,
                "metrics": metrics,
            }, f)
        os.replace(tmp, VERSION_FILE)

        print(
            f"🏆 CHAMPION v{self.version:06d} [{reason}] | "
            f"Full Intro {metrics.get('full_intro_permille',0)/10:.1f}% | "
            f"Treppe {metrics.get('full_stairs_permille',0)/10:.1f}% | "
            f"Exit {metrics.get('full_exit_permille',0)/10:.1f}% | "
            f"Starter {metrics.get('full_starter_permille',0)/10:.1f}%"
        )
        self.version += 1
        # Keine Episoden verschiedener Modellversionen vermischen.
        self.recent.clear()
        self.recent_full.clear()
        self.full_live.clear()

    def _protected_regression(self, candidate):
        old = self.champion_metrics or {}

        # Echte neue Tiefe (mehr Orden / Maps / Level in einem Beginning-Full-Run)
        # ist immer ein Fortschritt und hebt jeden Schutz auf.
        if int(candidate.get("max_badges", 0)) > int(old.get("max_badges", 0)):
            return False
        if int(candidate.get("max_maps", 0)) > int(old.get("max_maps", 0)):
            return False
        if int(candidate.get("max_level", 0)) > int(old.get("max_level", 0)):
            return False

        n = int(candidate.get("full_episodes", 0))
        if n < 8:
            # Zu wenig abgeschlossene Full-Runs -> nicht befoerdern, aber auch
            # nicht als harte Regression zaehlen.
            return True

        # WICHTIG: full_stairs_permille / full_exit_permille messen nur die
        # Endposition der Episode (stage in F1_TO_EXIT/OUTDOOR) und sind
        # strukturell nahe 0, sobald ein Run entweder frueher scheitert oder
        # weiter kommt. Der einzige verlaessliche kumulative Full-Indikator ist
        # der Starter (level>=5 oder has_starter) sowie die Tiefe oben.
        old_starter = int(old.get("full_starter_permille", 0))
        new_starter = int(candidate.get("full_starter_permille", 0))
        if old_starter >= 50 and new_starter < max(old_starter * 0.5,
                                                   old_starter - 200):
            print(
                f"🛡️ Champion geschuetzt: Full-Starter "
                f"{old_starter/10:.1f}% -> {new_starter/10:.1f}%"
            )
            return True

        old_intro = int(old.get("full_intro_permille", 0))
        new_intro = int(candidate.get("full_intro_permille", 0))
        if old_intro >= 300 and new_intro < old_intro - 250:
            print(
                f"🛡️ Champion geschuetzt: Full-Intro "
                f"{old_intro/10:.1f}% -> {new_intro/10:.1f}%"
            )
            return True

        return False

    def _evaluate(self):
        metrics = self._metrics()
        if metrics["episodes"] < self.min_eval_episodes or metrics["full_episodes"] < self.min_full_episodes:
            return False

        self.last_eval_metrics = dict(metrics)
        score = self._score(metrics)
        self.model.save(CANDIDATE_MODEL)

        if self._protected_regression(metrics):
            self.regression_strikes += 1
            self.steps_since_champion_update += int(metrics["full_episodes"])
            print(
                f"⚠️ Full-Regression erkannt "
                f"(Messung {self.regression_strikes}). "
                "Champion und Skill Vault bleiben erhalten; "
                "Learner lernt ohne Gewichtsverlust weiter."
            )
        elif self.champion_score is None or score >= self.champion_score:
            self.regression_strikes = 0
            self._publish_champion(score, metrics, "Recent-Eval")
        else:
            self.regression_strikes = 0
            self.steps_since_champion_update += int(metrics["full_episodes"])
            print(
                "🧪 Candidate noch nicht Champion, aber ohne harte Regression "
                "-> Training innerhalb der Schutzgrenze geht weiter."
            )

        self.recent.clear()
        self.recent_full.clear()
        return True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # V10.9.4 LIVE FULL SNAPSHOT:
        # Each Full-from-start VecEnv slot contributes one current probe.
        for i, live_info in enumerate(infos):
            if not isinstance(live_info, dict):
                continue
            live_policy = str(live_info.get("policy_objective", live_info.get("training_objective", "")))
            if live_policy != "full":
                continue
            if str(live_info.get("episode_start", "")) != "beginning":
                continue
            self.full_live[i] = self._episode_record(live_info)


        # V10.7 FRONTIER-CHAMPION:
        # New real Full-from-start story depth is protected immediately.
        champion = self.champion_metrics or {}
        champion_key = (
            int(champion.get("max_badges", 0)),
            1 if int(champion.get("full_starter_permille", 0)) > 0 else 0,
            1 if int(champion.get("full_exit_permille", 0)) > 0 else 0,
            1 if int(champion.get("full_stairs_permille", 0)) > 0 else 0,
            int(champion.get("max_maps", 0)),
            int(champion.get("max_level", 0)),
        )

        for info in infos:
            if not isinstance(info, dict):
                continue
            policy_obj = str(info.get("policy_objective", info.get("training_objective", "")))
            if policy_obj != "full":
                continue
            if str(info.get("episode_start", "")) != "beginning":
                continue

            stage = str(info.get("story_stage", "INTRO"))
            badges = int(info.get("badges_count", 0) or 0)
            level = int(info.get("level", info.get("p1_level", 0)) or 0)
            maps = int(info.get("frontier_maps", info.get("visited_maps", 0)) or 0)
            starter = int(bool(info.get("has_starter", False)) or level >= 5)
            stairs = int(stage in ("F1_TO_EXIT", "OUTDOOR"))
            exit_done = int(stage == "OUTDOOR")

            key = (badges, starter, exit_done, stairs, maps, level)
            if key <= champion_key:
                continue

            metrics = self._metrics_floor(self._metrics())
            metrics["max_badges"] = max(int(metrics.get("max_badges", 0)), badges)
            metrics["max_maps"] = max(int(metrics.get("max_maps", 0)), maps)
            metrics["max_level"] = max(int(metrics.get("max_level", 0)), level)
            if stairs:
                metrics["full_stairs_permille"] = max(int(metrics.get("full_stairs_permille", 0)), 1)
            if exit_done:
                metrics["full_exit_permille"] = max(int(metrics.get("full_exit_permille", 0)), 1)
            if starter:
                metrics["full_starter_permille"] = max(int(metrics.get("full_starter_permille", 0)), 1)

            self._publish_champion(
                self._score(metrics),
                metrics,
                "V10.7 FRONTIER-CHAMPION "
                f"badge={badges} starter={starter} exit={exit_done} "
                f"stairs={stairs} maps={maps} lvl={level}",
            )
            self.recent.clear()
            champion_key = key
            break

        current_best_stage = self._champion_full_stage_rank()
        stage_names = {1:"STAIRS", 2:"EXIT", 3:"STARTER", 4:"BADGE1"}

        for info in infos:
            if not isinstance(info, dict):
                continue
            rank = self._full_stage_rank(info)
            if rank > current_best_stage:
                metrics = self._metrics_floor(self._metrics())
                if rank >= 1:
                    metrics["full_stairs_permille"] = max(int(metrics.get("full_stairs_permille", 0)), 1)
                if rank >= 2:
                    metrics["full_exit_permille"] = max(int(metrics.get("full_exit_permille", 0)), 1)
                if rank >= 3:
                    metrics["full_starter_permille"] = max(int(metrics.get("full_starter_permille", 0)), 1)
                if rank >= 4:
                    metrics["max_badges"] = max(int(metrics.get("max_badges", 0)), 1)

                self._publish_champion(
                    self._score(metrics),
                    metrics,
                    f"FULL-MILESTONE {stage_names.get(rank, rank)}",
                )
                self.recent.clear()
                current_best_stage = rank
                break
        dones = self.locals.get("dones", [])

        # V10.9 FULL-EVAL FIX:
        # Vor dem Auswerten jedes Slots dessen letzten
        # brauchbaren Episode-Zustand sichern.
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            role = str(info.get("training_objective", ""))
            start = str(info.get("episode_start", ""))

            if role and start:
                self.last_episode_info[i] = dict(info)

        if dones is not None:
            for i, done in enumerate(dones):
                if not done:
                    continue

                terminal_info = None

                if i < len(infos) and isinstance(infos[i], dict):
                    terminal_info = dict(infos[i])

                cached_info = self.last_episode_info.get(i)

                # Falls der terminale Info-Satz durch VecEnv-
                # Auto-Reset unvollstaendig geworden ist,
                # verwenden wir den letzten sicheren Zustand.
                if cached_info:
                    if not terminal_info:
                        terminal_info = dict(cached_info)
                    else:
                        role = str(
                            terminal_info.get(
                                "training_objective", ""
                            )
                        )
                        start = str(
                            terminal_info.get(
                                "episode_start", ""
                            )
                        )

                        if not role or not start:
                            terminal_info = dict(cached_info)

                if isinstance(terminal_info, dict):

                    # TEMP V10.9 DEBUG:
                    # Zeige bei den Slots 96-119 exakt, was SB3
                    # beim Episode-Ende an den Callback liefert.
                    if 82 <= i <= 119:
                        print(
                            "🔬 FULL-SLOT DONE "
                            f"slot={i} | "
                            f"role={terminal_info.get('training_objective')!r} | "
                            f"start={terminal_info.get('episode_start')!r} | "
                            f"stage={terminal_info.get('story_stage')!r} | "
                            f"steps={terminal_info.get('episode_steps')!r} | "
                            f"level={terminal_info.get('level', terminal_info.get('p1_level'))!r} | "
                            f"terminated={terminal_info.get('terminated')!r} | "
                            f"truncated={terminal_info.get('TimeLimit.truncated')!r}"
                        )

                    record = self._episode_record(
                        terminal_info
                    )
                    self.recent.append(record)

                    if (
                        record["role"] == "full"
                        and record["start"] == "beginning"
                    ):
                        self.recent_full.append(record)

                    if record["role"] == "full":
                        print(
                            "🧬 FULL EPISODE: "
                            f"start={record['start']} | "
                            f"intro={record['intro']} | "
                            f"stairs={record['stairs']} | "
                            f"exit={record['exit']} | "
                            f"starter={record['starter']} | "
                            f"badge={record['badge']}"
                        )

                self.last_episode_info.pop(i, None)

        for info in infos:
            if not isinstance(info, dict):
                continue
            badges = self._badge_count(info)
            if badges > int(self.champion_metrics.get("max_badges", 0)):
                metrics = self._metrics()
                metrics["max_badges"] = badges
                metrics["badge_episodes"] = max(1, int(metrics.get("badge_episodes", 0)))
                self._publish_champion(self._score(metrics), metrics, f"Orden {badges}")
                self.recent.clear()
                return True

        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self.last_check_step = int(self.num_timesteps)
            self._update_skill_vault()
            if not self._evaluate():
                full_done_n = len(self.recent_full)
                print(
                    f"🧪 Champion-Eval: {len(self.recent)}/{self.min_eval_episodes} Episoden | "
                    f"Beginning-FullDone {full_done_n}/{self.min_full_episodes}"
                )
        # V10.9.4 AUTO-RESUME SAVE
        if self.num_timesteps - self.last_resume_save_step >= self.resume_save_freq:
            self.last_resume_save_step = int(self.num_timesteps)
            self.model.save(RESUME_MODEL)
            print(f"💾 Resume-Stand gespeichert bei {self.num_timesteps:,} Steps")

        # V10.10.1 LIVE TRAINER STATUS
        self._write_trainer_status()

        return True

    def final_candidate_save(self):
        self.model.save(CANDIDATE_MODEL)
        self.model.save(RESUME_MODEL)
        print(f"💾 Resume-Modell gespeichert bei {self.num_timesteps:,} Steps")

def _raise_nofile_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 65536
        target = max(soft, min(int(target), 65536))
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        now_soft, now_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"🧰 NOFILE soft/hard: {now_soft}/{now_hard}")
    except Exception as e:
        print(f"⚠️ NOFILE konnte nicht angehoben werden: {e}")

def main():
    _raise_nofile_limit()
    if TRAIN_DEVICE == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = TRAIN_DEVICE

    print(
        f"🔥 Starte Turbo-Training mit "
        f"{NUM_ENVS} Instanzen auf Device: "
        f"{device.upper()}"
    )

    print(
        "🧭 Curriculum: 65% der Episoden starten von vorne; "
        "35% duerfen spaeter von selbst erreichten Zwischenstaenden starten."
    )

    print(
        f"⏱️ Episodenlaenge: "
        f"{PokemonFireRedEnv.MAX_EPISODE_STEPS:,} Agent-Schritte"
    )

    # ------------------------------------------------------------
    # TRAINING V2: gemeinsamer Curriculum- und Mapping-Speicher
    # ------------------------------------------------------------
    seeded_curriculum = seed_shared_curriculum()
    if seeded_curriculum:
        print(
            "🎓 Shared Curriculum initialisiert: "
            + ", ".join(seeded_curriculum)
        )

    seed_edges, seed_maps, seed_transitions = load_global_exploration()

    manager = mp.Manager()
    shared_edges = manager.dict(seed_edges)
    shared_maps = manager.dict(seed_maps)
    shared_transitions = manager.dict(seed_transitions)

    # V8: persistierter ECHTER Episoden-Depth-Rekord.
    progress_file = os.path.join(
        EXPLORATION_MEMORY_DIR, "global_progress.json"
    )
    persisted_depth = 0
    try:
        with open(progress_file, "r") as f:
            persisted_depth = int(
                (json.load(f) or {}).get("max_episode_maps", 0)
            )
    except Exception:
        persisted_depth = 0

    shared_progress = manager.dict({
        "max_episode_maps": persisted_depth,
    })
    shared_lock = manager.RLock()

    print(
        "🗺️ Global Exploration geladen: "
        f"{len(seed_edges)} Kanten | "
        f"{len(seed_maps)} Maps | "
        f"{len(seed_transitions)} Warps"
    )
    print("🎯 V11 CLEAN EXPLORE: Reward = Exploration dominant (+2.0/Tile), keine Straf-Suppe, kein Nord-Prior. Bootcamp: intro>treppe>exit>starter>Welt (auto ab 88%)")

    print("🎮 V7.7 Actions: A | B | START | UP | DOWN | LEFT | RIGHT")
    print("🧭 V7.7 Observation: UNVERAENDERT 64x64 Bild + 28 RAM/Nav Features")
    print("🌲 V10.27B Fokus: Vertania City erreichen, Markt/Paket erkunden und zu Professor Eich zurueckkehren.\n⚡ V7.5 Speed Cache: adjacency/frontier/distance caching aktiv")

    vec_env = SubprocVecEnv(
        [
            make_env(
                i,
                shared_edges,
                shared_maps,
                shared_transitions,
                shared_progress,
                shared_lock,
            )
            for i in range(NUM_ENVS)
        ]
    , start_method="spawn")

    if os.path.exists(RESUME_MODEL):
        load_model = RESUME_MODEL
    elif os.path.exists(BEST_MODEL):
        load_model = BEST_MODEL
    else:
        load_model = LATEST_MODEL

    if os.path.exists(load_model):
        print(
            "🏆 Lade Champion-Modell..."
            if load_model == BEST_MODEL
            else "🧠 Lade aktuelles Modell als Champion-Basis..."
        )
        model = PPO.load(
            load_model,
            env=vec_env,
            device=device,
            custom_objects={
                "learning_rate": LEARNING_RATE,
                "n_steps": PPO_N_STEPS,
                "batch_size": PPO_BATCH_SIZE,
                "n_epochs": PPO_N_EPOCHS,
                "gamma": PPO_GAMMA,
                "ent_coef": PPO_ENT_COEF,
            },
        )
        if not os.path.exists(BEST_MODEL):
            shutil.copy2(load_model, BEST_MODEL)

        # V11: nach einem Reset ist RESUME eine neu geseedete Skill-Policy.
        # Ihr eingebauter Step-Zaehler (~11 Mio) ist irrefuehrend - es ist ein
        # frischer V11-Lauf. Ohne echten Champion (keine champion_score.json)
        # den Zaehler auf 0 setzen, damit das Dashboard „V11-Steps" zeigt.
        if not os.path.exists(CHAMPION_FILE):
            model.num_timesteps = 0
            try:
                model._num_timesteps_at_start = 0
            except Exception:
                pass
            print("🔄 V11-Reset erkannt -> Step-Zaehler auf 0")

    else:
        print("🌱 Initialisiere neues PPO-Modell...")

        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            ent_coef=PPO_ENT_COEF,
            verbose=1,
            device=device
        )

    print(
        "⚡ V10.6 EFFECTIVE PPO: "
        f"n_steps={model.n_steps} | "
        f"rollout={model.n_steps * model.n_envs} | "
        f"lr={LEARNING_RATE} | "
        f"batch={model.batch_size} | "
        f"epochs={model.n_epochs}"
    )

    print(
        "🌙 V10.7 EFFECTIVE PPO: "
        f"agents={NUM_ENVS} | n_steps={model.n_steps} | "
        f"rollout={model.n_steps * model.n_envs} | lr={LEARNING_RATE}"
    )
    print(
        "🏆 Frontier Champion: neue Full-from-start Tiefe wird sofort geschützt; "
        "24 Episoden + 8 abgeschlossene Full-Runs für same-depth Optimierung."
    )

    print("🧬 V10.9.2 FULL BUFFER ACTIVE: Full-Runs bleiben separat erhalten.")

    print("🧬 V10.9.4 LIVE FULL + RESUME ACTIVE")

    print("🧰 V10.27 ECHTES BOOTCAMP: jede fehlende Stufe wird nacheinander gelernt und sofort im Vault eingefroren.")
    print("🛡️ V10.27 Keine Auto-Rollbacks; Phasenwechsel nur durch geschuetzte Skill-Scores.")

    print("🧠 V10.13 FULL-POLICY UNIFIED ACTIVE: Story-Spezialisten trainieren jetzt dasselbe Full-Brain wie der Watcher.")

    # V10.27: geladene PPO-Modelle behalten sonst ihre alte LR.
    # Daher LR auch effektiv im geladenen Optimizer erzwingen.
    model.learning_rate = LEARNING_RATE
    model.lr_schedule = lambda _progress_remaining: LEARNING_RATE
    # V10.28: dasselbe gilt fuer ent_coef - ein geladenes Modell behaelt
    # sonst den alten, zu niedrigen gespeicherten Wert (0.008) und das
    # Entropy-Re-Heat greift nicht.
    model.ent_coef = PPO_ENT_COEF
    try:
        for group in model.policy.optimizer.param_groups:
            group["lr"] = LEARNING_RATE
    except Exception:
        pass
    print(f"🚀 V10.27 Effective LR: {LEARNING_RATE:.8f}")

    callback = MilestoneCheckpointCallback(
        check_freq=SAVE_EVERY_TIMESTEPS
    )

    # Solange TRAIN_FOREVER=True, laeuft PPO blockweise endlos weiter.
    # reset_num_timesteps=False behaelt den globalen Step-Zaehler bei.
    try:
        if TRAIN_FOREVER:
            print(
                f"♾️ Endlos-Training aktiv: "
                f"{TRAIN_CHUNK_TIMESTEPS:,} Steps pro Block | Stop mit Ctrl+C"
            )
            while True:
                model.learn(
                    total_timesteps=TRAIN_CHUNK_TIMESTEPS,
                    callback=callback,
                    reset_num_timesteps=False
                )
        else:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=False
            )
    except KeyboardInterrupt:
        print("🛑 Training wird beendet - speichere letzten Stand ...")
    finally:
        try:
            callback.final_candidate_save()
            print("💾 Finaler Candidate gespeichert; Champion bleibt geschützt.")
        except Exception as exc:
            print(f"⚠️ Candidate-Final-Save fehlgeschlagen: {exc}")
        try:
            vec_env.close()
        except (EOFError, BrokenPipeError):
            print("🧹 Worker waren bereits beendet; Shutdown abgeschlossen.")
        try:
            manager.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()