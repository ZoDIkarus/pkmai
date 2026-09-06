import os
import json
import math
import torch
import multiprocessing as mp
import shutil
import resource
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import pokemon_env
from pokemon_env import PokemonFireRedEnv


# ================================================================
# USER CONFIG / TRAINING TUNING
# ================================================================
# Die wichtigsten Werte stehen absichtlich hier oben.

# V16: 50 headless Envs; jeder liefert 512 zusammenhaengende Entscheidungen.
# Das ergibt 25.600 Samples pro PPO-Update und laesst Rewards innerhalb langer
# Intro-/Navigationsfolgen wesentlich weiter zurueckwirken.
# Sichtbar gerendert wird nur der unabhaengige Watcher; Rendering trainiert nicht.
NUM_ENVS = 60

# Endlos-Training: laeuft in Bloecken weiter, bis du Ctrl+C drueckst.
# TRAIN_CHUNK_TIMESTEPS ist nur die Groesse eines learn()-Blocks.
TRAIN_FOREVER = True
TRAIN_CHUNK_TIMESTEPS = 1_000_000

# Nur benutzt, wenn TRAIN_FOREVER = False.
TOTAL_TIMESTEPS = 100_000_000

# Kandidaten nur in ausreichend grossen Generationen bewerten.
SAVE_EVERY_TIMESTEPS = 250_000

# PPO
# Experiment 2026-09-05 beendet: 0.0005 zeigte nach 2+ Mio. Steps ohne
# Champion-Fortschritt echte Verschlechterung (alle 60 Agenten exakt Level 6,
# kaum noch Kaempfe, viel weniger Route-1-Ankuenfte als vorher) - zurueck auf
# den seit V11 bewaehrten Wert.
LEARNING_RATE = 7.5e-05
# V17.3: 512 -> 256, damit PPO doppelt so oft aktualisiert (haeufigeres,
# frischeres Feedback bei den langen 12.000-Schritte-Episoden) statt auf
# einem einzigen sehr grossen Rollout zu sitzen. 256 x 60 Envs = 15.360
# Rollout-Samples, exakt 60 Minibatches à 256.
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_GAMMA = 0.999
PPO_GAE_LAMBDA = 0.995
# 50 parallele Agenten liefern bereits viel Exploration. Eine niedrigere
# Entropie laesst erfolgreiche Wege und Kampfsequenzen sauberer wiederholen,
# ohne die Suche im Wald abzuwuergen.
PPO_ENT_COEF = 0.02

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
BEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_champion.zip")
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
    # Eigenes V2-Ziel: nur Schiggy. Der alte Starter-Vault belohnte alle drei
    # Starter und darf nicht als bereits gemeisterte Basis weitergelten.
    "starter": os.path.join(MODEL_DIR, "pokemon_skill_squirtle_best.zip"),
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
    shared_species,
    shared_tiles,
    n_envs=NUM_ENVS,
):
    def _init():
        return PokemonFireRedEnv(
            rank=rank,
            shared_edges=shared_edges,
            shared_maps=shared_maps,
            shared_transitions=shared_transitions,
            shared_progress=shared_progress,
            shared_lock=shared_lock,
            shared_species=shared_species,
            shared_tiles=shared_tiles,
            n_envs=n_envs,
        )
    return _init


def save_model_atomic(model, path):
    """Publish a complete snapshot; a watcher never reads a half-written ZIP."""
    tmp = path + f".tmp.{os.getpid()}.zip"
    try:
        model.save(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class MilestoneCheckpointCallback(BaseCallback):
    """V16: promote measured full-run candidates and protect the champion."""

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
        self.min_eval_episodes = 32
        self.min_full_episodes = 32
        self.champion_score = None
        self.champion_metrics = {}
        self.rollback_count = 0
        self.regression_strikes = 0
        self.last_eval_metrics = {}
        self.last_eval_result = ""
        self.last_eval_at_step = 0
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
                self.champion_metrics = dict(data.get("metrics") or {})
                if data.get("progress_schema") != PokemonFireRedEnv.PROGRESS_SCHEMA:
                    old = int(self.champion_metrics.get("max_stage", 0))
                    self.champion_metrics["max_stage"] = {4: 1, 5: 1, 6: 4, 7: 5, 8: 6, 9: 6}.get(old, old)
                    raw = None
                calculated_score = self._score(self.champion_metrics)
                if (
                    isinstance(raw, list)
                    and len(raw) == len(calculated_score)
                ):
                    self.champion_score = tuple(int(x) for x in raw)
                else:
                    # Score-Schema wurde um den Full-Speed-Tie-Breaker
                    # erweitert. Bestehende Metriken neu einordnen, ohne
                    # Modell, Version oder Champion-Step zu veraendern.
                    self.champion_score = calculated_score
                self.champion_published_at_step = int(
                    data.get("timesteps", 0) or 0
                )
            except Exception:
                pass

        # Den letzten ausgewerteten Candidate nach einem sauberen Prozess-
        # Neustart weiter im Status zeigen. Bei einem echten Runtime-Reset ist
        # die Datei nicht vorhanden und es wird folgerichtig nichts geerbt.
        if os.path.exists(TRAINER_STATUS_FILE):
            try:
                with open(TRAINER_STATUS_FILE, "r") as f:
                    previous_status = json.load(f) or {}
                self.last_eval_metrics = dict(
                    previous_status.get("last_eval_metrics") or {}
                )
                self.last_eval_result = str(
                    previous_status.get("last_eval_result", "") or ""
                )
                self.last_eval_at_step = int(
                    previous_status.get("last_eval_at_step", 0) or 0
                )
                # Migration vom alten Statusformat: Die Metriken wurden schon
                # gespeichert, aber das Ergebnis noch nicht benannt. Liegt der
                # Candidate unter dem bestaetigten Champion, war er eindeutig
                # nicht uebernommen worden.
                if (
                    self.last_eval_metrics
                    and not self.last_eval_result
                    and self.champion_score is not None
                    and self._score(self.last_eval_metrics)
                        <= self.champion_score
                ):
                    self.last_eval_result = "rejected"
            except Exception:
                pass

        if self.champion_score is None:
            self._seed_baseline_from_training_stats()

        if PokemonFireRedEnv.FULL_ONLY_MODE:
            # V16 kennt weder Skill-Brains noch Skill-Phasen. Insbesondere
            # keine leere Legacy-Datei erzeugen, die im Dashboard den Eindruck
            # erweckt, ein Progress-/Skill-Agent koenne noch eingreifen.
            self.skill_scores = {}
            self.skill_health_seed = {}
        else:
            self._load_skill_scores()
            self.skill_health_seed = self._load_skill_health_seed()

    def _load_skill_health_seed(self):
        """Erfolgszaehler vor diesem Trainerstart fuer Retention beibehalten."""
        pairs = {
            "intro": ("v2_intro_success", "v2_intro_episodes"),
            "stairs": ("v2_stairs_success", "v2_stairs_episodes"),
            "exit": ("v2_exit_success", "v2_exit_episodes"),
            "starter": ("v8_starter_success", "v8_starter_episodes"),
        }
        seed = {k: {"success": 0, "episodes": 0} for k in pairs}
        stats_dir = os.path.join(RUNTIME_DIR, "training_stats")
        try:
            names = os.listdir(stats_dir)
        except Exception:
            names = []
        for name in names:
            if not name.startswith("agent_") or not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(stats_dir, name), "r") as f:
                    data = json.load(f) or {}
                for skill, (success_key, episode_key) in pairs.items():
                    seed[skill]["success"] += int(data.get(success_key, 0) or 0)
                    seed[skill]["episodes"] += int(data.get(episode_key, 0) or 0)
            except Exception:
                pass
        return seed

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
            "max_stage": 0,
        }
        self.champion_score = self._score(self.champion_metrics)

        # Bei einem echten Clean-Start ist dies nur die interne Nullbasis,
        # noch kein bestaetigter Champion. Erst eine ausgewertete Generation
        # darf Champion-Datei und Bestmodell gemeinsam veroeffentlichen.
        if full_ep > 0 or os.path.exists(BEST_MODEL):
            self._write_champion_score(
                self.champion_score, self.champion_metrics
            )

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
        # Migration vom alten "beliebiger Starter"-Vault: Solange noch kein
        # bestaetigter Schiggy-Vault existiert, beginnt dieser Skill bei 0 und
        # aktiviert automatisch erneut das grosse Starter-Bootcamp.
        if not os.path.exists(SKILL_MODELS["starter"]):
            defaults["starter"] = 0
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
                + int(r.get("stage", 0)) * 1000
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
        if bool(info.get("has_target_starter", False)):
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
            # V14: Starter zaehlt NUR wenn der Agent damit auch DRAUSSEN war
            # (story_stage OUTDOOR) - nicht schon im Labor. Das ist die echte
            # "raus aus Eichs Labor"-Wand.
            "starter": int(
                bool(info.get("has_target_starter", False))
                and stage == "OUTDOOR"
            ),
            "badge": int(badges >= 1),
            "badges": badges,
            "maps": int(info.get("visited_maps", 0)),
            "stage": int(info.get("world_stage", 0)),
            "level": level,
            "steps": int(info.get("ppo_episode_steps", info.get("episode_steps", 0)) or 0),
            "arrivals": dict(info.get("stage_arrival_steps", {})),
        }

    @staticmethod
    def _score(m):
        # Tiefe zuerst, danach die reproduzierbare komplette Storykette.
        return (
            int(m.get("max_badges", 0)),
            int(m.get("max_stage", 0)),
            int(m.get("full_starter_permille", 0)),
            int(m.get("full_exit_permille", 0)),
            int(m.get("full_stairs_permille", 0)),
            int(m.get("full_intro_permille", 0)),
            int(m.get("max_level", 0)),
            int(m.get("max_maps", 0)),
            -int(m.get("full_best_stage_steps", 1_000_000) or 1_000_000),
        )

    # Ein einzelner Glueckslauf unter vielen darf keine neue Weltstufe fuers
    # Champion-Ranking freigeben - sonst koennte ein Candidate befoerdert
    # werden, der die Tiefe nur einmal erreicht hat, obwohl alle anderen
    # Full-Laeufe dort nie hinkamen. Mindestens dieser Anteil der Full-Laeufe
    # muss die Stufe reproduzierbar schaffen.
    STAGE_RELIABILITY_FRACTION = 0.12

    def _reliable_max_stage(self, full):
        if not full:
            return 0
        n = len(full)
        min_count = max(1, math.ceil(n * self.STAGE_RELIABILITY_FRACTION))
        stages = sorted({int(r.get("stage", 0)) for r in full}, reverse=True)
        for s in stages:
            if s <= 0:
                break
            reached = sum(1 for r in full if int(r.get("stage", 0)) >= s)
            if reached >= min_count:
                return s
        return 0

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

        max_stage = self._reliable_max_stage(full)
        stage_steps = [
            int(r.get("steps", 0)) for r in full
            if int(r.get("stage", 0)) == int(max_stage)
            and int(r.get("steps", 0)) > 0
        ]
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
            # Der Champion bewertet ausschliesslich abgeschlossene Full-Runs
            # vom Spielanfang. Curriculum-Progress darf ihn nicht befoerdern.
            "badge_episodes": sum(r["badge"] for r in full),
            "max_badges": max((r["badges"] for r in full), default=0),
            "max_maps": max((r["maps"] for r in full), default=0),
            "max_stage": max_stage,
            "max_level": max((r.get("level", 0) for r in full), default=0),
            "full_best_stage_steps": min(stage_steps) if stage_steps else 0,
        }

    def _metrics_floor(self, metrics):
        """Direkt nach Neustart ist recent_full leer -> alle full_*_permille
        aus _metrics() sind 0. Ein Frontier-/Milestone-Publish in diesem
        Moment wuerde sonst die bekannten guten Champion-Raten mit 0
        ueberschreiben und den Regressions-Schutz aushebeln. Deshalb die
        alten Champion-Werte als Untergrenze behalten, solange noch keine
        echten Beginning-Full-Runs abgeschlossen sind."""
        old = self.champion_metrics or {}
        measured_stage = int(metrics.get("max_stage", 0))
        old_stage = int(old.get("max_stage", 0))
        # world_stage / Tiefe immer als Untergrenze halten (auch mit Full-Runs).
        for k in ("max_stage", "max_level", "max_badges"):
            metrics[k] = max(int(metrics.get(k, 0)), int(old.get(k, 0)))
        if int(metrics.get("full_episodes", 0)) > 0:
            # Eine schnelle Episode auf einer flacheren Stufe ist kein
            # Geschwindigkeitsbeweis fuer die tiefere Champion-Stufe.
            if measured_stage < old_stage:
                metrics["full_best_stage_steps"] = int(
                    old.get("full_best_stage_steps", 0) or 0
                )
            return metrics
        for k in ("full_intro_permille", "full_stairs_permille",
                  "full_exit_permille", "full_starter_permille"):
            metrics[k] = max(int(metrics.get(k, 0)), int(old.get(k, 0)))
        metrics["full_best_stage_steps"] = int(
            old.get("full_best_stage_steps", 0) or 0
        )
        return metrics

    def _live_skill_health(self):
        """Rollende aktuelle Retention, getrennt von Lebenszeit/Vault."""
        rows = list(self.recent)
        health = {}
        for skill, key in (
            ("intro", "intro"),
            ("stairs", "stairs"),
            ("exit", "exit"),
            ("starter", "starter"),
        ):
            # Alte Fehlversuche duerfen einen inzwischen neu gelernten Skill
            # nicht fuer Stunden in derselben Bootcamp-Phase festhalten.
            # Maximal 64 aktuelle, abgeschlossene Episoden beurteilen das
            # gemeinsame Learner-Gehirn. Nach Neustart gilt eine Schonfrist,
            # bis wieder mindestens 16 aktuelle Tests vorhanden sind.
            group = [r for r in rows if r["role"] == skill][-64:]
            successes = sum(int(r[key]) for r in group)
            episodes = len(group)
            score = (
                round(1000 * successes / episodes)
                if episodes else 0
            )
            health[skill] = {"score": int(score), "episodes": int(episodes)}
        return health

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
            full_only = bool(PokemonFireRedEnv.FULL_ONLY_MODE)
            live_skill_health = {} if full_only else self._live_skill_health()
            effective_scores = {} if full_only else dict(self.skill_scores)
            if not full_only:
                for skill in ("intro", "stairs", "exit", "starter"):
                    h = live_skill_health.get(skill, {})
                    if (
                        int(effective_scores.get(skill, 0)) >= 880
                        and int(h.get("episodes", 0)) >= 16
                        and int(h.get("score", 0)) < 650
                    ):
                        effective_scores[skill] = int(h.get("score", 0))

            payload = {
                "trainer_pid": int(os.getpid()),
                "learner_steps": int(self.num_timesteps),
                "champion_steps": champion_steps,
                "champion_version": champion_version,
                "delta_steps": int(self.num_timesteps) - champion_steps,
                "recent_full_done": len(getattr(self, "recent_full", [])),
                "mode": "v16_clean_full_brain",
                "rollback_count": int(self.rollback_count),
                "regression_strikes": int(self.regression_strikes),
                "skill_scores": dict(self.skill_scores),
                "live_skill_health": live_skill_health,
                "effective_skill_scores": effective_scores,
                "last_eval_metrics": dict(self.last_eval_metrics),
                "last_eval_result": str(self.last_eval_result),
                "last_eval_at_step": int(self.last_eval_at_step),
                "training_phase": "full_brain" if full_only else (
                    "1_intro" if int(effective_scores.get("intro", 1000)) < 880
                    else "2_stairs" if int(effective_scores.get("stairs", 1000)) < 880
                    else "3_exit" if int(effective_scores.get("exit", 1000)) < 880
                    else "4_starter" if int(effective_scores.get("starter", 1000)) < 880
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
                "progress_schema": PokemonFireRedEnv.PROGRESS_SCHEMA,
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
                "progress_schema": PokemonFireRedEnv.PROGRESS_SCHEMA,
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
        # Ein schlechter Kandidat wird zwar nicht Champion, konnte bisher aber
        # unbegrenzt als Learner weitertrainieren und dabei den Anfang komplett
        # vergessen. Tieferer echter Fortschritt darf passieren; bei gleicher
        # Tiefe muessen zentrale Full-Faehigkeiten erhalten bleiben.
        old = self.champion_metrics or {}

        n = int(candidate.get("full_episodes", 0))
        if n < self.min_full_episodes:
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
        intro_floor = min(900, max(850, old_intro - 100))
        if old_intro >= 850 and new_intro < intro_floor:
            print(
                f"🛡️ Champion geschuetzt: Full-Intro "
                f"{old_intro/10:.1f}% -> {new_intro/10:.1f}%"
            )
            return True

        return False

    def _rollback_to_champion(self):
        """Restore policy + optimizer and make the safe state restartable."""
        if not os.path.exists(BEST_MODEL):
            return False
        try:
            self.model.set_parameters(BEST_MODEL, exact_match=True)
            save_model_atomic(self.model, RESUME_MODEL)
            self.rollback_count += 1
            self.regression_strikes = 0
            self.recent.clear()
            self.recent_full.clear()
            self.full_live.clear()
            print(
                "🔄 AUTO-ROLLBACK: Learner auf bestätigten Champion "
                f"zurückgesetzt (Rollback {self.rollback_count})."
            )
            return True
        except Exception as exc:
            print(f"⚠️ Auto-Rollback fehlgeschlagen: {exc}")
            return False

    def _evaluate(self):
        metrics = self._metrics()
        if metrics["episodes"] < self.min_eval_episodes or metrics["full_episodes"] < self.min_full_episodes:
            return False

        self.last_eval_metrics = dict(metrics)
        self.last_eval_at_step = int(self.num_timesteps)
        score = self._score(metrics)
        self.model.save(CANDIDATE_MODEL)

        if self._protected_regression(metrics):
            self.last_eval_result = "regression"
            self.regression_strikes += 1
            self.steps_since_champion_update += int(metrics["full_episodes"])
            print(
                f"⚠️ Full-Regression erkannt "
                f"(Messung {self.regression_strikes}). "
                "Champion bleibt erhalten."
            )
            if self.regression_strikes >= 3:
                self._rollback_to_champion()
        elif self.champion_score is None or score > self.champion_score:
            self.last_eval_result = "champion"
            self.regression_strikes = 0
            self._publish_champion(score, metrics, "Recent-Eval")
        else:
            self.last_eval_result = "rejected"
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
            int(champion.get("max_stage", 0)),
            1 if int(champion.get("full_starter_permille", 0)) > 0 else 0,
            1 if int(champion.get("full_exit_permille", 0)) > 0 else 0,
            1 if int(champion.get("full_stairs_permille", 0)) > 0 else 0,
            int(champion.get("max_level", 0)),
        )

        for info in infos:
            if not isinstance(info, dict):
                continue
            if PokemonFireRedEnv.FULL_ONLY_MODE:
                # V16-Champions entstehen nur aus einer abgeschlossenen
                # Generation, niemals aus einem einzelnen Live-Meilenstein.
                continue
            policy_obj = str(info.get("policy_objective", info.get("training_objective", "")))
            if policy_obj != "full":
                continue
            if str(info.get("episode_start", "")) != "beginning":
                continue

            stage = str(info.get("story_stage", "INTRO"))
            badges = int(info.get("badges_count", 0) or 0)
            level = int(info.get("level", info.get("p1_level", 0)) or 0)
            wstage = int(info.get("world_stage", 0) or 0)
            starter = int(
                bool(info.get("has_target_starter", False))
                and stage == "OUTDOOR"
            )
            stairs = int(stage in ("F1_TO_EXIT", "OUTDOOR"))
            exit_done = int(stage == "OUTDOOR")

            key = (badges, wstage, starter, exit_done, stairs, level)
            if key <= champion_key:
                continue

            metrics = self._metrics_floor(self._metrics())
            metrics["max_badges"] = max(int(metrics.get("max_badges", 0)), badges)
            metrics["max_stage"] = max(int(metrics.get("max_stage", 0)), wstage)
            metrics["max_level"] = max(int(metrics.get("max_level", 0)), level)
            metrics["full_best_stage_steps"] = int(
                info.get("episode_steps", 0) or 0
            )
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
                f"stairs={stairs} stage={wstage} lvl={level}",
            )
            self.recent.clear()
            champion_key = key
            break

        current_best_stage = self._champion_full_stage_rank()
        stage_names = {1:"STAIRS", 2:"EXIT", 3:"STARTER", 4:"BADGE1"}

        for info in infos:
            if not isinstance(info, dict):
                continue
            if PokemonFireRedEnv.FULL_ONLY_MODE:
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
                metrics["full_best_stage_steps"] = int(
                    info.get("episode_steps", 0) or 0
                )

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
                    record = self._episode_record(
                        terminal_info
                    )
                    self.recent.append(record)

                    if (
                        record["role"] == "full"
                        and record["start"] == "beginning"
                    ):
                        self.recent_full.append(record)

                self.last_episode_info.pop(i, None)

        for info in infos:
            if not isinstance(info, dict):
                continue
            if PokemonFireRedEnv.FULL_ONLY_MODE:
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
            if not PokemonFireRedEnv.FULL_ONLY_MODE:
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
            save_model_atomic(self.model, RESUME_MODEL)
            print(f"💾 Resume-Stand gespeichert bei {self.num_timesteps:,} Steps")

        # V10.10.1 LIVE TRAINER STATUS
        self._write_trainer_status()

        return True

    def final_candidate_save(self):
        self.model.save(CANDIDATE_MODEL)
        save_model_atomic(self.model, RESUME_MODEL)
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
    legacy_champion = os.path.join(MODEL_DIR, "pokemon_model_best.zip")
    if not os.path.exists(BEST_MODEL) and os.path.exists(legacy_champion):
        os.replace(legacy_champion, BEST_MODEL)

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

    print("🧬 V16 Full-Brain: alle Episoden trainieren dasselbe PPO-Netz.")

    print(
        f"⏱️ Episodenlaenge: "
        f"{PokemonFireRedEnv.MAX_EPISODE_STEPS:,} Agent-Schritte "
        f"(lange Full-Probe: {PokemonFireRedEnv.LONG_FULL_PROBE_STEPS:,})"
    )

    if getattr(PokemonFireRedEnv, "V20_CURRICULUM", False):
        import curriculum_v20
        from nav_transitions_v20 import KnownTransitions, KNOWN as _NK

        _alloc = curriculum_v20.allocation_summary(NUM_ENVS)

        # Seed discovered_stage from the checkpoints/records already on disk so
        # BRIDGE/FRONTIER do not have to rediscover the world from scratch on a
        # cold start. mastered_stage is intentionally NOT seeded - it must be
        # re-earned by Full-from-start confirmations.
        _cur = curriculum_v20.CurriculumState.load(pokemon_env.V20_STATE_FILE)
        _seed_stage = int(_cur.discovered_stage)
        try:
            for _n in PokemonFireRedEnv.SCOUT_STAGES:
                _mp = os.path.join(
                    RUNTIME_DIR, "curriculum_shared", f"stage_{_n}.meta.json"
                )
                if os.path.exists(_mp):
                    with open(_mp) as _f:
                        _m = json.load(_f) or {}
                    if int(_m.get("state_validation", 0)) == 1:
                        _seed_stage = max(_seed_stage, int(_n))
        except Exception:
            pass
        try:
            _gpf = os.path.join(
                EXPLORATION_MEMORY_DIR, "global_progress.json"
            )
            with open(_gpf) as _f:
                _gp = json.load(_f) or {}
            _seed_stage = max(
                _seed_stage, int(_gp.get("max_world_stage", 0) or 0)
            )
        except Exception:
            pass
        _dirty = _seed_stage > int(_cur.discovered_stage)
        if _dirty:
            _cur.record_discovery(_seed_stage)

        # One-time bootstrap of transition 1 (Pallet->Route1). The whole
        # premise is "full runners fail AROUND Route 1" - i.e. they DO leave
        # Pallet reliably. If the champion / global record already shows Full
        # depth >= Route 1 and transition 1 has no stats yet, pre-confirm it so
        # the detected bottleneck starts at Route1->Viridian (brief section 17)
        # instead of spending the first eval windows re-proving Pallet->Route1.
        try:
            _champ_stage = 0
            _cf = os.path.join(RUNTIME_DIR, "champion_score.json")
            if os.path.exists(_cf):
                with open(_cf) as _f:
                    _champ_stage = int(
                        (json.load(_f) or {}).get("metrics", {}).get(
                            "max_stage", 0
                        ) or 0
                    )
            _evidence_stage = max(_champ_stage, _seed_stage)
            if (_evidence_stage >= 2
                    and 1 not in _cur.transitions):
                for _ in range(curriculum_v20.TRANSITION_MASTERY_MIN_ATTEMPTS):
                    _cur.record_transition_attempt(1, True, full_chain=True)
                _dirty = True
                print(
                    "🧭 V20 bootstrap: Pallet->Route1 pre-confirmed from "
                    f"existing Full depth (stage {_evidence_stage})"
                )
        except Exception:
            pass

        if _dirty:
            _cur.save(pokemon_env.V20_STATE_FILE)
        _cur = curriculum_v20.CurriculumState.load(
            pokemon_env.V20_STATE_FILE
        )
        _known = KnownTransitions.load(
            pokemon_env.V20_KNOWN_TRANSITIONS_FILE
        )
        _bn = _cur.current_bottleneck
        print(
            "🧭 V20 CURRICULUM MODES | "
            f"FULL={_alloc['FULL']} BRIDGE={_alloc['BRIDGE']} "
            f"FRONTIER={_alloc['FRONTIER']} RETENTION={_alloc['RETENTION']}"
        )
        if _bn is not None:
            _bn_txt = (
                f"bottleneck={curriculum_v20.transition_name(_bn)} "
                f"({_known.navigation_state(_bn)})"
            )
        elif int(_cur.discovered_stage) <= 1:
            _bn_txt = (
                "bottleneck=Pallet->Route1 (nothing discovered past stage 1 "
                "yet - whole fleet runs FULL until the net holds Route 1)"
            )
        else:
            _bn_txt = "bottleneck=none (everything discovered is mastered)"
        print(
            f"🧭 discovered_stage={_cur.discovered_stage} "
            f"mastered_stage={_cur.mastered_stage} {_bn_txt}"
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
    # V17.2: welche Spezies wurden fleet-weit schon mindestens einmal gefangen.
    # Bewusst nicht aus alten Laeufen geseedet - jeder Full-Reset soll die
    # Erstfang-Boni wieder frisch vergeben koennen.
    shared_species = manager.dict()
    # V17.4-Fix: welche Kacheln (bank,map,x,y) wurden fleet-weit schon jemals
    # betreten - ersetzt shared_edges als Basis fuer den einmaligen globalen
    # Explorationsbonus (+2). Anders als shared_species (bewusst leer, siehe
    # oben) MUSS das aus der Kanten-Historie geseedet werden: shared_tiles
    # ist ein reines In-Memory-Manager-Dict, das bei JEDEM Trainer-Neustart
    # (nicht nur bei einem vollen Reset) neu entsteht. Ohne Seeding wuerde
    # jede laengst besuchte Kachel nach jedem Neustart erneut als "global
    # neu" gelten und +2 auszahlen - live beobachtet nach einem gezielten
    # Trainer-Neustart heute Nacht. Es gibt keine eigene persistierte
    # Tile-Historie, aber jede bekannte Kante verbindet zwei tatsaechlich
    # betretene Kacheln - beide Endpunkte aus der ohnehin schon geladenen
    # Kanten-Historie ergeben eine realistische Naeherung (identisch zur
    # Watcher-Seed-Logik in watcher_runtime.make_evaluation_env()).
    seed_tiles = {}
    for e in seed_edges:
        if len(e) == 6:
            bank, map_id, x1, y1, x2, y2 = e
            seed_tiles[(bank, map_id, x1, y1)] = 1
            seed_tiles[(bank, map_id, x2, y2)] = 1
    shared_tiles = manager.dict(seed_tiles)

    # V15: persistierter world_stage-Rekord (1=Alabastia .. 9=Orden).
    progress_file = os.path.join(
        EXPLORATION_MEMORY_DIR, "global_progress.json"
    )
    persisted_depth = 0
    _gp = {}
    try:
        with open(progress_file, "r") as f:
            _gp = json.load(f) or {}
            persisted_depth = int(
                _gp.get("max_world_stage", _gp.get("max_episode_maps", 0))
            )
    except Exception:
        persisted_depth = 0

    if _gp.get("progress_schema") != PokemonFireRedEnv.PROGRESS_SCHEMA:
        persisted_depth = max((PokemonFireRedEnv.WORLD_STAGE_BY_MAP.get(tuple(m), 0)
                               for m in seed_maps), default=1)
        with open(progress_file + ".tmp", "w") as f:
            json.dump({"max_world_stage": persisted_depth,
                       "progress_schema": PokemonFireRedEnv.PROGRESS_SCHEMA}, f)
        os.replace(progress_file + ".tmp", progress_file)

    shared_progress = manager.dict({
        "max_world_stage": persisted_depth,
    })
    shared_lock = manager.RLock()

    print(
        "🗺️ Global Exploration geladen: "
        f"{len(seed_edges)} Kanten | "
        f"{len(seed_maps)} Maps | "
        f"{len(seed_transitions)} Warps"
    )
    print(
        f"🎯 V16: {NUM_ENVS} Envs × {PPO_N_STEPS} Schritte = "
        f"{NUM_ENVS * PPO_N_STEPS:,} Samples/Update | "
        "keine Bewegungs-, Tile- oder Warp-Punkte"
    )
    print("🎮 Aktionen: A | B | START | HOCH | RUNTER | LINKS | RECHTS")
    print("🧭 Beobachtung: 4 Bilder + RAM-, Navigation- und Storywerte")

    vec_env = SubprocVecEnv(
        [
            make_env(
                i,
                shared_edges,
                shared_maps,
                shared_transitions,
                shared_progress,
                shared_lock,
                shared_species,
                shared_tiles,
                NUM_ENVS,
            )
            for i in range(NUM_ENVS)
        ]
    , start_method="spawn")

    # Jeder neue Trainingsprozess beginnt an der letzten bestaetigten Basis.
    # Ein unbewerteter Resume-Zwischenstand darf keine Regression konservieren.
    if os.environ.get("PKMAI_RESUME_SAVED") == "1" and os.path.exists(RESUME_MODEL):
        # Explicit maintenance restart: continue the just-saved learner without
        # promoting it to champion or discarding work since the last evaluation.
        load_model = RESUME_MODEL
    elif os.path.exists(BEST_MODEL):
        load_model = BEST_MODEL
    elif os.path.exists(RESUME_MODEL):
        load_model = RESUME_MODEL
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
                "gae_lambda": PPO_GAE_LAMBDA,
                "ent_coef": PPO_ENT_COEF,
            },
        )
        # Ein Resume ist nur ein fortsetzbarer Learner-Zwischenstand. Er darf
        # bei einem Neustart vor der ersten Evaluation nicht stillschweigend
        # zum bestaetigten Champion werden.

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
            print("🔄 Clean-Reset erkannt -> Step-Zaehler auf 0")

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
            gae_lambda=PPO_GAE_LAMBDA,
            ent_coef=PPO_ENT_COEF,
            verbose=1,
            device=device
        )

    print(
        "⚡ PPO: "
        f"n_steps={model.n_steps} | rollout={model.n_steps * model.n_envs} | "
        f"lr={LEARNING_RATE} | batch={model.batch_size} | "
        f"epochs={model.n_epochs} | gae_lambda={model.gae_lambda}"
    )
    print(
        "🏆 Champion-Regel: mindestens 32 abgeschlossene Full-Runs; "
        "nur bessere Kandidaten werden übernommen, Regression wird zurückgerollt."
    )

    # V10.27: geladene PPO-Modelle behalten sonst ihre alte LR.
    # Daher LR auch effektiv im geladenen Optimizer erzwingen.
    model.learning_rate = LEARNING_RATE
    model.lr_schedule = lambda _progress_remaining: LEARNING_RATE
    # V10.28: dasselbe gilt fuer ent_coef - ein geladenes Modell behaelt
    # sonst den alten, zu niedrigen gespeicherten Wert (0.008) und das
    # Entropy-Re-Heat greift nicht.
    model.ent_coef = PPO_ENT_COEF
    model.gae_lambda = PPO_GAE_LAMBDA
    try:
        for group in model.policy.optimizer.param_groups:
            group["lr"] = LEARNING_RATE
    except Exception:
        pass
    print(f"🚀 Effektive Lernrate: {LEARNING_RATE:.8f}")

    callback = MilestoneCheckpointCallback(
        check_freq=SAVE_EVERY_TIMESTEPS
    )

    # Solange TRAIN_FOREVER=True, laeuft PPO blockweise endlos weiter.
    # reset_num_timesteps=False behaelt den globalen Step-Zaehler bei.
    try:
        if TRAIN_FOREVER:
            print(
                "♾️ Training läuft fortlaufend | "
                f"interner Speicherblock {TRAIN_CHUNK_TIMESTEPS:,} Gesamtsteps | "
                f"Episode max. {PokemonFireRedEnv.MAX_EPISODE_STEPS:,} Steps | "
                "Stop mit Ctrl+C"
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
