import os
import json
import torch
import multiprocessing as mp
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from pokemon_env import PokemonFireRedEnv


# ================================================================
# USER CONFIG / TRAINING TUNING
# ================================================================
# Die wichtigsten Werte stehen absichtlich hier oben.

NUM_ENVS = 30

# Endlos-Training: laeuft in Bloecken weiter, bis du Ctrl+C drueckst.
# TRAIN_CHUNK_TIMESTEPS ist nur die Groesse eines learn()-Blocks.
TRAIN_FOREVER = True
TRAIN_CHUNK_TIMESTEPS = 1_000_000

# Nur benutzt, wenn TRAIN_FOREVER = False.
TOTAL_TIMESTEPS = 100_000_000

SAVE_EVERY_TIMESTEPS = 25_000

# PPO
LEARNING_RATE = 0.00010
PPO_N_STEPS = 128
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_GAMMA = 0.995
PPO_ENT_COEF = 0.008

# "auto" = MPS auf Apple Silicon wenn verfuegbar, sonst CPU.
# Alternativ: "cpu" oder "mps"
TRAIN_DEVICE = "auto"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
BASE_DIR = PROJECT_ROOT
EXPLORATION_MEMORY_DIR = os.path.join(RUNTIME_DIR, "exploration_memory")
CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_states")
SHARED_CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_shared")

# ================================================================
# INTERNAL PATHS - normalerweise nicht aendern
# ================================================================
MODEL_DIR = os.path.join(RUNTIME_DIR, "checkpoints")
LATEST_MODEL = os.path.join(MODEL_DIR, "pokemon_model_latest.zip")
VERSION_FILE = os.path.join(RUNTIME_DIR, "model_version.json")

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
    shared_lock,
):
    def _init():
        return PokemonFireRedEnv(
            rank=rank,
            shared_edges=shared_edges,
            shared_maps=shared_maps,
            shared_transitions=shared_transitions,
            shared_lock=shared_lock,
        )
    return _init


class MilestoneCheckpointCallback(BaseCallback):
    def __init__(self, check_freq=50_000, verbose=1):
        super().__init__(verbose)

        self.check_freq = check_freq
        self.last_saved_step = 0

        self.max_level_seen = 0
        self.max_badges_seen = 0
        self.max_maps_seen = 0

        self.version = 1

        if os.path.exists(VERSION_FILE):
            try:
                with open(VERSION_FILE, "r") as f:
                    self.version = (
                        int(json.load(f).get("version", 0)) + 1
                    )
            except Exception:
                self.version = 1

    def _save_brain(self, reason="Step-Intervall"):
        self.model.save(LATEST_MODEL)

        tmp_version = VERSION_FILE + ".tmp"
        with open(tmp_version, "w") as f:
            json.dump(
                {
                    "version": self.version,
                    "timesteps": int(self.num_timesteps),
                    "max_level": int(self.max_level_seen),
                    "max_badges": int(self.max_badges_seen),
                    "max_maps": int(self.max_maps_seen),
                },
                f
            )
        os.replace(tmp_version, VERSION_FILE)

        print(
            f"💾 [{reason}] Checkpoint bei "
            f"{self.num_timesteps:,} Steps: "
            f"PKMAI v{self.version:06d}"
        )

        self.version += 1

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue

            p_lvl = int(info.get("p1_level", 0))

            badges_raw = int(info.get("badges", 0))
            badges = (
                bin(badges_raw).count("1")
                if badges_raw > 0
                else 0
            )

            visited_maps = int(
                info.get("visited_maps", 0)
            )

            milestone_saved = info.get(
                "milestone_saved"
            )

            if (
                self.max_level_seen < 5 and
                p_lvl >= 5
            ):
                self.max_level_seen = p_lvl
                self._save_brain(
                    reason="🌟 Starter erstmals erreicht"
                )
                self.last_saved_step = self.num_timesteps

            if badges > self.max_badges_seen:
                self.max_badges_seen = badges
                self._save_brain(
                    reason=f"🏆 Orden {badges}/8 erreicht"
                )
                self.last_saved_step = self.num_timesteps

            # Auch deutlicher Karten-/Story-Fortschritt kann einen Brain-Save
            # ausloesen. Das erzeugt nicht bei jeder einzelnen neuen Map eine
            # Datei, sondern nur bei groesseren Spruengen.
            if visited_maps >= self.max_maps_seen + 5:
                self.max_maps_seen = visited_maps
                self._save_brain(
                    reason=f"🗺️ {visited_maps} Maps in Episode besucht"
                )
                self.last_saved_step = self.num_timesteps

            if milestone_saved:
                print(
                    f"📍 Curriculum-State gespeichert: "
                    f"{milestone_saved}"
                )

        if (
            self.num_timesteps - self.last_saved_step
        ) >= self.check_freq:
            self.last_saved_step = self.num_timesteps
            self._save_brain(
                reason="Zyklisches Training"
            )

        return True


def main():
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
    shared_lock = manager.RLock()

    print(
        "🗺️ Global Exploration geladen: "
        f"{len(seed_edges)} Kanten | "
        f"{len(seed_maps)} Maps | "
        f"{len(seed_transitions)} Warps"
    )
    print(
        "🎓 V7.1 Rollen: "
        "5 Intro | 5 Treppe | 5 Ausgang | 10 Progress | 5 Full Chain"
    )

    print("🎮 V7.1 Actions: A | B | START | UP | DOWN | LEFT | RIGHT")
    print("🧭 V7.1 Observation: 64x64 Bild + 20 RAM/Nav Features")
    print("🌉 V7.1 Progress Bridge: Checkpoints + Stall-Restart aktiv")

    vec_env = SubprocVecEnv(
        [
            make_env(
                i,
                shared_edges,
                shared_maps,
                shared_transitions,
                shared_lock,
            )
            for i in range(NUM_ENVS)
        ]
    )

    if os.path.exists(LATEST_MODEL):
        print("🧠 Lade existierendes Modell...")
        print(
            "⚠️ Hinweis: Fuer Training V2 wird ein frischer Brain empfohlen. "
            "Das alte 10M-Modell hat die alte Reward-Policy gelernt."
        )

        model = PPO.load(
            LATEST_MODEL,
            env=vec_env,
            device=device
        )

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
            callback._save_brain(reason="Final-Save beim Beenden")
        except Exception as exc:
            # Fallback: wenigstens das Modell sichern.
            print(f"⚠️ Callback-Final-Save fehlgeschlagen: {exc}")
            model.save(LATEST_MODEL)
        vec_env.close()
        try:
            manager.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
