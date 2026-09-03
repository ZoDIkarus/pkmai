import gymnasium as gym
from gymnasium import spaces
import stable_retro as retro
import numpy as np
import cv2
import os
import json
import gzip
import random
from firered_ram import read_player_location, read_enemy_party, read_player_party


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
LOCAL_DIR = os.path.join(PROJECT_ROOT, "local")
BASE_DIR = PROJECT_ROOT
CUSTOM_DIR = os.path.join(LOCAL_DIR, "custom_integrations")
INSTANCES_DIR = os.path.join(RUNTIME_DIR, "instances_data")
EXPLORATION_MEMORY_DIR = os.path.join(RUNTIME_DIR, "exploration_memory")
os.makedirs(EXPLORATION_MEMORY_DIR, exist_ok=True)
CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_states")
SHARED_CURRICULUM_DIR = os.path.join(RUNTIME_DIR, "curriculum_shared")
STATS_DIR = os.path.join(RUNTIME_DIR, "training_stats")
GLOBAL_PROGRESS_FILE = os.path.join(EXPLORATION_MEMORY_DIR, "global_progress.json")

os.makedirs(INSTANCES_DIR, exist_ok=True)
os.makedirs(CURRICULUM_DIR, exist_ok=True)
os.makedirs(SHARED_CURRICULUM_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


class PokemonFireRedEnv(gym.Env):
    BUILD_TAG = "V14_WORLD_STAGE_SINGLE_TRUTH"
    metadata = {"render_modes": []}

    # Training V2 nutzt feste Spezialisten statt einer 90/10-Zufallsquote.

    # Fuer einen langen 4-5h-Lauf deutlich laengere Episoden als bisher.
    # V10.19_1_EARLY_ROUTE_SAFE
    MAX_EPISODE_STEPS = 65536
    PROGRESS_STALL_TIMEOUT = 12000
    POST_STARTER_STALL_TIMEOUT = 12000
    STARTER_RUSH_TIMEOUT = 5000
    STARTER_RUSH_OBJECTIVE_BONUS = 100.0
    PROGRESS_CHECKPOINT_COOLDOWN = 800
    STARTER_SPECIALIST_TIMEOUT = 9000
    BATTLE_SPECIALIST_TIMEOUT = 14000
    LEVEL_SPECIALIST_TIMEOUT = 18000
    BADGE_SPECIALIST_TIMEOUT = 32768
    PARTY_READ_EVERY = 64
    SPECIALIST_SUCCESS_BONUS = 200.0
    # V11: Stage-Caps grosszuegig - eine langsam lernende Policy muss die
    # Belohnung am Stufen-Ende ueberhaupt erreichen koennen, sonst lernt sie
    # die Stufe nie. Anti-Loop / STUCK faengt echtes Feststecken separat ab.
    FULL_INTRO_STAGE_CAP = 6000
    FULL_STAIRS_STAGE_CAP = 9000
    FULL_EXIT_STAGE_CAP = 18000

    LONG_FULL_PROBE_STEPS = 32768

    # V10.29 NORD-SCHUB: Die Welt haengt global bei 5 Episoden-Maps
    # (Route 1) fest - niemand schafft die Route-1-Durchquerung nach
    # Vertania. Solange die Welt noch nicht ueber Vertania hinaus offen
    # ist, bekommt echter Nord-Fortschritt (neuer Y-Bestwert auf der
    # aktuellen Overworld-Map) einen kleinen, gedeckelten, nicht
    # farmbaren Bonus. Deaktiviert sich selbst, sobald global genug
    # Tiefe existiert.
    # ============================================================
    # TRAINING V2 / REWARD TUNING
    # ============================================================
    # ========================================================
    # V11 REWARD-PHILOSOPHIE (ML-optimiert, Marke "Pokemon Red RL"):
    # Exploration ist GRATIS und die dominante Belohnung. Bestraft wird nur
    # echter Schaden (HP-Verlust, Party-Wipe, Kampf-Flucht) und dauerhaftes
    # Einfrieren auf EXAKT einer Kachel. Kein Zeitdruck, keine
    # Backtrack-Strafe, keine "du warst hier schon"-Strafe - PPO muss frei
    # herumprobieren duerfen, um den Weg zu finden.
    # ========================================================
    INTRO_STEP_COST = 0.0
    GAMEPLAY_STEP_COST = 0.0
    KNOWN_PATH_NEUTRAL = True

    # START bleibt fuer Menues frei nutzbar - nicht mehr bestraft.
    START_HOUSE_PENALTY = 0.0
    START_REPEAT_PENALTY_2 = 0.0
    START_REPEAT_PENALTY_3PLUS = 0.0
    START_SPAM_RESET_STEPS = 6

    NEW_EDGE_REWARD = 0.10
    NEW_MAP_REWARD = 250.0
    EPISODE_NEW_MAP_REWARD = 0.0
    NEW_GLOBAL_DEPTH_REWARD = 300.0
    STARTER_REWARD = 500.0
    ENEMY_DAMAGE_REWARD_PER_HP = 0.75
    ENEMY_FAINT_REWARD = 20.0
    ENEMY_HP_READ_EVERY = 2
    NEW_TRANSITION_REWARD = 100.0
    REPLAY_MAP_REWARD = 20.0
    REPLAY_EDGE_REWARD = 0.0
    REPLAY_TRANSITION_REWARD = 8.0

    # V11: Backtracking ist erlaubt - keine Kanten-Wiederholungs-Strafen mehr.
    SECOND_EDGE_VISIT_PENALTY = 0.0
    REPEAT_EDGE_VISITS_FOR_LOOP = 3
    REPEAT_EDGE_PENALTY = 0.0
    INDOOR_SECOND_EDGE_PENALTY = 0.0
    INDOOR_REPEAT_EDGE_PENALTY = 0.0
    # Praktisch aus - faengt nur noch echtes Dauer-Campen ab.
    INDOOR_STALL_SOFT_STEPS = 6000
    INDOOR_STALL_HARD_STEPS = 15000

    # Sobald ein bekannter Story-Uebergang existiert, wird jede Bewegung
    # in Richtung des Ziels wiederholbar belohnt, weg davon symmetrisch
    # bestraft. So vergisst PPO den guten Weg nicht, wenn Novelty weg ist.
    TARGET_PROGRESS_REWARD = 2.5

    EXPLORATION_MEMORY_ENABLED = True
    CONFIRMED_WARP_MIN_AGENTS = 2
    CONFIRMED_WARP_REWARD = 6.0
    # V11: nur noch echtes Einfrieren (exakt gleiche Kachel, lange) wird
    # milde bestraft - nicht mehr "-2.0 alle 96 Steps".
    V9_STUCK_SAME_POS_STEPS = 240
    V9_STUCK_PENALTY = -0.5
    # V11: DOMINANTE Belohnung. Jede neue Kachel = +2.0. Wiederholung ist
    # gratis (0), nicht bestraft.
    V9_EXPLORER_NEW_TILE_BONUS = 2.0
    V9_EXPLORER_REPEAT_TILE_PENALTY = 0.0

    # V13.4 NORD-KORRIDOR: NUR die geraden Nord-Strecken vor dem Wald. Auf
    # diesen Maps ist "hoch = Weg". Pro neuer noerdlichster Y-Reihe gibt es
    # dichten Reward -> aus den 60 blinden Schritten wird eine Belohnungsrampe.
    # Der Wald (Labyrinth) ist BEWUSST NICHT dabei; dort loest reine
    # Exploration + Warp-/Tile-Reward. (3,0)=Alabastia (3,19)=Route 1.
    # Vertania/Route 2 kommen dazu, sobald die Flotte dort zuverlaessig steht.
    NORTH_CORRIDOR_MAPS = {(3, 0), (3, 19)}
    NORTH_CORRIDOR_ROW_REWARD = 1.2
    NORTH_CORRIDOR_MAX_ROWS = 45
    EXIT_ROUTE_EDGE_REWARD = 1.0
    EXIT_ROUTE_REVERSE_PENALTY = 0.0
    EXIT_ROUTE_REPEAT2_PENALTY = 0.0
    EXIT_ROUTE_REPEAT3_PENALTY = 0.0
    EXIT_ROUTE_CONFIRM_AGENTS = 2
    EXIT_ROUTE_MAX_EDGES = 256

    # V10.33 WEGE-GEDAECHTNIS: dieselbe Mechanik wie EXIT_ROUTE, aber fuer den
    # gesamten Weg NACH dem Haus (Eichs Labor -> raus, Alabastia -> Route 1 ->
    # Vertania). Der erste Agent, der durchkommt, legt die Kanten ab; sobald
    # >=2 Agenten dieselbe Kante bestaetigen, bekommt JEDER Agent eine
    # Brotkrumen-Spur (+Reward pro erstem Begehen, Strafe fuer Zurueck/Loop).
    JOURNEY_ROUTE_EDGE_REWARD = 1.5
    JOURNEY_ROUTE_REVERSE_PENALTY = 0.0
    JOURNEY_ROUTE_REPEAT2_PENALTY = 0.0
    JOURNEY_ROUTE_REPEAT3_PENALTY = 0.0
    JOURNEY_ROUTE_CONFIRM_AGENTS = 2
    JOURNEY_ROUTE_MAX_EDGES = 400
    BATTLE_BLOCKED_START_PENALTY = 0.0

    # Early-game Safety / kurze, dichte Lern-Episoden.
    OUTDOOR_CONFIRM_READS = 3
    INTRO_TIMEOUT_STEPS = 1800
    STAIRS_TIMEOUT_STEPS = 1800
    EXIT_TIMEOUT_STEPS = 7500
    EARLY_HOUSE_HARD_CAP = 12000

    # Dynamic FireRed SaveBlock RAM is relatively expensive to copy.
    # Exploration position is sampled every 4 agent steps (~32 emulator frames).
    LOCATION_READ_EVERY = 4
    LOCATION_DISCOVERY_EVERY = 512

    # V7.3.2 Performance:
    # RAM cadence bleibt gleich; nur Python-Navigation wird gecacht.
    NAV_TARGET_REFRESH_EVERY = 8
    SHARED_SNAPSHOT_EVERY = 256
    EXPLORATION_SAVE_EVERY = 512

    # FireRed: Bank 3 = Towns/Routes (Aussenwelt).
    OVERWORLD_BANK = 3

    # V14: EINE WAHRHEIT fuer "wie weit sind wir". Aus (bank,map) + Badges.
    #   0 = Alabastia-Innen / Intro (noch nie Bank!=4)
    #   1 = Alabastia aussen (3,0)
    #   2 = Route 1 (3,19)
    #   3 = jenseits Route 1 (Vertania City) - 1 neue Overworld-Map nach Route 1
    #   4 = Route 2                           - 2 neue
    #   5 = Vertania-Wald / weiter            - 3+ neue
    #   6 = >= 1 Orden
    # Champion, Skill-Vault, Checkpoints, Journey-Routen bewerten NUR world_stage.
    STAGE_PALLET = (3, 0)
    STAGE_ROUTE1 = (3, 19)

    def __init__(
        self,
        rank=0,
        shared_edges=None,
        shared_maps=None,
        shared_transitions=None,
        shared_progress=None,
        shared_lock=None,
        n_envs=120,
    ):
        super().__init__()

        self.rank = rank
        # V13.2: Flottengroesse. _agent_role / _choose_episode_start /
        # _is_long_full_probe verteilen die Rollen relativ dazu, statt feste
        # Slot-Zahlen fuer genau 120 Envs zu hardcoden.
        self.n_envs = max(1, int(n_envs))
        self.shared_edges = shared_edges
        self.shared_maps = shared_maps
        self.shared_transitions = shared_transitions
        self.shared_progress = shared_progress
        self.shared_lock = shared_lock
        self.shared_edge_snapshot = set()
        self.shared_transition_snapshot = set()

        # V7.3.2 Navigation caches.
        # Revision wird nur erhoeht, wenn sich die bekannte Graph-Struktur aendert.
        self.navigation_revision = 0
        self._adjacency_cache = {}
        self._frontier_cache = {}
        self._transition_target_cache = {}
        self._distance_field_cache = {}
        self._nav_target_cache = None
        self._nav_target_cache_step = -999999
        self._last_exploration_save_step = -999999
        self.training_objective = "full"
        self.full_chain_ready = False
        self.objective_success = False
        self.last_gameplay_ready = False
        self.last_in_battle = 0
        self.episode_battles_started = 0
        self.episode_battles_completed = 0
        self.enemy_party_cache = []
        self._last_enemy_seen_step = -999
        self.player_party_cache = []
        self.last_party_total_hp = 0
        self.faints_in_current_battle = 0
        self.last_party_size = 0
        self.last_party_total_level = 0
        self.indoor_steps_without_transition = 0
        self.battle_activity_open = False
        self.enemy_hp_min = {}
        self.enemy_fainted_rewarded = set()
        self.episode_enemy_damage_hp = 0
        self.episode_enemy_damage_reward = 0.0
        self.episode_enemy_faints = 0
        self.journey_seen_starter = False
        self.journey_seen_map5 = False
        self.journey_seen_map10 = False
        self.journey_seen_warp5 = False
        self.journey_seen_progress_checkpoint = False
        self.journey_seen_badge1 = False
        self.start_spam_count = 0
        self.last_start_step = -999999
        self.rank_state_dir = os.path.join(
            CURRICULUM_DIR,
            f"agent_{self.rank:02d}"
        )
        os.makedirs(self.rank_state_dir, exist_ok=True)

        retro.data.Integrations.add_custom_path(CUSTOM_DIR)

        self.env = retro.make(
            game="PokemonFireRed-Gba",
            state=retro.State.NONE,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode=None
        )

        self.env.auto_render = False
        if hasattr(self.env, "viewer"):
            self.env.viewer = None

        self.btn_list = list(self.env.buttons)
        self.num_buttons = len(self.btn_list)

        def get_btn_mask(name):
            mask = [0] * self.num_buttons
            if name in self.btn_list:
                mask[self.btn_list.index(name)] = 1
            return mask

        self.btn_none = [0] * self.num_buttons

        # V6: NONE ist nicht lernbar. Neutrale Release-Frames bleiben.
        # START bleibt voll verfuegbar fuer Menues.
        self.action_map = [
            get_btn_mask("A"),
            get_btn_mask("B"),
            get_btn_mask("START"),
            get_btn_mask("UP"),
            get_btn_mask("DOWN"),
            get_btn_mask("LEFT"),
            get_btn_mask("RIGHT"),
        ]

        self.action_space = spaces.Discrete(len(self.action_map))

        # V7: Policy sieht Bild + kompakten RAM/Navigationszustand.
        # Channel-first verhindert automatische Transpose-Magie in SB3.
        self.NAV_DIM = 28
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(1, 64, 64),
                dtype=np.uint8
            ),
            "nav": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.NAV_DIM,),
                dtype=np.float32
            ),
        })

        self.total_steps = 0
        self.seen_coords = set()
        self.visited_maps = set()
        self._saved_outdoor_depth = 0

        # V10.3 non-persistent learning memory
        self.learning_seen_maps = set()
        self.learning_seen_edges = set()
        self.learning_seen_transitions = set()
        self.recent_path = []

        # Persistentes Explorationsgedaechtnis (bleibt ueber Episoden erhalten).
        self.persistent_known_edges = set()
        self.persistent_known_maps = set()
        self.persistent_known_transitions = set()
        self.exploration_memory_dirty = False
        self.last_exploration_coord = None
        self.last_exploration_map = None
        self.episode_edge_visits = {}
        self.steps_since_new_edge = 0
        self.last_progress_advance_step = 0
        self.last_progress_checkpoint_step = -999999
        self.progress_checkpoint_index = 0
        self.last_exit_seek_distance = None
        self._load_exploration_memory()

        self.current_reward = 0.0
        self.last_level = 0
        self.last_badges = 0
        self.has_starter = False
        # V10.31: die echte Wand ist "raus aus Eichs Labor nach dem Starter"
        # (nicht Route 1). Dafuer gab es bisher keinen Reward-Gradienten.
        self.starter_outdoor_rewarded = False
        self.starter_obtained_step = None

        # Persistente Lernstatistik: bleibt ueber Episoden erhalten.
        self.completed_episodes = 0
        self.total_episode_reward = 0.0
        self.best_episode_reward = None
        self.anti_loop_resets = 0
        self.reward_event_counts = {
            "intro_state": 0,
            "intro_complete": 0,
            "stairs_down": 0,
            "stairs_back": 0,
            "left_house": 0,
            "outdoor_first_step": 0,
            "north_progress": 0,
            "north_to_grass": 0,
            "first_pokemon": 0,
            "next_outdoor_map": 0,
            "level_up": 0,
            "badge": 0,
        }
        self.episode_milestone_steps = {}

        # Saubere, persistente Erfolgsstatistik. Beginning-Runs werden getrennt
        # von Curriculum-Runs ausgewertet, damit Intro/Haus-Quoten nicht durch
        # Episoden verfälscht werden, die das Intro gar nicht spielen.
        self.stats_file = os.path.join(
            STATS_DIR, f"agent_{self.rank:02d}.json"
        )
        self.run_stats = {
            "all_episodes": 0,
            "beginning_episodes": 0,
            "curriculum_episodes": 0,
            "beginning_intro_complete": 0,
            "beginning_stairs_down": 0,
            "beginning_left_house": 0,
            "beginning_grass": 0,
            "beginning_starter": 0,
            "beginning_next_map": 0,
            "beginning_loop_resets": 0,
            "curriculum_loop_resets": 0,

            # Training-V2 Skill-Statistik
            "v2_intro_episodes": 0,
            "v2_intro_success": 0,
            "v2_stairs_episodes": 0,
            "v2_stairs_success": 0,
            "v2_exit_episodes": 0,
            "v2_exit_success": 0,
            "v2_full_episodes": 0,
            "v2_full_intro": 0,
            "v2_full_stairs": 0,
            "v2_full_left_house": 0,
            "v2_full_grass": 0,
            "v2_full_starter": 0,
            "v7_progress_episodes": 0,
            "v7_progress_badge1": 0,
            "v7_full_episodes": 0,
            "v7_full_badge1": 0,
            "v8_starter_episodes": 0,
            "v8_starter_success": 0,
            "v8_battle_episodes": 0,
            "v8_battle_success": 0,
            "v8_level_episodes": 0,
            "v8_level_success": 0,
            "v8_badge_episodes": 0,
            "v8_badge_success": 0,
            "battles_started": 0,
            "battles_completed": 0,
            "journey_starter": 0,
            "journey_map5": 0,
            "journey_map10": 0,
            "journey_warp5": 0,
            "journey_progress_checkpoint": 0,
            "journey_badge1": 0,
            "global_depth_records": 0,
            "enemy_damage_hp": 0,
            "enemy_damage_reward": 0.0,
            "enemy_faints": 0,
        }
        self.episode_anti_loop_resets = 0
        self._load_run_stats()

        # Visuelles Intro-Shaping: funktioniert auch bevor FireRed eine
        # verlaessliche Weltposition im SaveBlock bereitstellt.
        self.intro_seen_states = set()
        self.intro_last_thumb = None
        self.intro_same_screen_steps = 0
        self.intro_novelty_reward_total = 0.0
        self.intro_complete_rewarded = False

        # Story-Reward-Meilensteine (pro Episode nur einmal).
        self.initial_indoor_map = None
        self.stairs_down_rewarded = False
        self.left_house_rewarded = False
        self.left_house_confirmed = False
        self.outdoor_confirm_reads = 0
        self.last_stage_timeout = None
        self.outdoor_first_step_rewarded = False
        self.outdoor_entry_coord = None
        self.north_grass_rewarded = False
        self.next_outdoor_map_rewarded = False
        self.first_outdoor_map = None
        self.outdoor_entry_y = None
        self.best_north_y = None
        self.north_progress_tiles_rewarded = 0
        self.previous_valid_bank = None
        self.previous_valid_map = None
        self.pending_exit_story_transition = None
        self.previous_valid_x = None
        self.previous_valid_y = None

        self.last_pos = None
        self.stuck_counter = 0
        self.last_progress_signature = None

        self.episode_start = "beginning"
        self.saved_milestones = self._discover_saved_milestones()

        self.cached_loc = {
            "valid": False,
            "source": "init",
            "map_bank": 0,
            "map_id": 0,
            "x_pos": 0,
            "y_pos": 0,
        }

    def _load_run_stats(self):
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    for key in self.run_stats:
                        if key in loaded:
                            self.run_stats[key] = int(loaded[key])
        except Exception:
            pass

    def _save_run_stats(self):
        try:
            tmp = self.stats_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.run_stats, f, separators=(",", ":"))
            os.replace(tmp, self.stats_file)
        except Exception:
            pass

    def _finalize_run_stats(self):
        if self.total_steps <= 0:
            return

        self.run_stats["all_episodes"] += 1

        if self.episode_start == "beginning":
            self.run_stats["beginning_episodes"] += 1
            if self.intro_complete_rewarded:
                self.run_stats["beginning_intro_complete"] += 1
            if self.stairs_down_rewarded:
                self.run_stats["beginning_stairs_down"] += 1
            if self.left_house_rewarded:
                self.run_stats["beginning_left_house"] += 1
            if self.north_grass_rewarded:
                self.run_stats["beginning_grass"] += 1
            if self.has_starter:
                self.run_stats["beginning_starter"] += 1
            if self.next_outdoor_map_rewarded:
                self.run_stats["beginning_next_map"] += 1
            self.run_stats["beginning_loop_resets"] += int(
                self.episode_anti_loop_resets
            )
        else:
            self.run_stats["curriculum_episodes"] += 1
            self.run_stats["curriculum_loop_resets"] += int(
                self.episode_anti_loop_resets
            )

        # V2: Erfolg immer gegen das tatsaechliche Trainingsziel messen.
        if self.training_objective == "intro":
            self.run_stats["v2_intro_episodes"] += 1
            if self.intro_complete_rewarded:
                self.run_stats["v2_intro_success"] += 1

        elif self.training_objective == "stairs":
            self.run_stats["v2_stairs_episodes"] += 1
            if self.stairs_down_rewarded:
                self.run_stats["v2_stairs_success"] += 1

        elif self.training_objective == "exit":
            self.run_stats["v2_exit_episodes"] += 1
            if self.left_house_confirmed:
                self.run_stats["v2_exit_success"] += 1

        elif self.training_objective == "full" and self.episode_start == "beginning":
            self.run_stats["v2_full_episodes"] += 1
            self.run_stats["v7_full_episodes"] += 1
            if self.intro_complete_rewarded:
                self.run_stats["v2_full_intro"] += 1
            if self.stairs_down_rewarded:
                self.run_stats["v2_full_stairs"] += 1
            if self.left_house_confirmed:
                self.run_stats["v2_full_left_house"] += 1
            if self.north_grass_rewarded:
                self.run_stats["v2_full_grass"] += 1
            if self.has_starter:
                self.run_stats["v2_full_starter"] += 1
            if self.last_badges >= 1:
                self.run_stats["v7_full_badge1"] += 1

        elif self.training_objective == "starter":
            self.run_stats["v8_starter_episodes"] += 1
            if self.objective_success or self.has_starter:
                self.run_stats["v8_starter_success"] += 1

        elif self.training_objective == "battle":
            self.run_stats["v8_battle_episodes"] += 1
            if self.objective_success or self.episode_enemy_faints > 0:
                self.run_stats["v8_battle_success"] += 1

        elif self.training_objective == "level":
            self.run_stats["v8_level_episodes"] += 1
            if self.objective_success:
                self.run_stats["v8_level_success"] += 1

        elif self.training_objective == "badge":
            self.run_stats["v8_badge_episodes"] += 1
            if self.objective_success or self.last_badges >= 1:
                self.run_stats["v8_badge_success"] += 1

        elif self.training_objective == "progress":
            self.run_stats["v7_progress_episodes"] += 1
            if self.last_badges >= 1:
                self.run_stats["v7_progress_badge1"] += 1

        self._save_run_stats()

    def _claim_journey_milestone(self, key, attr_name):
        if getattr(self, attr_name, False):
            return
        setattr(self, attr_name, True)
        self.run_stats[key] = int(self.run_stats.get(key, 0)) + 1
        self._save_run_stats()

    def _process_image(self, screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (64, 64),
            interpolation=cv2.INTER_NEAREST
        )
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def _policy_objective(self):
        # V10.13 FULL-POLICY UNIFICATION:
        # Story specialists keep their own rewards and curriculum starts,
        # but they all train the exact Full-Journey policy context used by
        # the Watcher. This prevents stairs/exit/starter skills from being
        # trapped behind different objective one-hot inputs.
        if self.training_objective in (
            "intro", "stairs", "exit", "starter", "battle",
            "level", "progress", "badge", "full"
        ):
            return "full"
        return self.training_objective

    def _objective_one_hot(self):
        names = (
            "intro", "stairs", "exit", "starter",
            "battle", "level", "progress", "badge", "full"
        )
        policy_objective = self._policy_objective()
        return [
            1.0 if policy_objective == name else 0.0
            for name in names
        ]

    def _invalidate_navigation_cache(self):
        self.navigation_revision += 1
        self._adjacency_cache.clear()
        self._frontier_cache.clear()
        self._transition_target_cache.clear()
        self._distance_field_cache.clear()
        self._nav_target_cache = None
        self._nav_target_cache_step = -999999

    def _combined_edges(self):
        # Keine unnoetigen set()-Kopien wenn eine Seite leer ist.
        if not self.shared_edge_snapshot:
            return self.persistent_known_edges
        if not self.persistent_known_edges:
            return self.shared_edge_snapshot
        return self.persistent_known_edges | self.shared_edge_snapshot

    def _confirmed_warp_dir(self):
        path = os.path.join(SHARED_CURRICULUM_DIR, "confirmed_story_warps")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_confirmed_story_warp(self, kind, transition):
        if kind not in ("stairs", "exit"):
            return
        if not transition or len(transition) != 8:
            return

        path = os.path.join(
            self._confirmed_warp_dir(),
            f"agent_{self.rank:03d}_{kind}.json"
        )
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "agent": int(self.rank),
                        "kind": kind,
                        "transition": [int(v) for v in transition],
                    },
                    f,
                    separators=(",", ":"),
                )
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _load_confirmed_story_warps(self, kind):
        votes = {}
        try:
            names = os.listdir(self._confirmed_warp_dir())
        except Exception:
            return set()

        suffix = f"_{kind}.json"
        for name in names:
            if not name.startswith("agent_") or not name.endswith(suffix):
                continue
            try:
                with open(os.path.join(self._confirmed_warp_dir(), name), "r") as f:
                    data = json.load(f)
                raw = data.get("transition", [])
                if isinstance(raw, list) and len(raw) == 8:
                    t = tuple(int(v) for v in raw)
                    votes[t] = votes.get(t, 0) + 1
            except Exception:
                pass

        return {
            t for t, count in votes.items()
            if count >= self.CONFIRMED_WARP_MIN_AGENTS
        }

    def _combined_transitions(self):
        # V10.11 NIGHT ROUTE FIX
        # Known transitions must be visible before the house is left.
        # _target_coords_for_stage filters bedroom to indoor targets and
        # F1/other indoor rooms to overworld targets.
        if not self.shared_transition_snapshot:
            return self.persistent_known_transitions
        if not self.persistent_known_transitions:
            return self.shared_transition_snapshot
        return self.persistent_known_transitions | self.shared_transition_snapshot

    def _adjacency_for_map(self, bank, map_id):
        key = (
            self.navigation_revision,
            int(bank),
            int(map_id),
        )
        cached = self._adjacency_cache.get(key)
        if cached is not None:
            return cached

        adjacency = {}
        nodes = set()
        for e in self._combined_edges():
            if len(e) != 6:
                continue
            eb, em, x1, y1, x2, y2 = e
            if int(eb) != int(bank) or int(em) != int(map_id):
                continue
            a = (int(x1), int(y1))
            b = (int(x2), int(y2))
            nodes.add(a)
            nodes.add(b)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        self._adjacency_cache[key] = (adjacency, nodes)
        return adjacency, nodes

    def _frontiers_for_map(self, bank, map_id):
        key = (
            self.navigation_revision,
            int(bank),
            int(map_id),
        )
        cached = self._frontier_cache.get(key)
        if cached is not None:
            return cached

        adjacency, nodes = self._adjacency_for_map(bank, map_id)
        frontiers = tuple(
            p for p in nodes
            if len(adjacency.get(p, ())) < 4
        )
        self._frontier_cache[key] = frontiers
        return frontiers

    def _progress_targets_for_map(self, bank, map_id, x, y):
        """
        V7.3.2:
        Graph/Frontier-Struktur wird pro Map + Revision nur einmal gebaut.
        Nur die billige Entfernungssortierung bleibt positionsabhaengig.
        """
        map_key = (int(bank), int(map_id))

        # V14: auf den Nord-Korridor-Maps (Alabastia, Route 1) KEIN generisches
        # Ziel. Die Frontier-/Warp-Auswahl koennte ein Haus als "naechstes Ziel"
        # nehmen; die +-2.5 Ziel-Formung wuerde dann die Nord-Rampe ueberstimmen
        # ("zum Gebaeude" statt "nach Vertania"). Auf diesen Maps ist die
        # Nord-Rampe der ALLEINIGE Gradient.
        if map_key in self.NORTH_CORRIDOR_MAPS:
            return []

        warp_targets = []
        for t in self._combined_transitions():
            if len(t) != 8:
                continue
            a = (
                int(t[0]), int(t[1]),
                int(t[2]), int(t[3])
            )
            b = (
                int(t[4]), int(t[5]),
                int(t[6]), int(t[7])
            )

            if (a[0], a[1]) == map_key:
                if (b[0], b[1]) not in self.visited_maps:
                    warp_targets.append((a[2], a[3]))

            if (b[0], b[1]) == map_key:
                if (a[0], a[1]) not in self.visited_maps:
                    warp_targets.append((b[2], b[3]))

        if warp_targets:
            return list(dict.fromkeys(warp_targets))

        frontiers = self._frontiers_for_map(bank, map_id)
        if not frontiers:
            return []

        # Kein komplettes sort() mehr: maximal acht beste Kandidaten
        # werden in einem kleinen laufenden Puffer gehalten.
        nearest = sorted(
            frontiers,
            key=lambda p: abs(p[0] - x) + abs(p[1] - y)
        )[:8]
        return nearest

    def _nav_target(self, bank, map_id, x, y):
        cache_key = (
            self.navigation_revision,
            self.training_objective,
            int(bank), int(map_id),
        )

        cached = self._nav_target_cache
        if (
            cached is not None
            and cached[0] == cache_key
            and (
                self.total_steps - self._nav_target_cache_step
                < self.NAV_TARGET_REFRESH_EVERY
            )
        ):
            return cached[1]

        targets = self._target_coords_for_stage(bank, map_id)

        if (
            not targets
            and self.left_house_rewarded
            and self.training_objective in ("progress", "full")
            and self._valid_coord(bank, map_id, x, y)
        ):
            targets = self._progress_targets_for_map(
                bank, map_id, x, y
            )

        if not targets:
            target = None
        else:
            target = min(
                targets,
                key=lambda p: abs(p[0] - x) + abs(p[1] - y)
            )

        self._nav_target_cache = (cache_key, target)
        self._nav_target_cache_step = self.total_steps
        return target

    def _build_nav_vector(
        self,
        bank,
        map_id,
        x,
        y,
        gameplay_ready,
        in_battle,
        p_lvl,
        badges,
    ):
        vec = list(self._objective_one_hot())

        vec.extend([
            1.0 if gameplay_ready else 0.0,
            1.0 if in_battle else 0.0,
            1.0 if self.stairs_down_rewarded else 0.0,
            1.0 if self.left_house_confirmed else 0.0,
            1.0 if self.has_starter else 0.0,
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
            self._nav_target(bank, map_id, x, y)
            if gameplay_ready else None
        )

        if target is None:
            vec.extend([0.0, 0.0, 0.0, 0.0])
        else:
            dx = int(target[0]) - int(x)
            dy = int(target[1]) - int(y)
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

        party = self.player_party_cache or []
        party_levels = [
            int(m.get("level", 0))
            for m in party
            if int(m.get("level", 0)) > 0
        ]
        party_size = len(party_levels)
        party_max_level = max(party_levels) if party_levels else 0
        party_avg_level = (
            sum(party_levels) / len(party_levels)
            if party_levels else 0.0
        )
        hp_values = [
            float(m.get("hp_ratio", 0.0))
            for m in party
            if int(m.get("max_hp", 0)) > 0
        ]
        party_hp_ratio = (
            sum(hp_values) / len(hp_values)
            if hp_values else 0.0
        )
        vec.extend([
            float(np.clip(party_size / 6.0, 0.0, 1.0)),
            float(np.clip(party_max_level / 100.0, 0.0, 1.0)),
            float(np.clip(party_avg_level / 100.0, 0.0, 1.0)),
            float(np.clip(party_hp_ratio, 0.0, 1.0)),
        ])

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (self.NAV_DIM,):
            raise RuntimeError(
                f"V7 nav shape {arr.shape} != {(self.NAV_DIM,)}"
            )
        return arr

    def _make_obs(
        self,
        screen,
        loc=None,
        info=None,
    ):
        loc = loc or self.cached_loc or {}
        info = info or {}

        valid = bool(
            loc.get("valid", False)
            and loc.get("trusted", False)
        )

        bank = int(loc.get("map_bank", 0)) if valid else 0
        map_id = int(loc.get("map_id", 0)) if valid else 0
        x = int(loc.get("x_pos", 0)) if valid else 0
        y = int(loc.get("y_pos", 0)) if valid else 0

        gameplay_ready = bool(
            valid and self._valid_coord(bank, map_id, x, y)
        )

        badges_raw = int(info.get("badges", self.last_badges))
        # Das RAM-Feld ist eine Bitmaske, kein bereits gezaehlter Wert.
        badges = (
            bin(badges_raw).count("1")
            if badges_raw > 0
            else 0
        )

        return {
            "image": self._process_image(screen),
            "nav": self._build_nav_vector(
                bank=bank,
                map_id=map_id,
                x=x,
                y=y,
                gameplay_ready=gameplay_ready,
                in_battle=int(info.get("in_battle", self.last_in_battle)),
                p_lvl=int(info.get("p1_level", self.last_level)),
                badges=int(badges),
            ),
        }

    @staticmethod
    def _intro_thumb(screen_rgb):
        """Kleiner robuster Screen-Fingerprint fuer Intro/Menu-Fortschritt."""
        gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(
            gray,
            (12, 8),
            interpolation=cv2.INTER_AREA
        ).astype(np.int16)

    @staticmethod
    def _valid_coord(bank, map_id, x, y):
        if bank == 0 and map_id == 0 and x == 0 and y == 0:
            return False
        return 0 <= x < 512 and 0 <= y < 512

    def _state_path(self, milestone_name):
        safe = "".join(
            c if c.isalnum() or c in ("_", "-") else "_"
            for c in milestone_name
        )
        return os.path.join(
            self.rank_state_dir,
            f"{safe}.state.gz"
        )

    def _shared_state_path(self, milestone_name):
        safe = "".join(
            c if c.isalnum() or c in ("_", "-") else "_"
            for c in milestone_name
        )
        return os.path.join(
            SHARED_CURRICULUM_DIR,
            f"{safe}.state.gz"
        )

    def _discover_saved_milestones(self):
        states = set()

        for directory in (self.rank_state_dir, SHARED_CURRICULUM_DIR):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                if name.endswith(".state.gz"):
                    states.add(name[:-9])

        return sorted(states)

    # ------------------------------------------------------------------
    # V14: world_stage = einzige Fortschritts-Wahrheit
    # ------------------------------------------------------------------
    def _world_stage(self):
        """0=Alabastia-Innen/Intro, 1=Alabastia aussen, 2=Route 1,
        3=Vertania, 4=Route 2, 5=Wald/weiter, 6=>=1 Orden."""
        if int(getattr(self, "last_badges", 0) or 0) >= 1:
            return 6
        ow = {m for m in self.visited_maps if int(m[0]) != 4}
        if not ow:
            return 0
        beyond = [
            m for m in ow
            if m != self.STAGE_PALLET and m != self.STAGE_ROUTE1
        ]
        if beyond:
            return min(5, 2 + len(beyond))
        if self.STAGE_ROUTE1 in ow:
            return 2
        return 1

    def _stage_meta_path(self, name, shared=True):
        d = SHARED_CURRICULUM_DIR if shared else self.rank_state_dir
        return os.path.join(d, name + ".meta.json")

    def _read_stage_meta(self, name):
        for shared in (True, False):
            try:
                with open(self._stage_meta_path(name, shared)) as f:
                    return json.load(f) or {}
            except Exception:
                continue
        return {}

    def _write_stage_meta(self, name, meta):
        for shared in (True, False):
            try:
                p = self._stage_meta_path(name, shared)
                with open(p + ".tmp", "w") as f:
                    json.dump(meta, f)
                os.replace(p + ".tmp", p)
            except Exception:
                pass

    def _valid_stage_checkpoints(self):
        """{n: 'stage_n'} - nur Checkpoints mit gueltiger Sidecar-Meta
        (Stage passt, Starter vorhanden, Overworld-Bank)."""
        out = {}
        for name in getattr(self, "saved_milestones", ()) or ():
            if not name.startswith("stage_"):
                continue
            try:
                n = int(name.split("_", 1)[1])
            except Exception:
                continue
            meta = self._read_stage_meta(name)
            if (int(meta.get("stage", -1)) == n
                    and bool(meta.get("has_starter"))
                    and int(meta.get("bank", -1)) == self.OVERWORLD_BANK):
                out[n] = name
        return out

    def _save_stage_checkpoint(self, stage, bank, map_id, x, y):
        name = f"stage_{int(stage)}"
        meta = {
            "stage": int(stage), "bank": int(bank), "map": int(map_id),
            "x": int(x), "y": int(y), "has_starter": True,
            "step": int(self.total_steps), "agent": int(self.rank),
        }
        existing = self._read_stage_meta(name)
        ok_existing = (
            int(existing.get("stage", -1)) == int(stage)
            and bool(existing.get("has_starter"))
            and int(existing.get("bank", -1)) == self.OVERWORLD_BANK
            and os.path.exists(self._shared_state_path(name))
        )
        if ok_existing:
            self._write_stage_meta(name, meta)
            return True
        for p in (self._state_path(name), self._shared_state_path(name)):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if self._save_curriculum_state(name):
            self._write_stage_meta(name, meta)
            return True
        return False

    def _save_curriculum_state(self, milestone_name):
        local_path = self._state_path(milestone_name)
        shared_path = self._shared_state_path(milestone_name)

        saved_any = False

        try:
            state_data = self.env.em.get_state()

            if not os.path.exists(local_path):
                tmp_local = (
                    local_path
                    + f".tmp.{os.getpid()}.{self.rank}"
                )
                with gzip.open(tmp_local, "wb") as f:
                    f.write(state_data)
                os.replace(tmp_local, local_path)
                saved_any = True

            # Globaler Curriculum-Bank: sobald EIN Agent einen Meilenstein
            # schafft, koennen ALLE 30 Agents davon lernen.
            if not os.path.exists(shared_path):
                tmp_shared = (
                    shared_path
                    + f".tmp.{os.getpid()}.{self.rank}"
                )
                with gzip.open(tmp_shared, "wb") as f:
                    f.write(state_data)
                try:
                    # Race ist harmlos: der letzte vollstaendige State gewinnt.
                    os.replace(tmp_shared, shared_path)
                    saved_any = True
                except Exception:
                    try:
                        os.remove(tmp_shared)
                    except Exception:
                        pass

            self.saved_milestones = self._discover_saved_milestones()
            return saved_any

        except Exception as e:
            print(
                f"[Agent {self.rank:02d}] Curriculum-State "
                f"{milestone_name} konnte nicht gespeichert werden: {e}"
            )
            return False

    def _load_curriculum_state(self, milestone_name):
        paths = [
            self._shared_state_path(milestone_name),
            self._state_path(milestone_name),
        ]

        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with gzip.open(path, "rb") as f:
                    state_data = f.read()
                self.env.em.set_state(state_data)
                return True
            except Exception:
                continue

        return False

    def _claim_shared(self, registry, key):
        """True nur beim allerersten globalen Fund ueber alle 30 Agents."""
        if registry is None:
            return True

        try:
            if key in registry:
                return False

            if self.shared_lock is not None:
                self.shared_lock.acquire()

            try:
                if key in registry:
                    return False
                registry[key] = 1
                return True
            finally:
                if self.shared_lock is not None:
                    self.shared_lock.release()
        except Exception:
            # Fail-open: Training soll bei IPC-Problemen nicht stehenbleiben.
            return True

    def _claim_global_depth(self, stage):
        """V14: True nur wenn dieser Agent erstmals den globalen world_stage-
        Rekord aller laufenden Agents uebertrifft. Nicht farmbar (shared lock).
        """
        stage = int(stage)
        if stage <= 0 or self.shared_progress is None:
            return False

        try:
            if self.shared_lock is not None:
                self.shared_lock.acquire()

            try:
                current = int(
                    self.shared_progress.get("max_world_stage", 0)
                )
                if stage <= current:
                    return False

                self.shared_progress["max_world_stage"] = stage
                try:
                    tmp = GLOBAL_PROGRESS_FILE + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(
                            {"max_world_stage": stage},
                            f,
                            separators=(",", ":")
                        )
                    os.replace(tmp, GLOBAL_PROGRESS_FILE)
                except Exception:
                    pass
                return True
            finally:
                if self.shared_lock is not None:
                    self.shared_lock.release()

        except Exception:
            # Fail-closed: ein IPC-Problem darf keinen farmbaren Bonus erzeugen.
            return False

    def _refresh_shared_snapshots(self):
        old_edges = self.shared_edge_snapshot
        old_transitions = self.shared_transition_snapshot

        try:
            new_edges = set(self.shared_edges) if self.shared_edges is not None else set()
        except Exception:
            new_edges = old_edges

        try:
            new_transitions = (
                set(self.shared_transitions)
                if self.shared_transitions is not None
                else set()
            )
        except Exception:
            new_transitions = old_transitions

        changed = (
            new_edges != old_edges
            or new_transitions != old_transitions
        )

        self.shared_edge_snapshot = new_edges
        self.shared_transition_snapshot = new_transitions

        if changed:
            self._invalidate_navigation_cache()


    def _exploration_memory_path(self):
        return os.path.join(
            EXPLORATION_MEMORY_DIR,
            f"agent_{self.rank:02d}.json"
        )

    @staticmethod
    def _edge_key(bank, map_id, x1, y1, x2, y2):
        a = (int(x1), int(y1))
        b = (int(x2), int(y2))
        if b < a:
            a, b = b, a
        return (
            int(bank), int(map_id),
            a[0], a[1], b[0], b[1]
        )

    @staticmethod
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
        # Undirected: durch dieselbe Tuer zurueck gibt keinen zweiten Reward.
        if b < a:
            a, b = b, a
        return a + b

    def _known_transition_targets_for_map(
        self,
        bank,
        map_id,
        require_overworld=False,
        require_indoor=False,
    ):
        cache_key = (
            self.navigation_revision,
            int(bank),
            int(map_id),
            bool(require_overworld),
            bool(require_indoor),
        )
        cached = self._transition_target_cache.get(cache_key)
        if cached is not None:
            return cached

        targets = []
        key = (int(bank), int(map_id))

        for t in self._combined_transitions():
            if len(t) != 8:
                continue

            a = (
                int(t[0]), int(t[1]),
                int(t[2]), int(t[3])
            )
            b = (
                int(t[4]), int(t[5]),
                int(t[6]), int(t[7])
            )

            for here, other in ((a, b), (b, a)):
                if (here[0], here[1]) != key:
                    continue
                if (
                    require_overworld
                    and other[0] != self.OVERWORLD_BANK
                ):
                    continue
                if (
                    require_indoor
                    and other[0] == self.OVERWORLD_BANK
                ):
                    continue
                targets.append((here[2], here[3]))

        result = list(dict.fromkeys(targets))
        self._transition_target_cache[cache_key] = result
        return result

    # V10.17_1_STARTER_STORY_GUARD_SAFE
    def _v10171_party_has_starter(self):
        try:
            step_no = int(getattr(self, "total_steps", 0) or 0)
        except Exception:
            step_no = 0

        cached = bool(getattr(self, "_v10171_has_starter_cached", False))
        if cached:
            return True

        last = int(getattr(self, "_v10171_party_check_step", -999999))
        if step_no - last < 24:
            return False

        self._v10171_party_check_step = step_no
        try:
            party = read_player_party(self.env)
            if party:
                self.player_party_cache = list(party)
            good = [
                mon for mon in (party or [])
                if int(mon.get("level", 0) or 0) >= 5
                and int(mon.get("max_hp", 0) or 0) > 0
            ]
            if good:
                self._v10171_has_starter_cached = True
                return True
        except Exception:
            pass
        return False

    def _v10171_story_guard(self, bank, map_id, x, y, reward):
        try:
            if not bool(getattr(self, "left_house_rewarded", False)):
                return reward
            if self._v10171_party_has_starter():
                return reward

            coord = (int(bank), int(map_id), int(x), int(y))
            hist = getattr(self, "_v10171_post_house_hist", None)
            if hist is None:
                hist = []
                self._v10171_post_house_hist = hist
            hist.append(coord)
            if len(hist) > 96:
                del hist[:-96]

            seen = getattr(self, "_v10171_post_house_seen", None)
            if seen is None:
                seen = set()
                self._v10171_post_house_seen = seen

            if coord not in seen:
                seen.add(coord)
                reward += 0.35

            recent = hist[-32:]
            unique = len(set(recent))
            if len(recent) >= 24 and unique <= 8:
                reward -= 0.75
            if len(recent) >= 32 and unique <= 5:
                reward -= 1.25

            if len(hist) >= 12 and len(set(hist[-12:])) <= 3:
                reward -= 1.00

            return reward
        except Exception:
            return reward

    def _target_coords_for_stage(self, bank, map_id):
        # V10.4: raw indoor transitions remain quarantined, but confirmed
        # story warps may guide every role.
        key = (int(bank), int(map_id))

        if not self.stairs_down_rewarded:
            targets = []
            for t in self._load_confirmed_story_warps("stairs"):
                if len(t) != 8:
                    continue
                a = tuple(int(v) for v in t[:4])
                b = tuple(int(v) for v in t[4:])
                if (a[0], a[1]) == key:
                    targets.append((a[2], a[3]))
                if (b[0], b[1]) == key:
                    targets.append((b[2], b[3]))
            if not targets:
                # V10.19.1: confirmed-first, aber nicht confirmed-only.
                # Solange noch kein bestaetigter Story-Warp existiert, darf
                # eine real beobachtete Indoor->Indoor-Transition als
                # Treppen-Ziel dienen. Keine FireRed-Koordinate wird hardcodiert.
                for t in self._combined_transitions():
                    if len(t) != 8:
                        continue
                    a = tuple(int(v) for v in t[:4])
                    b = tuple(int(v) for v in t[4:])
                    if (a[0], a[1]) == key and b[0] != self.OVERWORLD_BANK:
                        targets.append((a[2], a[3]))
                    if (b[0], b[1]) == key and a[0] != self.OVERWORLD_BANK:
                        targets.append((b[2], b[3]))
            return list(dict.fromkeys(targets))

        if not self.left_house_rewarded:
            targets = []
            for t in self._load_confirmed_story_warps("exit"):
                if len(t) != 8:
                    continue
                a = tuple(int(v) for v in t[:4])
                b = tuple(int(v) for v in t[4:])
                if (a[0], a[1]) == key:
                    targets.append((a[2], a[3]))
                if (b[0], b[1]) == key:
                    targets.append((b[2], b[3]))
            return list(dict.fromkeys(targets))

        if bank != self.OVERWORLD_BANK:
            # V11: In einem Gebaeude erst dann Richtung Ausgang ziehen, wenn
            # der Starter schon da ist. Sonst wuerde der Agent in Eichs Labor
            # direkt zur Tuer laufen, statt vorher den Pokeball vom Tisch zu
            # holen. Ohne Starter -> kein Ziel -> reine Exploration (laeuft so
            # von selbst an den Tisch / loest den Auswahl-Dialog aus).
            if self.has_starter:
                return self._known_transition_targets_for_map(
                    bank, map_id, require_overworld=True
                )
            return []

        return []

    def _graph_distance(self, bank, map_id, start_xy, targets):
        if not targets:
            return None

        target_tuple = tuple(sorted(set(targets)))
        if start_xy in target_tuple:
            return 0

        cache_key = (
            self.navigation_revision,
            int(bank),
            int(map_id),
            target_tuple,
        )
        distance_field = self._distance_field_cache.get(cache_key)

        if distance_field is None:
            adjacency, _ = self._adjacency_for_map(bank, map_id)

            # Reverse BFS von allen Targets gleichzeitig.
            distance_field = {}
            queue = []
            for target in target_tuple:
                distance_field[target] = 0
                queue.append(target)

            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                next_dist = distance_field[node] + 1

                for nxt in adjacency.get(node, ()):
                    if nxt in distance_field:
                        continue
                    distance_field[nxt] = next_dist
                    queue.append(nxt)

            self._distance_field_cache[cache_key] = distance_field

        cached_dist = distance_field.get(start_xy)
        if cached_dist is not None:
            return cached_dist

        # Karte hat noch Luecken: Manhattan-Fallback wie vorher.
        sx, sy = start_xy
        return min(
            abs(sx - tx) + abs(sy - ty)
            for tx, ty in target_tuple
        )

    def _load_exploration_memory(self):
        if not self.EXPLORATION_MEMORY_ENABLED:
            return
        path = self._exploration_memory_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.persistent_known_edges = {
                tuple(x) for x in data.get("edges", [])
                if isinstance(x, list) and len(x) == 6
            }
            self.persistent_known_maps = {
                tuple(x) for x in data.get("maps", [])
                if isinstance(x, list) and len(x) == 2
            }
            self.persistent_known_transitions = {
                tuple(x) for x in data.get("transitions", [])
                if isinstance(x, list) and len(x) == 8
            }
        except Exception:
            self.persistent_known_edges = set()
            self.persistent_known_maps = set()
            self.persistent_known_transitions = set()

    def _save_exploration_memory(self, force=False):
        if (
            not self.EXPLORATION_MEMORY_ENABLED
            or (not self.exploration_memory_dirty and not force)
        ):
            return

        if (
            not force
            and (
                self.total_steps - self._last_exploration_save_step
                < self.EXPLORATION_SAVE_EVERY
            )
        ):
            return

        path = self._exploration_memory_path()
        tmp = path + ".tmp"
        try:
            data = {
                "schema": 1,
                "edges": [
                    list(x) for x in self.persistent_known_edges
                ],
                "maps": [
                    list(x) for x in self.persistent_known_maps
                ],
                "transitions": [
                    list(x)
                    for x in self.persistent_known_transitions
                ],
            }
            with open(tmp, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp, path)
            self.exploration_memory_dirty = False
            self._last_exploration_save_step = self.total_steps
        except Exception:
            pass


    def _milestone_number(self, name, prefix):
        try:
            return int(name.split(prefix, 1)[1])
        except Exception:
            return -1

    def _best_progress_milestone(self):
        saved = set(self.saved_milestones)

        badge_states = [
            m for m in saved if m.startswith("badge_")
        ]
        if badge_states:
            return max(
                badge_states,
                key=lambda m: self._milestone_number(m, "badge_")
            )

        # V14: tiefste VALIDIERTE Stage-Front (Sidecar-Meta: Stage/Starter/
        # Overworld-Bank stimmen). Kein blindes outdoor_N mehr - ein Glitch-
        # Save kann so keine Flotte mehr in einen Innenraum ziehen.
        stage_cps = self._valid_stage_checkpoints()
        if stage_cps:
            deepest_n = max(stage_cps)
            # ~85% an die tiefste Front, ~15% frisch von starter_outdoor.
            if "starter_outdoor" in saved and (self.rank % 7 == 0):
                return "starter_outdoor"
            return stage_cps[deepest_n]

        progress_states = [
            m for m in saved if m.startswith("progress_")
        ]
        map_states = [
            m for m in saved if m.startswith("maps_")
        ]
        level_states = [
            m for m in saved if m.startswith("level_")
        ]

        # Hoehere generische Progress-Checkpoints zuerst.
        candidates = []

        candidates.extend(sorted(
            progress_states,
            key=lambda m: self._milestone_number(m, "progress_"),
            reverse=True
        ))

        candidates.extend(sorted(
            map_states,
            key=lambda m: self._milestone_number(m, "maps_"),
            reverse=True
        ))
        candidates.extend(sorted(
            level_states,
            key=lambda m: self._milestone_number(m, "level_"),
            reverse=True
        ))

        # Starter ist chronologisch wertvoller als sehr fruehe maps_3 States.
        if "starter" in saved or "starter_outdoor" in saved:
            high_maps = [
                m for m in candidates
                if m.startswith("maps_")
                and self._milestone_number(m, "maps_") >= 6
            ]
            high_levels = [
                m for m in candidates
                if m.startswith("level_")
                and self._milestone_number(m, "level_") >= 7
            ]
            if high_maps:
                return high_maps[0]
            if high_levels:
                return high_levels[0]
            # V13.3: "draussen mit Starter" ist der Frontier-Startpunkt. ~80%
            # starten dort (raus auf die Route), nur ~20% aus "starter" (im
            # Labor), damit die Labor-Ausgang-Passage weiter geuebt wird - aber
            # nicht mehr die Haelfte der Flotte im Labor parken.
            if "starter_outdoor" in saved and (self.rank % 5 != 0):
                return "starter_outdoor"
            if "starter" in saved:
                return "starter"
            return "starter_outdoor"

        if candidates:
            return candidates[0]

        if "stairs_down" in saved:
            return "stairs_down"
        if "intro_complete" in saved:
            return "intro_complete"
        return "beginning"

    def _maybe_save_progress_bridge(self, reason):
        """
        V7.1: generischer Fortschritts-Checkpoint.
        Wird nur bei echten Fortschrittsereignissen gespeichert.
        Keine hartcodierten FireRed-Koordinaten oder Story-Loesung.
        """
        if self.training_objective not in ("progress", "full"):
            return None

        if (
            self.total_steps - self.last_progress_checkpoint_step
            < self.PROGRESS_CHECKPOINT_COOLDOWN
        ):
            return None

        self.progress_checkpoint_index += 1
        name = f"progress_{self.progress_checkpoint_index}"

        if self._save_curriculum_state(name):
            self.last_progress_checkpoint_step = self.total_steps
            self.last_progress_advance_step = self.total_steps
            self._claim_journey_milestone("journey_progress_checkpoint","journey_seen_progress_checkpoint")
            return name

        return None

    def _is_starter_rusher(self):
        # V10.15: only actual Starter specialists.
        return self.training_objective == "starter"

    def _champion_full_starter_ready(self):
        try:
            path = os.path.join(RUNTIME_DIR, "champion_score.json")
            with open(path, "r") as f:
                data = json.load(f) or {}
            metrics = data.get("metrics") or {}
            return int(metrics.get("full_starter_permille", 0)) > 0
        except Exception:
            return False

    def _skill_vault_scores(self):
        default = {"intro": 0, "stairs": 0, "exit": 0, "starter": 0, "progress": 0}
        try:
            path = os.path.join(RUNTIME_DIR, "skill_vault_scores.json")
            with open(path, "r") as f:
                data = json.load(f) or {}
            for k in default:
                default[k] = int(data.get(k, 0))
            return default
        except Exception:
            return default

    def _agent_role(self):
        # V11 SEQUENTIELLES SKILL-BOOTCAMP:
        # Ein Skill nach dem anderen. Fast die ganze Flotte uebt die aktuelle
        # Stufe (schnelle, saubere Gradienten), ein kleiner Sockel haelt die
        # bereits gelernten Stufen wach + ein paar Probes testen die naechste.
        # Umschalten passiert automatisch, sobald der gemessene Skill-Score
        # (skill_vault_scores.json, 0-1000 = Erfolgsquote x1000) die Schwelle
        # erreicht. Ab "starter >= 800" -> freie Welt-Exploration mit
        # gestaffelten Horizonten (8k/16k/28k).
        n = self.n_envs
        slot = self.rank % n
        s = self._skill_vault_scores()
        DONE = 880  # ~88% Erfolgsquote = Stufe gilt als gelernt

        def _pct(frac, lo):
            # relatives Band: mind. `lo` Slots, sonst `frac` der Flotte
            return max(int(lo), int(round(n * frac)))

        # --- Phase 1: INTRO ---
        if s["intro"] < DONE:
            probe = _pct(0.10, 6)
            if slot < n - probe: return "intro", f"Intro Bootcamp {slot + 1:03d}"
            return "stairs", f"Stairs Probe {slot - (n - probe) + 1:02d}"

        # --- Phase 2: TREPPE ---
        if s["stairs"] < DONE:
            ret = _pct(0.06, 4)
            probe = _pct(0.13, 8)
            if slot < ret: return "intro", f"Intro Retention {slot + 1:02d}"
            if slot < n - probe: return "stairs", f"Stairs Bootcamp {slot - ret + 1:03d}"
            return "exit", f"Exit Probe {slot - (n - probe) + 1:02d}"

        # --- Phase 3: HAUSAUSGANG ---
        if s["exit"] < DONE:
            r1 = _pct(0.03, 3)
            r2 = r1 + _pct(0.08, 6)
            probe = _pct(0.11, 8)
            if slot < r1: return "intro", f"Intro Retention {slot + 1:02d}"
            if slot < r2: return "stairs", f"Stairs Retention {slot - r1 + 1:02d}"
            if slot < n - probe: return "exit", f"Exit Bootcamp {slot - r2 + 1:03d}"
            return "starter", f"Starter Probe {slot - (n - probe) + 1:02d}"

        # --- Phase 4: STARTER (inkl. raus aus Eichs Labor) ---
        if s["starter"] < DONE:
            r1 = _pct(0.03, 3)
            r2 = r1 + _pct(0.06, 6)
            r3 = r2 + _pct(0.08, 8)
            probe = _pct(0.10, 8)
            if slot < r1: return "intro", f"Intro Retention {slot + 1:02d}"
            if slot < r2: return "stairs", f"Stairs Retention {slot - r1 + 1:02d}"
            if slot < r3: return "exit", f"Exit Retention {slot - r2 + 1:02d}"
            if slot < n - probe: return "starter", f"Starter Bootcamp {slot - r3 + 1:03d}"
            return "progress", f"World Probe {slot - (n - probe) + 1:02d}"

        # --- Phase 5: FREIE WELT-EXPLORATION + KAMPF ---
        # Frueh-Skills minimaler Erhaltungs-Sockel. Loewenanteil in progress
        # (Route/Wald erkunden). battle+level fuer wilde Pokemon + Brock.
        # full klein, aber NICHT null: nur Full-from-Beginning-Runs koennen den
        # Champion befoerdern (Frontier/Milestone/Recent-Eval) und den Rollback-
        # Punkt frisch halten. Ohne full friert der Champion fuer immer ein.
        b_intro   = 2
        b_stairs  = b_intro + 3
        b_exit    = b_stairs + 3
        b_starter = b_exit + 4
        b_battle  = b_starter + _pct(0.11, 8)
        b_level   = b_battle + _pct(0.07, 6)
        b_full    = n - _pct(0.05, 4)          # letzte ~5% = Full Journey
        if slot < b_intro:   return "intro", f"Intro Vault {slot + 1:02d}"
        if slot < b_stairs:  return "stairs", f"Stairs Vault {slot - b_intro + 1:02d}"
        if slot < b_exit:    return "exit", f"Exit Vault {slot - b_stairs + 1:02d}"
        if slot < b_starter: return "starter", f"Starter Vault {slot - b_exit + 1:02d}"
        if slot < b_battle:  return "battle", f"Battle {slot - b_starter + 1:02d}"
        if slot < b_level:   return "level", f"Level {slot - b_battle + 1:02d}"
        if slot < b_full:    return "progress", f"World Push {slot - b_level + 1:03d}"
        return "full", f"Full Journey {slot - b_full + 1:02d}"

    def _choose_episode_start(self):
        self.saved_milestones = self._discover_saved_milestones()
        self.full_chain_ready = self._champion_full_starter_ready()
        role, _ = self._agent_role()
        self.training_objective = role
        saved = set(self.saved_milestones)

        if role == "intro": return "beginning"
        if role == "stairs": return "intro_complete" if "intro_complete" in saved else "beginning"
        if role == "exit":
            if "stairs_down" in saved: return "stairs_down"
            if "intro_complete" in saved: return "intro_complete"
            return "beginning"
        if role == "starter":
            if "left_house" in saved: return "left_house"
            if "stairs_down" in saved: return "stairs_down"
            if "intro_complete" in saved: return "intro_complete"
            return "beginning"
        if role in ("battle", "level"):
            # V12.5: NICHT im Labor (Rivalenkampf) starten, sondern DRAUSSEN im
            # Gras (Route 1) - dort sind wilde Kaempfe haeufig und die
            # Battle-Erkennung (0x02023BC8) ist bestaetigt. Rivalenkampf-Signal
            # war unzuverlaessig.
            stage_cps = self._valid_stage_checkpoints()
            if stage_cps:
                return stage_cps[max(stage_cps)]
            if "starter_outdoor" in saved: return "starter_outdoor"
            if "starter" in saved: return "starter"
            if "left_house" in saved: return "left_house"
            return self._best_progress_milestone()
        if role in ("progress", "badge"):
            if "starter" in saved: return self._best_progress_milestone()
            if "left_house" in saved: return "left_house"
            if "stairs_down" in saved: return "stairs_down"
            if "intro_complete" in saved: return "intro_complete"
            return "beginning"

        # Recovery / full: der Champion wird NUR an Full-from-Beginning-Runs
        # gemessen -> die grosse Mehrheit der (wenigen) full-Agenten startet
        # vorne. Erst wenn die Full-Kette steht, darf ein einzelner als
        # Bruecken-Start tiefer einsteigen (haelt spaete Stages warm).
        n = self.n_envs
        slot = self.rank % n
        b_full = n - max(4, int(round(n * 0.05)))
        full_idx = max(0, slot - b_full)   # 0 = erster full-Agent

        if not bool(getattr(self, "full_chain_ready", False)):
            return "beginning"

        if full_idx == 0:
            if "left_house" in saved: return "left_house"
            if "stairs_down" in saved: return "stairs_down"
            if "intro_complete" in saved: return "intro_complete"
        return "beginning"

    def _is_long_full_probe(self):
        # V13.2: relativ zur Flotte. Die HINTERE Haelfte des full-Bandes sind
        # lange, cap-freie Probes (laufen bis zur natuerlichen Truncation) -
        # die vordere Haelfte bleibt gecappt, damit ueberhaupt regelmaessig
        # Full-Episoden abschliessen (Recent-Eval braucht min_full_episodes).
        n = self.n_envs
        slot = self.rank % n
        b_full = n - max(4, int(round(n * 0.05)))
        lo = b_full + max(1, (n - b_full) // 2)
        return (
            self.training_objective == "full"
            and self.episode_start == "beginning"
            and slot >= lo
        )

    def _read_info_with_idle_frame(self):
        """
        Ein neutrales Frame liefert nach Reset/State-Restore die aktuellen
        data.json-Werte. Diese Werte werden als Baseline gesetzt, damit ein
        Curriculum-Start nicht alte Level/Badges erneut belohnt.
        """
        step_res = self.env.step(self.btn_none)

        info = (
            step_res[4]
            if len(step_res) == 5
            else step_res[3]
        )

        if not isinstance(info, dict):
            return {}

        return info

    def _set_baseline_from_info(self, info, loc_override=None):
        p_lvl = int(info.get("p1_level", 0))
        badges_raw = int(info.get("badges", 0))
        badges = (
            bin(badges_raw).count("1")
            if badges_raw > 0
            else 0
        )

        self.last_level = p_lvl
        self.last_badges = badges
        self.has_starter = (
            p_lvl >= 5 or self._v10171_party_has_starter()
        )
        if self.has_starter:
            # Der Party-Reader ist ebenso gueltig wie p1_level. Sonst zeigt
            # die Journey-Karte trotz sichtbarem Starter weiterhin 0%.
            self._claim_journey_milestone(
                "journey_starter", "journey_seen_starter"
            )

        loc = loc_override
        if loc is None:
            loc = read_player_location(self.env, allow_scan=False)
        self.cached_loc = loc
        bank = int(loc["map_bank"]) if loc["valid"] else 0
        map_id = int(loc["map_id"]) if loc["valid"] else 0
        x = int(loc["x_pos"]) if loc["valid"] else 0
        y = int(loc["y_pos"]) if loc["valid"] else 0

        if self._valid_coord(bank, map_id, x, y):
            coord_key = (bank, map_id, x, y)
            map_key = (bank, map_id)

            self.last_pos = coord_key
            self.seen_coords.add(coord_key)
            self.visited_maps.add(map_key)
            self.recent_path.append([bank, map_id, x, y])

            self.previous_valid_bank = bank
            self.previous_valid_map = map_id
            self.previous_valid_x = x
            self.previous_valid_y = y

            if (
                self.episode_start != "beginning"
                and bank == self.OVERWORLD_BANK
            ):
                self.left_house_rewarded = True
                self.left_house_confirmed = True
                self.outdoor_confirm_reads = self.OUTDOOR_CONFIRM_READS
                self.first_outdoor_map = map_id
                self.outdoor_entry_y = y
                # Start bereits draussen MIT Starter -> "raus aus dem Labor"
                # war schon erledigt, nicht erneut belohnen.
                if self.has_starter:
                    self.starter_outdoor_rewarded = True

        self.last_progress_signature = (
            bank,
            map_id,
            x,
            y,
            p_lvl,
            badges,
            int(info.get("in_battle", 0))
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Vorherige Episode fuer die Lernstatistik abschliessen.
        if self.total_steps > 0:
            self._finalize_run_stats()
            self.completed_episodes += 1
            self.total_episode_reward += float(self.current_reward)
            if (
                self.best_episode_reward is None
                or self.current_reward > self.best_episode_reward
            ):
                self.best_episode_reward = float(self.current_reward)

        # Persistente Exploration vor Episode-Reset sichern.
        self._save_exploration_memory()

        # Erst echter Spielstart.
        self.env.reset()

        self.total_steps = 0
        self.episode_battles_started = 0
        self.episode_battles_completed = 0
        self.enemy_party_cache = []
        self.player_party_cache = []
        self.last_party_total_hp = 0
        self.faints_in_current_battle = 0
        self.last_party_size = 0
        self.last_party_total_level = 0
        self.indoor_steps_without_transition = 0
        self.battle_activity_open = False
        self.enemy_hp_min = {}
        self.enemy_fainted_rewarded = set()
        self.episode_enemy_damage_hp = 0
        self.episode_enemy_damage_reward = 0.0
        self.episode_enemy_faints = 0
        self.seen_coords = set()
        self.visited_maps = set()
        self._saved_outdoor_depth = 0
        # V10.23: pro Episode erneut belohnen, wenn eine bekannte Map oder ein
        # bekannter Warp korrekt wiederholt wird. Persistente Weltkenntnis
        # bleibt davon getrennt und wird nicht geloescht.
        self.learning_seen_maps = set()
        self.learning_seen_edges = set()
        self.learning_seen_transitions = set()
        self.recent_path = []
        self.current_reward = 0.0
        self.last_exploration_coord = None
        self.last_exploration_map = None
        self.episode_edge_visits = {}
        self.steps_since_new_edge = 0
        self.last_progress_advance_step = 0
        self.last_progress_checkpoint_step = -999999
        self.progress_checkpoint_index = 0
        self.last_exit_seek_distance = None

        self.last_level = 0
        self.last_badges = 0
        self.has_starter = False
        self.starter_outdoor_rewarded = False
        self.starter_obtained_step = None
        self._v10171_has_starter_cached = False
        self._v10171_party_check_step = -999999

        self.initial_indoor_map = None
        self.stairs_down_rewarded = False
        self.left_house_rewarded = False
        self.left_house_confirmed = False
        self.outdoor_confirm_reads = 0
        self.last_stage_timeout = None
        self.outdoor_first_step_rewarded = False
        self.outdoor_entry_coord = None
        self.north_grass_rewarded = False
        self.next_outdoor_map_rewarded = False
        self.first_outdoor_map = None
        self.outdoor_entry_y = None
        self.best_north_y = None
        self.north_progress_tiles_rewarded = 0
        self.episode_milestone_steps = {}

        self.intro_seen_states = set()
        self.intro_last_thumb = None
        self.intro_same_screen_steps = 0
        self.intro_novelty_reward_total = 0.0
        self.intro_complete_rewarded = False

        self.previous_valid_bank = None
        self.previous_valid_map = None
        self.previous_valid_x = None
        self.previous_valid_y = None
        self.pending_exit_story_transition = None

        self.last_pos = None
        self.stuck_counter = 0
        self.last_progress_signature = None
        self.episode_anti_loop_resets = 0

        self.objective_success = False
        self.last_gameplay_ready = False
        self.last_in_battle = 0
        self.start_spam_count = 0
        self.last_start_step = -999999
        self._refresh_shared_snapshots()

        chosen_start = self._choose_episode_start()
        self.episode_start = chosen_start

        if chosen_start != "beginning":
            loaded = self._load_curriculum_state(chosen_start)
            if not loaded:
                self.episode_start = "beginning"

        baseline_info = self._read_info_with_idle_frame()

        verified_loc = None
        if self.episode_start != "beginning":
            try:
                verified_loc = read_player_location(
                    self.env,
                    allow_scan=True
                )
            except Exception:
                verified_loc = None

        self._set_baseline_from_info(
            baseline_info,
            loc_override=verified_loc
        )

        if self.episode_start != "beginning":
            self.intro_complete_rewarded = True

            # Gezielte Early-Game-Curriculum-States muessen ihre Story-Stufe
            # explizit wiederherstellen. Der RAM-Ort allein sagt nicht sicher,
            # ob die Treppe bereits als Lern-Meilenstein erreicht wurde.
            if self.episode_start == "intro_complete":
                self.episode_milestone_steps["intro_complete"] = 0

            elif self.episode_start == "stairs_down":
                self.stairs_down_rewarded = True
                self.episode_milestone_steps["intro_complete"] = 0
                self.episode_milestone_steps["stairs_down"] = 0

            elif (
                self.episode_start == "left_house"
                or self.episode_start == "starter"
                or self.episode_start == "starter_outdoor"
                or self.episode_start.startswith("progress_")
                or self.episode_start.startswith("maps_")
                or self.episode_start.startswith("outdoor_")
                or self.episode_start.startswith("level_")
                or self.episode_start.startswith("badge_")
            ):
                # Spaetere Curriculum-States liegen nach dem Haus.
                self.stairs_down_rewarded = True
                self.left_house_rewarded = True
                self.left_house_confirmed = True
                self.outdoor_confirm_reads = self.OUTDOOR_CONFIRM_READS
                self.episode_milestone_steps["intro_complete"] = 0
                self.episode_milestone_steps["stairs_down"] = 0
                # starter_outdoor: schon draussen mit Starter -> nicht erneut
                # fuer "raus aus dem Labor" belohnen.
                if self.episode_start == "starter_outdoor":
                    self.starter_outdoor_rewarded = True
                    self.episode_milestone_steps["starter_outdoor"] = 0

        return (
            self._make_obs(
                self.env.get_screen(),
                loc=self.cached_loc,
                info=baseline_info,
            ),
            {
                "episode_start": self.episode_start,
                "curriculum_states": len(self.saved_milestones)
            }
        )


    def _exit_route_dir(self):
        path = os.path.join(SHARED_CURRICULUM_DIR, "exit_routes")
        os.makedirs(path, exist_ok=True)
        return path

    def _load_confirmed_exit_route_edges(self):
        votes = {}
        try:
            names = os.listdir(self._exit_route_dir())
        except Exception:
            names = []

        for name in names:
            if not name.startswith("agent_") or not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._exit_route_dir(), name), "r") as f:
                    data = json.load(f)
                seen = set()
                for raw in data.get("edges", []):
                    if isinstance(raw, list) and len(raw) == 8:
                        edge = tuple(int(v) for v in raw)
                        if edge not in seen:
                            votes[edge] = votes.get(edge, 0) + 1
                            seen.add(edge)
            except Exception:
                pass

        return {
            edge for edge, count in votes.items()
            if count >= self.EXIT_ROUTE_CONFIRM_AGENTS
        }

    def _commit_successful_exit_route(self):
        edges = list(dict.fromkeys(
            getattr(self, "episode_exit_route_edges", [])
        ))
        if not edges:
            return

        edges = edges[-self.EXIT_ROUTE_MAX_EDGES:]
        path = os.path.join(
            self._exit_route_dir(),
            f"agent_{self.rank:02d}.json"
        )
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "agent": int(self.rank),
                        "edges": [list(edge) for edge in edges],
                    },
                    f
                )
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    # ---- V10.33 Wege-Gedaechtnis (Weg NACH dem Haus) ----
    def _journey_route_dir(self):
        path = os.path.join(SHARED_CURRICULUM_DIR, "journey_routes")
        os.makedirs(path, exist_ok=True)
        return path

    def _load_confirmed_journey_edges(self):
        votes = {}
        try:
            names = os.listdir(self._journey_route_dir())
        except Exception:
            names = []
        for name in names:
            if not name.startswith("agent_") or not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._journey_route_dir(), name), "r") as f:
                    data = json.load(f)
                seen = set()
                for raw in data.get("edges", []):
                    if isinstance(raw, list) and len(raw) == 8:
                        edge = tuple(int(v) for v in raw)
                        if edge not in seen:
                            votes[edge] = votes.get(edge, 0) + 1
                            seen.add(edge)
            except Exception:
                pass
        return {
            edge for edge, count in votes.items()
            if count >= self.JOURNEY_ROUTE_CONFIRM_AGENTS
        }

    def _commit_journey_route(self):
        edges = list(dict.fromkeys(
            getattr(self, "episode_journey_edges", [])
        ))
        if not edges:
            return
        edges = edges[-self.JOURNEY_ROUTE_MAX_EDGES:]
        path = os.path.join(
            self._journey_route_dir(),
            f"agent_{self.rank:02d}.json"
        )
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "agent": int(self.rank),
                        "edges": [list(edge) for edge in edges],
                    },
                    f
                )
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def step(self, action):
        requested_action = int(action)
        effective_action = requested_action

        # V8.7.1: Action 2 = START. SELECT is not in the current
        # seven-action space. During an already detected battle START
        # becomes NO-OP and receives a small penalty below.
        battle_was_active = int(getattr(self, "last_in_battle", 0)) == 1
        blocked_battle_start = bool(
            battle_was_active and requested_action == 2
        )
        raw_act = (
            self.btn_none
            if blocked_battle_start
            else self.action_map[effective_action]
        )

        for _ in range(4):
            step_res = self.env.step(raw_act)

        for _ in range(4):
            step_res = self.env.step(self.btn_none)

        self.total_steps += 1
        if self.total_steps == 1 or not hasattr(self, "v9_last_pos"):
            self.v9_last_pos = None
            self.v9_same_pos_steps = 0
            self.v9_episode_tiles = set()
            self._north_corridor_best = {}  # (bank,map) -> noerdlichste Y

        if (
            self.total_steps == 1
            or not hasattr(self, "episode_exit_route_edges")
        ):
            self.episode_exit_route_edges = []
            self.exit_route_edge_visits = {}
            self.confirmed_exit_route_edges = (
                self._load_confirmed_exit_route_edges()
            )

        if (
            self.total_steps == 1
            or not hasattr(self, "episode_journey_edges")
        ):
            self.episode_journey_edges = []
            self.journey_edge_visits = {}
            self.confirmed_journey_edges = (
                self._load_confirmed_journey_edges()
            )


        info = (
            step_res[4]
            if len(step_res) == 5
            else step_res[3]
        )

        if not isinstance(info, dict):
            info = {}

        raw_screen = step_res[0]

        # Full RAM copy only periodically. Level/battle/data.json rewards below
        # are still evaluated on every agent step.
        if (
            self.total_steps == 1
            or self.total_steps % self.LOCATION_READ_EVERY == 0
        ):
            allow_discovery_scan = False

            if not self.cached_loc.get("valid", False):
                # V6 FIX: guaranteed discovery slot for every rank.
                discovery_slots = max(
                    1,
                    self.LOCATION_DISCOVERY_EVERY
                    // self.LOCATION_READ_EVERY
                )
                discovery_slot = (
                    (self.rank * 17) % discovery_slots
                )
                read_slot = (
                    self.total_steps // self.LOCATION_READ_EVERY
                ) % discovery_slots
                allow_discovery_scan = (
                    read_slot == discovery_slot
                )

            self.cached_loc = read_player_location(
                self.env,
                allow_scan=allow_discovery_scan
            )

        loc = self.cached_loc
        bank = int(loc["map_bank"]) if loc["valid"] else 0
        map_id = int(loc["map_id"]) if loc["valid"] else 0
        x = int(loc["x_pos"]) if loc["valid"] else 0
        y = int(loc["y_pos"]) if loc["valid"] else 0
        # V12: Battle-Erkennung robust ueber MEHRERE Signale.
        # 1) RAM-Flag-Kandidaten (deutsche BPRD01, siehe data.json)
        # 2) FALLBACK: gueltige Gegner-Party im RAM = wir sind im Kampf.
        #    read_enemy_party liest an der (bestaetigten) US/DE-Adresse 0x2402C
        #    -> zuverlaessig, unabhaengig von den Flag-Bytes.
        _bt_any = (
            int(info.get("battle_flags", 0) or 0)
            or int(info.get("bt_c1", 0) or 0)
            or int(info.get("bt_c2", 0) or 0)
            or int(info.get("bt_c3", 0) or 0)
            or int(info.get("in_battle", 0) or 0)
        )
        # Frische Gegner-Sichtung (Zaehler laeuft nach Kampfende aus).
        _enemy_present = (
            self.total_steps - int(getattr(self, "_last_enemy_seen_step", -999))
            <= 3
        )
        in_battle = 1 if (_bt_any or _enemy_present) else 0
        previous_battle_state = int(self.last_in_battle)
        battle_ended_without_faint = False
        if previous_battle_state == 0 and in_battle == 1:
            self.faints_in_current_battle = 0
            self.run_stats["battles_started"] += 1
            self.episode_battles_started += 1
            self.battle_activity_open = True
            self.enemy_party_cache = []
            self.enemy_hp_min = {}
            self.enemy_fainted_rewarded = set()
            self._save_run_stats()
        elif previous_battle_state == 1 and in_battle == 0:
            alive_pokemon = sum(
                1 for mon in self.player_party_cache
                if int(mon.get("cur_hp", 0)) > 0
            )
            battle_ended_without_faint = bool(
                alive_pokemon > 0
                and self.faints_in_current_battle == 0
            )
            self.run_stats["battles_completed"] += 1
            self.episode_battles_completed += 1
            self.battle_activity_open = False
            self.enemy_party_cache = []
            self.enemy_hp_min = {}
            self.enemy_fainted_rewarded = set()
            self._save_run_stats()

        p_lvl = int(info.get("p1_level", 0))
        badges_raw = int(info.get("badges", 0))
        badges = (
            bin(badges_raw).count("1")
            if badges_raw > 0
            else 0
        )

        gameplay_ready = bool(
            loc.get("valid", False)
            and loc.get("trusted", False)
            and self._valid_coord(bank, map_id, x, y)
        )

        if (
            p_lvl >= 5
            and (
                not self.player_party_cache
                or self.total_steps % self.PARTY_READ_EVERY == 0
            )
        ):
            try:
                decoded_party = read_player_party(self.env)
                if decoded_party:
                    self.player_party_cache = decoded_party
            except Exception:
                pass

        if (
            self.total_steps == 1
            or self.total_steps % self.SHARED_SNAPSHOT_EVERY == 0
        ):
            self._refresh_shared_snapshots()

        self.last_gameplay_ready = gameplay_ready
        self.last_in_battle = in_battle

        reward = (
            self.GAMEPLAY_STEP_COST
            if gameplay_ready
            else self.INTRO_STEP_COST
        )
        milestone_saved = None
        reward_events = []
        truncated = False
        objective_done = False

        if battle_ended_without_faint:
            reward -= 3.0
            reward_events.append("fled_battle:-3.0")

        # V10.27A: Die gesamte Party statt nur Pokemon-Slot 1 bewerten.
        if p_lvl >= 5 and self.player_party_cache:
            valid_party = [
                mon for mon in self.player_party_cache
                if int(mon.get("species_id", 0)) > 0
            ]
            current_party_size = len(valid_party)
            current_total_level = sum(
                int(mon.get("level", 0)) for mon in valid_party
            )
            current_total_hp = sum(
                int(mon.get("cur_hp", 0)) for mon in valid_party
            )
            max_total_hp = sum(
                int(mon.get("max_hp", 0)) for mon in valid_party
            )

            if (
                self.last_party_size > 0
                and current_party_size > self.last_party_size
                and current_party_size <= 4
            ):
                reward += 100.0
                reward_events.append("caught_pokemon:+100.0")

            if (
                self.last_party_total_level > 0
                and current_total_level > self.last_party_total_level
            ):
                level_gain = current_total_level - self.last_party_total_level
                reward += level_gain * 25.0
                reward_events.append(
                    f"team_level_up:+{level_gain * 25.0:.1f}"
                )
                self.reward_event_counts["level_up"] += level_gain
                self.last_progress_advance_step = self.total_steps
                if self.training_objective == "level":
                    reward += self.SPECIALIST_SUCCESS_BONUS
                    reward_events.append(
                        f"objective_level:+{self.SPECIALIST_SUCCESS_BONUS:.0f}"
                    )
                    self.objective_success = True
                    objective_done = True
                bridge = self._maybe_save_progress_bridge("team_level_up")
                if bridge:
                    milestone_saved = bridge

            if current_total_hp == 0 and max_total_hp > 0 and not truncated:
                reward -= 2.0
                reward_events.append("party_wiped:-2.0")
                truncated = True
                self.last_stage_timeout = "party_wiped"
                info["last_stage_timeout"] = "party_wiped"
            elif self.last_party_total_hp > 0:
                hp_diff = current_total_hp - self.last_party_total_hp
                if hp_diff < 0:
                    hp_reward = abs(hp_diff) * -0.5
                    reward += hp_reward
                    reward_events.append(f"took_damage:{hp_reward:.1f}")
                elif hp_diff > 0:
                    hp_reward = hp_diff * 0.2
                    reward += hp_reward
                    reward_events.append(f"healed:+{hp_reward:.1f}")

            # Auch nach der ersten Baseline und bei unveraenderter Party
            # aktualisieren; sonst blieben die vorgeschlagenen Werte bei 0.
            self.last_party_size = current_party_size
            self.last_party_total_level = current_total_level
            self.last_party_total_hp = current_total_hp

        if blocked_battle_start:
            reward += self.BATTLE_BLOCKED_START_PENALTY
            reward_events.append("battle_start_blocked:-0.10")

        # V7.5.1: reward only NEW opponent HP damage.
        if p_lvl >= 5 and self.total_steps % self.ENEMY_HP_READ_EVERY == 0:
            try:
                enemy_party = read_enemy_party(self.env)
            except Exception:
                enemy_party = []

            if enemy_party:
                self.enemy_party_cache = enemy_party
                if any(
                    int(m.get("species_id", 0)) > 0
                    and int(m.get("max_hp", 0)) > 0
                    for m in enemy_party
                ):
                    self._last_enemy_seen_step = self.total_steps
                for mon in enemy_party:
                    slot = int(mon.get("slot", -1))
                    species = int(mon.get("species_id", 0))
                    personality = int(mon.get("personality", 0))
                    cur_hp = int(mon.get("cur_hp", 0))
                    max_hp = int(mon.get("max_hp", 0))
                    if slot < 0 or species <= 0 or max_hp <= 0 or not (0 <= cur_hp <= max_hp):
                        continue

                    mon_key = (slot, species, personality)
                    if mon_key not in self.enemy_hp_min:
                        self.enemy_hp_min[mon_key] = cur_hp
                        continue

                    previous_min = int(self.enemy_hp_min[mon_key])
                    if cur_hp < previous_min:
                        hp_damage = previous_min - cur_hp
                        if not self.battle_activity_open:
                            self.battle_activity_open = True
                            self.run_stats["battles_started"] += 1
                            self.episode_battles_started += 1
                            reward_events.append("battle_fallback_start")
                        damage_reward = hp_damage * self.ENEMY_DAMAGE_REWARD_PER_HP
                        reward += damage_reward
                        self.enemy_hp_min[mon_key] = cur_hp

                        self.episode_enemy_damage_hp += hp_damage
                        self.episode_enemy_damage_reward += damage_reward
                        self.run_stats["enemy_damage_hp"] = int(
                            self.run_stats.get("enemy_damage_hp", 0)
                        ) + hp_damage
                        self.run_stats["enemy_damage_reward"] = round(
                            float(self.run_stats.get("enemy_damage_reward", 0.0))
                            + damage_reward, 3
                        )
                        reward_events.append(
                            f"enemy_damage:{hp_damage}hp:+{damage_reward:.2f}"
                        )

                        if cur_hp == 0 and previous_min > 0 and mon_key not in self.enemy_fainted_rewarded:
                            reward += self.ENEMY_FAINT_REWARD
                            self.enemy_fainted_rewarded.add(mon_key)
                            self.episode_enemy_faints += 1
                            self.faints_in_current_battle += 1
                            if self.battle_activity_open:
                                self.run_stats["battles_completed"] += 1
                                self.episode_battles_completed += 1
                                self.battle_activity_open = False
                            if self.training_objective == "battle":
                                reward += self.SPECIALIST_SUCCESS_BONUS
                                reward_events.append(
                                    f"objective_battle:+{self.SPECIALIST_SUCCESS_BONUS:.0f}"
                                )
                                self.objective_success = True
                                objective_done = True
                            self.run_stats["enemy_faints"] = int(
                                self.run_stats.get("enemy_faints", 0)
                            ) + 1
                            reward_events.append(
                                f"enemy_faint:+{self.ENEMY_FAINT_REWARD:.2f}"
                            )
                        self._save_run_stats()

        # ---------------------------------------------------------
        # START ANTI-SPAM
        # ---------------------------------------------------------
        # action 3 = START. Im Intro darf START frei benutzt werden.
        # Nach Intro im Haus ist START unnoetig und wird bestraft.
        # Spaeter ausserhalb des Hauses bleibt der erste START frei;
        # nur schnelles wiederholtes Spammen wird bestraft.
        if requested_action == 2:
            if (
                self.total_steps - self.last_start_step
                <= self.START_SPAM_RESET_STEPS
            ):
                self.start_spam_count += 1
            else:
                self.start_spam_count = 1

            self.last_start_step = self.total_steps

            if gameplay_ready and in_battle == 0:
                if not self.left_house_confirmed:
                    reward += self.START_HOUSE_PENALTY
                    reward_events.append(
                        f"start_house:{self.START_HOUSE_PENALTY:.2f}"
                    )

                if self.start_spam_count == 2:
                    reward += self.START_REPEAT_PENALTY_2
                    reward_events.append(
                        "start_repeat2:"
                        f"{self.START_REPEAT_PENALTY_2:.2f}"
                    )
                elif self.start_spam_count >= 3:
                    reward += self.START_REPEAT_PENALTY_3PLUS
                    reward_events.append(
                        "start_repeat3+:"
                        f"{self.START_REPEAT_PENALTY_3PLUS:.2f}"
                    )
        elif (
            self.total_steps - self.last_start_step
            > self.START_SPAM_RESET_STEPS
        ):
            self.start_spam_count = 0

        # ---------------------------------------------------------
        # INTRO / NAMENSVERGABE SHAPING
        # ---------------------------------------------------------
        if not gameplay_ready and self.episode_start == "beginning":
            thumb = self._intro_thumb(raw_screen)

            if self.intro_last_thumb is None:
                self.intro_last_thumb = thumb
                quant = (thumb // 32).astype(np.uint8)
                self.intro_seen_states.add(quant.tobytes())
            else:
                diff = float(
                    np.mean(np.abs(thumb - self.intro_last_thumb))
                )

                if diff < 4.0:
                    self.intro_same_screen_steps += 1
                else:
                    self.intro_same_screen_steps = 0

                quant = (thumb // 32).astype(np.uint8)
                state_key = quant.tobytes()

                # Nur deutliche, neue Screens belohnen. Cursor-Flackern allein
                # reicht normalerweise nicht. Gesamtbonus ist auf +25 gedeckelt.
                if (
                    diff >= 10.0
                    and state_key not in self.intro_seen_states
                    and self.intro_novelty_reward_total < 25.0
                ):
                    bonus = 1.0 if diff >= 28.0 else 0.5
                    bonus = min(
                        bonus,
                        25.0 - self.intro_novelty_reward_total
                    )
                    reward += bonus
                    self.intro_novelty_reward_total += bonus
                    self.intro_seen_states.add(state_key)
                    self.reward_event_counts["intro_state"] += 1
                    reward_events.append(
                        f"intro_state:+{bonus:.1f}"
                    )

                self.intro_last_thumb = thumb

                # Sehr lang derselbe Screen ist auch im Intro unerwuenscht.
                if self.intro_same_screen_steps >= 120:
                    reward -= 0.01
                if self.intro_same_screen_steps >= 300:
                    reward -= 0.03
                if self.intro_same_screen_steps >= 900:
                    truncated = True
                    info["intro_loop_reset"] = True
                    if self.intro_same_screen_steps == 900:
                        self.episode_anti_loop_resets += 1
                else:
                    info["intro_loop_reset"] = False

        elif (
            gameplay_ready
            and self.episode_start == "beginning"
            and not self.intro_complete_rewarded
        ):
            # Der grosse Zielreward: Intro/Namenswahl wirklich abgeschlossen.
            self.intro_complete_rewarded = True
            reward += 100.0
            reward_events.append("intro_complete:+100")

            if self.training_objective == "intro":
                reward += 50.0
                reward_events.append("objective_intro:+50")
                self.objective_success = True
                objective_done = True
            self.reward_event_counts["intro_complete"] += 1
            self.episode_milestone_steps.setdefault(
                "intro_complete", self.total_steps
            )
            if in_battle == 0:
                if self._save_curriculum_state("intro_complete"):
                    milestone_saved = "intro_complete"
            info["intro_loop_reset"] = False
        else:
            info["intro_loop_reset"] = False

        # V8.7: route learning from previously successful exits.
        if (
            gameplay_ready
            and in_battle == 0
            and not self.left_house_rewarded
            and self.previous_valid_bank is not None
            and self.previous_valid_map is not None
            and getattr(self, "previous_valid_x", None) is not None
            and getattr(self, "previous_valid_y", None) is not None
        ):
            edge = (
                int(self.previous_valid_bank),
                int(self.previous_valid_map),
                int(self.previous_valid_x),
                int(self.previous_valid_y),
                int(bank),
                int(map_id),
                int(x),
                int(y),
            )

            if edge[:4] != edge[4:]:
                self.episode_exit_route_edges.append(edge)
                if len(self.episode_exit_route_edges) > self.EXIT_ROUTE_MAX_EDGES:
                    self.episode_exit_route_edges = (
                        self.episode_exit_route_edges[
                            -self.EXIT_ROUTE_MAX_EDGES:
                        ]
                    )

                visits = self.exit_route_edge_visits.get(edge, 0) + 1
                self.exit_route_edge_visits[edge] = visits
                reverse = edge[4:] + edge[:4]

                if edge in self.confirmed_exit_route_edges:
                    if visits == 1:
                        reward += self.EXIT_ROUTE_EDGE_REWARD
                        reward_events.append(
                            f"exit_route_edge:+{self.EXIT_ROUTE_EDGE_REWARD:.2f}"
                        )
                    elif visits == 2:
                        reward += self.EXIT_ROUTE_REPEAT2_PENALTY
                        reward_events.append(
                            f"exit_route_repeat2:{self.EXIT_ROUTE_REPEAT2_PENALTY:.2f}"
                        )
                    elif visits == 3:
                        reward += self.EXIT_ROUTE_REPEAT3_PENALTY
                        reward_events.append(
                            f"exit_route_repeat3:{self.EXIT_ROUTE_REPEAT3_PENALTY:.2f}"
                        )
                elif reverse in self.confirmed_exit_route_edges:
                    reward += self.EXIT_ROUTE_REVERSE_PENALTY
                    reward_events.append(
                        f"exit_route_reverse:{self.EXIT_ROUTE_REVERSE_PENALTY:.2f}"
                    )

        # V10.33 WEGE-GEDAECHTNIS: gleicher Mechanismus fuer den Weg NACH dem
        # Haus (Labor-Ausgang, Alabastia -> Route 1 -> Vertania). Brotkrumen-
        # Spur aus den Kanten, die frueher erfolgreiche Agenten gegangen sind.
        if (
            gameplay_ready
            and in_battle == 0
            and self.left_house_rewarded
            and self.previous_valid_bank is not None
            and self.previous_valid_map is not None
            and getattr(self, "previous_valid_x", None) is not None
            and getattr(self, "previous_valid_y", None) is not None
        ):
            jedge = (
                int(self.previous_valid_bank),
                int(self.previous_valid_map),
                int(self.previous_valid_x),
                int(self.previous_valid_y),
                int(bank),
                int(map_id),
                int(x),
                int(y),
            )
            if jedge[:4] != jedge[4:]:
                self.episode_journey_edges.append(jedge)
                if len(self.episode_journey_edges) > self.JOURNEY_ROUTE_MAX_EDGES:
                    self.episode_journey_edges = (
                        self.episode_journey_edges[-self.JOURNEY_ROUTE_MAX_EDGES:]
                    )
                jvisits = self.journey_edge_visits.get(jedge, 0) + 1
                self.journey_edge_visits[jedge] = jvisits
                jreverse = jedge[4:] + jedge[:4]
                if jedge in self.confirmed_journey_edges:
                    if jvisits == 1:
                        reward += self.JOURNEY_ROUTE_EDGE_REWARD
                        reward_events.append(
                            f"journey_route_edge:+{self.JOURNEY_ROUTE_EDGE_REWARD:.2f}"
                        )
                    elif jvisits == 2:
                        reward += self.JOURNEY_ROUTE_REPEAT2_PENALTY
                        reward_events.append(
                            f"journey_route_repeat2:{self.JOURNEY_ROUTE_REPEAT2_PENALTY:.2f}"
                        )
                    elif jvisits >= 3:
                        reward += self.JOURNEY_ROUTE_REPEAT3_PENALTY
                        reward_events.append(
                            f"journey_route_repeat3:{self.JOURNEY_ROUTE_REPEAT3_PENALTY:.2f}"
                        )
                elif jreverse in self.confirmed_journey_edges:
                    reward += self.JOURNEY_ROUTE_REVERSE_PENALTY
                    reward_events.append(
                        f"journey_route_reverse:{self.JOURNEY_ROUTE_REVERSE_PENALTY:.2f}"
                    )

        # V9 anti-camping / exploration shaping.
        if gameplay_ready and in_battle == 0:
            pos_key = (int(bank), int(map_id), int(x), int(y))
            if self.v9_last_pos == pos_key:
                self.v9_same_pos_steps += 1
            else:
                self.v9_same_pos_steps = 0
                self.v9_last_pos = pos_key

            if self.v9_same_pos_steps == self.V9_STUCK_SAME_POS_STEPS:
                reward += self.V9_STUCK_PENALTY
                reward_events.append("v9_stuck_same_pos:-2")

            # V10.27C: Aufenthalt auf derselben Indoor-Map wird nach einer
            # grosszuegigen Navigationsphase zunehmend negativ. Ein echter
            # Raumwechsel setzt den Zaehler sofort zurueck.
            same_map = (
                self.previous_valid_bank is not None
                and self.previous_valid_map is not None
                and (int(self.previous_valid_bank), int(self.previous_valid_map))
                    == (int(bank), int(map_id))
            )
            if bank != self.OVERWORLD_BANK:
                self.indoor_steps_without_transition = (
                    self.indoor_steps_without_transition + 1
                    if same_map else 0
                )
                if self.indoor_steps_without_transition >= self.INDOOR_STALL_HARD_STEPS:
                    reward -= 0.10
                    reward_events.append("indoor_stall_hard:-0.10")
                elif self.indoor_steps_without_transition >= self.INDOOR_STALL_SOFT_STEPS:
                    reward -= 0.02
                    reward_events.append("indoor_stall_soft:-0.02")
            else:
                self.indoor_steps_without_transition = 0

            if self.training_objective == "progress":
                if pos_key not in self.v9_episode_tiles:
                    self.v9_episode_tiles.add(pos_key)
                    # V13.3: neue Kachel DRAUSSEN voll (+2.0), DRINNEN nur ein
                    # Bruchteil (+0.35). Sonst farmt der Progress-Agent Eichs
                    # Labor / Haeuser mit +2.0/Kachel ab und geht nie auf die
                    # Route. Ein Pflicht-Gebaeude (Mart/Arena) bleibt trotzdem
                    # leicht positiv, wird aber nicht attraktiver als Route 1.
                    if bank == self.OVERWORLD_BANK:
                        _tile_bonus = self.V9_EXPLORER_NEW_TILE_BONUS
                        reward_events.append("v9_explorer_new_tile:+2.0")
                    else:
                        _tile_bonus = min(0.35, self.V9_EXPLORER_NEW_TILE_BONUS)
                        reward_events.append("v9_explorer_new_tile_indoor:+0.35")
                    reward += _tile_bonus
                else:
                    reward += self.V9_EXPLORER_REPEAT_TILE_PENALTY

            # V13.3: Progress/Battle/Level mit Starter, die drinnen herumhaengen
            # (kein Pflicht-Gebaeude direkt am Anfang), kriegen einen kleinen
            # Sog nach draussen. Die Route ist damit immer die profitablere Wahl.
            if (
                self.training_objective in ("progress", "battle", "level")
                and self.has_starter
                and bank != self.OVERWORLD_BANK
                and self.indoor_steps_without_transition > 150
            ):
                reward -= 0.05
                reward_events.append("indoor_drift:-0.05")

            # V13.4: NORD-KORRIDOR. Auf den geraden Vor-Wald-Strecken
            # (Alabastia, Route 1) dichter Reward pro neuer noerdlichster
            # Reihe -> die 60 blinden Schritte werden zur Belohnungsrampe.
            # NUR fuer die Welt-Roller, NIE im Wald. Nicht farmbar: es zahlt
            # ausschliesslich fuer eine ECHTE neue noerdlichste Y-Reihe.
            if (
                self.training_objective in ("progress", "battle", "level", "full")
                and (int(bank), int(map_id)) in self.NORTH_CORRIDOR_MAPS
            ):
                _mk = (int(bank), int(map_id))
                _prev_best = self._north_corridor_best.get(_mk, int(y) + 1)
                if int(y) < _prev_best:
                    _rows = min(_prev_best - int(y), self.NORTH_CORRIDOR_MAX_ROWS)
                    _r = _rows * self.NORTH_CORRIDOR_ROW_REWARD
                    reward += _r
                    reward_events.append(f"north_corridor:{map_id}:+{_r:.1f}")
                    self._north_corridor_best[_mk] = int(y)

        # ---------------------------------------------------------
        # STORY SHAPING
        # ---------------------------------------------------------
        if gameplay_ready and in_battle == 0:
            # 0) Start-Haus gezielt formen statt Treppen-Loops zu belohnen.
            if bank != self.OVERWORLD_BANK and self.initial_indoor_map is None:
                self.initial_indoor_map = (bank, map_id)

            # Erster Indoor-Mapwechsel vom Startzimmer weg = Treppe/F1.
            if (
                not self.stairs_down_rewarded
                and self.initial_indoor_map is not None
                and bank != self.OVERWORLD_BANK
                and (bank, map_id) != self.initial_indoor_map
            ):
                self.stairs_down_rewarded = True

                if (
                    getattr(self, "previous_valid_bank", None) is not None
                    and getattr(self, "previous_valid_map", None) is not None
                    and getattr(self, "previous_valid_x", None) is not None
                    and getattr(self, "previous_valid_y", None) is not None
                ):
                    stairs_transition = (
                        int(self.previous_valid_bank),
                        int(self.previous_valid_map),
                        int(self.previous_valid_x),
                        int(self.previous_valid_y),
                        int(bank), int(map_id), int(x), int(y),
                    )
                    self._save_confirmed_story_warp(
                        "stairs", stairs_transition
                    )
                    reward += self.CONFIRMED_WARP_REWARD
                    reward_events.append("confirmed_stairs_warp:+6")

                reward += 150.0
                reward_events.append("stairs_down:+150")

                if self.training_objective == "stairs":
                    reward += 50.0
                    reward_events.append("objective_stairs:+50")
                    self.objective_success = True
                    objective_done = True
                self.reward_event_counts["stairs_down"] += 1
                self.episode_milestone_steps.setdefault(
                    "stairs_down", self.total_steps
                )
                if in_battle == 0:
                    if self._save_curriculum_state("stairs_down"):
                        milestone_saved = "stairs_down"

            # Zurueck zum Startzimmer bevor das Haus verlassen wurde:
            # klar negativ, damit F2 <-> F1 nicht zur Reward-Schleife wird.
            if (
                self.stairs_down_rewarded
                and not self.left_house_rewarded
                and self.initial_indoor_map is not None
                and self.previous_valid_bank is not None
                and (self.previous_valid_bank, self.previous_valid_map)
                    != self.initial_indoor_map
                and (bank, map_id) == self.initial_indoor_map
            ):
                reward -= 30.0
                reward_events.append("stairs_back:-30")
                self.reward_event_counts["stairs_back"] += 1

            # Haus verlassen: nur nach erkannter Treppe UND mehreren
            # vertrauenswuerdigen Outdoor-Reads bestaetigen. Das verhindert,
            # dass ein einzelner falscher/staler RAM-Read die Early-Game-
            # Timeouts deaktiviert.
            if (
                self.stairs_down_rewarded
                and bank == self.OVERWORLD_BANK
                and self.previous_valid_bank is not None
                and self.previous_valid_bank != self.OVERWORLD_BANK
            ):
                if (
                    getattr(self, "previous_valid_x", None) is not None
                    and getattr(self, "previous_valid_y", None) is not None
                ):
                    self.pending_exit_story_transition = (
                        int(self.previous_valid_bank),
                        int(self.previous_valid_map),
                        int(self.previous_valid_x),
                        int(self.previous_valid_y),
                        int(bank), int(map_id), int(x), int(y),
                    )
                self.outdoor_confirm_reads = 1

            elif (
                self.stairs_down_rewarded
                and bank == self.OVERWORLD_BANK
                and self.outdoor_confirm_reads > 0
                and not self.left_house_confirmed
            ):
                self.outdoor_confirm_reads += 1

            elif (
                bank != self.OVERWORLD_BANK
                and not self.left_house_confirmed
            ):
                self.outdoor_confirm_reads = 0

            if (
                not self.left_house_confirmed
                and self.outdoor_confirm_reads >= self.OUTDOOR_CONFIRM_READS
            ):
                self.left_house_confirmed = True
                self.left_house_rewarded = True
                self.first_outdoor_map = map_id
                self.outdoor_entry_y = y
                self.outdoor_entry_coord = (x, y)
                self.best_north_y = y
                if self.pending_exit_story_transition is not None:
                    self._save_confirmed_story_warp(
                        "exit", self.pending_exit_story_transition
                    )
                    reward += self.CONFIRMED_WARP_REWARD
                    reward_events.append("confirmed_exit_warp:+6")

                reward += 500.0
                reward_events.append("left_house_confirmed:+500")
                self._commit_successful_exit_route()
                self.confirmed_exit_route_edges = (
                    self._load_confirmed_exit_route_edges()
                )

                if self.training_objective == "exit":
                    reward += 300.0
                    reward_events.append("objective_exit:+300")
                    self.objective_success = True
                    objective_done = True
                self.reward_event_counts["left_house"] += 1
                self.episode_milestone_steps.setdefault(
                    "left_house", self.total_steps
                )
                if in_battle == 0:
                    if self._save_curriculum_state("left_house"):
                        milestone_saved = "left_house"

            # Erster echter Schritt vom Hauseingang weg.
            if (
                self.left_house_rewarded
                and not self.outdoor_first_step_rewarded
                and bank == self.OVERWORLD_BANK
                and map_id == self.first_outdoor_map
                and self.outdoor_entry_coord is not None
                and abs(x - self.outdoor_entry_coord[0])
                    + abs(y - self.outdoor_entry_coord[1]) >= 1
            ):
                self.outdoor_first_step_rewarded = True
                reward += 30.0
                reward_events.append("outdoor_first_step:+30")
                self.reward_event_counts["outdoor_first_step"] += 1
                self.last_progress_advance_step = self.total_steps
                bridge = self._maybe_save_progress_bridge(
                    "outdoor_first_step"
                )
                if bridge:
                    milestone_saved = bridge

            # Falls der erste valide RAM-Read erst draussen gelingt, setzen wir
            # nur den Anker, ohne einen unverdienten Reward zu erzeugen.
            if (
                bank == self.OVERWORLD_BANK
                and self.first_outdoor_map is None
            ):
                self.first_outdoor_map = map_id
                self.outdoor_entry_y = y
                self.best_north_y = y

            # 3) Erste neue Aussenwelt-Map nach der Start-Aussenmap.
            if (
                self.first_outdoor_map is not None
                and not self.next_outdoor_map_rewarded
                and bank == self.OVERWORLD_BANK
                and map_id != self.first_outdoor_map
            ):
                self.next_outdoor_map_rewarded = True
                reward += 150.0
                reward_events.append("next_outdoor_map:+150")
                # V10.33: Alabastia -> Route 1 geschafft -> Weg als Spur sichern.
                self._commit_journey_route()
                self.reward_event_counts["next_outdoor_map"] += 1
                self.episode_milestone_steps.setdefault(
                    "next_outdoor_map", self.total_steps
                )
                self.last_progress_advance_step = self.total_steps
                bridge = self._maybe_save_progress_bridge(
                    "next_outdoor_map"
                )
                if bridge:
                    milestone_saved = bridge

        # V11: NORD-SCHUB ENTFERNT. War ein hartkodierter Richtungs-Prior
        # ("Norden = gut"). Das verfaelscht die Lernkurve UND ist im Vertania-
        # Wald direkt falsch (dort: rechts, hoch, links, runter, links, hoch -
        # ein Labyrinth, kein Nordweg). Der reine Neue-Kachel-Bonus (+2.0)
        # zieht die Agenten sauber und richtungs-neutral ins Unerkundete -
        # von Alabastia zufaellig nach Norden, im Wald den Labyrinthpfad lang.

        # 4) Erstes Pokemon/Starter: starkes Storysignal.
        # Den normalen Levelreward fuer Level 1->5 unterdruecken wir dabei,
        # sonst waeren es unbeabsichtigt +225.
        party_has_starter = self._v10171_party_has_starter()
        if not self.has_starter and (p_lvl >= 5 or party_has_starter):
            self.has_starter = True
            reward += self.STARTER_REWARD
            reward_events.append(
                f"first_pokemon:+{self.STARTER_REWARD:.0f}"
            )
            self.reward_event_counts["first_pokemon"] += 1
            self.episode_milestone_steps.setdefault(
                "first_pokemon", self.total_steps
            )
            self.last_level = max(p_lvl, 5 if party_has_starter else 0)
            self.last_progress_advance_step = self.total_steps
            self.starter_obtained_step = self.total_steps
            self._claim_journey_milestone(
                "journey_starter", "journey_seen_starter"
            )

            bridge = self._maybe_save_progress_bridge("starter")
            if bridge:
                milestone_saved = bridge

            if in_battle == 0:
                if self._save_curriculum_state("starter"):
                    milestone_saved = "starter"

            # V10.31: die Starter-Spezialisten-Episode endet NICHT mehr sofort
            # beim Erhalt des Starters. Der halbe Bonus jetzt, der Rest erst
            # beim Verlassen des Labors (starter_outdoor unten). Sonst lernt
            # kein Agent je, aus Eichs Labor rauszulaufen.
            if self.training_objective == "starter":
                reward += self.SPECIALIST_SUCCESS_BONUS * 0.4
                reward_events.append(
                    f"objective_starter_got:+{self.SPECIALIST_SUCCESS_BONUS * 0.4:.0f}"
                )
                self.objective_success = True

            if self._is_starter_rusher():
                reward += self.STARTER_RUSH_OBJECTIVE_BONUS
                reward_events.append(
                    f"starter_rush_success:+{self.STARTER_RUSH_OBJECTIVE_BONUS:.0f}"
                )
                self.objective_success = True
                objective_done = True

        # V10.31 STARTER-OUTDOOR: erster Overworld-Schritt MIT Starter.
        # Das ist die eigentliche Wand ("raus aus Eichs Labor"). Einmal pro
        # Episode, nicht farmbar. Belohnt + speichert einen Curriculum-State,
        # damit Progress-Agenten direkt draussen mit Starter starten koennen.
        if (
            self.has_starter
            and not self.starter_outdoor_rewarded
            and bank == self.OVERWORLD_BANK
            and loc["valid"]
        ):
            self.starter_outdoor_rewarded = True
            reward += 150.0
            reward_events.append("starter_outdoor:+150")
            # V10.33: den Weg raus aus dem Labor als Brotkrumen-Spur sichern.
            self._commit_journey_route()
            self.reward_event_counts["starter_outdoor"] = (
                self.reward_event_counts.get("starter_outdoor", 0) + 1
            )
            self.episode_milestone_steps.setdefault(
                "starter_outdoor", self.total_steps
            )
            self.last_progress_advance_step = self.total_steps
            if in_battle == 0:
                if self._save_curriculum_state("starter_outdoor"):
                    milestone_saved = "starter_outdoor"
            bridge = self._maybe_save_progress_bridge("starter_outdoor")
            if bridge:
                milestone_saved = bridge
            if self.training_objective == "starter":
                reward += self.SPECIALIST_SUCCESS_BONUS * 0.6
                reward_events.append(
                    f"objective_starter_out:+{self.SPECIALIST_SUCCESS_BONUS * 0.6:.0f}"
                )
                self.objective_success = True
                objective_done = True

        # Einzel-Slot-Levelreward entfernt: V10.27A nutzt oben Team-Level.
        self.last_level = max(self.last_level, p_lvl)

        if badges > self.last_badges:
            badge_gain = badges - self.last_badges
            reward += badge_gain * 500.0
            self.reward_event_counts["badge"] += badge_gain
            reward_events.append(f"badge:+{badge_gain * 500}")
            self.last_badges = badges
            self.last_progress_advance_step = self.total_steps
            if self.training_objective == "badge":
                reward += self.SPECIALIST_SUCCESS_BONUS
                reward_events.append(
                    f"objective_badge:+{self.SPECIALIST_SUCCESS_BONUS:.0f}"
                )
                self.objective_success = True
                objective_done = True
            if badges >= 1:
                self._claim_journey_milestone("journey_badge1","journey_seen_badge1")
            bridge = self._maybe_save_progress_bridge("badge")
            if bridge:
                milestone_saved = bridge

            if in_battle == 0:
                badge_name = f"badge_{badges}"
                if self._save_curriculum_state(badge_name):
                    milestone_saved = badge_name

        if (
            self._valid_coord(bank, map_id, x, y) and
            in_battle == 0
        ):
            map_key = (bank, map_id)
            coord_key = (bank, map_id, x, y)

            # V14: Welt-Roller, die Route 1 noch nicht ueberwunden haben
            # (stage < 3), bekommen fuer Alabastia-INNENRAEUME (Bank 4)
            # NULL Map-Reward. Eine Haus-Tour darf weder Score noch Progress
            # erzeugen. Story-Rollen (intro/stairs/exit/starter) sind davon
            # ausgenommen - die brauchen die Innenraeume.
            _route_roller = self.training_objective in (
                "progress", "battle", "level", "full"
            )
            _indoor_muted = (
                _route_roller
                and int(bank) == 4
                and self._world_stage() < 3
            )

            if map_key not in self.visited_maps:
                self.visited_maps.add(map_key)

                if map_key not in self.learning_seen_maps:
                    self.learning_seen_maps.add(map_key)
                    if map_key in self.persistent_known_maps and not _indoor_muted:
                        reward += self.REPLAY_MAP_REWARD
                        reward_events.append(f"replay_map:+{self.REPLAY_MAP_REWARD:.2f}")

                if map_key not in self.persistent_known_maps:
                    self.persistent_known_maps.add(map_key)
                    self.exploration_memory_dirty = True
                    self._nav_target_cache = None

                    if _indoor_muted:
                        reward_events.append("new_map_indoor_muted:+0")
                    elif self._claim_shared(self.shared_maps, map_key):
                        self.last_progress_advance_step = self.total_steps
                        reward += self.NEW_MAP_REWARD
                        reward_events.append(
                            f"new_map_global:+{self.NEW_MAP_REWARD:.2f}"
                        )
                    else:
                        local_map_reward = self.NEW_MAP_REWARD * 0.25
                        reward += local_map_reward
                        self.last_progress_advance_step = self.total_steps
                        reward_events.append(
                            f"new_map_local:+{local_map_reward:.2f}"
                        )

                # V14: TIEFEN-BONUS an world_stage gekoppelt (nicht an
                # Raumanzahl). Nur ein echter Stage-Anstieg zahlt.
                _stage = self._world_stage()
                if _stage >= 2 and self._claim_global_depth(_stage):
                    depth_bonus = self.NEW_GLOBAL_DEPTH_REWARD * _stage
                    reward += depth_bonus
                    reward_events.append(
                        f"world_depth:{_stage}:+{depth_bonus:.0f}"
                    )
                    self.reward_event_counts["world_depth"] = (
                        self.reward_event_counts.get("world_depth", 0) + 1
                    )
                    self.run_stats["global_depth_records"] += 1
                    self._save_run_stats()
                    self.last_progress_advance_step = self.total_steps

            # V14: STAGE-CHECKPOINT - EIGENER Block, jeden Schritt, NICHT im
            # "neue Map"-Zweig (dort lief der Zaehler nur 1x und erreichte nie
            # 30). Speichert einen validierten Resume-Punkt (Sidecar-Meta:
            # Stage/Position/Starter) sobald der Agent >= 25 Schritte am Stueck
            # tief auf einer neuen Stage steht - auf Korridor-Maps nur weit im
            # Norden (y <= 18).
            if (
                _route_roller
                and self.has_starter
                and in_battle == 0
                and loc["valid"]
                and int(bank) == self.OVERWORLD_BANK
            ):
                _stage_now = self._world_stage()
                _mk = (int(bank), int(map_id))
                if _mk == getattr(self, "_stage_hold_map", None):
                    self._stage_hold_steps = getattr(self, "_stage_hold_steps", 0) + 1
                else:
                    self._stage_hold_map = _mk
                    self._stage_hold_steps = 1
                _north_ok = (
                    _mk not in self.NORTH_CORRIDOR_MAPS or int(y) <= 18
                )
                if (
                    _stage_now >= 2
                    and _stage_now > int(getattr(self, "_saved_stage", 0))
                    and self._stage_hold_steps >= 25
                    and _north_ok
                ):
                    if self._save_stage_checkpoint(_stage_now, bank, map_id, x, y):
                        milestone_saved = f"stage_{_stage_now}"
                        self._saved_stage = _stage_now
                        self._commit_journey_route()
                        self.episode_journey_edges = []
                        bridge = self._maybe_save_progress_bridge(
                            f"stage_{_stage_now}"
                        )
                        if bridge:
                            milestone_saved = bridge

            if coord_key not in self.seen_coords:
                self.seen_coords.add(coord_key)

            # -----------------------------------------------------
            # PERSISTENTE BLUE-LINE / EDGE EXPLORATION
            # -----------------------------------------------------
            # Ein "blauer Strich" ist eine echte Bewegung um genau ein Tile auf
            # derselben Map. Nur ein noch nie bekannter Linienabschnitt gibt Reward.
            # A->B und B->A sind derselbe Abschnitt.
            if self.last_exploration_coord is not None:
                pb, pm, px, py = self.last_exploration_coord

                if (pb, pm) == (bank, map_id):
                    manhattan = abs(x - px) + abs(y - py)

                    if manhattan == 1:
                        edge_key = self._edge_key(
                            bank, map_id, px, py, x, y
                        )

                        if edge_key not in self.learning_seen_edges:
                            self.learning_seen_edges.add(edge_key)
                            if edge_key in self.persistent_known_edges:
                                reward += self.REPLAY_EDGE_REWARD
                                reward_events.append(f"replay_edge:+{self.REPLAY_EDGE_REWARD:.2f}")
                        visit_count = self.episode_edge_visits.get(
                            edge_key, 0
                        ) + 1
                        self.episode_edge_visits[edge_key] = visit_count

                        local_new = (
                            edge_key not in self.persistent_known_edges
                        )

                        if local_new:
                            self.persistent_known_edges.add(edge_key)
                            self.exploration_memory_dirty = True
                            self._invalidate_navigation_cache()

                            if self._claim_shared(
                                self.shared_edges, edge_key
                            ):
                                self.steps_since_new_edge = 0
                                reward += self.NEW_EDGE_REWARD
                                reward_events.append(
                                    "new_edge_global:"
                                    f"+{self.NEW_EDGE_REWARD:.2f}"
                                )
                            else:
                                # V10.2:
                                # Edge ist global bekannt, aber fuer diesen
                                # Agent erstmals gelaufen. Positives Imitations-
                                # signal statt den richtigen Weg neutral zu machen.
                                local_edge_reward = self.NEW_EDGE_REWARD * 0.20
                                self.steps_since_new_edge = 0
                                reward += local_edge_reward
                                reward_events.append(
                                    "new_edge_local:"
                                    f"+{local_edge_reward:.2f}"
                                )
                        else:
                            self.steps_since_new_edge += (
                                self.LOCATION_READ_EVERY
                            )

                        # Ein benoetigter bekannter Weg darf einmal pro Episode
                        # kostenlos benutzt werden. Wiederholung wird negativ.
                        if visit_count == 2:
                            revisit_penalty = (
                                self.INDOOR_SECOND_EDGE_PENALTY
                                if bank != self.OVERWORLD_BANK
                                else self.SECOND_EDGE_VISIT_PENALTY
                            )
                            reward += revisit_penalty
                            reward_events.append(
                                "edge_revisit:"
                                f"{revisit_penalty:.2f}"
                            )
                        elif (
                            visit_count
                            >= self.REPEAT_EDGE_VISITS_FOR_LOOP
                        ):
                            repeat_penalty = (
                                self.INDOOR_REPEAT_EDGE_PENALTY
                                if bank != self.OVERWORLD_BANK
                                else self.REPEAT_EDGE_PENALTY
                            )
                            reward += repeat_penalty
                            reward_events.append(
                                "repeat_edge:"
                                f"{repeat_penalty:.2f}"
                            )

                        # Wiederholbarer, nicht farmbarer Ziel-Fortschritt:
                        # naeher und weiter sind symmetrisch.
                        targets = self._target_coords_for_stage(
                            bank, map_id
                        )
                        if (
                            not targets
                            and self.left_house_rewarded
                            and self.training_objective
                                in ("progress", "full")
                        ):
                            targets = self._progress_targets_for_map(
                                bank, map_id, x, y
                            )
                        if targets:
                            prev_d = self._graph_distance(
                                bank, map_id, (px, py), targets
                            )
                            new_d = self._graph_distance(
                                bank, map_id, (x, y), targets
                            )

                            if (
                                prev_d is not None
                                and new_d is not None
                            ):
                                if new_d < prev_d:
                                    reward += self.TARGET_PROGRESS_REWARD
                                    reward_events.append(
                                        "target_closer:"
                                        f"+{self.TARGET_PROGRESS_REWARD:.2f}"
                                    )
                                elif new_d > prev_d:
                                    reward -= self.TARGET_PROGRESS_REWARD
                                    reward_events.append(
                                        "target_farther:"
                                        f"-{self.TARGET_PROGRESS_REWARD:.2f}"
                                    )

                else:
                    # Mapwechsel / Warp: Ein konkreter Ein-/Ausgangspunkt wird
                    # persistent gespeichert. Rueckweg durch dieselbe Tuer = bekannt.
                    transition_key = self._transition_key(
                        pb, pm, px, py,
                        bank, map_id, x, y
                    )

                    if transition_key not in self.learning_seen_transitions:
                        self.learning_seen_transitions.add(transition_key)
                        if transition_key in self.persistent_known_transitions:
                            reward += self.REPLAY_TRANSITION_REWARD
                            self.last_progress_advance_step = self.total_steps
                            reward_events.append(f"replay_transition:+{self.REPLAY_TRANSITION_REWARD:.2f}")
                    if (
                        transition_key
                        not in self.persistent_known_transitions
                    ):
                        self.persistent_known_transitions.add(
                            transition_key
                        )
                        self.exploration_memory_dirty = True
                        self._invalidate_navigation_cache()
                        self.steps_since_new_edge = 0

                        if self._claim_shared(
                            self.shared_transitions,
                            transition_key
                        ):
                            self.last_progress_advance_step = self.total_steps
                            reward += self.NEW_TRANSITION_REWARD
                            reward_events.append(
                                "new_transition_global:"
                                f"+{self.NEW_TRANSITION_REWARD:.2f}"
                            )
                        else:
                            # V10.2:
                            # Besonders wichtig fuer Treppen/Tueren:
                            # Jeder Agent bekommt beim ersten eigenen Benutzen
                            # eines bereits bekannten Warps ein Lernsignal.
                            local_transition_reward = (
                                self.NEW_TRANSITION_REWARD * 0.50
                            )
                            reward += local_transition_reward
                            self.last_progress_advance_step = self.total_steps
                            reward_events.append(
                                "new_transition_local:"
                                f"+{local_transition_reward:.2f}"
                            )

                        # Sofort fuer alle lokalen Zielabfragen sichtbar.
                        self.shared_transition_snapshot.add(
                            transition_key
                        )
                        if len(self.persistent_known_transitions | self.shared_transition_snapshot) >= 5:
                            self._claim_journey_milestone("journey_warp5","journey_seen_warp5")
                        bridge = self._maybe_save_progress_bridge(
                            "new_transition"
                        )
                        if bridge:
                            milestone_saved = bridge

            self.last_exploration_coord = coord_key
            self.last_exploration_map = map_key

            if coord_key != self.last_pos:
                self.recent_path.append([bank, map_id, x, y])
                self.recent_path = self.recent_path[-300:]
                self.last_pos = coord_key

        # ---------------------------------------------------------
        # EARLY-GAME STORY TIMEOUTS
        # ---------------------------------------------------------
        # Spezialisten werden aggressiv neu gestartet. Full-Runs besitzen
        # weiter unten eigene Stage-Caps bzw. den langen 32k-Horizont und
        # duerfen hier deshalb nicht schon nach 1800 Schritten enden.
        #
        # - Intro nicht innerhalb 900 Episode-Steps fertig -> Reset
        # - Nach Intro: max. 1500 weitere Steps bis F1/Treppe
        # - Nach Treppe: max. 2000 weitere Steps bis Hausausgang
        # - Nach Verlassen des Hauses: keine Early-Game-Begrenzung mehr
        if (
            not self.left_house_confirmed
            and self.training_objective != "full"
        ):
            stage_timeout = None

            # Absoluter Failsafe: Solange das Start-Haus nicht bestaetigt
            # verlassen wurde, darf KEINE Episode (Beginning oder Curriculum)
            # bis zum globalen 8192-Limit laufen.
            if self.total_steps >= self.EARLY_HOUSE_HARD_CAP:
                stage_timeout = "early_house_hard_cap"

            elif (
                not self.intro_complete_rewarded
                and self.total_steps >= self.INTRO_TIMEOUT_STEPS
            ):
                stage_timeout = "intro_timeout"

            elif (
                self.intro_complete_rewarded
                and not self.stairs_down_rewarded
            ):
                intro_step = self.episode_milestone_steps.get(
                    "intro_complete", 0
                )
                if self.total_steps - intro_step >= self.STAIRS_TIMEOUT_STEPS:
                    stage_timeout = "stairs_timeout"

            elif self.stairs_down_rewarded:
                stairs_step = self.episode_milestone_steps.get(
                    "stairs_down", 0
                )
                if self.total_steps - stairs_step >= self.EXIT_TIMEOUT_STEPS:
                    stage_timeout = "house_exit_timeout"

            if stage_timeout is not None and not truncated:
                truncated = True
                info["stage_timeout"] = stage_timeout
                self.last_stage_timeout = stage_timeout
                reward -= 1.0
                reward_events.append(f"{stage_timeout}:-1")
                self.anti_loop_resets += 1
                self.episode_anti_loop_resets += 1
            else:
                info["stage_timeout"] = None

        # Anti-Loop erst nach echter, vertrauenswuerdiger Weltposition.
        if gameplay_ready:
            progress_signature = (
                bank,
                map_id,
                x,
                y,
                p_lvl,
                badges,
                in_battle
            )

            if progress_signature != self.last_progress_signature:
                self.stuck_counter = 0
                self.last_progress_signature = progress_signature
            else:
                self.stuck_counter += 1

            if in_battle == 0 and self.stuck_counter >= 60:
                reward -= 0.03
            if in_battle == 0 and self.stuck_counter >= 180:
                reward -= 0.12
            if in_battle == 0 and self.stuck_counter >= 400:
                reward -= 0.40

            if in_battle == 0 and self.stuck_counter >= 900:
                truncated = True
                info["anti_loop_reset"] = True
                self.anti_loop_resets += 1
                self.episode_anti_loop_resets += 1
            else:
                info["anti_loop_reset"] = False

            self.previous_valid_bank = bank
            self.previous_valid_map = map_id
        else:
            self.stuck_counter = 0
            self.last_progress_signature = None
            info["anti_loop_reset"] = False

        # V7.7: Starter-Rusher trainieren nur Beginning -> Starter.
        if (
            self._is_starter_rusher()
            and not self.has_starter
            and in_battle == 0
            and self.total_steps >= self.STARTER_RUSH_TIMEOUT
        ):
            truncated = True
            self.last_stage_timeout = "starter_rush_timeout"
            reward_events.append("starter_rush_timeout:truncate")

        # V7.6.1: Progress-Agent nach Starter neu ansetzen,
        # falls 3000 Steps kein Map/Story/Level-Fortschritt kam.
        if (
            self.training_objective == "progress"
            and self.has_starter
            and in_battle == 0
            and self.total_steps - self.last_progress_advance_step
                >= self.POST_STARTER_STALL_TIMEOUT
        ):
            truncated = True
            self.last_stage_timeout = "post_starter_stall"
            reward_events.append("post_starter_stall:truncate")

        # V10.25: kurze Full-Probes behalten die Stage-Caps. Die langen
        # Beginning-Probes muessen davon ausgenommen sein, sonst enden sie wie
        # bisher bereits bei ca. 1800 statt erst bei 32768 Schritten.
        if (
            self.training_objective == "full"
            and not self._is_long_full_probe()
            and not truncated
        ):
            if not self.intro_complete_rewarded and self.total_steps >= self.FULL_INTRO_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_intro_cap"
                reward_events.append("full_intro_cap:truncate")
            elif self.intro_complete_rewarded and not self.stairs_down_rewarded and self.total_steps >= self.FULL_STAIRS_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_stairs_cap"
                reward_events.append("full_stairs_cap:truncate")
            elif self.stairs_down_rewarded and not self.left_house_rewarded and self.total_steps >= self.FULL_EXIT_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_exit_cap"
                reward_events.append("full_exit_cap:truncate")

        # V10.15: ten long full probes get a real 32k horizon.
        if self._is_long_full_probe() and self.total_steps >= self.LONG_FULL_PROBE_STEPS and not truncated:
            truncated = True
            self.last_stage_timeout = "long_full_32k"
            reward_events.append("long_full_32k:truncate")
        # V10.31: Starter-Spezialist hat den Starter, kommt aber seit 4000
        # Schritten nicht aus dem Labor -> neu ansetzen (schnelle Iteration
        # fuer genau diese Passage).
        if (
            self.training_objective == "starter"
            and self.has_starter
            and not self.starter_outdoor_rewarded
            and self.starter_obtained_step is not None
            and self.total_steps - self.starter_obtained_step >= 4000
            and not truncated
        ):
            truncated = True
            self.last_stage_timeout = "starter_exit_stall"
            reward_events.append("starter_exit_stall:truncate")

        # V8 specialist timeouts.
        specialist_timeout = None
        if self.training_objective == "starter" and not self.has_starter:
            specialist_timeout = self.STARTER_SPECIALIST_TIMEOUT
        elif self.training_objective == "battle":
            specialist_timeout = self.BATTLE_SPECIALIST_TIMEOUT
        elif self.training_objective == "level":
            specialist_timeout = self.LEVEL_SPECIALIST_TIMEOUT
        elif self.training_objective == "badge":
            specialist_timeout = self.BADGE_SPECIALIST_TIMEOUT

        if (
            specialist_timeout is not None
            and not objective_done
            and self.total_steps >= specialist_timeout
            and not truncated
        ):
            truncated = True
            self.last_stage_timeout = f"{self.training_objective}_timeout"
            reward_events.append(
                f"{self.training_objective}_timeout:truncate"
            )

        # V10.30 GESTAFFELTE PROGRESS-HORIZONTE: statt eines einzigen langen
        # 65k-Laufs bekommt jeder Progress-Agent je nach Slot 8k / 16k / 28k
        # Steps. Kurze Laeufe = viele Resets = schnelleres Credit-Assignment
        # fuer "Nordausgang finden + Route 1 anfangen"; lange Laeufe halten
        # die Route-1->Vertania->Route-2-Ketten. PROGRESS_STALL bleibt aktiv.
        if self.training_objective == "progress" and not truncated:
            _tier = (self.rank % self.n_envs) % 3
            _prog_cap = (12000, 22000, 40000)[_tier]
            # V11.4: Wer von der TIEFSTEN Aussen-Position resumt, steht an der
            # echten Grenze (z.B. Route 1 -> Vertania) und braucht Zeit zum
            # Durchqueren - nicht alle 8k Steps zurueckgesetzt werden.
            if str(self.episode_start).startswith("outdoor_"):
                _prog_cap = max(_prog_cap, 45000)
            if self.total_steps >= _prog_cap:
                truncated = True
                self.last_stage_timeout = f"progress_tier{_tier}_cap"
                reward_events.append(f"progress_tier{_tier}_cap:truncate")

        self.current_reward += reward
        info["step_reward"] = round(float(reward), 4)
        info["episode_reward"] = round(float(self.current_reward), 2)
        info["reward_events"] = reward_events
        info["ram_valid"] = bool(loc.get("valid", False))
        info["ram_trusted"] = bool(loc.get("trusted", False))
        info["ram_source"] = loc.get("source", "unknown")

        # Zusaetzliche Infos fuer Training/Callback/Debug.
        info["episode_start"] = self.episode_start
        info["training_objective"] = self.training_objective
        info["training_role"] = self._agent_role()[0]
        info["curriculum_states"] = len(self.saved_milestones)
        info["milestone_saved"] = milestone_saved
        info["explored_tiles"] = len(self.seen_coords)
        info["visited_maps"] = len(self.visited_maps)
        info["stuck_counter"] = self.stuck_counter
        info["episode_steps"] = self.total_steps
        info["story_stage"] = (
            "OUTDOOR"
            if self.left_house_confirmed else
            "F1_TO_EXIT"
            if self.stairs_down_rewarded else
            "F2_TO_STAIRS"
            if self.intro_complete_rewarded else
            "INTRO"
        )
        info["left_house_confirmed"] = self.left_house_confirmed
        info["has_starter"] = bool(self.has_starter)
        info["level"] = int(self.last_level)
        info["badges_count"] = int(self.last_badges)
        info["world_stage"] = int(self._world_stage())
        info["frontier_maps"] = int(len(self.visited_maps))
        info["has_starter"] = bool(self.has_starter)
        info["level"] = int(self.last_level)
        info["badges_count"] = int(self.last_badges)
        info["outdoor_confirm_reads"] = self.outdoor_confirm_reads
        info["last_stage_timeout"] = self.last_stage_timeout

        if self.total_steps % 80 == 0:
            inst_file = os.path.join(
                INSTANCES_DIR,
                f"inst_{self.rank:02d}.json"
            )
            tmp_file = os.path.join(
                INSTANCES_DIR,
                f"tmp_{self.rank:02d}.json"
            )

            try:
                data = {
                    "id": self.rank,
                    "name": self._agent_role()[1],
                    "agent_role": self._agent_role()[0],
                    "bank": bank,
                    "map": map_id,
                    "x": x,
                    "y": y,
                    "path": self.recent_path,
                    "room": f"Bank {bank} / Map {map_id}",
                    "steps": self.total_steps,
                    "reward": round(self.current_reward, 2),
                    "level": self.last_level,
                    "badges": self.last_badges,
                    "party": self.player_party_cache,
                    "party_summary": {
                        "size": len(self.player_party_cache or []),
                        "max_level": max(
                            [int(m.get("level", 0)) for m in (self.player_party_cache or [])]
                            or [0]
                        ),
                    },
                    "has_starter": bool(self.has_starter),
                    "training_phase": (
                        "forest_push"
                        if self.full_chain_ready
                        else "chain_repair"
                        if "starter" in set(self.saved_milestones)
                        else "starter_breakthrough"
                    ),
                    "in_battle": in_battle,
                    "battle_stats": {
                        "started": int(self.run_stats.get("battles_started", 0)),
                        "completed": int(self.run_stats.get("battles_completed", 0)),
                        "episode_started": int(self.episode_battles_started),
                        "episode_completed": int(self.episode_battles_completed),
                        "enemy_damage_hp": int(self.episode_enemy_damage_hp),
                        "enemy_damage_reward": round(float(self.episode_enemy_damage_reward), 2),
                        "enemy_faints": int(self.episode_enemy_faints),
                    },
                    "enemy_party": self.enemy_party_cache,
                    "world_stage": int(self._world_stage()),
                    "global_depth": {
                        "episode_maps": int(len(self.visited_maps)),
                        "episode_stage": int(self._world_stage()),
                        "record_stage": int(
                            self.shared_progress.get("max_world_stage", 0)
                        ) if self.shared_progress is not None else 0,
                    },
                    "journey_stats": {
                        "starter": int(self.run_stats.get("journey_starter", 0)),
                        "map5": int(self.run_stats.get("journey_map5", 0)),
                        "map10": int(self.run_stats.get("journey_map10", 0)),
                        "warp5": int(self.run_stats.get("journey_warp5", 0)),
                        "progress": int(self.run_stats.get("journey_progress_checkpoint", 0)),
                        "badge1": int(self.run_stats.get("journey_badge1", 0)),
                    },
                    "explored_tiles": len(self.seen_coords),
                    "visited_maps": len(self.visited_maps),
                    "stuck_counter": self.stuck_counter,
                    "episode_start": self.episode_start,
                    "training_objective": self.training_objective,
                    "training_role": (
                        "starter_rush"
                        if self._is_starter_rusher()
                        else self.training_objective
                    ),
                    "requested_action": requested_action,
                    "effective_action": effective_action,
                    "start_spam_count": self.start_spam_count,
                    "objective_success": self.objective_success,
                    "left_house_confirmed": self.left_house_confirmed,
                    "outdoor_confirm_reads": self.outdoor_confirm_reads,
                    "last_stage_timeout": self.last_stage_timeout,
                    "story_stage": (
                        "OUTDOOR"
                        if self.left_house_confirmed else
                        "F1_TO_EXIT"
                        if self.stairs_down_rewarded else
                        "F2_TO_STAIRS"
                        if self.intro_complete_rewarded else
                        "INTRO"
                    ),
                    "curriculum_states": len(self.saved_milestones),
                    "milestone_saved": milestone_saved,
                    "reward_events": reward_events,
                    "persistent_exploration": {
                        "known_edges": len(self.persistent_known_edges),
                        "known_maps": len(self.persistent_known_maps),
                        "known_transitions":
                            len(self.persistent_known_transitions),
                        "steps_since_new_edge":
                            int(self.steps_since_new_edge),
                        "target_guidance_active":
                            bool(
                                self._nav_target(
                                    bank, map_id, x, y
                                )
                            ),
                    },
                    "reward_stats": {
                        "episodes": self.completed_episodes,
                        "event_counts": dict(self.reward_event_counts),
                        "anti_loop_resets": self.anti_loop_resets,
                        "avg_episode_reward": round(
                            self.total_episode_reward / self.completed_episodes,
                            2
                        ) if self.completed_episodes else 0.0,
                        "best_episode_reward": round(
                            self.best_episode_reward, 2
                        ) if self.best_episode_reward is not None else 0.0,
                        "current_milestone_steps":
                            dict(self.episode_milestone_steps),
                        "run_stats": dict(self.run_stats),
                    }
                }

                with open(tmp_file, "w") as f:
                    json.dump(data, f)

                os.replace(tmp_file, inst_file)
                self._save_exploration_memory()

            except Exception:
                pass

        # V7.1 Progress Bridge:
        # Progress-Agenten werden neu gestartet, wenn ueber lange Zeit
        # weder neue Map/Warp/Level/Starter/Badge noch anderer echter
        # Fortschritt erreicht wurde. Full-Chain darf weiterlaufen.
        if (
            self.training_objective == "progress"
            and self.total_steps >= self.PROGRESS_STALL_TIMEOUT
            and (
                self.total_steps - self.last_progress_advance_step
                >= self.PROGRESS_STALL_TIMEOUT
            )
        ):
            truncated = True
            info["progress_stall_reset"] = True
        else:
            info["progress_stall_reset"] = False

        episode_limit = (
            self.MAX_EPISODE_STEPS
            if self.training_objective in ("progress", "badge", "full")
            else 32768
        )
        terminated = bool(
            objective_done
            or self.total_steps >= episode_limit
        )

        reward = self._v10171_story_guard(bank, map_id, x, y, reward)

        return (
            self._make_obs(
                raw_screen,
                loc=loc,
                info=info,
            ),
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):
        return None

    def close(self):
        self._save_exploration_memory(force=True)
        self.env.close()
