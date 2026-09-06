import gymnasium as gym
from gymnasium import spaces
import stable_retro as retro
import numpy as np
import cv2
import os
import json
import gzip
import hashlib
from contextlib import nullcontext
import random
from battle_state import BattleState, MainBattleReader
from reward_state import claim_event
from loop_guard import LocalLoopGuard
from firered_ram import (
    read_battle_type_flags,
    read_enemy_party,
    read_player_location,
    read_player_party,
)


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
    BUILD_TAG = "V19_BROCK_RUSH"
    metadata = {"render_modes": []}

    # Training V2 nutzt feste Spezialisten statt einer 90/10-Zufallsquote.

    # Fuer einen langen 4-5h-Lauf deutlich laengere Episoden als bisher.
    # V10.19_1_EARLY_ROUTE_SAFE
    MAX_EPISODE_STEPS = 12000
    # V17.3: Scouts (siehe FRONTIER_SCOUT_SLOTS) resetten bewusst frueher als
    # der Rest der Flotte - haeufigere, kuerzere Durchlaeufe an derselben
    # Front statt eines einzelnen sehr langen Laufs, damit sie die jeweils
    # neue Map schneller und oefter ueben statt sich einmal sehr weit
    # wegzubewegen.
    SCOUT_EPISODE_STEPS = 12000
    MAX_EPISODE_BATTLE_STEPS = 6000
    MAX_SINGLE_BATTLE_STEPS = 2000
    # Aktionsausfuehrung: Taste halten + neutrale Frames. 16 Halte-Frames sind
    # noetig, damit ein D-Pad-Druck wirklich eine Kachel laeuft statt die Figur
    # nur zu drehen; die Ruhe-Frames trennen aufeinanderfolgende Menue-Drücke
    # sauber. MUSS mit watch.py (ACTION_HOLD_FRAMES/ACTION_RELEASE_FRAMES)
    # identisch sein, sonst rendert der Watcher ein anderes Spiel als trainiert.
    # A/B-getestet 2026-09-05 direkt gegen echte Bewegung (nicht nur FPS):
    # 8 Hold-Frames brechen (3 von 12 Tastendruecken bewegen die Figur nicht),
    # 9 Hold-Frames sind exakt so zuverlaessig wie die vorherigen 12 (11/12,
    # der eine "Fehlschlag" ist in beiden Faellen dieselbe echte Wand). 14
    # statt 18 Frames/Entscheidung = ~22% weniger Emulator-Arbeit pro Schritt.
    ACTION_HOLD_FRAMES = 9
    ACTION_RELEASE_FRAMES = 5
    # V15.3: Kein Spezialisten-Bootcamp mehr. Fast alle Agenten spielen den
    # vollen Lauf ab Spielanfang (inkl. Intro) - genau das, woran der Champion
    # gemessen wird. Kein Ueberfitting auf einzelne Resume-States mehr.
    FULL_ONLY_MODE = True
    # V17.3: Anzahl Scout-Slots PRO validierter Stage, die statt eines
    # kompletten Laufs ab Pallet Town vom Stage-Checkpoint aus weiterspielen
    # (siehe _agent_role() / _scout_assigned_stage()). Bewusst klein
    # gehalten, damit der Loewenanteil der Flotte weiterhin vollstaendige
    # Champion-vergleichbare Laeufe liefert.
    # V17.4: gilt jetzt PRO STAGE, nicht mehr als fester Gesamtwert - jede
    # neu validierte Stage bekommt ihre eigenen FRONTIER_SCOUT_SLOTS Scouts
    # dazu, bestehende Stages behalten ihre.
    # V18: 5 -> 2 zurueck. Die Scouts kommen gut durch, die Full-Runner nicht -
    # bei 5 Scouts/Stage (und mehreren Stages) fraessen sie zu viel Flotte auf.
    # V19: 2 -> 3. Der Weg bis Brock ist laenger (Route 2 / Wald / Marmoria /
    # Arena) - etwas mehr Scout-Lerndaten an den tiefen Checkpoints helfen,
    # ohne die Full-Runner-Mehrheit zu gefaehrden.
    FRONTIER_SCOUT_SLOTS = 3
    PROGRESS_STALL_TIMEOUT = 12000
    POST_STARTER_STALL_TIMEOUT = 12000
    STARTER_RUSH_TIMEOUT = 5000
    STARTER_RUSH_OBJECTIVE_BONUS = 100.0
    PROGRESS_CHECKPOINT_COOLDOWN = 800
    STARTER_SPECIALIST_TIMEOUT = 9000
    BATTLE_SPECIALIST_TIMEOUT = 14000
    LEVEL_SPECIALIST_TIMEOUT = 18000
    BADGE_SPECIALIST_TIMEOUT = 32768
    PARTY_READ_EVERY = 8
    SPECIALIST_SUCCESS_BONUS = 200.0
    STARTER_SPECIALIST_BONUS = 400.0
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
    # Winzige Zeitgebuehr statt Bewegungsstrafe: Jede unproduktive Aktion
    # kostet gleich viel. So lohnt sich weder Herumlaufen noch gegen Waende
    # laufen/Menu-Camping. Echte Ziele bleiben um Groessenordnungen staerker
    # (Starter ~1550, neue Map 250, Weltstufe >=600).
    INTRO_STEP_COST = 0.0
    # V17.3: winzige, aber von Null verschiedene Zeitgebuehr - jeder Schritt
    # kostet minimal, damit Herumstehen/Trippeln nicht mehr strikt neutral
    # ist. Bleibt Groessenordnungen kleiner als jedes echte Ziel.
    # V17.4: 5x angehoben (-0.001 -> -0.005). Seit dem heutigen Reward-
    # Umbau sind die echten Ziele viel groesser geworden (Map 100, Stadt
    # 250, Orden 2000), aber das reine Herumstehen/-laufen in laengst
    # bekannten Gebaeuden/an Waenden blieb genauso billig wie vorher - die
    # relative Buchse zwischen "nichts tun" und "weiterziehen" wurde also
    # groesser statt kleiner. Bleibt weiterhin Groessenordnungen kleiner
    # als jedes echte Ziel, macht aber stundenlanges Nichtstun spuerbarer.
    GAMEPLAY_STEP_COST = -0.005
    INTRO_NOVELTY_REWARD = 2.0
    INTRO_NOVELTY_REWARD_CAP = 20.0
    # Meilenstein-Spezialisten (intro/stairs/exit/starter + full vor dem Starter)
    # werden auf Tempo bewertet: fester Zielbonus minus feste Kosten pro Step.
    # Schnellster Lauf = hoechster Reward -> PPO lernt die kuerzeste Route.
    # Ziel-Boni (stairs +150, exit +500/+800, starter ~1550) bleiben um
    # Groessenordnungen groesser, damit Erfolg immer den Timeout schlaegt.
    # -0.10 war zu hart: ueber eine 5000-Step-Episode -500, bevor ueberhaupt
    # ein Erfolg moeglich ist -> Netz lernt "Bewegen = Verlust" und resigniert.
    # -0.02 haelt das Tempo-Signal, ohne Bewegung netto-negativ zu machen.
    SPECIALIST_STEP_COST = -0.02
    SPECIALIST_SPEED_ROLES = ("intro", "stairs", "exit", "starter")
    KNOWN_PATH_NEUTRAL = True

    # START bleibt fuer Menues frei nutzbar - nicht mehr bestraft.
    START_HOUSE_PENALTY = 0.0
    START_REPEAT_PENALTY_2 = 0.0
    START_REPEAT_PENALTY_3PLUS = 0.0
    START_SPAM_RESET_STEPS = 6
    # Dialoge brauchen mehrere A-Druecke. Erst deutlich darueber gilt A auf
    # derselben Kachel ohne Story-/Kampf-/Level-Fortschritt als Regal-/NPC-Loop.
    INTERACTION_SPAM_PENALTY_AFTER = 24
    INTERACTION_SPAM_RESET_AT = 64
    INTERACTION_SPAM_PENALTY = -0.5

    # V17.4: Kanten-Reward komplett deaktiviert (nie wieder, auch nicht beim
    # allerersten Mal). Grund: EPISODE_EDGE_REWARD wurde ueber
    # learning_seen_edges nur PRO EPISODE dedupliziert, nicht ueber die
    # Lebenszeit des Agenten - ein laengst bekannter Loop von 8-10 Kanten
    # direkt am Savestate-Spawn gab dadurch bei JEDEM Episodenstart erneut
    # vollen Reward fuers reine Abklappern, ohne echten Fortschritt. Live
    # beobachtet: Agenten liefen absichtlich moeglichst viele Ecken/Kanten ab
    # statt weiterzuziehen. Tracking (persistent_known_edges/shared_edges,
    # Kanten-Zaehler + Kartenlinien im Dashboard) bleibt bestehen, zahlt aber
    # nie wieder aus.
    NEW_EDGE_REWARD = 0.0
    EPISODE_EDGE_REWARD = 0.0
    # Ersatz fuer den Explorationsanreiz: nicht an die KANTE (Bewegung A->B)
    # gekoppelt, sondern an die einzelne KACHEL (Koordinate) selbst - nur
    # EINMAL UEBER DIE GESAMTE FLOTTE FUER IMMER (shared_tiles +
    # _claim_shared(), ueberlebt auch Episodenwechsel).
    # V17.4-Fix: EPISODE_TILE_REWARD (0.05 pro Kachel, einmal pro Episode)
    # war strukturell dieselbe Farm-Luecke wie beim alten Kanten-Reward -
    # laengst bekannte Innenraeume (z.B. Reds Haus) gaben dafuer risikofrei
    # kleines Dauer-Einkommen, ohne jemals draussen ins hohe Gras/Kampf-
    # Risiko zu muessen. Live beobachtet: viele Agenten liefen absichtlich
    # zurueck ins Starterhaus statt durch Route 1 zu ziehen. Komplett auf 0,
    # damit nur noch echte fleet-weite Erstfunde und Kampf punkten.
    # V18: Erstfund einer Kachel zahlt PRO LAUF (seen_coords, jede Episode
    # frisch) - NICHT fleet-weit einmalig, sonst sieht ein Agent auf laengst
    # bekanntem Boden (z.B. der Watcher) nie einen Kachel-Reward.
    # Handgesetzte, STEILE Leiter pro Story-Aussenmap (an _current_world_stage
    # der Kachel gebunden, NICHT am Gesamtfortschritt): Alabastia zahlt fast
    # nichts (Spawn-Gebiet), ab Route 1 springt es hoch. So schlaegt die
    # "Menge Kacheln hier" (Pallet) nie die "hoehere Rate da vorne" (Route 1+).
    # V19 BROCK RUSH: flachere Kachel-Leiter. Kacheln sind nur noch die
    # "bleib in Bewegung"-Wuerze; den echten Zug nach vorn machen jetzt
    # STAGE_ADVANCE_REWARD (+250/neue Stufe), TARGET_PROGRESS_REWARD
    # (Graph-Distanz +/-0.20) und die Stadt-/Gym-Meilensteine.
    TILE_REWARD_BY_STAGE = {
        1: 0.1,   # Alabastia / Pallet Town - Spawn, praktisch wertlos
        2: 1.5,   # Route 1
        3: 2.0,   # Vertania / Viridian City
        4: 2.5,   # Route 2
        5: 3.0,   # Vertania-Wald / Viridian Forest
        6: 3.0,   # Marmoria / Pewter City
    }
    # Innenraeume nach Stadt-Bank, jeweils unter der Aussen-Kachel ihrer Stadt:
    # Alabastia-Haeuser (4) = 0.1 (wie Alabastia aussen), Vertania (5) = 1.0,
    # Marmoria (6) = 1.5 (~halbe Stadt).
    INTERIOR_TILE_REWARD_BY_BANK = {4: 0.1, 5: 1.0, 6: 1.5}
    INTERIOR_TILE_REWARD_DEFAULT = 1.0
    # Fleet-weit EINMALIGER Zusatz obendrauf, sobald irgendein Agent eine
    # Kachel zum allerersten Mal ueberhaupt betritt (shared_tiles). Trimmt das
    # Hirn auf echten Vorstoss: der erste Fuss nach Marmoria gibt 6 (pro Lauf)
    # + 1 (global) = 7.
    GLOBAL_NEW_TILE_BONUS = 1.0
    NEW_TILE_REWARD = 2.0        # Fallback fuer Aussenmaps ohne Stage-Eintrag
    EPISODE_TILE_REWARD = 0.0
    # V18: pro Karte UND Episode zahlen nur die ersten TILE_REWARD_CAP_PER_MAP
    # neuen Kacheln die volle Leiter, danach nur noch der Bruchteil
    # TILE_REWARD_AFTER_CAP_FACTOR. Ohne das wird das reine Abgrasen einer
    # grossen Startmap (Pallet hat ~80 begehbare Kacheln * 2 = ~160 Reward,
    # jede Episode neu) ein farmbarer Loop, der die Flotte in Alabastia haelt.
    # Der fleet-weit einmalige +1-Zusatz bleibt ungedeckelt (echter Frontier-
    # Fund, nicht farmbar).
    TILE_REWARD_CAP_PER_MAP = 20
    # Innenraeume kleiner gedeckelt (15): ein neues Haus soll man einmal
    # aufdecken koennen, danach zaehlt nichts mehr - kein Gebaeude-Tour-Farm.
    INTERIOR_TILE_CAP_PER_MAP = 15
    # V18/V19: nach dem Deckel zahlen neue Kacheln nur noch 10 % - minimaler
    # "lauf ins Unerkundete"-Krumen gegen Totzonen, aber die Graph-Distanz-
    # Belohnung (TARGET_PROGRESS_REWARD) uebernimmt jetzt die Wegfuehrung.
    TILE_REWARD_AFTER_CAP_FACTOR = 0.1
    # V17.4: kein fleet-weiter Einmal-Jackpot mehr fuer die allererste Map-
    # Entdeckung (ehem. NEW_MAP_REWARD=500, ging strukturell nur an einen
    # einzigen Agenten je Map) - jetzt EIN Wert pro Run, fuer JEDEN Agenten
    # gleich. Route/Gebaeude 100, echte Stadt 250 (siehe CITY_EPISODE_REWARD
    # unten).
    EPISODE_NEW_MAP_REWARD = 50.0   # V19: 25 -> 50 (neue Route erstmals/Lauf)
    # V18: Innenraeume einer echten Stadt (Bank 5 = Vertania, Bank 6 = Marmoria)
    # sind es wert, einmal reinzuschauen - Arena (Orden!), Laden, Haeuser.
    # Fleet-weit EINMALIGER Fund pro Gebaeude-Map (claim_event key
    # building_<b>_<m>, ueberlebt Neustarts). Die Alabastia-Schuppen (Bank 4)
    # bleiben beim kleinen +25.
    CITY_BUILDING_BANKS = {5, 6}
    BUILDING_FIRST_GLOBAL_REWARD = 500.0
    NEW_GLOBAL_DEPTH_REWARD = 0.0
    # V17.2: Wenn ein Agent den bisher tiefsten world_stage ueberhaupt (ueber
    # ALLE Agenten und Episoden seit dem letzten Reset) als Erster erreicht,
    # zahlt das einmalig fuers ganze Brain - danach nie wieder fuer diese
    # Stufe. _claim_global_depth() trug das bereits in run_stats ein, zahlte
    # aber nie Reward aus. Ergaenzt NEW_GLOBAL_DEPTH_REWARD (das pro Episode
    # jedem Agenten seinen eigenen Stufenanstieg belohnt) um einen echten
    # Fleet-weiten Meilenstein-Bonus fuer Route 1 / Vertania / Route 2 /
    # Vertania-Wald / Marmoria usw.
    GLOBAL_STAGE_RECORD_REWARD = 1000.0
    STARTER_REWARD = 1000.0
    # Ein eindeutiges Ziel verhindert, dass drei verschiedene Starterpfade
    # denselben Reward teilen. Schiggy (Species 7) erleichtert Rocko und kann
    # spaeter Surfer lernen; Zerschneider uebernimmt ein zweites Pokemon.
    TARGET_STARTER_SPECIES = 7
    STARTER_SPECIES = {1, 4, 7}
    WRONG_STARTER_PENALTY = -500.0
    # V17.3: Faenge sollen Artenvielfalt lernen statt immer dieselbe haeufige
    # Spezies (Taubsi/Rattfratz/Raupy) zu wiederholen. Species-ID = Pokedex-
    # Nummer (Gen3-interne SPECIES_-Konstanten entsprechen 1:1 der Nummer fuer
    # alle Kanto/Johto-Spezies, z.B. Pikachu = 25).
    # V17.4: NICHT mehr fleet-weit einmalig ueber die gesamte Trainingszeit
    # (jede Episode startet ohne gefangene Mons neu - ein Lifetime-Claim haette
    # ab dem zweiten jemals gefangenen Exemplar einer Art fuer den Rest des
    # Trainings NIE WIEDER Anreiz gegeben, sie zu fangen). Dedup laeuft jetzt
    # ueber episode_caught_species (pro Episode, pro Agent) - jede neue Art
    # in DIESEM Run zahlt, ein zweites Exemplar derselben Art im selben Run
    # ist neutral.
    # V18: Der V17.x-Audit hat fast alle Farm-Loops (Kachel/Kante/Episoden-
    # Kachel/Dup-Fang) auf 0 gesetzt und Stadt/Weltstufe stark angehoben - ein
    # 50-Punkt-Taubsi war danach kaum noch attraktiver als einfach
    # weiterzulaufen, entsprechend selten wurde noch gefangen. Basiswert
    # deutlich hoeher (aber unter CITY_EPISODE_REWARD=250, damit Vorstoss die
    # bessere Wahl bleibt) plus ein levelskalierter Aufschlag: bei gleicher
    # Art lohnt sich das staerkere Exemplar, ohne dass Grinden in tiefem Gras
    # sinnvoll wird (Cap bei Level 20).
    # V19: 120 -> 50 / Level-Bonus 4 -> 2. Fangen soll bis Brock kein
    # ernsthafter Anreiz gegen den Story-Vorstoss sein.
    SPECIES_CAUGHT_FIRST_REWARD = 50.0
    SPECIES_CAUGHT_LEVEL_BONUS = 2.0
    SPECIES_CAUGHT_LEVEL_BONUS_CAP = 20
    SPECIES_CAUGHT_DUPLICATE_PENALTY = 0.0
    # Pikachu ist im Vertania-Wald selten und nicht der reguelaere Weg
    # vorwaerts - ein eigener, deutlich groesserer Bonus obendrauf, nur fuer
    # genau diese Art an genau diesem Ort. Auch dieser ist pro Run (nicht
    # fleet-lifetime): Pikachu wird fuer Orden 2 (Wasser-Typ-Konter) in JEDEM
    # Run wieder gebraucht, nicht nur beim allerersten Fund ueberhaupt.
    PIKACHU_SPECIES_ID = 25
    PIKACHU_FOREST_MAP = (1, 0)
    # V19: 1000 -> 400. Pikachu bleibt sinnvoll (Wasser-Konter fuer Rocko/Misty)
    # und soll VOR Misty mitgenommen werden, ist aber KEIN Pflichtziel fuer
    # Brock - der grosse Zug geht direkt Richtung Marmoria/Arena.
    PIKACHU_FOREST_CAUGHT_REWARD = 400.0
    # V19: einmal pro Episode, wenn Marmoria (Pewter) zum ersten Mal erreicht
    # wird UND Pikachu schon in der Party ist. Anti-Farm: Episode-Flag.
    PEWTER_WITH_PIKACHU_REWARD = 300.0
    # Kaempfe sind unbegrenzt wiederholbar (Wildgras respawnt), anders als
    # Kanten/Maps/Stufen, die pro Fleet-Leben nur einmal zahlen. Auf dem alten
    # Niveau (0.5/30/50) waere ein Kampf ~80-100 Reward in 20-40 Schritten -
    # das schlaegt jede Erkundung, sobald die leichten neuen Kanten ausgehen,
    # und der Agent haengt im Gras fest statt weiterzuziehen. Auf ein Drittel
    # gekuerzt, damit Erkunden strukturell immer die bessere Wahl bleibt.
    # V18: 0.15 -> 0.08. Wild-Gras respawnt endlos, Kampf-Reward war weiter
    # ein positiver Dauerstrom, sobald die einmaligen Vorwaerts-Boni (Kachel/
    # Map/Stadt) in Reichweite abgegrast waren - Live 45% aller Steps im Kampf.
    ENEMY_DAMAGE_REWARD_PER_HP = 0.08
    # V18: 2.0 -> 0.0. Im Kampf zaehlen nur noch KONTINUIERLICHE Signale:
    # zugefuegter Schaden (ENEMY_DAMAGE_REWARD_PER_HP), erlittener Schaden
    # (-0.1/HP), Heilung (+0.1/HP), Level-Up (LEVEL_GAIN_REWARD) und Fangen
    # (SPECIES_CAUGHT_*). KEINE pauschalen Discrete-Boni mehr fuers KO
    # (enemy_faint) oder den Sieg/EP-Anstieg (BATTLE_WIN_REWARD) - die haben
    # das Kaempfen ueberbewertet.
    ENEMY_FAINT_REWARD = 0.0
    # V18: Wildkampf-Abklingen pro Episode. Die ersten WILD_BATTLE_DECAY_AFTER
    # besiegten Wild-Pokemon auf einer WILD_TRAINING_MAP zahlen voll, ab dem
    # naechsten sinken Schaden- UND Level-Up-Reward auf WILD_BATTLE_DECAY_FACTOR.
    # Fruehes Leveln (Orden 1/2 brauchen es) bleibt voll bezahlt, der Dauer-
    # Grind an derselben Stelle wird strukturell unrentabel. Trainer-/Story-
    # Kaempfe und Nicht-Wild-Maps sind nicht betroffen.
    WILD_BATTLE_DECAY_AFTER = 6
    WILD_BATTLE_DECAY_FACTOR = 0.3
    # V18: Trainer-/Arena-Kaempfe zahlen doppelt (Schaden) und sind vom
    # Wild-Abklingen ausgenommen - sie sind der eigentliche Story-Weg.
    TRAINER_BATTLE_REWARD_MULT = 2.0
    LEVEL_GAIN_REWARD = 15.0   # V19: 10 -> 15
    # V17.4: erster echter Orden-Reward als benannte Konstante statt
    # hartcodierter Inline-Zahl - gilt pro gewonnenem Orden (jede Episode
    # neu, kein Fleet-Claim: die Party wird bei jedem Reset zurueckgesetzt,
    # der Orden muss also in jedem Run neu erkaempft werden).
    BADGE_EARNED_REWARD = 3000.0   # V19: 2000 -> 3000 (pro Lauf)
    # V18: fleet-weit EINMALIGER Bonus, wenn die Flotte einen Orden zum
    # allerersten Mal ueberhaupt holt (pro Ordensnummer, reward_events.json
    # key badge_<n>_ever). Der 2000er oben bleibt pro Lauf (Party resettet).
    BADGE_FIRST_GLOBAL_REWARD = 5000.0
    # Kleiner, von der Party-Level-Summe komplett unabhaengiger Anreiz, das
    # Center ueberhaupt aufzusuchen. Nur einmal pro Episode (Flag oben), keine
    # Kopplung an Levelsumme/Party-Groesse - kann also nie durch PC-Box-
    # Ein-/Auslagern verzerrt werden wie LEVEL_GAIN_REWARD vorher.
    POKEMON_CENTER_FIRST_HEAL_REWARD = 10.0
    # Einmaliger Flotten-Bonus fuers allererste Mal ueberhaupt komplett
    # geheilt worden zu sein (RAM-basiert, siehe POKEMON_CENTER_FIRST_HEAL_
    # REWARD oben fuer die genaue Erkennung).
    POKEMON_CENTER_VISIT_GLOBAL_REWARD = 100.0
    # V18: Ein Pokemon-Center ist mehr als eine Heil-Station. Beim ersten
    # Betreten/Heilen setzt FireRed den Wiedereinstiegspunkt: nach einem
    # Party-Wipe erwacht die Party ab dann in DIESEM Center statt im vorigen
    # (Alabastia -> Vertania -> Marmoria ...). Ein weiter vorne liegendes
    # Center zu erschliessen ist damit echter, im Lauf bleibender Fortschritt
    # - unabhaengig von der Himmelsrichtung.
    #   * Betreten (erste Mal pro Lauf, pro Center): kleiner Anreiz reinzugehen
    #   * Erste Heilung in einem Center tiefer als jedes bisher im Lauf
    #     genutzte: Stadt-grosser Bonus, weil der Respawn vorgerueckt ist
    #   * Allererste Heilung ueberhaupt in GENAU diesem Center, fleet-weit
    #     dauerhaft (reward_events.json, key pc_heal_<bank>_<map>)
    POKECENTER_ENTER_REWARD = 50.0        # V19: 100 -> 50
    POKECENTER_ADVANCE_HEAL_REWARD = 500.0  # V19: 250 -> 500 (Respawn-Anker!)
    POKECENTER_FIRST_HEAL_GLOBAL_REWARD = 1000.0
    # Bekannte Center-Innenraum-Maps -> world_stage ihrer Stadt. Wird
    # erweitert, sobald Scouts weitere Center-Map-IDs bestaetigen.
    # POKECENTER_MAPS: alle Etagen (fuer den Betreten-Bonus).
    # POKECENTER_HEAL_MAPS: nur Erdgeschoss mit Schwester (Heil-/Respawn-Bonus).
    POKECENTER_MAPS = {
        (5, 4): 3,   # Vertania City (Viridian) - Center Erdgeschoss
        (5, 5): 3,   # Vertania City (Viridian) - Center Obergeschoss
        (6, 5): 6,   # Marmoria City (Pewter) - Center Erdgeschoss
    }
    POKECENTER_HEAL_MAPS = {
        (5, 4): 3,   # Vertania City (Viridian)
        (6, 5): 6,   # Marmoria City (Pewter)
    }
    # V18: Poke-Markt. Erstmals betreten im Lauf zahlt, allererster Fund
    # fleet-weit einmalig (key mart_<b>_<m>). Ziel: das Hirn weiss ueberhaupt,
    # dass es den Laden gibt (-> Pokebaelle kaufen). Wird erweitert.
    POKEMART_MAPS = {
        (5, 3),   # Vertania City (Viridian) - Poke-Markt
    }
    POKEMART_ENTER_REWARD = 100.0
    POKEMART_FIRST_GLOBAL_REWARD = 1000.0
    EXPERIENCE_GAIN_REWARD_PER_POINT = 0.0
    # V18: 2.0 -> 0.0. Kein pauschaler Reward mehr fuers Gewinnen / den
    # EP-Anstieg im Kampf - im Kampf zaehlen nur noch Schaden, erlittener
    # Schaden, Heilung, Level-Up und Fangen (siehe ENEMY_FAINT_REWARD).
    BATTLE_WIN_REWARD = 0.0
    ENEMY_HP_READ_EVERY = 2
    ENEMY_ACTIVITY_TTL = 96
    FLED_BATTLE_PENALTY = -25.0
    # V17.2: Ein Party-Wipe beendet die Episode NICHT mehr. FireRed
    # teleportiert nach einem Wipe automatisch zum letzten Pokemon-Center
    # (geheilt), die Episode laeuft danach ganz normal weiter - genau wie im
    # echten Spiel. Vorher zwang jeder Wipe einen Reset auf den
    # Savestate-Startpunkt zurueck; die Party kam dadurch nie ueber
    # Level 6-7 hinaus, weil ein laengerer Lauf mit echtem Grinding nie
    # zustande kam. Die -100-Strafe bleibt, nur der Episodenabbruch faellt
    # weg. Damit der Teleport zum Pokecenter nicht als "neue Map" +25/+500
    # durchrutscht, werden Map-/Kanten-Boni fuer POST_WIPE_REWARD_COOLDOWN_
    # STEPS Route-Schritte nach dem Wipe unterdrueckt (Buchfuehrung/Claims
    # laufen normal weiter, nur die Auszahlung pausiert).
    POST_WIPE_REWARD_COOLDOWN_STEPS = 40
    # V19 POST_WIPE_RECOVERY_MODE. Nach einem Wipe zahlen laengst besuchte
    # Tiles/Maps zu Recht nicht erneut - dadurch kann Wildkampf am Respawn
    # der attraktivste Rest-Rewardstrom werden und die Policy bleibt im Gras
    # kaempfen statt zur Front zurueckzulaufen. Waehrend post_wipe_recovery:
    #   * Wildkampf-Rewards zusaetzlich * POST_WIPE_WILD_BATTLE_SCALE
    #     (Trainer-/Gym-/Brock-Kaempfe NICHT reduziert)
    #   * generische Fang-Rewards -> 0 (Pikachu-Wald-Bonus bleibt, eigener if)
    #   * Graph-Distanz zur alten Front mit POST_WIPE_TARGET_PROGRESS_REWARD
    #     (symmetrisch +/-, kein Vor/Zurueck-Loop)
    # Endet, sobald die aktuelle Standort-Stufe >= pre_wipe_best_stage ist,
    # ODER ein tieferer Center-Respawn aktiviert wurde, ODER ein Orden fiel.
    # Dann einmalig +POST_WIPE_FRONT_RECOVERED_REWARD.
    POST_WIPE_WILD_BATTLE_SCALE = 0.05
    POST_WIPE_TARGET_PROGRESS_REWARD = 0.50
    POST_WIPE_FRONT_RECOVERED_REWARD = 300.0
    WILD_TRAINING_MAPS = {(3, 19), (3, 20), (1, 0)}
    # V17.3: Warps/Tueren waren bisher komplett neutral ("reine Kartendaten").
    # V17.4: kein Run-Bonus mehr fuer laengst bekannte Warps (0.0) - nur noch
    # der fleet-weite Einmal-Fund einer bislang unbekannten Tuer zahlt (100).
    # Ein bekannter Warp ist "frei" (kostet nichts), gibt aber auch nichts
    # mehr - Tueren sollen kein wiederholbarer Farm-Loop sein.
    NEW_TRANSITION_REWARD = 100.0
    EPISODE_TRANSITION_REWARD = 0.0
    REPLAY_MAP_REWARD = 25.0
    REPLAY_EDGE_REWARD = 0.0
    REPLAY_TRANSITION_REWARD = 0.0

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
    # Welt-Exploration bekommt keine generischen Koordinaten-Ziele mehr.
    # Diese Ziele bevorzugten wiederholt Haeuser und Sackgassen.
    # V19 BROCK RUSH: Graph-Distanz-Wegfuehrung wieder AN. Naeher zum
    # bekannten Stage-Ziel (_target_coords_for_stage / _progress_targets_for_map,
    # Graph-Distanz - NICHT Kompassrichtung) = +0.20, weiter weg = -0.20.
    # Symmetrisch -> Hin-/Rueckweg netto 0, kein Nord-Bonus.
    TARGET_PROGRESS_REWARD = 0.20
    EARLY_STORY_STEP_REWARD = 0.0

    EXPLORATION_MEMORY_ENABLED = True
    CONFIRMED_WARP_MIN_AGENTS = 2
    CONFIRMED_WARP_REWARD = 0.0
    # V11: nur noch echtes Einfrieren (exakt gleiche Kachel, lange) wird
    # milde bestraft - nicht mehr "-2.0 alle 96 Steps".
    # V15.1: schneller ausloesen, damit Auf-der-Stelle-Drehen frueh kostet.
    V9_STUCK_SAME_POS_STEPS = 60
    V9_STUCK_PENALTY = -0.5
    # V15: Der Welt-Reward bleibt absichtlich klein und eindeutig:
    # neue echte Weltkachel positiv, Wiederholung KOSTET jetzt etwas -
    # sonst ist Im-Kreis-Laufen gratis (V15.1).
    V9_EXPLORER_NEW_TILE_BONUS = 0.0
    V9_EXPLORER_REPEAT_TILE_PENALTY = 0.0
    # V19 BROCK RUSH: 0 -> 250. Zahlt NUR beim neuen Episode-Bestwert der
    # world_stage (episode_best_stage) - Zuruecklaufen auf eine schon erreichte
    # Stufe zahlt nichts (bestehende Logik, siehe unten). * stage_gain, also
    # ein Stufensprung 1->2 = +250.
    STAGE_ADVANCE_REWARD = 250.0

    # V13.4 NORD-KORRIDOR: NUR die geraden Nord-Strecken vor dem Wald. Auf
    # diesen Maps ist "hoch = Weg". Pro neuer noerdlichster Y-Reihe gibt es
    # dichten Reward -> aus den 60 blinden Schritten wird eine Belohnungsrampe.
    # Der Wald (Labyrinth) ist BEWUSST NICHT dabei; dort loest reine
    # Exploration. (3,0)=Alabastia, (3,19)=Route 1, (3,20)=Route 2.
    # Vertania bleibt frei, weil dort der Paket-Rueckweg notwendig ist.
    NORTH_CORRIDOR_MAPS = {(3, 0), (3, 19), (3, 20)}
    PARCEL_RETURN_MAPS = {(3, 1), (3, 19), (3, 0)}
    NORTH_CORRIDOR_ROW_REWARD = 0.0
    # Potential-Differenz pro gelaufener Reihe. Nord->Sued nimmt exakt den
    # zuvor erhaltenen Betrag wieder weg; Hin-und-her-Laufen ist damit kein
    # Reward-Loop. Anders als der reine Rekordbonus liefert das auch nach
    # einem Rueckschritt wieder ein brauchbares Richtungssignal.
    CORRIDOR_STEP_REWARD = 0.0
    NORTH_CORRIDOR_MAX_ROWS = 45
    EXIT_ROUTE_EDGE_REWARD = 0.0
    EXIT_ROUTE_REVERSE_PENALTY = 0.0
    EXIT_ROUTE_REPEAT2_PENALTY = 0.0
    EXIT_ROUTE_REPEAT3_PENALTY = 0.0
    EXIT_ROUTE_CONFIRM_AGENTS = 2
    EXIT_ROUTE_MAX_EDGES = 256

    # V15: Journey-Routen sind deaktiviert. Haeufig gelaufene Rueckwege duerfen
    # nie wieder als vermeintlich guter Pfad positiv verstaerkt werden.
    JOURNEY_ROUTE_ENABLED = False
    JOURNEY_ROUTE_EDGE_REWARD = 0.0
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

    # V15: EINE explizite Wahrheit fuer "wie weit sind wir".
    # Nur echte Story-Maps zaehlen; Gebaeude koennen die Stage nicht erhoehen.
    #   0 = Innen / Intro
    #   1 = Alabastia aussen (3,0)
    #   2 = Route 1 (3,19)
    #   3 = Vertania City (3,1)
    #   4 = Eichs Paket im Vertania-Markt erhalten
    #   5 = Paket bei Eich abgegeben / Pokédex erhalten
    #   6 = Route 2 (3,20)
    #   7 = Vertania-Wald (1,0)
    #   8 = Marmoria City (3,2)
    #   9 = >= 1 Orden
    # Champion, Skill-Vault, Checkpoints, Journey-Routen bewerten NUR world_stage.
    STAGE_PALLET = (3, 0)
    STAGE_ROUTE1 = (3, 19)
    STAGE_VIRIDIAN = (3, 1)
    STAGE_ROUTE2 = (3, 20)
    STAGE_FOREST = (1, 0)
    STAGE_PEWTER = (3, 2)
    PROGRESS_SCHEMA = "geography_v1"
    WORLD_STAGE_BY_MAP = {
        STAGE_PALLET: 1,
        STAGE_ROUTE1: 2,
        STAGE_VIRIDIAN: 3,
        STAGE_ROUTE2: 4,
        STAGE_FOREST: 5,
        STAGE_PEWTER: 6,
    }
    # V17.3: Ankunft in einer echten Stadt zaehlt JEDE Episode, nicht nur
    # beim allerersten Fleet-Fund - Routen bleiben beim generischen
    # EPISODE_NEW_MAP_REWARD. Ziel: schneller/haeufiger bis zur naechsten
    # Stadt vorstossen statt nur beim einmaligen globalen Fund belohnt zu
    # werden.
    CITY_MAPS = {STAGE_PALLET, STAGE_VIRIDIAN, STAGE_PEWTER}
    CITY_EPISODE_REWARD = 300.0   # V19: 250 -> 300

    # V19 BROCK RUSH: Pewter/Brock in kleine, anti-farm-gesicherte Meilensteine
    # aufgeteilt. Alle nur EINMAL pro Episode (Episode-Flags), nicht durch
    # Rein-/Rauslaufen oder wiederholtes Kampfstarten wiederholbar.
    #   PEWTER_GYM_MAPS: Innenraum-Map(s) der Marmoria-Arena. ID noch NICHT
    #     bestaetigt (Bank 6, wie Center (6,5)). Bis ein Scout sie meldet
    #     bleibt der Betreten-Bonus inert; die Brock-/Trainer-Erkennung laeuft
    #     ueber Trainer-Flag + world_stage 6 und ist davon unabhaengig.
    PEWTER_GYM_MAPS = set()          # z.B. {(6, 4)} sobald bestaetigt
    PEWTER_GYM_ENTER_REWARD = 200.0
    BROCK_BATTLE_START_REWARD = 500.0
    PEWTER_GYM_TRAINER_REWARD = 300.0
    SCOUT_STAGES = (2, 3, 4, 5, 6)  # FRONTIER_SCOUT_SLOTS scouts per validated checkpoint after Pallet.
    WORLD_ROLES = ("progress", "battle", "level", "badge", "full", "scout")

    def __init__(
        self,
        rank=0,
        shared_edges=None,
        shared_maps=None,
        shared_transitions=None,
        shared_progress=None,
        shared_lock=None,
        shared_species=None,
        shared_tiles=None,
        n_envs=32,
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
        self.shared_species = shared_species
        # V17.4: fleet-weite Lifetime-Registry fuer "diese Kachel wurde
        # jemals von irgendeinem Agenten betreten" - ersetzt shared_edges als
        # Grundlage fuer den einmaligen globalen Explorations-Bonus.
        self.shared_tiles = shared_tiles
        self.shared_edge_snapshot = set()
        self.shared_transition_snapshot = set()
        # V18: grobe Karten-Paare (map_a<->map_b), fuer die die Flotte schon
        # eine Verbindung kennt - aus der geladenen 8-Tupel-Transition-Historie
        # abgeleitet. Ein Warp-Paar hier zahlt NIE (mehr) den Global-Bonus,
        # auch nicht beim ersten Ueberqueren nach einem Neustart. Genuin neue
        # Paare zahlen einmal und werden dann ueber reward_events.json dauerhaft
        # vermerkt (claim_event), sodass sie auch nach jedem Neustart erledigt
        # bleiben - der Nutzer sah sonst bei jedem Watcher-Neustart wieder
        # new_warp_global fuer Alabastia<->Route 1.
        # Leer starten: das Iterieren des manager.dict-Proxys ist waehrend
        # __init__ im SubprocVecEnv-Worker noch nicht sicher moeglich. Die
        # bekannten Warp-Paare werden beim ersten _refresh_shared_snapshots()
        # (laeuft in step() vor jeder Warp-Reward-Pruefung) aus
        # shared_transitions nachgezogen.
        self._known_warp_pairs = set()

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
        self.last_party_total_experience = 0
        self.faints_in_current_battle = 0
        self.last_party_size = 0
        self.pokemon_center_healed_this_episode = False
        # V18: {(bank,map)} bereits betretener Center + tiefste Stufe, in deren
        # Center diese Episode schon geheilt wurde (Respawn-Punkt-Fortschritt).
        self.pokecenter_entered_this_episode = set()
        self.pokemart_entered_this_episode = set()
        self.best_pokecenter_heal_stage = 0
        # V19 BROCK RUSH: Pewter/Brock-Meilensteine - je einmal pro Episode.
        self.episode_pewter_reached = False
        self.episode_pewter_with_pikachu_rewarded = False
        self.episode_pewter_gym_entered = False
        self.episode_brock_battle_started = False
        self.episode_pewter_gym_trainer_beaten = False
        self.last_party_total_level = 0
        self.indoor_steps_without_transition = 0
        self.battle_activity_open = False
        self.enemy_hp_min = {}
        self.enemy_fainted_rewarded = set()
        # V17.2: fehlte hier komplett. total_steps setzt jede Episode auf 0
        # zurueck, dieser Wert aber nicht - nach der ersten Episode wird
        # total_steps - _last_enemy_seen_step sofort stark negativ, was die
        # "<=96"-Pruefung immer erfuellt. Folge: in_battle haengt am Anfang
        # JEDER Episode ausser der ersten faelschlich auf 1, bis total_steps
        # den alten (episodenfremden) Wert wieder eingeholt hat - oft
        # tausende Schritte. Erklaert sowohl die "haengt in Kampf fest ohne
        # sichtbaren Kampf"-Agenten als auch die ausbleibende Fluchtstrafe
        # (battle_just_ended kann nie feuern, wenn in_battle nie auf 0 faellt).
        self._last_enemy_seen_step = -999
        self.episode_enemy_damage_hp = 0
        self.episode_enemy_damage_reward = 0.0
        self.episode_enemy_faints = 0
        # V18: besiegte Wild-Pokemon auf WILD_TRAINING_MAPS dieser Episode -
        # ab WILD_BATTLE_DECAY_AFTER greift WILD_BATTLE_DECAY_FACTOR.
        self.episode_wild_faints = 0
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

        # Stable-Retro setzt bei State.NONE beim spaeteren reset() keinen
        # Emulatorzustand zurueck. Es fuehrt dann nur einen neutralen Frame
        # aus; Party, Karte und Story-RAM der vorigen Episode bleiben stehen.
        # Einen festen Startzustand deshalb einmal in self.initial_state
        # laden und von nun an bei JEDEM Episodenreset wiederherstellen.
        #
        # Immutable user-recorded master after Oak's parcel: every full runner
        # restores this exact state; scouts alone use geographic checkpoints.
        self.env.load_state(
            "StartGame", inttype=retro.data.Integrations.CUSTOM_ONLY
        )
        self.env.reset()

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

        # V15: vier Bilder geben der Policy Bewegungsrichtung und kurze
        # Aktionsfolgen; ein einzelnes Standbild konnte das nicht zeigen.
        self.IMAGE_STACK = 4
        self._image_frames = []
        # Policy sieht Bildfolge + kompakten RAM/Navigationszustand.
        # Channel-first verhindert automatische Transpose-Magie in SB3.
        self.NAV_DIM = 31
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(self.IMAGE_STACK, 64, 64),
                dtype=np.uint8
            ),
            "nav": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.NAV_DIM,),
                dtype=np.float32
            ),
        })

        # total_steps = jede PPO-Entscheidung. route_steps pausiert im Kampf
        # und ist die faire Uhr fuer Story-Horizonte und Champion-Tempo.
        self.total_steps = 0
        self.route_steps = 0
        self.battle_steps = 0
        self.current_battle_steps = 0
        self.seen_coords = set()
        self._episode_tiles_by_map = {}
        self.visited_maps = set()
        self._saved_outdoor_depth = 0

        # V10.3 non-persistent learning memory
        self.learning_seen_maps = set()
        self.learning_seen_edges = set()
        self.learning_seen_transitions = set()
        self.recent_path = []
        # V17.2: Dashboard-Klick auf einen Agenten zeigte bisher nur die
        # reward_events des einen Steps, in dem die inst_XX.json zufaellig
        # geschrieben wurde (alle 80 Steps) - fast immer leer. Stattdessen
        # rollierendes Log der letzten tatsaechlichen Reward-Ereignisse.
        self.recent_reward_events = []
        # V17.2: Route-Schritt, bis zu dem Map-/Kanten-Boni nach einem
        # Party-Wipe unterdrueckt werden (siehe POST_WIPE_REWARD_COOLDOWN_
        # STEPS). -1 = kein aktiver Wipe-Cooldown.
        self._post_wipe_reward_cooldown_until = -1
        # V19 POST_WIPE_RECOVERY_MODE (siehe POST_WIPE_* Konstanten).
        self.post_wipe_recovery = False
        self.pre_wipe_best_stage = 0
        self.pre_wipe_best_center_stage = 0
        self.pre_wipe_badges = 0

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
        self.has_target_starter = False
        self._starter_species_cached = 0
        self.viridian_mart_scene = 0
        self.viridian_old_man_scene = 0
        self.pallet_oaks_lab_scene = 0
        # The user master is after parcel delivery. Preserve story telemetry;
        # these flags do not contribute to geographic progression.
        self.parcel_obtained_confirmed = True
        self.parcel_delivered_confirmed = True
        self.parcel_obtained_confirm_reads = 0
        self.parcel_delivered_confirm_reads = 0
        # V10.31: die echte Wand ist "raus aus Eichs Labor nach dem Starter"
        # (nicht Route 1). Dafuer gab es bisher keinen Reward-Gradienten.
        self.starter_outdoor_rewarded = False
        # V17.4: Spezies-Faenge sind PRO RUN, nicht mehr fleet-weit einmalig -
        # jede Episode startet ohne gefangene Mons neu, ein fleet-weiter
        # Einmal-Claim wuerde also ab dem zweiten Fund jeder Art fuer den Rest
        # der gesamten Trainingszeit nie wieder Anreiz geben, ueberhaupt noch
        # etwas zu fangen (auch Pikachu, das fuer Orden 2 in JEDEM Run wieder
        # gebraucht wird).
        self.episode_caught_species = set()
        self.episode_pikachu_forest_caught = False
        # V17: wurde bisher nur beim Uebergang "kein Starter -> Starter"
        # gesetzt. Der Startpunkt hat den Starter aber schon ab Step 0, dieser
        # Uebergang passiert also nie mehr - das "muss innerhalb 4000 Steps aus
        # dem Labor"-Sicherheitsnetz waere sonst permanent tot. 0 statt None:
        # die 4000 Steps zaehlen jetzt ab Episodenstart.
        self.starter_obtained_step = 0

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
            "experience_wins": 0,
            "party_wipes": 0,
        }
        self.episode_anti_loop_resets = 0
        self._load_run_stats()

        # Visuelles Intro-Shaping: funktioniert auch bevor FireRed eine
        # verlaessliche Weltposition im SaveBlock bereitstellt.
        self.intro_seen_states = set()
        self.intro_last_thumb = None
        self.intro_same_screen_steps = 0
        self.intro_novelty_reward_total = 0.0
        # V17: der feste Startpunkt liegt bereits nach Intro/Treppe/Hausausgang.
        # Diese Uebergaenge werden nur durch In-Episode-Beobachtung gesetzt und
        # koennen ab dem neuen Startpunkt nie mehr beobachtet werden - False
        # wuerde bei INTRO_TIMEOUT_STEPS/STAIRS_TIMEOUT_STEPS/EXIT_TIMEOUT_STEPS
        # JEDE Episode zwangsweise frueh abbrechen (siehe Zeile ~4330).
        self.intro_complete_rewarded = True

        # Story-Reward-Meilensteine (pro Episode nur einmal).
        self.initial_indoor_map = None
        self.stairs_down_rewarded = True
        self.left_house_rewarded = True
        # V17: separat von left_house_rewarded, aber gleiches Problem - nur
        # Telemetrie (v2_full_left_house%), keine Reward-Wirkung, aber sonst
        # dauerhaft fälschlich 0% obwohl der Hausausgang laengst erledigt ist.
        self.left_house_confirmed = True
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
        self.interaction_anchor = None
        self.interaction_count = 0

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
            if self.has_target_starter:
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
            if self.has_target_starter:
                self.run_stats["v2_full_starter"] += 1
            if self.last_badges >= 1:
                self.run_stats["v7_full_badge1"] += 1

        elif self.training_objective == "starter":
            self.run_stats["v8_starter_episodes"] += 1
            if self.starter_outdoor_rewarded:
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

    @staticmethod
    def _valid_confirmed_story_warp(kind, transition):
        """Allow only the known German FireRed house story map pairs."""
        if not transition or len(transition) != 8:
            return False
        maps = frozenset((
            (int(transition[0]), int(transition[1])),
            (int(transition[4]), int(transition[5])),
        ))
        expected = {
            "stairs": frozenset(((4, 1), (4, 0))),
            "exit": frozenset(((4, 0), (3, 0))),
        }
        return maps == expected.get(kind)

    def _save_confirmed_story_warp(self, kind, transition):
        if not self._valid_confirmed_story_warp(kind, transition):
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
                if self._valid_confirmed_story_warp(kind, raw):
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

        # V15: Welt-Agenten explorieren frei. Generische Frontier-/Warp-Ziele
        # haben zuvor Haeuser, Waende und Rueckwege bevorzugt.
        if self.training_objective in self.WORLD_ROLES:
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

    def _v19_forward_targets(self, bank, map_id):
        """V19 BROCK RUSH: Koordinaten auf DIESER Karte, deren bekannte
        Transition auf eine Karte mit HOEHERER world_stage fuehrt - bzw. auf
        das Center der aktuellen Stadt (solange dieser Lauf dort noch nicht
        geheilt hat) oder die Marmoria-Arena (solange Brock noch nicht lief).
        Rein graph-basiert (_combined_transitions + _graph_distance im Aufrufer),
        KEINE Kompassrichtung. Leer -> keine Wegfuehrung, dann greifen nur
        Kachel-/Stage-Reward. Der Aufrufer bewertet symmetrisch (+/-), also
        kein Farm-Loop durch Hin-und-Her.
        """
        here = (int(bank), int(map_id))
        here_stage = self._current_world_stage(bank, map_id)
        fwd_maps = {m for m, s in self.WORLD_STAGE_BY_MAP.items() if s > here_stage}
        if here == self.STAGE_VIRIDIAN and self.best_pokecenter_heal_stage < 3:
            fwd_maps |= {m for m, s in self.POKECENTER_MAPS.items() if s == 3}
        if here == self.STAGE_PEWTER and not getattr(
            self, "episode_brock_battle_started", False
        ):
            fwd_maps |= set(self.PEWTER_GYM_MAPS)
        if not fwd_maps:
            return []
        targets = []
        for t in self._combined_transitions():
            if len(t) != 8:
                continue
            a = (int(t[0]), int(t[1]), int(t[2]), int(t[3]))
            b = (int(t[4]), int(t[5]), int(t[6]), int(t[7]))
            if (a[0], a[1]) == here and (b[0], b[1]) in fwd_maps:
                targets.append((a[2], a[3]))
            if (b[0], b[1]) == here and (a[0], a[1]) in fwd_maps:
                targets.append((b[2], b[3]))
        return list(dict.fromkeys(targets))

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
            and self.training_objective in ("progress", "full", "scout")
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
            float(np.clip(self.viridian_mart_scene / 2.0, 0.0, 1.0)),
            float(np.clip(self.pallet_oaks_lab_scene / 6.0, 0.0, 1.0)),
            float(np.clip(self.viridian_old_man_scene / 2.0, 0.0, 1.0)),
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

        frame = self._process_image(screen)
        if not self._image_frames:
            self._image_frames = [frame.copy() for _ in range(self.IMAGE_STACK)]
        else:
            self._image_frames.append(frame)
            self._image_frames = self._image_frames[-self.IMAGE_STACK:]

        return {
            "image": np.concatenate(self._image_frames, axis=0),
            "nav": self._build_nav_vector(
                bank=bank,
                map_id=map_id,
                x=x,
                y=y,
                gameplay_ready=gameplay_ready,
                in_battle=int(self.last_in_battle),
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
    # V15: world_stage = einzige Fortschritts-Wahrheit
    # ------------------------------------------------------------------
    @staticmethod
    def _party_identity(party):
        return tuple(sorted((int(m.get("personality", 0)), int(m.get("species_id", 0)))
                            for m in party))

    def _record_party_wipe(self, events, info):
        if getattr(self, "wipe_active", False):
            return 0.0
        self.wipe_active = True
        self.run_stats["party_wipes"] += 1
        self._save_run_stats()
        self._post_wipe_reward_cooldown_until = self.route_steps + self.POST_WIPE_REWARD_COOLDOWN_STEPS
        self.last_stage_timeout = "party_wiped"
        info["last_stage_timeout"] = "party_wiped"
        events.append("party_wiped:-100.0")
        # V19 POST_WIPE_RECOVERY_MODE: KEIN Reset von seen_coords/visited_maps
        # (absichtliches Sterben darf kein Farm-Trick werden). Wir merken uns
        # nur, wie weit die Episode schon war, und schalten den Recovery-Modus
        # ein: bis die alte Front (oder ein tieferer Center / Orden) wieder
        # erreicht ist, sind Wildkaempfe fast wertlos und der Rueckweg zur
        # Storyfront stark belohnt.
        self.post_wipe_recovery = True
        self.pre_wipe_best_stage = int(getattr(self, "episode_best_stage", 0))
        self.pre_wipe_best_center_stage = int(getattr(self, "best_pokecenter_heal_stage", 0))
        self.pre_wipe_badges = int(getattr(self, "last_badges", 0))
        return -100.0

    def _is_trainer_battle(self):
        """gBattleTypeFlags Bit 0x8 = BATTLE_TYPE_TRAINER (Gen3)."""
        try:
            return bool(int(getattr(self.battle_state, "raw_flags", 0)) & 0x8)
        except Exception:
            return False

    def _battle_reward_scale(self, bank, map_id):
        """V18/V19: Multiplikator auf Kampf-Rewards (Schaden / Level-Up).
        * Trainer-/Arena-/Brock-Kaempfe: TRAINER_BATTLE_REWARD_MULT (doppelt),
          nie vom Wild-Abklingen ODER dem Post-Wipe-Modus betroffen.
        * Wildkampf auf einer WILD_TRAINING_MAP: 1.0 bis WILD_BATTLE_DECAY_AFTER
          Siege/Episode, danach WILD_BATTLE_DECAY_FACTOR.
        * V19: waehrend post_wipe_recovery zusaetzlich * POST_WIPE_WILD_BATTLE_SCALE
          - Wildkampf am Respawn soll nicht der lohnendste Rest-Strom sein.
        * Sonst 1.0."""
        if self._is_trainer_battle():
            return self.TRAINER_BATTLE_REWARD_MULT
        scale = 1.0
        if (
            (int(bank), int(map_id)) in self.WILD_TRAINING_MAPS
            and self.episode_wild_faints >= self.WILD_BATTLE_DECAY_AFTER
        ):
            scale = self.WILD_BATTLE_DECAY_FACTOR
        if getattr(self, "post_wipe_recovery", False):
            scale *= self.POST_WIPE_WILD_BATTLE_SCALE
        return scale

    # Rueckwaerts-kompatibler Alias (aeltere Aufrufer/Tests).
    _wild_battle_scale = _battle_reward_scale

    def _can_reward_map_arrival(self, bank, map_id):
        # A scout must be paid for reaching the next map too. Only revisiting
        # stages at/before its own spawn is excluded, not all scout arrivals.
        stage = self._current_world_stage(bank, map_id)
        if (bank, map_id) == self.STAGE_PALLET:
            return False
        return (self.training_objective != "scout" or stage == 0 or
                stage > getattr(self, "episode_start_stage", 1))

    def _current_world_stage(self, bank, map_id):
        return int(self.WORLD_STAGE_BY_MAP.get((int(bank), int(map_id)), 0))

    def _update_story_state_from_loc(self, loc):
        map_key = (
            int(loc.get("map_bank", -1)),
            int(loc.get("map_id", -1)),
        )
        self.viridian_mart_scene = max(
            self.viridian_mart_scene,
            int(loc.get("viridian_mart_scene", 0) or 0),
        )
        self.viridian_old_man_scene = max(
            self.viridian_old_man_scene,
            int(loc.get("viridian_old_man_scene", 0) or 0),
        )
        self.pallet_oaks_lab_scene = max(
            self.pallet_oaks_lab_scene,
            int(loc.get("pallet_oaks_lab_scene", 0) or 0),
        )

        # Ein einzelner plausibler u16-Wert darf keine permanente Story-Stufe
        # mehr setzen. Paket und Abgabe werden nur auf der passenden Karte,
        # in der richtigen Reihenfolge und ueber drei RAM-Lesezyklen bestaetigt.
        if (
            map_key == (5, 3)
            and self.has_target_starter
            and int(loc.get("viridian_mart_scene", 0) or 0) >= 1
        ):
            self.parcel_obtained_confirm_reads += 1
            if self.parcel_obtained_confirm_reads >= 3:
                self.parcel_obtained_confirmed = True
        elif not self.parcel_obtained_confirmed:
            self.parcel_obtained_confirm_reads = 0

        if (
            map_key == (4, 3)
            and self.has_target_starter
            and self.parcel_obtained_confirmed
            and int(loc.get("pallet_oaks_lab_scene", 0) or 0) >= 6
        ):
            self.parcel_delivered_confirm_reads += 1
            if self.parcel_delivered_confirm_reads >= 3:
                self.parcel_delivered_confirmed = True
        elif not self.parcel_delivered_confirmed:
            self.parcel_delivered_confirm_reads = 0

    def _world_stage(self):
        """Geographic progress only. Story flags never manufacture a stage."""
        return max(1, max(
            (self._current_world_stage(*m) for m in self.visited_maps), default=1
        ))

    def _stage_at_current_location(self, bank, map_id):
        return self._current_world_stage(bank, map_id)

    def _meta_checkpoint_stage(self, meta):
        return self._current_world_stage(meta.get("bank", -1), meta.get("map", -1))

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
        """{n: 'stage_n'} fuer Sidecars, deren aktuelle Map exakt zur Stage passt."""
        out = {}
        for name in getattr(self, "saved_milestones", ()) or ():
            if not name.startswith("stage_"):
                continue
            try:
                n = int(name.split("_", 1)[1])
            except Exception:
                continue
            meta = self._read_stage_meta(name)
            if (meta.get("state_validation") == 1
                    and int(meta.get("stage", -1)) == n
                    and bool(meta.get("has_starter"))
                    and self._meta_checkpoint_stage(meta) == n):
                out[n] = name
        return out

    def _save_stage_checkpoint(self, stage, bank, map_id, x, y, episode_reward=0.0):
        # Cached navigation may lag a warp. Read the actual emulator now.
        live = read_player_location(self.env, allow_scan=True)
        if not live.get("trusted") or (
            int(live.get("map_bank", -1)), int(live.get("map_id", -1)),
            int(live.get("x_pos", -1)), int(live.get("y_pos", -1))
        ) != (int(bank), int(map_id), int(x), int(y)):
            return False
        name = f"stage_{int(stage)}"
        meta = {
            "stage": int(stage), "progress_schema": self.PROGRESS_SCHEMA,
            "bank": int(bank), "map": int(map_id),
            "x": int(x), "y": int(y), "has_starter": True,
            "starter_species": int(self._starter_species()),
            "step": int(self.route_steps), "agent": int(self.rank),
            "viridian_mart_scene": int(self.viridian_mart_scene),
            "viridian_old_man_scene": int(self.viridian_old_man_scene),
            "pallet_oaks_lab_scene": int(self.pallet_oaks_lab_scene),
            "parcel_obtained_confirmed": bool(self.parcel_obtained_confirmed),
            "parcel_delivered_confirmed": bool(self.parcel_delivered_confirmed),
            "episode_reward": float(episode_reward),
        }
        if self._stage_at_current_location(bank, map_id) != int(stage):
            return False

        if self.shared_lock is not None:
            self.shared_lock.acquire()
        try:
            existing = self._read_stage_meta(name)
            ok_existing = (
                existing.get("state_validation") == 1
                and int(existing.get("stage", -1)) == int(stage)
                and bool(existing.get("has_starter"))
                and self._meta_checkpoint_stage(existing) == int(stage)
                and os.path.exists(self._shared_state_path(name))
            )
            if ok_existing:
                # North position wins first (smaller map Y). At equal Y,
                # require strictly higher reward. Never move a checkpoint south.
                # Each stage competes independently, including full runners.
                old_y = int(existing.get("y", y))
                old_reward = float(existing.get("episode_reward", 0.0) or 0.0)
                if int(y) > old_y or (
                    int(y) == old_y and float(episode_reward) <= old_reward
                ):
                    return False

            # Auch beim Verbessern atomar ersetzen. So bleibt bei einem
            # Abbruch entweder der alte oder der neue vollstaendige State.
            state_data = self.env.em.get_state()
            meta["state_validation"] = 1
            meta["state_sha256"] = hashlib.sha256(state_data).hexdigest()
            saved_any = False
            for p in (self._state_path(name), self._shared_state_path(name)):
                tmp = p + f".tmp.{os.getpid()}.{self.rank}"
                with gzip.open(tmp, "wb") as f:
                    f.write(state_data)
                os.replace(tmp, p)
                saved_any = True
            if saved_any:
                self._write_stage_meta(name, meta)
                self.saved_milestones = self._discover_saved_milestones()
            return saved_any
        finally:
            if self.shared_lock is not None:
                self.shared_lock.release()

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
        paths = [self._shared_state_path(milestone_name), self._state_path(milestone_name)]
        original = self.env.em.get_state()
        lock = self.shared_lock if self.shared_lock is not None else nullcontext()
        with lock:
            for path in paths:
                if not os.path.exists(path):
                    continue
                try:
                    with gzip.open(path, "rb") as f:
                        state_data = f.read()
                    meta = None
                    if milestone_name.startswith("stage_"):
                        with open(path[:-9] + ".meta.json") as f:
                            meta = json.load(f)
                        if (meta.get("state_validation") != 1 or
                                meta.get("state_sha256") != hashlib.sha256(state_data).hexdigest()):
                            continue
                    self.env.em.set_state(state_data)
                    if meta is not None:
                        live = read_player_location(self.env, allow_scan=True)
                        stage = int(milestone_name.split("_", 1)[1])
                        if not live.get("trusted") or (
                            live.get("map_bank"), live.get("map_id"),
                            live.get("x_pos"), live.get("y_pos")
                        ) != (meta.get("bank"), meta.get("map"), meta.get("x"), meta.get("y")) or (
                            self._stage_at_current_location(live.get("map_bank", -1),
                                                            live.get("map_id", -1)) != stage
                        ):
                            self.env.em.set_state(original)
                            continue
                    return True
                except Exception:
                    self.env.em.set_state(original)
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

                # Consult durable history too, even if a fresh registry is empty.
                if os.path.exists(GLOBAL_PROGRESS_FILE):
                    with open(GLOBAL_PROGRESS_FILE) as f:
                        current = max(current, int(json.load(f).get("max_world_stage", 0)))
                if stage <= current:
                    self.shared_progress["max_world_stage"] = current
                    return False
                tmp = GLOBAL_PROGRESS_FILE + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"max_world_stage": stage,
                               "progress_schema": self.PROGRESS_SCHEMA}, f)
                os.replace(tmp, GLOBAL_PROGRESS_FILE)
                self.shared_progress["max_world_stage"] = stage
                return True
            finally:
                if self.shared_lock is not None:
                    self.shared_lock.release()

        except Exception:
            # Fail-closed: ein IPC-Problem darf keinen farmbaren Bonus erzeugen.
            return False

    @staticmethod
    def _warp_pair_key(from_bank, from_map, to_bank, to_map):
        return tuple(sorted((
            (int(from_bank), int(from_map)), (int(to_bank), int(to_map))
        )))

    @classmethod
    def _derive_warp_pairs(cls, transitions):
        """Grobe (map_a<->map_b)-Paare aus koordinatengenauen 8-Tupel-Warps.

        Robust gegen einen noch nicht verbundenen multiprocessing-Manager-Proxy
        (waehrend __init__ im SubprocVecEnv-Worker) - dann eben leer, die
        Paare werden beim ersten _refresh_shared_snapshots() nachgezogen.
        """
        pairs = set()
        try:
            snapshot = list(transitions) if transitions else []
        except Exception:
            return pairs
        for t in snapshot:
            try:
                if len(t) == 8:
                    pairs.add(cls._warp_pair_key(t[0], t[1], t[4], t[5]))
            except (TypeError, ValueError):
                continue
        return pairs

    def _refresh_shared_snapshots(self):
        old_edges = self.shared_edge_snapshot
        old_transitions = self.shared_transition_snapshot
        # V18: neue Transition-Endpunkte aus dem 5-Min-Nachladen (Watcher) bzw.
        # dem laufenden Fleet-Dict in die bekannten Warp-Paare mergen.
        try:
            self._known_warp_pairs |= self._derive_warp_pairs(self.shared_transitions)
        except Exception:
            pass

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
                and int(mon.get("species_id", 0) or 0) in self.STARTER_SPECIES
            ]
            if good:
                self._starter_species_cached = int(
                    good[0].get("species_id", 0) or 0
                )
                self._v10171_has_starter_cached = True
                return True
        except Exception:
            pass
        return False

    def _starter_species(self):
        self._v10171_party_has_starter()
        return int(getattr(self, "_starter_species_cached", 0) or 0)

    def _has_target_starter(self):
        return self._starter_species() == self.TARGET_STARTER_SPECIES

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
        if self.training_objective not in ("progress", "full", "scout"):
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
            self.last_progress_advance_step = self.route_steps
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
        except Exception:
            pass

        # Der Vault ist ein Hoechstwert. Fuer die Rollenverteilung zaehlt aber
        # die aktuelle Retention des gemeinsamen Learners. Bei echtem
        # Vergessen liefert der Trainer hier bewusst einen kleineren Wert und
        # das Curriculum springt automatisch zur Reparaturphase zurueck.
        try:
            with open(os.path.join(RUNTIME_DIR, "trainer_status.json"), "r") as f:
                status = json.load(f) or {}
            # Beim Start existiert kurz noch der Status des vorherigen
            # Trainerprozesses. Dessen alte Reparaturphase darf nicht die
            # allerersten Rollen des neuen Laufs bestimmen. Spawn-Worker sind
            # direkte Kinder des aktuellen Trainerprozesses.
            status_pid = int(status.get("trainer_pid", 0) or 0)
            effective = (
                status.get("effective_skill_scores") or {}
                if status_pid > 0 and status_pid == os.getppid()
                else {}
            )
            for k in default:
                if k in effective:
                    default[k] = int(effective[k])
        except Exception:
            pass
        return default

    def _scout_assigned_stage(self):
        """Five fixed ranks per approved stage; missing stages keep full runners.

        Fixed bands prevent late checkpoint discovery or replacement from moving
        existing scouts between stages during asynchronous episode resets.
        """
        stage_cps = self._valid_stage_checkpoints()
        slots_from_end = self.n_envs - 1 - (self.rank % self.n_envs)
        stage_index = slots_from_end // self.FRONTIER_SCOUT_SLOTS
        if stage_index >= len(self.SCOUT_STAGES):
            return None
        stage = self.SCOUT_STAGES[stage_index]
        return stage if stage in stage_cps else None

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

        # V16 CLEAN: alle Agenten spielen dieselbe komplette Aufgabe vom
        # Spielanfang. Keine Skills und keine gemischten Checkpoint-Starts im
        # selben PPO-Rollout.
        #
        # V17.3: eigene Rolle "scout" (statt weiter "full" oder das alte,
        # in vielen Legacy-Verzweigungen ueberladene "progress") fuer die
        # FRONTIER_SCOUT_SLOTS - im Web-Dashboard filterbar, aber bewusst
        # ein brandneuer String, der in KEINER bestehenden
        # training_objective-Pruefung sonst irgendwo im Code vorkommt.
        # Dadurch faellt "scout" ueberall dort, wo Reward-Code nach Rolle
        # unterscheidet, automatisch auf dasselbe Verhalten wie "full"
        # zurueck - ausser an der einen Stelle (episode_limit unten), wo es
        # explizit ergaenzt wurde, damit Scouts ihr eigenes, kuerzeres
        # SCOUT_EPISODE_STEPS-Limit bekommen statt versehentlich das lange
        # "full"-Limit oder gar 32768 zu erben.
        if getattr(self, "FULL_ONLY_MODE", False):
            assigned_stage = self._scout_assigned_stage()
            if assigned_stage is not None:
                return (
                    "scout",
                    f"Frontier Scout S{assigned_stage} {slot + 1:03d}"
                )
            return "full", f"Full Journey {slot + 1:03d}"

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
            # Bei 32 Envs machten die alten Mindestwerte 17 Retention-Slots
            # und nur 7 echte Starter-Lerner. Retention bleibt vorhanden,
            # aber die automatische Live-Pruefung schaltet bei Vergessen
            # ohnehin gezielt in die jeweilige Reparaturphase zurueck.
            r1 = _pct(0.03, 1)
            r2 = r1 + _pct(0.06, 2)
            r3 = r2 + _pct(0.10, 3)
            # V15.3: DEUTLICH mehr echte End-to-End-Laeufe. Der Champion wird
            # NUR an Full-from-Beginning gemessen und braucht >=8 abgeschlossene
            # Full-Episoden pro Eval-Fenster. Mit nur 1 full-Agenten kam das
            # praktisch nie zustande -> Champion 23 Mio Steps eingefroren, und
            # das Verketten der Abschnitte wurde fast nie geuebt (nur der
            # Watcher sah das im echten Durchlauf). Jetzt ~6 full + 2 progress.
            full_block = _pct(0.19, 6)
            progress_block = _pct(0.06, 2)
            starter_end = n - full_block - progress_block
            if slot < r1: return "intro", f"Intro Retention {slot + 1:02d}"
            if slot < r2: return "stairs", f"Stairs Retention {slot - r1 + 1:02d}"
            if slot < r3: return "exit", f"Exit Retention {slot - r2 + 1:02d}"
            if slot < starter_end:
                return "starter", f"Starter Bootcamp {slot - r3 + 1:03d}"
            if slot < starter_end + progress_block:
                return "progress", f"World Probe {slot - starter_end + 1:02d}"
            return "full", f"Full Journey {slot - (starter_end + progress_block) + 1:02d}"

        # --- Phase 5: FREIE WELT ---
        # Vor Vertania trainieren keine Battle-/Level-Spezialisten. Der
        # Loewenanteil schiebt die Weltfront; vier Slots pruefen Full-Runs.
        try:
            global_stage = int(self.shared_progress.get("max_world_stage", 0))
        except Exception:
            global_stage = 0
        b_intro = 1
        b_stairs = b_intro + 1
        b_exit = b_stairs + 1
        b_starter = b_exit + 3
        if global_stage < 4:
            b_battle = b_starter
            b_level = b_battle
            b_badge = b_level
        elif global_stage < 5:
            # Paket ist abgegeben, aber Route 2/Wald noch nicht erreicht:
            # nur ein kleiner Kampfsockel. Zehn von 32 Slots grindeten vorher
            # Level, waehrend die eigentliche Weltfront stehen blieb.
            b_battle = b_starter + max(2, int(round(n * 0.0625)))
            b_level = b_battle + max(2, int(round(n * 0.0625)))
            b_badge = b_level
        else:
            b_battle = b_starter + max(3, int(round(n * 0.15625)))
            b_level = b_battle + max(2, int(round(n * 0.09375)))
            b_badge = b_level + max(2, int(round(n * 0.125)))
        # Ab der Paket-Rueckgabe muessen genug komplette Starts den
        # geschuetzten Champion pruefen; Curriculum-Fortschritt allein darf
        # ihn weiterhin nicht ersetzen.
        full_count = max(2, int(round(n * 0.125)))
        if global_stage >= 4:
            full_count = max(8, int(round(n * 0.25)))
        b_full = n - full_count
        if slot < b_intro:   return "intro", f"Intro Vault {slot + 1:02d}"
        if slot < b_stairs:  return "stairs", f"Stairs Vault {slot - b_intro + 1:02d}"
        if slot < b_exit:    return "exit", f"Exit Vault {slot - b_stairs + 1:02d}"
        if slot < b_starter: return "starter", f"Starter Vault {slot - b_exit + 1:02d}"
        if slot < b_battle:  return "battle", f"Battle {slot - b_starter + 1:02d}"
        if slot < b_level:   return "level", f"Level {slot - b_battle + 1:02d}"
        if slot < b_badge:   return "badge", f"Badge {slot - b_level + 1:02d}"
        if slot < b_full:    return "progress", f"World Push {slot - b_badge + 1:03d}"
        return "full", f"Full Journey {slot - b_full + 1:02d}"

    def _choose_episode_start(self):
        self.saved_milestones = self._discover_saved_milestones()
        self.full_chain_ready = self._champion_full_starter_ready()
        role, _ = self._agent_role()
        self.training_objective = role
        saved = set(self.saved_milestones)

        # V15.3c: im All-Full-Regime startet JEDER full-Agent am Spielanfang.
        #
        # V17.3: bisher startete unter FULL_ONLY_MODE ausnahmslos jeder
        # Agent "beginning" - die ganze Flotte spielte dadurch JEDE Episode
        # wieder komplett ab Pallet Town los, egal wie tief die Front schon
        # war (_save_stage_checkpoint() sammelte brav Checkpoints, aber
        # niemand lud sie je wieder). Neue, tiefere Maps (Route 2/Wald/
        # Marmoria) bekamen dadurch praktisch nie gezielte Uebung. Die
        # "scout"-Rolle (siehe _agent_role()) resumt jetzt vom tiefsten
        # validierten Checkpoint statt neu ab Spielanfang.
        #
        # V17.4: jede Stage hat jetzt ihre EIGENEN Scouts (siehe
        # _scout_assigned_stage()) statt dass alle immer zur tiefsten Front
        # wandern - hier also den PASSENDEN Checkpoint fuer die diesem Rank
        # zugewiesene Stage laden, nicht pauschal den tiefsten.
        if getattr(self, "FULL_ONLY_MODE", False):
            if role == "scout":
                assigned_stage = self._scout_assigned_stage()
                if assigned_stage is not None:
                    stage_cps = self._valid_stage_checkpoints()
                    if assigned_stage in stage_cps:
                        return stage_cps[assigned_stage]
                self.training_objective = "full"
                return "beginning"
            return "beginning"

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
            # Ein tiefer Story-Checkpoint kann indoor liegen (stage_5 ist
            # Eichs Labor). Kampf- und Level-Spezialisten starten deshalb nur
            # an einem gesunden Gras-Checkpoint oder am sicheren Hausausgang.
            if "squirtle_battle_ready" in saved: return "squirtle_battle_ready"
            if "squirtle_outdoor" in saved: return "squirtle_outdoor"
            if "squirtle" in saved: return "squirtle"
            if "battle_ready" in saved: return "battle_ready"
            if "starter_outdoor" in saved: return "starter_outdoor"
            if "starter" in saved: return "starter"
            if "left_house" in saved: return "left_house"
            return self._best_progress_milestone()
        if role in ("progress", "badge"):
            # Tiefe Legacy-States bleiben als Bruecke erhalten. Sobald die
            # Schiggy-Kette aufholt, ersetzt sie Stage fuer Stage atomar.
            if "starter" in saved: return self._best_progress_milestone()
            if "squirtle" in saved: return self._best_progress_milestone()
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
        try:
            global_stage = int(self.shared_progress.get("max_world_stage", 0))
        except Exception:
            global_stage = 0
        full_count = max(2, int(round(n * 0.125)))
        if global_stage >= 4:
            full_count = max(8, int(round(n * 0.25)))
        b_full = n - full_count
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
        try:
            global_stage = int(self.shared_progress.get("max_world_stage", 0))
        except Exception:
            global_stage = 0
        full_count = max(2, int(round(n * 0.125)))
        if global_stage >= 4:
            full_count = max(4, int(round(n * 0.1875)))
        b_full = n - full_count
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

        self.last_badges = badges
        self.has_starter = self._v10171_party_has_starter()
        self.has_target_starter = self._has_target_starter()
        valid_party = [
            mon for mon in (self.player_party_cache or [])
            if int(mon.get("species_id", 0)) > 0
        ]
        party_levels = [int(mon.get("level", 0)) for mon in valid_party]
        self.last_level = max([p_lvl] + party_levels)
        self.last_party_size = len(valid_party)
        self.last_party_total_level = sum(party_levels)
        self.last_party_total_hp = sum(
            int(mon.get("cur_hp", 0)) for mon in valid_party
        )
        self.last_party_identity = self._party_identity(valid_party)
        self.last_party_total_experience = sum(
            int(mon.get("experience", 0)) for mon in valid_party
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
        self._update_story_state_from_loc(loc)
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
                bank == self.OVERWORLD_BANK
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

    # States, deren Name mit einem dieser Praefixe beginnt, liegen im
    # Curriculum NACH dem Hausausgang (mit Starter). "stage_" gehoert
    # zwingend dazu: stage_2..3 liegen draussen, stage_4/5 im Vertania-
    # Markt bzw. Eichs Labor (Bank 5/4) - dort greift der Overworld-Zweig
    # in _set_baseline_from_info nicht, also MUSS der Resume die Haus-Flags
    # hier explizit setzen, sonst kappt der early-house-Failsafe den Run.
    POST_HOUSE_STATE_PREFIXES = (
        "progress_", "maps_", "outdoor_", "stage_", "level_", "badge_",
        "battle_",
    )

    def _apply_curriculum_resume_flags(self):
        """Setzt die Story-Flags, die ein Curriculum-Resume bereits erfuellt hat.
        Der RAM-Ort allein sagt nicht sicher, welche Lern-Meilensteine schon
        erreicht wurden - gezielte States stellen ihre Stufe explizit her."""
        start = self.episode_start

        if start == "intro_complete":
            self.episode_milestone_steps["intro_complete"] = 0
            return

        if start == "stairs_down":
            self.stairs_down_rewarded = True
            self.episode_milestone_steps["intro_complete"] = 0
            self.episode_milestone_steps["stairs_down"] = 0
            return

        post_house = (
            start in ("left_house", "starter", "starter_outdoor")
            or start.startswith(self.POST_HOUSE_STATE_PREFIXES)
        )
        if not post_house:
            return

        # Spaetere Curriculum-States liegen nach dem Haus.
        self.stairs_down_rewarded = True
        self.left_house_rewarded = True
        self.left_house_confirmed = True
        self.outdoor_confirm_reads = self.OUTDOOR_CONFIRM_READS
        self.episode_milestone_steps["intro_complete"] = 0
        self.episode_milestone_steps["stairs_down"] = 0

        # starter_outdoor und jeder stage_N (N>=2) sind laengst mit Starter
        # aus dem Labor -> "raus aus Eichs Labor" nicht erneut belohnen.
        if start == "starter_outdoor" or start.startswith("stage_"):
            self.starter_outdoor_rewarded = True
            self.episode_milestone_steps["starter_outdoor"] = 0

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
        self.battle_state = BattleState(read_enemy_party(self.env))
        self.main_battle_reader = MainBattleReader()
        self.wipe_active = False
        # V19 POST_WIPE_RECOVERY_MODE
        self.post_wipe_recovery = False
        self.pre_wipe_best_stage = 0
        self.pre_wipe_best_center_stage = 0
        self.pre_wipe_badges = 0
        self._image_frames = []

        self.total_steps = 0
        self.route_steps = 0
        self.battle_steps = 0
        self.current_battle_steps = 0
        self.viridian_mart_scene = 0
        self.viridian_old_man_scene = 0
        self.pallet_oaks_lab_scene = 0
        # The user master is after parcel delivery. Preserve story telemetry;
        # these flags do not contribute to geographic progression.
        self.parcel_obtained_confirmed = True
        self.parcel_delivered_confirmed = True
        self.parcel_obtained_confirm_reads = 0
        self.parcel_delivered_confirm_reads = 0
        self.episode_battles_started = 0
        self.episode_battles_completed = 0
        self.enemy_party_cache = []
        self.player_party_cache = []
        self.last_party_total_hp = 0
        self.last_party_total_experience = 0
        self.faints_in_current_battle = 0
        self.last_party_size = 0
        self.pokemon_center_healed_this_episode = False
        # V18: {(bank,map)} bereits betretener Center + tiefste Stufe, in deren
        # Center diese Episode schon geheilt wurde (Respawn-Punkt-Fortschritt).
        self.pokecenter_entered_this_episode = set()
        self.pokemart_entered_this_episode = set()
        self.best_pokecenter_heal_stage = 0
        # V19 BROCK RUSH: Pewter/Brock-Meilensteine - je einmal pro Episode.
        self.episode_pewter_reached = False
        self.episode_pewter_with_pikachu_rewarded = False
        self.episode_pewter_gym_entered = False
        self.episode_brock_battle_started = False
        self.episode_pewter_gym_trainer_beaten = False
        self.last_party_total_level = 0
        self.indoor_steps_without_transition = 0
        self.battle_activity_open = False
        self.enemy_hp_min = {}
        self.enemy_fainted_rewarded = set()
        # V17.2: fehlte hier komplett. total_steps setzt jede Episode auf 0
        # zurueck, dieser Wert aber nicht - nach der ersten Episode wird
        # total_steps - _last_enemy_seen_step sofort stark negativ, was die
        # "<=96"-Pruefung immer erfuellt. Folge: in_battle haengt am Anfang
        # JEDER Episode ausser der ersten faelschlich auf 1, bis total_steps
        # den alten (episodenfremden) Wert wieder eingeholt hat - oft
        # tausende Schritte. Erklaert sowohl die "haengt in Kampf fest ohne
        # sichtbaren Kampf"-Agenten als auch die ausbleibende Fluchtstrafe
        # (battle_just_ended kann nie feuern, wenn in_battle nie auf 0 faellt).
        self._last_enemy_seen_step = -999
        self.episode_enemy_damage_hp = 0
        self.episode_enemy_damage_reward = 0.0
        self.episode_enemy_faints = 0
        # V18: besiegte Wild-Pokemon auf WILD_TRAINING_MAPS dieser Episode -
        # ab WILD_BATTLE_DECAY_AFTER greift WILD_BATTLE_DECAY_FACTOR.
        self.episode_wild_faints = 0
        self.seen_coords = set()
        self._episode_tiles_by_map = {}
        self.visited_maps = set()
        self._saved_outdoor_depth = 0
        # V10.23: pro Episode erneut belohnen, wenn eine bekannte Map oder ein
        # bekannter Warp korrekt wiederholt wird. Persistente Weltkenntnis
        # bleibt davon getrennt und wird nicht geloescht.
        self.learning_seen_maps = set()
        self.learning_seen_edges = set()
        self.learning_seen_transitions = set()
        self.recent_path = []
        # V17.2: Dashboard-Klick auf einen Agenten zeigte bisher nur die
        # reward_events des einen Steps, in dem die inst_XX.json zufaellig
        # geschrieben wurde (alle 80 Steps) - fast immer leer. Stattdessen
        # rollierendes Log der letzten tatsaechlichen Reward-Ereignisse.
        self.recent_reward_events = []
        # V17.2: Route-Schritt, bis zu dem Map-/Kanten-Boni nach einem
        # Party-Wipe unterdrueckt werden (siehe POST_WIPE_REWARD_COOLDOWN_
        # STEPS). -1 = kein aktiver Wipe-Cooldown.
        self._post_wipe_reward_cooldown_until = -1
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
        self.has_target_starter = False
        self.starter_outdoor_rewarded = False
        self.episode_caught_species = set()
        self.episode_pikachu_forest_caught = False
        # V17: wurde bisher nur beim Uebergang "kein Starter -> Starter"
        # gesetzt. Der Startpunkt hat den Starter aber schon ab Step 0, dieser
        # Uebergang passiert also nie mehr - das "muss innerhalb 4000 Steps aus
        # dem Labor"-Sicherheitsnetz waere sonst permanent tot. 0 statt None:
        # die 4000 Steps zaehlen jetzt ab Episodenstart.
        self.starter_obtained_step = 0
        self._v10171_has_starter_cached = False
        self._starter_species_cached = 0
        self._v10171_party_check_step = -999999

        self.initial_indoor_map = None
        # V17: siehe __init__ - Startpunkt liegt bereits nach Treppe/Hausausgang.
        self.stairs_down_rewarded = True
        self.left_house_rewarded = True
        # V17: separat von left_house_rewarded, aber gleiches Problem - nur
        # Telemetrie (v2_full_left_house%), keine Reward-Wirkung, aber sonst
        # dauerhaft fälschlich 0% obwohl der Hausausgang laengst erledigt ist.
        self.left_house_confirmed = True
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
        # V17: siehe __init__ - Startpunkt liegt bereits nach dem Intro.
        self.intro_complete_rewarded = True

        self.previous_valid_bank = None
        self.previous_valid_map = None
        self.previous_valid_x = None
        self.previous_valid_y = None
        self.pending_exit_story_transition = None

        self.last_pos = None
        self.stuck_counter = 0
        self.last_progress_signature = None
        self.interaction_anchor = None
        self.interaction_count = 0
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
                self.training_objective = "full"

        baseline_info = self._read_info_with_idle_frame()

        verified_loc = None
        try:
            verified_loc = read_player_location(self.env, allow_scan=True)
        except Exception:
            verified_loc = None

        self._set_baseline_from_info(
            baseline_info,
            loc_override=verified_loc
        )
        self.battle_state = BattleState(read_enemy_party(self.env))
        self.main_battle_reader = MainBattleReader()
        self.episode_best_stage = int(self._world_stage())
        self.episode_start_stage = self.episode_best_stage
        self.stage_arrival_steps = {str(self.episode_best_stage): 0}
        self.local_loop_guard = LocalLoopGuard()
        self._stage_hold_map = None
        self._stage_hold_steps = 0

        if self.episode_start != "beginning":
            self.intro_complete_rewarded = True
            self._apply_curriculum_resume_flags()

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
        if not self.JOURNEY_ROUTE_ENABLED:
            return set()
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
        if not self.JOURNEY_ROUTE_ENABLED:
            return
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

        for _ in range(self.ACTION_HOLD_FRAMES):
            step_res = self.env.step(raw_act)

        for _ in range(self.ACTION_RELEASE_FRAMES):
            step_res = self.env.step(self.btn_none)

        self.total_steps += 1
        if self.total_steps == 1 or not hasattr(self, "v9_last_pos"):
            self.v9_last_pos = None
            self.v9_same_pos_steps = 0
            self.v9_episode_tiles = set()
            self._north_corridor_best = {}  # (bank,map) -> noerdlichste Y
            self._south_corridor_best = {}  # Paket-Rueckweg: suedlichste Y

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
        self._update_story_state_from_loc(loc)
        bank = int(loc["map_bank"]) if loc["valid"] else 0
        map_id = int(loc["map_id"]) if loc["valid"] else 0
        x = int(loc["x_pos"]) if loc["valid"] else 0
        y = int(loc["y_pos"]) if loc["valid"] else 0
        # Battle types can be zero for wild encounters and remain in RAM
        # after exit. Track fresh enemy data and actual overworld movement.
        if not hasattr(self, "battle_state"):
            self.battle_state = BattleState()
        try:
            battle_party = read_enemy_party(self.env)
        except Exception:
            battle_party = self.enemy_party_cache
        position = (bank, map_id, x, y) if loc.get("trusted") else None
        if not hasattr(self, "main_battle_reader"):
            self.main_battle_reader = MainBattleReader()
        live_battle = self.main_battle_reader.read(
            self.env.get_ram(), self.ACTION_HOLD_FRAMES + self.ACTION_RELEASE_FRAMES
        )
        in_battle = self.battle_state.update(
            battle_party, position,
            flags=read_battle_type_flags(self.env),
            live=live_battle,
            signal=int(info.get("in_battle", 0) or 0),
        )
        info["in_battle"] = in_battle
        info["battle_detection"] = self.battle_state.reason
        info["battle_type_flags"] = self.battle_state.raw_flags
        if in_battle:
            self.battle_steps += 1
            self.current_battle_steps += 1
        else:
            self.route_steps += 1
            self.current_battle_steps = 0
        reward = 0.0
        reward_events = []
        truncated = False
        previous_party = self.player_party_cache
        try:
            fresh_party = read_player_party(self.env)
            if fresh_party:
                self.player_party_cache = fresh_party
        except Exception:
            pass
        if not hasattr(self, "wipe_active"):
            self.wipe_active = False
        previous_battle_state = int(self.last_in_battle)
        battle_just_ended = previous_battle_state == 1 and in_battle == 0
        battle_ended_without_faint = False
        if previous_battle_state == 0 and in_battle == 1:
            self.faints_in_current_battle = 0
            self.battle_rewarded_win = False
            self.battle_caught = False
            self.run_stats["battles_started"] += 1
            self.episode_battles_started += 1
            self.battle_activity_open = True
            self.enemy_party_cache = []
            self.enemy_hp_min = {}
            self.enemy_fainted_rewarded = set()
            self._save_run_stats()
            # V19 BROCK RUSH: erster Trainerkampf in Marmoria (Stadt-Map oder
            # ein Bank-6-Innenraum = Arena) pro Episode = Brock-/Arenakampf
            # gestartet. Anti-Farm: Episode-Flag, kein Re-Trigger durch
            # Kampf-Neustart. Kompass-/Trainer-ID-unabhaengig.
            _in_pewter = (
                int(bank) == 6
                or self._current_world_stage(bank, map_id) == 6
            )
            if (
                _in_pewter
                and self._is_trainer_battle()
                and not self.episode_brock_battle_started
            ):
                self.episode_brock_battle_started = True
                if self.BROCK_BATTLE_START_REWARD:
                    reward += self.BROCK_BATTLE_START_REWARD
                    reward_events.append(
                        f"brock_battle_start:+{self.BROCK_BATTLE_START_REWARD:.0f}"
                    )
        elif battle_just_ended:
            alive_pokemon = sum(
                1 for mon in self.player_party_cache
                if int(mon.get("cur_hp", 0)) > 0
            )
            # V17.2: Party-Wipe HIER erkennen, nicht nur ueber das periodische
            # current_total_hp==0 unten (Party wird nur alle PARTY_READ_EVERY=8
            # Schritte neu gelesen). Nach einem Wipe teleportiert das Spiel
            # automatisch zu einem Pokecenter - das kann als "neue Map" +25/
            # +500 durchrutschen, wenn die Strafe hier verpasst wird, weil die
            # exakte HP==0-Lesung zwischen zwei 8-Schritt-Samples faellt. Der
            # Kampf-Ende-Uebergang wird dagegen JEDEN Schritt zuverlaessig
            # erkannt, deshalb hier pruefen statt nur periodisch zu hoffen.
            if alive_pokemon == 0 and self.player_party_cache:
                reward += self._record_party_wipe(reward_events, info)
            # Der letzte Angriff kann Gegner-KP und Battle-Flag im selben
            # Emulator-Step auf 0 setzen. Die alte Reihenfolge loeschte hier
            # enemy_hp_min und verlor dadurch genau diesen K.O.
            try:
                ending_enemy_party = read_enemy_party(self.env)
            except Exception:
                ending_enemy_party = []
            ending_faint_visible = False
            for mon in ending_enemy_party:
                mon_key = (
                    int(mon.get("slot", -1)),
                    int(mon.get("species_id", 0)),
                    int(mon.get("personality", 0)),
                )
                if (
                    int(mon.get("cur_hp", -1)) == 0
                    and int(self.enemy_hp_min.get(mon_key, 0)) > 0
                    and mon_key not in self.enemy_fainted_rewarded
                ):
                    ending_faint_visible = True
                    break
            battle_ended_without_faint = bool(
                alive_pokemon > 0
                and not self.wipe_active
                and not getattr(self, "battle_rewarded_win", False)
                and not getattr(self, "battle_caught", False)
                and len(self.player_party_cache) <= len(previous_party)
                and sum(int(m.get("experience", 0)) for m in self.player_party_cache)
                    <= sum(int(m.get("experience", 0)) for m in previous_party)
                and self.faints_in_current_battle == 0
                and not ending_faint_visible
            )
            self.run_stats["battles_completed"] += 1
            self.episode_battles_completed += 1
            self.battle_activity_open = False
            # Bis zum Damage-Scan unten behalten; er verbucht einen im
            # Abschlussframe sichtbaren K.O. noch sauber. Beim naechsten
            # Kampfstart werden diese Felder ohnehin zurueckgesetzt.
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
            self.total_steps == 1
            or self.total_steps % self.SHARED_SNAPSHOT_EVERY == 0
        ):
            self._refresh_shared_snapshots()

        self.last_gameplay_ready = gameplay_ready
        self.last_in_battle = in_battle

        if not gameplay_ready:
            # V16: normale Aktionen sind auch im Intro neutral. Fortschritt
            # entsteht nur durch gedeckelte neue Screens und Meilensteine.
            reward += self.INTRO_STEP_COST
        elif self.FULL_ONLY_MODE:
            # V16: keine Bewegungs- oder Zeitpunkte im Full-Brain-Modus.
            reward += self.GAMEPLAY_STEP_COST
        else:
            _speed_graded = (
                self.training_objective in self.SPECIALIST_SPEED_ROLES
                or (self.training_objective == "full" and not self.has_starter)
            )
            reward += (
                self.SPECIALIST_STEP_COST
                if _speed_graded
                else self.GAMEPLAY_STEP_COST
            )
        milestone_saved = None
        objective_done = False

        if battle_ended_without_faint:
            reward += self.FLED_BATTLE_PENALTY
            reward_events.append(
                f"fled_battle:{self.FLED_BATTLE_PENALTY:.1f}"
            )

        # V18: Pokemon-Center betreten - erste Mal pro Lauf, pro Center.
        # Unabhaengig von visited_maps (auch das Center des Startpunkts zaehlt)
        # und von einer Heilung. Der automatische Wipe-Teleport ins Center zahlt
        # NICHT (wipe_active / Post-Wipe-Cooldown), der Key wird dann aber schon
        # gesetzt, damit es nach Ablauf des Cooldowns nicht doch noch ausloest.
        _pc_key = (int(bank), int(map_id))
        if (
            _pc_key in self.POKECENTER_MAPS
            and _pc_key not in self.pokecenter_entered_this_episode
            and in_battle == 0
        ):
            self.pokecenter_entered_this_episode.add(_pc_key)
            if (
                not self.wipe_active
                and self.route_steps >= self._post_wipe_reward_cooldown_until
                and self.POKECENTER_ENTER_REWARD
            ):
                reward += self.POKECENTER_ENTER_REWARD
                reward_events.append(
                    f"pokecenter_enter:{_pc_key[0]}_{_pc_key[1]}:"
                    f"+{self.POKECENTER_ENTER_REWARD:.0f}"
                )

        # V18: Poke-Markt betreten - erstmals pro Lauf +POKEMART_ENTER_REWARD,
        # allererster Fund fleet-weit +POKEMART_FIRST_GLOBAL_REWARD. Soll dem
        # Hirn zeigen, dass es den Laden ueberhaupt gibt (-> Pokebaelle kaufen).
        _mart_key = (int(bank), int(map_id))
        if (
            _mart_key in self.POKEMART_MAPS
            and _mart_key not in self.pokemart_entered_this_episode
            and in_battle == 0
        ):
            self.pokemart_entered_this_episode.add(_mart_key)
            if (
                not self.wipe_active
                and self.route_steps >= self._post_wipe_reward_cooldown_until
            ):
                if self.POKEMART_ENTER_REWARD:
                    reward += self.POKEMART_ENTER_REWARD
                    reward_events.append(
                        f"pokemart_enter:{_mart_key[0]}_{_mart_key[1]}:"
                        f"+{self.POKEMART_ENTER_REWARD:.0f}"
                    )
                if self.POKEMART_FIRST_GLOBAL_REWARD and claim_event(
                    EXPLORATION_MEMORY_DIR,
                    f"mart_{_mart_key[0]}_{_mart_key[1]}",
                    self.shared_species, self.shared_lock
                ):
                    reward += self.POKEMART_FIRST_GLOBAL_REWARD
                    reward_events.append(
                        f"pokemart_first_global:{_mart_key[0]}_{_mart_key[1]}:"
                        f"+{self.POKEMART_FIRST_GLOBAL_REWARD:.0f}"
                    )

        # V19 BROCK RUSH: Marmoria-Arena erstmals pro Episode betreten.
        # Anti-Farm: Episode-Flag. Inert bis PEWTER_GYM_MAPS eine bestaetigte
        # Innenraum-Map enthaelt.
        if (
            (int(bank), int(map_id)) in self.PEWTER_GYM_MAPS
            and not self.episode_pewter_gym_entered
            and in_battle == 0
        ):
            self.episode_pewter_gym_entered = True
            if (
                not self.wipe_active
                and self.route_steps >= self._post_wipe_reward_cooldown_until
                and self.PEWTER_GYM_ENTER_REWARD
            ):
                reward += self.PEWTER_GYM_ENTER_REWARD
                reward_events.append(
                    f"pewter_gym_enter:+{self.PEWTER_GYM_ENTER_REWARD:.0f}"
                )

        # V10.27A: Die gesamte Party statt nur Pokemon-Slot 1 bewerten.
        if self.player_party_cache:
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
            current_total_experience = sum(
                int(mon.get("experience", 0)) for mon in valid_party
            )
            max_total_hp = sum(
                int(mon.get("max_hp", 0)) for mon in valid_party
            )

            if (
                self.last_party_size > 0
                and current_party_size > self.last_party_size
                and current_party_size <= 6
                and (in_battle or battle_just_ended)
            ):
                # V17.2: Erstfang einer Spezies fleet-weit belohnen, jeden
                # weiteren Fang derselben Art bestrafen - lernt Artenvielfalt
                # statt denselben haeufigen Wildpokemon (Taubsi/Raupy/...)
                # immer wieder zu fangen. Neu gefangene Mons haengen sich ans
                # Party-Ende an, valid_party[-1] ist daher der Neuzugang.
                new_species = (
                    int(valid_party[-1].get("species_id", 0))
                    if valid_party else 0
                )
                if new_species > 0:
                    self.battle_caught = True
                    if new_species not in self.episode_caught_species:
                        self.episode_caught_species.add(new_species)
                        _caught_level = min(
                            int(valid_party[-1].get("level", 0) or 0),
                            self.SPECIES_CAUGHT_LEVEL_BONUS_CAP,
                        )
                        _catch_reward = (
                            self.SPECIES_CAUGHT_FIRST_REWARD
                            + max(_caught_level, 0) * self.SPECIES_CAUGHT_LEVEL_BONUS
                        )
                        # V19: waehrend Post-Wipe-Recovery kein generischer
                        # Fang-Reward - der Rueckweg zur Front soll die klar
                        # beste Wahl sein. Der Pikachu-Wald-Bonus unten ist ein
                        # eigener if und bleibt (wichtiger einmaliger Storyfang).
                        if getattr(self, "post_wipe_recovery", False):
                            _catch_reward = 0.0
                        reward += _catch_reward
                        reward_events.append(
                            f"species_caught_first:{new_species}:L{_caught_level}:"
                            f"+{_catch_reward:.0f}"
                        )
                    elif self.SPECIES_CAUGHT_DUPLICATE_PENALTY:
                        reward += self.SPECIES_CAUGHT_DUPLICATE_PENALTY
                        reward_events.append(
                            f"species_caught_dup:{new_species}:"
                            f"{self.SPECIES_CAUGHT_DUPLICATE_PENALTY:.0f}"
                        )
                    # V17.4: Pikachu ist im Vertania-Wald selten und kein
                    # Fortschrittsweg - eigener, viel groesserer Bonus obendrauf,
                    # unabhaengig vom generischen Fang-Reward oben. Pro Run
                    # (nicht fleet-lifetime): ein zweiter Pikachu im selben Run
                    # zahlt nichts mehr, aber der naechste Run wieder von vorn.
                    if (
                        new_species == self.PIKACHU_SPECIES_ID
                        and (bank, map_id) == self.PIKACHU_FOREST_MAP
                        and not self.episode_pikachu_forest_caught
                    ):
                        self.episode_pikachu_forest_caught = True
                        reward += self.PIKACHU_FOREST_CAUGHT_REWARD
                        reward_events.append(
                            "pikachu_forest_first:"
                            f"+{self.PIKACHU_FOREST_CAUGHT_REWARD:.0f}"
                        )
                else:
                    reward_events.append("caught_pokemon:+0")

            if (
                self.last_party_total_level > 0
                and current_total_level > self.last_party_total_level
            ):
                level_gain = current_total_level - self.last_party_total_level
                # V18: Level-Up klingt auf einer Wild-Map nach 6 Kaempfen
                # ebenfalls auf WILD_BATTLE_DECAY_FACTOR ab - stures Grinden an
                # derselben Stelle soll nicht ueber Levels doch noch lohnen.
                _lvl_scale = self._battle_reward_scale(bank, map_id)
                _lvl_reward = level_gain * self.LEVEL_GAIN_REWARD * _lvl_scale
                reward += _lvl_reward
                _lvl_tag = (":2x" if _lvl_scale > 1.0
                            else ":decayed" if _lvl_scale < 1.0 else "")
                reward_events.append(f"team_level_up{_lvl_tag}:+{_lvl_reward:.1f}")
                self.reward_event_counts["level_up"] += level_gain
                self.last_progress_advance_step = self.route_steps
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

            # EP sind der robuste Sieg-Nachweis: FireRed kann das einzelne
            # Gegner-0-KP-Frame zwischen zwei Agent-Aktionen ueberspringen,
            # der dauerhafte EP-Anstieg bleibt jedoch im Party-Struct stehen.
            if (
                self.last_party_total_experience > 0
                and current_party_size == self.last_party_size
                and current_total_experience > self.last_party_total_experience
                and not getattr(self, "battle_rewarded_win", False)
                and self._party_identity(valid_party) == getattr(self, "last_party_identity", ())
                and (in_battle or battle_just_ended)
            ):
                experience_gain = (
                    current_total_experience - self.last_party_total_experience
                )
                self.battle_rewarded_win = True
                # V18: BATTLE_WIN_REWARD ist 0 - Buchfuehrung laeuft weiter
                # (verhindert Doppel-Zaehlen), aber kein Reward/Event fuer den
                # reinen EP-Anstieg mehr.
                experience_reward = (
                    self.BATTLE_WIN_REWARD * self._battle_reward_scale(bank, map_id)
                )
                if experience_reward:
                    reward += experience_reward
                    reward_events.append(
                        f"experience_gain:{experience_gain}:+{experience_reward:.1f}"
                    )
                self.run_stats["experience_wins"] = int(
                    self.run_stats.get("experience_wins", 0)
                ) + 1
                if experience_reward:
                    self.last_progress_advance_step = self.route_steps
                if self.training_objective == "battle":
                    reward += self.SPECIALIST_SUCCESS_BONUS
                    reward_events.append(
                        f"objective_battle_xp:+{self.SPECIALIST_SUCCESS_BONUS:.0f}"
                    )
                    self.objective_success = True
                    objective_done = True
                self._save_run_stats()

            if current_total_hp == 0 and max_total_hp > 0:
                reward += self._record_party_wipe(reward_events, info)
            elif self.last_party_total_hp > 0:
                hp_diff = current_total_hp - self.last_party_total_hp
                if hp_diff < 0:
                    hp_reward = abs(hp_diff) * -0.1
                    reward += hp_reward
                    reward_events.append(f"took_damage:{hp_reward:.1f}")
                elif hp_diff > 0:
                    # V18: Vollheilung ausserhalb des Kampfes in einem bekannten
                    # Center-Erdgeschoss - das Signal, das nur die Schwester
                    # liefert (Items/Attacken heilen fast nie exakt bis max_hp).
                    _pc_heal_stage = self.POKECENTER_HEAL_MAPS.get(
                        (int(bank), int(map_id)), 0
                    )
                    _healed_full = (
                        _pc_heal_stage > 0
                        and not self.wipe_active
                        and in_battle == 0
                        and max_total_hp > 0
                        and current_total_hp == max_total_hp
                        and self.last_party_total_hp < max_total_hp
                        and self.route_steps >= self._post_wipe_reward_cooldown_until
                    )
                    if _healed_full and not self.pokemon_center_healed_this_episode:
                        # V17.3: erster Komplett-Heal dieser Episode + einmaliger
                        # Flotten-Bonus fuers allererste Mal ueberhaupt.
                        self.pokemon_center_healed_this_episode = True
                        if self.POKEMON_CENTER_FIRST_HEAL_REWARD:
                            reward += self.POKEMON_CENTER_FIRST_HEAL_REWARD
                        reward_events.append(
                            "pokemon_center_first_heal:"
                            f"+{self.POKEMON_CENTER_FIRST_HEAL_REWARD:.1f}"
                        )
                        if claim_event(
                            EXPLORATION_MEMORY_DIR, "pokemon_center_ever",
                            self.shared_species, self.shared_lock
                        ):
                            reward += self.POKEMON_CENTER_VISIT_GLOBAL_REWARD
                            reward_events.append(
                                "pokemon_center_visit_global:"
                                f"+{self.POKEMON_CENTER_VISIT_GLOBAL_REWARD:.0f}"
                            )
                    if _healed_full and _pc_heal_stage > self.best_pokecenter_heal_stage:
                        # V18: Heilung in einem Center weiter vorne als jedes
                        # bisher in diesem Lauf genutzte -> der Wiedereinstiegs-
                        # punkt nach einem Party-Wipe ist dauerhaft vorgerueckt.
                        # Bewusst NICHT an pokemon_center_healed_this_episode
                        # gekoppelt: ein zweites, tieferes Center im selben Lauf
                        # (Vertania -> Marmoria) zaehlt ebenfalls.
                        self.best_pokecenter_heal_stage = _pc_heal_stage
                        self.last_progress_advance_step = self.route_steps
                        if self.POKECENTER_ADVANCE_HEAL_REWARD:
                            reward += self.POKECENTER_ADVANCE_HEAL_REWARD
                            reward_events.append(
                                f"pokecenter_advance_heal:{_pc_heal_stage}:"
                                f"+{self.POKECENTER_ADVANCE_HEAL_REWARD:.0f}"
                            )
                        if claim_event(
                            EXPLORATION_MEMORY_DIR,
                            f"pc_heal_{int(bank)}_{int(map_id)}",
                            self.shared_species, self.shared_lock
                        ):
                            reward += self.POKECENTER_FIRST_HEAL_GLOBAL_REWARD
                            reward_events.append(
                                f"pokecenter_first_heal_global:{_pc_heal_stage}:"
                                f"+{self.POKECENTER_FIRST_HEAL_GLOBAL_REWARD:.0f}"
                            )
                    if not _healed_full:
                        # Jede andere Heilung (Items/Kampfattacken/Teilheilung):
                        # proportional zur wiederhergestellten HP, symmetrisch
                        # zur Schadensstrafe (-0.1/HP) statt pauschal neutral.
                        hp_reward = 0.0 if self.wipe_active else hp_diff * 0.1
                        reward += hp_reward
                        reward_events.append(f"healed_partial:+{hp_reward:.1f}")

            if current_total_hp > 0 and self.wipe_active:
                self.wipe_active = False

            # Auch nach der ersten Baseline und bei unveraenderter Party
            # aktualisieren; sonst blieben die vorgeschlagenen Werte bei 0.
            self.last_party_size = current_party_size
            # NIE absenken: sonst waere ein PC-Boxwechsel (Pokemon einlagern
            # senkt die Party-Summe, wieder rausholen hebt sie zurueck auf den
            # alten Wert) als "Level-Up" farmbar, weil current_total_level
            # ueber den vorherigen (gesenkten) last-Wert steigt, ohne dass
            # irgendein Pokemon wirklich ein Level gewonnen hat. Die Baseline
            # bleibt beim bisherigen Maximum, bis ein ECHTER Level-Up sie hebt.
            self.last_party_total_level = max(
                self.last_party_total_level, current_total_level
            )
            self.last_party_total_hp = current_total_hp
            self.last_party_total_experience = current_total_experience
            self.last_party_identity = self._party_identity(valid_party)
            self.last_level = max(
                [self.last_level, p_lvl]
                + [int(mon.get("level", 0)) for mon in valid_party]
            )

            # Einmaliger, gesunder Gras-Start fuer Kampf-/Level-Spezialisten.
            # Er ist kein Reward und kann daher nicht gefarmt werden.
            if (
                self.has_target_starter
                and in_battle == 0
                and gameplay_ready
                and (bank, map_id) in self.WILD_TRAINING_MAPS
                and "squirtle_battle_ready" not in set(self.saved_milestones)
                and max_total_hp > 0
                and current_total_hp / max_total_hp >= 0.75
            ):
                if self._save_curriculum_state("squirtle_battle_ready"):
                    milestone_saved = "squirtle_battle_ready"

        if blocked_battle_start and self.BATTLE_BLOCKED_START_PENALTY:
            reward += self.BATTLE_BLOCKED_START_PENALTY
            reward_events.append(
                "battle_start_blocked:"
                f"{self.BATTLE_BLOCKED_START_PENALTY:.2f}"
            )

        # V7.5.1: reward only NEW opponent HP damage.
        if self.has_starter and self.total_steps % self.ENEMY_HP_READ_EVERY == 0:
            try:
                enemy_party = read_enemy_party(self.env)
            except Exception:
                enemy_party = []

            if enemy_party:
                old_enemy_fingerprint = tuple(
                    (
                        int(mon.get("slot", -1)),
                        int(mon.get("species_id", 0)),
                        int(mon.get("personality", 0)),
                        int(mon.get("cur_hp", 0)),
                        tuple(
                            (int(move.get("id", 0)), int(move.get("pp", 0)))
                            for move in mon.get("moves", [])
                        ),
                    )
                    for mon in self.enemy_party_cache
                )
                new_enemy_fingerprint = tuple(
                    (
                        int(mon.get("slot", -1)),
                        int(mon.get("species_id", 0)),
                        int(mon.get("personality", 0)),
                        int(mon.get("cur_hp", 0)),
                        tuple(
                            (int(move.get("id", 0)), int(move.get("pp", 0)))
                            for move in mon.get("moves", [])
                        ),
                    )
                    for mon in enemy_party
                )
                if old_enemy_fingerprint and new_enemy_fingerprint != old_enemy_fingerprint:
                    self._last_enemy_seen_step = self.total_steps
                self.enemy_party_cache = enemy_party
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
                        self._last_enemy_seen_step = self.total_steps
                        hp_damage = previous_min - cur_hp
                        if not self.battle_activity_open and not battle_just_ended:
                            self.battle_activity_open = True
                            self.run_stats["battles_started"] += 1
                            self.episode_battles_started += 1
                            reward_events.append("battle_fallback_start")
                        _wild_scale = self._battle_reward_scale(bank, map_id)
                        damage_reward = (
                            hp_damage * self.ENEMY_DAMAGE_REWARD_PER_HP * _wild_scale
                        )
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
                        _scale_tag = (":2x" if _wild_scale > 1.0
                                      else ":decayed" if _wild_scale < 1.0 else "")
                        reward_events.append(
                            f"enemy_damage:{hp_damage}hp{_scale_tag}:+{damage_reward:.2f}"
                        )

                        if cur_hp == 0 and previous_min > 0 and mon_key not in self.enemy_fainted_rewarded:
                            # V18: ENEMY_FAINT_REWARD ist 0 - kein Reward/Event
                            # fuer das KO selbst. Die Buchfuehrung (v.a.
                            # episode_wild_faints fuer das Abklingen!) laeuft
                            # unveraendert weiter.
                            faint_reward = self.ENEMY_FAINT_REWARD * _wild_scale
                            reward += faint_reward
                            self.enemy_fainted_rewarded.add(mon_key)
                            self.episode_enemy_faints += 1
                            self.faints_in_current_battle += 1
                            if (int(bank), int(map_id)) in self.WILD_TRAINING_MAPS:
                                self.episode_wild_faints += 1
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
                            if faint_reward:
                                _faint_tag = (":2x" if _wild_scale > 1.0
                                              else ":decayed" if _wild_scale < 1.0 else "")
                                reward_events.append(
                                    f"enemy_faint{_faint_tag}:+{faint_reward:.2f}"
                                )
                            # V19 BROCK RUSH: erstes gegnerisches KO in einem
                            # Trainerkampf in Marmoria pro Episode. NAEHERUNG:
                            # ohne Trainer-ID-RAM nicht sicher vom ersten
                            # Brock-Pokemon zu trennen, falls der Arena-Trainer
                            # uebersprungen wird. Anti-Farm: Episode-Flag.
                            if (
                                not self.episode_pewter_gym_trainer_beaten
                                and self._is_trainer_battle()
                                and (int(bank) == 6
                                     or self._current_world_stage(bank, map_id) == 6)
                            ):
                                self.episode_pewter_gym_trainer_beaten = True
                                if self.PEWTER_GYM_TRAINER_REWARD:
                                    reward += self.PEWTER_GYM_TRAINER_REWARD
                                    reward_events.append(
                                        "pewter_gym_trainer_ko:"
                                        f"+{self.PEWTER_GYM_TRAINER_REWARD:.0f}"
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
                if (
                    not self.left_house_confirmed
                    and self.START_HOUSE_PENALTY
                ):
                    reward += self.START_HOUSE_PENALTY
                    reward_events.append(
                        f"start_house:{self.START_HOUSE_PENALTY:.2f}"
                    )

                if self.start_spam_count == 2 and self.START_REPEAT_PENALTY_2:
                    reward += self.START_REPEAT_PENALTY_2
                    reward_events.append(
                        "start_repeat2:"
                        f"{self.START_REPEAT_PENALTY_2:.2f}"
                    )
                elif (
                    self.start_spam_count >= 3
                    and self.START_REPEAT_PENALTY_3PLUS
                ):
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
        # V17.2: "nicht gameplay_ready" heisst nur "Positions-RAM gerade
        # nicht vertrauenswuerdig" - das gilt fuer JEDES Dialogfenster, Menue
        # oder jeden Raumwechsel im normalen Spiel, nicht nur fuer das echte
        # Intro. Seit dem Savestate-Start ist das Intro immer schon erledigt
        # (intro_complete_rewarded startet auf True), der Block feuerte aber
        # trotzdem bei jedem "kurz nicht lesbar"-Moment weiter: unverdiente
        # intro_state-Boni (+2, gedeckelt +20) UND ein 900-Schritte-Anti-
        # Loop-Abbruch, der eine laengere Dialogszene mitten im echten Spiel
        # als Intro-Loop missverstehen und die Episode grundlos beenden
        # konnte. Zusaetzliche Bedingung macht den ganzen Block dauerhaft
        # inaktiv, sobald das Intro einmal (im Savestate) erledigt ist.
        if (
            not gameplay_ready
            and self.episode_start == "beginning"
            and not self.intro_complete_rewarded
        ):
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
                # V16: auch im Full-Modus aktiv. Jeder deutlich neue, grob
                # quantisierte Bildschirm zaehlt nur einmal und der gesamte
                # Bonus ist hart gedeckelt; damit bleibt das Intro im PPO-
                # Gedaechtnis, ohne Cursor-Flackern unbegrenzt zu farmen.
                if (
                    diff >= 10.0
                    and state_key not in self.intro_seen_states
                    and self.intro_novelty_reward_total
                        < self.INTRO_NOVELTY_REWARD_CAP
                ):
                    bonus = self.INTRO_NOVELTY_REWARD
                    bonus = min(
                        bonus,
                        self.INTRO_NOVELTY_REWARD_CAP
                        - self.intro_novelty_reward_total
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
                "intro_complete", self.route_steps
            )
            if in_battle == 0:
                if self._save_curriculum_state("intro_complete"):
                    milestone_saved = "intro_complete"
            info["intro_loop_reset"] = False
        else:
            info["intro_loop_reset"] = False

        # V8.7: route learning from previously successful exits.
        # V15.3: aus im All-Full-Regime - "viel gelaufener Weg = guter Weg"
        # ist genau die Ruecklauf-Schleife, die wir loswerden wollen.
        if (
            gameplay_ready
            and in_battle == 0
            and not self.FULL_ONLY_MODE
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
        # V15.3: im All-Full-Regime komplett aus - v9_explorer_new_tile (+1.0
        # farmbar), indoor_stall, v9_stuck. Erkundung laeuft allein ueber
        # new_edge_global (+0.10, un-farmbar) und die Meilensteine.
        if gameplay_ready and in_battle == 0 and not self.FULL_ONLY_MODE:
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

            if self.training_objective in self.WORLD_ROLES:
                # Innenraeume sind fuer Paket/Labor/Gym notwendig, bleiben aber
                # reward-neutral. Nur echte Weltkarten formen Exploration.
                if self._current_world_stage(bank, map_id) > 0:
                    if pos_key not in self.v9_episode_tiles:
                        self.v9_episode_tiles.add(pos_key)
                        _tile_bonus = self.V9_EXPLORER_NEW_TILE_BONUS
                    else:
                        _tile_bonus = self.V9_EXPLORER_REPEAT_TILE_PENALTY
                    if _tile_bonus:
                        reward += _tile_bonus
                        reward_events.append(
                            f"v9_tile:{_tile_bonus:+.2f}"
                        )

            # V13.3: Progress/Battle/Level mit Starter, die drinnen herumhaengen
            # (kein Pflicht-Gebaeude direkt am Anfang), kriegen einen kleinen
            # Sog nach draussen. Die Route ist damit immer die profitablere Wahl.
            if (
                self.training_objective in ("progress", "battle", "level")
                and self.has_starter
                and bank != self.OVERWORLD_BANK
                and self._world_stage() < 3
                and self.indoor_steps_without_transition > 150
            ):
                reward -= 0.05
                reward_events.append("indoor_drift:-0.05")

            # V15: Richtung ist Story-abhaengig. Mit Eichs Paket geht die
            # Rampe nach Sueden zurueck zum Labor; davor/danach auf den
            # geraden Verbindungen nach Norden. Im Wald gibt es keine Rampe.
            _returning_parcel = (
                self.viridian_mart_scene >= 1
                and self.pallet_oaks_lab_scene < 6
                and self.viridian_old_man_scene < 1
            )
            _mk = (int(bank), int(map_id))

            # Dichter, schleifenfreier Richtungsreward als Potential-
            # Differenz zwischen zwei echten Positionen. Ein Schritt zurueck
            # kostet genau so viel wie der Hinweg gebracht hat. Damit bleibt
            # freie Exploration moeglich, aber Route 1 verliert nach einem
            # Absatz/Rueckschritt nicht dauerhaft ihren Lern-Gradienten.
            _previous_coord = getattr(self, "last_exploration_coord", None)
            if (
                self.training_objective in self.WORLD_ROLES
                and self.has_starter
                and _previous_coord is not None
                and tuple(_previous_coord[:2]) == _mk
            ):
                _px, _py = int(_previous_coord[2]), int(_previous_coord[3])
                _dx, _dy = int(x) - _px, int(y) - _py
                # Normale Bewegung plus kleine In-Map-Spruenge zulassen,
                # kaputte RAM-Spruenge aber nie belohnen.
                if 0 < abs(_dx) + abs(_dy) <= 4:
                    _north_target = (
                        not _returning_parcel
                        and (
                            _mk in self.NORTH_CORRIDOR_MAPS
                            or (_mk == self.STAGE_VIRIDIAN and self.parcel_delivered_confirmed)
                        )
                    )
                    _south_target = _returning_parcel and _mk in self.PARCEL_RETURN_MAPS
                    _directed_rows = _dy if _south_target else -_dy if _north_target else 0
                    if _directed_rows and self.CORRIDOR_STEP_REWARD:
                        _step_r = _directed_rows * self.CORRIDOR_STEP_REWARD
                        reward += _step_r
                        reward_events.append(f"corridor_step:{_step_r:+.2f}")

            if (
                self.training_objective in self.WORLD_ROLES
                and self.has_starter
                and _returning_parcel
                and _mk in self.PARCEL_RETURN_MAPS
            ):
                _prev_best = self._south_corridor_best.get(_mk, int(y) - 1)
                if int(y) > _prev_best and self.NORTH_CORRIDOR_ROW_REWARD:
                    _rows = min(int(y) - _prev_best, self.NORTH_CORRIDOR_MAX_ROWS)
                    _r = _rows * self.NORTH_CORRIDOR_ROW_REWARD
                    reward += _r
                    reward_events.append(f"parcel_return_south:{map_id}:+{_r:.1f}")
                    self._south_corridor_best[_mk] = int(y)
            elif (
                self.training_objective in self.WORLD_ROLES
                and self.has_starter
                and not _returning_parcel
                and (
                    _mk in self.NORTH_CORRIDOR_MAPS
                    or (_mk == self.STAGE_VIRIDIAN and self.parcel_delivered_confirmed)
                )
            ):
                _prev_best = self._north_corridor_best.get(_mk, int(y) + 1)
                if int(y) < _prev_best and self.NORTH_CORRIDOR_ROW_REWARD:
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
                    reward_events.append("confirmed_stairs_warp:+0")

                reward += 150.0
                reward_events.append("stairs_down:+150")

                if self.training_objective == "stairs":
                    reward += 50.0
                    reward_events.append("objective_stairs:+50")
                    self.objective_success = True
                    objective_done = True
                self.reward_event_counts["stairs_down"] += 1
                self.episode_milestone_steps.setdefault(
                    "stairs_down", self.route_steps
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
                    reward_events.append("confirmed_exit_warp:+0")

                reward += 300.0
                reward_events.append("left_house_confirmed:+300")
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
                    "left_house", self.route_steps
                )
                if in_battle == 0:
                    if self._save_curriculum_state("left_house"):
                        milestone_saved = "left_house"

            # Erster echter Schritt vom Hauseingang weg.
            if (
                self.left_house_rewarded
                and not self.outdoor_first_step_rewarded
                and self.training_objective != "starter"
                and bank == self.OVERWORLD_BANK
                and map_id == self.first_outdoor_map
                and self.outdoor_entry_coord is not None
                and abs(x - self.outdoor_entry_coord[0])
                    + abs(y - self.outdoor_entry_coord[1]) >= 1
            ):
                self.outdoor_first_step_rewarded = True
                reward_events.append("outdoor_first_step:+0")
                self.reward_event_counts["outdoor_first_step"] += 1
                self.last_progress_advance_step = self.route_steps
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
                reward_events.append("next_outdoor_map:+0")
                # V10.33: Alabastia -> Route 1 geschafft -> Weg als Spur sichern.
                self._commit_journey_route()
                self.reward_event_counts["next_outdoor_map"] += 1
                self.episode_milestone_steps.setdefault(
                    "next_outdoor_map", self.route_steps
                )
                self.last_progress_advance_step = self.route_steps
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
        starter_species = self._starter_species()
        if not self.has_starter and party_has_starter:
            self.has_starter = True
            self.has_target_starter = (
                starter_species == self.TARGET_STARTER_SPECIES
            )
            self.last_level = max(p_lvl, 5)

            if not self.has_target_starter:
                # Falsche Wahl sofort beenden. So bekommt PPO ein eindeutiges
                # Signal und kein Bisasam-/Glumanda-State gelangt neu in das
                # Schiggy-Curriculum.
                reward += self.WRONG_STARTER_PENALTY
                reward_events.append(
                    f"wrong_starter_species_{starter_species}:"
                    f"{self.WRONG_STARTER_PENALTY:.0f}"
                )
                info["wrong_starter_species"] = int(starter_species)
                truncated = True
                self.last_stage_timeout = "wrong_starter"
                info["last_stage_timeout"] = "wrong_starter"
            else:
                reward += self.STARTER_REWARD
                reward_events.append(
                    f"target_starter_squirtle:+{self.STARTER_REWARD:.0f}"
                )
                self.reward_event_counts["first_pokemon"] += 1
                self.episode_milestone_steps.setdefault(
                    "first_pokemon", self.route_steps
                )
                self.last_progress_advance_step = self.route_steps
                self.starter_obtained_step = self.route_steps
                self._claim_journey_milestone(
                    "journey_starter", "journey_seen_starter"
                )

                bridge = self._maybe_save_progress_bridge("starter")
                if bridge:
                    milestone_saved = bridge

                if in_battle == 0:
                    if self._save_curriculum_state("squirtle"):
                        milestone_saved = "squirtle"

                # Erfolg bleibt an "Schiggy erhalten UND Labor verlassen"
                # gekoppelt. Der erste Teil liefert nur 40 Prozent.
                if self.training_objective == "starter":
                    reward += self.STARTER_SPECIALIST_BONUS * 0.4
                    reward_events.append(
                        "objective_starter_got_squirtle:+"
                        f"{self.STARTER_SPECIALIST_BONUS * 0.4:.0f}"
                    )

        # Sicherheitsnetz: Die Party kann in einem anderen RAM-Lesezyklus
        # sichtbar werden als das Levelsignal. Deshalb eine falsche Wahl in
        # echten Starter-/Full-from-Beginning-Runs auf JEDEM Schritt beenden,
        # nicht nur im ersten Erkennungsframe. Tiefe Legacy-Curriculum-Runs
        # bleiben als voruebergehende Fortschrittsbruecke erlaubt.
        starter_species = self._starter_species()
        wrong_fresh_starter = (
            starter_species in self.STARTER_SPECIES
            and starter_species != self.TARGET_STARTER_SPECIES
            and (
                self.training_objective == "starter"
                or (
                    self.training_objective == "full"
                    and self.episode_start == "beginning"
                )
            )
        )
        if wrong_fresh_starter and not truncated:
            reward += self.WRONG_STARTER_PENALTY
            reward_events.append(
                f"wrong_starter_guard_{starter_species}:"
                f"{self.WRONG_STARTER_PENALTY:.0f}"
            )
            info["wrong_starter_species"] = int(starter_species)
            truncated = True
            self.last_stage_timeout = "wrong_starter"
            info["last_stage_timeout"] = "wrong_starter"

        # V10.31 STARTER-OUTDOOR: erster Overworld-Schritt MIT Starter.
        # Das ist die eigentliche Wand ("raus aus Eichs Labor"). Einmal pro
        # Episode, nicht farmbar. Belohnt + speichert einen Curriculum-State,
        # damit Progress-Agenten direkt draussen mit Starter starten koennen.
        if (
            self.has_target_starter
            and not self.starter_outdoor_rewarded
            and bank == self.OVERWORLD_BANK
            and loc["valid"]
        ):
            self.starter_outdoor_rewarded = True
            # V17.3: war 500 - eine echte Huerde ("aus Eichs Labor raus")
            # in der alten Kaltstart-Architektur. Seit dem Savestate-Start
            # ist has_target_starter von Episodenbeginn an True und der
            # erste gueltige Aussen-Schritt kommt praktisch sofort - das
            # gab bis zu 500 Gratis-Reward jede einzelne Episode, ohne
            # jeden echten Fortschritt. Auf 25 reduziert (wie replay_map_once).
            reward += 25.0
            reward_events.append("starter_outdoor:+25")
            # V10.33: den Weg raus aus dem Labor als Brotkrumen-Spur sichern.
            self._commit_journey_route()
            self.reward_event_counts["starter_outdoor"] = (
                self.reward_event_counts.get("starter_outdoor", 0) + 1
            )
            self.episode_milestone_steps.setdefault(
                "starter_outdoor", self.route_steps
            )
            self.last_progress_advance_step = self.route_steps
            if in_battle == 0:
                if self._save_curriculum_state("squirtle_outdoor"):
                    milestone_saved = "squirtle_outdoor"
            bridge = self._maybe_save_progress_bridge("starter_outdoor")
            if bridge:
                milestone_saved = bridge
            if self.training_objective == "starter":
                reward += self.STARTER_SPECIALIST_BONUS * 0.6
                reward_events.append(
                    f"objective_starter_out:+{self.STARTER_SPECIALIST_BONUS * 0.6:.0f}"
                )
                self.objective_success = True
                objective_done = True

        # Einzel-Slot-Levelreward entfernt: V10.27A nutzt oben Team-Level.
        self.last_level = max(self.last_level, p_lvl)

        if badges > self.last_badges:
            badge_gain = badges - self.last_badges
            badge_reward = badge_gain * self.BADGE_EARNED_REWARD
            reward += badge_reward
            self.reward_event_counts["badge"] += badge_gain
            reward_events.append(f"badge:+{badge_reward:.0f}")
            if self.BADGE_FIRST_GLOBAL_REWARD and claim_event(
                EXPLORATION_MEMORY_DIR, f"badge_{int(badges)}_ever",
                self.shared_species, self.shared_lock
            ):
                reward += self.BADGE_FIRST_GLOBAL_REWARD
                reward_events.append(
                    f"badge_first_global:{int(badges)}:"
                    f"+{self.BADGE_FIRST_GLOBAL_REWARD:.0f}"
                )
            self.last_badges = badges
            self.last_progress_advance_step = self.route_steps
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

            _route_roller = self.training_objective in self.WORLD_ROLES

            # V17.2: Der Teleport zum Pokecenter nach einem Party-Wipe darf
            # nie als "neue Map" bezahlt werden. Buchfuehrung/globaler Claim
            # laufen unveraendert (die Map gilt danach fuer alle als bekannt),
            # nur die Auszahlung pausiert waehrend des Cooldowns.
            _wipe_cooldown_active = (
                self.wipe_active or self.route_steps < self._post_wipe_reward_cooldown_until
            )
            if map_key not in self.visited_maps:
                self.visited_maps.add(map_key)
                self.learning_seen_maps.add(map_key)
                if map_key not in self.persistent_known_maps:
                    self.persistent_known_maps.add(map_key)
                    self.exploration_memory_dirty = True
                    self._nav_target_cache = None
                    # V17.4: kein fleet-weiter Einmal-Jackpot mehr fuer
                    # Route/Stadt - nur noch EIN Wert pro Run, fuer JEDEN
                    # Agenten gleich. Vorher bekam ausschliesslich der EINE
                    # Agent, der eine Map als Erster ueberhaupt fand, den
                    # grossen Bonus (500) - alle folgenden Agenten (und
                    # derselbe Agent in spaeteren Episoden) nur den kleinen
                    # Episodenwert. Das bremste den Vorstoss nach Norden,
                    # sobald die "billigen" Erstfunde in Reichweite
                    # ausgegangen waren.
                    _claimed_globally = self._claim_shared(
                        self.shared_maps, map_key
                    )
                    _is_route_or_city = (
                        bank == self.OVERWORLD_BANK
                        or self._current_world_stage(bank, map_id) > 0
                    )
                    if _wipe_cooldown_active:
                        reward_events.append("new_map_suppressed_post_wipe:+0")
                    elif _is_route_or_city and self._can_reward_map_arrival(bank, map_id):
                        _map_reward = (
                            self.CITY_EPISODE_REWARD
                            if map_key in self.CITY_MAPS
                            else self.EPISODE_NEW_MAP_REWARD
                        )
                        reward += _map_reward
                        self.last_progress_advance_step = self.route_steps
                        reward_events.append(
                            "new_map_episode:"
                            f"+{_map_reward:.2f}"
                        )
                    elif not _is_route_or_city:
                        # Innenraeume: fleet-weit EINMALIGER Fund pro Gebaeude-
                        # Map. Alabastia-Schuppen (Bank 4) +25 (kein Fortschritt,
                        # nur "gesehen"); Stadt-Gebaeude (Vertania/Marmoria,
                        # Bank 5/6) +BUILDING_FIRST_GLOBAL_REWARD - Arena, Laden,
                        # Haeuser sollen einmal angespielt werden. Ueber
                        # claim_event (reward_events.json), damit es auch fuer
                        # Gebaeude feuert, die shared_maps schon kennt, und nach
                        # Neustarts erledigt bleibt.
                        # NUR Bank 5 (Vertania) und Bank 6 (Marmoria) duerfen
                        # den +500-Erstfund bekommen. Bank 4 = Alabastia/Pallet-
                        # Innenraeume NIEMALS - weder +500, noch ein
                        # building_4_x-claim_event, nur der normale kleine
                        # Fleet-Erstfund (EPISODE_NEW_MAP_REWARD ueber shared_maps,
                        # nicht wiederholbar). Der explizite `!= 4` ist eine
                        # Sicherung, falls jemand spaeter versehentlich 4 in
                        # CITY_BUILDING_BANKS aufnimmt.
                        _is_city_building = (
                            int(bank) in self.CITY_BUILDING_BANKS
                            and int(bank) != 4
                        )
                        if _is_city_building:
                            _bld_reward = self.BUILDING_FIRST_GLOBAL_REWARD
                            _bld_key = f"building_{int(bank)}_{int(map_id)}"
                            _bld_claimed = claim_event(
                                EXPLORATION_MEMORY_DIR, _bld_key,
                                self.shared_species, self.shared_lock,
                            )
                            _bld_event = "new_building_global"
                        else:
                            _bld_reward = self.EPISODE_NEW_MAP_REWARD
                            _bld_key = None
                            _bld_claimed = _claimed_globally
                            _bld_event = "new_building_seen"
                        if _bld_reward and _bld_claimed:
                            reward += _bld_reward
                            reward_events.append(
                                f"{_bld_event}:{int(bank)}_{int(map_id)}:"
                                f"+{_bld_reward:.2f}"
                            )
                elif not _wipe_cooldown_active and (
                    bank == self.OVERWORLD_BANK or self._current_world_stage(bank, map_id) > 0
                ) and self._can_reward_map_arrival(bank, map_id):
                    # V17.3: nur draussen. Innenraeume rund um den fixen
                    # Savestate-Start (Reds Haus, Rivalenhaus, Eichs Labor)
                    # sind JEDEM Agenten in JEDER Episode sofort bekannt -
                    # das gab bis zu ~100 Gratis-Reward pro Episode allein
                    # dafuer, kurz alle bekannten Raeume abzuklappern, bevor
                    # ueberhaupt Route 1 erreicht wird. Live beobachtet:
                    # Agenten "duempelten im Haus rum" statt loszulaufen,
                    # gerade seit Episoden nach einem Wipe nicht mehr enden.
                    # Echte Staedte zahlen bewusst mehr und JEDE Episode neu,
                    # nicht nur beim einmaligen globalen Fund - Ziel:
                    # schneller/haeufiger bis zur naechsten Stadt vorstossen.
                    # V17.4: Scouts ausgenommen - die resumen jede Episode
                    # per Savestate mitten in fremdem Terrain und liefen
                    # gezielt zurueck ins laengst bekannte Alabastia/Vertania,
                    # nur um dort den Wiederholungs-Bonus fuer eine Stadt
                    # abzugreifen, die sie schon x-mal besucht haben - das
                    # bremste den Vorstoss nach vorn staerker als es half.
                    # V17.4-Fix: Alabastia (Pallet Town) selbst ist EBENFALLS
                    # ausgenommen. Der feste Savestate startet "full"-Agenten
                    # in Eichs Labor (Bank 4/Map 3), direkt nach Paketabgabe -
                    # Pallet Town liegt buchstaeblich 1-2 Schritte vor der
                    # Tuer und ist als CITY_MAP selbst reward-berechtigt.
                    # Ohne diesen Ausschluss zahlte das JEDE einzelne Episode
                    # automatisch +250 quasi nur fuers Rauslaufen, ganz ohne
                    # echte Erkundung - live beobachtet direkt bei
                    # Episodenstart (starter_outdoor + replay_map_once:+250
                    # binnen der ersten ~200 Steps, bei praktisch JEDEM
                    # Agenten). Ein erster Versuch, stattdessen die exakte
                    # Episoden-Startmap zu tracken, griff nicht, weil die
                    # Startmap Eichs Labor ist, nicht Pallet Town selbst.
                    _map_reward = (
                        self.CITY_EPISODE_REWARD
                        if map_key in self.CITY_MAPS
                        else self.EPISODE_NEW_MAP_REWARD
                    )
                    reward += _map_reward
                    reward_events.append(
                        "replay_map_once:"
                        f"+{_map_reward:.2f}"
                    )

            # Jede Policy lernt einen echten lokalen Stage-Anstieg. Der grosse
            # globale Bonus wird weiterhin nur einmal flottenweit ausgezahlt.
            _stage = self._world_stage()
            # Count actual first arrival, including battle decisions in elapsed time.
            _location_stage = self._stage_at_current_location(bank, map_id)
            if loc.get("trusted") and _location_stage > 0:
                self.stage_arrival_steps.setdefault(str(_location_stage), int(self.total_steps))
            _old_stage = int(getattr(self, "episode_best_stage", 0))
            if self.has_starter and _stage > _old_stage:
                stage_gain = _stage - _old_stage
                if self.STAGE_ADVANCE_REWARD:
                    stage_reward = self.STAGE_ADVANCE_REWARD * stage_gain
                    reward += stage_reward
                    reward_events.append(
                        f"stage_advance:{_old_stage}->{_stage}:+{stage_reward:.0f}"
                    )
                self.episode_best_stage = _stage
                self.last_progress_advance_step = self.route_steps

                # Pro neu erreichter Stufe und Episode exakt ein fester Bonus.
                # Kein Multiplizieren mit der Stufennummer und kein Reward bei
                # Rueckkehr auf eine bereits erreichte Stufe.
                depth_bonus = self.NEW_GLOBAL_DEPTH_REWARD * stage_gain
                if depth_bonus:
                    reward += depth_bonus
                    reward_events.append(f"world_depth:{_stage}:+{depth_bonus:.0f}")
                self.reward_event_counts["world_depth"] = (
                    self.reward_event_counts.get("world_depth", 0) + 1
                )
                if self._claim_global_depth(_stage):
                    self.run_stats["global_depth_records"] += 1
                    self._save_run_stats()
                    if self.GLOBAL_STAGE_RECORD_REWARD:
                        reward += self.GLOBAL_STAGE_RECORD_REWARD
                        reward_events.append(
                            f"global_stage_record:{_stage}:"
                            f"+{self.GLOBAL_STAGE_RECORD_REWARD:.0f}"
                        )

            # V19 BROCK RUSH: Marmoria (Pewter, Stage 6) erstmals in dieser
            # Episode erreicht - mit Pikachu bereits in der Party einmalig
            # extra (Pikachu VOR Misty mitnehmen, aber kein Brock-Pflichtziel).
            # Anti-Farm: Episode-Flag.
            if (
                _location_stage == 6
                and loc.get("trusted")
                and not self.episode_pewter_reached
            ):
                self.episode_pewter_reached = True
                _has_pika = any(
                    int(m.get("species_id", 0)) == self.PIKACHU_SPECIES_ID
                    for m in (self.player_party_cache or [])
                )
                if _has_pika and self.PEWTER_WITH_PIKACHU_REWARD:
                    reward += self.PEWTER_WITH_PIKACHU_REWARD
                    reward_events.append(
                        f"pewter_with_pikachu:+{self.PEWTER_WITH_PIKACHU_REWARD:.0f}"
                    )

            # V19 POST_WIPE_RECOVERY_MODE: Ende + Einmal-Bonus. Erreicht der
            # Agent (aussen) wieder die Standort-Stufe der Vor-Wipe-Front, ODER
            # hat er unterwegs einen tieferen Center-Respawn aktiviert, ODER
            # einen Orden geholt -> Recovery beendet, einmal +300.
            if getattr(self, "post_wipe_recovery", False):
                _loc_now = self._current_world_stage(bank, map_id)
                if (
                    (_loc_now > 0 and _loc_now >= int(self.pre_wipe_best_stage))
                    or int(self.best_pokecenter_heal_stage)
                        > int(self.pre_wipe_best_center_stage)
                    or int(self.last_badges) > int(self.pre_wipe_badges)
                ):
                    self.post_wipe_recovery = False
                    if self.POST_WIPE_FRONT_RECOVERED_REWARD:
                        reward += self.POST_WIPE_FRONT_RECOVERED_REWARD
                        reward_events.append(
                            "post_wipe_front_recovered:"
                            f"+{self.POST_WIPE_FRONT_RECOVERED_REWARD:.0f}"
                        )

            # Each geographic stage competes by north position, then reward.
            # A full runner may improve Route 1 even after reaching the forest.
            if (
                _route_roller
                and self.has_target_starter
                and in_battle == 0
                and not _wipe_cooldown_active
                and any(int(m.get("cur_hp", 0)) > 0 for m in self.player_party_cache)
                and loc["valid"]
                and self._stage_at_current_location(bank, map_id) >= 2
            ):
                _stage_now = self._stage_at_current_location(bank, map_id)
                _mk = (int(bank), int(map_id))
                if _mk == getattr(self, "_stage_hold_map", None):
                    self._stage_hold_steps = getattr(self, "_stage_hold_steps", 0) + 1
                else:
                    self._stage_hold_map = _mk
                    self._stage_hold_steps = 1
                # Require three stable location reads before capturing.
                _hold_required = 3
                if (
                    _stage_now >= 2
                    and self._stage_hold_steps >= _hold_required
                ):
                    _saved_now = self._save_stage_checkpoint(
                        _stage_now, bank, map_id, x, y,
                        episode_reward=self.current_reward + reward,
                    )
                    if _saved_now:
                        milestone_saved = f"stage_{_stage_now}"
                        self._commit_journey_route()
                        self.episode_journey_edges = []
                        bridge = self._maybe_save_progress_bridge(
                            f"stage_{_stage_now}"
                        )
                        if bridge:
                            milestone_saved = bridge


            if coord_key not in self.seen_coords:
                self.seen_coords.add(coord_key)
                if _wipe_cooldown_active:
                    reward_events.append("new_tile_suppressed_post_wipe:+0")
                else:
                    # V18: Erstfund einer Kachel PRO LAUF (seen_coords, jede
                    # Episode frisch). Handgesetzte Leiter nach Kachel-Karte:
                    #   * Story-Aussenmap  -> TILE_REWARD_BY_STAGE
                    #   * Innenraum        -> INTERIOR_TILE_REWARD_BY_BANK
                    # Selbstbegrenzend: dieselben ~30 Startkacheln zahlen pro
                    # Episode nur einmal, danach zwingt die Zeitgebuehr weiter.
                    # Zusatz obendrauf: fleet-weit einmalig +GLOBAL_NEW_TILE_BONUS
                    # beim allerersten Betreten dieser Kachel ueberhaupt.
                    _tile_stage = self._current_world_stage(bank, map_id)
                    # V18: Ein Scout, der zurueck auf/vor seine Spawn-Stufe
                    # laeuft (z.B. Route-1-Scout, der nach Alabastia abwandert),
                    # bekommt dafuer KEINEN Kachel-Reward - genau wie beim
                    # Stadt-/Map-Arrival (_can_reward_map_arrival). Sonst zieht
                    # das Zurueckwandern trotzdem noch +2/Kachel.
                    _scout_backtrack = (
                        self.training_objective == "scout"
                        and _tile_stage != 0
                        and _tile_stage <= int(getattr(self, "episode_start_stage", 1))
                    )
                    if _scout_backtrack:
                        reward_events.append("new_tile_scout_backtrack:+0")
                    else:
                        if _tile_stage > 0:
                            _tile_reward = self.TILE_REWARD_BY_STAGE.get(
                                _tile_stage, self.NEW_TILE_REWARD
                            )
                        else:
                            _tile_reward = self.INTERIOR_TILE_REWARD_BY_BANK.get(
                                int(bank), self.INTERIOR_TILE_REWARD_DEFAULT
                            )
                        # Pro Karte/Episode: nach dem Deckel nur noch der
                        # Bruchteil - Abgrasen einer grossen Karte / Gebaeude-
                        # Touren sind dann kein farmbarer Loop mehr. Innenraeume
                        # kleiner gedeckelt als Aussenmaps.
                        _map_tiles = self._episode_tiles_by_map.get(map_key, 0)
                        self._episode_tiles_by_map[map_key] = _map_tiles + 1
                        _cap = (
                            self.TILE_REWARD_CAP_PER_MAP if _tile_stage > 0
                            else self.INTERIOR_TILE_CAP_PER_MAP
                        )
                        _capped = _map_tiles >= _cap
                        if _capped:
                            _tile_reward *= self.TILE_REWARD_AFTER_CAP_FACTOR
                        _tile_global = (
                            self.GLOBAL_NEW_TILE_BONUS
                            if self._claim_shared(self.shared_tiles, coord_key)
                            else 0.0
                        )
                        _tile_reward += _tile_global
                        reward += _tile_reward
                        reward_events.append(
                            f"new_tile:s{_tile_stage}"
                            f"{'+g' if _tile_global else ''}"
                            f"{':capped' if _capped else ''}:+{_tile_reward:.2f}"
                        )

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
                            if (
                                edge_key in self.persistent_known_edges
                                and self.EPISODE_EDGE_REWARD
                            ):
                                reward += self.EPISODE_EDGE_REWARD
                                reward_events.append(f"replay_edge:+{self.EPISODE_EDGE_REWARD:.2f}")
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

                            _claimed_edge_globally = self._claim_shared(
                                self.shared_edges, edge_key
                            )
                            self.steps_since_new_edge = 0
                            if _wipe_cooldown_active:
                                reward_events.append(
                                    "new_edge_suppressed_post_wipe:+0"
                                )
                            elif _claimed_edge_globally:
                                if self.NEW_EDGE_REWARD:
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
                                local_edge_reward = self.EPISODE_EDGE_REWARD
                                if local_edge_reward:
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
                            if revisit_penalty:
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
                            if repeat_penalty:
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
                                in ("progress", "full", "scout")
                        ):
                            targets = self._progress_targets_for_map(
                                bank, map_id, x, y
                            )
                        # V19: fuer die Welt-Rollen auf einer Aussenmap liefern
                        # die beiden oben bewusst nichts (generische Ziele
                        # bevorzugten Haeuser/Sackgassen). Stattdessen exakt die
                        # Transition Richtung naechster Stufe / Center / Arena.
                        if not targets and self.left_house_rewarded:
                            targets = self._v19_forward_targets(bank, map_id)
                        if targets:
                            target_step_reward = (
                                self.EARLY_STORY_STEP_REWARD
                                if (
                                    not self.left_house_rewarded
                                    and self.training_objective
                                    in ("stairs", "exit", "starter", "full")
                                )
                                else self.TARGET_PROGRESS_REWARD
                            )
                            # V19: waehrend Post-Wipe-Recovery zieht der Rueckweg
                            # zur alten Front deutlich staerker (+/-0.50).
                            if getattr(self, "post_wipe_recovery", False):
                                target_step_reward = (
                                    self.POST_WIPE_TARGET_PROGRESS_REWARD
                                )
                            prev_d = self._graph_distance(
                                bank, map_id, (px, py), targets
                            )
                            new_d = self._graph_distance(
                                bank, map_id, (x, y), targets
                            )

                            if (
                                target_step_reward
                                and
                                prev_d is not None
                                and new_d is not None
                            ):
                                if new_d < prev_d:
                                    reward += target_step_reward
                                    reward_events.append(
                                        "target_closer:"
                                        f"+{target_step_reward:.2f}"
                                    )
                                elif new_d > prev_d:
                                    reward -= target_step_reward
                                    reward_events.append(
                                        "target_farther:"
                                        f"-{target_step_reward:.2f}"
                                    )

                else:
                    # Mapwechsel / Warp: Ein konkreter Ein-/Ausgangspunkt wird
                    # persistent gespeichert. Rueckweg durch dieselbe Tuer = bekannt.
                    transition_key = self._transition_key(
                        pb, pm, px, py,
                        bank, map_id, x, y
                    )
                    # V18: Der REWARD-Claim laeuft ueber ein grobes Kartenpaar
                    # statt den koordinatengenauen transition_key. Eine Stadt-/
                    # Routen-Grenze kann an jedem x/y ueberquert werden - mit
                    # Koordinaten im Key war der Vorrat an "global neuen" Warps
                    # praktisch unerschoepflich (Live: der isolierte Watcher
                    # meldete staendig new_warp_global fuer laengst bekannte
                    # Grenzen). "Neuer Warp" heisst jetzt: die Flotte hat diese
                    # zwei Karten zum ersten Mal ueberhaupt verbunden - begrenzt
                    # auf die Zahl echter Karten-Nachbarschaften, nicht farmbar.
                    # Die koordinatengenaue persistent_known_transitions (fuer
                    # Navigation) bleibt unveraendert.
                    warp_pair_key = self._warp_pair_key(pb, pm, bank, map_id)

                    # V17.4-Fix: der Wipe-Cooldown schuetzte bisher nur Map-/
                    # Kachel-Rewards vor dem automatischen Pokecenter-
                    # Teleport nach einem Party-Wipe - der Warp-Reward-Block
                    # hier hatte NIE eine solche Pruefung, obwohl derselbe
                    # Teleport einen (von,nach)-Uebergang zwischen einer fast
                    # beliebigen Kampf-Position und dem Pokecenter erzeugt.
                    # Da die genaue Kampf-Position praktisch nie zweimal
                    # gleich ist, war das ein fast unerschoepflicher, immer
                    # "global neuer" +100-Warp-Fund - jeder Party-Wipe konnte
                    # so zusaetzlich zur -100-Strafe einen Bonus einbringen.
                    # Buchfuehrung/Claims laufen wie bei Maps/Kacheln normal
                    # weiter, nur die Auszahlung pausiert waehrend des
                    # Cooldowns.
                    if transition_key not in self.learning_seen_transitions:
                        self.learning_seen_transitions.add(transition_key)
                        if (
                            transition_key in self.persistent_known_transitions
                            and self.EPISODE_TRANSITION_REWARD
                            and not _wipe_cooldown_active
                        ):
                            reward += self.EPISODE_TRANSITION_REWARD
                            reward_events.append(
                                "replay_warp:"
                                f"+{self.EPISODE_TRANSITION_REWARD:.2f}"
                            )
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

                        # V18: kein Warp-Reward auf dem Step, an dem ein Kampf
                        # endet (Pseudo-Uebergang durch die Vor-/Nach-Kampf-
                        # Positionsdifferenz), und nicht waehrend des Wipe-
                        # Cooldowns. Der Global-Bonus faellt nur, wenn das
                        # Karten-Paar (a) nicht schon aus der Navigations-
                        # Historie bekannt ist UND (b) noch nie in
                        # reward_events.json vermerkt wurde - Letzteres macht
                        # ihn dauerhaft ueber Neustarts hinweg einmalig.
                        _pair_known = warp_pair_key in self._known_warp_pairs
                        self._known_warp_pairs.add(warp_pair_key)
                        if _wipe_cooldown_active or battle_just_ended:
                            reward_events.append("new_warp_suppressed:+0")
                        elif _pair_known:
                            reward_events.append("known_warp:+0")
                        elif self.NEW_TRANSITION_REWARD and claim_event(
                            EXPLORATION_MEMORY_DIR,
                            f"warp_{warp_pair_key[0][0]}_{warp_pair_key[0][1]}"
                            f"_{warp_pair_key[1][0]}_{warp_pair_key[1][1]}",
                            self.shared_species, self.shared_lock,
                        ):
                            reward += self.NEW_TRANSITION_REWARD
                            reward_events.append(
                                "new_warp_global:"
                                f"+{self.NEW_TRANSITION_REWARD:.2f}"
                            )
                        else:
                            reward_events.append("known_warp:+0")

                        # Sofort fuer alle lokalen Zielabfragen sichtbar.
                        self.shared_transition_snapshot.add(
                            transition_key
                        )
                        if len(self.persistent_known_transitions | self.shared_transition_snapshot) >= 5:
                            self._claim_journey_milestone("journey_warp5","journey_seen_warp5")

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
            if self.route_steps >= self.EARLY_HOUSE_HARD_CAP:
                stage_timeout = "early_house_hard_cap"

            elif (
                not self.intro_complete_rewarded
                and self.route_steps >= self.INTRO_TIMEOUT_STEPS
            ):
                stage_timeout = "intro_timeout"

            elif (
                self.intro_complete_rewarded
                and not self.stairs_down_rewarded
            ):
                intro_step = self.episode_milestone_steps.get(
                    "intro_complete", 0
                )
                if self.route_steps - intro_step >= self.STAIRS_TIMEOUT_STEPS:
                    stage_timeout = "stairs_timeout"

            elif self.stairs_down_rewarded:
                stairs_step = self.episode_milestone_steps.get(
                    "stairs_down", 0
                )
                if self.route_steps - stairs_step >= self.EXIT_TIMEOUT_STEPS:
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

            # V17.3: durchgehend zunehmend negativ statt drei fester Sprungstufen
            # (0.03/0.12/0.40) - der Druck, aus einer Schleife auszubrechen,
            # waechst jetzt jeden Schritt weiter statt in Spruengen, bleibt an
            # denselben Eckpunkten (60/180/400/900 Schritte) aber vergleichbar.
            if in_battle == 0 and self.stuck_counter >= 60:
                reward -= 0.001 * (self.stuck_counter - 59)

            local_loop = self.local_loop_guard.update(
                (bank, map_id, x, y) if loc.get("trusted") else None,
                (self._world_stage(), self.last_badges,
                 self.last_party_total_experience, len(self.seen_coords)),
                in_battle=bool(in_battle),
            )
            if in_battle == 0 and (self.stuck_counter >= 900 or local_loop):
                truncated = True
                info["anti_loop_reset"] = True
                self.last_stage_timeout = "local_loop" if local_loop else "stationary_loop"
                reward_events.append(self.last_stage_timeout + ":truncate")
                self.anti_loop_resets += 1
                self.episode_anti_loop_resets += 1
            else:
                info["anti_loop_reset"] = False

            # Wiederholtes A an exakt derselben Stelle ist typischerweise ein
            # Buch/Regal/NPC-Loop. Die hohe Freigrenze laesst normale Dialoge
            # durch; Bewegung oder echter Fortschritt setzt den Zaehler zurueck.
            interaction_anchor = (
                bank,
                map_id,
                x,
                y,
                int(self.intro_complete_rewarded),
                int(self.stairs_down_rewarded),
                int(self.left_house_rewarded),
                int(self.has_starter),
                p_lvl,
                badges,
                in_battle,
            )
            if interaction_anchor != self.interaction_anchor:
                self.interaction_anchor = interaction_anchor
                self.interaction_count = 0

            if requested_action == 0 and in_battle == 0:
                self.interaction_count += 1
                if self.interaction_count > self.INTERACTION_SPAM_PENALTY_AFTER:
                    reward += self.INTERACTION_SPAM_PENALTY
                    reward_events.append(
                        "interaction_spam:"
                        f"{self.INTERACTION_SPAM_PENALTY:.1f}"
                    )
                if self.interaction_count >= self.INTERACTION_SPAM_RESET_AT:
                    truncated = True
                    self.last_stage_timeout = "interaction_spam"
                    info["last_stage_timeout"] = "interaction_spam"
                    info["interaction_spam_reset"] = True
                    self.anti_loop_resets += 1
                    self.episode_anti_loop_resets += 1
                else:
                    info["interaction_spam_reset"] = False
            else:
                info["interaction_spam_reset"] = False

            self.previous_valid_bank = bank
            self.previous_valid_map = map_id
        else:
            self.stuck_counter = 0
            self.last_progress_signature = None
            self.interaction_anchor = None
            self.interaction_count = 0
            info["anti_loop_reset"] = False
            info["interaction_spam_reset"] = False

        # V7.7: Starter-Rusher trainieren nur Beginning -> Starter.
        if (
            self._is_starter_rusher()
            and not self.has_starter
            and in_battle == 0
            and self.route_steps >= self.STARTER_RUSH_TIMEOUT
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
            and self.route_steps - self.last_progress_advance_step
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
            if not self.intro_complete_rewarded and self.route_steps >= self.FULL_INTRO_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_intro_cap"
                reward_events.append("full_intro_cap:truncate")
            elif self.intro_complete_rewarded and not self.stairs_down_rewarded and self.route_steps >= self.FULL_STAIRS_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_stairs_cap"
                reward_events.append("full_stairs_cap:truncate")
            elif self.stairs_down_rewarded and not self.left_house_rewarded and self.route_steps >= self.FULL_EXIT_STAGE_CAP:
                truncated = True
                self.last_stage_timeout = "full_exit_cap"
                reward_events.append("full_exit_cap:truncate")

        # V10.15: ten long full probes get a real 32k horizon.
        if self._is_long_full_probe() and self.route_steps >= self.LONG_FULL_PROBE_STEPS and not truncated:
            truncated = True
            self.last_stage_timeout = "long_full_32k"
            reward_events.append("long_full_32k:truncate")
        # V10.31/V15.3: hat den Starter, kommt aber seit 4000 Schritten nicht
        # aus dem Labor -> neu ansetzen (schnelle Iteration fuer genau diese
        # Passage). War auf die Rolle "starter" beschraenkt (Spezialisten-Ära);
        # im All-Full-Regime ist "full" die einzige Rolle, die das ueberhaupt
        # noch erlebt - ohne diese Erweiterung wandern Full-Agenten mit
        # Starter beliebig lange im Labor herum (kein Timeout griff mehr).
        if (
            self.training_objective in ("starter", "full", "progress")
            and self.has_starter
            and not self.starter_outdoor_rewarded
            and self.starter_obtained_step is not None
            and self.route_steps - self.starter_obtained_step >= 4000
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
            and self.route_steps >= specialist_timeout
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
            if self.route_steps >= _prog_cap:
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
        info["episode_steps"] = self.route_steps
        info["stage_arrival_steps"] = dict(self.stage_arrival_steps)
        info["route_steps"] = self.route_steps
        info["battle_steps"] = self.battle_steps
        info["ppo_episode_steps"] = self.total_steps
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
        info["has_target_starter"] = bool(self.has_target_starter)
        info["starter_species_id"] = int(self._starter_species())
        info["level"] = int(self.last_level)
        info["badges_count"] = int(self.last_badges)
        info["progress_schema"] = self.PROGRESS_SCHEMA
        info["world_stage"] = int(self._world_stage())
        info["frontier_maps"] = int(len(self.visited_maps))
        info["has_starter"] = bool(self.has_starter)
        info["level"] = int(self.last_level)
        info["badges_count"] = int(self.last_badges)
        info["outdoor_confirm_reads"] = self.outdoor_confirm_reads
        info["last_stage_timeout"] = self.last_stage_timeout

        if reward_events:
            self.recent_reward_events.extend(
                f"{self.route_steps}:{ev}" for ev in reward_events
            )
            self.recent_reward_events = self.recent_reward_events[-40:]

        # Auch sehr kurze Spezialisten-Episoden publizieren. Sonst bleibt im
        # Dashboard nach einem Neustart die alte Party/Rolle stehen, wenn der
        # neue Lauf schon vor dem regulaeren 80-Step-Punkt endet.
        if self.total_steps % 80 == 0 or truncated or objective_done:
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
                    "ram_valid": bool(loc.get("valid", False)),
                    "bank": bank,
                    "map": map_id,
                    "x": x,
                    "y": y,
                    "path": self.recent_path,
                    "room": f"Bank {bank} / Map {map_id}",
                    "steps": self.route_steps,
                    "route_steps": self.route_steps,
                    "battle_steps": self.battle_steps,
                    "ppo_episode_steps": self.total_steps,
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
                    "has_target_starter": bool(self.has_target_starter),
                    "starter_species_id": int(self._starter_species()),
                    "training_phase": "full_brain" if self.FULL_ONLY_MODE else (
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
                    "progress_schema": self.PROGRESS_SCHEMA,
                    "world_stage": int(self._world_stage()),
                    "story_progress": {
                        "viridian_mart_scene": int(self.viridian_mart_scene),
                        "pallet_oaks_lab_scene": int(self.pallet_oaks_lab_scene),
                        "viridian_old_man_scene": int(self.viridian_old_man_scene),
                    },
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
                    "reward_events": self.recent_reward_events,
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
            and self.route_steps >= self.PROGRESS_STALL_TIMEOUT
            and (
                self.route_steps - self.last_progress_advance_step
                >= self.PROGRESS_STALL_TIMEOUT
            )
        ):
            truncated = True
            info["progress_stall_reset"] = True
        else:
            info["progress_stall_reset"] = False

        episode_limit = (
            self.SCOUT_EPISODE_STEPS
            if self.training_objective == "scout"
            else self.MAX_EPISODE_STEPS
            if self.training_objective in ("progress", "badge", "full")
            else 32768
        )
        if (
            self.current_battle_steps >= self.MAX_SINGLE_BATTLE_STEPS
            or self.battle_steps >= self.MAX_EPISODE_BATTLE_STEPS
        ):
            truncated = True
            self.last_stage_timeout = "battle_step_cap"
            info["last_stage_timeout"] = "battle_step_cap"
            reward_events.append("battle_step_cap:truncate")

        terminated = bool(
            objective_done
            or self.route_steps >= episode_limit
        )

        # V15.3: das Post-Haus-Wander-Shaping (+0.35/Kachel, Loop-Strafen) ist
        # genau die "bewegt sich fuer kleine Scores in Alabastia"-Logik - im
        # All-Full-Regime aus.
        if not self.FULL_ONLY_MODE:
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
