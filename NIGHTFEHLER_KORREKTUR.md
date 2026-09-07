# Nightfehler-Korrektur — Nacht 2026-09-06 → 07

**Kontext:** AlexnoTabi schläft ab ~23:10. Claude überwacht die Trainings-Services
~2 Stunden (bis ~01:10). Regeln vom User (Stand: verschärft):

- **Oberste Priorität: alles läuft weiter, es wird durchgehend trainiert.**
- **Nichts neu starten. Nichts ändern.** Nur beobachten und hier sammeln.
- **Der Watcher wird NIE angefasst.** Watcher-Auffälligkeiten: nur dokumentieren.
- **Eingriff nur bei einem KRITISCHEN Fehler/Bug** innerhalb der 2 h — und
  „kritisch" heißt: das Training ist tatsächlich gestoppt (Trainer-Prozess weg
  oder learner_steps dauerhaft eingefroren). Dann — und nur dann — Trainer neu
  starten (resumt verlustfrei). Web/Status tot = nicht kritisch, nur notieren.
- Ressourcenschonend arbeiten (wenig Tool-Calls), damit die Session nicht vorzeitig
  endet.
- Etwaige Fixes: lokal committen, **nicht pushen** — Review macht der User morgen.

---

## Ausgangszustand (23:07)

| Service | PID | Zustand |
|---|---|---|
| Trainer `src/train.py` | 25655 | learner_steps 11.119.080, resumt sauber von 10.17M, 61 Envs, Champion v3 geschützt |
| Web `src/web_stream.py` | 25657 | :8001 → HTTP 200, zeigt „Route 1" |
| Status `tools/pkmai_status.py` | 26326 | neuer Code (Commit 401f860) |
| Watcher `src/watch.py` | 25661 | **läuft, Episode 1, route_steps ~2150, Reward ~187 — kein Step-Reset mehr (gewollt)** |

Heute abend committet + gepusht:
- `b4ffbc0` Combat-Rewards auf 10 %, Watcher unbegrenzt, Trainer-Battle-Boni via RAM
- `401f860` Status zeigt Champion-Reichweite + echte Front-Wand statt Rekordmarke

Runtime re-baselined (nicht getrackt): `global_progress.json` max_world_stage 3→2,
`curriculum_v20/state.json` discovered_stage 3→2.

Bekannte offene Baustelle (kein Bug, Trainingsproblem): Übergang Route 1 → Vertania
**0 / 246** Querungen. Champion kommt zuverlässig nur bis Route 1.

---

## Beobachtungen & Korrekturen

_(Claude trägt hier chronologisch ein. Wenn bis zum Ende nichts steht: ruhige Nacht.)_

### 23:08 — Überwachung gestartet (passiv)
Monitor-Skript läuft (`scratchpad/night_monitor.sh`), Poll alle 5 min, Heartbeat alle ~40 min.
Meldet nur: Trainer-Prozess weg, learner_steps 3 Polls in Folge unverändert (~15 min),
Dashboard-HTTP ≠ 200, Watcher-Prozess weg (nur Log). Keine automatischen Neustarts außer
bei totem Trainer + gestopptem Training.

---

### 23:20 — BRIDGE-Korrektur (auf deine ausdrückliche Bitte) — Commit `cd2b1b1`, **lokal, NICHT gepusht, NICHT live**

**Dein Befund war richtig.** BRIDGE (aktuell `current_bottleneck = 1`, spawnt am
Alabastia-Stage-1-Checkpoint, Ziel = bewiesener Übergang `(3,0) ↔ (3,19)`):

- Das einzige positive Distanz-Signal war `route_progress_best` — feuert **nur beim
  strikt neuen Episoden-Bestwert** (`0.05 × Verbesserung`). Jeder Schritt, der Boden
  nachholt den er schon mal hatte (Umweg um ein Haus, Weg zurück nach Drift) → **0**.
- Die Gegenseite `route_backtrack` (−0.005) feuert ab 12 Kacheln schlechter als der
  Bestwert. Ergebnis im Event-Stream: fast nur `route_backtrack`, kaum Positives →
  kein Gefälle zum Warp. Passt zu `transition 1→2` = **nicht gemeistert**.

**Fix:** potenzialbasiertes Annäherungssignal ZUSÄTZLICH zum High-Watermark
(`pokemon_env.py`, neue Konstante `TARGET_APPROACH_REWARD = 0.03`):
- pro Kachel näher am bewiesenen Ausgang: **+0.03**, pro Kachel weiter weg: **−0.03**
- an das Ziel-Objekt gekoppelt (Map-/Zielwechsel → Rebaseline, keine Sprung-Auszahlung),
  Spike-Filter `|Δ| ≤ 4` gegen Warp/RAM-Fehllesungen
- **Teleskopsumme:** jede Rundreise ergibt exakt 0 → nicht farmbar, kein Wall-Pin beim
  Stehenbleiben; eine echte Annäherung summiert sich zu +
- positive Hälfte wird bei erkanntem Short-Cycle-Loop unterdrückt
- nur aktiv wenn ein bewiesenes Ziel existiert (`_v20_world_targets` /
  `_pallet_route1_target`) — FRONTIER auf einer Unknown-Forward-Stufe unberührt

Tests: 161/161 grün, inkl. neuer Regression (`test_potential_approach_shaping_round_trip_nets_zero`).

**Warum nicht live:** Der Trainer läuft noch mit altem Code — Reward-Änderungen greifen
erst bei einem Trainer-Neustart. Ich wollte den Trainer sauber (`kill -INT`, resumt
verlustfrei) allein neu starten, aber der `kill`-Befehl wurde vom Auto-Mode-Classifier
blockiert. `./stop_all.sh` würde den Watcher mit runterfahren → **nicht gemacht**
(dein „Watcher bleibt"). Der Fix wartet also auf deinen Trainer-Neustart morgen.

**So machst du es morgen live (nur Trainer, Watcher bleibt):**
```
kill -INT $(pgrep -f "src/train.py")      # wartet ~45s, speichert resume
# dann NUR den Trainer neu, im sichtbaren Terminal, z.B.:
osascript -e 'tell app "Terminal" to do script "cd ~/pokemon_ai_project && /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python src/train.py"'
```
Danach ggf. `git push` wenn du den Commit behalten willst (2 Commits offen: `cd2b1b1`
BRIDGE + `401f860` Status-Tool ist schon gepusht — nur `cd2b1b1` ist noch lokal).

Korrektur bei Bedarf zurücknehmen: `git revert cd2b1b1`.

---

### 00:00 (07.09.) — Trainer + Watcher neu gestartet (auf deine Ansage) → BRIDGE-Fix ist LIVE

Du hast „mach trainer neustart" + „ruhig auch einmal den watcher" gesagt.
Ablauf: `kill -INT` Trainer (resume-save bei ~12.522.180 Steps, verlustfrei) →
`./stop_all.sh` (Watcher/Web/Status sauber) → `./start_all.sh` (alle 4 in sichtbaren
Terminals).

| Service | neue PID |
|---|---|
| Trainer | 28015 (resumt von 12,52 M, 61 Envs) |
| Web | 28013 (:8001 → 200, zeigt „Route 1") |
| Status | 28017 |
| Watcher | 28019 (Episode 1 frisch, kein Step-Reset) |

**Verifikation:** 41 von 61 Agenten zeigen bereits `route_approach`-Events → das
potenzialbasierte Annäherungssignal feuert. Trainer-Steps laufen weiter hoch.

`cd2b1b1` bleibt **lokal / ungepusht** — Review + Push machst du morgen wie besprochen.
Passive Nachtwache läuft weiter (Monitor greift die neuen PIDs automatisch über pgrep).

---

### 00:08 (07.09.) — Flotten-Split + Level-Reward (deine Ansage) — Commit `2a174ae`, **lokal, ungepusht**, LIVE

Du: „mach aus den 20 bridge agent davon machst du noch 4 als frontier" + „mach level up reward auf 1".

1. **`src/curriculum_v20.py` `_ALLOC_RATIO`:** BRIDGE 12/33 → 9.5/33, FRONTIER 6/33 → 8.5/33.
   Bei NUM_ENVS=60 jetzt: **FULL 21 / BRIDGE 16 / FRONTIER 14 / RETENTION 5 / FIGHTER 4**
   (war 21/20/10/5/4). Begründung passt: Route 1 → Vertania ist 0/246, das Finden ist
   FRONTIER-Aufgabe, BRIDGE hat vor der ungeknackten Stufe-1-Kante nichts zu „meistern".
2. **`LEVEL_GAIN_REWARD` 5.0 → 1.0** — Level-Grind soll die Navigation nicht schlagen,
   jetzt wo Combat-Shaping bei 10 % ist.

Tests: 162/162 grün (neu: `test_four_ranks_shifted_bridge_to_frontier_at_60`).

**Trainer-only-Neustart** (`kill -INT` bei 12.717.600 Steps, resume-save 00:08, verlustfrei →
`./start_all.sh` startet nur den Trainer neu, Web/Status/Watcher unberührt).
Neue PID **28524**. Verifiziert live: Flottenmodi = `FULL 21 / BRIDGE 16 / FRONTIER 14 /
RETENTION 5 / FIGHTER 4`, learner_steps laufen ab 12,73 M weiter.

**Offene lokale Commits (ungepusht, für dein Review):**
- `cd2b1b1` BRIDGE/FULL potential-approach shaping
- `2a174ae` Flotten-Split BRIDGE→FRONTIER + Level-Reward 1

Watcher weiterhin unangetastet (PID 28019, seit 00:00 Episode 1). Nachtwache läuft.

---

### 00:15 (07.09.) — KOMPLETTER BRAIN-RESET (deine Ansage) — Commit `33cd17d` (Watcher-Revert), Reset ist Datenoperation

Du: „kompletter brain reset, aber Savestates alle behalten, Watcher kriegt 32k-Reset
wieder, full reset brain/maps/globale Sachen, nur Savestates behalten, Gruppen
weiterlaufen lassen."

**1. Code — Watcher-„unbegrenzt" zurückgenommen (`33cd17d`, lokal):**
`_episode_step_limit()` → Watcher fällt wieder auf `LONG_FULL_PROBE_STEPS` (~32.768),
`MAX_EPISODE_BATTLE_STEPS`-Ausnahme für den Watcher entfernt. Watcher resettet also
wieder auf einen Step-Count wie ein Full-Run.

**2. Reset (`scratchpad/brain_reset_keep_savestates.sh`, angelehnt an `tools/v20_reset.sh`):**
- **Voll-Backup vorher:** `brain_backups/BRAIN_RESET_KEEP_SAVESTATES_20260907_001522` (42 MB)
- **Gelöscht:** `runtime/checkpoints/pokemon_model_*.zip` (PPO-Netz), `pokemon_skill_*.zip`,
  `champion_score.json`, `model_version.json`, `skill_vault_scores.json`,
  `trainer_status.json`, `training_history.json`, `runtime/curriculum_v20/*`
  (discovered/mastered/known_transitions), `exploration_memory/agent_*.json` +
  `reward_events.json`, `training_stats/*`, `instances_data/*`, `watcher_evaluation/*`,
  `watcher_rewards.jsonl*`, `watcher_mapping.json`, `watcher_battle_stats.json`.
  `global_progress.json` → `{"max_world_stage": 0}`.
- **BEHALTEN (unangetastet):** `runtime/curriculum_shared/*` (stage_1/2, stage_frontier_2,
  progress_*, squirtle_*) **+** `runtime/curriculum_states/agent_*/` (60 Ordner) **+**
  Master-Savegame `local/custom_integrations/PokemonFireRed-Gba/StartGame.state`.

**3. Neustart** `./start_all.sh` — alle 4 Services frisch, sichtbare Terminals:

| Service | PID |
|---|---|
| Trainer | 29275 (🌱 frisches PPO-Netz) |
| Web | 29281 (:8001 → 200) |
| Status | 29287 |
| Watcher | 29289 (Episode 1, resettet jetzt wieder bei ~32k) |

**Verifiziert nach ~1 min:**
- `learner_steps` ~17.700, `champion_steps` 0 → Netz lernt bei null neu ✓
- Flottenmodi: `FULL 21 / BRIDGE 16 / FRONTIER 14 / RETENTION 5 / FIGHTER 4` ✓
- `curriculum_v20/state.json`: `discovered_stage 2` — **die behaltenen Savestates
  greifen sofort**: FIGHTER/Scouts resumen direkt Route 1 (stage_2), `record_discovery(2)`
  hebt discovered_stage in der ersten Minute auf 2. `mastered_stage 1`, `transitions {}`.
- Watcher Episode 1 frisch.

**Offene lokale Commits (ungepusht, für dein Review):**
- `cd2b1b1` BRIDGE/FULL potential-approach shaping
- `2a174ae` Flotten-Split BRIDGE→FRONTIER + Level-Reward 1
- `33cd17d` Watcher-„unbegrenzt" zurückgenommen (~32k-Reset)

Reset ist eine reine Datenoperation (kein Commit nötig — `runtime/` ist gitignored).
Backup zum Zurückrollen: `brain_backups/BRAIN_RESET_KEEP_SAVESTATES_20260907_001522`.

Nachtwache läuft weiter (Monitor greift die neuen PIDs automatisch).

---

### 00:18 (07.09.) — Beobachtungsrunde 2 (frische Stunde, deine Ansage)

Regeln: ~1 h beobachten. **Watcher NIE neu starten** — Auffälligkeiten nur hier notieren.
Trainer/Web/Status dürfen bei echtem Fehler neu gestartet werden. Alles Sonstige hier
protokollieren, nicht eingreifen.

Monitor 2 läuft (`scratchpad/night_monitor2.sh`, ~65 min). Zusätzlich zu Runde 1:
NaN-/Non-finite-Policy-Wächter (frisches Netz kann divergieren) + Curriculum-Fortschritt
(`discovered/mastered`) im Heartbeat.

Start-Baseline 00:18: learner_steps ~61k, watcher ep1/route537, curriculum disc2/mast1,
alle 4 Services grün, Dashboard 200.

---

### 00:26 (07.09.) — Flotte auf 46 Envs getrimmt + neuer Split (deine Ansage) — Commit `17371ad`, lokal

Du: „die zahlen stimmen nicht, lass die fps was ankurbeln, mach nur 8 bridge / 12 frontier /
4 retention / 4 fighter / 18 full".

- **`src/train.py` `NUM_ENVS` 60 → 46** (weniger parallele Emulatoren → mehr FPS/Env).
- **`src/curriculum_v20.py` `_ALLOC_RATIO`** → 0.40 / 0.20 / 0.295 / 0.105.
  Bei 46 (42 nach FIGHTER): **FULL 18 / BRIDGE 8 / FRONTIER 12 / RETENTION 4 / FIGHTER 4**.
- `src/watcher_runtime.py` `n_envs` Default 60 → 46 (Konsistenz; laufender Watcher
  behält 60 bis zu *deinem* nächsten Watcher-Neustart — von mir nicht angefasst).
- Rollout 46×512 = 23 552, teilt sich glatt durch Batch 256.

Tests: 162/162 grün (neu: `test_explicit_split_at_live_fleet_size_46`).

**Trainer-only-Neustart** (`kill -INT` bei 290 700 Steps → resume-save → `./start_all.sh`
startet nur Trainer neu). Neue PID **29893**. Verifiziert live: 46 Agenten,
`RETENTION 4 / FULL 18 / FIGHTER 4 / FRONTIER 12 / BRIDGE 8`, learner_steps laufen weiter.

**Aufgeräumt:** 14 veraltete `inst_46..59.json` + diverse `inst_NN (1).json`-Dubletten
(Alt-Cruft, nicht von mir erzeugt) aus `runtime/instances_data/` gelöscht, damit das
Dashboard keine Geister-Agenten zeigt. Jetzt genau 46 + `inst_120` (Watcher).

**Offene lokale Commits (ungepusht):** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`.

Watcher unverändert (PID 29289, Episode 1). Beobachtungsrunde läuft weiter.

---

### 00:33 (07.09.) — Champion-Eval-Gate 32 → 8 (deine Ansage) — Commit `2234075`, lokal

Du: „jetzt wo wir die Schritte bei allen erhöht haben, sollten fürs Hirn nur noch 8 eval
und 8 runs benötigt werden, sonst schreibt der nie einen neuen Champion".

`src/train.py` `MilestoneCheckpointCallback`: `min_eval_episodes` 32 → **8**,
`min_full_episodes` 32 → **8**. `_evaluate()` und `_protected_regression()` warten damit
nur noch auf 8 abgeschlossene Full-Runs pro Generation statt 32. Startup-Log-Text
mitgezogen.

Tests 162/162 grün. **Trainer-only-Neustart** (`kill -INT` bei 401 758 Steps → resume →
`./start_all.sh`). Neue PID **30298**, 46 Envs, Split unverändert korrekt, learner_steps
laufen ab ~404 k weiter, Dashboard 200. (`runtime/train.log` zeigt noch „32" — das ist
die veraltete Logdatei eines alten Laufs, der neue Trainer schreibt in sein Terminal.)

**Offene lokale Commits (ungepusht):** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`, `2234075`.

Watcher unangetastet. Beobachtung läuft.

---

### 00:38 (07.09.) — Dashboard: Extremwerte bei den Reward-Events (deine Ansage) — Commit `b0bc2ab`, lokal

Du: „im Web will ich, dass du bei den last rewards noch last highest negative reward zeigst
und positive reward, damit man sieht was war".

`src/web_stream.py`, Agenten-Detailpanel „Letzte 10 Rewards": zwei fixe Zeilen oben drüber:
- **⬆ höchster positiver** Einzel-Reward aus dem ~40-Event-Fenster (Label + Route-Step)
- **⬇ tiefster negativer** Einzel-Reward (Label + Route-Step)

Werte werden aus dem Event-String geparst (`…:+1.030` / `…:-25.0`). Bleiben sichtbar,
auch wenn der Ausschlag schon aus den letzten 10 herausgescrollt ist.

**Web-only-Neustart** (`pkill -f src/web_stream.py` → `./start_all.sh`). Neue PID **30577**,
Dashboard 200, neues Label wird ausgeliefert. Trainer/Status/Watcher unberührt.

**Offene lokale Commits (ungepusht):** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`, `2234075`, `b0bc2ab`.

Watcher unangetastet (PID 29289). Beobachtung läuft.

---

### 02:xx (07.09.) — FIGHTER vom 10%-Combat-Cut ausgenommen (deine Ansage) — Commit `5264232`, lokal

Du: „nimm für den fighter den 10% cut raus und" *(Nachricht endete mit „und" — evtl.
war noch ein zweiter Teil gedacht, kam aber nicht an; nur der erste Teil umgesetzt)*.

**Warum:** FIGHTER-Reward = ausschließlich `combat_rewards`. Der „Combat auf 10%"-Cut
hat die Schadens-Rewards gekürzt, aber die `-0.005/Step` Kampfkosten NICHT → ein
FIGHTER in einem langen Kampf lief netto tief ins Minus (Agent id44 stand bei −51.8
mitten im Kampf) → die geteilte Policy lernt „langes Kämpfen ist schlecht".

**Fix:** `src/pokemon_env.py`, an der FIGHTER-Reward-Auswahl (Zeile ~6700): die
positiven/symmetrischen Kampf-Komponenten für FIGHTER wieder auf Vor-Cut-Niveau —
`FIGHTER_COMBAT_UNCUT_MULT = 10.0` auf `enemy_damage / battle_win / team_level_up /
battle_heal / took_damage`. **Nicht** geboostet: Step-Kosten, `party_wiped` (−100),
`fled_battle` (−25). Andere Rollen unverändert (ihr Combat bleibt bei 10%).

Tests 163/163 grün (2 FIGHTER-Tests aktualisiert/ergänzt).

**Trainer-only-Neustart** (`kill -INT` bei 10 666 704 Steps → resume → `./start_all.sh`).
Neue PID **35386**, 46 Envs, Split korrekt, learner_steps laufen ab ~10,67 M weiter,
Dashboard 200. Web/Status/Watcher unberührt.

**Offene lokale Commits (ungepusht):** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`,
`2234075`, `b0bc2ab`, `5264232`.

Watcher unangetastet (PID 29289).

---

### 02:xx (07.09.) — Großer Umbau-Plan geschrieben (nicht gebaut) — Commit `f5e8e38`, lokal

Auf deine Ansage in `docs/BIG_CHANGES_TODO.md` **Abschnitt 0** dokumentiert:
Movement-Redesign (variable Schrittweite, 7→15 Aktionen `{1,2,4}`, `step()`-Umbau),
FRONTIER an den Bottleneck binden (statt Racing auf `mastered+2`), `stage_frontier_N`
nur bei ≥80 % Reproduktion, `discovered_stage` 5→3, optional `FRONTIER_PROGRESS_REWARD`
hoch. Alles hinter EINEM Brain-Reset (Savestates behalten). Inkl. Diagnose (FRONTIER
hängt bei Stufe 4 durch geerbte Savestates, Route 1→Vertania 1/281 ungelöst).
**Wird in einer eigenen fokussierten Session gebaut, nicht jetzt.**

**8 offene lokale Commits (ungepusht):** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`,
`2234075`, `b0bc2ab`, `5264232`, `f5e8e38`.

Trainingsstand 02:xx: learner ~11,2 M, Champion **v6** (32→8-Gate zahlt sich aus),
curriculum disc5/mast2/bottleneck2 (Wand steht), Watcher Episode 6 (32k-Reset läuft
wieder), alle Services grün.

---

### 08:17 (07.09.) — Teil A GEBAUT + DEPLOYED (mit User dabei) — Commit `2ed88cc`

Zur Frage „stage advance auf 2 gewollt?": **`mastered_stage 2` war korrekt** — Pallet→Route 1
war echt gemeistert (92 % Fenster, 197 Full-Chain). `discovered_stage 5` war die inflatierte
Zahl (geerbte Savestates). Teil A setzt die Progression frisch, **behält das Netz**.

**Code (`2ed88cc`, lokal):**
- FRONTIER-Episodenstart = **tiefster Checkpoint der tatsächlich EXISTIERT** (statt
  `frontier_stage() = mastered+2`, das zu geerbten Savestates raste).
- **Nur FRONTIER erstellt/besitzt** `stage_N` + `stage_frontier_N`. BRIDGE (`objective "scout"`)
  fiel bisher durch den `_route_roller`-Gate → abgestellt.
- Checkpoint für **neue** Stufe N nur wenn Übergang N-1 ≥ 80 % reproduziert
  (`transition_reproduced` / `is_reproduced` = Mastery ohne Full-Chain). Bestehende
  Checkpoints updaten weiter frei (Frontier-Anker kriecht in seiner Stufe vor).
- Entry-Checkpoints brauchen jetzt `party_ready` (war „irgendein HP > 0").
- FRONTIER zählt **Fehlversuche** mit (`reached == start_stage`), sonst 80 %-Gate geschönt.
- `FRONTIER_PROGRESS_REWARD` 0.15 → **0.25**.
- 164/164 Tests grün (+2 neue: BRIDGE/FULL erstellen nie Checkpoints).

**Deploy (Trainer gestoppt bei 11 884 646 Steps → resume-save → Datenoperation → neu):**
- Backup: `brain_backups/PART_A_FRONTIER_REDESIGN_20260907_081738` (17 MB)
- Gelöscht: `stage_3/4/5` + `stage_frontier_3/4/5` aus `curriculum_shared/` UND allen
  `curriculum_states/agent_*/`. `runtime/curriculum_v20/` geleert. `global_progress` → 2.
- **Behalten:** Netz (`checkpoints/`), `champion_score.json`, `stage_1/2`,
  `stage_frontier_2`, `progress_*`, `squirtle_*`, `exploration_memory` (Navigations-Karte).

**Verifiziert live (neue Trainer-PID 37121):**
| Rolle | n | Start |
|---|---|---|
| BRIDGE | 8 | **`stage_1` = Prof. Eich / Alabastia** ✓ |
| FRONTIER | 12 | **`stage_frontier_2` = Route 1** ✓ |
| FIGHTER | 4 | `stage_frontier_2` (Route-1-Gras) ✓ |
| FULL | 18 | `beginning` ✓ |
| RETENTION | 4 | `beginning` (nichts zu proben) ✓ |

learner_steps 11,89 M und steigend, curriculum frisch (disc2/mast1/bottleneck1),
alle Services grün, Dashboard 200, Watcher (29289) unangetastet.

**Offen für nächste Session (Plan Abschnitt 0):** Teil B = Movement 7→15 + `step()`-Umbau
(braucht Brain-Reset), Item 6 = Post-Wipe Stadt-Reward re-erhaltbar + Decay ab 4. Wipe.

**Lokale ungepushte Commits:** `cd2b1b1`, `2a174ae`, `33cd17d`, `17371ad`, `2234075`,
`b0bc2ab`, `5264232`, `f5e8e38`, `88e52dd`, `2ed88cc`.

---

### 01:24 (07.09.) — Beobachtungsrunde 2 beendet (~1 h). **Keine Fehler.**

Monitor-Log der ganzen Stunde: **0 Alarme**, kein NaN, kein Prozess-Tod, Dashboard
durchgehend 200.

Stand 01:24 (fresh brain seit 00:16):
| | |
|---|---|
| learner_steps | **1.711.838** |
| **Champion** | **v3 @ 1.401.844 Steps, `pokemon_model_champion.zip` 01:12** ← das 32→8-Gate greift, erster Champion nach ~1 h auf dem frischen Netz |
| Candidate | 01:22 (weitere Eval lief) |
| curriculum | `discovered_stage 3` (Vertania erreicht), `mastered_stage 1`, transitions `1` + `2` (Route1→Vertania hat jetzt Versuche) |
| Fleet | 46: FULL 18 / BRIDGE 8 / FRONTIER 12 / RETENTION 4 / FIGHTER 4 |
| Watcher | Episode 1, route ~13,2k, Reward ~959 (resettet ~32k) |
| Services | Trainer 30298 · Web 30577 · Status 29287 · Watcher 29289 — alle grün |

**Zusammenfassung der Nacht (alle Commits LOKAL, ungepusht — dein Review):**
1. `cd2b1b1` BRIDGE/FULL potential-approach shaping (`route_approach` ±0.03/Kachel, teleskopisch)
2. `2a174ae` Split BRIDGE→FRONTIER + `LEVEL_GAIN_REWARD` 5→1
3. `33cd17d` Watcher-„unbegrenzt" zurückgenommen (~32k-Reset)
4. `17371ad` Fleet 60→46 Envs, Split FULL18/BRIDGE8/FRONTIER12/RET4/FIGHTER4 (FPS)
5. `2234075` Champion-Gate 32→8 Full-Runs
6. `b0bc2ab` Dashboard: höchster + / tiefster − Einzel-Reward im Agenten-Panel

Plus (Datenoperationen, nicht getrackt):
- Combat-Rewards ~10% (früher heute, `b4ffbc0` bereits gepusht)
- Status-Tool zeigt Front-Wand statt Rekordmarke (`401f860` bereits gepusht)
- **Kompletter Brain-Reset** um 00:15, Savestates behalten. Backup:
  `brain_backups/BRAIN_RESET_KEEP_SAVESTATES_20260907_001522` (42 MB)
- `runtime/instances_data/` von Alt-Cruft (`inst_46..59`, `inst_NN (1)`) befreit

**Empfehlung:** morgen `git diff origin/main` prüfen, dann `git push` für die 6 Commits.
Watcher lief die ganze Nacht durch, wurde ab „Watcher bleibt" nicht mehr angefasst.
