# Pokemon FireRed AI – PROJECT STATUS / TODO

**Projekt:** Stable-Retro / Gym-Retro + PPO – Pokemon FireRed AI auf macOS  
**Arbeitsordner:** `~/pokemon_ai_project`  
**Python:** `/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python`  
**Letzter gepflegter Stand:** 2026-09-02

---

## 🆕 OFFENE TODOs AUS DER V17-SESSION (2026-09-05, nachts) — ZUERST LESEN

Diese Punkte sind bewusst NICHT mehr in derselben Session umgesetzt worden,
damit der frische V17-Trainingslauf (Savestate-Start ab Oaks Labor, dauerhafter
Kanten-Reward, Kampf-Rebalance, Reproduzierbarkeits-Schwelle - siehe README
"Current release") ungestoert ueber Nacht laufen kann. Der Rest dieser Datei
ist AELTER (Stand 2026-09-02, 30-Agenten-Architektur) und teilweise ueberholt -
im Zweifel zaehlt README.md + der aktuelle Code, nicht der Rest dieser Datei.

- 🔴 **Item-/Pokeball-Pickup-Reward:** Herumliegende Items/Pokebaelle (z.B. am
  Wegesrand) sollen beim Aufsammeln einen Reward geben (~+15 vorgeschlagen).
  Noch nicht recherchiert, ob/wie sich Item-Pickup zuverlaessig per RAM
  erkennen laesst (aehnliches Vorsichtsprinzip wie beim Geld-RAM-Bug unten:
  erst empirisch verifizieren, nicht blind eine Adresse raten).
- 🔴 **Pokeball-Kauf-Reward im Pokemarkt** (nur wenn Geld > 500 Pokedollar,
  gegen Kaufen-bis-pleite-Loop): Geld-RAM-Adresse fuer die DEUTSCHE FireRed-
  ROM ist noch nicht verifiziert - die extern recherchierte Adresse
  (0x02023CC4 bzw. 0x0202494C fuer US) hat sich im Test als falsch erwiesen
  (kompletter Bereich nur Nullen; auch eine Volltextsuche nach exakt 3000 im
  RAM ergab 0 Treffer). Ausgelagert als Hintergrund-Task `task_3e5cd4bf`
  ("Finde und verifiziere Geld-RAM-Adresse (deutsches FireRed)"), Status beim
  Fortsetzen pruefen.
- 🔴 **Watcher: Schiggy-Sprite fehlt.** Das Team-Panel im Watcher-Overlay zeigt
  aktuell nur Name+Level+HP-Balken als Text, kein echtes Pokemon-Sprite-Bild.
  Sollte durch ein echtes Sprite ersetzt/ergaenzt werden (Sprite-Quelle noch
  zu klaeren - ROM-intern extrahieren oder Asset-Datei).
- 🔴 **Watcher-Overlay: eigene Reward-Logik hat vermutlich denselben
  Savestate-Start-Bug wie pokemon_env.py hatte** (intro_complete_rewarded/
  left_house_rewarded/starter_obtained_step in src/watch.py, siehe Zeilen um
  1025-1970). Betrifft NUR die sichtbare Watcher-Anzeige, NICHT das
  eigentliche Training (train.py/pokemon_env.py sind bereits gefixt und
  laufen unabhaengig vom Watcher) - deshalb bewusst nicht mehr in dieser
  Session angefasst. Sollte aber denselben Fix bekommen wie pokemon_env.py:
  diese Flags am Watcher-Episodenstart auf "bereits erledigt" setzen statt
  auf False, sonst resettet der sichtbare Lauf vermutlich immer wieder frueh.
- 🟡 **Dashboard-Feinschliff (angefangen, nicht abgeschlossen):**
  - Watcher-Overlay-Titel "LIVE MAP TILING" -> "LiveMap" umbenannt und ueber
    die tatsaechliche Kartenflaeche zentriert (MAP_X0 statt GAME_PANEL_W+18) -
    ERLEDIGT in src/watch.py.
  - Gruener Header-Text gekuerzt (kein "Pokemon FireRed AI by Alex" mehr am
    Ende, das verursachte die Ueberlappung mit "LIVE MAP TILING") -
    ERLEDIGT, jetzt "Live Brain v{version} | Learner ... | Nx nachgeladen".
  - "Hintergrund etwas modernere Farben, aktuell zu braun" - NICHT
    umgesetzt, unklar wo genau (Web-Dashboard oder Watcher-Canvas?). Mit User
    klaeren, wo genau das braun auftaucht, bevor Farben geaendert werden.
  - "Team-Panel huebscher, Pokemon-Sprites, ggf. Status-Werte" - NICHT
    umgesetzt, haengt am Sprite-Punkt oben.
  - Countdown "noch X Steps bis Timeout" pro laufendem Agenten - NICHT
    umgesetzt, komplett neues Feature.
  - Trainer-Sprite-Pfad-Nachbau der ersten Agenten + Spielernamen fuer
    spaeteres Online-Training + feste (nicht zoombare) Kartengroesse -
    explizit vom User als spaetere Baustelle markiert, nicht jetzt.

---

> Diese Datei ist die zentrale Übergabe zwischen ChatGPT, Claude und Gemini.
> Bitte **vor Änderungen zuerst diese Datei lesen** und anschließend die aktuell
> vorhandenen Projektdateien prüfen. Keine Änderung als "eingebaut" bezeichnen,
> solange sie nicht wirklich in der Projektdatei/Drive-Datei verifiziert wurde.

## Status-Legende

- **✅ ERLEDIGT** – umgesetzt und praktisch bestätigt
- **🟡 BEOBACHTEN** – grundsätzlich umgesetzt, aber noch weiter prüfen
- **🔴 OFFEN** – noch zu lösen
- **🧪 VORBEREITET** – Patch/Datei existiert, aber noch nicht sicher im echten Projektstand verifiziert
- **ℹ️ INFO** – Architektur-/Hintergrundinformation

---


## 1. Hauptziel

30 PPO-Trainingsagenten sollen Pokemon FireRed möglichst autonom lernen und
langfristig durchspielen.

Parallel läuft ein **Watcher**, der immer das neueste gemeinsame PPO-Modell lädt,
das Spiel sichtbar ausführt und die Weltkarte optisch aufbaut.

Wichtig:

- Die 30 Agents trainieren **ein gemeinsames PPO-Modell**.
- Der Watcher trainiert nicht.
- Modellversionen wie `v00357` sind nur Save-/Versionsnummern, keine direkten
  "Intelligenzstufen".
- Zuletzt beobachtet: ungefähr **16 Mio. Trainingssteps**, Modell ungefähr
  **v00357**.
- Der User möchte das aktuelle Gehirn grundsätzlich weitertrainieren und
  **nicht ohne ausdrücklichen Wunsch resetten**.

---

## 2. Unverhandelbare Regeln

### Mapping
- **Nur der Watcher darf echte Screenshot-/Tile-Map-Bilder schreiben.**
- Trainingsagenten dürfen **keine Screenshots mappen**, keine Tile-PNGs schreiben
  und keinen zusätzlichen OpenCV-Map-Aufwand bekommen.
- Trainingsagenten dürfen ihre ohnehin vorhandene `(bank, map, x, y)`-Telemetrie
  liefern.
- Ein separater Skeleton-Mapper darf daraus reine Map-Kästen/Bounds/Transitions
  bauen, solange dadurch praktisch keine FPS verloren gehen.

### Performance
- Keine dauernden RAM-Vollscans.
- Keine Änderungen, die die Trainingsleistung wieder von ~1000+ FPS auf ~60 FPS
  drücken.
- Teure Watcher-Map-Arbeit darf den Input-/Emulatorloop nicht blockieren.

### Startup
- **KEIN TIMER**.
- User beendet Training/Server selbst.
- Bei normalem Neustart Modell, Checkpoints und Curriculum erhalten.
- Kein Fresh-Brain-Reset ohne explizite Zustimmung.

---

## 2.1 Aktueller Kurzstatus

| Bereich | Status | Aktueller Stand |
|---|---|---|
| Watcher FPS Intro/Menu | ✅ ERLEDIGT | konstant ca. 120–150 FPS |
| Watcher FPS im Spiel | ✅ ERLEDIGT | ca. 120–150 FPS |
| Intro-Rewards | 🟡 BEOBACHTEN | Intro wird inzwischen deutlich schneller überwunden |
| Haus verlassen | 🔴 OFFEN | Agents hängen noch häufig bei F2/F1 bzw. Treppen-Loops |
| Reward-Statistik | 🟡 BEOBACHTEN | Grundstruktur vorhanden; sinnvoll auswerten und History sichern |
| Graph-Dashboard | 🧪 VORBEREITET | Integration/Drive-Stand noch verifizieren |
| Skeleton Map | 🧪 VORBEREITET | soll ohne zusätzliche Agent-FPS-Kosten arbeiten |
| Watcher Input-Anzeige | 🧪 VORBEREITET | mit aktuellem Watcher-Fix zusammenführen |
| Watcher minutenlang ohne Input | 🔴 OFFEN | Ursache weiter prüfen; Mapping/Hot-Reload verdächtig |
| Tile-Map-Ausrichtung | 🔴 OFFEN | feste Kameraannahme `(7,5)` weiterhin problematisch |
| Fresh Reset | ✅ ERLEDIGT / VERMEIDEN | kein weiterer Brain-Reset ohne ausdrücklichen Wunsch |
| Timer im Startup | ✅ ERLEDIGT | kein Timer erwünscht |

---

## 3. Architektur

### `train.py`
- `NUM_ENVS = 30`
- Shared PPO `CnnPolicy`
- ungefähr:
  - learning rate: `0.0003`
  - n_steps: `128`
  - batch_size: `256`
  - n_epochs: `4`
  - gamma: `0.995`
  - ent_coef: `0.02`
- lädt `checkpoints/pokemon_model_latest.zip`, falls vorhanden
- zyklischer Save ungefähr alle 50k Steps
- Curriculum:
  - ca. 65 % Beginn
  - ca. 35 % gespeicherte Milestones

### `pokemon_env.py`
Training Environment:
- 8 Emulatorframes je Agentenaktion:
  - 4 aktive
  - 4 Ruheframes
- Location wird nicht in jedem Step gelesen.
- Dynamic SaveBlock RAM via `firered_ram.py`
- Maps/Screenshots werden hier NICHT geschrieben.

### `watch.py`
- sichtbarer Emulator
- lädt neueste PPO-Version
- ungefähr 300 Emulator-FPS Ziel
- GUI ca. 50 FPS
- baut echte Tile-Map
- schreibt Watcher-Telemetrie `inst_99.json`

### `web_stream.py`
- FastAPI
- Main Map / Agentenanzeige
- Web-Dashboard
- ngrok-Link:
  `https://thesaurus-gladiator-zoology.ngrok-free.dev/`

---

## 4. Dynamic FireRed RAM

Feste FireRed-SaveBlock-Adressen sind nicht zuverlässig, weil SaveBlock1
dynamisch liegen kann.

Bekannte Annahmen:

- `gSaveBlock1Ptr` Bus: `0x03005008`
- kombinierter Stable-Retro RAM-Slot: `0x45008`
- IWRAM-only Slot: `0x5008`
- SaveBlock1:
  - `+0` x
  - `+2` y
  - `+4` mapGroup
  - `+5` mapNum

`firered_ram.py` versucht bekannte Pointer-Slots und kann bei Bedarf scannen.

### WICHTIGER PERFORMANCE-PUNKT
Vollscan nicht ständig durchführen.

Training sollte Pointer-Discovery nur selten/staggered machen.

Frühere problematische Logik:
```python
total_steps % LOCATION_DISCOVERY_EVERY == (
    rank * 17
) % LOCATION_DISCOVERY_EVERY
```

Das ist bei `LOCATION_READ_EVERY=4` für viele Ranks mathematisch ungünstig.

Besser:
```python
slots = LOCATION_DISCOVERY_EVERY // LOCATION_READ_EVERY

discovery_slot = (rank * 17) % slots

allow_scan = (
    (total_steps // LOCATION_READ_EVERY) % slots
    == discovery_slot
)
```

Diesen Punkt bei der nächsten Änderung unbedingt prüfen.

---

## 5. Reward-System – aktueller verifizierter Stand

Der zuletzt tatsächlich aus Drive gelesene `pokemon_env.py` enthielt:

### Intro
- bei echter Gameplay-Position Basisreward: `+0.01`
- Intro visuelle Novelty:
  - deutlicher neuer Screen: ungefähr `+0.5`
  - großer Screenwechsel: ungefähr `+1.0`
  - Gesamt-Novelty gedeckelt auf `+25`
- Intro vollständig geschafft / erste echte Gameplay-Position:
  - `+35`
- lang derselbe Intro-Screen:
  - ab ~120 Steps kleine Strafe
  - ab ~300 stärkere Strafe
  - ab ~900 Intro-Loop Truncate

### Early Game / Story
Im verifizierten Drive-Stand:
- erster Indoor-Mapwechsel: `+25`
- Haus verlassen (Indoor -> Overworld Bank 3): `+52`
- Richtung Norden:
  - `+2` pro neuem Nord-Tile
  - max. 12 Tiles / `+24`
- Gras-/Nord-Meilenstein: `+75`
- nächste Overworld-Map: `+150`
- erster Pokemon/Starter: `+100`
- Level-Up: `+25` pro Level
- Badge: `+500`
- neue Map: `+10`
- neues Feld: `+0.25`

### Problem beobachtet
Agents zeigen häufig:
**Treppe runter -> Treppe hoch -> Treppe runter -> ...**

Das deutet darauf hin, dass der Indoor-Mapwechsel derzeit zu attraktiv ist und
das Haus-Verlassen zu weit entfernt / zu schwach belohnt wird.

---

## 6. House-Reward-Patch — 🧪 VORBEREITET / DRIVE-STAND PRÜFEN

Es wurde lokal ein neuer Patch vorbereitet und syntaxgeprüft:

`pokemon_env_house_reward_patch.py`

Geplante Änderungen:

- Treppe runter / erster Indoor-Wechsel: `+40`
- zurück ins Startzimmer vor Haus-Ausgang: `-15`
- Haus verlassen: `+150`
- erster Schritt vom Hauseingang weg: `+20`
- neue Tiles im Haus: `+0.10`
- neue Tiles draußen: `+0.25`
- allgemeiner `new map +10` erst **nach** Haus-Verlassen
- Starter: `+150`

### TODO
**Diesen Patch erst in das echte `pokemon_env.py` übernehmen und danach die
Drive-/Projektdatei erneut lesen/verifizieren.**

Nicht behaupten, dass dieser Patch aktiv ist, bevor die echte Datei geprüft wurde.

---

## 7. Reward-Statistik

In `pokemon_env.py` gibt es bereits persistente Zähler wie:

- `intro_state`
- `intro_complete`
- `indoor_progress`
- `left_house`
- `north_progress`
- `north_to_grass`
- `first_pokemon`
- `next_outdoor_map`
- `level_up`
- `badge`
- Anti-Loop-Resets
- completed episodes
- average episode reward
- best episode reward

Diese werden in `reward_stats` der Agent-Telemetrie geschrieben.

### So soll die Statistik interpretiert werden

Nicht nur Gesamt-Reward anschauen.

Beispiel:
```text
Intro fertig      90 %
Treppe/F1         80 %
Haus verlassen     5 %
Gras               1 %
Starter             0 %
```

=> Problem liegt klar im Erdgeschoss / Haustürbereich.

Guter Lernverlauf:
```text
Haus:      20 % -> 45 % -> 75 %
Gras:       5 % -> 20 % -> 50 %
Starter:    0 % ->  8 % -> 30 %
```

Wenn Reward steigt, aber Story-Erfolgsquoten nicht steigen:
=> Reward-Hacking / falsches Shaping.

---

## 8. Graph-Dashboard — 🧪 VORBEREITET / INTEGRATION PRÜFEN

User möchte auf der Main Map einen Button:

**📈 Graphs**

Beim Klick soll die Kartenfläche durch sinnvolle Lernkurven ersetzt werden.

Gewünschte Graphen:

1. **Lernkurve**
   - Ø Episode-Reward vs. echte PPO-Gesamtsteps

2. **Story-Erfolgsquote**
   - Intro geschafft
   - Haus verlassen
   - Gras erreicht
   - Starter
   - nächste Overworld-Map

3. **Spiel-Fortschritt**
   - Max Level
   - Orden
   - Maps / Story-Fortschritt

4. **Anti-Loop / Stuck**
   - Reset-Anzahl / Rate
   - idealerweise relativ zu Episoden oder Steps

Oben KPI-Kacheln:
- PPO Steps
- Modellversion
- abgeschlossene Episoden
- Ø Reward
- Haus verlassen %
- Starter %

### Vorbereitete lokale Dateien
- `train_graph_stats.py`
- `web_stream_graphs.py`

Sie wurden syntaxgeprüft.

### TODO
- echten `train.py` und `web_stream.py` damit aktualisieren
- danach Dateien erneut verifizieren
- History dauerhaft z.B. in:
  `~/pokemon_ai_project/training_history.json`
- pro Modellversion nur einen Messpunkt speichern

Wichtig:
Browser-only-History reicht NICHT für Overnight-Auswertung.

---

## 9. Skeleton Map — 🧪 VORBEREITET / INTEGRATION PRÜFEN

Ziel:
Trainingsagenten helfen beim schnellen Aufbau der **Kartenstruktur**, aber nicht
bei echten Tiles.

Vorgesehen:

`Skeleton Map Builder`
- liest nur bestehende `instances_data/inst_XX.json`
- ungefähr 1x pro Sekunde
- schreibt ungefähr 1x pro 5 Sekunden
- keine RAM-Reads
- keine Screenshots
- kein OpenCV in den Training Agents
- keine Tile-Bilder

Daten:
- Bank
- Map
- min/max X/Y
- ungefähre Breite/Höhe
- welche Agents dort waren
- beobachtete Map-Transitions

Output:
`~/pokemon_ai_project/skeleton_map.json`

### Vorbereitete lokale Datei
`skeleton_map_builder.py`

### Geplanter Webstream
- leere/gestrichelte Kästen auf der Global Map
- echte Tiles füllt später nur der Watcher

### TODO
Vor Aktivierung prüfen, ob die echte Projektversion dies wirklich enthält.
Performance nach Start kontrollieren.

---

## 10. Watcher-Probleme

### A. Menü-FPS — ✅ ERLEDIGT
Frühere Beobachtung:
- im eigentlichen Spiel ungefähr 150 FPS
- Intro/Menu teilweise nur 30–50 FPS

Ursache:
- teure RAM-Pointer-Vollscans solange noch kein gültiger Pointer existierte

Umgesetzter Fix:
- bekannte Pointer-Slots billig weiter prüfen
- Vollscan im Intro stark drosseln, ungefähr alle `0.75s`

**Aktueller Praxisstand:**
- ✅ Watcher läuft jetzt auch im Intro konstant ungefähr **120–150 FPS**
- Dieser Punkt gilt derzeit als behoben.

### B. Watcher stoppt minutenlang mit Inputs — 🔴 OFFEN / BEOBACHTEN
Beobachtung:
- kommt schnell ins Zimmer/F2
- steht dann teils 1–3 Minuten
- später läuft er plötzlich wieder

Das sieht eher nach blockierendem Watcher-Loop aus als nach PPO.

Verdächtig:
- `mapper.add_frame()`
- `mapper.get_preview()`
- riesiger Canvas durch falsche RAM-Koordinate
- Hot-Reload `PPO.load()`

### Vorbereiteter Schutz
Lokale Dateien:
- `tile_map_builder_guarded.py`
- `watch_nonblocking_map.py`

Idee:
- unrealistische Player-Sprünge innerhalb einer Map ignorieren
- absurd große Map-Ausdehnungen nicht rendern
- Preview nur gedrosselt rebuilden
- Map-Arbeit messen (`MapWork XXXms`)

### TODO
Diese Schutzmaßnahmen in die echten Dateien übernehmen und danach verifizieren.

---

## 11. Watcher Input-Anzeige — 🧪 VORBEREITET / ZUSAMMENFÜHREN

Gewünscht:
unter/über dem Emulator-Screen anzeigen:

```text
INPUT: LEFT
LAST: LEFT  LEFT  UP  A  RIGHT ...
```

Damit kann unterschieden werden:

- Policy spammt wirklich dieselbe Taste
oder
- Inputs wechseln, aber Figur bewegt sich trotzdem nicht.

Vorbereiteter lokaler Watcher:
`watch_with_input_display.py`

### TODO
Input-Anzeige mit dem Nonblocking-Map-Fix zusammenführen.
Nicht einen älteren Watcher über einen neueren Patch kopieren.

---

## 12. Map Builder — 🔴 OFFENES KERNPROBLEM

Aktueller Tile-Mapper nimmt sinngemäß an, dass der Spieler immer ungefähr auf
Screen-Tile `(7,5)` liegt.

Das ist bei Kamera-Clamping / Kartenrändern nicht immer korrekt.

Folgen:
- Tiles verschieben sich
- Bereiche überschreiben sich
- Indoor-/Randbereiche sehen falsch aus

### Bessere zukünftige Lösung
Frame Registration / Kamera-Origin-Inferenz:

- neuen Screen mit vorherigem Screen vergleichen
- kleine 16px-Shifts testen
- erkennen, ob Kamera wirklich gescrollt hat
- bei Map-Wechsel Registration resetten
- statische Tiles nur nach starker Mehrheit überschreiben

### Global Map
Nur Overworld:
- FireRed Bank 3

Indoor:
- separat als Room Maps
- nicht in globale Kanto Map mischen

---

## 13. Aktuelle Telemetrie-Dateien

Training Agents:
`~/pokemon_ai_project/instances_data/inst_00.json`
bis ungefähr:
`inst_29.json`

Watcher:
`inst_99.json`

Training-Agenten liefern u.a.:
- id
- bank/map/x/y
- path
- steps
- reward
- level
- badges
- in_battle
- explored_tiles
- visited_maps
- stuck_counter
- reward_stats

Watcher zusätzlich:
- Watcher Reward
- Mapping-Tiles
- RAM source/trusted
- ggf. Input-History nach entsprechendem Patch

---

## 14. Start / Stop

Normaler kompletter Start soll langfristig sein:

```bash
cd ~/pokemon_ai_project
bash start_all.sh
```

Gewünschte Prozesse:
- Training – 30 Agents
- Skeleton Mapper
- Web Stream
- Watcher
- ngrok

### Wichtig
Vor Nutzung von `start_all.sh` den echten Inhalt kontrollieren.

Eine ältere Drive-Version enthielt noch einen:
`ONE-TIME FRESH AI RESET`

mit Marker:
`.fresh_ai_reset_done_v3`

Das darf **nicht versehentlich erneut das aktuelle Gehirn löschen**.

Der gewünschte neue `start_all.sh` soll:
- alte Prozesse stoppen
- temporäre Live-/Map-Daten ggf. leeren
- Modell/Checkpoints/Curriculum behalten
- Skeleton Map behalten
- keinen Timer haben
- 5 Prozesse starten

---

## 15. Nächste Prioritäten

### PRIORITÄT 1 – echten Projektstand verifizieren
Vor allem:
- `pokemon_env.py`
- `train.py`
- `watch.py`
- `web_stream.py`
- `tile_map_builder.py`
- `start_all.sh`
- `skeleton_map_builder.py`

Nicht von Chat-Aussagen ausgehen – echte Dateien lesen.

### PRIORITÄT 2 – House Reward Fix aktivieren
Ziel:
Treppe-rauf/runter Reward-Hacking beenden.

### PRIORITÄT 3 – Graphs fertig integrieren
Damit Lernfortschritt messbar wird.

### PRIORITÄT 4 – Watcher Freeze beseitigen
Input-Loop darf niemals minutenlang durch Mapping blockiert werden.

### PRIORITÄT 5 – Map-Registration
Feste `(7,5)` Kameraannahme ersetzen.

---

## 16. Prüfwerte nach Neustart

Nach Änderungen ungefähr prüfen:

### Training
```bash
ps aux | grep "[t]rain.py"

TRAIN_PID=$(pgrep -f "python.*train.py" | head -1)
echo "Train PID: $TRAIN_PID"
pgrep -P "$TRAIN_PID" | wc -l
```

Bei 30 Envs sollten ungefähr 30 Worker-Prozesse existieren.

### Watcher
Beobachten:
- Emulator FPS
- GUI reagiert
- Input ändert sich regelmäßig
- kein minutenlanges Hängen
- MapWork möglichst niedrig

### Statistik
Nach einigen Modellversionen:
- Intro %
- Treppe/F1 %
- Haus verlassen %
- Gras %
- Starter %
- Anti-Loop-Rate

---

## 17. Wichtige Drive IDs

Projektordner:
`1XsXyYGVWuzoJdNUF5HNBBt_6u-N5KTZR`

Custom Integrations:
`1kD7tEhs17RXpXYSSajoJ2PkHeBkRS5ip`

FireRed Integration:
`1PUssf1nb2vBj2OGbg18gFmkpN8Wsfd2z`

Dateien:
- `watch.py`
  `158pbg4oK_M2XpoP1lZccYgFCq2PcFAWh`
- `pokemon_env.py`
  `1QtknO5nNU1Fh7N_sg7u8kL2cTTOBPITp`
- `train.py`
  `14tt1cOjOO-7qfTX8vGT61yKuUYuGVMaf`
- `web_stream.py`
  `1LVYDbBlHuWRkGXSpdl5vfX0gcWiMOlau`
- `start_all.sh`
  `1YFwC5LDzpQNhaU69fE6R-KgTlWkuHoYo`
- `tile_map_builder.py`
  `1O2QzH-fsuAdO36z-b9WUVQC0ej2bdhes`
- `firered_ram.py`
  `1svMWII6NdeAC9i1SQx1hzs3dp_J_ccoj`
- `model_version.json`
  `1SlPV-YTsU1f7kG4qAY1iTMECZBKoN0Fy`

---

## 18. Arbeitsregel für jeden nächsten Assistenten

1. Diese Datei lesen.
2. Aktuelle echte Projektdatei öffnen/fetchen.
3. Änderung auf dem aktuellen Stand aufbauen.
4. Syntax prüfen.
5. Wirklich in Projekt/Drive schreiben.
6. Datei erneut lesen/fetchen.
7. Erst dann sagen: **"Patch ist aktiv."**
8. Bei Unsicherheit klar sagen: **"nur vorbereitet / nicht verifiziert".**


---

## 19. Update 2026-09-02 11:17 – Endlos-Training + RAM-Skeleton Streamingkarte

### ✅ AKTIV / VERIFIZIERT

#### train.py
Drive-ID: `14tt1cOjOO-7qfTX8vGT61yKuUYuGVMaf`

- `TRAIN_FOREVER = True`
- `TRAIN_CHUNK_TIMESTEPS = 1_000_000`
- PPO laeuft blockweise unbegrenzt weiter, bis der Benutzer `Ctrl+C` drueckt.
- `reset_num_timesteps=False`: globaler PPO-Step-Zaehler bleibt erhalten.
- `TOTAL_TIMESTEPS` wird nur noch verwendet, wenn `TRAIN_FOREVER = False` gesetzt wird.
- Zyklische Checkpoints bleiben aktiv.
- Final-Save bei `Ctrl+C` bleibt aktiv.
- Drive nach Update erneut gefetcht: **7681 Bytes**, Modified `2026-09-02T09:17:00.731Z`.

#### watch.py
Drive-ID: `158pbg4oK_M2XpoP1lZccYgFCq2PcFAWh`

- Screenshot-/Tile-Stitching im Watcher entfernt.
- Watcher-Karte basiert jetzt nur auf bereits gelesenen RAM-Daten `(Bank, Map, X, Y)`.
- Keine zusätzlichen RAM-Reads fuer die Karte.
- Keine Screenshot-Tiles / kein OpenCV-Matching fuer Mapping.
- Pro aktueller Map wird der echte RAM-Pfad als Skeleton dargestellt.
- Nur echte 1-Tile-Nachbarbewegungen werden verbunden; Spruenge/Mapwechsel erzeugen keine falschen Linien.
- Watcher-Telemetrie, Reward-Spiegel und Statistik laufen weiterhin unabhaengig weiter.
- Drive nach Update erneut gefetcht: **31030 Bytes**, Modified `2026-09-02T09:17:08.858Z`.

#### web_stream.py
Drive-ID: `1LVYDbBlHuWRkGXSpdl5vfX0gcWiMOlau`

- Browserkarte nutzt jetzt RAM-Skeleton statt `kanto_map.png` als Hauptdarstellung.
- `/api/skeleton` wird direkt aus bestehenden `instances_data/inst_*.json` berechnet.
- Kein separater Skeleton-Prozess noetig.
- Kein Screenshot-Mapping und keine zusaetzlichen RAM-Reads.
- Jeder Trainingsagent bekommt eine dauerhaft eigene Farbe.
- Watcher bleibt gruen.
- Klick auf Agent: Fokusmodus, nur dieser Agent wird auf der Karte gezeigt.
- Zweiter Klick auf denselben Agenten: wieder alle anzeigen.
- Bei fokussiertem Agenten werden einzelne RAM-Schritte als Punkte + Route dargestellt.
- Detailpanel listet die letzten ca. 24 Schritte mit Richtung (`↑ ↓ ← →`), Bank/Map und X/Y auf.
- Watcher-Telemetrie laeuft immer weiter, auch wenn ein Trainingsagent fokussiert ist.
- Map-Umrisse werden aus Min/Max der von den Agents beobachteten RAM-Koordinaten gebaut.
- Python-Syntax und Inline-JavaScript (`node --check`) wurden lokal geprueft.
- Drive nach Update erneut gefetcht: **55408 Bytes**, Modified `2026-09-02T09:17:16.676Z`.

### 🟡 WICHTIG / NAECHSTER AUSBAU DER KARTE

Die Skeleton-Karte zeigt aktuell **erkundete Umrisse und echte Agent-Routen**, aber noch keine echten FireRed-Grafiktiles. Das ist absichtlich der robuste Zwischenstand fuer Streaming.

Spaeter moeglicher Ausbau:
1. echte FireRed-ROM-Maplayouts/Tiles auslesen,
2. diese als statischen Hintergrund pro `(Bank, Map)` rendern,
3. RAM-Skeleton + Agent-Routen darueberlegen,
4. Map-Uebergaenge aus beobachteten Transitions automatisch relativ platzieren.

### 🔴 OFFEN / NICHT MIT DIESEM PATCH ERLEDIGT

- Potion/PC-Item-Reward (+100 nur bei echter Item-Uebernahme, kein PC-Loop) ist noch nicht als aktiv verifiziert.
- Early-Game-Patch `90% Beginning / 10% Curriculum` sowie gestufte `900 / 1500 / 2000` Timeouts wurde zuvor besprochen, war im zuletzt kontrollierten `pokemon_env.py` aber noch NICHT vollstaendig aktiv. Vor weiterem langen Training diesen Punkt separat verifizieren/patchen.

### Arbeitsregel erweitert
Nach jedem groesseren Patch diese MD-Datei aktualisieren, damit bei Kontext-/Modellwechsel direkt weitergearbeitet werden kann.


## 2026-09-02 – Early-Game Stage-Timeout Fix

✅ `pokemon_env.py` wurde auf dem Drive aktualisiert und verifiziert.

Aktiver Early-Game-Fokus:
- `START_FROM_BEGINNING_PROB = 0.90`
- Curriculum damit effektiv 90 % Beginning / 10 % gespeicherte States
- Treppe F2→F1: `+80`
- Zurück F1→F2 vor Hausausgang: `-30`
- Haus verlassen: `+200`
- erster Schritt draußen: `+30`

Aktive Beginning-Timeouts:
- Intro nicht nach 900 Episode-Steps fertig → `intro_timeout`
- Intro fertig, aber Treppe/F1 nicht innerhalb weiterer 1500 Steps → `stairs_timeout`
- Treppe/F1 erreicht, aber Haus nicht innerhalb weiterer 2000 Steps verlassen → `house_exit_timeout`
- nach Hausausgang greifen diese Early-Game-Timeouts nicht mehr; dann gilt wieder das normale Episodenlimit.

Wichtig:
- Der vorherige Drive-Stand hatte weiterhin `START_FROM_BEGINNING_PROB = 0.65` und KEINE 900/1500/2000 Story-Timeouts. Deshalb liefen viele Agents wieder bis 5k–8k Steps.
- Der Fix wurde am 2026-09-02 gegen 11:23 MESZ tatsächlich via Drive `update_file` eingespielt und danach erneut mit Dateigröße 37393 Bytes verifiziert.


## 2026-09-02 – Persistenter Blue-Line Exploration Reward

✅ `pokemon_env.py` aktualisiert:
- `NEW_EDGE_REWARD = 0.25`
- `NEW_MAP_REWARD = 10.0`
- `NEW_TRANSITION_REWARD = 15.0`
- normale Exploration greift erst nach Verlassen des Start-Hauses
- Reward basiert auf denselben RAM-Koordinaten, aus denen die blauen Linien entstehen
- neuer Linienabschnitt = genau 1 Tile Bewegung auf derselben Map
- A→B und B→A zählen als derselbe Abschnitt
- neuer Map-/Warp-Übergang wird anhand konkreter Ein-/Ausgangskoordinaten erkannt
- Rückweg durch denselben Übergang gibt keinen zweiten Reward
- persistentes Gedächtnis pro Agent in `exploration_memory/agent_##.json`
- dadurch gibt derselbe Weg nach Episode-Reset keinen neuen Reward
- keine zusätzlichen RAM-Reads; nutzt die bereits gelesene `(bank,map,x,y)` Telemetrie

✅ `web_stream.py` aktualisiert:
- Detailpanel zeigt `Known Edges`, `Known Maps`, `Transitions`
- letztes Reward-Event sichtbar
- Agentenliste kennzeichnet angezeigte Steps als Episode-Steps (`ep`)

Drive-Verifikation:
- `pokemon_env.py` via `update_file` erfolgreich ersetzt, Größe 43902 Bytes
- `web_stream.py` via `update_file` erfolgreich ersetzt, Größe 56700 Bytes

Testziel nach Statistik-Reset:
- neue `new_edge:+0.25`, `new_map_persistent:+10.00` und `new_transition:+15.00` Events beobachten
- prüfen, ob bekannte blaue Linien bei späteren Episoden keinen Reward mehr auslösen
- prüfen, ob Early-Game weiterhin nicht durch Schlafzimmer-Exploration exploitet wird


## 2026-09-02 – Beginning-8k-Run Diagnose + Timeout-Härtung

Diagnose:
- Die 900/1500/2000-Timeouts waren vorhanden.
- Sie waren aber an `not self.left_house_rewarded` gekoppelt.
- `left_house_rewarded` konnte schon durch einen einzelnen falschen/stalen RAM-Wechsel auf Overworld gesetzt werden.
- Sobald das geschah, waren alle Early-Game-Timeouts deaktiviert und Beginning-Runs konnten wieder bis zum globalen 8192-Step-Limit laufen.
- Das erklärt die beobachteten 8k-Episoden trotz eingebauter Stage-Timeouts.

Lokaler Fix erstellt:
- `OUTDOOR_CONFIRM_READS = 3`
- `BEGINNING_HARD_EARLY_CAP = 4000`
- Hausausgang wird bei Beginning erst nach erkannter Treppe + 3 vertrauenswürdigen Outdoor-Reads bestätigt.
- Early-Game-Timeouts hängen jetzt an `left_house_confirmed`, nicht am zu leicht auslösbaren Reward-Flag.
- Absoluter Failsafe: Beginning ohne bestätigten Hausausgang endet spätestens bei 4000 Steps.
- Debug-Telemetrie: `story_stage`, `left_house_confirmed`, `outdoor_confirm_reads`, `last_stage_timeout`.

Wichtig:
- Diese Version wurde lokal erzeugt und syntaktisch geprüft.
- Mit den aktuell verfügbaren Drive-Aktionen konnte sie in diesem Turn NICHT in Google Drive geschrieben werden.


## 2026-09-02 – Curriculum Early-Game + Explore→Navigate Reward

✅ Drive-Stand `pokemon_env.py` aktualisiert.

Early-Game:
- Intro-Reward jetzt `+100`.
- Ziel-Curriculum-States werden gespeichert:
  - `intro_complete`
  - `stairs_down`
- Curriculum priorisiert diese frühen States.
- Stage-Timeouts gelten jetzt für Beginning UND Curriculum, solange das Start-Haus nicht bestätigt verlassen wurde.
- `OUTDOOR_CONFIRM_READS = 3`
- `BEGINNING_HARD_EARLY_CAP = 4000`
- `stairs_down`-Curriculum stellt das Story-Flag explizit wieder her.
- Hausausgang wird auch bei Indoor-Curriculum korrekt erkannt/bestätigt.

Exploration:
- `NEW_EDGE_REWARD = +0.35`
- `KNOWN_EDGE_PENALTY = -0.01`
- ab 3 Wiederholungen desselben bekannten Edges in einer Episode zusätzlich `REPEAT_EDGE_PENALTY = -0.08`
- `NEW_MAP_REWARD = +20`
- `NEW_TRANSITION_REWARD = +30`
- persistentes Gedächtnis pro Agent bleibt in `exploration_memory/agent_##.json`
- bekannte Kante gibt nach Episode-Reset keinen neuen positiven Reward.

Explore→Navigate:
- Wenn 250 Steps lang kein neuer Edge entdeckt wurde und auf der aktuellen Map ein bekannter Warp existiert:
  - Modus `EXIT SEEK`
  - Distanzverringerung zum nächsten bekannten Warp: `+0.05`
- Sobald wieder ein neuer Edge entdeckt wird, zurück zu `EXPLORE`.
- Dies ist eine Heuristik; vollständige Map-Grenzen werden nicht behauptet.

Webstream:
- Agenten haben weiterhin individuelle Farben.
- Klick auf Agent fokussiert nur diesen Agenten; nochmal klicken = alle.
- Historische, persistente bekannte Edges des fokussierten Trainingsagenten werden direkt aus `exploration_memory` geladen und bleiben sichtbar.
- Bekannte Warps/Transitions werden gestrichelt dargestellt.
- Die kurze Live-Route darf weiterhin begrenzt sein; die persistente Karte verschwindet dadurch nicht mehr.
- Detailpanel zeigt `EXPLORE` bzw. `EXIT SEEK`.

Drive-Verifikation:
- `pokemon_env.py`: 52444 Bytes, Update erfolgreich.
- `web_stream.py`: 61165 Bytes, Update erfolgreich.
- `model_version.json` nach versehentlichem falschem MD-Upload wieder als gültiges JSON mit 86 Bytes repariert.


## 2026-09-02 – TRAINING V2: Reward-/Curriculum-Redesign nach ~10M Steps ohne Hausausgang

### Diagnose
Die alte Lernlogik war strukturell problematisch:
- `reward = +0.01` fuer jeden gameplay-ready Step belohnte auch langes Herumlaufen/Looping.
- Bei 8192 Steps konnten dadurch ca. +81.92 Reward ohne Story-Fortschritt entstehen.
- `gamma=0.995` macht sehr spaete Story-Rewards fuer weit zurueckliegende Aktionen extrem schwach.
- 90 % Beginning-Runs erzeugten zu viele aehnliche Failure-Samples.
- Curriculum-States lagen pro Agent getrennt; 30 parallele Agents teilten ihre Meilensteine nicht sauber.
- Exploration-Memory lag pro Agent; derselbe Weg konnte dadurch bei mehreren Agents als "neu" gelten.
- Permanente One-Time-Novelty allein ist fuer on-policy PPO nicht genug: sobald Novelty weg ist, kann die Policy gutes Verhalten wieder vergessen.

### Neue Training-V2-Architektur
`pokemon_env.py`:
- positiver Step-/Survival-Reward entfernt
- `INTRO_STEP_COST = -0.002`
- `GAMEPLAY_STEP_COST = -0.003`
- Mapping-Novelty stark reduziert:
  - neuer globaler Edge `+0.10`
  - neue globale Map `+5`
  - neuer globaler Transition/Warp `+20`
- Mapping-Reward wird ueber alle 30 Agents global dedupliziert.
- ein bekannter Edge gibt beim ersten notwendigen Durchlauf in einer Episode keinen positiven Reward
- zweiter Besuch desselben Edge: `-0.01`
- ab drittem Besuch: `-0.05`
- bekannte Story-Warps erzeugen wiederholbares Ziel-Shaping:
  - Richtung Ziel `+0.08`
  - vom Ziel weg `-0.08`
- Zielentfernung wird bevorzugt ueber den selbst entdeckten Edge-Graph berechnet, Manhattan nur als Fallback.
- Mapping-Novelty greift nun auch im Start-Haus, aber global nur einmal.

### 30 Agent Spezialisten
Alle 30 Agents trainieren weiterhin dieselbe PPO-Policy:
- Agent 00-07: INTRO-Skill
- Agent 08-15: TREPPEN-Skill
- Agent 16-23: EXIT-Skill
- Agent 24-29: FULL CHAIN

Skill-Episoden terminieren direkt nach Erfolg:
- Intro-Spezialist: Intro abgeschlossen -> Episode Erfolg/Reset
- Treppen-Spezialist: F2->F1 -> Episode Erfolg/Reset
- Exit-Spezialist: Haus verlassen -> Episode Erfolg/Reset
- Full Chain laeuft weiter.

Story-Rewards:
- Intro `+100`, Spezialisten-Ziel zusaetzlich `+50`
- Treppe `+150`, Spezialisten-Ziel zusaetzlich `+50`
- bestaetigter Hausausgang `+500`, Exit-Ziel zusaetzlich `+100`

Timeouts:
- Intro: 900 Steps
- nach Intro bis Treppe: 1000 Steps
- nach Treppe bis Hausausgang: 1200 Steps
- absoluter Indoor-Hardcap: 2500 Steps

### Gemeinsamer Curriculum-Bank
- neuer Ordner `curriculum_shared/`
- beim Start kopiert `train.py` vorhandene per-Agent States automatisch einmal in den Shared-Bank
- wenn irgendein Agent spaeter `intro_complete`, `stairs_down` oder andere Milestones erreicht, wird der State zusaetzlich global gespeichert
- Spezialisten koennen dadurch sofort von Erfolgen anderer Agents starten

### Globales Exploration-Gedaechtnis waehrend Training
`train.py` startet einen `multiprocessing.Manager` mit:
- shared_edges
- shared_maps
- shared_transitions
- shared_lock

Bestehende `exploration_memory/agent_##.json` Dateien werden beim Start als globale Seed-Map geladen.
Ein neuer Edge kann dadurch trainingweit nur einmal Novelty-Reward ausloesen.

### PPO Tuning V2
- Learning Rate: `0.00015` statt `0.0003`
- Entropy Coef: `0.005` statt `0.02`
- Save every: `25_000` statt `50_000`
- Endlos-Training bleibt aktiv bis manueller Ctrl+C.

### V2 Statistik / Dashboard
Neue objektivbezogene Zaehler:
- v2_intro_episodes / v2_intro_success
- v2_stairs_episodes / v2_stairs_success
- v2_exit_episodes / v2_exit_success
- v2_full_episodes
- v2_full_intro
- v2_full_stairs
- v2_full_left_house

`web_stream.py` nutzt `stats_schema = 3` und zeigt getrennt:
- Intro Skill
- Treppen Skill
- Exit Skill
- Full: Intro
- Full: Treppe
- Full: Haus raus

Damit zaehlt ein absichtlich nach dem Intro beendeter Intro-Spezialist NICHT mehr als Hausausgang-Fehlschlag.

### Drive – tatsaechlich geschrieben und frisch verifiziert
- `pokemon_env.py` Drive-ID `1QtknO5nNU1Fh7N_sg7u8kL2cTTOBPITp`
  - 62168 Bytes
  - modified 2026-09-02T10:41:31.717Z
- `train.py` Drive-ID `14tt1cOjOO-7qfTX8vGT61yKuUYuGVMaf`
  - 11702 Bytes
  - modified 2026-09-02T10:39:19.434Z
- `web_stream.py` Drive-ID `1LVYDbBlHuWRkGXSpdl5vfX0gcWiMOlau`
  - 67415 Bytes
  - modified 2026-09-02T10:41:44.233Z

### Neustart-Empfehlung V2
Der alte ~10M-Brain soll NICHT geloescht, sondern archiviert werden.
Fuer V2 wird ein frischer PPO-Brain empfohlen, weil der alte Brain die alte positive Loop-/Survival-Reward-Struktur gelernt hat.

Behalten:
- Exploration Memory
- Curriculum States
- Live Map Tiling
- alter Brain als Backup

Reset:
- aktiver `pokemon_model_latest.zip` nach Backup entfernen
- training_history.json
- training_stats/*
