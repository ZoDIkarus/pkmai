# PKMAI — Live Work Log

## 2026-09-06 — V18: per-run tile ladder, Pokécenter/Mart/badge globals, battle rebalance, dashboard fixes

Full trainer + watcher + web restart, brain kept (learner ~21.5M, champion v9).
V17.4 was the starting point. All 67 unit tests pass.

**Reward model (`src/pokemon_env.py`):**
- Tile first-find now pays **per run** (`seen_coords`, every episode fresh) — NOT
  fleet-once. A watcher / any agent on long-known ground was seeing zero tile
  reward. Hand-set ladder keyed on the tile's own map stage:
  `TILE_REWARD_BY_STAGE = {1:0.2, 2:3, 3:4, 4:5, 5:5, 6:6}` — Alabastia (Spawn) fast
  wertlos, ab Route 1 steil, sonst schlaegt die Kachel-Menge in Pallet die
  hoehere Rate weiter vorne. Deckel 20 neue Kacheln pro Karte/Episode, danach 0
  (`new_tile:...:capped`). Pallet-Haeuser (Bank 4) ebenfalls 0.2.
  Interiors `INTERIOR_TILE_REWARD_BY_BANK = {4:1, 5:2, 6:3}` (Pallet/Vertania/
  Marmoria houses, ~half their city). Plus `GLOBAL_NEW_TILE_BONUS = 1.0`
  fleet-once on top (`shared_tiles`). Event `new_tile:s<n>[+g]:+N`.
- City 250/run, route/building 25/run, first global stage unlock 1000-once,
  per-run stage bonus 0, `NORTH_CORRIDOR_ROW_REWARD` 0 — unchanged from the
  values confirmed this session; no blanket directional bonus.
- Scout arrival: `_can_reward_map_arrival` pays scouts for stages deeper than
  `episode_start_stage`, nothing for backtracking; Pallet excluded for all.
- **Pokécenter:** `POKECENTER_ENTER_REWARD 100`/run/center, `POKECENTER_ADVANCE_HEAL_REWARD 250`
  (first heal at a center deeper than any used this run — wipe respawn advanced),
  `POKECENTER_FIRST_HEAL_GLOBAL_REWARD 1000` once ever per center (`pc_heal_<b>_<m>`).
  Tables `POKECENTER_MAPS` / `POKECENTER_HEAL_MAPS` — (5,4)/(5,5) Vertania, (6,5) Marmoria.
- **Poké Mart:** `(5,3)` Vertania — `POKEMART_ENTER_REWARD 100`/run + `POKEMART_FIRST_GLOBAL_REWARD 1000`
  once ever (`mart_<b>_<m>`). So the brain knows the shop exists (buy Poké Balls).
- **Badge:** `BADGE_EARNED_REWARD 2000`/run kept + `BADGE_FIRST_GLOBAL_REWARD 5000`
  fleet-once per badge number (`badge_<n>_ever`).
- **Battle rebalance** (watcher was 45% of steps in battle): `ENEMY_DAMAGE_REWARD_PER_HP`
  0.15→0.08; per-episode wild decay `WILD_BATTLE_DECAY_AFTER 6` / `WILD_BATTLE_DECAY_FACTOR 0.3`
  on `WILD_TRAINING_MAPS` (`episode_wild_faints`, `_battle_reward_scale`); trainer battles
  `TRAINER_BATTLE_REWARD_MULT 2.0` on damage/faint/win and exempt from wild decay
  (`_is_trainer_battle`, `raw_flags & 0x8`). Flee penalty left at −25.
- **Catch:** `SPECIES_CAUGHT_FIRST_REWARD` 50→120 + `min(level,20)*4` (`SPECIES_CAUGHT_LEVEL_BONUS`).
- **Scouts:** `FRONTIER_SCOUT_SLOTS` 5→2 per checkpoint — more of the fleet runs
  full journeys; scouts were getting through but full runners were not.
- **Warp reward:** the global bonus now claims a coarse `(map_a↔map_b)` pair,
  not the coordinate-rich `_transition_key`, AND it is doubly guarded: a pair
  already present in the loaded navigation history counts as known immediately
  (`_derive_warp_pairs` / `_known_warp_pairs`, also merged on the watcher's 5-min
  refresh), and a genuinely new pair pays once via `claim_event` into
  `reward_events.json` so it stays paid across every restart. Fixes the user's
  repeated `new_warp_global:+100` for entering/leaving Pallet after each watcher
  restart. `persistent_known_transitions` stays coordinate-rich for navigation.
  No warp reward on the step a battle ends. Events `new_warp_global` (once) /
  `known_warp:+0` / `new_warp_suppressed:+0`.
- **Scout backtrack:** a scout on/behind its spawn stage (Route 1 scout drifting
  into Pallet) now earns no per-run tile reward there either
  (`new_tile_scout_backtrack:+0`), matching `_can_reward_map_arrival`. Note: 2
  scouts seen in Pallet right after a restart is expected — they resume on
  Route 1 (stage_2 checkpoint) and the not-yet-retrained policy walks them south;
  no reward is paid for it.

**Dashboard (`src/web_stream.py`, `assets/ui/dashboard-language.js`):**
- Watcher live image removed from the Overworld-Map left column ("● AGENT · DETAIL"
  now); the live feed only shows under the Watcher tab.
- Clicked-agent detail panel gained a "Letzte 10 Rewards" list (`#detail-recent-rewards`)
  for a quick read of what an agent is doing.
- i18n: ~40 missing de→en pairs added (status pills, badge names, Poké-Mart, Eich-Szene,
  Instanzen, "…jetzt", empty-slot titles …). Removed the `['Beste','Best']` pair that
  turned "Bester" into "Bestr". Translation now runs synchronously in the
  MutationObserver microtask (was `requestAnimationFrame` — paused in a background tab
  and silently stopped translating; an interim `setTimeout` caused a visible German
  flicker on every refresh) + a `setInterval(translatePage, 700)` safety net + per-node
  try/catch. Verified live on all 5 tabs, EN and DE, no leaks, no flicker.

**Deferred to `docs/BIG_CHANGES_TODO.md`:** FighterBrain (separate combat policy +
champion loop), and the "house after Viridian Forest" special-case (needs its map id).

## 2026-09-06 — Route 2 included in scout allocation

User clarified three reached checkpoints must mean 15 scouts. SCOUT_STAGES now
(2,3,4,5,6): Route 1, Viridian, Route 2, Forest, Pewter. Five fixed ranks per stage;
Pallet remains excluded. Route 2 had mistakenly been omitted from the earlier
interpreted list. Full runners use the unchanged master. 64 tests pass, including
exactly 15 scouts and 45 full slots for stages 2/3/4 in a 60-agent fleet.
Trainer saved at 18,727,920 before restart. Watcher 63749 not restarted.


## 2026-09-06 — Watcher follows learner, trainer alone learns

User clarified watcher should evaluate new brains, not wait for champion approval.
Trainer retains champion separately as pokemon_model_champion.zip (original weights
renamed intact). The running watcher's old best-path fallback now resolves to
pokemon_model_resume.zip: verified live telemetry, same PID 63749, episode 13
continued, about 59 FPS. No watcher restart or optimizer operation occurred.
Source watch.py now explicitly prioritizes resume snapshots. Trainer publishes
resume ZIPs atomically so reload cannot consume partially written snapshots.
Champion ranking was NOT changed or forced; user clarified separation instead.

Rewards active in trainer: 25 per new map, 50 per new city under existing eligibility;
per-run world-stage bonus is zero, lifetime global depth unlock is 500. Durable
claim history is checked even for empty registries and write failure pays nothing.
63 tests pass including persistence, publication failure and watcher routing.
Trainer resumed at 18,621,720; PID 70747, first save 18,621,780.
Running watcher reward code still has old values because its module was loaded
before these edits; model reload does not reload Python code. Tell user explicitly.


## 2026-09-06 — Route 2 checkpoint gate corrected

Route 2 was really visited at y76–79, but a reused Route 1 checkpoint gate required
y <=34 on every north corridor. Removed that absolute gate: first validated
arrival can save any geographic stage; north-first replacement remains unchanged.
Real isolated stage_3 scout advanced to Route 2 and produced stage_4 at x11/y76.
61 unit tests pass. Trainer saved at 18,585,900 and restarted alone to activate.
Watcher PID 63749 continues Champion v8 (7,900,560 training steps), not the current
18.5M learner; candidate evaluations are rejected. No forced champion promotion.
Fleet snapshots show many full runners oscillating at Pallet/Route 1, earning
+100 map and +1000 stage once then usually paying only -0.005 per action.
This indicates unresolved policy/reward shortcomings; checkpoint fixes alone are
not evidence of successful end-to-end learning. No claim that the loop is solved.


## 2026-09-06 — False Vertania checkpoint verified and removed from active use

Isolated restores proved stage_3 metadata said Viridian (3/1) while its actual
state was Route 1 (3/19), x2/y34. Previous explanation of ten scouts as valid
geographic progress was incorrect. Quarantined stage_3 and legacy lab stage_5
under runtime/checkpoints/quarantine_20260906_014530. Actual stage_2 verified at
Route 1 x5/y27, marked with state_validation=1 and SHA-256. Backed up and capped
false depth records to validated stage 2; model weights preserved.
Saving now checks fresh RAM location against requested metadata. Loading checks
validation marker, snapshot hash and actual restored coordinates/stage under the
shared lock; mismatch restores the previous master. Unverified metadata cannot
allocate scout bands. 61 unit tests and real Scout reset passed.
Trainer saved at 8,970,540 and restarted alone. Watcher 63749 untouched.


## 2026-09-06 — Standing restart permission clarified

User permits trainer and other project services to restart when needed. Watcher
must NEVER be restarted during the current stream. Trainer-only maintenance:
saved learner at 8,870,460 steps, then restarted with PKMAI_RESUME_SAVED=1.
This loads north-first checkpoint ranking, hardened scout RAM lookup and the
wider idle-loop guard. Watcher PID 63749 is explicitly excluded; its loaded code
remains the earlier version until the user authorizes a watcher restart.


## 2026-09-06 — Checkpoint ordering corrected (restart pending)

Latest user rule: on the same geographic stage/map, smaller Y (further north)
replaces the old checkpoint regardless of reward. At equal Y, only strictly
higher episode reward replaces it. Larger Y (south) is always rejected.
This supersedes reward-only selection. Independent stage updates, exactly five
fixed scouts per approved stage, and the immutable full-run master are unchanged.
Regression covers south with huge reward, equal north with lower/equal/higher
reward, and further north with lower reward after the forest was already reached.
No runtime savestate was rewritten and no process was restarted.


## 2026-09-06 — Scout position resolver hardened (source only)

Current shared Route 1 checkpoint restored correctly three times before and after
resolver changes. Discovery previously scanned EWRAM data as well as IWRAM and
could accept unrelated three-pointer sequences. It now searches IWRAM only,
selects known slots by RAM layout, rejects impossible link-room map IDs, and
retains a structurally valid pointer slot during transient invalid coordinates.
Regression test demonstrates false EWRAM candidate rejection and subsequent
location recovery without rediscovery. 60 tests pass. No process restarted.
This fixes a demonstrated resolver weakness; the intermittent live scout failure
has not yet been reproduced end-to-end, so live verification remains required.


## 2026-09-06 — Wider watcher door loop: source fix, not yet active

Live watcher at step 1023 repeatedly moved between Oak's lab and Pallet. Last
160 decisions had no reward events, only -0.005 time cost. The 900-step local
guard only covered at most eight tiles; the observed path spans more tiles.
Added an independent 1,800 valid overworld-decision limit without new tile,
EXP, stage or badge progress. Battles pause counting; real progress resets it.
59 unit tests pass. No additional process restart performed.
Separate scout RAM issue remains open: scouts 56/58 intermittently reported
invalid or implausible bank 0/maps 0 or 57, despite Route 1 checkpoint assignment.
Do not label this merely cosmetic until RAM resolver and restore are validated.


## 2026-09-06 — Restart explicitly authorized and completed

Trainer, watcher, webserver and status monitor restarted in four visible Terminal
windows. Trainer saved at 8,482,200 steps and resumed that learner explicitly via
PKMAI_RESUME_SAVED=1; new output confirmed 8,482,260 steps. Web API responds;
watcher telemetry is fresh, reports gMain.inBattle and approximately 59 FPS.
PIDs at verification: trainer 63733, web 63739, status 63745, watcher 63749.
Master savestate remains unchanged. Mapper remains off.


## 2026-09-06 — Geographic progression and battle/loop corrections (restart pending)

These changes supersede the older parcel-based stage descriptions below. No trainer,
watcher, web or mapper process was stopped or restarted during this change.

- Immutable master: the user-recorded `local/custom_integrations/PokemonFireRed-Gba/StartGame.state`
  after Oak's parcel, verified inside the lab at bank 4/map 3, x6/y4. Every full runner
  restores this exact original; only scouts load geographic checkpoints.
  The briefly prepared outdoor derivative was removed, per the latest instruction.
  Lab/indoor baseline counts as stage 1 and produces no geographic stage jump.
- Geography: Pallet 1, Route 1 2, Viridian 3, Route 2 4, Viridian Forest 5, Pewter 6.
  Parcel flags, stairs, buildings and badges do not manufacture geographic stages.
- Exactly five scouts per valid checkpoint on Route 1, Viridian, Forest and Pewter.
  Fixed rank bands prevent reassignment when another checkpoint appears. Replacing
  a savestate never adds scouts. No Pallet or Route 2 scouts. Other agents continue full runs.
- Each stage independently accepts strictly higher episode reward, including later
  full runners returning through earlier maps. Battle/wipe states are not captured.
  Legacy lab checkpoints are rejected by their actual map; old fleet depth is
  migrated from persisted visited maps at trainer startup, preserving model weights.
- Battle detection uses dynamically validated gMain.inBattle (not battle-type flags).
  An isolated real-ROM test detected a zero-type-flag wild encounter and cleared the
  battle after fleeing without overworld movement. Observations use this same signal.
- Fresh party HP on every decision; one wipe charge until recovery; wipe healing
  receives no heal bonus. EXP rewards require the same party and battle context.
- Global center-heal claim now persists in exploration_memory/reward_events.json.
  Historical claims made before this file existed cannot be reconstructed reliably;
  the first post-migration qualifying claim establishes the persistent baseline.
- A shared local-loop guard ends an episode after 900 overworld decisions on at most
  eight tiles without exploration/EXP/stage/badge progress. Battles pause counting.
  This also bounds short back-and-forth loops that reset the old stationary counter.
- Watcher remains policy evaluation without learning. Updated code takes effect only
  on the next explicitly approved restart; refreshing navigation is not code reload.

Validation: 58 unit tests, plus isolated emulator checks. Full environment smoke
validation: 1,000 isolated decisions. Two final resets restored the original master
location and stage 1; its SHA-256 remained unchanged.
No user savestate was modified.


**Dieses Dokument ist der aktuelle Arbeitsstand.** Jede AI / jeder Entwickler
liest das ZUERST (vor `README.md` und `AI_HANDOFF.md`), trägt neue Änderungen
oben ein und lässt den Verlauf stehen. Kurz halten, keine Romane.

Laufende Runtime-Werte immer aus `tools/pkmai_status.py` lesen, nicht hier abschreiben.

---

## Watcher-FPS wieder sichtbar (2026-09-05)

- Gemessene Emulator-FPS im Watcher-Header und Fenstertitel, alle 0,5 Sekunden
  aktualisiert; zählt Hold-/Release-Frames je real verstrichener Sekunde.
- Sollwert zusätzlich im Header; Ist-FPS auch in Watcher-Telemetrie.

## Watcher = Trainer-Umgebung / Battle-Fix / DE–EN (2026-09-05)

- `watch.main()` nutzt `watcher_runtime.run()` und direkt `PokemonFireRedEnv`;
  alte separate Reward-Schleife entfernt. Keine `learn()`-/Optimizer-Aufrufe.
  Aktionen, Beobachtungen, Rewards und Resets stammen aus dem Trainer-Code.
- Watcher-State/Statistiken/Curriculum unter `runtime/watcher_evaluation/`;
  öffentliche Telemetrie nur `inst_120.json`, niemals Trainingsstatistiken.
  Einmal-Boni haben einen privaten Evaluationskontext (Navigations-Snapshot
  beim Start). Identische Regeln, nicht identische Flottenhistorie/Reward-Summen.
- Kampfstatus nun gemeinsam in `battle_state.py`: frische Gegnerdaten erkennen
  auch Wildkämpfe mit Flags 0; kein 96-Step-Ablauf mitten im Kampfmenü.
  Bestätigte Bewegung beendet den Status trotz verbleibender Gegnerdaten.
  Ende-Erkennung bleibt bei stillstehender Figur bis zur nächsten Bewegung
  konservativ; keine Behauptung einer perfekten RAM-Kampfstatusadresse.
- Echter Bug korrigiert: Party-Wipe beim Kampfende verwendete `reward`,
  `reward_events`, `truncated` vor Initialisierung. Initialisierung vorgezogen,
  spätere Zeitkosten addieren sich statt Wipe-Strafe/Abbruch zu überschreiben.
- Dark-Watcher in neutralem Grau, Log zeigt Reward-Ereignisse und Kampfgrund.
  Rotierende Audit-Logs: `runtime/watcher_rewards.jsonl` (5 MB × 4 Dateien).
- Web links separater `/watcher-emulator.jpg` (240×160, nur echte Spielpixel),
  ohne Party-Sidebar im linken Detailbereich. Vollständige Diagnosegrafik
  weiterhin im Watcher-Tab und nativen Fenster verfügbar.
- Indoor-Ansicht ohne Warp-Punkte/-Legende/-Verbindungen. Ortsnamen für frühe
  Karten anhand https://raw.githubusercontent.com/pret/pokefirered/master/data/maps/map_groups.json
  (Reds Haus, Rivalenhaus, Eichs Labor usw.); unbekannte Orte neutral nummeriert.
- Oben DE/EN-Schalter, Englisch als Standard, Browser merkt Auswahl. UI-Labels
  und Ortsnamen wechseln; technische Reward-Event-IDs bleiben zum Debuggen.
  Das Spiel selbst bleibt die deutsche ROM; UI-Sprache ändert keine Spieltexte.
- Verifiziert: 50 Unit-Tests; 120 echte identische Emulator-Aktionen in
  Trainer-/Watcher-Instanzen mit identischen Observations, Rewards, Events und
  Done-Flags; zusätzlich erzwungener Wipe mit -100 und Abbruch. RAM-Discovery-
  Cache für den Vergleich jeweils identisch initialisiert.
- Live-Bild und Audit bestätigen Wildkampf mit Flags 0 als Kampf erkannt.
- Trainer kontrolliert bei 1.649.880 Steps gespeichert. Worker-Shutdown hing
  nach bestätigt erfolgreichem Final-Save; alte Prozesse gezielt beendet.
  Wartungsstart via `PKMAI_RESUME_SAVED=1` setzt den gespeicherten Learner fort
  statt beim älteren Champion zu beginnen. Neuer Trainer überschritt 1.651.200
  Steps; Champion unverändert v2. Watcher/Web in sichtbaren Terminals gestartet.

## Festes Browserlayout + Web-Terminal (2026-09-05)

- Nutzerpräferenz: Webserver bei jedem Neustart in einem sichtbaren eigenen
  Terminalfenster starten, damit Alex die Webserver-Ausgabe sehen kann.
- Oben volle Browserbreite für Statusanzeigen. Darunter ein festes Raster
  bis zum unteren Browserrand: links Alex-Watcher samt Details, mittig
  Live-Karte, rechts scrollbare Agentenliste. Keine frei schwebenden Fenster.
- Watcher-Bild wird jetzt auch auf dem Karten-Tab alle 500 ms aktualisiert.
- Näherer Zoom und dynamische Kartenbegrenzung bleiben erhalten.
- Python-/JavaScript-Syntax und HTML-Rendering geprüft.

## Dashboard-Update (2026-09-05, V17.2)

- Weblayout als durchgehende Seite: feste Abschnitte statt verschiebbarer
  Fenster; Global-Status und Filter außerhalb der Karte, Agenten darunter.
- Fester Kartenzoom 0 → 0.5 (ca. 41 % näher). Pan-Grenzen aus Live-Daten,
  vier statt 200 Felder Rand; mindestens Viewport-Größe, bei Resize neu berechnet.
  Harte Drag-Grenzen und keine Trägheit verhindern weites Wegschieben.
- Python-/JavaScript-Syntax und Dashboard-Rendering geprüft. Keine Browser-Sichtprüfung.
- Aktueller Trainings-/V17.2-Übergabestand: `STATUS_TODO.md` und README.
  Der folgende V15-Stand ist historisch.

---

## Wo wir gerade stehen (2026-09-04, V15.2)

**Das Ziel:** von Spielanfang → Starter → Alabastia raus → Route 1 →
**Vertania City** → Vertania-Wald → Orden 1 (Rocko).

**Aktueller Blocker (Stand 2026-09-04 ~12:45):** Der Champion stand ~23 Mio
Steps still (v6 @ 9,87 Mio). Ursache-Kette: die Flottenverteilung `_agent_role()`
schaltet Phase 5 (freie Welt / World-Push zum Wald) erst frei, wenn der
gemessene Starter-Skill ≥ 880 ist. Der lag bei **0** (Lifetime 21/1160 ≈ 1,9 %
Erfolg), weil ~alle 22 Starter-Agenten in Alabastia im Kreis liefen und nie
Eichs Labor erreichten. Kein Rollback auf V14 möglich: Champion v6 nutzt schon
den V15-Obs-Space (4×64×64 + 31), V14-Code macht 1×64×64 + 28 → `PPO.load`
inkompatibel.

**Heute umgesetzt (V15.2, siehe Änderungs-Log):** Tastendruck-Timing 4/4 → 16/8
Frames, Learner von Champion v6 neu aufgesetzt, echte Step-Kosten für
Meilenstein-Spezialisten, Anti-Kreis-Penalty, Status-Terminal um SHIGGY-Block
erweitert. Erste Wirkung: 16 von 22 Starter-Agenten stehen nun IM Labor.

**Offen / TODO:**
- Beobachten ob Starter-Agenten mit 16/8-Timing jetzt wirklich Schiggy holen
  (`SHIGGY/STARTER … mit Schiggy N` im Status). Wenn ja, steigt der Skill über
  880 und Phase 5 öffnet von selbst.
- Falls sie im Labor hängen: `left_house`-Resume-State prüfen/löschen (der
  Spezialist startet dort und schafft trotzdem nur 1,9 %, während Full-Runs von
  vorne 25 % schaffen — verdächtig).
- Step-Kosten `SPECIALIST_STEP_COST = -0.10` ist ein Startwert. Nach Beobachtung
  nachziehen: zu hoch → Agent lernt „schnell scheitern" statt „schnell lösen".
- Watcher-Exit dauert lang (Sekunden) — niedrige Priorität.
- Champion-`_score` hat einen Tempo-Tiebreaker (`-full_best_stage_steps`), aber
  Skill-Score selbst ist rein binär. Ggf. später Tempo in die Skill-Bewertung.

**Neustart:** V15 nutzt ein frisches Modell. Alte Modelle, Champion,
Curriculum-States, Karten und Statistiken werden gemeinsam archiviert.
Fortschritt wird nur noch als explizite Feuerrot-Kette 0–9 gezählt: Alabastia,
Route 1, Vertania, Eichs Paket, Paketabgabe/Pokédex, Route 2, Wald, Marmoria,
Orden. Beliebige Räume und Warps sind kein Fortschritt.

---

## Änderungs-Log (neueste zuerst)

### V15.3 — ALL-FULL + Reward radikal vereinfacht + kein Champion-Gate (2026-09-04)
Kurskorrektur Richtung PWhiddy-Modell. Grund: Spezialisten überfitten auf ihren
einen Resume-State ("Skill 1000" ≠ echter Lauf), Champion 23 Mio Steps eingefroren.

- **`FULL_ONLY_MODE = True`** (`pokemon_env.py`). `_agent_role` kurzgeschlossen:
  30 Agenten = `full` ab Spielanfang (inkl. Intro), 2 = `progress` Deep-Warm
  (halten die erreichte Spätphase warm). Kein Spezialisten-Bootcamp mehr.
- **Reward-Chirurgie** — hinter `FULL_ONLY_MODE` (ein Schalter, reversibel):
  RAUS: V9-Anti-Camping-Block komplett (v9_explorer_new_tile +1.0 farmbar,
  indoor_stall, v9_stuck), Korridor-/Nord-Richtungsreward, exit_/journey_route_edge
  ("viel gelaufen = gut"), intro_novelty-Screens, `_v10171_story_guard`
  (+0.35/Kachel Post-Haus-Wandern), rollenabhängiges Step-Cost-Tuning.
  BLEIBT: alle Meilenstein-Einmalboni (Intro/Treppe/Haus/Schiggi +1000/Map
  +250/Stufe +250/Orden), `new_edge_global` +0.10 (un-farmbare Erkundungs-Spur),
  Level +150, Kampf, Blackout, `EARLY_STORY_STEP_REWARD` ±1.0 (Haus-Ausgang,
  symmetrisch), einheitliche Zeitgebühr −0.002.
- **Kein Champion-Gate mehr** (`train.py`): `_protected_regression` → immer False,
  `min_full_episodes` 8→4, `_score` auf reine Tiefe+Tempo
  `(orden, stufe, level, maps, -best_stage_steps, starter_permille)` — die
  fragilen permille-Endpositions-Raten sind raus.
- **Learner-Reset** auf Champion v6 (`best.zip` → `resume.zip`). Nicht Random:
  v6 kann laufen/menüen/Intro/Treppe/Haus + teilweise Schiggi.
- Frames bleiben 12/6. **Neustart: Trainer** (+ Watcher hat 12/6 schon).
- `_choose_episode_start`: im `FULL_ONLY_MODE` startet JEDER full-Agent bei
  "beginning" (ganz von vorn, inkl. Intro); nur die 2 progress-Slots resumen tief.
- **Web (`web_stream.py`)**: Watcher-Tab hat jetzt eine klickbare Agenten-Liste
  (Watcher + alle Runner) mit Live-Detail-Panel (Episode-Reward, Steps, Stufe,
  Level, Schiggi, Eich-Szene, Kämpfe, Taste, letzter Abbruchgrund,
  Reward-Events-Aufschlüsselung). Mobile: Seiten-Overflow behoben (kam aus der
  v81-Skill-Zeile in der Graphs-View).
- **Champion-Promotion:** strikte Logik erst mal beobachten (1-2 h). Wenn v6
  hält und die Flotte bei Stufe 2-3 klemmt → `_score`-Primärachse von max auf
  mittlere Full-Lauf-Tiefe umstellen (progressiv, schreibt öfter neue Brains),
  v6 vorher als best_v6.zip sichern.
- Offen: Watcher-„Brain-Modus" (echtes resume/best-Netz end-to-end statt der
  ~14h-alten Skill-Snapshot-Umschaltung in `get_watcher_model_path`).

### V15.2b — Nachjustierung nach 40-Min-Test (2026-09-04)
- Starter-Skill stieg mit V15.2 von **0 → 538** (Gate 880), plateaute dann.
  Ursache: `SPECIALIST_STEP_COST = -0.10` zu hart (−500 über eine 5000-Step-
  Episode, bevor Erfolg möglich → Netz resigniert, Agenten hocken in Alabastia).
- `SPECIALIST_STEP_COST` −0.10 → **−0.02**, `INTRO_STEP_COST` −0.02 → **−0.01**.
- Frames 16/8 → **12/6** (12 reicht meist für echten Kachel-Schritt, weniger
  Umgewöhnung fürs 4/4-trainierte Champion-Netz, 2× statt 3× langsamer).
- Learner NICHT resettet — die ~500k Steps „wie komme ich ins Labor" behalten.
- Anti-Loop (`REPEAT_TILE_PENALTY -0.05`, `STUCK_SAME_POS 60`) unverändert.

### V15.2 — Timing-Fix + Tempo-Reward + Learner-Reset (2026-09-04)
- **Tastendruck 4/4 → 16/8 Frames.** War hart kodiert als `for _ in range(4)`
  in `PokemonFireRedEnv.step()` und als `ACTION_HOLD_FRAMES/RELEASE = 4` in
  `watch.py`. 4 Halte-Frames *drehen* die Figur in FireRed nur — ein echter
  Kachel-Schritt braucht ~16. Die Hälfte aller „Geh"-Aktionen war wirkungslos
  → Agenten drehten auf der Stelle; kurze START-Taps wurden in der
  Namensvergabe geschluckt. Jetzt Klassen-Konstanten `ACTION_HOLD_FRAMES = 16`
  / `ACTION_RELEASE_FRAMES = 8` in `pokemon_env.py`, identische Modul-Konstanten
  in `watch.py` (MÜSSEN gleich bleiben). ~3× langsamer Wall-Clock/Step,
  Step-basierte Timeouts unberührt.
- **Echte Step-Kosten für Meilenstein-Spezialisten.** `INTRO_STEP_COST`
  −0.002 → −0.02; neu `SPECIALIST_STEP_COST = -0.10` für Rollen
  intro/stairs/exit/starter (und `full` vor dem Starter). Reward = fester
  Zielbonus − feste Kosten/Step → schnellster Lauf = höchster Reward. Ziel-Boni
  (stairs +150, exit +500/+800, starter ~1550) bleiben Größenordnungen größer,
  Erfolg schlägt immer den Timeout.
- **Anti-Kreis.** `V9_EXPLORER_REPEAT_TILE_PENALTY` 0.0 → −0.05 (bekannte
  Kachel kostet jetzt), `V9_EXPLORER_NEW_TILE_BONUS` 2.0 → 1.0,
  `V9_STUCK_SAME_POS_STEPS` 240 → 60.
- **Learner-Reset.** `pokemon_model_best.zip` (Champion v6) → `…_resume.zip`.
  Die ~23 Mio regredierten Steps verworfen, Training läuft vom bekannt-guten
  Champion + neuem Timing weiter.
- **Status-Terminal** (`tools/pkmai_status.py`): Einzel-Agenten-Dump raus, neuer
  `SHIGGY/STARTER`-Block (Vault/880-Gate, Live-Health, Lifetime-%, Champion
  Full-Starter %, Live-Agentenzahl + wie viele Schiggy haben, Timeout-Histogramm).
- **Start-Skripte**: `start_all.sh` startet Cloudflare + Mapper standardmäßig
  nicht mehr (`--cloudflare` / `--mapper` zum Reaktivieren). Dashboard bleibt
  HTTP-only auf :8001 — immer `http://` benutzen, nie `https://` (sonst
  `Invalid HTTP request` im uvicorn-Log).
- **Neustart: Trainer + Watcher** (beide laden `pokemon_env` bzw. die
  Timing-Konstanten beim Start).

### V15.1 — `stage_`-Resume-Bug behoben (pokemon_env.py)
- **Warum:** die Resume-Flag-Logik in `reset()` kannte nur
  `progress_/maps_/outdoor_/level_/badge_`, nicht `stage_`. Ein Resume von
  `stage_4` (Vertania-Markt, Bank 5) oder `stage_5` (Eichs Labor, Bank 4)
  bekam `left_house_confirmed` nicht gesetzt (der Overworld-Zweig in
  `_set_baseline_from_info` greift nur bei Bank 3) → der early-house-Failsafe
  hätte den Run per `stairs_timeout` / `early_house_hard_cap` gekappt. Dazu
  hätte der Agent beim Rausgehen erneut `starter_outdoor:+150` kassiert.
- **Fix:** Flag-Logik in `_apply_curriculum_resume_flags()` ausgelagert,
  `stage_` in die Post-Haus-Präfixe aufgenommen. `stage_N` (N≥2) setzt jetzt
  `stairs_down/left_house/left_house_confirmed` + `starter_outdoor_rewarded`.
- Neue Unit-Tests (`CurriculumResumeFlagTests`). Noch nicht scharf: es gibt
  bislang keine `stage_N`-Savestates (globale Stufe 2). **Neustart: Trainer**
  (Subprozess-Envs laden `pokemon_env` beim Spawn).

### V15 — FireRed-Story-Curriculum + Frame-Stack + frisches Gehirn
- 4 Bilder statt eines Standbilds; 31 RAM-/Nav-/Storyfeatures.
- 32 headless Envs × 128 Schritte = 4096 Samples je PPO-Rollout.
- Deutsche `BPRD`-ROM: Paket-/Eich-/Old-Man-Status direkt aus den
  FireRed-SaveBlock-Variablen, mit Plausibilitätsprüfung.
- Explizite Stufen 0–9; Stage-Savestates validieren aktuelle Map und Storywert.
- Weltrollen erhalten keinen Map-/Warp-Reward; Journey-Routen deaktiviert.
- Starter-Erfolg erst draußen, nicht beim Erhalt im Labor.
- Rollen: World-Push zuerst, Kampf nach Paketabgabe, Orden ab Wald.
- Neue Unit-Tests decken Stufen, Story-Checkpoints und Rollenverteilung ab.

### V14.1 — Champion-Log NameError behoben (train.py)
- Beim ersten V14-Frontier-Fortschritt referenzierte die Logmeldung noch die
  entfernte Variable `maps` und beendete den Trainer mit `NameError`.
- Logausgabe verwendet jetzt den tatsächlichen V14-Wert `stage={wstage}`.
- Das Resume-Modell war vor dem Fehler bereits sicher gespeichert.

### V13.4 — NORD-KORRIDOR-RAMPE (pokemon_env.py)
- **Warum:** in 30 Mio Steps hat KEIN Agent Route 1 durchquert. Kein „langsam",
  sondern strukturell: (1) 1 Standbild = keine Bewegungsinfo (Route-1-Absätze!),
  (2) ~60 richtige Schritte fast ohne Zwischenreward, (3) 30M Steps Gewohnheit.
- **Fix:** `NORTH_CORRIDOR_MAPS = {(3,0) Alabastia, (3,19) Route 1}` — auf genau
  diesen geraden Vor-Wald-Strecken gibt es `+1.2` pro neuer nördlichster Y-Reihe
  (nicht farmbar, nur echte neue Reihe). Aus den 60 blinden Schritten wird eine
  Belohnungsrampe Richtung Vertania. **Der Wald ist BEWUSST NICHT dabei**
  (Labyrinth → reine Exploration). Gilt nur für progress/battle/level/full.
- `outdoor_N`-Checkpoint-Gate: 60→30 Schritte, ABER auf einer Korridor-Map nur
  wenn `y <= 20` (echt weit im Norden) → kein „outdoor_2 am Südrand" mehr.

### V13.3 — outdoor_1 war GARBAGE + Indoor-Farm-Nerf + 128 Envs + Web-Panel
- **Fund nach V13.2-Start:** 82/96 Agenten standen DRINNEN. Ein frisch von
  `outdoor_1` gestarteter Agent landet bei Bank 4, Map 0, kein Starter →
  `outdoor_1.state.gz` ist ein Alabastia-Innenraum, kein Aussen-State
  (Glitch-RAM-Read beim Warp).
- **Indoor-Farm:** `progress` bekam `+2.0`/Kachel auch drinnen → Labor abgrasen
  = +100. Jetzt draußen `+2.0`, drinnen `+0.35`; dazu `-0.05/Schritt` für
  Progress/Battle/Level mit Starter die >150 Schritte drinnen hängen.
- `_best_progress_milestone`/`_choose_episode_start`: `outdoor_N` nur als Resume
  wenn **N ≥ 2**. Fallback ~80% `starter_outdoor` (Bank-3-geprüft beim Save →
  verlässlich), ~20% `starter`. `outdoor_1`-Dateien beim Neustart gelöscht.
- `NUM_ENVS 96 → 128`, `PPO_N_STEPS 40 → 32` (Rollout 4096).
- **Web:** neues `🧭 FLOTTEN-STATUS`-Panel im Graphen-Tab (Welt-Tiefe + Balken +
  Map-Name, draußen/drinnen/Kampf, Tiefen-Durchbrüche, K.O./Schaden, Rollen,
  Live world_depth-Events). `/api/state` → `fleet`-Block.
- Watcher: `btf/c1/c2/c3`-Debug-Leiste raus.

### V13.2 — Flotte auf 96 Envs, Rollen skalieren automatisch (train.py, pokemon_env.py)
- **Grund:** 120 Envs auf M4 Max (12 P-Kerne) = ~8× überbucht, Load 60, jeder
  Emulator lief mit ~60–70 % Speed. `NUM_ENVS 120 → 96`, `PPO_N_STEPS 32 → 40`
  (Rollout-Buffer bleibt 3840).
- **`_agent_role` / `_choose_episode_start` / `_is_long_full_probe`** rechnen
  jetzt **relativ zur Flottengröße** (`self.n_envs`, via Konstruktor) statt fester
  Slot-Zahlen für genau 120. NUM_ENVS ändern reicht künftig.
- Phase 5 @ 96: intro 2 / stairs 3 / exit 3 / starter 4 / battle 11 / level 7 /
  **progress 61** / **full 5**.
- **`full` NICHT entfernt** (User-Frage): nur Full-from-Beginning-Runs können den
  Champion befördern (Frontier / Milestone / Recent-Eval) **und** den Rollback-
  Punkt frisch halten. Ohne `full` friert der Champion für immer ein. Deshalb 5
  behalten statt 0. `min_full_episodes 12 → 8`, `min_eval_episodes 24 → 16`.
- **GPU:** kann keine Emulatoren rechnen (mGBA = sequenzieller CPU-Code). Die
  M4-Max-GPU macht bereits die NN-Forward/Backward-Pässe (MPS). Mehr Clients als
  ~1–2× Kernzahl bringt nichts — reine Context-Switch-Kosten.

### V13.1 — Watcher: nie wieder Treppe nach dem ersten Draußen (watch.py)
- **Bug (User beobachtet):** Watcher lief versehentlich ins Rivalen-/Schwesterhaus
  (das Gebäude über dem Labor). Routing wählte „stairs" (weil `initial_indoor_room`
  nicht gelockt + `house_rooms` leer) → Treppen-Skill lief nach Norden gegen die
  Wand → kam nicht mehr raus.
- **Fix:** neues stickiges Flag `watcher_ever_outdoors`. Sobald der Watcher EINMAL
  auf Bank 3 war, ist „stairs" komplett vom Tisch — jedes Gebäude ohne Starter →
  Exit-Skill. Nur der Anti-Loop-Reset (Spielanfang) löscht das Flag.
- Info-Zeile im Watcher zeigt jetzt `Aussenwelt` / `Innen (Bank N)` / `IM KAMPF`
  statt immer nur „Overworld".

### V13 — Welt-Tiefe zählt jetzt auch den Wald + Route-1-Fokus (pokemon_env.py, watch.py)
- **Welt-Tiefe** = jetzt „alle Maps außer Bank 4" (Bank 4 = alle Alabastia-
  Innenräume inkl. Eichs Labor). Vorher „nur Bank 3" — das hätte den Vertania-
  **Wald** (eigene Map-Bank!) und Arenen nie als Tiefe gezählt. Checkpoints
  (`outdoor_N`) bleiben bewusst nur auf Bank 3 (sicherer Resume-Boden).

### V13 — Route-1-Fokus + eskalierende Tiefen-Belohnung (pokemon_env.py, watch.py)
- **Eskalierender Tiefen-Reward:** `world_depth` gibt jetzt `300 * outdoor_count`
  statt flat 300 → Route 1 +600, Vertania +900, Route 2 +1200, Wald +1500.
  Der erste Agent der durchbricht bekommt eine massive Prägung; Checkpoint +
  journey_route ziehen die anderen 68 nach.
- **~85% der Progress-Agenten** starten direkt am tiefsten `outdoor_N` (statt
  50/50 mit `starter_outdoor`).
- **battle/level-Rollen** starten auf Route 1 (Gras = wilde Kämpfe, Erkennung
  bestätigt) statt im Labor (Rivalenkampf-Signal unzuverlässig).
- Watcher V12.6: merkt sich das erste Nicht-Haus-Gebäude als Eichs Labor →
  Rivalenhaus & Co. werden mit Exit-Skill wieder verlassen.
- `pkmai_status.py` zeigt jetzt `KAEMPFE`-Zeile.

### ML-Hebel für „schneller lernen" (Kontext, noch nicht umgesetzt)
- **Frame-Stacking (4 Frames)** = größter Hebel. Gibt der Policy Bewegung/Kontext
  → viel weniger verwirrte/unnötige Schritte. Braucht Obs-Änderung + Netz-Reset.
- Rendern von Clients bringt NICHTS fürs Lernen (nur Emulator-Overhead). Weniger
  Envs = weniger Erfahrung/s = langsamer. 120 parallel ist richtig.
- `target_closer/farther ±2.5` feuert sehr oft → könnte „am Ziel oszillieren"
  begünstigen; Kandidat zum Reduzieren.

### V12.4 — mehr Exploration, sauberere Checkpoints (pokemon_env.py, watch.py)
- **Bug:** alle 50 progress-Agenten resumten von `outdoor_2` — der State saß am
  Route-1-Südrand → Agenten liefen sofort zurück nach Alabastia, manche fielen
  bis ins Schlafzimmer. Deshalb „keiner erkundet oben".
- Fix: `outdoor_N`-Checkpoint wird erst gespeichert wenn der Agent **≥ 60
  Schritte am Stück tief auf der neuen Außen-Map** steht (nicht am Rand, nicht
  bei Glitch-Map-Read). `_best_progress_milestone`: Hälfte der Agenten startet
  von `starter_outdoor` (sauber), Hälfte vom tiefsten `outdoor_N`.
- Phase 5: progress **50 → 68**, full **32 → 16**, battle 12, level 10.
- Battle-Adresse **gefunden** (Probe): `in_battle` = 146376 (0x02023BC8, Wert 1).
- Watcher V12.3: Haus-Räume werden gemerkt → eigenes Haus wird nicht mehr mit
  Eichs Labor verwechselt.
- **AKTION nötig:** alte `outdoor_1/2.state.gz` löschen (sitzen schlecht).
  `full_stairs/full_exit_permille` (0.1 %) sind kaputte End-Positions-Metriken,
  harmloses Rauschen - die `full`-ROLLE bleibt (nötig für Champion-Frontier).

### V12 — Kampf + Level Rollen, Watcher-Startraum-Fix (pokemon_env.py, watch.py)
- Phase-5-Verteilung: **battle (10) + level (8) Rollen wieder aktiv** (für
  Vertania-Wald wilde Pokémon + Brock). 2/6/6/6 Erhaltung, 50 progress, 32 full.
- Watcher-Bug: transienter RAM-Read während Intro-Cutscene konnte den
  „Startraum" auf eine falsche Map locken → Routing schickte den Watcher in die
  Exit-Skill statt Treppe → hing ewig im Schlafzimmer. Fix: Startraum erst nach
  bestätigtem Intro-Ende + 3 stabilen Frames festnageln; bis dahin default `stairs`.
- Champion `max_maps` 6→1 (Skalen-Mismatch nach V11.3 Außen-Zählung, hatte
  Champion eingefroren).
- Watcher in der Agentenliste: **nicht mehr forciert/gehighlighted** — sortiert
  normal nach ID.
- `#watcher-view` fehlte `display:none` → Live-JPEG lag über der Overworld-Karte.

### CODE-REVIEW Findings (2026-09-03)
1. **Battle-Erkennung (`in_battle` RAM 147074 = immer 0)** — der eigentliche
   Blocker. `fled_battle`-Malus feuert nie, `battle_stats` bleibt 0. ABER:
   `enemy_damage`/`enemy_faint`-Rewards laufen über einen HP-Fallback
   (`read_enemy_party` + Gegner-HP sinkt) — die funktionieren auch ohne
   `in_battle`. → `enemy_faints: 0` heißt: Agenten **gewinnen keine Kämpfe**
   (fliehen / Kampfmenü-Navigation scheitert). battle-Rolle (V12) trainiert genau das.
2. **Keine Item-/Geld-/HM-Logik** — für „Spiel komplett durchspielen" nötig
   (Route-1-Potion, Wald-Items, VM Zerschneider für Vertania-Wald-Ausgang…),
   aber RAM-Adressen (Bag, Money) fehlen in `data.json`. Muss geprobt werden.
3. `isW` in web_stream.py nach De-Highlighting ungenutzt (harmlos).
4. `_agent_role` nutzt hartkodiert `rank % 120` — bricht bei NUM_ENVS ≠ 120.

### V11.3 — Welt-Tiefe = nur Außen-Maps (pokemon_env.py)  ← AKTUELL
- **Bug gefunden:** `max_episode_maps` zählte ALLE besuchten Maps inkl.
  Innenräume. „6 Maps" kam durch einen optionalen **Rivalenhaus**-Besuch in
  Alabastia zustande — **Vertania (3,1) wurde NIE erreicht**, Route 1 (3,19)
  aber von 60 Agenten. `maps_6.state.gz` lag wahrscheinlich im Rivalenhaus →
  Progress-Agenten resumten dort → landeten wieder in Alabastia.
- Fix: Welt-Tiefe zählt nur `bank == 3` (Außen). Neue `outdoor_N`-Checkpoints
  werden nur draußen gespeichert (garantiert guter Spot). `outdoor_2` = Route 1,
  `outdoor_3` = Vertania …
- `_best_progress_milestone` bevorzugt jetzt den tiefsten `outdoor_N` →
  „schnell zurück an die weiteste Stelle" (genau der User-Wunsch).
- **Cleanup nötig:** alte `maps_*.state.gz` + `progress_*.state.gz` löschen
  (teils im Rivalenhaus). Progress-Agenten fallen dann sauber auf
  `starter_outdoor` zurück, bis `outdoor_2` neu gespeichert wird.
- **Neustart: Trainer.**

### V11.4 — Battle-Erkennung + Kampf-Rewards (pokemon_env.py, watch.py, data.json)
- **Ursache:** `in_battle` (RAM 147074) lieferte immer 0 → keine Kampf-Rewards,
  kein Flucht-Malus, AI "wusste" nie dass sie kämpft.
- Fix: `gBattleTypeFlags` (0x02022FEC → EWRAM-Offset **143340**, u4) als neues
  Feld `battle_flags` in `data.json`. Env + Watcher: `in_battle = 1` wenn altes
  Feld ODER `battle_flags` != 0.
- `fled_battle` -1 → **-3** (Flucht stärker bestraft).
- Schadens-Moves (Kratzer statt Kampfschrei): `ENEMY_DAMAGE_REWARD_PER_HP=0.75`
  belohnt jeden abgezogenen Gegner-HP bereits — greift ab jetzt, weil die
  Erkennung endlich funktioniert. Kein neuer Code nötig.
- **Prüfen nach Neustart:** `Kaempfe`-Zähler im Watcher-Fenster steigt,
  `/api/state` `battle_stats.started` > 0. Wenn weiter 0 → RAM-Adresse ist für
  diese ROM-Speicherlayout falsch, dann RAM-Diff-Probe.
- **Neustart: Trainer + Watcher.**

### V11.3b — Web-Dashboard an V11 angeglichen (web_stream.py)
- `/api/state` liefert jetzt `world_depth` (echte Außen-Map-Tiefe) +
  `deepest_outdoor_checkpoint`. Neue KPI-Kacheln im Graphs-Tab.
- Globale Overworld-Karte: Hintergrundbild `assets/maps/kanto_map.png` FEHLT
  (404). Karte lädt es jetzt optional — fehlt es, bleibt sie mit dezentem
  Raster + Live-Tiles nutzbar. Bild muss der Nutzer selbst bereitstellen
  (Copyright), oder wir bleiben bild-frei.
- **Neustart: nur Webserver.**

### Offene Bugs
- **Battle-Erkennung kaputt:** `in_battle` (RAM-Adr. 147074 in
  `local/custom_integrations/.../data.json`) liest immer 0 → `battle_stats`
  überall 0, keine `enemy_faint`/`enemy_damage`-Rewards. **Für Brock nötig.**
- Dashboard: „Battles"-Kachel 0 % (Folge des obigen Bugs).
- Wunsch: Skill-Chart-Filter (alle / einzelne Linien) — TODO web_stream.py.

### V11.2 — Watcher-Anzeige aufgeräumt (watch.py)
- Info-Zeile modernisiert: `Bank X / Map Y (x,y) - Overworld/IM KAMPF -
  Kaempfe N - Skill: STARTER - Reward +0.02`. **START-spam raus.** „Battle"
  zeigte nur 0/1 (Zustand) → jetzt echte Kampf-Anzahl aus
  `watcher_battle_stats["completed"]`. Aktive Skill wird angezeigt.
- **Watcher startet weiterhin BEWUSST immer vom Spielanfang** (End-to-End-Demo
  des Hirns). Anti-Loop-Reset ebenfalls → Spielanfang. Savestates sind nur
  für die Trainings-Clients. (Kurz getestete Savestate-Resume-Idee wieder
  verworfen — Watcher soll den ganzen Run zeigen.)
- **Neustart: Watcher.**

### V11.1 — Fixes nach erstem Lauf (pokemon_env.py, train.py)
- **Labor-Ziel-Bug:** `_target_coords_for_stage` zog einen Agenten OHNE Starter
  in Eichs Labor direkt zur Ausgangstür (+2.5/Step). Watcher stand nach 3k Steps
  am Auswahltisch und lief nach unten raus statt einen Pokéball zu nehmen. Fix:
  im Gebäude erst dann Richtung Ausgang zeigen, wenn `has_starter` — vorher kein
  Ziel, reine Exploration (läuft von selbst an den Tisch).
- `training_phase`-Label im Status/Dashboard auf V11-Phasen umgestellt
  (`1_intro` … `5_world_explore`).
- Step-Zähler wird bei V11-Reset (keine champion_score.json) auf 0 gesetzt.
- **Neustart: Trainer.**

### V11 — CLEAN EXPLORE (pokemon_env.py, train.py)  ← AKTUELL
Reward-Logik ML-optimiert nach dem „Pokémon Red RL"-Prinzip:
- **Exploration ist gratis und dominant:** `V9_EXPLORER_NEW_TILE_BONUS` 0.75 → **2.0**
  pro neuer Kachel. Wiederholung = 0 (nicht bestraft).
- **Straf-Suppe raus:** `GAMEPLAY_STEP_COST` 0, alle edge-revisit-Penalties 0,
  `indoor_stall` praktisch aus (6k/15k), `V9_STUCK` -2.0 → -0.5 & erst nach 240
  Steps, START-Penalties 0, `exit_route`/`journey_route`-Reverse-Penalties 0,
  `fled_battle` -5 → -1, stage-timeout -5 → -1.
- **Stage-Caps großzügig:** intro 2500→6000, treppe 3500→9000, exit 10k→18k
  (langsame Policy muss die Belohnung am Stufenende erreichen können).
- **NORTH_PUSH KOMPLETT ENTFERNT** — war ein hartkodierter Richtungs-Prior,
  verfälschte die Lernkurve und ist im Vertania-Wald direkt falsch (Labyrinth:
  rechts/hoch/links/runter). Reiner Neue-Kachel-Bonus macht das richtungs-neutral.
- **Entropie 0.035 → 0.05** (Re-Heat, um die alte Policy aus der Sackgasse zu holen).
- **V11 Sequentielles Bootcamp** (`_agent_role`): intro → treppe → exit → starter → Welt.
  ~90 % der Flotte auf der aktuellen Stufe, kleiner Erhaltungs-Sockel + Probes.
  Auto-Umschalten sobald `skill_vault_scores` die Stufe ≥ 88 % misst. Ab
  „starter ≥ 88 %" → Phase 5: freie Welt-Exploration, Horizonte 8k/16k/28k.
- Reset-Tool: `tools/v11_reset.sh`
  - **soft** (Standard): Skills + `skill_vault_scores.json` + Curriculum-Savestates
    + Routen BLEIBEN. Learner + Champion + latest werden aus
    `pokemon_skill_starter_best.zip` neu geseedet (kann Frühspiel, hat die
    Sackgasse nicht). `champion_score.json`, `model_version.json`,
    `exploration_memory/agent_*.json`, `global_progress.json`, `training_stats/*`
    → weg (das sind die „verfälschten" Kennzahlen vom festgefahrenen Lauf).
    Bootcamp → direkt Phase 5.
  - **--hard**: zusätzlich Skill-zips + skill_vault_scores + Savestates weg →
    Phase 1 (Intro) von Null.
- **Voll-Backup vor V11:** `brain_backups/PRE_V11_20260903_154716/` (46 MB, alle 9 Modelle).
- PPO: LR 7.5e-5, n_steps 32, ent_coef **0.05**.
- **Neustart: Trainer + Watcher** (nach `tools/v11_reset.sh`).

### V10.33 — Wege-Gedächtnis (pokemon_env.py)
- Die bereits existierende „exit_route"-Mechanik (Haus-Ausgang wird als
  bestätigte Kanten-Spur gemerkt, +Reward fürs Nachlaufen) auf den **gesamten
  Weg NACH dem Haus** erweitert: `curriculum_shared/journey_routes/`.
- Jede Kante, die ein Agent nach `left_house` geht, wird aufgezeichnet. Bei
  `starter_outdoor`, `next_outdoor_map` (Route 1) und jedem `global_depth`-Rekord
  wird die Spur committet. Sobald ≥2 Agenten dieselbe Kante bestätigen →
  `+1.5` fürs erste Begehen, Strafe für Zurück/Loop. Der erste Pionier legt
  den Weg, alle anderen kriegen eine Brotkrumen-Spur.
- **Neustart: Trainer.**

### V10.32 — Watcher: raum-basiertes Skill-Routing (watch.py)
- `watcher_gameplay_ready` jetzt **sticky** (ein einzelner untrusted RAM-Read wirft
  den Watcher nicht mehr zurück in die Intro-Skill).
- **Routing nach aktuellem Raum statt monotoner Flags:** im Startraum (2F) → immer
  Treppe (kann „runter"), in anderem Innenraum (1F) → immer Exit (kann „raus"),
  draußen ohne Starter → starter, Starter+drinnen → exit (Labor raus),
  Starter+draußen → progress. Behebt: Treppen-Dauerloop, wenn die Exit-Skill ins
  Straucheln kommt und wieder hoch nach 2F läuft — vorher blieb Routing auf „exit"
  und die Exit-Policy war auf 2F verloren. Jetzt macht jede Policy ihren lokalen Job.
- **Neustart: nur Watcher.**

### V10.31 — Labor-Ausgang-Gradient (pokemon_env.py, train.py)
- **`starter_outdoor` +150**: erster Overworld-Schritt MIT Starter. Vorher gab es
  für „Labor verlassen" null Reward → kein Agent hat es je gelernt.
- Starter-Spezialist-Episode endet **nicht mehr beim Erhalt des Starters**, sondern
  erst DRAUSSEN (40 % Bonus fürs Kriegen, 60 % fürs Rausgehen). `starter_exit_stall`
  4k-Timeout für „im Labor festgefahren".
- Neuer Curriculum-State `starter_outdoor` → halbe Progress-Flotte startet „draußen
  mit Starter", andere Hälfte drillt weiter den Laborausgang (`rank % 2`).
- Entropie 0.025 → **0.035** (Policy klebt am Alabastia-Rand).
- **Neustart: Trainer.**

### V10.30 — Vertania-Durchbruch: Verteilung + gestaffelte Horizonte (pokemon_env.py)
- Rehearsal auf Erhaltungs-Sockel: **2 intro / 8 stairs / 8 exit / 6 starter**,
  Rest **58 progress + 38 full**.
- Progress-Horizonte gestaffelt: `(rank%120)%3` → **8k / 16k / 28k** Steps.
- `_is_long_full_probe`: full-Slots 82-99 (18) = saubere 32k-Probes.
- `maps_4` / `maps_5` Curriculum-Saves (vorher erst ab 6).

### V10.29 — NORTH_PUSH (pokemon_env.py)
- Kleiner, gedeckelter, nicht farmbarer Bonus für Nord-Fortschritt (neuer
  Y-Bestwert je Overworld-Map), `+1.5`/Tile, max `+60`/Map.
- **Nur draußen** (`bank==3`), **nur mit Starter**, **nur** solange global
  `max_episode_maps < 8` → schaltet sich selbst ab.

### V10.28.1 — Champion-Schutz repariert (train.py)
- Zeitbasierten Stale-Champion-Fallback **entfernt** (ersetzte guten Champion durch
  frühgame-vergessliche Policy, nur weil Zeit verging).
- `_protected_regression` neu: echte Tiefe (Orden/Maps/Level) hebt Schutz auf;
  sonst Schutz bei echtem `full_starter_permille`-Einbruch. `full_episodes < 8` → nie
  befördern.
- `_metrics_floor`: solange keine echten Beginning-Full-Runs abgeschlossen sind,
  bleiben bekannte Champion-Raten als Untergrenze (kein 0-Clobbern nach Neustart).
- `min_full_episodes` 24 → 12. `champion_score.json` einmalig ent-vergiftet
  (Backup: `brain_backups/SAFETY_v155_champion_20260903_pre_stale_fix/`).

### V10.28 — Rehearsal-Rebalance (überholt von V10.30)
### V10.27A/B/C — Team-Rewards, Indoor-Stall-Escape, große Map-Rewards, Entropy-Re-Heat

---

## Web-Dashboard (web_stream.py) — separate Spur

- Mobile: `<meta viewport>` ergänzt, alles fester vertikaler Scroll, nur Leaflet-Karte
  behält feste Höhe. Alle Panels sind auf dem Handy feste Kacheln (kein Drag/Minimize).
- Graphs-KPIs: „Intro/Treppen Skill" ersetzt durch Champion Steps / Learner−Champion /
  Full Starter. Live-Brain + Champion-Karten auf Nicht-Map-Views ausgeblendet.
- Agenten-Filter (Map-Ansicht): Rolle / Map / Starter / Stage / Sortierung + Reset.
- Watcher immer erste Zeile + Label „👁️ Watcher". Kopfzeile (Team/Orden) zeigt immer
  den Watcher; Klick auf Agent wechselt nur das Detail-Panel (kein Full-Rerender mehr).
- **Neustart: nur Webserver.**

---

## Nächste Schritte / offene Punkte

1. **V10.31–V10.33 laufen lassen (~2 h)** und in `pkmai_status.py` prüfen:
   `starter_outdoor:` / `journey_route_edge:` in Reward-Events? `Agenten @Map`
   verschiebt sich von `4,3` weg? `max_episode_maps` → 6?
2. **Wenn nach 2 h nichts → V10.34 „EXPLORATION MODE" (großer Hebel, noch offen):**
   Der Vergleich mit Peter Whiddens „Pokémon Red RL" (kam schnell bis Vertania-Wald)
   ist berechtigt. Dessen Rezept: **eine dominante Belohnung = neue Tiles/Koordinaten**,
   plus Level/Events/Badges — und **KEINE Straf-Suppe**. PKMAI bestraft aktuell
   Backtracking / Loitern / „verschwendete" Schritte (step_cost, indoor_stall,
   edge_revisit, repeat_edge, exit_route_reverse, START-spam …). Netto ist der
   Gradient „ins Unbekannte laufen" schwach/negativ → Policy ist auf „sicher am
   Alabastia-Rand kreisen" konvergiert (klassisches Over-Shaping).
   **Vorschlag:** für outdoor progress/full-Agenten, solange `max_episode_maps < 8`:
   Straf-Terme aus, `V9_EXPLORER_NEW_TILE_BONUS` 0.75 → ~2.0, Repeat-Tile-Penalty
   aus, Stall-Truncation nur wenn wirklich keine neuen Tiles mehr. Milestone-Rewards
   (neue Map 250, depth 300, journey_route) bleiben. Danach wieder zurückschalten.
3. Alternative wenn 2 h nichts: Learner aus `pokemon_skill_starter_best.zip` neu
   seeden (Skills + Champion bleiben unangetastet).
4. Nach erstem `max_episode_maps ≥ 6`: NORTH_PUSH-Deckel prüfen, Horizonte/Rehearsal
   nachjustieren. Sobald Vertania stabil: Battle-/Level-/Badge-Rollen hochfahren.

## Architektur-Review: kann ein FRISCHES Hirn hiermit das Spiel lernen?

Kurzantwort: **Frühspiel ja (Skill-Vault beweist es). Ab Starter kämpft das
Design gegen sich selbst.** Wenn eh resettet wird, ist das DIE Gelegenheit, die
Dinge zu fixen, die sowieso eine Obs-/Reward-Änderung brauchen.

**Schwächen fürs Lernen von Null:**
1. **Kein Frame-Stacking.** Obs-Bild ist `(1,64,64)` — EIN Graustufen-Frame.
   Kein Bewegungs-/Animations-/Menü-Kontext. Red RL nutzte 3 Frames. → `(4,64,64)`.
2. **Keine Recurrence.** Reaktive Policy, kein Gedächtnis „wo kam ich her". Navi
   ohne Speicher ist hart; RAM-Navi-Features + V10.33-Wege-Gedächtnis mildern das.
3. **Reward-Suppe.** ~36 Event-Typen, viele kleine widersprüchliche Terme. Der
   Explorations-Bonus (0.75) konkurriert mit ~10 Straf-Termen (step_cost,
   edge_revisit ×4, indoor_stall ×2, v9_stuck −2.0, exit_route_reverse …). Netto
   lernt ein frisches Hirn: „bloß nichts riskieren".
4. **Aggressive Early-Truncation.** Stage-Caps 2500/3500/10000 setzen Kompetenz
   voraus. Ein frisches Hirn wird abgeschnitten, bevor es die Belohnung sieht →
   kann die Stufe nie lernen. (Nur „long full probes" ausgenommen.)
5. **Curriculum-Resume ist GUT** — frisches Hirn übt Segmente statt Kaltstart. Behalten.

**Reset-freundliches V11 (echter Refactor, ~1 Tag + Retraining):**
- Obs-Bild → `(4,64,64)` Frame-Stack (braucht eh Reset).
- Reward-Kern auf ~8 Terme: `+neues Tile` (dominant), `+neue Map`, `+neuer Warp`,
  `+Level`, `+Event/Orden`, `+HP-Heilung`, `−HP-Verlust`, `−Party-Wipe`.
- Weg: step_cost, alle edge-revisit-Penalties, indoor_stall, v9_stuck (oder nur
  bei echtem Dauer-Stillstand), START-spam-Penalty, exit_route_reverse.
- Behalten: Curriculum-Resume, 4+4-Frame-Timing, Champion-Schutz (vereinfacht),
  journey_route-Gedächtnis.
- Episode: 8k–16k Steps, Truncation nur bei „2000 Steps kein neues Tile".

## Warum ist es langsamer als „Pokémon Red RL"?

- Red RL: ~5 Reward-Terme, **Exploration dominiert**, simple Obs (Frame-Stack + paar
  RAM-Werte), kurze Episoden, kein Curriculum-Mikromanagement, keine Skill-Routing.
- PKMAI: 28 RAM-Features, ~40 Reward-Terme (viele negativ), Curriculum-States,
  Skill-Vault, Champion-Schutz, Rollen-Allokation. Mächtiger, aber die vielen
  Straf-Gradienten dämpfen Exploration. → V10.34-Vorschlag oben.

## Prozess-Steuerung (welcher Neustart wofür)

| Geändert | Neustart |
|---|---|
| `pokemon_env.py`, `train.py` | Trainer (Ctrl+C, auf `💾 Resume-Stand gespeichert` warten) |
| `watch.py` | Watcher |
| `web_stream.py` | nur Webserver |

Befehle:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python ~/pokemon_ai_project/src/train.py
/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python ~/pokemon_ai_project/src/watch.py
/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python ~/pokemon_ai_project/src/web_stream.py
/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python ~/pokemon_ai_project/tools/pkmai_status.py
```
