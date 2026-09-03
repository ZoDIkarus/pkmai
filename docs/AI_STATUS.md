# PKMAI — Live Work Log

**Dieses Dokument ist der aktuelle Arbeitsstand.** Jede AI / jeder Entwickler
liest das ZUERST (vor `README.md` und `AI_HANDOFF.md`), trägt neue Änderungen
oben ein und lässt den Verlauf stehen. Kurz halten, keine Romane.

Laufende Runtime-Werte immer aus `tools/pkmai_status.py` lesen, nicht hier abschreiben.

---

## Wo wir gerade stehen (2026-09-03)

**Das Ziel:** Full-Champion soll zuverlässig von Spielanfang → Starter → Alabastia
raus → Route 1 → **Vertania City** kommen. Danach Kämpfe / Level / Orden 1.

**Der Engpass, Stand jetzt:** Nach 22 Mio Steps hängt `max_episode_maps` global bei
**5** (= alle Alabastia-lokalen Maps: Schlafzimmer 2F, Haus 1F, Eichs Labor,
Rivalenhaus, Alabastia außen). Route 1 (`bank 3, map 19`) wurde in 22 Mio Steps
**1×** berührt, Vertania (`3,1`) **nie**. Laut `pkmai_status.py` sitzen ~60–89 von
120 Agenten dauerhaft **in Eichs Labor** — die eigentliche Wand ist
**„raus aus Eichs Labor nach dem Starter"**, nicht die Route-1-Durchquerung.

**Champion:** v158 @ ~21,7 Mio Steps. Advanct nur noch über Frontier-Publishes
(echte neue Tiefe). Seit V10.28.1 **nicht mehr durch eine schwächere Policy
ersetzbar** — Skill-Vault + Champion liegen getrennt vom Learner.

---

## Änderungs-Log (neueste zuerst)

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
