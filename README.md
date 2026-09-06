# PKMAI — Pokémon FireRed AI by Alex

PKMAI is an experimental reinforcement-learning project that trains a PPO agent to play Pokémon FireRed with Stable-Retro. It combines visual input, RAM-derived navigation features, persistent exploration memory, curriculum states, a live watcher and a browser dashboard.

> This repository does not contain a Pokémon ROM or proprietary game assets. You must provide your own legally obtained game data and local Stable-Retro integration.

## Current training and watcher behavior

Only the trainer learns. The watcher evaluates current `pokemon_model_resume.zip`
snapshots, independently of champion promotion. The protected fallback is stored
as `pokemon_model_champion.zip`. Snapshot publication is atomic.

Reward model as of **V19 BROCK RUSH** (see the dated section in `docs/STATUS_TODO.md`
for the full list): the push forward comes from `STAGE_ADVANCE_REWARD` (+250 per new
world-stage, new episode best only) and `TARGET_PROGRESS_REWARD` (±0.20 graph-distance
toward the transition that leads to the next stage / the city's Center / the gym —
symmetric, no compass bias). Tiles are just a flat "keep moving" trickle (Pallet 0.1
… Pewter 3.0, first 20/map/episode then 10 %, +1 fleet-once). New route 50/run, city
300/run, city building 500-once, first global stage unlock 1000-once. Pokécenter
enter 50/run, **deeper heal 500/run** (wipe respawn anchor), 1000-once; Poké Mart
100/run + 1000-once; badge 3000/run + 5000-once. Pewter/Brock split into small
episode-flagged milestones (reach Pewter with Pikachu +300, gym enter +200, Brock
battle start +500, first gym KO +300). **In a battle only continuous signals pay** —
dealt/taken damage, healing, level-up, catching; no flat KO or win bonus. Trainer
battles pay double on damage and skip the wild decay (30 % after 6 wild wins).
All edge/warp/corridor farm rewards stay off. Persistent claim history
(`reward_events.json`) keeps every one-time bonus one-time across restarts.

After a party wipe a **recovery mode** kicks in (no novelty-memory reset, so
dying is never a farm): wild-battle rewards are cut to 5 %, generic catches pay 0,
and graph-distance guidance back to the pre-wipe story front pulls at ±0.50 until
that front (or a deeper Center respawn, or a badge) is re-reached — then a
one-time +300. The −100 wipe penalty and the Center-respawn teleport are unchanged.

## 2026-09-06 — V18: per-run tile ladder, one-time fleet bonuses, battle rebalance, dashboard

Full trainer + watcher + web restart (brain kept: learner ~21.5M, champion v9).
67 unit tests pass. Live-verified in the watcher and dashboard.

**Tiles pay per run now.** The fleet-once tile bonus meant the watcher (and any
agent on long-known ground) saw no tile reward at all. First-find of a tile is
now rewarded every episode (`seen_coords`), on a hand-set ladder tied to the
tile's own map: `TILE_REWARD_BY_STAGE = {1:0.2, 2:3, 3:4, 4:5, 5:5, 6:6}` — Alabastia
is the spawn so exploring it barely pays; Route 1 onward is where it kicks in, so
the pile of tiles in Pallet never beats the higher rate ahead. Interior tiles by
city bank: Pallet houses 0.2, Vertania 2, Marmoria 3. Only the first 20 new tiles
per map per episode pay (`new_tile:…:capped` after). On top, `GLOBAL_NEW_TILE_BONUS = 1`
fleet-once for the very first agent ever to step on a tile — first foot into
Pewter = 6 + 1.

**One-time fleet-wide bonuses** (persisted in `reward_events.json`, cannot repay
after a restart), each on top of a per-run value:
- Pokécenter: +100/run to enter, +250/run for a heal at a center deeper than any
  used this run (the wipe respawn point advances), **+1000 once per center**.
- Poké Mart (Vertania `5,3`): +100/run to enter, **+1000 once** — so the brain
  learns the shop exists and can buy Poké Balls.
- First badge: +2000/run kept, **+5000 once** per badge number.

**Battle rebalance** (watcher was 45 % of steps fighting): enemy-damage reward
0.15 → 0.08 per HP; after 6 wild wins in an episode on a wild map, wild
damage/faint/win drop to 30 %; **trainer/gym battles pay ×2** and are exempt from
that decay. Catch reward 50 → 120 + `min(level,20)×4`.

**Scouts** back to 2 per checkpoint (from 5) — scouts were getting through,
full runners were not.

**Warp reward** now claims a coarse `(map_a↔map_b)` pair instead of exact
coordinates, and it is persisted (`claim_event` → `reward_events.json`) plus
pre-seeded from the navigation history, so the first global warp bonus truly
fires once ever and does not repay after a restart (the watcher was granting
+100 for Pallet↔Route 1 on every restart). Navigation data stays
coordinate-exact. No warp reward on the step a battle ends. A scout that drifts
back onto/behind its spawn stage now also earns no per-run tile reward there.

**Dashboard:** the watcher live image is gone from the Overworld-Map side column
(it only shows under the Watcher tab now); that column shows the clicked agent's
detail plus its **last 10 reward events**. Localisation: ~40 missing German→English
strings added, the `Beste`→`Best` substring bug that produced "Bestr" removed, and
the translation pass rewritten to run synchronously in the mutation observer
(`requestAnimationFrame` was paused in background tabs and silently stopped
translating) with a 700 ms safety interval.

**Deferred** to `docs/BIG_CHANGES_TODO.md`: a separate combat policy (FighterBrain)
beside the champion, and special handling for the house past Viridian Forest
(needs its map id first).

## 2026-09-06 — Geographic progression and battle/loop corrections

These changes supersede the older parcel-based stage descriptions below. No trainer,
watcher, web or mapper process was stopped or restarted during this change.

- Immutable master: the user-recorded `local/custom_integrations/PokemonFireRed-Gba/StartGame.state`
  after Oak's parcel, verified inside the lab at bank 4/map 3, x6/y4. Every full runner
  restores this exact original; only scouts load geographic checkpoints.
  The briefly prepared outdoor derivative was removed, per the latest instruction.
  Lab/indoor baseline counts as stage 1 and produces no geographic stage jump.
- Geography: Pallet 1, Route 1 2, Viridian 3, Route 2 4, Viridian Forest 5, Pewter 6.
  Parcel flags, stairs, buildings and badges do not manufacture geographic stages.
- Exactly five scouts per valid checkpoint on Route 1, Viridian, Route 2, Forest and Pewter.
  Fixed rank bands prevent reassignment when another checkpoint appears. Replacing
  a savestate never adds scouts. No Pallet scouts. Other agents continue full runs.
- Each stage independently prefers a further-north position (smaller Y), even
  with lower reward. At equal Y, strictly higher reward wins; south is rejected.
  This also applies to later full runners returning through earlier maps. Battle/wipe states are not captured.
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

## Previous release (historical)

**V17.4 — Reward-Rebalance gegen Farm-Loops, Frontier-Scouts pro Stage, kritische Checkpoint-/Warp-Bugs behoben** (2026-09-06, Nachtsession)

Ausgangspunkt war ein wiederkehrendes Live-Symptom: die Flotte lief lieber
zurück in längst bekanntes Gebiet (Alabastia, Starterhaus) statt weiter nach
Norden vorzustoßen, weil dort garantierter, risikofreier Reward wartete.
Mehrere Runden aus Live-Beobachtung → Root-Cause-Suche → Fix haben dabei vier
unabhängige, teils schon länger aktive Bugs zutage gefördert.

**Reward-Rebalance (Farm-Loops geschlossen):**
- Kanten-Reward komplett auf 0 (auch nicht mehr beim ersten Mal) — der
  Pro-Episode-Anteil resettete sich bei jedem Reset und machte bekannte
  Kurz-Loops am Spawn jede Episode neu profitabel.
- Kachel-Reward: Pro-Run-Anteil (`EPISODE_TILE_REWARD`) auf 0 — dieselbe
  Farm-Lücke wie bei Kanten, nur eine Stufe kleiner (sicheres Rumlaufen in
  bekannten Innenräumen gab bisher noch kleines Dauer-Einkommen). Nur der
  fleet-weite Einmal-Fund (`NEW_TILE_REWARD=2`) bleibt.
- Map/Stadt-Reward: kein fleet-weiter Einmal-Jackpot mehr (ging strukturell
  nur an den einen Agenten, der eine Map zuerst fand) — jetzt ein fester Wert
  pro Run für JEDEN Agenten (`EPISODE_NEW_MAP_REWARD=100`,
  `CITY_EPISODE_REWARD=250`). Gebäude-Erstfund (Reds Haus etc.) zahlt separat
  nur noch einmal für die ganze Flotte, nicht mehr pro Agent.
- Warp-Reward: kein Pro-Run-Bonus mehr für bekannte Türen
  (`EPISODE_TRANSITION_REWARD=0`), nur der fleet-weite Einmal-Fund
  (`NEW_TRANSITION_REWARD=100`) bleibt.
- Neuer, benannter `BADGE_EARNED_REWARD=2000` (vorher unbenannt inline 2500).
- Artenvielfalt-Fang und Pikachu-Wald-Bonus von fleet-weit-einmalig auf
  pro-Run umgestellt (`SPECIES_CAUGHT_FIRST_REWARD=50`,
  `PIKACHU_FOREST_CAUGHT_REWARD=1000`) — die Party resettet ja jede Episode,
  ein Lifetime-Claim hätte ab dem zweiten jemals gefangenen Exemplar nie
  wieder Anreiz gegeben, überhaupt zu fangen.
- `GAMEPLAY_STEP_COST` 5× verschärft (`-0.001` → `-0.005`), da die echten
  Ziele deutlich größer wurden, Herumstehen aber gleich billig blieb.

**Vier kritische Bugs live gefunden und behoben:**
1. **Stage-Checkpoints für Route 1/Vertania konnten nie entstehen.** Der
   feste Savestate startet mit bereits bestätigter Paketabgabe an Prof.
   Eich, wodurch `_world_stage()` (ein reiner Ratchet) ab Step 0 immer
   mindestens 5 zurückgibt — unabhängig vom tatsächlichen Standort. Der
   Checkpoint-Code nutzte diesen Wert sowohl (a) als zu speichernde
   Stufennummer als auch (b) als Baseline für `_saved_stage`
   (Anti-Doppel-Speicher-Schutz). Beide Stellen verglichen ihn gegen den rein
   standortbasierten `_stage_at_current_location()` (Route 1 = 2, Vertania =
   3) — 5 ≠ 2/3 schlug immer fehl. Beide Stellen jetzt auf den
   standortbasierten Wert umgestellt (`episode_best_stage`, das den
   Depth-Reward vor genau demselben Exploit schützt, bewusst NICHT
   angefasst).
2. **Alabastia (Pallet Town) gab jede Episode automatisch +250.** Der
   Spawnpunkt liegt in Eichs Labor, 1-2 Schritte vor Alabastia — da
   Alabastia selbst ein `CITY_MAPS`-Eintrag ist und längst bekannt, feuerte
   der Stadt-Wiederholungsbonus jede einzelne Episode automatisch, quasi nur
   fürs Rauslaufen. Alabastia explizit von diesem Bonus ausgenommen (andere
   Städte bleiben unverändert, da echte wiederholte Lauf-Leistung nötig).
3. **`shared_tiles` verlor seinen Fortschritt bei jedem Trainer-Neustart**
   (nicht nur bei einem vollen Reset) — anders als `shared_edges`/`shared_maps`/
   `shared_transitions` wurde es nie aus der persistierten Kanten-Historie
   neu geseedet. Jeder gezielte Trainer-Neustart öffnete dadurch kurzzeitig
   wieder das +2-Kachel-Fenster für längst bekanntes Terrain. Jetzt beim
   Start aus den Endpunkten der geladenen Kanten-Historie vorbefüllt
   (identische Logik im isolierten Watcher-Eval-Env, das zusätzlich alle 5
   Minuten automatisch nachlädt, ohne den Prozess neu zu starten).
4. **Party-Wipe-Teleport konnte einen Warp-Bonus auslösen.** Der
   Wipe-Cooldown schützte bisher nur Map-/Kachel-Rewards vor dem
   automatischen Pokecenter-Teleport nach einem Total-K.O. — der
   Warp-Reward-Block hatte nie eine solche Prüfung. Da die exakte
   Kampf-Position beim Fainten praktisch nie zweimal gleich ist, war das ein
   fast unerschöpflicher, immer "global neuer" Warp-Fund — jeder Wipe konnte
   zusätzlich zur -100-Strafe einen +100-Bonus einbringen. Jetzt ebenfalls
   während des Cooldowns unterdrückt (Buchführung/Claims laufen normal
   weiter).

**Frontier-Scout-System überarbeitet:** vorher wanderten alle
`FRONTIER_SCOUT_SLOTS` Scouts sofort zur tiefsten Front, sobald diese einen
Checkpoint bekam — die vorherige Front wurde komplett verwaist. Jetzt bekommt
jede validierte Stage ihre eigenen Scout-Slots dazu (`_scout_assigned_stage()`),
bestehende Paare bleiben stabil auf ihrer Stage. `FRONTIER_SCOUT_SLOTS` 2 → 5
(realistisch validiert die Flotte ohnehin nur 2-3 Stages pro Nacht). Die
Checkpoint-Haltezeit (`_hold_required`) von 25 auf 3 Lesezyklen gesenkt — die
eigentliche Qualitätssicherung passiert beim Ersetzen selbst (nur bei
höherem Episoden-Reward oder weiter nördlicher Position).

**Betrieb:** kompletter Reset aller Explorations-/Curriculum-/
Statistik-Daten (Backup in `runtime_reset_backup_*/`, nicht gelöscht) — das
trainierte Brain (`runtime/checkpoints/`), Champion-Metadaten und die
Web-Karten-Layout-Konstanten bleiben unangetastet. **Neue Standing-Regel:**
lief der Watcher gerade live (z.B. Stream), NIE über `stop_all.sh`/
`start_all.sh` mitneustarten — stattdessen den Trainer gezielt per PID
(`kill -INT`) stoppen und `start_all.sh` erneut aufrufen; das Skript erkennt
bereits laufende Prozesse (Watcher/Web/Status) automatisch und lässt sie in
Ruhe.

**V17.2 — Savestate-Start, Fleet-Rekorde, Artenvielfalt** (2026-09-05)

Seit V17 beginnt jede Episode nicht mehr am kalten Spielanfang, sondern an
einem festen, manuell erspielten Savestate (`StartGame.state`) kurz nach
Intro, Namensvergabe und Starterwahl (PWhiddy-Stil). Das Nachspielen der
~10-minütigen Introsequenz entfällt für jede der 60 parallelen Umgebungen bei
jedem Reset — der gesamte Trainingshorizont geht in echte Weltexploration statt
in wiederholtes Intro-Abspulen.

### Was sich seit V16 geändert hat

- **Savestate statt Kaltstart:** `env.load_state("StartGame", ...)` ersetzt den
  Kaltboot-Snapshot in `pokemon_env.py` und `watch.py`. Intro-/Treppen-/
  Hausausgangs- und Paket-Flags starten als bereits erledigt, damit kein
  Reset mehr fälschlich am alten Intro-Timeout (1800 Schritte) abbricht.
- **60 parallele Umgebungen** (`NUM_ENVS`), synchron mit `PPO_N_STEPS=512` →
  30.720 Samples/Update. `ACTION_HOLD_FRAMES=9` / `ACTION_RELEASE_FRAMES=5`
  (empirisch gegen 12/6 verifiziert: identische Bewegungszuverlässigkeit,
  +7–13 % Steps/Sekunde).
- **Persistenter, nicht farmbarer Kanten-Reward:** `NEW_EDGE_REWARD=1.0` für
  jede fleet-weit erstmals gelaufene Kachel-Kante (`_claim_shared`, genau
  einmal über alle Agenten und Episoden). Bereits bekannte Kanten geben dem
  einzelnen Agenten höchstens einmal `+0.20` (Imitationssignal), nie erneut.
- **Fleet-Rekord-Bonus (neu in V17.2):** `GLOBAL_STAGE_RECORD_REWARD=1000`.
  Wer als Erster im gesamten Brain einen neuen `world_stage`-Tiefenrekord
  erreicht (Route 2, Vertania-Wald, Marmoria, erster Orden), bekommt diesen
  Bonus einmalig fleet-weit — zusätzlich zum bestehenden, pro Episode und
  Agent wiederholbaren `NEW_GLOBAL_DEPTH_REWARD=1000 × Stufenanstieg`.
- **Artenvielfalt-Fangbonus (neu in V17.2):** erste je Spezies fleet-weit
  gefangene Pokémon geben `+1000` (`shared_species`, ebenfalls `_claim_shared`-
  geschützt); jeder weitere Fang derselben Art kostet `-500`. Verhindert
  Farmen häufiger Wildpokémon (Taubsi, Raupy) und belohnt seltene Funde wie
  Pikachu im Vertania-Wald.
- **Kampf-Rebalance:** `ENEMY_DAMAGE_REWARD_PER_HP=0.15`,
  `ENEMY_FAINT_REWARD=10`, `BATTLE_WIN_REWARD=15` (von 0,5/30/50 gekürzt,
  damit Dauerkämpfen im Wildgras Exploration nicht mehr strukturell schlägt).
  `FLED_BATTLE_PENALTY=-25`, kompletter Party-K.O. `-100` und sofortiges
  Episodenende (auch direkt beim Kampfende erkannt, nicht erst beim
  nächsten HP-Sample — verhindert einen Exploit, bei dem der automatische
  Pokémon-Center-Teleport nach einem Wipe als "neue Map" belohnt wurde).
- **Reproduzierbarkeits-Schwelle:** `STAGE_RELIABILITY_FRACTION=0.12` — ein
  neuer world_stage zählt für Champion-Aufstiege erst, wenn mindestens 12 %
  aller vollständigen Läufe ihn erreichen. Verhindert Champion-Beförderungen
  durch einen einzelnen glücklichen Ausreißer-Run.
- **Watcher-Anti-Loop korrigiert:** die alten, für teure Kaltstart-Resets
  gedachten Gnadenfristen (bis zu 8000 Schritte) ließen den sichtbaren
  Watcher-Lauf nach dem Savestate-Umbau unbegrenzt weiterlaufen, ohne je zu
  terminieren. Auf 900/1800 Schritte zurückgesetzt.
- **Dashboard-Weltkarte repariert:** eine zu klein bemessene feste
  `setMaxBounds`-Box zwang Leaflet, bis auf den Minimalzoom
  herauszuzoomen, egal welches Seitenverhältnis der Browser hatte — die
  Karte war dadurch faktisch unsichtbar ("links abgeschnitten"). Jetzt fester,
  nicht veränderbarer Zoom (nur noch verschiebbar, nicht mehr zoombar) und
  eine an die tatsächlich aufgedeckte Welt gekoppelte, automatisch
  mitwachsende Grenze mit 200 Kacheln unsichtbarem Rand — die Karte lässt
  sich dadurch nie mehr komplett aus dem Bild schieben.
- **Rollierendes Reward-Event-Log:** ein Klick auf einen Agenten im
  Dashboard zeigte bisher nur die Reward-Events des einen Schritts, in dem
  zufällig die Instanzdatei geschrieben wurde (alle 80 Schritte) — fast immer
  leer. Jetzt ein echtes Log der letzten ~40 tatsächlichen Ereignisse.
- `LEARNING_RATE=7.5e-05` (ein Experiment mit `0.0005` wurde nach über 2 Mio.
  Steps ohne Champion-Fortschritt als gescheitert bewertet und zurückgesetzt).

### Vorheriges V16 — Clean Full-Brain Generations (2026-09-04)

V16 beginnt nach einem vollständigen, gesicherten Reset aller alten Modelle,
Statistiken, Karten- und Curriculum-Daten. Der Stand unmittelbar vor dem Reset
wird als datiertes Backup erhalten. ROM, Quellcode und lokale Konfiguration
werden niemals gelöscht.

### Ziel und Grundprinzip

- Es existiert genau **ein gemeinsames PPO-Brain**. Keine Skill-Modelle, keine
  Progress-Agenten und keine gemischten Savestate-Starts.
- Alle **50 Trainings-Clients** beginnen jede Episode am echten Spielanfang und
  lernen die vollständige Kette Intro → Name → Haus → Labor → Schiggi → Welt.
- Die Clients laufen headless und ungebremst. Sie werden nicht auf 60 FPS
  reduziert und schreiben keine Screenshot-Karten.
- Der Mapper bleibt vollständig ausgeschaltet. Ein visueller Mapper kann später
  separat entworfen werden, beeinflusst aber weder Reward noch Champion.
- PPO sammelt pro Client 512 zusammenhängende Entscheidungen. Das ergibt
  25.600 Samples pro synchronem PPO-Update (`50 × 512`).
- `gamma=0,995` und `gae_lambda=0,98` lassen einen Erfolg weiter auf die
  vorherigen Entscheidungen zurückwirken als im alten 128er-Setup.
- Eine Episode hat zunächst höchstens 12.000 **Weg-Schritte**. Eindeutiger
  Stillstand beendet sie früher. Kampfentscheidungen werden separat gezählt:
  Sie gehören weiterhin zum PPO-Lernen, verbrauchen aber weder Intro- noch
  Routen-Horizont. Ein einzelner Kampf endet spätestens nach 2.000, alle Kämpfe
  einer Episode zusammen nach 6.000 Kampfentscheidungen als Sicherheitsgrenze.
- Stable-Retro erhält beim Erzeugen jedes Clients einen unveränderlichen
  Kaltstart-Snapshot. Jeder Episodenreset stellt genau diesen Snapshot wieder
  her. Damit können Party, Starter, Karte oder Story-RAM aus einem beendeten
  Lauf niemals in die nächste Episode durchsickern.

### V16 Reward-Vertrag

Normale Bewegung, einzelne Tiles und Tür-/Warp-Wechsel geben keinen Reward.
Alle positiven Ereignisse sind pro Episode oder projektweit einmalig geschützt.

| Ereignis | Reward |
| --- | ---: |
| normaler Schritt / bekanntes Tile | 0 |
| deutlich neuer Intro-/Dialogbildschirm | +2, insgesamt höchstens +20 |
| Intro abgeschlossen | +100 |
| Treppe erreicht | +150 |
| Haus bestätigt verlassen | +300 |
| bekannte Map erstmals in dieser Episode | +25 |
| projektweit wirklich neue Map | +500 |
| Warp / Tür / bekannte Transition | 0 |
| neue Weltstufe in dieser Episode | +1000 je Stufe |
| Schiggi gewählt | +1000 |
| Bisasam oder Glumanda gewählt | -500 und sofortiges Episodenende |
| Labor mit Schiggi verlassen | +500 |
| Gegner verliert neue HP | +0,5 je HP |
| Gegner K.O. | +30 |
| Kampf gewonnen / Erfahrung erhalten | +50 |
| Levelaufstieg | +25 je Level |
| eigene HP verloren | -0,1 je HP |
| Flucht aus einem begonnenen Kampf | -25 |
| komplette Party besiegt | -100 und Episodenende |
| Orden | +2500 |
| Heilung / Pokémon-Center | 0 |

Screenshot-Unterschiede geben außerhalb des Intros bewusst keinen Reward:
Menüs, Kampfanimationen, NPCs und Bildschirmeffekte wären leicht farmbar. Maps,
Positionen, Gegner-HP, Party und Story werden stattdessen aus bestätigten
RAM-Daten gelesen. Begegnungen sollen Gegner, eigenes Pokémon, gewählte Attacke,
Schaden, PP und Ergebnis als Telemetrie erfassen; proportionaler echter
HP-Schaden lehrt die Policy wirksame Attacken, ohne eine Kampftabelle
hartzukodieren.

Die Paket-Story wird nicht aus einem einzelnen RAM-Wert abgeleitet. „Paket
erhalten“ und „Paket abgegeben“ brauchen Schiggi, die jeweils richtige Karte,
die richtige Reihenfolge und drei aufeinanderfolgende bestätigende RAM-Lesungen.
Route 2, Wald, Marmoria und Orden dürfen die Weltstufe erst nach dieser
bestätigten Paketkette erhöhen. So kann ein kurzzeitig falsch gelesener Wert den
Webstatus und die Champion-Bewertung nicht mehr vorspulen.

### Brain-Pflege und Generationen

Alle Clients besitzen innerhalb eines Trainingsblocks dieselbe Policy. Ein
einzelner Agent besitzt daher kein eigenes Brain, das kopiert werden könnte.
Aus den synchron gesammelten Rollouts erzeugt PPO gemeinsam einen Candidate.

1. Alle 50 Clients sammeln mit derselben Ausgangspolicy Rollouts.
2. PPO aktualisiert daraus den Candidate in synchronen 25.600-Sample-Schritten.
3. Nach einem festen Trainingsblock wird der Candidate eingefroren.
4. Der Candidate wird in vollständigen Episoden vom Spielanfang ohne Lernen
   bewertet.
5. Vergleichsreihenfolge: Orden, Weltstufe, bestätigte Storykette, Schiggi plus
   Laborausgang, Maps, Reproduzierbarkeit und erst danach Reward/Tempo.
6. Nur ein nachweislich besserer Candidate wird neuer Champion und gemeinsame
   Basis der nächsten Generation. Alte und neue Messwerte dürfen niemals zu
   einer künstlichen Champion-Metrik vermischt werden.
7. Eine neue Tiefe darf das Intro nicht verdecken. Bei einem etablierten
   Champion muss der Candidate mindestens 85–90 % Intro-Retention halten.
8. Ein nicht besserer, aber stabiler Candidate darf begrenzt weiterlernen; bei
   klarer Regression wird wieder vom unveränderten Champion begonnen.

PPO kann Vergessen nicht mathematisch ausschließen. V16 verhindert das praktisch
durch wiederkehrende, gedeckelte Intro-Rewards, lange zusammenhängende Rollouts,
Full-from-Beginning-Episoden und eine unabhängige Retention-Prüfung vor jeder
Champion-Beförderung. Der Watcher zeigt ausschließlich den bestätigten
vollständigen Champion und verwendet keinerlei Skill-Snapshot.

### Vorheriger Stand

**V15.3 — All-Full, radikal vereinfachter Reward, kein Champion-Gate**
(2026-09-04). Aktueller Arbeitsstand und offene Punkte stehen in
`docs/AI_STATUS.md` (zuerst lesen). Laufende Zahlen aus
`python tools/pkmai_status.py`.

V15.3 ist eine bewusste Kurskorrektur weg vom Spezialisten-Curriculum
(intro/stairs/exit/starter-Bootcamp) hin zu einem einfacheren, PWhiddy-
artigen Modell: **fast die gesamte Flotte spielt jede Episode ab Spielanfang**
(inkl. Intro), der Reward ist auf Meilensteine + eine einzige un-farmbare
Erkundungsspur eingedampft, und der Champion wird nicht mehr durch einen
Schutzwall vor Verbesserung abgeschirmt.

### Lernlogik

- Eine gemeinsame PPO-Policy lernt alles; es gibt keine getrennten
  Spezialisten-Gehirne. `FULL_ONLY_MODE` (in `pokemon_env.py`) lässt **alle
  32 Trainings-Envs** die Rolle `full` spielen — jede Episode startet am
  Spielanfang, spielt Intro → Treppe → Haus → Labor → Starter → Welt, bis
  Blackout, falscher Starter oder ein Step-Cap.
- Die Observation enthält vier aufeinanderfolgende 64×64-Bilder sowie 31
  RAM-/Navigations-/Storywerte.
- 32 unsichtbare Trainingsumgebungen × 128 zusammenhängende Schritte ergeben
  4096 Samples pro PPO-Update. Nur der Watcher wird gerendert.
- Jede Agenten-Aktion hält die Taste **12 Emulator-Frames** und lässt sie
  dann 6 Frames los (`ACTION_HOLD_FRAMES` / `ACTION_RELEASE_FRAMES`,
  identisch in `pokemon_env.py` und `watch.py`). Kürzere Drücke drehen die
  Spielfigur in FireRed nur, statt eine Kachel zu laufen.
- **Reward, deutlich eingedampft:** Meilenstein-Einmalboni (Intro +100,
  Treppe +150, Haus +500, **Starter (Schiggy) +1000**, falscher Starter
  (Bisasam/Glumanda) −500 + Episodenende, Orden +500), ein pro Lauf
  eskalierender Tiefen-Bonus für jede neu erreichte Weltstufe
  (`NEW_GLOBAL_DEPTH_REWARD × Stufe`, gilt für **jeden** Agenten, nicht nur
  den Flotten-Ersten) und **eine einzige** Erkundungs-Spur: eine wirklich neue,
  global nie zuvor gelaufene Kachel-Kante zahlt `+0.10`. Alles Farmbare ist
  raus: die alten Korridor-/Nord-Richtungsreward, Routen-Imitation,
  Intro-Novelty-Screens und das rollenabhängige Step-Cost-Tuning sind
  deaktiviert. Eine winzige einheitliche Zeitgebühr (`-0.002`/Step) bleibt.
- Kein Champion-Schutzwall mehr: ein neuer Champion wird veröffentlicht,
  sobald ein Kandidat mit ≥4 abgeschlossenen Full-Läufen den aktuellen
  Champion-Score erreicht oder schlägt (`_score` = Orden, Weltstufe, Level,
  Maps, Tempo, Starter-Rate — reine Tiefe statt fragiler Endpositions-Raten).
- Der Watcher läuft im **Brain-Modus**: er lädt immer das aktuell trainierte
  Netz (`pokemon_model_resume.zip`) end-to-end, ohne Skill-Umschaltung, und
  zeigt Learner-Steps + Nachlade-Zähler live an — kein eingefrorener
  Skill-Snapshot mehr.

### Echte Feuerrot-Fortschrittskette

`world_stage` ist die einzige Fortschritts-Wahrheit:

0. Spielanfang/innen
1. Alabastia `(3,0)`
2. Route 1 `(3,19)`
3. Vertania City `(3,1)`
4. Eichs Paket im Vertania-Markt `(5,3)` erhalten
5. Paket bei Eich im Labor `(4,3)` abgegeben / Pokédex erhalten
6. Route 2 `(3,20)`
7. Vertania-Wald `(1,0)`
8. Marmoria City `(3,2)`
9. erster Orden

Die Paket-Stufen kommen aus den echten FireRed-SaveBlock-Variablen. Sie sind
nicht aus englischem Bildschirmtext abgeleitet und funktionieren daher auch
mit der deutschen `BPRD`-ROM. Jeder Checkpoint enthält die aktuelle Map und die
relevanten Storywerte als Sidecar; unpassende oder alte States werden ignoriert.

### Rollen bei 32 Umgebungen (V15.3)

Alle 32 Envs spielen `full` ab Spielanfang — keine Rollen-Aufteilung mehr.
`_agent_role()` ist auf `FULL_ONLY_MODE` kurzgeschlossen; die alte
sequentielle Bootcamp-Logik (intro → stairs → exit → starter →
free-world-Phasen, siehe Git-Historie) bleibt im Code als deaktivierter
Fallback erhalten, ist aber nicht aktiv. Ein Rest-Reservat für Deep-Resume-
Agenten (früher 2 Slots, resumten aus einem gespeicherten `stage_N`-Savestate)
wurde entfernt, weil der einzige verfügbare Checkpoint (Eichs Labor, ein
kleiner Innenraum) ein Datenrest der alten Reward-Ära war und praktisch
nichts beitrug.

Die Stage-, Checkpoint- und Rollenlogik wird durch Unit-Tests abgesichert.

---

Die vorherige Architektur war **V10.25 — Skill Vault + Full Chain**
(Spezialisten-Bootcamp je Story-Abschnitt, Champion-Schutzwall). Details und
warum V15.3 davon abgerückt ist: `docs/AI_STATUS.md`.

## Architecture

- Stable-Baselines3 PPO with `MultiInputPolicy`
- 32 parallel headless Stable-Retro environments
- four stacked 64×64 grayscale observations
- 31 RAM/navigation/story features
- 7 actions: A, B, START, UP, DOWN, LEFT, RIGHT
- identical action timing in training and watcher: 12 held frames + 6 release frames
- shared curriculum checkpoints and confirmed story transitions
- persistent exploration/navigation memory
- unprotected champion (best-score-wins, no regression shield) + live "brain mode" watcher
- browser dashboard with a clickable per-agent stats panel (Status and Watcher tabs)
- eigener sichtbarer Mapper (ID 121) mit einem Karten-Schritt pro Sekunde,
  separatem PPO-Modell und aus Screenshots zusammengesetzten 16×16-Tiles

### Mapper und Laufwege

`src/mapper.py` startet am tiefsten validierten Fortschritts-Checkpoint, lernt
aber in `pokemon_mapper_latest.zip` getrennt vom Haupt-Learner und Champion.
Er startet bevorzugt aus dem vorhandenen `battle_ready`-Savestate (Starter,
gesund, draußen im Gras). Der Emulator läuft dabei kontinuierlich mit 60 echten
Frames pro Sekunde. In der Welt wählt ein persistenter Frontier-Graph alle 1,5
Sekunden systematisch eine noch ungeprüfte Richtung und findet über bekannte
Kanten den kürzesten Rückweg zum nächsten offenen Feld. Das PPO-Modell wird nur
noch als Fallback in Kämpfen und Menüs verwendet. Nach jeder Aktion folgen 750
ms neutrale Beruhigungsframes; anschließend werden drei zeitlich versetzte,
ruhige Bilder desselben RAM-Standpunkts ausgewertet.
Jedes neue Feld der laufenden Episode gibt Reward;
ein projektweit erstmals kartiertes Feld, neue sichtbare Kartentiles und neue
Maps geben Extra-Reward. Dadurch bleibt der Übungsgradient erhalten, während
echte Entdeckung deutlich wertvoller ist. Story-, Kampf-, Level- und sonstige
Rewards des Haupttrainings werden vollständig verworfen. Sein 60-FPS-HD-Fenster zeigt zwischen den Entscheidungen ein ruhiges
Bild; exakt einmal pro Sekunde wird gehandelt und danach ein Mapping-Screenshot
ausgewertet. Während eines Kampfes werden weder Karten-Tiles noch Web-JPEGs
gespeichert; im Dashboard bleibt das letzte saubere Weltbild stehen. Die erzeugten Einzelkarten und der Atlas liegen unter
`runtime/mapper/` und bleiben über Neustarts erhalten.

Im Web ist `Alex (Watcher)` immer der erste Agent und sein Laufweg wird
standardmäßig angezeigt. Zusätzlich erscheinen nur Full-Journey-Agenten, deren
Episode wirklich mit `beginning` begonnen hat; Savestate-/Curriculum-Wege und
deren historische Navigationskanten sind standardmäßig aus. Linien werden ausschließlich zwischen zwei
benachbarten Koordinaten derselben Map gezogen. Savestate-Sprünge, Teleports und
Warps erscheinen deshalb nicht als Weg. Die Wege der übrigen Runner sind aus
Performance- und Lesbarkeitsgründen zunächst aus und lassen sich über
`👣 Weitere Wege` einblenden. Der Reiter `🧩 Mapper` zeigt Live-HD-Bild und den
aus echten Screenshots zusammengesetzten Atlas.

Die einzelnen Screenshot-Karten des Mappers werden außerdem alle zwei Sekunden
direkt als Bild-Overlays in der normalen `🗺️ Overworld Map` aktualisiert, sobald
ihre Kameraausrichtung durch einen echten Scrollschritt kalibriert wurde. Bis
dahin zeigt die Overworld eine exakte, animationsunabhängige RAM-Tilemap der vom
Mapper wirklich betretenen Felder. So können Wasser, Blumen, NPCs oder ein
geratener Kameraursprung den Watcher-Marker nicht mehr in Bäume verschieben.

Der Mapper besitzt eine harte Schreibgrenze: Curriculum-Savestates,
Story-Warps, Exit-/Journey-Routen und globale Weltstufe des Haupttrainings sind
für ihn schreibgeschützt. Sein Navigationsgedächtnis, seine Statistik, sein
PPO-Modell und sämtliche Bildkarten liegen ausschließlich unter
`runtime/mapper/` beziehungsweise im separaten Mapper-Checkpoint. Er liest die
validierten Trainings-Savestates nur für seinen Start und veröffentlicht außer
seiner Web-Telemetrie nichts in die Daten des Haupt-Learners.

Current PPO settings in V17.2 (`src/train.py`):

| Setting | Value |
| --- | ---: |
| Learning rate | `7.5e-05` |
| Environments | `60` |
| Steps per environment | `512` |
| Rollout size | `30720` |
| Batch size | `256` |
| Epochs | `4` |
| Gamma | `0.995` |
| GAE lambda | `0.98` |
| Entropy coefficient | `0.02` |

> Ältere Abschnitte unten (Mapper-Beschreibung, „Adaptive agent roles" mit 120
> Envs) stammen aus früheren Architekturversionen und sind teils veraltet —
> maßgeblich sind immer der aktuelle Code und die Angaben oben.

## Adaptive agent roles

The 120 environments change distribution automatically as the shared curriculum and Full Champion advance.

| Role | Starter breakthrough | Chain repair | Forest push |
| --- | ---: | ---: | ---: |
| Intro | 4 | 4 | 4 |
| Stairs | 12 | 20 | 10 |
| Exit | 20 | 20 | 12 |
| Starter | 52 | 18 | 12 |
| Battle | 0 | 4 | 8 |
| Level | 0 | 2 | 4 |
| Progress | 8 | 16 | 34 |
| Badge | 0 | 0 | 4 |
| Full journey | 24 | 36 | 32 |

All roles train the same Full-policy observation context. Their role only changes curriculum start, episode horizon and reward focus.

The current migration normally enters **Chain repair** when a shared Starter state exists but the protected Full Champion has not yet reached the Starter from the beginning. Once that happens, training switches automatically to **Forest push**.

## Checkpoints

Generated checkpoints live below `runtime/checkpoints/` and are intentionally ignored by Git.

| File | Purpose |
| --- | --- |
| `pokemon_model_resume.zip` | current learner, optimizer state and PPO step counter |
| `pokemon_model_best.zip` | protected end-to-end Full Champion |
| `pokemon_model_candidate.zip` | latest evaluated/final candidate |
| `pokemon_skill_intro_best.zip` | protected Intro policy |
| `pokemon_skill_stairs_best.zip` | protected stair policy |
| `pokemon_skill_exit_best.zip` | protected house-exit policy |
| `pokemon_skill_starter_best.zip` | protected Starter policy |
| `pokemon_skill_progress_best.zip` | protected post-Starter progress policy |

**V15.3 update:** the Skill Vault files above are historical — they were
written by the old specialist bootcamp and are no longer read by the watcher.
`WATCHER_BRAIN_MODE = True` in `watch.py` makes the watcher always load
`pokemon_model_resume.zip` (the live, continuously-training network) end to
end, with no per-stage skill switching. This is the honest "what you see is
what's training" view; see `docs/AI_STATUS.md` for why the switch was made.

## Repository layout

```text
src/        application, environment, training, watcher and web code
scripts/    start/stop utilities
tools/      development, RAM and map utilities
assets/     distributable static assets
docs/       architecture and AI handoff documentation
runtime/    generated models, telemetry, maps and curriculum (gitignored)
local/      private integration and game files (gitignored)
```

## Setup

Create a Python environment and install the project dependencies. Then create the local configuration:

```bash
cp .env.example .env
```

If ngrok is already configured globally, no token is required in `.env`.

Start all project processes:

```bash
./start_all.sh
```

Stop all project processes:

```bash
./stop_all.sh
```

During source-only or documentation changes, the trainer, watcher, webserver and ngrok do not need to be stopped. Changes to `train.py`, `pokemon_env.py` or `watch.py` require a controlled process restart before they become active.

## Runtime status

Useful local status files:

```text
runtime/trainer_status.json
runtime/champion_score.json
runtime/skill_vault_scores.json
runtime/instances_data/
```

Quick inspection:

```bash
cat runtime/trainer_status.json
echo
cat runtime/champion_score.json
echo
cat runtime/skill_vault_scores.json
```

## Development safety

- Never commit ROMs, private Stable-Retro integrations, `.env`, ngrok credentials, model checkpoints, runtime data or savestates.
- Do not reset `pokemon_model_resume.zip` merely because the Full Champion is older. The learner and champion intentionally advance on separate tracks.
- Do not reintroduce automatic hard rollback. It previously erased later-stage learning repeatedly.
- Keep watcher and training observation construction and action timing identical.
- Count only completed Full-from-beginning episodes for same-depth champion evaluation.
- Long Full probes must remain exempt from early house/stage caps; otherwise they terminate near 1,800 steps instead of their 32,768-step horizon.
- Avoid hard-coded map coordinates. Navigation should use RAM positions, discovered edges and confirmed transitions.

**Current work log: [docs/AI_STATUS.md](docs/AI_STATUS.md)** — read this first. It tracks
what changed, why, what is running, and what the next step is. Update it every session.

See [docs/AI_HANDOFF.md](docs/AI_HANDOFF.md) for the deeper technical background, invariants and tests.

Live runtime numbers: `python tools/pkmai_status.py`.

## Legal

No Pokémon ROM or proprietary game assets are included in this repository. Users must provide their own legally obtained game data and local Stable-Retro integration.

## Security

Never commit `.env`, ROMs, model checkpoints, save states, runtime data, backups or ngrok credentials. Review staged files with `git diff --cached` before every push.
