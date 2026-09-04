# PKMAI — Pokémon FireRed AI by Alex

PKMAI is an experimental reinforcement-learning project that trains a PPO agent to play Pokémon FireRed with Stable-Retro. It combines visual input, RAM-derived navigation features, persistent exploration memory, curriculum states, a live watcher and a browser dashboard.

> This repository does not contain a Pokémon ROM or proprietary game assets. You must provide your own legally obtained game data and local Stable-Retro integration.

## Current release

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

Current PPO settings in V15:

| Setting | Value |
| --- | ---: |
| Learning rate | `7.5e-05` |
| Environments | `32` |
| Steps per environment | `128` |
| Rollout size | `4096` |
| Batch size | `256` |
| Epochs | `4` |
| Gamma | `0.995` |
| Entropy coefficient | `0.05` |

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
