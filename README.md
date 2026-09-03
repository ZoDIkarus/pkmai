# PKMAI — Pokémon FireRed AI by Alex

PKMAI is an experimental reinforcement-learning project that trains a PPO agent to play Pokémon FireRed with Stable-Retro. It combines visual input, RAM-derived navigation features, persistent exploration memory, curriculum states, a live watcher and a browser dashboard.

> This repository does not contain a Pokémon ROM or proprietary game assets. You must provide your own legally obtained game data and local Stable-Retro integration.

## Current release

**V14 — `world_stage` als einzige Fortschritts-Wahrheit** (2026-09-03).

### Warum V14

30 Mio Steps lang hat kein Agent Route 1 durchquert. Ursache war nicht „zu
langsam", sondern eine kaputte Bewertungs- und Checkpoint-Logik:

1. **Kein einheitliches Fortschrittsmaß.** `max_episode_maps`, `outdoor_count`,
   Champion-`max_maps` maßen alle unterschiedliche Dinge und zählten Alabastia-
   Innenräume mit. Eine Haus-Tour konnte den Progress-Skill verbessern, den
   Champion schützen und den Score heben — ohne Vertania zu erreichen.
2. **Der Checkpoint-Zähler stand im „neue Map"-Block.** `_deep_outdoor_steps`
   lief auf Route 1 genau einmal und erreichte nie 30 → `outdoor_2` wurde nie
   gespeichert. Jede Episode begann den Alabastia→Route-1-Weg von vorn.
3. **`outdoor_1` war Müll.** Ein Glitch-RAM-Read beim Warp (Bank kurz 3) hatte
   einen Innenraum-State als „outdoor_1" gespeichert. Agenten resumten drinnen.
4. **Falscher Ziel-Reward.** Die generische Navigation konnte ein Haus als
   nächstes Ziel wählen; die ±2.5-Ziel-Formung überstimmte dann jede
   Nord-Belohnung. „Zum Gebäude laufen" war profitabler als „nach Vertania".
5. **Kontaminiertes Wege-Gedächtnis.** Alte `journey_routes` belohnten
   Alabastia→Route 1 **und** Route 1→Alabastia.

### Was V14 ändert

- **`_world_stage()`** — eine Funktion, ein Wert (0–6):
  `0` Alabastia-Innen/Intro · `1` Alabastia außen `(3,0)` · `2` Route 1 `(3,19)`
  · `3` Vertania · `4` Route 2 · `5` Wald/weiter · `6` ≥1 Orden.
  Champion, Skill-Vault, Checkpoints und der Tiefen-Bonus bewerten **nur** noch
  diesen Wert. `_score`/`_protected_regression`/Frontier-Key: `max_stage` ist
  die primäre Achse (nach Orden), `max_maps` nur noch letzter Tie-Breaker.
- **Stage-Checkpoint als eigener Block**, jeden Schritt, aus dem „neue Map"-
  Zweig herausgezogen. Speichert `stage_N.state.gz` **plus Sidecar**
  `stage_N.meta.json` (Stage, Position, `has_starter`) sobald der Agent ≥25
  Schritte am Stück tief auf einer neuen Stage steht — auf Korridor-Maps nur
  weit im Norden (`y ≤ 18`). Resume nimmt nur Checkpoints mit **validierter**
  Meta (`_valid_stage_checkpoints()`).
- **Nord-Rampe** auf `NORTH_CORRIDOR_MAPS = {(3,0), (3,19)}`: `+1.2` pro neuer
  nördlichster Y-Reihe (nicht farmbar). Der **Wald ist bewusst ausgenommen**
  (Labyrinth). Auf diesen zwei Maps liefert `_progress_targets_for_map` **kein**
  generisches Ziel mehr — die Nord-Rampe ist der alleinige Gradient.
- **Indoor stumm** für Welt-Roller mit `stage < 3`: Bank-4-Räume geben null
  Map-/Replay-Reward. Dazu `−0.05`/Schritt fürs Drinnen-Rumhängen mit Starter,
  Indoor-Kachel-Bonus `2.0 → 0.35`.
- **Starter zählt erst bei `story_stage == OUTDOOR`** — nicht schon im Labor.
- **Journey-Routen** nur noch bei echtem Stage-Anstieg gespeichert, Kantenliste
  danach geleert (kein Rückweg mehr in der Spur).
- **128 Envs** (Test: 96 vs 120 = gleiche FPS), `PPO_N_STEPS 32`, Rollen
  skalieren automatisch mit `NUM_ENVS`.
- **Web-Dashboard:** `🧭 FLOTTEN-STATUS`-Panel (Welt-Tiefe + Balken + Map-Name,
  draußen/drinnen/Kampf-Zähler, Tiefen-Durchbrüche, K.O./Schaden, Rollen, Live
  `world_depth`-Events). `/api/state` → `fleet`-Block + `world_stage`.
- **Watcher V13.1:** stickiges `watcher_ever_outdoors` — nach dem ersten
  Draußen nie wieder Treppen-Skill; `btf/c1/c2/c3`-Debug-Leiste entfernt.

### Nötige Bereinigung vor dem ersten V14-Start

```bash
cd ~/pokemon_ai_project && ./stop_all.sh
rm -f runtime/curriculum_shared/outdoor_*.state.gz runtime/curriculum_shared/maps_*.state.gz \
      runtime/curriculum_shared/progress_*.state.gz
rm -f runtime/curriculum_states/agent_*/outdoor_*.state.gz \
      runtime/curriculum_states/agent_*/maps_*.state.gz \
      runtime/curriculum_states/agent_*/progress_*.state.gz
rm -rf runtime/curriculum_shared/journey_routes runtime/curriculum_states/agent_{96..127}
echo '{"max_world_stage":1}' > runtime/exploration_memory/global_progress.json
rm -f runtime/instances_data/*.json
./start_all.sh
```

Erhalten bleiben: `intro_complete`, `stairs_down`, `left_house`, `starter`,
`starter_outdoor` (alle Bank-3-geprüft), Learner + Champion-Modell.

### Offen / ungetestet

Reward-, Checkpoint-, Rollen- und Stage-Logik haben **keine Unit-Tests**. Die
zwei vorhandenen Tests prüfen nur das Web-Dashboard. Erfolgskriterium: im
`🧭 FLOTTEN-STATUS` muss „draußen" über „drinnen" steigen und die Welt-Tiefe
innerhalb ~1 h von `1` auf `2` (Route 1) gehen. Passiert das nicht, ist der
nächste Schritt Frame-Stacking (V15, frisches Netz).

---

Die vorherige Architektur war **V10.25 — Skill Vault + Full Chain**.

The central design separates three responsibilities:

- **Learner:** continues PPO training without automatic weight rollback.
- **Full Champion:** remains protected and is replaced only by verified full-journey progress.
- **Skill Vault:** stores the strongest complete policy snapshot found for Intro, Stairs, Exit, Starter and Progress.

Skill Vault files are complete PPO policies, not independently composable neural-network layers. The watcher routes between them according to the persistent story stage.

## Architecture

- Stable-Baselines3 PPO with `MultiInputPolicy`
- 120 parallel Stable-Retro environments
- 64×64 grayscale image observation
- 28 RAM/navigation features
- 7 actions: A, B, START, UP, DOWN, LEFT, RIGHT
- identical action timing in training and watcher: 4 held frames + 4 release frames
- shared curriculum checkpoints and confirmed story transitions
- persistent exploration/navigation memory
- protected champion and stage-specific Skill Vault
- live watcher plus browser dashboard

Default PPO settings in V10.25:

| Setting | Value |
| --- | ---: |
| Learning rate | `2.5e-05` |
| Environments | `120` |
| Steps per environment | `64` |
| Rollout size | `7680` |
| Batch size | `256` |
| Epochs | `4` |
| Gamma | `0.995` |
| Entropy coefficient | `0.008` |

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

The watcher loads the appropriate protected skill immediately when the story stage changes. It falls back to the Full Champion if a skill file does not exist.

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
