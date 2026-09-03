# PKMAI — Pokémon FireRed AI by Alex

PKMAI is an experimental reinforcement-learning project that trains a PPO agent to play Pokémon FireRed with Stable-Retro. It combines visual input, RAM-derived navigation features, persistent exploration memory, curriculum states, a live watcher and a browser dashboard.

> This repository does not contain a Pokémon ROM or proprietary game assets. You must provide your own legally obtained game data and local Stable-Retro integration.

## Current release

The current architecture is **V10.25 — Skill Vault + Full Chain**.

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
