# PKMAI — Pokémon FireRed AI by Alex

Experimental reinforcement-learning project that teaches a PPO agent to
play Pokémon FireRed using Stable-Retro.

## Current architecture

- Stable-Baselines3 PPO
- `MultiInputPolicy`
- 30 parallel environments
- 64×64 grayscale visual observation
- compact RAM/navigation observation
- shared curriculum checkpoints
- persistent exploration memory
- live watcher and browser dashboard

## Agent roles

- 5 Intro specialists
- 5 Stair specialists
- 5 Exit specialists
- 10 Progress agents
- 5 Full-chain agents

## Repository layout

```text
src/        application/training code
scripts/    start/stop utilities
tools/      development and map utilities
assets/     distributable static assets
docs/       project documentation
runtime/    local generated state (gitignored)
local/      private integrations / game files (gitignored)
```

## Setup

Create a Python environment and install the project dependencies.

Copy:

```bash
cp .env.example .env
```

If ngrok is already configured globally, no token is required in `.env`.

Start:

```bash
./start_all.sh
```

Stop:

```bash
./stop_all.sh
```

## Legal

No Pokémon ROM or proprietary game assets are included in this repository.
You must provide your own legally obtained game data and local Stable-Retro
integration.

## Security

Never commit `.env`, ROMs, model checkpoints, save states, runtime data, or
ngrok credentials.
