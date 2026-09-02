# PKMAI Architecture

## Training

The project trains one shared PPO `MultiInputPolicy` using 30 parallel
Stable-Retro environments.

The observation combines:
- grayscale game image
- objective
- RAM position/map information
- battle/story state
- learned navigation target direction and distance
- player level and badge count

## Curriculum

Runtime curriculum data is stored outside Git in `runtime/`.

Specialists learn short early-game skills while Progress agents resume from
the most advanced self-discovered checkpoints. Full-chain agents continue to
validate end-to-end behavior from the beginning.

## Mapping

Training agents contribute coordinate/edge/warp metadata only.
The Watcher performs the visual live-map tiling.

## Secrets

Secrets are local-only. `.env` is ignored by Git.
ngrok may alternatively use its standard user-level config outside the repo.
