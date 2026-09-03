# AI-server deployment

This deployment is maintained on the `sascha` branch and uses Docker Compose.

## Services

- `trainer`: one CPU-only Stable-Retro/PPO worker on `10.10.15.110`.
- `web`: dashboard on `10.10.15.111:8000`.

Both services use the external Docker network `wireguard_net`. No host ports are
published. The dashboard is therefore available to WireGuard peers at:

```text
http://10.10.15.111:8000/
```

## Local, untracked game integration

Do not commit ROMs, their hashes, checkpoints, or runtime data. The ignored local
integration must contain:

```text
local/custom_integrations/PokemonFireRed-Gba/
  rom.gba
  rom.sha
```

Generate the required SHA-1 manifest after placing the legally obtained ROM:

```bash
sha1sum local/custom_integrations/PokemonFireRed-Gba/rom.gba \
  | cut -d' ' -f1 > local/custom_integrations/PokemonFireRed-Gba/rom.sha
```

Verify registration without starting training:

```bash
docker run --rm --network none -e PYTHONPATH=/app/src \
  -v "$PWD/local:/app/local:ro" --entrypoint python pkmai-trainer:local \
  -c 'import stable_retro as retro; retro.data.Integrations.add_custom_path("/app/local/custom_integrations"); print(retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY))'
```

## Build and service control

```bash
# Build the CPU-only image.
docker compose --profile trainer build trainer

# Start the dashboard and the single trainer.
docker compose --profile web up -d web
docker compose --profile trainer up -d trainer

# Follow trainer telemetry and dashboard logs.
docker logs -f pkmai-trainer
docker logs -f pkmai-web
```

The local profile at `local/training_settings.json` configures one environment,
CPU device, 64 rollout steps and batch size 64. It is intentionally Git-ignored.

To stop training gracefully and allow the trainer's KeyboardInterrupt handler to
write its final checkpoint:

```bash
docker kill --signal=SIGINT pkmai-trainer
docker stop pkmai-web
```

Never use `docker compose down -v`: runtime state is stored in the bind-mounted
`runtime/` directory and must be preserved.
