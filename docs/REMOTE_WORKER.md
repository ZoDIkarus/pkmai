# PKMAI remote Ray worker

Use this on a separate WireGuard-connected machine. The central brain runs on
the ai-server; workers provide emulator rollouts and never write the brain or
cluster checkpoints.

## Prerequisites

- Checkout the same `sascha` commit as the brain.
- Provide local, untracked files:
  - `local/custom_integrations/PokemonFireRed-Gba/rom.gba`
  - `local/custom_integrations/PokemonFireRed-Gba/rom.sha`
  - `local/cluster_key.txt`
- Attach the host to the appropriate local `wireguard_net` Docker network.

Create a local, untracked `.worker.env` with the WireGuard addresses supplied
by the brain operator:

```text
PKMAI_RAY_ADDRESS=10.10.15.113:6379
PKMAI_WORKER_IP=<this worker's WireGuard Docker IP>
PKMAI_CLUSTER_MASTER_URL=http://10.10.15.112:8765
PKMAI_WORKER_ID=<unique worker name>
PKMAI_WORKER_AGENTS=1
```

Build and start one constrained worker:

```bash
docker compose --env-file .worker.env -f compose.remote-worker.yaml build worker
docker compose --env-file .worker.env -f compose.remote-worker.yaml up -d worker
```

The default limit is one CPU and one emulator. Confirm registration from the
brain host via the authenticated cluster registry before increasing capacity.
