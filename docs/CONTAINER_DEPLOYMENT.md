# PKMAI container deployment

The current rollout architecture is Ray-free: the brain learns batches uploaded
by independent workers. The authenticated cluster master is the only required
network endpoint for workers. Bind user-facing services only to loopback or a
private VPN interface; never publish them as `0.0.0.0`.

## Containers and host ports

| Container | Role | Host port(s) |
|---|---|---|
| `pkmai-web` | Dashboard | `8001 -> 8000` |
| `pkmai-cluster-master` | Authenticated worker registry and rollout inbox | `8765` |
| `pkmai-cluster-brain` | Dynamic PPO batch learner | no published port |
| `pkmai-cluster-worker` | Independent rollout uploader | outbound access to master only |
| `pkmai-trainer` | Legacy local SB3 trainer | no published port |

## VPS / Linux brain host

Use `compose.yaml`. Set the host variables when using a remote VPN deployment.

```bash
git checkout sascha
git pull --ff-only origin sascha
export PKMAI_CLUSTER_HOST=10.10.15.1
export PKMAI_WEB_HOST=10.10.15.1
docker compose --profile trainer build trainer
docker compose --profile web up -d web
docker compose --profile cluster up -d cluster-master cluster-brain
```

Dashboard: `http://10.10.15.1:8001/`

Firewall: allow the master port only from trusted VPN subnets. Do not expose it through a public interface.

## Local Windows Docker Desktop host: ten trainers

For an all-local rollout pool, use the default loopback host (`127.0.0.1`) and
the local private ROM/integration files. `cluster-worker` intentionally has no
fixed container name or worker ID, so Compose can scale it into ten independent
emulator workers. Each scaled hostname maps to a distinct zero-based curriculum
rank and shares the writable exploration/curriculum runtime volume.

```bash
docker compose --profile trainer build trainer
docker compose --profile cluster --profile cluster-local-worker up -d \
  --scale cluster-worker=10 \
  cluster-master cluster-brain cluster-worker
```

The expected CPU reservation is 11.25 CPUs: ten one-CPU rollout workers, one
brain CPU, and the master. Docker Desktop must be configured with at least 12
CPUs and enough memory for ten 1 GiB worker limits plus the 2 GiB brain limit.

Verify that all workers have independent identities and actively upload
rollouts:

```bash
docker compose --profile cluster --profile cluster-local-worker ps
docker compose --profile cluster --profile cluster-local-worker logs --tail 50 cluster-worker
curl -fsS http://127.0.0.1:8765/health
```

`workers_online` must reach `10`; brain logs must show increasing
`policy_version` and `timesteps`. Stop the local pool with:

```bash
docker compose --profile cluster --profile cluster-local-worker down
```

## VPS / Linux remote worker

Use `compose.remote-worker.yaml` and an untracked `.worker.env`:

```text
PKMAI_WORKER_HOST_IP=<worker VPN host IP>
PKMAI_CLUSTER_MASTER_URL=http://10.10.15.1:8765
PKMAI_WORKER_ID=<unique worker name>
PKMAI_WORKER_AGENTS=1
```

Clone the exact `sascha` branch first:

```bash
git clone --branch sascha https://github.com/ZoDIkarus/pkmai.git /opt/pkmai
cd /opt/pkmai
```

Then copy the local integration and cluster key:

```bash
# Run from the new worker after its SSH access to ai-server is configured.
mkdir -p local/custom_integrations/PokemonFireRed-Gba

scp ai-server:/opt/pkmai/runtime/cluster/cluster_key.txt \
  local/cluster_key.txt
scp ai-server:/opt/pkmai/local/custom_integrations/PokemonFireRed-Gba/rom.gba \
  local/custom_integrations/PokemonFireRed-Gba/rom.gba
scp ai-server:/opt/pkmai/local/custom_integrations/PokemonFireRed-Gba/rom.sha \
  local/custom_integrations/PokemonFireRed-Gba/rom.sha
chmod 600 local/cluster_key.txt
```

Create the local `.worker.env` from the template above, then build and start one worker:

```bash
docker compose --env-file .worker.env -f compose.remote-worker.yaml build worker
docker compose --env-file .worker.env -f compose.remote-worker.yaml up -d worker
docker logs -f pkmai-cluster-worker
```

The worker has `restart: unless-stopped`; it repeatedly loads the current policy,
collects rollouts, and uploads batches to the master. Its only required remote
reachability is the master port `8765` over the VPN. Do not publish it to a
public interface.

## Windows worker

Use Docker Desktop with the WSL2 backend and run the Linux-worker commands in an Ubuntu WSL terminal. Configure the Windows/WSL VPN so the worker host can reach the brain host ports above. Keep `.worker.env`, ROM and key local; do not commit them.

## macOS worker

Use Docker Desktop for Mac. Ensure the host VPN routes reach the brain host. Create the same `.worker.env`, provide local ROM/key files, then run the Linux-worker compose commands from Terminal. If the Docker Desktop VM cannot reach the VPN interface, use a Linux/WSL worker instead.

## Operations

```bash
# Status
docker ps --filter name=pkmai-
# Brain logs
docker logs -f pkmai-cluster-brain
# Worker logs
docker logs -f pkmai-cluster-worker
# Stop only the cluster
docker compose --profile cluster down
```

The brain is the sole writer for `runtime/cluster/`; workers never copy or write brain checkpoints.
