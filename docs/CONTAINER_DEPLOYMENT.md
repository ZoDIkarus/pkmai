# Local Docker deployment

This project is deployed only on the local Windows/Docker Desktop host.

## Services

`compose.yaml` defines:

- `cluster-master` — local rollout coordination;
- `cluster-brain` — dynamic policy learning and best-policy publication;
- `web` — watcher-first dashboard on the same configurable host and container port (`8001` by default);
- `dynamic-watcher` (watcher profile) — non-training emulator stream.

Start the core services:

```bash
docker compose up -d cluster-master cluster-brain web
docker compose --profile watcher up -d dynamic-watcher
```

Start exactly ten explicit trainer containers:

```bash
./scripts/start_local_trainers.sh
```

Do not use Compose scaling for the trainer service with Docker Desktop host networking. The explicit launcher supplies distinct IDs/ranks and avoids shared-hostname registration collisions.

## Dashboard

Set `PKMAI_WEB_HOST` to the intended LAN/VPN interface and `PKMAI_WEB_PORT` to `8001`. The web application listens and publishes on that same port.

- Dashboard: `http://<host>:8001/`
- Watcher page: `http://<host>:8001/watcher`
- JPEG stream: `http://<host>:8001/watcher.jpg`
- Watcher API: `http://<host>:8001/api/watchers`

The start page renders the watcher list and automatically selects the first watcher as the live preview. The watcher writes status metadata without private model paths.

## Required verification

After any worker, brain or watcher deployment:

1. confirm all ten uniquely named trainer containers are online and registered;
2. confirm the learner serves an increasing policy version;
3. compare several sequential watcher JPEG hashes and inspect selected watcher actions;
4. verify dashboard and JPEG availability through the configured external interface.

A running container or an HTTP 200 alone is not sufficient evidence that the watcher is controlling the emulator.