# PKMAI

PKMAI is a local reinforcement-learning system for Pokémon FireRed. It uses ten independently identified Docker rollout trainers, a dynamic policy learner, and a visible watcher that runs the best published policy from the real initial game state.

> The repository contains no ROM or private Stable-Retro integration. Provide legally obtained game data through the ignored `local/` directory.

## Local topology

- `cluster-master` accepts authenticated worker registrations and rollout uploads.
- `cluster-brain` trains and atomically publishes `runtime/cluster/dynamic_policy.pt` plus the best evaluated `dynamic_policy_best.pt`.
- Ten `local-trainer-N` containers execute rollouts with explicit IDs and ranks `0`–`9`.
- `dynamic-watcher` never trains. It samples the best policy distribution, writes `runtime/watcher.jpg` and publishes public watcher status.
- `web` serves the dashboard on `PKMAI_WEB_PORT` (default `8001`). Its start page is a watcher list; the first watcher is shown live.

## Start

1. Create the private local integration in `local/` and keep it out of Git.
2. Configure the private dashboard host as needed, for example `PKMAI_WEB_HOST=192.168.2.88`.
3. Start the control plane, dashboard and watcher with Docker Compose:

   ```bash
   docker compose up -d cluster-master cluster-brain web
   docker compose --profile watcher up -d dynamic-watcher
   ```

4. Start exactly ten local trainers:

   ```bash
   ./scripts/start_local_trainers.sh
   ```

Open `http://<PKMAI_WEB_HOST>:8001/`. The first watcher preview refreshes automatically; `/watcher.jpg` remains available for direct stream embedding.

## Verification

- Confirm ten distinct `local-trainer-0` through `local-trainer-9` containers and IDs in the master status.
- Confirm learner policy versions/timesteps increase.
- Confirm several sequential `/watcher.jpg` frame hashes differ and the watcher status reports changing actions.
- Confirm the dashboard loads the watcher list and live first preview on port `8001`.

## Development safety

Never commit ROMs, private integrations, `.env` files, credentials, checkpoints or `runtime/` data. Run the test suite and inspect staged changes before pushing.