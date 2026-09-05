# PKMAI collaboration rules

This repository is actively used from multiple devices and by multiple coding agents.

- Before starting work, fetch `origin` and inspect whether `sascha` advanced.
- Preserve local work when syncing: use a named stash or commit; never reset, clean, or overwrite uncommitted changes.
- After every verified change, commit it to `sascha` and push it immediately with native Windows Git when running on Windows.
- Before pushing, re-fetch/rebase or otherwise integrate the latest `origin/sascha`; resolve conflicts by preserving both valid changes and re-run focused tests.
- Keep this `AGENTS.md` current whenever project conventions, architecture, deployment steps, tooling, or verification requirements change; commit and push that documentation update with the related change.
- Never force-push or rewrite shared history.
- Never commit `.env`, `.worker.env`, `local/`, ROMs, cluster keys, runtime data, checkpoints, or other credentials.
- For worker or brain changes, verify the relevant container/service after the push. Do not claim emulator capacity from telemetry alone; confirm Ray resources and active rollout demand.
- On Docker Desktop local training, start ten rollout workers with `scripts/start_local_trainers.sh`; do not use Compose scaling with `network_mode: host`, because Docker Desktop collapses scaled workers onto a shared hostname.
- The dashboard listens and publishes on the same configurable `PKMAI_WEB_PORT` (default `8001`); set `PKMAI_WEB_HOST` to the intended private VPN/LAN interface before external access and verify that interface after restart.
