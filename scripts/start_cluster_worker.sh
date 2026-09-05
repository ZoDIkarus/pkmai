#!/usr/bin/env bash
set -euo pipefail

: "${RAY_ADDRESS:?set RAY_ADDRESS, e.g. 10.10.15.113:6379}"
: "${RAY_NODE_IP:?set RAY_NODE_IP to this worker VPN IP}"
RAY_WORKER_CPUS="${PKMAI_WORKER_CPUS%%.*}"
: "${RAY_WORKER_CPUS:=1}"
trap 'ray stop --force || true' EXIT INT TERM
ray start --address="$RAY_ADDRESS" \
  --node-ip-address="$RAY_NODE_IP" \
  --num-cpus="$RAY_WORKER_CPUS" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT:-10001}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT:-10002}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_PORT:-10003}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-11000}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-11100}" \
  --disable-usage-stats
exec python -u src/cluster_worker.py
