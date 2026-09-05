#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

trainer_count="${PKMAI_LOCAL_TRAINERS:-10}"
if ! [[ "$trainer_count" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PKMAI_LOCAL_TRAINERS must be a positive integer; got %s\n' "$trainer_count" >&2
  exit 2
fi

master_url="${PKMAI_CLUSTER_MASTER_URL:-http://127.0.0.1:8765}"
PKMAI_CLUSTER_HOST=127.0.0.1 docker compose --profile cluster up -d cluster-master cluster-brain

for rank in $(seq 0 "$((trainer_count - 1))"); do
  container="pkmai-local-trainer-${rank}"
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run -d \
    --name "$container" \
    --restart unless-stopped \
    --network host \
    --cpus 1.0 \
    -e "PKMAI_CLUSTER_MASTER_URL=$master_url" \
    -e PKMAI_CLUSTER_KEY_FILE=/app/runtime/cluster/cluster_key.txt \
    -e PKMAI_WORKER_ID="local-trainer-${rank}" \
    -e PKMAI_WORKER_RANK="$rank" \
    -e PKMAI_WORKER_AGENTS=1 \
    -e PKMAI_WORKER_FLEET_SIZE="$trainer_count" \
    -v "$project_root/runtime:/app/runtime" \
    -v "$project_root/local:/app/local:ro" \
    --entrypoint bash \
    pkmai-trainer:local scripts/start_cluster_worker.sh >/dev/null
done

printf 'Started %s local trainer containers.\n' "$trainer_count"
