#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$PROJECT/runtime/brain_backups/brain_reset_$(date +%Y%m%d_%H%M%S)"

if pgrep -f "$PROJECT/src/train.py" >/dev/null 2>&1; then
  printf 'Trainer läuft noch. Erst mit scripts/stop_trainers_wsl.sh beenden.\n' >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
moved=0
for path in \
  "$PROJECT/runtime/checkpoints/pokemon_model_latest.zip" \
  "$PROJECT/runtime/checkpoints/pokemon_model_best.zip" \
  "$PROJECT/runtime/checkpoints/pokemon_model_candidate.zip" \
  "$PROJECT/runtime/checkpoints/pokemon_model_resume.zip" \
  "$PROJECT/runtime/checkpoints"/pokemon_skill_*_best.zip \
  "$PROJECT/runtime/model_version.json" \
  "$PROJECT/runtime/champion_score.json" \
  "$PROJECT/runtime/skill_vault_scores.json"; do
  if [[ -e "$path" ]]; then
    mv "$path" "$BACKUP_DIR/"
    moved=1
  fi
done

if [[ "$moved" -eq 0 ]]; then
  rmdir "$BACKUP_DIR"
  printf 'Kein vorhandenes Brain gefunden; der nächste Start erzeugt bereits ein frisches Modell.\n'
  exit 0
fi

printf 'Vorheriges Brain archiviert: %s\n' "$BACKUP_DIR"
printf 'Der nächste Trainerstart initialisiert ein neues PPO-Modell. Curriculum und Exploration bleiben erhalten.\n'
