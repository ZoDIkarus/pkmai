#!/bin/bash
# ==========================================================================
# PKMAI V20 — 100% CLEAN RESET
#
# ZIEL: komplett sauberes Bild fuer die V20-Curriculum-Architektur
# (FULL / BRIDGE / FRONTIER / RETENTION, discovered_stage vs mastered_stage,
# nicht-farmbares Ziel-Shaping, Entry/Frontier-Checkpoints).
#
# BEHALTEN: NUR das Master-Savegame `StartGame` (fixe Spielposition nach Prof.
#   Eichs Paket - liegt in local/custom_integrations/, wird von KEINEM Reset
#   angefasst) + Quellcode + ROM/Assets.
#
# GELOESCHT (frisches Netz lernt komplett neu):
#   - runtime/checkpoints/pokemon_model_*.zip   (PPO-Netz -> frisch)
#   - runtime/checkpoints/pokemon_skill_*.zip   (Skill-Vault)
#   - champion_score.json, model_version.json, skill_vault_scores.json,
#     trainer_status.json, training_history.json
#   - runtime/curriculum_shared/*  + runtime/curriculum_states/*
#     (alle Savestates/Checkpoints -> werden im Lauf NEU gesetzt)
#   - runtime/curriculum_v20/*     (discovered/mastered/known_transitions)
#   - runtime/exploration_memory/agent_*.json + reward_events.json
#   - global_progress.json -> {"max_world_stage": 0}
#   - runtime/training_stats/*, runtime/instances_data/*
#   - runtime/watcher_evaluation/*, watcher_rewards.jsonl*, watcher_mapping.json,
#     watcher_battle_stats.json
#
# VORHER Trainer + Watcher stoppen (Ctrl+C, auf Resume-Save warten) oder
#   bash scripts/stop_all.sh
#
# Nutzung:
#   bash tools/v20_reset.sh              # interaktiv (tippe RESET)
#   bash tools/v20_reset.sh --yes        # ohne Rueckfrage
#   bash tools/v20_reset.sh --yes --no-backup
# ==========================================================================
set -euo pipefail

ASSUME_YES=false
DO_BACKUP=true
for a in "$@"; do
  case "$a" in
    --yes|-y) ASSUME_YES=true ;;
    --no-backup) DO_BACKUP=false ;;
    *) echo "Unbekannte Option: $a" >&2; exit 2 ;;
  esac
done

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
RT="$PROJECT/runtime"
CK="$RT/checkpoints"
EM="$RT/exploration_memory"
STAMP="$(date +%Y%m%d_%H%M%S)"
BK="$PROJECT/brain_backups/V20_CLEAN_RESET_${STAMP}"

if pgrep -f "$PROJECT/src/train.py" >/dev/null 2>&1; then
  echo "❌ Trainer laeuft noch. Erst 'bash scripts/stop_all.sh' oder Ctrl+C."
  exit 1
fi

echo "PKMAI V20 — 100% CLEAN RESET"
echo "  Projekt : $PROJECT"
echo "  Backup  : $([ "$DO_BACKUP" = true ] && echo "$BK" || echo "(uebersprungen)")"
echo "  Behalten: NUR StartGame-Master-Savegame + Code + ROM/Assets"
if [ "$ASSUME_YES" != true ]; then
  read -r -p "Wirklich alles zuruecksetzen? Tippe RESET: " A
  [ "$A" = "RESET" ] || { echo "Abgebrochen."; exit 1; }
fi

# ---------- Voll-Backup ----------
if [ "$DO_BACKUP" = true ]; then
  mkdir -p "$BK"
  for d in checkpoints curriculum_shared curriculum_states exploration_memory \
           training_stats curriculum_v20; do
    [ -d "$RT/$d" ] && cp -Rp "$RT/$d" "$BK/$d" || true
  done
  for f in champion_score.json model_version.json skill_vault_scores.json \
           trainer_status.json training_history.json watcher_mapping.json \
           watcher_battle_stats.json; do
    [ -f "$RT/$f" ] && cp -p "$RT/$f" "$BK/" || true
  done
  echo "V20 CLEAN RESET backup $(date)" > "$BK/README.txt"
  echo "✓ Backup: $(du -sh "$BK" | cut -f1)"
fi

# ---------- PPO-Netz + Champion + Skills ----------
rm -f "$CK"/pokemon_model_*.zip
rm -f "$CK"/pokemon_skill_*.zip
rm -f "$RT/champion_score.json" "$RT/model_version.json" \
      "$RT/skill_vault_scores.json" "$RT/trainer_status.json" \
      "$RT/training_history.json"

# ---------- Curriculum / Savestates / Checkpoints ----------
rm -rf "$RT/curriculum_shared" && mkdir -p "$RT/curriculum_shared"
rm -rf "$RT/curriculum_states" && mkdir -p "$RT/curriculum_states"
rm -rf "$RT/curriculum_v20"    && mkdir -p "$RT/curriculum_v20"
rm -rf "$PROJECT/curriculum_shared" 2>/dev/null || true

# ---------- Welt / Exploration ----------
find "$EM" -name 'agent_*.json' -delete 2>/dev/null || true
rm -f "$EM/reward_events.json"
echo '{"max_world_stage": 0, "progress_schema": "geography_v1"}' \
  > "$EM/global_progress.json"

# ---------- Statistik / Dashboard / Watcher ----------
rm -f "$RT"/training_stats/* 2>/dev/null || true
rm -f "$RT"/instances_data/* 2>/dev/null || true
rm -rf "$RT/watcher_evaluation" && mkdir -p "$RT/watcher_evaluation"
rm -f "$RT"/watcher_rewards.jsonl* "$RT/watcher_mapping.json" \
      "$RT/watcher_battle_stats.json" 2>/dev/null || true

# ---------- Marker ----------
rm -f "$PROJECT"/.fresh_ai_reset_done_v3 "$PROJECT"/.v*_*_done 2>/dev/null || true
: > "$PROJECT/.v20_clean_reset_done"

echo
echo "✓ V20 100% CLEAN RESET fertig."
echo "  Frisches PPO-Netz, world_stage 0, keine Checkpoints, leere Statistik."
echo "  discovered_stage/mastered_stage = 1, Bottleneck = Pallet->Route1"
echo "  (steigt automatisch, sobald das Netz Route 1 wirklich haelt)."
echo
echo "Neu starten:"
echo "  bash scripts/start_all.sh"
echo "  # oder einzeln:"
echo "  \"$PROJECT\"/.. ; PY=/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python"
echo "  \$PY src/train.py   &   \$PY src/watch.py"
