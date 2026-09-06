#!/bin/bash
# ==========================================================================
# PKMAI V11 — RESET
#
# ZIEL: sauberer Neustart auf der V11-Reward-Logik. Alles, was das
# festgefahrene Verhalten ODER verfaelschte Kennzahlen enthaelt, fliegt raus.
# Nur die reinen Skill-Policies (und savestate-basierte Checkpoints) bleiben.
#
# --------------------------------------------------------------------------
# SOFT (Standard):  bash tools/v11_reset.sh
#   BEHALTEN:
#     - pokemon_skill_{intro,stairs,exit,starter,progress}_best.zip   (Skills)
#     - skill_vault_scores.json        -> Bootcamp startet in Phase 5 (Welt)
#     - curriculum_shared/*.state.gz   (Savestates = Spiel-Fakten, kein "Hirn")
#     - journey_routes / exit_routes / confirmed_story_warps
#   NEU GESEEDET (aus pokemon_skill_starter_best.zip):
#     - pokemon_model_resume.zip / _best.zip / _latest.zip
#       (kann Fruehspiel perfekt, hat die "Alabastia-Rand-Sackgasse" NICHT)
#   GELOESCHT:
#     - champion_score.json, model_version.json, candidate.zip
#     - exploration_memory/agent_*.json  (Weltkarte -> frische volle Rewards)
#     - global_progress.json -> {"max_episode_maps": 0}
#     - training_stats/*  (Dashboard-Kurven starten frisch)
#
# --hard :  bash tools/v11_reset.sh --hard
#   Zusaetzlich: alle Skill-zips + skill_vault_scores.json weg + Curriculum-
#   Savestates weg. Frisches Netz lernt komplett von Phase 1 (Intro).
#
# VORHER Trainer + Watcher stoppen (Ctrl+C, auf Resume-Save warten).
# ==========================================================================
set -euo pipefail

MODE="soft"; [ "${1:-}" = "--hard" ] && MODE="hard"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
RT="$PROJECT/runtime"; CK="$RT/checkpoints"; EM="$RT/exploration_memory"
STAMP="$(date +%Y%m%d_%H%M%S)"
BK="$PROJECT/brain_backups/V11_${MODE}_$STAMP"

if pgrep -f "$PROJECT/src/train.py" >/dev/null 2>&1; then
  echo "❌ Trainer laeuft noch. Erst mit Ctrl+C stoppen."; exit 1
fi

echo "PKMAI V11 Reset  [$MODE]"
echo "Backup: $BK"
read -r -p "Weiter? Tippe RESET: " A
[ "$A" = "RESET" ] || { echo "Abgebrochen."; exit 1; }

# ---------- Voll-Backup ----------
mkdir -p "$BK/checkpoints"
cp -p "$CK"/*.zip "$BK/checkpoints/" 2>/dev/null || true
for f in champion_score.json model_version.json skill_vault_scores.json trainer_status.json; do
  [ -f "$RT/$f" ] && cp -p "$RT/$f" "$BK/" || true
done
[ -f "$EM/global_progress.json" ] && cp -p "$EM/global_progress.json" "$BK/" || true
[ -d "$RT/training_stats" ] && cp -pr "$RT/training_stats" "$BK/" || true
[ -d "$RT/curriculum_shared" ] && cp -pr "$RT/curriculum_shared" "$BK/" || true
echo "✓ Backup geschrieben ($(du -sh "$BK" | cut -f1))"

# ---------- gemeinsame Loeschungen ----------
rm -f "$RT"/training_stats/* 2>/dev/null || true
rm -f "$CK/pokemon_model_candidate.zip"
rm -f "$RT/champion_score.json" "$RT/model_version.json"
find "$EM" -name 'agent_*.json' -delete 2>/dev/null || true
echo '{"max_episode_maps": 0}' > "$EM/global_progress.json"

if [ "$MODE" = "soft" ]; then
  [ -f "$CK/pokemon_skill_starter_best.zip" ] || {
    echo "❌ pokemon_skill_starter_best.zip fehlt. Nutze --hard."; exit 1; }
  cp -p "$CK/pokemon_skill_starter_best.zip" "$CK/pokemon_model_resume.zip"
  cp -p "$CK/pokemon_skill_starter_best.zip" "$CK/pokemon_model_champion.zip"
  cp -p "$CK/pokemon_skill_starter_best.zip" "$CK/pokemon_model_latest.zip"
  echo "✓ Learner + Champion + latest aus Starter-Skill neu geseedet"
  echo "  BEHALTEN: 5 Skill-zips, skill_vault_scores.json (-> Phase 5),"
  echo "            $(ls "$RT/curriculum_shared"/*.state.gz 2>/dev/null | wc -l | tr -d ' ') Savestates, Routen"
else
  rm -f "$CK"/pokemon_model_*.zip "$CK"/pokemon_skill_*_best.zip
  rm -f "$RT/skill_vault_scores.json"
  rm -f "$RT/curriculum_shared"/*.state.gz 2>/dev/null || true
  echo "✓ Alles auf Null. Bootcamp startet bei Phase 1 (Intro)."
fi

echo
echo "Jetzt neu starten:"
echo "  /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python $PROJECT/src/train.py"
echo "  /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python $PROJECT/src/watch.py"
echo "Status:"
echo "  /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python $PROJECT/tools/pkmai_status.py"
