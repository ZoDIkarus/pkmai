# PKMAI — BIG CHANGES TODO

Larger rebuilds that need their own focused session + a clean restart (not a
quick in-between change). Small reward/doc tuning stays in
[`docs/STATUS_TODO.md`](STATUS_TODO.md). Current implemented behaviour:
[`docs/CURRENT_LOGIC.md`](CURRENT_LOGIC.md). Full 2×2 design + per-module status:
[`docs/BATTLE_ARCHITECTURE.md`](BATTLE_ARCHITECTURE.md).

---

## 0. THE 2×2 LIVE CUTOVER  ← DONE (2026-09-07)

The cutover ran and the split stack is **training now** (see
[`CURRENT_LOGIC.md`](CURRENT_LOGIC.md) → "What is running now"):

- `tools/migrate_to_2x2.py --execute` → `runtime/model_manifest.json`
  (`migration_state: executed`); champion → `navigation_champion.zip`, resume →
  `navigation_learner.zip`, battle champion = `battle_champion.rule.json` marker,
  originals archived.
- `reset_navigation.py --apply` + `reset_battle.py --apply` (learners/stats only).
- `scripts/start_2x2_visible.sh` → NAVIGATION 40 (all FULL) · BATTLE 9 · mirror ·
  FULL watcher · web · status, `PKMAI_TWOBY2_LIVE=1`.
- Pre-cutover backup: `brain_backups/pre_2x2_activation_20260907_200507/`
  (`RESTORE.md` + `MANIFEST.sha256`).

**Open follow-ups on the live 2×2 system:**

- **Battle win rate is 0** (1699 episodes, 37 KOs, no win, no promoted PPO
  champion — still the rule fallback). The battle learner needs attention:
  reward shaping / scenario variety / eval gate. Currently only 1 validated
  scenario (route1).
- **Live Pokémon SWITCH is still masked** — needs the party-list cursor RAM
  address verified for BPRD (`docs/RAM_PROBE_GUIDE.md` method), then unmask in
  `src/twoby2/emulator_battle_driver.py::legal_macros`.
- **Legacy-stack cleanup** — with 2×2 live, the FIGHTER role and single-brain
  assumptions in `pokemon_env.py` / `curriculum_v20.py` / `watcher_runtime.py`
  are inert but still present (rollback path). Formal removal is its own session,
  only once the rollback path is no longer wanted.
- `src/twoby2/__init__.py` docstring still says "Nothing here is imported by
  pokemon_env.py / …" — stale; editing it re-times the preflight staleness check,
  so do it alongside a test re-record.

**Standing constraints:** the 2×2 + nav-blocker + catch-v2 + grass/shiny work was
committed and pushed to `origin/main` on 2026-09-08 at the operator's request
(one "major update" commit); commit only when asked. Never modify the protected
master / Route-1 seeds / Frontier anchor; no `SimulatedBattleDriver` in the live
path; reuse `twoby2` modules; nothing is "done" as a stub/sim/fake; start scripts
use visible `osascript` windows, never `nohup`; never restart the watcher without
the operator present.

**Rollback:** stop the stack, unset `PKMAI_TWOBY2_LIVE`, follow
`brain_backups/pre_2x2_activation_20260907_200507/RESTORE.md`, start
`scripts/start_all.sh`.

---

## 0b. NAVIGATION BLOCKER FIX + CATCH-v2 + GRASS/SHINY — BUILT 2026-09-08, awaiting go-live

Code-complete, 827 tests green, **committed but not live-activated**. Full write-up:
[`AI_STATUS.md`](AI_STATUS.md) (2026-09-08). Plan: `.claude/plans/goofy-strolling-cray.md`.

**Navigation directed graph + Route-1 blocker fix** (`src/nav_graph.py`,
`src/nav_shaping_state.py`, `src/loop_guard.py`, `src/pokemon_env.py`,
`src/train.py`, `tools/clean_nav_graph.py`). Root cause of the
`max_world_stage = 2` ceiling: `target_valid = false` on Route 1 (the target
`(10,0)` is a warp-trigger tile that `directed_bfs` can never confirm) + an
`is_blocked` per-episode TTL bug + ~1380 sampling-artefact ledges + a per-episode
`stage_advance 1→2` +250 magnet. Fix: symmetric geometric approach shaping toward
a verified exit, run-wide `stage_advance` dedup, `_score()` as the sole geographic
promotion authority, bounded eval workers, `nav_obs_v3_directed` (`NAV_DIM 36`).

- [ ] **Operator go-live** (needs a fresh nav brain — obs width changed, savestates kept):
  1. stop the nav trainer;
  2. `PYTHONPATH=src python tools/reset_navigation.py --fresh-obs-schema --apply`;
  3. `PYTHONPATH=src python tools/clean_nav_graph.py --apply` (backs up first);
  4. restart the nav trainer on the new code;
  5. sample `inst_*` after ~20–30 min — Route-1 reach must not regress > 5 pp,
     `target_valid` honest, blocked-direction count down, ledge counter ~0,
     ≥ 1 agent reaches Stage 3 in a live canary.
- [ ] Reproduce Stage 3 in multiple beginning runs before trusting the change.

**Catch-v2 Battle-PPO** (`src/battle_catch.py`, `src/catch_planner.py`,
`src/battle_executor.py`, `src/battle_train.py`). Schema v2 (obs 140, actions 12,
+CATCH). v1 untouched. Live catch **fail-closed** — `catch_ram_ready()` is False.

- [ ] `PYTHONPATH=src python tools/catch_ram_probe.py` (isolated emulator, operator
  present) → verify bag / ball-pocket / dex-owned / catch-result RAM for BPRD over
  ≥ 2 distinct encounters → flip `twoby2.battle_ram_live.catch_ram_ready`.
- [ ] Then opt-in `battle_train.py --schema v2` (fresh MaskablePPO 140/12); promote
  only through `battle_catch_gate`. No v2 champion until then.

**Grass harvester + shiny telemetry** (`src/shiny.py`,
`src/twoby2/{wild_encounter_harvester,shiny_ram,shiny_counters,scenario_pool}.py`).
Harvester grows wild-scenario variety; shiny is **telemetry only** and
`SHINY_RAM_VERIFIED = False`.

- [ ] Capture real harvest scenarios: `PYTHONPATH=src python tools/capture_battle_scenario.py …`
  (no `harvest:true` entry exists yet — do not invent one).
- [ ] `PYTHONPATH=src python tools/shiny_ram_probe.py` → verify the player TID/SID
  (SaveBlock2 `playerTrainerId`) offset for BPRD (cross-check vs own-mon `otId`,
  PID stability over ≥ 2 encounters) → set `SHINY_RAM_VERIFIED = True`.
- [ ] Phase 2 (shiny-catch priority, catch-v2 training/promotion) is a separate
  approval after both probes pass.

### Optional — variable step size (needs a nav brain reset)

Not done. Would need a navigation learner reset, so it belongs in a dedicated
session or the next reset.

**Variable step size, policy-chosen.** Action space `7 → 15`:
`A / B / START` + `↑ ↓ ← →` × lengths `{1, 2, 4}`. `step()` runs an N-tile move
as **N internal 1-tile steps** (movement + position read + tile/edge/blue-line
reward per tile), then one `return` — otherwise `seen_coords`, the
`manhattan == 1` edge block, `ShortCycleGuard` and `route_approach` all break.
`A / B / START` stay one short press (`ACTION_HOLD_FRAMES = 9`), or menus/battles
break. Mirror `ACTION_HOLD_FRAMES` / `ACTION_RELEASE_FRAMES` in `src/watch.py`.
Rationale: one policy step = one tile makes corridors/mazes near-impossible to
cross with a random walk; a fixed larger stride just oscillates. Not `{1,2,3,4}`
(= 19 actions) — the near-duplicates slow learning; `{1,2,4}` covers
fine / medium / sprint. **Needs a nav brain reset**, so it only belongs in a
cutover or a dedicated session.

---

## Superseded / done

- **2026-09-06 — V20 CURRICULUM MODES.** `FULL` / `BRIDGE` / `FRONTIER` /
  `RETENTION` / `FIGHTER` on one PPO. This is the **legacy single-PPO** stack —
  now the inactive rollback target. The live 2×2 navigation stack replaces the
  role split with 40 identical canonical-start FULL workers.
- **2026-09-07 — Navigation/Battle 2×2, built AND cut over.** Phase 1
  (type/damage engine, Gen-III DB, verified battle RAM, rule controller) +
  Phases 2–5 (macro executor, `BattleEnv` / `battle_train.py`, router +
  `NavigationBattleWrapper` in the live env, watcher/web/reset split, migration
  tool) implemented, tested (592), and **live** — see section 0.
- **FighterBrain — a second, permanently independent combat PPO.** Built as the
  **Battle Learner / Battle Champion** of the 2×2 system: own PPO / optimizer /
  rollout buffer / step counters / eval suite / checkpoints
  (`runtime/battle/checkpoints/`), its own promotion loop in `src/battle_train.py`
  (`BATTLE_PPO_N_STEPS = 256`), combat-only anti-farming reward in
  `src/battle_env.py` (no navigation/tile/map/stage/story term), untouched by a
  navigation full-reset. The earlier lightweight `FIGHTER` role on the shared net
  stays in the legacy stack only.

---

## 1. "House after Viridian Forest" — special handling  *(still open)*

Independent of the 2×2 cutover; applies to the navigation reward model.

Needs the **bank/map id** of that house first (currently unknown; a scout must
reach it, or read it from the watcher status while an agent stands inside).

Then:
- First entry per run: **+100** (like a new map).
- First fleet-wide discovery: **+250 global, once** (really only the first).
- Interior tiles of this house: **+5** per new tile/run (instead of the normal
  bank-interior value), so the agent doesn't rate the house as "worse than the
  forest" and walk back out.

Implement analogous to `POKECENTER_MAPS` / `POKEMART_MAPS` +
`INTERIOR_TILE_REWARD_BY_BANK` special case.

---

## 2. Post-wipe: new route / new city re-earnable, with anti-farm decay  *(still open, reward-only)*

User request 2026-09-07. Legacy-stack reward change; check whether it still
matters once navigation runs under the 2×2 split.

- Today (`_record_party_wipe`): `visited_maps` / `seen_coords` are deliberately
  **not** cleared on a wipe — recovery mode (graph distance to the old front +
  wild battle ×0.05), so "die on purpose" is not a farm trick.
- Wanted: after a wipe, `CITY_EPISODE_REWARD` / `EPISODE_NEW_MAP_REWARD` become
  **re-earnable** (the walk back to town pays), BUT **×0.01 from the 4th wipe of
  the episode** so die→collect-town-again→die is not a loop.
- Implementation: per-episode wipe counter (`self.episode_party_wipes`, reset in
  `reset()`, incremented in `_record_party_wipe`). On a wipe, remove cities/routes
  from `pre_wipe_best_stage` onward from `visited_maps` + replay flags (NOT
  `seen_coords`). When paying `new_map_episode` / `replay_map_once`: if
  `episode_party_wipes >= 4` → `_map_reward *= 0.01`.
- Check the interaction with `post_wipe_recovery` /
  `POST_WIPE_TARGET_PROGRESS_REWARD` — soften the distance-to-front shaping if the
  town rewards now cover it (avoid a double incentive). Do **not** touch the wipe
  cooldown (`POST_WIPE_REWARD_COOLDOWN_STEPS = 40`).

**Deliberately NOT done:** re-enabling edge reward. Discussed, but the farm risk
(A↔B / A→B→C→A loops) has bitten this project twice. If FULL/BRIDGE still walk
Route 1 too little after 1–2 above: `FULL_FRONTIER_TILE_REWARD` 0.3 → 0.5, or a
capped edge reward (~0.02 + a hard 40/map cap that stays active on unproven
stages too).
