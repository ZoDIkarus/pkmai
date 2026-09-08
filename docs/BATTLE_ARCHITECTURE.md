# PKMAI — Navigation / Battle 2×2 architecture

Authoritative current behaviour: [`CURRENT_LOGIC.md`](CURRENT_LOGIC.md)
("What is running now" + delta rounds 1–8). This file is the design + per-module
reference.

**The 2×2 split is LIVE** — cut over 2026-09-07, running via
`scripts/start_2x2_visible.sh` (`PKMAI_TWOBY2_LIVE=1`).

Status by facet (see §2b for the per-module table):

| Facet | State |
|---|---|
| Phase 1 (engine / RAM reads / rule controller / pokédb) | **implemented + hardened + tested** |
| Phases 2–5 logic (config, horizon, retention, promotion, isolation, scenarios, router, manifest, summary, executor, battle env, battle trainer, reset + migration tools) | **implemented + tested (592), LIVE** |
| Battle-menu RAM (`gBattleMons` / `gBattlerPartyIndexes` / cursors / menu state) | **VERIFIED for BPRD** (2026-09-07, 20 real dumps `runtime/ram_probe/20260907_184350`) — `src/twoby2/battle_ram_live.py` |
| Real emulator battle canary | **PASSED** (`runtime/live_battle_canary/20260907_194636/`, wild + trainer) |
| Runtime migration (`tools/migrate_to_2x2.py`) | **executed** — `runtime/model_manifest.json` `migration_state: executed` |
| Navigation stack | **live**: `src/train.py` `NUM_ENVS = 40`, **all FULL agents from the master start** (`_v20_mode()` forces `MODE_FULL`), one PPO; learner ≈ 2.72 M, champion v5 @ 2.5 M |
| Battle stack | **live**: `src/battle_train.py --workers 9`, separate PPO, ≈ 200 K steps, learner_version 43; **battle champion still the verified RULE controller** (`battle_champion.rule.json`) — no PPO champion promoted; 0 wins / 37 KOs / 1699 episodes so far |
| `twoby2.FEATURES` gates | all `False` in source; `PKMAI_TWOBY2_LIVE=1` (set by the start script) flips the six live gates ON at runtime |
| Live Pokémon SWITCH | **still masked** — the party-list cursor is not yet RAM-verified; `EmulatorBattleDriver` fail-closes it |
| Legacy removal (FIGHTER role, single-brain assumptions) | **kept** — inert under 2×2 but is the rollback path |

The legacy single-PPO trainer (`src/train.py` **without** `PKMAI_TWOBY2_LIVE`,
`NUM_ENVS = 46`, roles FULL/BRIDGE/FRONTIER/RETENTION/FIGHTER) is the **rollback
target** — `bash scripts/start_all.sh` after restoring from
`brain_backups/pre_2x2_activation_20260907_200507/`.

**The 2×2 seam in the live modules.** `pokemon_env.py` / `train.py` / `watch.py` /
`watcher_runtime.py` / `battle_train.py` / `battle_watch.py` import
`src/twoby2/live_integration.py` (and `twoby2.router` / `twoby2.nav_chunk_rollout`)
through guarded imports that are no-ops when the gates are off and active when
`PKMAI_TWOBY2_LIVE=1`.
`tests/test_twoby2_import_isolation.py::test_live_modules_only_touch_twoby2_via_the_gated_seam`
enforces that the live modules reference `twoby2` only through that guarded seam.
`battle_ram_live` / `battle_executor` only *read from* `firered_ram` /
`battle_state` (shared, unchanged) — never the reverse.

---

## 1. Target architecture

Four permanently separate model states (live layout):

```
runtime/navigation/checkpoints/navigation_learner.zip   navigation_champion.zip
runtime/battle/checkpoints/battle_learner.zip           battle_champion.zip  (currently battle_champion.rule.json)
```

Temporary `navigation_candidate.zip` / `battle_candidate.zip` are allowed; they
are not a permanent brain. Under the legacy fallback the navigation models live
at `runtime/checkpoints/pokemon_model_*.zip`.

```
                         ┌──────────────────────────────┐
                         │   runtime/model_manifest.json │  (schema, timestamps,
                         │  couples the two systems      │   versions, steps,
                         └──────────────┬───────────────┘   eval metrics, the
                                        │                   watcher's active combo)
        ┌───────────────────────────────┴────────────────────────────────┐
        │                                                                │
┌───────▼─────────┐   overworld obs                             battle obs┌▼─────────────────┐
│ NAVIGATION PPO   │  (64x64x4 image + nav vector)      (compact RAM vector│ BATTLE PPO        │
│ learner ⇄ champ  │                                     + unknown masks) │ learner ⇄ champ   │
│ 40 workers       │                                    8 headless + 1    │ macro actions    │
│ actions: 7       │                                    frame-mirror = 9  │ SWITCH_1..6 / RUN │
│ ↑↓←→ A B START   │                                                      │ + rule fallback  │
└───────┬─────────┘                                              ┌────────┴─────────┐
        │  "start a battle"                                       │ rule controller  │
        │  (overworld action triggers encounter)                  │ (battle_controller│
        ▼                                                         │  .py) — baseline  │
┌───────────────────────────────────────────────┐                │  champion + safety│
│  NAVIGATION-ONLY WRAPPER  (Phase 3)            │                │  fallback + eval  │
│  - nav picks an overworld action              │                │  reference        │
│  - if a battle starts, the Battle controller  │◄───────────────┘  opponent         │
│    plays it to the end / wipe / safety-limit  │
│  - nav then gets the NEXT overworld obs plus  │
│    a SUMMARISED battle outcome (never the     │
│    individual battle actions)                 │
└───────────────────────────────────────────────┘
```

### Hard isolation contract

Outside a battle → only Navigation PPO decides and only Navigation PPO collects
navigation rollouts.

Inside a battle → the Battle champion / learner decides. Navigation PPO sees no
battle screen, no battle menu, no raw battle button. Battle trajectories never
enter the navigation rollout. **Overwriting a navigation action with a battle
action afterwards is explicitly forbidden** — PPO would still be credited with a
foreign action. The Phase-3 wrapper implements a real boundary: the battle
sub-episode runs entirely inside the wrapper and returns one *summary* as the
consequence of the navigation decision.

The navigation battle summary may contain: win / loss / wipe, party HP lost,
enemies defeated, PP / items spent, number of turns, fled, resulting world
state. It contains **no** per-turn battle reward and **no** micro-combat reward.
Navigation learns the strategic risk of a route, not battle menus.

---

## 2. Phase 1 — what is built (this change)

| File | Purpose | Tests |
|---|---|---|
| `src/battle_types.py` | Gen-III type ids, full effectiveness chart, STAB, **physical/special by move type** (the Gen-III rule) | `tests/test_battle_types.py` |
| `src/battle_engine.py` | **Exact Gen-III integer** damage path (pret/pokefirered order: integer stat-stage ratios → `A*power` → `*(2*level//5+2)` → `//D` → `//50` → burn → `+2` → crit ×2 → STAB `*15//10` → per-type `*mult//10` with min-1 → 16 rolls `dmg*roll//100`, `roll` 85..100, min-1). Returns **all 16 roll values**. Confidence contract (`mechanics_complete`, `damage_is_estimate`, `unknown_reasons` enum). KO split (`ko_on_hit_guaranteed` / `ko_on_hit_possible` / `hit_probability` / `ko_probability` / `true_guaranteed`). Accuracy, integer effective speed, action order, burn / stat-stage / paralysis. **Fail-closed**: unknown input → `unknown=True`, never a number; `true_guaranteed` only with complete mechanics **and** certain hit | `tests/test_battle_engine.py` |
| `tools/extract_gen3_data.py` | Reproducible extraction of the species + move tables **from the local ROM** (BPRD rev 0, md5 `6648a0484a56097ca75d6af87ebce225`). Self-verifying by signature search + cross-check. No ROM bytes stored | run once, read-only |
| `src/pokedb/species_gen3.json`, `moves_gen3.json` | 411 species (types + base stats + items + abilities), 355 moves (type / power / accuracy / pp / priority / status flag). `_meta` records the ROM md5 and table offsets | — |
| `src/pokedb.py` | Fail-closed loader. Unknown id → `None` | `tests/test_pokedb.py` |
| `src/battle_ram.py` | Battle-state RAM assembly. Reuses **only verified** reads; adds held-item decode from the already-decrypted Growth substruct; everything unverified is `None` + an explicit `*_known=False` flag. `read_battle_type_flags_checked` distinguishes a real `flags==0` wild battle from an unverified read; `escape_allowed` needs a verified flag read **and** a confirmed active battle; `battle_snapshot_ready_for_execution` is the central Phase-2 readiness gate (always `False` in Phase 1). `UNVERIFIED_BATTLE_ADDRESSES` registry | `tests/test_battle_ram.py` |
| `src/battle_controller.py` | Rule-based **offline baseline / reference controller** (Battle-PPO eval opponent + safety fallback once readiness is confirmed). **Not** an immediately-usable live champion. Macro actions only. Conservative on unknowns; never claims "guaranteed" (strongest phrasing "estimated KO on hit"); every result carries `executable` (`False` until the snapshot is execution-ready); switch safety uses **current** HP and refuses any switch while an enemy move is unreadable | `tests/test_battle_controller.py` |

67 battle tests. Full suite: **592/592 green** (2026-09-07,
`runtime/twoby2_test_result.json`).

### RAM schema — verified vs fail-closed

**Verified (reused from `firered_ram.py`, confirmed earlier against
pret/pokefirered + this ROM):**

| Datum | Location | Notes |
|---|---|---|
| player party (×6) | EWRAM+0x24284 | Gen-III box decrypt + checksum; species, level, exp, cur/max HP, status u32, the 5 computed stats, move ids + PP |
| enemy party (×6) | EWRAM+0x2402C | same decoder |
| held item | Growth substruct +2 (u16) | decoded in `battle_ram`, `firered_ram` unchanged |
| `gBattleTypeFlags` | EWRAM+0x22B4C (u32) | trainer / double / link / safari bits. Read via `read_battle_type_flags_checked` → `(flags, known)`; a buffer too short to contain the field → `(None, False)`, never a silent `0` |
| trainer id / outcome | 0x386AE / 0x23E8A | fail-closed pair |
| `gMain.inBattle` | located by ROM-callback signature | version-independent (`MainBattleReader`) |

**Fail-closed / UNVERIFIED for BPRD — returned as unknown, never guessed:**

| Datum | Would need | Consequence today |
|---|---|---|
| in-battle stat stages (−6..+6) | `gBattleMons[b].statStages` | treated as neutral (0), `stat_stages_known=False` |
| authoritative active party slot | `gBattlerPartyIndexes` | heuristic = first non-fainted mon (`player_active_source="heuristic_first_alive"`, `active_slot_authoritative=False`) — telemetry / offline eval only; a live SWITCH must never fall back to it |
| in-battle types / ability / item | `gBattleMons` | types come from the DB via species id (safe); ability not modelled |
| weather | `gBattleWeather` | `weather_known=False`; not applied |
| battle-menu / cursor state | `gActionSelectionCursor` / `gMoveSelectionCursor` | `menu_cursor_known=False`; **blocks the Phase-2 macro executor** and `battle_snapshot_ready_for_execution` — see gate |
| escape-attempt bookkeeping | `gBattleStruct` | snapshot `can_escape` = `escape_allowed(flags, flags_known, in_battle)`: needs a **verified** flag read **and** `in_battle is True` **and** an escapable-wild classification with no hard blocker (trainer/link/safari/tutorial/roamer/legendary). The `can_escape(flags)` helper is only the flag-level half; the executor must still verify the real menu response |

`battle_snapshot_ready_for_execution(snapshot)` → `(ready, missing)` is the one
place that decides a snapshot is fit to drive the real game. It requires:
confirmed `in_battle`, verified `gBattleTypeFlags`, `gBattleMons`
(`stat_stages_known`), `gBattlerPartyIndexes` (`active_slot_authoritative`),
menu cursors (`menu_cursor_known`), and a readable active mon on both sides.
**In Phase 1 it always returns `False`.** Phase 2 / Router / Executor /
Battle-PPO must never be live-activated while it is `False`. Verifying
`gBattleMons` and the menu-cursor addresses against this exact ROM is the
**hard blocker for Phase 2**.

---

## 2b. Phase 2+ implementation — `src/twoby2/` + top-level battle modules

Real implementations behind **default-off feature gates** (`twoby2.FEATURES`,
all `False`; `PKMAI_TWOBY2_LIVE=1` flips the six live gates: `battle_env`,
`battle_executor_live`, `battle_router_live`, `nav_battle_wrapper`,
`adaptive_nav_horizon`, `watcher_battle_champion`). The live modules reach this
code **only** through the guarded seam in `src/twoby2/live_integration.py` —
enforced by `test_live_modules_only_touch_twoby2_via_the_gated_seam`.
`twoby2.activation` remains a chokepoint for the finer per-action live-battle
readiness. The 5 mandatory battle RAM addresses are verified;
`src/twoby2/battle_ram_live.py` is the trusted live reader.

### Final architecture (no anchor / FRONTIER / BRIDGE / RETENTION / FIGHTER)

- **40 navigation workers**, ALL `start_kind="beginning"`, ALL loading the one
  canonical start (`StartGame.state`), all training the single navigation
  learner with one reward/observation semantics. `twoby2/config.py` enforces
  `NAV_BEGINNING_WORKERS == 40`, `NAV_ANCHOR_WORKERS == 0`.
- **9 fully isolated battle workers** (`BATTLE_WORKERS = 9` =
  `BATTLE_HEADLESS_WORKERS 8 + BATTLE_VISIBLE_WORKERS 1`) on their own
  `BattleEnv` scenario emulators, with their own PPO / optimizer / rollout
  buffer / counters / logs / checkpoints. Worker 9 is a real member of the
  9-worker learner and additionally publishes its frames to a mirror window
  (`tools/battle_mirror_watch.py`) — no own optimizer/model, no extra emulator.
- **1 FULL watcher emulator** (`FULL_WATCHER_EMULATORS = 1`), inference only.
- Total: `40 + 9 + 1 = 50 = MAX_TOTAL_EMULATORS`. `validate_worker_config()`
  returns `total_emulators == 50`; `battle_worker_split()` → `{headless: 8,
  visible: 1, total: 9}`.
- Every worker practising the whole path so far comes for free from starting at
  the beginning (PWhiddy principle).

| Module | Role | Tests |
|---|---|---|
| `twoby2/__init__.py` | `FEATURES` gate dict (all off), `feature_enabled`, `all_gates_closed` | `test_twoby2_activation.py` |
| `twoby2/config.py` | **40 beginning + 0 anchor + 9 battle (8+1) + 1 FULL-watcher = 50**; `validate_worker_config` rejects any anchor; `battle_worker_split()` → `{headless:8, visible:1, total:9}`; `worker_roster` / `assert_all_navigation_starts_identical` | `test_twoby2_config.py` |
| `twoby2/horizon.py` | adaptive nav episode-length ladder `[2000 … 163840]` (NOT `PPO_N_STEPS`=512); fresh learner → 2000; confirmed champion → derived from real signals (episode horizon used, deepest stage, arrival distribution), floor = old live horizon 32768, never reset below; 80/20 current/next-rung probe split (every probe still a canonical beginning run); +1 rung only, monotonic; atomic persist + no-regress reload | `test_twoby2_horizon.py` |
| `twoby2/nav_map.py` | RAM-only `(6,64,64)` local coordinate map centred on the player; per-episode `seen_coords` resets so the early path stays rewarding; global channels for context | `test_twoby2_nav_map.py` |
| `twoby2/nav_feature_extractor.py` | versioned `CombinedExtractor` subclass adding a zero-column-migrated map branch; **`migrate_navigation_policy` verified against the real `pokemon_model_champion.zip` → bit-identical logits/values, map branch trainable** | `test_twoby2_nav_feature_extractor.py` |
| `twoby2/nav_wrapper.py` | `NavigationBattleWrapper`: semi-MDP battle sub-episode via injected `BattleDriver`; battle steps never touch the nav step counter; nav gets only a leak-checked summary + coarse reward; real post-battle / respawn observation | `test_twoby2_nav_wrapper.py` |
| `twoby2/retention.py` | early-game retention gates (`intro` … `route1_viridian`, +later); per-gate baseline/sample/abs+rel bound; `hard_failure` ⇒ nav-learner-only reset | `test_twoby2_retention.py` |
| `twoby2/promotion.py` | nav champion promotion: beginning-runs only, ≥100 evaluable, **anchor runs rejected & counted**, single lucky deep run discounted, retention veto, same-battle-version comparability required | `test_twoby2_promotion.py` |
| `twoby2/isolation.py` / `ppo_isolation.py` | disjoint nav/battle counter sets, `ModelPathGuard`, `NavigationImmutableDuringBattle`; **real two-PPO test: no shared param tensor/optimizer, a step of one is bit-frozen for the other** | `test_twoby2_ppo_isolation.py` |
| `twoby2/scenario_pool.py` | atomic-persisted pool; signature includes HP/PP/status/enemy/party (no false dedup); quarantine on ROM/schema mismatch; `AreaUnlockLedger` (≥50 runs, ≥80% reach, ≥2 seeds, champion version); weighted concrete scenario selection; 60/25/15 + ≥5 Route-1 workers | `test_twoby2_scenario_pool.py` |
| `twoby2/router.py` | 4 consumer modes (nav-training→learner / nav-eval→under-eval / watcher→champion / battle-training→learner); `PolicyVersion` compares number+sha+source+generation+obs-schema; `GenerationPin` / `EvalBatchPin` / `WatcherBattlePin` — no hot-swap; `route` rejects a raw `execution_ready` boolean | `test_twoby2_router.py` |
| `twoby2/manifest.py` | `model_manifest_v2` (ROM hash, obs schemas, learner/candidate/champion versions+steps+sha+eval, migration state, generation); `evals_comparable({},{})==False`; crash-between-tempwrite-and-replace safe | `test_twoby2_manifest.py` |
| `twoby2/battle_summary.py` | the ONLY nav↔battle channel: allow-listed keys, **recursive** leak scan (`{resulting_world_state:{per_turn_reward:…}}` rejected), strict `WORLD_STATE_SCHEMA`, fail-closed number normalisation | `test_twoby2_battle_summary.py` |
| `twoby2/battle_promotion.py` | fixed eval suite ≥200 episodes; per-area / trainer / wild / wipe / residual-HP / invalid / switch-loop regressions; hard regression ⇒ battle-learner-only reset | `test_twoby2_battle_promotion.py` |
| `twoby2/activation.py` | `live_battle_allowed(snapshot, action=…)`, per-action readiness, **dynamic `status_report`** (not hard-coded), `assert_not_live_router`, `battle_ppo_may_be_live_champion` | `test_twoby2_activation.py` |
| `twoby2/ram_battle_probe.py` | RAM verification harness: documented pokefirered-US reference + integration guesses, per-field validators + cross-consistency, **`verify_field` needs ≥2 independent in-battle dumps**; `blocker_report()` is the honest zero-dump state | `test_twoby2_ram_battle_probe.py` |
| `twoby2/reset_tools.py` + `tools/reset_{navigation,battle,full}.py` | bounded reset matrix; nav↔battle strictly separated; scenarios kept unless explicit `--wipe-scenarios`; full needs a second confirmation; savestates/curriculum/exploration always preserved; dry-run default | `test_twoby2_reset_tools.py` |
| `battle_executor.py` (top level) | real macro executor state machine over `BattleIO`; idempotent menu navigation, cursor re-read + verify after every relevant press, timeouts / max presses / max message-advances, safe abort on an unexpected menu, `action_mask(None)` → all zeros, fail-closed legality (0 PP / fainted / trainer-RUN masked), gated by `twoby2.activation` unless `dry_run` | `test_battle_executor.py` |
| `battle_env.py` (top level) | real `gymnasium.Env` (`check_env` passes); RAM-derived obs with known-masks; anti-farming reward (damage capped at real prior enemy HP, KO/win once, faint/wipe/invalid/switch-loop negative, menu 0, trainer-flee illegal); **no navigation/tile/map/stage/story reward**; `SimulatedBattleDriver` (deterministic, `battle_engine`-based) for offline bring-up, `EmulatorBattleDriver` gated | `test_battle_env.py` |
| `battle_train.py` (top level) | Battle-PPO trainer: 9 workers, own PPO/optimizer/rollout/counters/logs/checkpoints, fixed eval seeds, resume, atomic checkpoints, promotion + battle-only auto-rollback; `BATTLE_PPO_N_STEPS`=256 independent of `train.PPO_N_STEPS`=512; `make_live_battle_env` + real `EmulatorBattleDriver` when the `battle_env` gate is ON (else `SimulatedBattleDriver`, tests only) | `test_battle_train.py` |
| `battle_watch.py` (top level) | worker 9 of the battle learner + frame mirror; no own optimizer/model; `battle_driver_for()` selects the real `EmulatorBattleDriver`; fail-closed if not real-training-ready | `test_battle_watch.py` |
| `twoby2/live_integration.py` | the single gated seam: `maybe_wrap_full_agent` (wraps `PokemonFireRedEnv` with `NavigationBattleWrapper` + `NavBattleDriverAdapter` → real `EmulatorBattleDriver`), `battle_driver_for`, `assert_no_simulated_driver_in_live_path`, `integration_status` | `test_twoby2_live_integration.py` |
| `twoby2/battle_ram_live.py` | verified BPRD readers + full `struct BattlePokemon`; `crosscheck_against_party` checks `gBattleMons[player]` vs the `gBattlerPartyIndexes[player]` slot (never `party[0]`) | `test_twoby2_battle_ram_live.py` |
| `twoby2/emulator_battle_driver.py` | real `EmulatorBattleDriver` + `EmulatorBattleIO` + `build_live_snapshot`; fail-closed (unknown menu → no button, double battle blocked, trainer-RUN masked, live SWITCH masked, bounded presses/timeouts); policy pinned per battle | `test_twoby2_emulator_battle_driver.py` |
| `twoby2/nav_progress.py` | navigation progress = geographic/story only; `navigation_progress_delta` ignores level/XP/KOs; `strip_level_from_promotion_metrics` (recursive) wired into `twoby2.promotion.evaluate_promotion` | `test_twoby2_nav_progress.py` |
| `twoby2/nav_chunk_rollout.py` | `NavChunkedRollout`: PPO update every 512 real navigation decisions regardless of horizon (≤163840); chunk boundary ≠ terminated/truncated, no env reset; battle sub-episode + watcher steps add nothing to nav counters | `test_twoby2_nav_chunk_rollout.py` |
| `tools/migrate_to_2x2.py` | idempotent, dry-run default, source hashes, atomic verified copies, `--verify-obs` runs the real logit-equivalence check, rollback plan, archives originals (never deletes); **EXECUTED 2026-09-07** (`runtime/model_manifest.json` `migration_state: executed`) | — |

### RAM verification status — VERIFIED (2026-09-07)

All 5 mandatory battle addresses are **verified for BPRD** from the 20-dump
dataset `runtime/ram_probe/20260907_184350` (`battle_dump_score.py` → ALL
MANDATORY FIELDS VERIFIED), and `tools/twoby2_preflight.py` re-checks them from
the dumps on every run:

| Field | Offset (RAM view) | GBA address |
|---|---|---|
| `gBattlerPartyIndexes` | `0x23BCE` | `0x02023BCE` |
| `gBattleMons` | `0x23BE4` (4 × `0x58`) | `0x02023BE4` |
| `gActionSelectionCursor` | `0x23FF8` | `0x02023FF8` |
| `gMoveSelectionCursor` | `0x23FFC` | `0x02023FFC` |
| `battle_menu_state` | `0x22BC4` (18/20/22 = main/move/party) | `0x02022BC4` |
| `gBattleWeather` (optional) | `0x23F1C` | `0x02023F1C` |

`src/twoby2/battle_ram_live.py` reads them and passes every cross-check against
all 20 real dumps (including the post-switch dump: `gBattleMons[player]` is
checked against `gBattlerPartyIndexes[player]`, never `party[0]`). The Phase-1
fail-closed `battle_snapshot_ready_for_*` gates in `src/battle_ram.py` are
superseded for the verified fields by `battle_ram_live` + the real
`EmulatorBattleDriver`. **Live SWITCH is still masked** — the party-list cursor
is not among the verified addresses.

### Delta round (2026-09-07, part 2) — full-reset preparation

| Piece | Module | State |
|---|---|---|
| Protected-savestate registry | `twoby2/protected_assets.py` | master `StartGame.state` (sha256 `0c0e26d9…`, Oak's lab 4/3/6/4) `protected_from_{delete,replace,normalize}`; symlink/`..`/alias resistant; `assert_write_target_ok` on every real write/publish target |
| Route-1 seed (CONFIRMED) | (registry) | `route1_manual_battle_seed` = `brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.{state.gz,meta.json}` (state `4244fa04…`, meta `37087431…`), `candidate_ambiguity=False`, type `immutable_user_battle_seed` |
| Live Route-1 anchor | (registry) | `protected_live_route1_frontier_anchor` = `runtime/curriculum_shared/stage_frontier_2.*` (state `4c22f65e…`, meta `dee031c1…`), type `protected_live_anchor` — mutable training anchor, protected from reset/migration/cleanup, NEVER normalised directly |
| Read-only copy of protected sources | `protected_assets.read_only_copy` | protected file → byte-identical UNPROTECTED temp dest, sha256 checked before **and** after; source is never a write target |
| Post-parcel retention fix | `twoby2/retention.py` | `intro/stairs_down/left_house/starter` → `precanonical`, never a gate; active chain = `leave_oak_lab → pallet_route1 → route1_viridian → …`; old metrics kept, just not gating |
| Reset baseline + reward split | `twoby2/reward_split.py` | `ResetBaseline` — initial starter/dex/parcel/party/level/flags/inventory/position pay 0; `navigation_reward` / `battle_reward` channel guards reject each other's terms; `no_double_count` (no battle_win in the nav channel) |
| 8 headless + 1 mirror battle worker | `twoby2/config.py`, `src/battle_watch.py` | `BATTLE_HEADLESS_WORKERS=8 + BATTLE_VISIBLE_WORKERS=1 == BATTLE_WORKERS=9` (superseded the earlier 9+1=10); the `PKMai – BATTLE` mirror is worker 9 of the central learner (no own optimizer/model), policy pinned per battle, render side-effect-free, **fail-closed** while not real-training-ready |
| 3×3 route groups | `twoby2/scenario_pool.py::route_group_plan` | three fixed groups of three headless workers on the newest three reliably-unlocked routes *that have a validated scenario*; older routes → regression core; visible watcher always the newest active route, swaps only after its current battle; concrete scenario IDs for all 9 + the watcher |
| Full-health scenario normalizer | `twoby2/scenario_normalizer.py` | protected source is **read-only copied** (sha checked before/after) into an unprotected temp working file; `assert_write_target_ok` on the working copy + `publish` target; **fail-closed** (Gen-III party-mutation + checksum re-encryption not verified for BPRD → nothing marked `normalized=true`, every original untouched, blocker reported); level-band (`low/current/high`) versioning with size cap, dominated-only non-core eviction |
| Full-watcher parity | `twoby2/watcher_parity.py` | shared-component list + deterministic parity check (two state copies, same seed/actions → identical obs/reward-components/position/flags/hp-pp/battle-detection/term-trunc/world-state) |
| Start/stop/status prep | `twoby2/orchestration.py` | 40 nav + 9 battle + FULL watcher + web + status; separate pid files, double-start guard, status shape; **starts nothing** (no subprocess/Popen/osascript). The actual start script is `scripts/start_2x2_visible.sh` (osascript, visible windows, `PKMAI_TWOBY2_LIVE=1`) |
| Manual RAM-probe kit | `tools/battle_dump_collect.py`, `tools/battle_dump_score.py`, `twoby2/ram_battle_probe.py`, `docs/RAM_PROBE_GUIDE.md` | isolated emulator; sidecar carries FULL verified party + `encounter_id` + `battle_kind` + labelled `menu_state` + expected cursor; 5 mandatory fields (`gBattleMons` cross-checked against the `gBattlerPartyIndexes` slot **after a switch**, never `party[0]`; `gActionSelectionCursor`/`gMoveSelectionCursor` vs labelled positions 0–3; `battle_menu_state` distinct/stable per main/move/party/out-of-battle); needs ≥2 **distinct encounters** (wild + trainer) with positive & negative contexts; `gBattleWeather` is optional, not a blocker; `battle_dump_score.py` **exits non-zero** while any mandatory field is unverified |

> **Delta rounds 4–8 (2026-09-07) — superseded this section's verdict.**
> The manual RAM probe was run (20 dumps), all 5 mandatory addresses verified,
> `battle_ram_live.py` + the real `EmulatorBattleDriver` built, the interactive
> battle canary PASSED, the 2×2 seam wired into the live modules, and (round 8)
> the **cutover was executed — the 2×2 stack is training now**. The
> `scenario_normalizer` is still fail-closed (Gen-III party mutation +
> checksum re-encryption not verified for BPRD) and live SWITCH is still masked;
> only 1 validated scenario (route1) so far. Full running record:
> [`CURRENT_LOGIC.md`](CURRENT_LOGIC.md) → "What is running now" + delta rounds 4–8.

592 tests, full suite green (`runtime/twoby2_test_result.json`).

---

## 3. Phases 2–5 — implemented in `src/twoby2/`, live under `PKMAI_TWOBY2_LIVE=1` (see §2b)

### Phase 2 — Battle execution + Battle PPO  *(RAM gate satisfied 2026-09-07; runs behind `battle_env` / `battle_executor_live`)*

- **Macro Action Executor** (`src/battle_executor.py`) — translates
  `MOVE_n` / `SWITCH_n` / `RUN` into verified A/B/↑/↓ menu sequences. Verifies
  cursor state after every relevant step; fail-closed abort on an unexpected
  menu; hard safety-limit on frames-in-menus; never presses a navigation key
  outside a confirmed battle.
- **Battle-only environment** (`src/battle_env.py`) — one emulator per battle
  env, resumes a validated **battle scenario** (not a curriculum savestate),
  compact RAM observation with unknown-masks, macro-action space, invalid
  actions masked *before* execution.
- **Battle scenario pool** (`runtime/battle_scenarios/`) — deduplicated,
  size-capped, valid savestates only, wild + trainer variety. Navigation / FULL
  runs may *capture* valid battle start states into the pool but never hand
  their live episode to the battle learner — the battle trainer works on copies
  in its own emulators.
- **Battle trainer process** (`src/battle_train.py`) — separate promotion loop,
  fixed reproducible eval suite, anti-farming reward
  (damage capped at real enemy HP; KO / win once; switch-loop detection; no
  reward for menu movement or menu open/close).
- **Battle Champion gate** — winrate overall / by matchup / trainer / wild,
  wipe-rate, avg residual HP, avg turns, invalid-action count, switch-loops,
  needless flees, scenario coverage. Promotion needs a real sample and no
  critical regression; a single lucky win never promotes. Several hard
  regressions → auto-reset Battle learner to Battle champion; Navigation
  champion untouched.

### Phase 3 — Navigation-only wrapper + router  *(built as)*

- `src/twoby2/router.py` (`BattlePolicyRouter`) — Overworld → Navigation
  champion; Battle → Battle champion; Battle-PPO missing/invalid → verified rule
  controller; Navigation model missing → hard error, no random fallback.
  4 consumer modes (nav-training / nav-eval / watcher / battle-training).
- `src/twoby2/nav_wrapper.py` (`NavigationBattleWrapper`) — wraps
  `PokemonFireRedEnv` so a battle sub-episode is played entirely inside the
  wrapper (real `EmulatorBattleDriver` via `NavBattleDriverAdapter`) and the
  navigation policy only ever receives overworld observations + the battle
  summary. Wired via `src/twoby2/live_integration.py::maybe_wrap_full_agent`.
- Navigation trainer = `src/train.py` (**no separate `nav_train.py`**);
  `make_env` gained a `battle_policy` param and calls `_twoby2_wrap`. The
  `PPO_N_STEPS = 512` chunked rollout is `src/twoby2/nav_chunk_rollout.py`.
  In the 2×2 config there are no FIGHTER slots — the 9 battle workers are their
  own PPO and produce **no** navigation rollouts.

### Phase 4 — Watcher / Web / Status / start-stop / reset

- Watcher + FULL eval use `twoby2.router.BattlePolicyRouter`. Dashboard shows: active brain
  (NAVIGATION / BATTLE), both champion + learner versions/steps and their
  learner→champion deltas, battle-controller source (PPO / rule), current own +
  enemy Pokémon, types, chosen macro action, STAB, type multiplier, expected
  damage, decision reason, battle winrate / wipe-rate, navigation progress
  separately. Old single-brain dashboard fields removed / migrated — no
  contradictory duplicates.
- `scripts/start_2x2_visible.sh` starts navigation trainer + battle trainer +
  battle mirror + FULL watcher + web + status, each in its own visible Terminal
  window (osascript), `PKMAI_TWOBY2_LIVE=1`. `scripts/stop_all.sh` stops them.
  `scripts/start_all.sh` is the legacy (rollback) stack. Clean PID files; one
  trainer crashing must not corrupt the other champion.

### Phase 5 — Legacy cleanup + one-time migration

Reference-checked removals (only when provably replaced and unused):
FIGHTER as a navigation-PPO role; FIGHTER overworld rewards + navigation leash
in the navigation trainer; old single-PPO combat specialists; stale
battle/level skill-model paths; unreachable skill-vault branches; single-model
dashboard assumptions; status fields with no consumer; contradictory
comments/docs; dead compat branches.

**Never** blindly removed: story/navigation curriculum; BRIDGE/FRONTIER/
RETENTION; Part-A/Part-B checkpoint logic (frontier viability, safe fallback);
exploration memory; existing savestates; current champion files before a
verified migration.

One-time idempotent runtime migration (`tools/migrate_to_2x2.py`, **executed 2026-09-07**):
- `runtime/checkpoints/pokemon_model_champion.zip` →
  `runtime/navigation/checkpoints/navigation_champion.zip` (atomic copy)
- `pokemon_model_resume.zip` (or `…_latest.zip`) → `navigation_learner.zip`
- battle champion starts as `battle_champion.rule.json` (rule-controller marker,
  `battle_controller.py`) until a Battle-PPO passes the gates — no fake `.zip`
- `runtime/model_manifest.json` written (schema `model_manifest_v2`)
- source files hashed first, every copy atomic + verified, idempotent no-op on a
  repeat run, originals archived under `runtime/brain_backups/<ts>/` **after**
  verification (never deleted), rollback plan printed
- run: `python tools/migrate_to_2x2.py --execute --i-understand-this-rewires-runtime`

---

## 4. Reset matrix

| Tool | Touches | Preserves |
|---|---|---|
| `tools/reset_navigation.py --apply` | navigation **learner/candidate + stats** (`navigation_learner <- navigation_champion`) | navigation champion, battle models, battle scenarios, savestates, curriculum, exploration memory |
| `tools/reset_battle.py --apply` | battle **learner/candidate + stats** (`battle_learner <- battle_champion` / rule marker); scenario pool kept unless `--wipe-scenarios` | navigation models, world progress, curriculum, savestates |
| `tools/reset_full.py` | both learners — **requires a separate explicit confirmation** | both champions, savestates, curriculum, exploration memory, **all protected assets** |

All three are **dry-run by default** (`--apply` to execute) and route every
delete/write through `twoby2.reset_tools` + `protected_assets.assert_not_protected`.
The legacy `tools/v20_reset.sh` / `v11_reset.sh` belong to the legacy stack.

---

## 5. Model manifest (`runtime/model_manifest.json`)

Written by `tools/migrate_to_2x2.py` via `twoby2.manifest.build_manifest`
(schema `model_manifest_v2`; also carries `rom_sha256`, per-model `sha256`,
`migration_state`, generation). Shape:

```json
{
  "schema": "model_manifest_v2",
  "updated": "<iso8601>",
  "navigation": {"learner": {"version": N, "steps": N}, "champion": {"version": N, "steps": N, "eval": {...}}},
  "battle":     {"learner": {"version": N, "steps": N}, "champion": {"version": N, "steps": N, "source": "ppo|rule", "eval": {...}}},
  "watcher_combo": {"navigation_champion_version": N, "battle_champion_version": N}
}
```

Navigation champion eval records the **pinned battle-champion version** used, so
swapping the battle champion is never mistaken for a navigation improvement;
comparison evals must reuse the same battle version or re-baseline the combo
explicitly.

---

## 6. Known limits / open uncertainties

1. **RESOLVED (2026-09-07):** `gBattleMons`, `gBattlerPartyIndexes`,
   `gActionSelectionCursor`, `gMoveSelectionCursor`, `battle_menu_state` are
   **verified for BPRD** (20 dumps) and read by `src/twoby2/battle_ram_live.py`.
   `gBattleWeather` also confirmed (optional). Still unverified: the **party-list
   cursor** — live SWITCH stays masked in `EmulatorBattleDriver`.
2. Out of battle / offline, active-battler identification via
   `battle_ram.py` is still a heuristic (first non-fainted). In a live battle the
   authoritative slot comes from `gBattlerPartyIndexes` (verified);
   `build_live_snapshot` sets `active_slot_authoritative` from the
   `gBattleMons` / `gBattlerPartyIndexes` cross-check.
3. `battle_ram_live` supplies real stat stages (`gBattleMons[b].statStages`) in a
   live battle. The offline `battle_ram.py` path still treats them as neutral and
   flags `damage_is_estimate=True`.
4. Abilities, held-item battle effects, weather, screens, dynamic-power moves
   (Seismic Toss / Low Kick / Flail / Hidden Power / …), multi-hit/recoil/drain
   effects, secondary-effect chances and trapping are **not** modelled. The
   engine flags them via `unknown_reasons` and never returns `true_guaranteed`;
   the controller stays conservative and only ever says "estimated KO on hit".
   Levitate / Wonder Guard / Sturdy therefore cannot produce a false guaranteed
   KO.
5. The real emulator path has been exercised once end-to-end — the interactive
   battle canary (`tools/live_battle_canary.py`, PASS `20260907_194636`, isolated
   emulator, wild + trainer). It has **not** run across many battles / a full
   training session; the 5-minute post-cutover monitor is the first such test.
6. `_MOVE_NAMES` / `_KANTO_SPECIES` in `firered_ram.py` remain (telemetry
   display only); the mechanics DB is `src/pokedb/`. Legacy cleanup removes the
   display maps if a consumer audit clears them — legacy cleanup is its own
   session (the rollback path still uses them).
