# PKMAI — Pokémon FireRed AI by Alex

PKMAI trains reinforcement-learning agents to play **Pokémon FireRed** through
[Stable-Retro](https://github.com/Farama-Foundation/stable-retro). It uses a
stacked game image, RAM-derived navigation features, a persistent exploration
memory, curriculum savestates, a live visible watcher and a browser dashboard.

> This repository contains **no Pokémon ROM and no proprietary game assets**.
> You must supply your own legally obtained game data and a local Stable-Retro
> integration under `local/custom_integrations/`.

**Authoritative current behaviour:** [`docs/CURRENT_LOGIC.md`](docs/CURRENT_LOGIC.md).
It supersedes any conflicting numbers in the historical notes / other docs.

---

## Status (2026-09-08)

The **2×2 Navigation/Battle split is LIVE** — cut over on 2026-09-07 and
training since, started with `scripts/start_2x2_visible.sh` (`PKMAI_TWOBY2_LIVE=1`,
which flips the six `twoby2.FEATURES` live gates ON at runtime).

| Path | State |
|---|---|
| **2×2 system** (`src/twoby2/` + `battle_*` + gated hooks in the live modules) | **LIVE.** Migration executed (`runtime/model_manifest.json`); split stack running: `train.py` (40 FULL nav workers, one PPO) + `battle_train.py --workers 9` (separate PPO) + watcher + web + status. Canary PASSED. |
| **Legacy single-PPO** (`train.py` *without* `PKMAI_TWOBY2_LIVE`, roles FULL/BRIDGE/FRONTIER/RETENTION/FIGHTER, 46 envs) | intact **rollback target**, not running |

Last reported (`tools/pkmai_status.py`, 2026-09-07 ~22:52): navigation learner
≈ 2.72 M steps, champion **v5 @ 2.50 M**; battle ≈ 200 K steps, champion still the
verified **rule fallback** (no PPO battle champion promoted yet). Details:
[`docs/CURRENT_LOGIC.md`](docs/CURRENT_LOGIC.md) → "What is running now".

Rollback: stop the stack, unset `PKMAI_TWOBY2_LIVE`, restore from
`brain_backups/pre_2x2_activation_20260907_200507/` (`RESTORE.md`), start
`scripts/start_all.sh`.

### Built and tested on 2026-09-08 — NOT yet live-activated

Four subsystems are code-complete and covered by the 827-test suite but are
**gated off / not loaded by the running processes** and wait for an explicit
operator go-live (stop trainer → apply → restart → live acceptance). Nothing
below has changed a champion, a running model or a savestate.

| Subsystem | What it is | Gate / go-live |
|---|---|---|
| **Directed navigation graph** (`src/nav_graph.py`, `src/nav_shaping_state.py`, `src/loop_guard.py`) | The old movement graph was **undirected** and dropped ledge jumps, so a one-way Route-1 ledge read as a valid back-path → endless `right-up → ledge-down → right-up` loop. New `DirectedNavGraph` keys every edge on `(map, from_xy, action)` with a movement kind (walk / ledge / warp / blocked); `directed_bfs`; blocked directions need 3 temporally-independent confirmations and decay; `RegionLoopGuard` catches ledge/region loops. Obs schema `nav_obs_v3_directed`, `NAV_DIM 31→36`. | Needs a **fresh navigation brain** (obs width change), savestates kept — `tools/reset_navigation.py --fresh-obs-schema` (dry-run verified, not applied). |
| **Route-1 progress-blocker fix** (`src/pokemon_env.py`, `src/train.py`, `tools/clean_nav_graph.py`) | Champion was stuck at `max_world_stage=2` despite 100 % Route-1 reach — **not** a training-time problem. Root causes: the strategic target `(10,0)` is a warp-trigger tile so `directed_bfs` could never confirm a route (`target_valid=false` → no gradient north); an `is_blocked` TTL bug walled off the south entry every episode; 4-frame position sampling stored continuous walking as ~1400 fake ledges; the `stage_advance 1→2` **+250 was paid per episode** (a "sprint to Route 1, then stop" magnet). Fix: symmetric potential-based **geometric approach shaping** toward a verified exit while the graph rebuilds (telescoping, clamped, gated), `stage_advance` deduped **run-wide** (survives reset/wipe/savestate-reload), `_score()` is the **sole** promotion authority (geographic only), bounded eval workers so champion eval can't starve. `tools/clean_nav_graph.py` backs up + removes only provably-wrong graph entries (dry-run: 1380 ledges → 0, ~2500 false statics → ~1050, Pallet→exit route coverage 0.4 % → 82 %). | Same fresh-brain reset + `clean_nav_graph.py --apply`, then live acceptance sampling of ≥10 Route-1 agents. |
| **Catch-v2 Battle-PPO** (`src/battle_catch.py`, `src/catch_planner.py`, `src/battle_executor.py`) | Battle schema **v2** with a **CATCH** action: obs 116→140, actions 11→12, `battle_reward_v2_catch`. v1 (116/11) is untouched and stays the live combat fallback. Gen-III catch maths, `_do_catch` state machine, promotion gate `battle_catch_gate` (success-rate / ball-efficiency / no unrequested or trainer catch; reward is never a criterion). | **Live catch is fail-closed** — `twoby2.battle_ram_live.catch_ram_ready()` returns `(False, …)` because the bag / ball-pocket / dex / catch-result RAM is not verified for BPRD. Unblock: `tools/catch_ram_probe.py` (isolated emulator, operator) → verify → flip the flag → opt-in `battle_train.py --schema v2`. |
| **Grass wild-encounter harvester + fail-closed shiny telemetry** (`src/shiny.py`, `src/twoby2/{wild_encounter_harvester,shiny_ram,shiny_counters}.py`) | Tile-aware corridor walker (±3 tiles from the scenario anchor, re-reads position every frame, aborts + restores on any anomaly) to grow real wild-battle scenario variety; per-bucket scenario dedup so one Rattata seed can't dominate. Shiny detection is **telemetry only** (`shiny_value = tid ^ sid ^ pid_hi ^ pid_lo < 8`); process-safe per-class counters (`fcntl.flock` around the whole RMW); one terminal outcome per encounter. Dashboard shows Seen/Caught/Lost/Unknown per Battle/FULL/Watcher. | **`SHINY_RAM_VERIFIED = False`** → shiny status is always `("unknown", "shiny_ram_unverified")`; the dashboard shows a red "SHINY-RAM NICHT VERIFIZIERT" banner. Unblock: `tools/shiny_ram_probe.py` verifies the player TID/SID offset for BPRD. |

Design + rationale: [`docs/AI_STATUS.md`](docs/AI_STATUS.md) (2026-09-08),
[`docs/BIG_CHANGES_TODO.md`](docs/BIG_CHANGES_TODO.md), plan
`.claude/plans/goofy-strolling-cray.md`.

---

## The 2×2 architecture

Four permanently separate model states, coupled only by
`runtime/model_manifest.json`:

```
runtime/navigation/checkpoints/   navigation_learner.zip   navigation_champion.zip
runtime/battle/checkpoints/       battle_learner.zip       battle_champion.zip
runtime/model_manifest.json       couples the two systems (schema model_manifest_v2)
```

At migration the battle champion starts as a **verified rule-controller marker**
(`battle_champion.rule.json`, `battle_controller.py`) and is only replaced by a
real `battle_champion.zip` once a Battle-PPO passes `twoby2.battle_promotion`.
No fake checkpoint is ever written.

**Workers (50 emulators total):**

| Group | Count | Role |
|---|---|---|
| Navigation | 40 | **all FULL agents** — every one starts at the identical canonical post-parcel master (`StartGame.state`). **No BRIDGE/FRONTIER/RETENTION/FIGHTER split** under 2×2 (`_v20_mode()` forces `MODE_FULL`). One navigation PPO. Overworld only. |
| Battle headless | 8 | isolated battle scenario emulators; train the one battle learner |
| Battle mirror | 1 | worker 9 of the same battle learner; its frames are shown in a window (`tools/battle_mirror_watch.py`) — no own optimizer/model, no extra actions |
| FULL watcher | 1 | inference-only; same wrapper / router / champions as a FULL worker, but never learns |

**Out of battle:** the navigation brain acts; only navigation transitions enter
the navigation rollout; only navigation reward reaches the navigation PPO.

**In battle:** `NavigationBattleWrapper` hands the whole fight to the pinned
**battle champion** driven by the real `EmulatorBattleDriver`; the wrapper steps
the raw emulator directly, so `PokemonFireRedEnv.step` and its combat-reward
pipeline never run during a battle — the navigation PPO **structurally** never
sees damage / KO / win / level. Navigation only receives a coarse strategic
summary (win/loss/wipe, HP/PP/items spent, turns, resulting world state).

**Level / XP / KOs are never navigation progress** and never gate a navigation
champion promotion (`src/twoby2/nav_progress.py`). They remain in the *battle*
observation because the battle brain needs them.

**PPO update timing:** navigation `PPO_N_STEPS = 512` — a rollout is processed
every 512 real navigation decisions **regardless of the episode horizon**
(2 000 → 163 840). A chunk boundary is neither `terminated` nor `truncated`; the
env is not reset; only a real terminal zeroes the value bootstrap
(`src/twoby2/nav_chunk_rollout.py`). The battle rollout size is separate.

Full design: [`docs/BATTLE_ARCHITECTURE.md`](docs/BATTLE_ARCHITECTURE.md).

### Verified BPRD battle RAM (2026-09-07)

From 20 labelled dumps in `runtime/ram_probe/20260907_184350` (wild + trainer
fight, every cursor position, a switch, an out-of-battle negative context).
Stable-Retro RAM-view offsets; GBA address = `0x02000000 + offset`:

| Field | Offset | Notes |
|---|---|---|
| `gBattlerPartyIndexes` | `0x23BCE` | `u16[4]`; player index changes after a switch |
| `gBattleMons` | `0x23BE4` | 4 × `0x58` `struct BattlePokemon` |
| `gActionSelectionCursor` | `0x23FF8` | 0/1/2/3 = FIGHT/BAG/POKEMON/RUN |
| `gMoveSelectionCursor` | `0x23FFC` | 0..3 |
| `battle_menu_state` | `0x22BC4` | 18/20/22 = CHOOSEACTION/CHOOSEMOVE/CHOOSEPOKEMON |
| `gBattleWeather` | `0x23F1C` | optional, not a blocker |

`gMain.inBattle` (`battle_state.MainBattleReader`) is the only trusted
battle-active latch; the pulsing byte at `0x23BC8` is not.

**Live SWITCH is still masked** — the party-list cursor is not yet RAM-verified,
so live switching is deliberately fail-closed.

---

## Protected assets (never modified)

`src/twoby2/protected_assets.py` registers, with symlink/`..`/alias-resistant
paths, files that no reset / migration / normalize / cleanup path may write:

| Logical id | File(s) | sha256 (state) |
|---|---|---|
| `canonical_post_parcel_master` | `local/custom_integrations/PokemonFireRed-Gba/StartGame.state` | `0c0e26d9…683d0` |
| `route1_manual_battle_seed` | `brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.*` | `4244fa04…0b48` |
| `protected_ram_probe_route1_seed` | `brain_backups/ram_probe_route1_seed/stage_frontier_2.*` | `4c22f65e…2563` |
| `protected_live_route1_frontier_anchor` | `runtime/curriculum_shared/stage_frontier_2.*` | `4c22f65e…2563` |

The verified dump set `runtime/ram_probe/20260907_184350/`, exploration memory
and curriculum/route savestates are also preserved by every reset.

---

## Setup

```bash
# Python 3.11 env (Apple Silicon / miniforge shown; adjust PKMAI_PYTHON otherwise)
conda create -n pokemon-ai python=3.11
conda activate pokemon-ai
pip install stable-retro stable-baselines3 torch gymnasium numpy opencv-python

# provide your own ROM + integration under:
#   local/custom_integrations/PokemonFireRed-Gba/{rom.gba,data.json,metadata.json,StartGame.state}
```

Expected ROM: German FireRed **BPRD** rev 0, md5 `6648a0484a56097ca75d6af87ebce225`,
sha256 `eed4fb02…b970507`.

---

## Running

### 2×2 split stack (live)

```bash
bash scripts/start_2x2_visible.sh   # sets PKMAI_TWOBY2_LIVE=1, opens visible Terminal windows
bash scripts/stop_all.sh
#   -> NAVIGATION 40 · BATTLE 9 · BATTLE mirror · FULL watcher · WEB · STATUS
```

`src/train.py` under `PKMAI_TWOBY2_LIVE` uses `runtime/navigation/checkpoints/`
and resumes `navigation_learner.zip`; `src/battle_train.py --workers 9` uses
`runtime/battle/`. Never restart the watcher mid-stream.

### Legacy single-PPO (rollback target, not running)

```bash
bash scripts/start_all.sh     # trainer + watcher + web + status, PKMAI_TWOBY2_LIVE unset
```

`train.py` without the env var runs `NUM_ENVS = 46` with the
FULL/BRIDGE/FRONTIER/RETENTION/FIGHTER curriculum and resumes
`runtime/checkpoints/pokemon_model_resume.zip`. (`scripts/start_all_legacy.sh` is
an older osascript variant that predates the `src/` + `runtime/` layout — kept
for reference only.)

### Dashboard

`src/web_stream.py` serves the dashboard on **http://localhost:8001** (http
only — https shows "can't connect"). The start scripts launch it.

---

## How the 2×2 cutover was done (2026-09-07)

```bash
PYTHONPATH=src python tools/twoby2_preflight.py                                   # ACTIVATION-READY
PYTHONPATH=src python tools/migrate_to_2x2.py --execute --i-understand-this-rewires-runtime
PYTHONPATH=src python tools/reset_navigation.py --apply
PYTHONPATH=src python tools/reset_battle.py --apply
bash scripts/start_2x2_visible.sh
```

Migration copied the single-PPO champion → `navigation_champion.zip` and
resume → `navigation_learner.zip`, wrote `runtime/model_manifest.json`
(`migration_state: executed`), and set the battle champion to the
`battle_champion.rule.json` rule marker. Originals were archived, not deleted.

**Rollback:** stop the stack, unset `PKMAI_TWOBY2_LIVE`, follow
`brain_backups/pre_2x2_activation_20260907_200507/RESTORE.md` (restore models,
start `scripts/start_all.sh`). Protected assets are never in a backup and never
touched.

### Pre-flight — six honest readiness levels

`tools/twoby2_preflight.py` distinguishes:

1. `component_prepared` — modules import + unit-testable
2. `production_wired` — `train.py` / `watcher_runtime.py` / `watch.py` /
   `pokemon_env.py` / `battle_train.py` / `battle_watch.py` really call the seam
   (call-site scan, not file existence)
3. `unit_tests_passed` — recorded full-suite result is green **and not stale**
4. `real_canary_passed` — a current, valid `live_battle_canary` report
   (`overall == PASS`, matching ROM + seed + addresses)
5. `migration_ready` — 1–4 hold and no protected file is a write target
6. `activation_ready` — 1–5 hold

### Real battle canary

```bash
PYTHONPATH=src python tools/live_battle_canary.py
```

Isolated emulator, read-only working copy of the probe seed (hash checked before
and after). You walk into grass / a trainer; the tool detects the battle via the
verified latch, then the **real `EmulatorBattleDriver`** runs the checks (reader
plausible, action mask, real MOVE macro + observed HP/PP/turn change, switch +
`gBattlerPartyIndexes`/`gBattleMons` cross-check, wild RUN, trainer-RUN masked,
stable OUT_OF_BATTLE). Report:
`runtime/live_battle_canary/<ts>/canary_report.json`. Guide:
[`docs/RAM_PROBE_GUIDE.md`](docs/RAM_PROBE_GUIDE.md).

Last PASS: `runtime/live_battle_canary/20260907_194636/`.

---

## Tests

```bash
PYTHONPATH=src:tests /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python -m unittest discover -s tests
```

827 tests. After changing any `src/` code, re-run and re-record
`runtime/twoby2_test_result.json` (the pre-flight rejects a stale result).

---

## Layout

```
src/
  pokemon_env.py        legacy single-PPO env (gated 2x2 seam + directed-nav + geometric approach)
  train.py  watch.py    legacy trainer / visible watcher (gated 2x2 seam)
  watcher_runtime.py    isolated evaluation env (FULL-watcher parity wrap)
  web_stream.py         dashboard :8001
  nav_graph.py nav_shaping_state.py loop_guard.py   directed movement graph + persistent shaping state + loop guard
  curriculum_v20.py frontier_v20.py nav_transitions_v20.py   legacy curriculum
  battle_engine.py battle_types.py battle_ram.py battle_controller.py
  battle_executor.py battle_env.py battle_train.py battle_watch.py   battle system
  battle_catch.py catch_planner.py shiny.py         catch-v2 maths / objective planner / Gen-3 shiny value
  battle_state.py firered_ram.py pokedb.py pokedb/                    RAM + Gen-III DB
  twoby2/              2x2 state machines, router, wrapper, drivers, live_integration,
                       protected_assets, nav_progress, nav_chunk_rollout, preflight logic,
                       wild_encounter_harvester, shiny_ram, shiny_counters, scenario_pool
tools/
  twoby2_preflight.py            6-level readiness
  live_battle_canary.py          interactive real canary
  battle_dump_collect.py battle_dump_score.py   RAM-address probe kit
  catch_ram_probe.py shiny_ram_probe.py          fail-closed RAM verification (operator-run)
  clean_nav_graph.py             back up + remove provably-wrong movement-graph entries
  migrate_to_2x2.py  reset_navigation.py  reset_battle.py  reset_full.py  reset_component.py
  battle_mirror_watch.py  pkmai_status.py
scripts/
  start_2x2_visible.sh  start_all.sh (legacy)  stop_all.sh
docs/
  CURRENT_LOGIC.md              authoritative current behaviour
  BATTLE_ARCHITECTURE.md        full 2x2 design + per-module status
  BIG_CHANGES_TODO.md           forward-looking big tasks
  STATUS_TODO.md  AI_STATUS.md  AI_HANDOFF.md   history / running log
runtime/    (git-ignored)  checkpoints, curriculum, exploration memory, ram_probe, canary
brain_backups/  (git-ignored)  timestamped safety copies incl. protected seeds
```

---

## Conventions

- Git commits: end with `Code by AlexnoTabi` + `https://www.youtube.com/@AlexnoTabi`.
- Never restart the watcher without the operator present.
- `scripts/start_2x2_visible.sh` launches each service in its own visible
  Terminal window (osascript), never `nohup`.
- Secrets are local-only; `.env` is git-ignored.

Historical release notes: `git log` and the dated sections of
`docs/STATUS_TODO.md` / `docs/AI_STATUS.md`.
