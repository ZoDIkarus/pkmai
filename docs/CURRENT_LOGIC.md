# Current training logic — 2×2 Navigation/Battle, LIVE since 2026-09-07

This is the consolidated description of what actually runs, replacing conflicting
historical release notes. Runtime processes load Python changes at their next start.

## What is running now (2×2 live)

The **2×2 Navigation/Battle split is LIVE** — started with
`scripts/start_2x2_visible.sh` (`PKMAI_TWOBY2_LIVE=1`, which flips the six
`twoby2.FEATURES` live gates ON at runtime). The migration
(`tools/migrate_to_2x2.py`) ran on 2026-09-07 (`runtime/model_manifest.json`,
`migration_state: executed`) and the learners were initialised from their champions.

Running processes (`scripts/start_2x2_visible.sh`):

| Window | Process | Role |
|---|---|---|
| NAVIGATION 40 | `src/train.py` (`NUM_ENVS = 40`) | **40 FULL agents, every one from the canonical master start `StartGame.state`.** There is **no** BRIDGE / FRONTIER / RETENTION / FIGHTER split under 2×2 — `_v20_mode()` returns `MODE_FULL` for every rank while the `nav_battle_wrapper` gate is on. One navigation PPO (learner ⇄ champion). Overworld only. |
| BATTLE 9 | `src/battle_train.py --workers 9` | 8 headless + 1 frame-mirror, **one separate battle PPO** (own optimizer / rollout buffer / step counter / checkpoints). Isolated `BattleEnv` scenario emulators. |
| BATTLE mirror | `tools/battle_mirror_watch.py` | shows battle worker 9's frames (no extra emulator) |
| FULL WATCHER | `src/watch.py` | inference only — navigation champion out of battle, latest released battle champion in battle (currently the verified rule controller). Never learns, never counts. |
| WEB / STATUS | `src/web_stream.py` :8001 · `tools/pkmai_status.py` | dashboard (`twoby2_brains_v1` status) + terminal monitor (`PKMai 2x2 STATUS`) |

**Last reported state (2026-09-07 ~22:52, from `tools/pkmai_status.py`):**

- Navigation: learner ≈ **2.72 M steps**, champion **v5 @ 2.50 M**, +≈0.22 M since
  champion; 8 completed FULL runs; every FULL run reaches intro/stairs/exit/starter
  (1000‰); max stage 2 (Route 1), 7 maps, 0 badges. `training_phase: full_brain`.
- Battle: **200 448** battle steps, 348 PPO updates, learner_version 43; champion
  still **RULE FALLBACK** (`battle_champion.rule.json`, no PPO champion promoted yet);
  1699 episodes, 0 wins, 37 KOs, 0 wipes; 1 validated scenario (route1); currently
  in champion-evaluation (training rollouts briefly paused).
- Watcher: navigation champion v5, no learning.

### Hard isolation (enforced, `tests/test_twoby2_ppo_isolation.py`)

Out of battle only the navigation PPO decides and only navigation transitions enter
its rollout. In battle `NavigationBattleWrapper` hands the whole fight to the pinned
battle champion via the real `EmulatorBattleDriver`, **stepping the raw emulator
directly** — `PokemonFireRedEnv.step` and its combat-reward pipeline do not run, so
the navigation PPO structurally never sees damage / KO / win / level. Navigation
gets only a coarse strategic summary. Battle rollouts never enter the navigation
buffer. `PPO_N_STEPS = 512` navigation updates run every 512 real navigation
decisions regardless of the episode horizon (`src/twoby2/nav_chunk_rollout.py`).

**Level / XP / KOs are never navigation progress** and never gate a navigation
champion promotion (`src/twoby2/nav_progress.py`; `max_level` was removed from the
champion key in `src/train.py`). `max_level` is still shown in metrics for display.

Verified BPRD battle RAM (2026-09-07, 20 dumps): `gBattlerPartyIndexes 0x23BCE`,
`gBattleMons 0x23BE4`, `gActionSelectionCursor 0x23FF8`,
`gMoveSelectionCursor 0x23FFC`, `battle_menu_state 0x22BC4`
(`src/twoby2/battle_ram_live.py`). **Live SWITCH is masked** — party-list cursor
not yet RAM-verified.

Full design + per-module status: [BATTLE_ARCHITECTURE.md](BATTLE_ARCHITECTURE.md).
Build history: the "2×2 delta rounds" near the end of this file.

### Built 2026-09-08, NOT yet in the running processes

The directed navigation graph (`nav_obs_v3_directed`, `NAV_DIM 36`), the Route-1
progress-blocker fix (geometric approach shaping, run-wide `stage_advance` dedup,
single geographic promotion authority), Catch-v2 (`--schema v2`, fail-closed) and
the grass harvester + fail-closed shiny telemetry are committed and tested but
**do not run** — the live nav trainer still uses the `NAV_DIM 31` champion and the
undirected graph, live catch and shiny are gated off. This section will describe
them as current only after the operator go-live in
[`BIG_CHANGES_TODO.md`](BIG_CHANGES_TODO.md) §0b. Details: [`AI_STATUS.md`](AI_STATUS.md)
(2026-09-08).

---

## Legacy single-PPO stack — INACTIVE fallback

> **Not running.** Sections 1–13 below describe `src/train.py` **without**
> `PKMAI_TWOBY2_LIVE=1` (`NUM_ENVS = 46`, roles FULL/BRIDGE/FRONTIER/RETENTION/FIGHTER
> on one shared PPO). That code path is intact and is the rollback target, but the
> 2×2 stack above is what trains today. The role split, the 60/46-env numbers, the
> combat-reward tables and the curriculum below apply only to that fallback.

## Roles and actual starts (legacy fallback only)

All environments train the same PPO policy. Roles are starting conditions and reward rules, not separate networks. IDs below are zero-based, as in Status (`A59`).

| Mode | IDs (46-env fallback) | Start and purpose |
|---|---|---|
| FULL | ~18 ranks | Original `StartGame`: learn the complete route from the master save. |
| BRIDGE | ~8 ranks | Immutable `stage_<bottleneck>` entry: repeat the earliest discovered transition not reliably mastered. If no valid entry exists, start from master. |
| FRONTIER | ~12 ranks | Frontier/entry near the deepest discovered stage: extend the explored area and discover the next transition. Missing checkpoints fall back to a validated entry or master. |
| RETENTION | ~4 ranks | Rotate entry states for mastered transitions; until any are mastered, start from master. |
| FIGHTER | 4 ranks | Healthy `stage_frontier_2`, then stage-2 entry, deepest available entry, or master: provide combat experience. No separate fighter network. |
| Watcher | 120 | Master start, evaluation only; does not train PPO. |

Implementation: `src/curriculum_v20.py:allocate_modes`, `src/pokemon_env.py:_v20_choose_episode_start`, `_choose_episode_start`.

BRIDGE success means reaching at least the next stage from its bottleneck entry. Mastery requires at least 20 attempts, at least 80% success in the rolling window of up to 50 results, and at least five Full-chain confirmations. Discovery alone does not satisfy mastery; trainer startup no longer fabricates 20 successful attempts from one discovery. Retention rehearses the mastered entries; Fighter submits no transition mastery attempts.

Current saved evidence at review: discovered stage 3, mastered stage 1. The current bottleneck remains Pallet→Route 1. Route 1→Viridian has no successful attempt in its stored transition statistics. A discovery record is therefore not proof of a repeatable route or of a usable stage-3 checkpoint.

## Combat rewards: identical numbers, selective Fighter accounting

| Event | Normal roles | Fighter |
|---|---:|---:|
| Enemy HP damage | +0.08 per HP | same |
| Battle win, detected by same-team EXP increase | +10 once per battle | same |
| Level gained | +10 per level | same, during/in the immediate end of battle |
| Opponent faint bonus | 0 | 0 |
| Party HP lost | −0.1 per HP | same, during/immediately after battle |
| Partial healing | +0.1 per HP | same, only inside battle, no wipe healing |
| Flee | −25 | same |
| Flee at mean party HP ratio ≤10% | −2 | same |
| Party wipe | −100 once | same; episode ends |
| Gameplay action cost | −0.005 | same, only during battle |

`_battle_reward_scale` doubles enemy-damage/win/level/faint components for trainer battles in both cases. It does not double every penalty. On wild-training maps it reduces these components to ×0.1 after three recorded opponent faints for normal roles; Fighter alone skips that reduction. The counter is opponent faints, not necessarily three fully completed multi-opponent battles.

The existing post-wipe ×0.05 applies to damage (win and level are exempt) in both roles' scaling function. Fighter now ends its episode on a wipe, so it normally restarts healthy instead of spending its next battles in recovery.

Fighter receives **zero exploration, tile, route, stage, story, building, capture or recovery bonuses**. Its returned PPO reward, episode total, and reported reward events use the same exact numeric combat accumulator. Display strings are not parsed to reconstruct reward. Other roles retain their original total. No extra win/level multiplier or bigger base reward was introduced.

Shared policy means combat practice can benefit all roles, but removing navigation rewards from Fighter is not a guarantee of faster learning. The other agents continue supplying navigation/story experience.

## One shared policy context

`_policy_objective()` normalises the objective one-hot so that **FULL, BRIDGE, FRONTIER, RETENTION and FIGHTER all present the exact same "full" objective context to the policy input**. Previously `training_objective="scout"` (BRIDGE/FRONTIER/RETENTION/FIGHTER) produced an all-zero objective one-hot rather than the FULL context, which blocked direct transfer of what the navigators learn into the champion-measured FULL policy. Reward selection and episode-start logic still branch on the real `training_objective` / `training_mode`, so this changes only the policy *input*, not behaviour. `NAV_DIM` (31) and the observation-space shape are unchanged — normalisation only moves which one-hot slot is 1.0.

## Navigation and exploration rewards

These apply to non-Fighter roles; ordinary combat rewards above remain available to them too.

| Event | Actual rule |
|---|---|
| New episode stage | +250 per gained stage, above the reset baseline |
| Approach a known target | +0.05 per unit of new best distance; returning to an already achieved distance does not repay |
| Large backtrack | −0.005 beyond a 12-tile margin; the environment explicitly passes this margin |
| FULL/BRIDGE/RETENTION new tile | +0.02 on proven ground, +0.3 when forward navigation is unknown |
| FRONTIER new tile | +0.3 while the forward transition is unknown; otherwise +0.02. Fleet-first tile adds +1. |
| Tile cap | No cap while the forward transition is unknown, for all navigation roles. On proven ground, after 20 outdoor tiles per episode the base component is ×0.1; separate interior cap. |
| FRONTIER topological progress | +0.15 × improvement over anchored best frontier score, minimum improvement 0.5 |
| Newly confirmed forward transition | +40 if the crossing makes it known and objective is `scout` |
| Already known forward crossing | +25 once per episode for FULL/BRIDGE |
| Door/map back-and-forth | −0.10 |
| Short local cycle | −0.05, escalating to −0.25; persistent loops can truncate |

Town/route, story, badges and catching retain their existing conditional rewards (e.g. new route +50, city +300, badge +2000). They are not additive unconditional rewards on every visit. Scouts receive no tile reward below their starting stage, and map arrival guards suppress old-stage reward farming. `src/pokemon_env.py:step` contains the conditions; constants alone are not a complete reward specification.

FRONTIER score comes from walked graph depth plus unknown-neighbor openness minus revisits, not a hardcoded north direction. Unknown/unconnected positions do not invent a target or score. This also means a deep dead-end can look promising; the metric cannot guarantee the exit is found.

## Healthy and timely savestates

`src/checkpoint_health.py` now defines **two** bars:

* **`party_ready` (strict, unchanged):** all checksums valid, every Pokémon ≥80% HP, no status, ≥1 move with PP each. **Entry checkpoints (`stage_<n>`) and FIGHTER anchors require this.** Entries stay immutable and only FRONTIER creates them, so BRIDGE/RETENTION always resume a fully healthy start.
* **`frontier_viable` (2026-09-07, `stage_frontier_<n>` only):** a hurt but still playable party — at least two Pokémon alive, total current HP ≥50% of total max HP over the valid party, and at least one alive Pokémon with a usable move. A strict `party_ready` party trivially qualifies. This lets an agent that fought its way deeper into a no-heal route (Route 1) **save that spatial progress** instead of losing it at the next wipe. `MIN_CHECKPOINT_HP_RATIO` is untouched.

**Only FRONTIER ever calls `_save_stage_checkpoint()`.** BRIDGE, FULL, RETENTION and FIGHTER never write `stage_<n>`, `stage_frontier_<n>` or the safe fallback. They only ever *resume* checkpoints.

**Frontier replacement rule (`may_replace_frontier`), exact — invariant: no path may ever lower `frontier_score`:**

1. A candidate that is neither `party_ready` nor `frontier_viable` may anchor nothing.
2. `party_ready` candidate vs `party_ready` anchor: strictly deeper score, **or** an HP refresh (≥+5 pp minimum-HP) at the same score.
3. `party_ready` candidate vs a hurt / legacy / poisoned anchor: reclaim it **only if `score >= old_score`** — health may restore the party but must not walk the anchor backwards. No new distance record is required (equal score is enough); a spatially worse healthy state is rejected.
4. Merely-viable (hurt) candidate vs **any** anchor (healthy or hurt): only on a real forward jump of **≥ +3.0** `frontier_score` (graph depth from the stage origin — never Y direction or hardcoded "north").

`frontier_score` therefore never decreases through any replacement path.

**Fixed per-stage healthy safe fallback (`stage_frontier_safe_<n>`):** the first time a `party_ready` frontier anchor is about to be overwritten by a merely-viable (hurt) one, the healthy state + meta are copied atomically into exactly **one** fixed file pair per stage (shared dir only). It is never overwritten by a weak state and holds no history; a later strictly-deeper healthy anchor may refresh it (never regressing its score). Every `FRONTIER_SAFE_FALLBACK_EVERY`th FRONTIER rank (default 3 → ~1/3) resumes this safe fallback **instead of** the main anchor **while the main anchor is a hurt state**, so a formally-viable-but-practically-worse main anchor cannot pin the whole FRONTIER fleet. Entry checkpoints, BRIDGE, FULL and RETENTION are unaffected.

**FIGHTER never starts from a hurt main anchor.** When `stage_frontier_2` is a hurt state, a FIGHTER episode start is: `stage_frontier_safe_2` → strict `stage_2` entry → deepest entry → master. It never loads the hurt main anchor. When the main anchor is `party_ready`, FIGHTER uses it exactly as before.

`_v20_stage_checkpoint_name(..., "frontier")` accepts a strict-ready anchor, an old meta that predates the health fields, **or** a `frontier_viable` anchor. `"frontier_safe"` is **strict-only**: it is returned only when `meta.get("party_ready") is True` — a missing or non-`True` field is rejected.

`_load_curriculum_state` checks the party in an explicit order:
* `stage_fighter_*` → `party_ready` required;
* `stage_frontier_safe_*` → `party_ready` required (checked *before* the general prefix so the safe fallback can never load on a merely-viable party);
* other `stage_frontier_*` → `frontier_viable` sufficient.

Old `party_ready=True` metas keep working unchanged. Viability fields (`frontier_viable`, `party_alive`, `party_total_hp_ratio`) are written additively into the meta.

Capture checks run on refreshed positions (every four agent actions), after three stable map readings, outside battle and wipe cooldown. There is no episode-end or thousands-of-steps wait. Entries stay immutable; only FRONTIER advances `stage_frontier_<n>`. The saved score is the **actual saved position's** value with fractional precision, not an old episode peak or truncated integer.

The local Route-1 anchor was repaired in an isolated emulator: Squirtle 31/31, Rattata 16/16, Pidgey 15/15, same Route 1 position (17,24), species, levels, PP and story. The previous files are backed up under `brain_backups/healthy_frontier_20260906_214032`. This is a local state repair, not a recurring heal during training, and is not a ROM/state payload shipped through Git.

FRONTIER/FIGHTER episodes end after a party wipe — **still episode-terminal, unchanged**. There is no mid-episode savestate reload and no artificial in-episode heal. A FRONTIER agent that wiped mid-route contributes its progress only through the `frontier_viable` anchor it may have saved *before* the wipe; the **next** normal episode reset then loads that (deeper) anchor. FRONTIER additionally truncates after 120 trusted non-battle steps below its starting stage; returning to the start stage or farther resets the counter. FIGHTER retains its 400-consecutive-non-battle-step leash and its combat-only reward selection. FULL/BRIDGE/RETENTION retain their recovery behavior.

The viability bar means a hurt-but-playable FRONTIER agent can now anchor its spatial progress; it no longer needs a pristine party at a new depth (unreachable on a no-heal route). Three things keep a weak state from trapping the fleet: the score invariant (a replacement never regresses depth), the +3.0-score gate for hurt candidates, and the fixed per-stage healthy safe fallback that ~1/3 of FRONTIER ranks resume while the main anchor is hurt.

## Episode length is not update frequency (legacy fallback)

- FULL/Watcher horizon: 32,768 **travel** actions. Scout-start roles: 12,000 travel actions. Battle actions are tracked separately.
- A single stuck battle is capped at 2,000 actions for everyone. Fighter is exempt from the total episode battle budget, not this single-battle cap.
- PPO: `NUM_ENVS` (46 in the fallback) × 512 actions per rollout/update; batch size 256, four epochs. Learning does not wait for an episode to finish. *(Under the live 2×2 path `NUM_ENVS = 40` and the navigation rollout is the 512-real-decision chunk in `src/twoby2/nav_chunk_rollout.py`.)*
- Resume publication: every 50,000 aggregate training steps. Champion checks: every 250,000 aggregate steps, subject to completed evaluation evidence. Longer episodes can delay completed-run evidence and champion promotion, not PPO updates.
- Restart chooses the resume checkpoint, then champion, then latest. Loaded policies keep their real counter.

Trainer remains unrestricted on `TRAIN_DEVICE=auto` (MPS here). Input cadence remains 9 held + 5 released emulator frames. Only watcher display is paced at 59.7 FPS. No audio is required.

## Dashboard and verification

Status and Watcher have search by ID/name/map/start, role filters, health/battle/checkpoint filters, current team health, and reported start health. Actual `training_mode` takes priority, with canonical role in the agent name as a legacy fallback. Agent 59 is Fighter. UI changes are loaded through the existing dashboard asset route without restarting the web server. Reload the dashboard when convenient; do not restart the streaming watcher. While training is stopped, cards retain the last reported training observations and do not represent the newly repaired start state until those agents reset.

149 regression tests passed. Isolated real-emulator starts and 32 actions for each of the five roles passed; Fighter outdoor reward was zero, Fighter/Frontier starts had 62/62 HP. A further isolated 1,500-action Fighter run produced battle wins, damage and level-up rewards with the documented amounts and no navigation/story reward events. Browser Status search for 59 returned exactly one Fighter. These checks establish implementation behavior, not an overnight learning result. Next training evidence should be real Viridian arrivals, stage-3/4/5 safe anchors, combat win/wipe rates and repeatable transitions toward Viridian Forest.

## Starting the consolidated version

Run `bash scripts/start_all.sh`. It starts trainer, watcher, dashboard (:8001) and status monitor in detached sessions, with logs in `runtime/logs/` and PID files in `runtime/`. Existing services are detected for both absolute and relative Python script paths. It never resets learning or starts external tunnels. Set `PKMAI_PYTHON` to use another installed Python environment. The status monitor accepts frontier checkpoint filenames.

On 2026-09-06 the user explicitly authorized restarting all services, including the watcher, after publishing this revision. Earlier stream-preservation notes describe the preceding maintenance, not a prohibition on this authorized restart.

## Exploration balance follow-up

All navigation roles (FULL/BRIDGE/FRONTIER/RETENTION) receive +0.3 for each tile first visited in the episode on a stage whose forward transition is still unknown, without the 20-tile reduction. A fleet-first tile adds +1 for the discovering navigation agent, including FULL. Repeated visits in the episode pay zero; scout backtracking below spawn stage stays unrewarded. Known-stage caps and all combat values are unchanged; Fighter still receives only combat rewards. This raises the incentive to reach unknown territory without increasing repeated-tile rewards. It does not directly change BRIDGE's mastery gate or promise faster convergence.

Wild battle decay follow-up: `WILD_BATTLE_DECAY_AFTER=3`, `WILD_BATTLE_DECAY_FACTOR=0.1`. Applies to positive damage/win/level/faint components on wild-training maps for navigation roles. Fighter remains exempt, trainer-battle multiplier and all penalties unchanged. The counter resets each episode.

## Trainer-battle bonuses

Normal combat shaping and trainer ×2 scaling remain. Each trainer ID pays +50 on battle start and +50 on the full battle victory (outcome=1), each once per episode. A single enemy KO/EXP increase does not grant the +50 victory bonus. Repeated losses/re-entry cannot repeat the start bonus within the episode. Fighter receives these combat bonuses too. Brock (ID 414) pays +500 instead of +50 at start, then the usual +50 trainer win bonus. A badge pays +2000; the previous extra global +5000 and approximate Pewter trainer-KO +300 bonus are disabled.

The former battle-flags offset 0x22FEC was incorrect; use 0x22B4C. Trainer ID is at 0x386AE and complete battle outcome at 0x23E8A. The local BPRD ROM contains the matching ordinary-trainer argument table at ROM offset 0x3C674C. Sources: [trainer argument table](https://raw.githubusercontent.com/pret/pokefirered/master/src/battle_setup.c), [battle RAM symbols](https://raw.githubusercontent.com/Skeli789/Complete-Fire-Red-Upgrade/master/BPRE.ld), [Brock ID](https://raw.githubusercontent.com/pret/pokefirered/master/include/constants/opponents.h). No ROM content is distributed.

## 2×2 Navigation/Battle architecture — build history

> **This section is the build record.** For what runs today see "What is running
> now (2×2 live)" at the top of this file — the 2×2 stack was cut over on
> 2026-09-07 and is training now. Delta rounds 2–8 below track how it got there.
> The "gated OFF / cutover pending" wording in rounds 2–7 was accurate when
> written; round 8 is the cutover.

The 2×2 split (navigation learner/champion + battle learner/champion) is
implemented and tested in `src/twoby2/` and the top-level `battle_*` modules.
`twoby2.FEATURES` ships all-OFF; `PKMAI_TWOBY2_LIVE=1` (set by
`scripts/start_2x2_visible.sh`) flips the six live gates ON at runtime — which is
how it runs today. The legacy single-PPO `train.py` path (no env var, roles
FULL/BRIDGE/FRONTIER/RETENTION/FIGHTER) is the rollback target. Full design +
per-module status: [docs/BATTLE_ARCHITECTURE.md](BATTLE_ARCHITECTURE.md) §2b.

Final architecture: **40 navigation workers, all starting at the identical
canonical start (`StartGame.state`), no anchor/FRONTIER/BRIDGE/RETENTION/FIGHTER
roles** + **9 fully isolated battle workers** (8 headless + 1 frame-mirror; own
PPO/optimizer/rollout/counters/checkpoints) + **1 FULL-watcher emulator** = 50
total (`MAX_TOTAL_EMULATORS`). Navigation episode length follows an adaptive
horizon ladder `[2000 … 163840]` that is separate from `PPO_N_STEPS=512`
(unchanged); an existing champion is migrated to a horizon derived from its real
metrics and never below the old live 32768.

**Legacy-removal audit (section 17):** `FIGHTER` / BRIDGE / FRONTIER / RETENTION
still appear in `pokemon_env.py`, `curriculum_v20.py`, `watcher_runtime.py`,
`checkpoint_health.py`, `train.py`. Under the live 2×2 path they are inert
(`_v20_mode()` forces `MODE_FULL`), but the code is the rollback target, so it is
**kept** until the legacy stack is formally retired. Nothing was removed.

**Battle-menu RAM blocker — CLEARED in delta round 4 (2026-09-07):**
`gBattleMons` (`0x23BE4`), `gBattlerPartyIndexes` (`0x23BCE`),
`gActionSelectionCursor` (`0x23FF8`), `gMoveSelectionCursor` (`0x23FFC`) and
`battle_menu_state` (`0x22BC4`) are **verified for BPRD** from 20 real dumps
(`runtime/ram_probe/20260907_184350`) and read by
`src/twoby2/battle_ram_live.py`. The still-open item is the **party-list
cursor** → live SWITCH stays masked. (This paragraph was written before the
probe; see delta rounds 4–7 below.)

## 2×2 delta round 2 (2026-09-07) — full-reset preparation (still gated OFF)

- **Protected savestates** (`twoby2/protected_assets.py`): the canonical master
  `local/custom_integrations/PokemonFireRed-Gba/StartGame.state`
  (sha256 `0c0e26d9b6e321381ffca4593bf2cc04e8e9d282ddbdd799c07e76b4c66683d0`,
  Oak's lab bank 4 / map 3 / x6 / y4, starter+parcel already done) is registered
  `immutable_user_master` — never delete/replace/normalize/rename, survives every
  reset incl. `reset_full`. The Route-1 fighter seed is **ambiguous**: two
  plausible pairs exist —
  `runtime/curriculum_shared/stage_frontier_2.{state.gz,meta.json}` (live FRONTIER
  anchor, Route 1 x10/y6, Squirtle L11/Rattata L4/Pidgey L3) and
  `brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.*` (documented
  manual repair, x17/y24). The runtime pair is protected as a candidate; the user
  must confirm which is "the" seed. Every reset/migration/normalize path calls
  `assert_not_protected` (symlink/`..`/alias resistant) before touching a file.
- **Retention corrected**: `intro/stairs_down/left_house/starter` are
  `precanonical` (before the master) and never block a promotion; the active
  chain is `leave_oak_lab → pallet_route1 → route1_viridian → …`.
- **Reset baseline + reward split** (`twoby2/reward_split.py`): existing
  starter/dex/parcel/party/level/flags/inventory/position at master-load pay 0;
  navigation and battle reward channels are separately versioned and reject each
  other's terms; a battle win never double-counts into navigation.
- **Battle workers**: 8 headless (route groups on the newest routes) + 1
  frame-mirror `PKMai – BATTLE` worker (`src/battle_watch.py`) = `BATTLE_WORKERS
  = 9`. (This superseded the earlier 9+1=10 split.) The mirror worker is
  worker 9 of the central battle learner, no own optimizer/model, fail-closed
  while not real-training-ready.
- **RAM probe kit**: `tools/battle_dump_collect.py` + `battle_dump_score.py` +
  `docs/RAM_PROBE_GUIDE.md`. Nothing was scripted/adopted — the user must run
  the manual probe. Until then: **NOT ready for full-reset / 2×2 start**.

## 2×2 delta round 3 (2026-09-07) — audit fixes (still gated OFF)

- **Route-1 seeds classified**: `route1_manual_battle_seed`
  (`immutable_user_battle_seed`, `candidate_ambiguity=False`) =
  `brain_backups/healthy_frontier_20260906_214032/stage_frontier_2.{state.gz,meta.json}`
  (SHA256 state `4244fa04db32fcd313ab159930787a613367904d5bd0f9c6ad8a0654ff6b0b48`,
  meta `37087431bf762a4cdaf2c885b800d194629f06b5bcd0e782d99c520d6fd57682`).
  `runtime/curriculum_shared/stage_frontier_2.*` is now a separate identity
  `protected_live_route1_frontier_anchor` (`protected_live_anchor`, SHA256 state
  `4c22f65e7566ef68b04d5a0ff98af69d858cce95abc73af846ceeef5e48e2563`, meta
  `dee031c1f09e146616724722195098af244629783cb3a0dedbbd5ed82433b8ad`) — mutable
  live anchor, protected from reset/migration/cleanup, never normalised directly.
- **Protected sources are read-only copyable**: `protected_assets.read_only_copy`
  makes a byte-identical copy into an UNPROTECTED temp path with the source
  sha256 verified before and after; `assert_write_target_ok` guards every real
  write / publish target (normalize/overwrite/replace/rename/move/delete/publish).
- **RAM probe kit completed**: `battle_dump_collect.py` sidecars carry the full
  verified party, `encounter_id`, `battle_kind`, labelled `menu_state`, expected
  cursor. `ram_battle_probe.verify_all` checks 5 mandatory fields with positive
  + negative contexts, ≥2 distinct encounters, and cross-checks `gBattleMons`
  against the `gBattlerPartyIndexes` slot after a switch (never `party[0]`).
  `gBattleWeather` = optional, not a blocker. `battle_dump_score.py` exits
  non-zero while any mandatory field is unverified.
- Status unchanged: all feature gates False; no full-reset/migration; `battle_watch.py`
  fail-closed; `watcher_parity.py` / `reward_split.py` are prepared components,
  not live-integrated. **NOT ready for full-reset** until real RAM addresses,
  `EmulatorBattleDriver` and live integration exist.

## 2×2 delta round 4 (2026-09-07) — verified RAM + real read path (NO-GO for live)

The 5 mandatory battle addresses are VERIFIED for BPRD from the 20-dump dataset
`runtime/ram_probe/20260907_184350` (`battle_dump_score.py` → ALL MANDATORY
FIELDS VERIFIED):

  gBattlerPartyIndexes 0x23BCE · gBattleMons 0x23BE4 (4×0x58) ·
  gActionSelectionCursor 0x23FF8 · gMoveSelectionCursor 0x23FFC ·
  battle_menu_state 0x22BC4 (18=main / 20=move / 22=party) ·
  gBattleWeather 0x23F1C (optional). GBA addr = 0x02000000 + offset.

New (isolated, gated OFF):
- `src/twoby2/battle_ram_live.py` — verified readers + full `struct BattlePokemon`
  (species/stats/moves/PP/types/ability/HP/level/status1-2/stat-stages). All 20
  real dumps cross-check; `gBattleMons[player]` is checked against
  `gBattlerPartyIndexes[player]` (never `party[0]`), incl. the post-switch dump.
- `src/twoby2/emulator_battle_driver.py` — real `EmulatorBattleDriver` +
  `EmulatorBattleIO` + `build_live_snapshot`. Read path verified against real
  dump RAM. Fail-closed: unknown menu → no button; double battle → blocked;
  trainer-RUN → masked; bounded presses / message-advances / per-turn wall-clock
  cap. Policy is pinned per battle.
- `tools/twoby2_preflight.py` — fail-closed preflight (ROM/master/seed hashes,
  RAM manifest verified from dumps, live-reader/dump cross-check, 40+9+1 config,
  isolation, protected-write rejection, model roles, feature gates, files).

**Preflight result: NO-GO.** The single blocking check is `live_battle_canary`
— executing a real MOVE macro in a live battle and observing HP change / switch
/ RUN / trainer-RUN block / handoff. That needs an interactively-reachable
battle (the encounter cannot be triggered programmatically from a raw
`set_state`). Migration / full-reset / process start / 5-min monitored rollout
are NOT performed. All feature gates stay False. Old trainer/watcher/web were
stopped externally; restart with `bash scripts/start_all.sh` (or the project's
resume flow) to abort the cutover.

## 2×2 delta round 5 (2026-09-07) — gaps closed, honest preflight, real canary tool

- `src/twoby2/nav_progress.py` — navigation progress is geographic/story ONLY;
  `navigation_progress_delta` ignores level/XP/KOs; `strip_level_from_promotion_metrics`
  now scrubs `twoby2/promotion.py` inputs (recursive). Level→progress 0, 100 wild
  wins same place → promotion impossible (tests F1–F5, F13–F14).
- `src/twoby2/nav_chunk_rollout.py` — `NavChunkedRollout`: PPO updates every 512
  real navigation decisions regardless of horizon (≤163 840); chunk boundary ≠
  terminated/truncated, no env reset, bootstrap = last value; real `terminated`
  → bootstrap 0; real TimeLimit truncation → bootstrap from terminal obs. Battle
  sub-episode + watcher steps add nothing to nav counters (F6–F12).
- `tools/live_battle_canary.py` — the ONE interactive real canary. Isolated
  emulator, read-only working copy of `brain_backups/ram_probe_route1_seed/…`
  (hash before+after), user walks in, tool auto-detects the battle via the
  verified latch, then the REAL `EmulatorBattleDriver` runs: reader plausible,
  action mask, real MOVE_1 macro + observed state change, switch to slot 1 +
  `gBattlerPartyIndexes`/`gBattleMons` cross-check, switch back, wild RUN →
  stable OUT_OF_BATTLE; trainer phase: `battle_kind=trainer`, RUN masked & no
  RUN sequence sent, one allowed MOVE, controlled stop. Resumable; JSON report.
- `tools/twoby2_preflight.py` — 6 levels: `component_prepared` / `production_wired`
  (real call-site scan of pokemon_env/train/watch + battle trainers using the
  real driver) / `unit_tests_passed` (recorded result) / `real_canary_passed`
  (current valid PASS report, ROM+seed+addr match) / `migration_ready` /
  `activation_ready`. **File-exists never counts as wired or canary-passed.**

Current preflight: component_prepared YES, unit_tests_passed YES,
**production_wired no, real_canary_passed no → activation_ready no → NOT READY.**
Live wiring of pokemon_env/train/watch and the activation are the next task,
after a real canary PASS.

## 2×2 delta round 6 (2026-09-07) — live path WIRED (gated), preflight ACTIVATION-READY

Canary PASSED (`runtime/live_battle_canary/20260907_194636`, wild+trainer, ROM+seed ok).

- `src/twoby2/live_integration.py` — the single gated seam: `maybe_wrap_full_agent`
  (wraps a PokemonFireRedEnv with `NavigationBattleWrapper` + `NavBattleDriverAdapter`
  → real `EmulatorBattleDriver`, never Simulated), `battle_driver_for`
  (EmulatorBattleDriver when `battle_env` gate ON + live env, else Simulated for
  tests only), `assert_no_simulated_driver_in_live_path`.
- `src/train.py` `make_env._init` → `_twoby2_wrap(env, learning=True, battle_policy=…)`
  (no-op while `nav_battle_wrapper` OFF).
- `src/watcher_runtime.py` `make_evaluation_env` → `maybe_wrap_full_agent(env, learning=False)`
  (FULL-watcher parity: same wrapper/router/champions, no learning).
- `src/watch.py` → `watcher_battle_policy_choice()` consults the same `BattlePolicyRouter`
  (WATCHER mode) — battle champion / verified rule fallback, never a learner.
- `src/pokemon_env.py` step → `info["twoby2_split"]` / `info["twoby2_battle_active"]`.
  With the gate ON the wrapper drives the battle by stepping the raw emulator, so
  `PokemonFireRedEnv.step` and its whole combat-reward pipeline never run in
  battle → navigation PPO structurally never sees damage/KO/win/level.
- `src/battle_train.py` → `make_live_battle_env` + `EmulatorBattleDriver` via
  `LiveBattleDriver` when `battle_env` ON (needs captured battle-start savestates;
  fail-closed otherwise). `src/battle_watch.py` → `battle_driver_for`.
- `src/twoby2/nav_chunk_rollout.py` wired for import in train.py (512-decision
  PPO updates over the persistent episode; battle/watcher steps add nothing).
- `tools/twoby2_preflight.py` — `unit_tests_passed` now also fails if the
  recorded result is STALE vs any src mtime.

**Preflight: ACTIVATION-READY** (component_prepared / production_wired /
unit_tests_passed(583, later 592) / real_canary_passed / migration_ready /
activation_ready all YES). **All FEATURES gates still OFF** — nothing runs until the operator
flips them + migrates + resets + starts + monitors. Backup:
`brain_backups/pre_2x2_activation_20260907_200507/` (RESTORE.md + MANIFEST.sha256).

## 2×2 delta round 7 (2026-09-07) — worker config finalised, docs rewritten

- **Worker config finalised**: `NAV_WORKERS = 40`, `BATTLE_WORKERS = 9`
  (`BATTLE_HEADLESS_WORKERS = 8` + `BATTLE_VISIBLE_WORKERS = 1`),
  `FULL_WATCHER_EMULATORS = 1` → **40 + 9 + 1 = 50** (`MAX_TOTAL_EMULATORS`).
  Battle worker 9 is a real member of the 9-worker learner that also mirrors its
  frames to a window (`tools/battle_mirror_watch.py`) — not an extra emulator.
  This superseded the earlier "9 headless + 1 visible = 10" split;
  `tools/twoby2_preflight.py` check renamed `worker_config_40_8_1_plus_watcher`.
- **Test record**: `runtime/twoby2_test_result.json` = **592 tests green**,
  fresh vs all `src/twoby2/*.py` + the wired live modules → preflight
  `unit_tests_passed` YES.
- **`scripts/start_2x2_visible.sh`** (osascript, visible windows,
  `PKMAI_TWOBY2_LIVE=1`, `PYTHONPATH=src`) starts NAVIGATION 40 (`src/train.py`)
  · BATTLE 9 (`src/battle_train.py --workers 9`) · BATTLE mirror
  (`tools/battle_mirror_watch.py`) · FULL WATCHER (`src/watch.py`) · WEB · STATUS.
  There is **no** `src/nav_train.py` — navigation is `src/train.py` with the
  gated `_twoby2_wrap` seam.
- **Docs rewritten to this state**: `README.md`, `docs/ARCHITECTURE.md`,
  `docs/BIG_CHANGES_TODO.md`, and `docs/BATTLE_ARCHITECTURE.md` (status table,
  worker counts, verified-RAM section, module table, migration/reset commands).
  `docs/AI_HANDOFF.md` (V10.25) and the older entries of `docs/AI_STATUS.md` /
  `docs/STATUS_TODO.md` are kept as history.
- **Preflight: still ACTIVATION-READY**, all 6 levels YES, all FEATURES gates
  still OFF. No migration, no reset, no gate flip, no process start performed in
  that round — the cutover is the operator's supervised step.

## 2×2 delta round 8 (2026-09-07) — CUTOVER EXECUTED, 2×2 stack LIVE

The operator ran the cutover:

- `tools/migrate_to_2x2.py --execute` → `runtime/model_manifest.json`
  (`migration_state: executed`, schema `model_manifest_v2`); single-PPO champion
  → `runtime/navigation/checkpoints/navigation_champion.zip`, resume →
  `navigation_learner.zip`; battle champion = `battle_champion.rule.json` rule
  marker; originals archived.
- Learners reset from their champions, then `scripts/start_2x2_visible.sh`
  launched the split stack with `PKMAI_TWOBY2_LIVE=1`.
- **Now training** (see "What is running now" at the top): navigation = 40 FULL
  agents from the master start on one PPO (learner ≈ 2.72 M, champion v5 @ 2.5 M);
  battle = 9 workers on a separate PPO (≈ 200 K steps, learner_version 43,
  champion still the RULE fallback, 0 wins / 37 KOs / 1699 episodes so far — a
  training concern, not a wiring one); watcher inference-only on navigation
  champion v5.
- `train.py` now: `NUM_ENVS = 40` under `PKMAI_TWOBY2_LIVE`; model/status paths
  move to `runtime/navigation/`; `max_level` dropped from the champion promotion
  key and metrics floor. `_v20_mode()` returns `MODE_FULL` for every rank while
  the `nav_battle_wrapper` gate is on — hence "only full agents".
- `src/web_stream.py` reworked to a single `twoby2_brains_v1` status
  (navigation + battle brain cards); `tools/pkmai_status.py` shows `PKMai 2x2
  STATUS`.
- Legacy-removal of the FIGHTER role / single-brain assumptions from
  `pokemon_env.py` / `curriculum_v20.py` is still **deferred** — that code is the
  rollback path and is only reached with `PKMAI_TWOBY2_LIVE` unset.
- Rollback: stop the stack, unset `PKMAI_TWOBY2_LIVE`, restore from
  `brain_backups/pre_2x2_activation_20260907_200507/` (`RESTORE.md`), start the
  legacy stack. No commit, no push.
