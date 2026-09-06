# PKMAI — Pokémon FireRed AI by Alex

PKMAI is an experimental reinforcement-learning project that trains a PPO agent to play Pokémon FireRed with Stable-Retro. It combines visual input, RAM-derived navigation features, persistent exploration memory, curriculum states, a live watcher and a browser dashboard.

> This repository does not contain a Pokémon ROM or proprietary game assets. You must provide your own legally obtained game data and local Stable-Retro integration.

## Current training and watcher behavior

Only the trainer learns. The watcher evaluates valid `pokemon_model_resume.zip`
snapshots, independently of champion promotion. Before switching, it checks all
policy parameters for finite values and verifies a prediction. A rejected
snapshot never replaces an already working policy; a changed snapshot is checked
again. At startup, fallback order is champion, latest, then the local
`runtime/checkpoints/watcher_recovery.zip`.

### Watcher display baseline — 2026-09-06

The default watcher now displays **every emulator frame**, targeting **59.7 FPS**.
The AI still holds each button for **9 frames** and releases it for **5 frames**:
training inputs, action boundaries and observations are unchanged. Only the
watcher enables the optional per-frame display callback. Deadline pacing avoids
catch-up bursts after a pause. No audio output is required.

Statistics refresh separately every 0.5 seconds. A background worker publishes
JPEGs at up to 30 FPS with only one pending frame, so slow encoding cannot build
an old-frame backlog. The browser uses `/watcher.mjpg` instead of polling a JPEG
every 500 ms; hidden watcher views disconnect. Snapshot endpoints remain available.
Model loading and reward computation can still introduce pauses between actions.

**Current run: fresh reset, as requested by the user.** No old learned policy is
active. The temporary recovery model and existing progress were moved into
`brain_backups/fresh_reset_20260906_141332/`. Trainer, watcher, web and status
were restarted with empty curriculum/exploration state and a newly initialized
PPO policy. CPU training is the current stability baseline; the origin of the
previous non-finite checkpoint values has not been conclusively established.
The trainer checks initial policy weights and atomically publishes the fresh
resume immediately, allowing the watcher to act before periodic checkpoints.
The 9+5 inputs and per-frame display remain enabled.

**Verified:** 106 tests passed; the running watcher executed actions and reported
59.7 displayed FPS, with about 24 JPEG updates/s measured locally. These are
observed values, not guaranteed performance on every machine.

This behavior is the source default used by `src/watch.py` and `src/web_stream.py`.
Restart those processes after updating and reload the dashboard. Model archives,
backups, ROMs and runtime data are deliberately excluded from Git. On another
machine, supply a compatible, valid checkpoint locally; Git alone does not carry
model weights. Details: [live work log](docs/AI_STATUS.md).

As of **V20 CURRICULUM MODES** (`BUILD_TAG = "V20_CURRICULUM_MODES"`) the fleet
is split into four mutually exclusive training modes that all train the **same**
PPO policy (no second network):

| Mode | Start | Purpose |
|---|---|---|
| `FULL` | real `StartGame` | can the shared policy chain everything together? |
| `BRIDGE` | entry checkpoint of the current bottleneck stage | learn the first hop Full runs cannot reproduce |
| `FRONTIER` | deepest discovered frontier | discover the next unknown story transition |
| `RETENTION` | rotates mastered-transition entry checkpoints | prevent catastrophic forgetting |

plus the existing dynamic `POST_WIPE_RECOVERY` overlay. Allocation is the
12 / 12 / 6 / 3 ratio scaled to `NUM_ENVS` (at 60 envs: 24 / 21 / 10 / 5).

Two separate progress concepts now exist:

- **`discovered_stage`** — deepest world-stage *any* agent has ever reached.
- **`mastered_stage`** — deepest transition the *shared* PPO policy reproduces
  reliably. A transition (Pallet→Route1, Route1→Viridian, …) is *mastered* only
  when its rolling window (50) has ≥ 20 attempts, ≥ 80 % success **and** ≥ 5
  Full-from-start confirmations. A lucky scout can never promote a stage; only
  Full-chain evidence moves `mastered_stage`.

The **`current_bottleneck`** is the earliest discovered transition that is not
mastered; `BRIDGE` agents concentrate there and automatically walk forward
(Route1→Viridian → Viridian→Route2 → Route2→Forest → Forest→Pewter) as each hop
is mastered.

Navigation targets come **only** from real recorded `stage N → stage N+1`
crossings (`runtime/curriculum_v20/known_transitions.json`). An undiscovered
next hop → `UNKNOWN_NEXT_TRANSITION`: exploration on, no target, no fake
coordinate (no north-most point, map edge, nearest frontier, house or dead end).

Reward model (see the dated section in `docs/STATUS_TODO.md` for the full list):
the push forward comes from `STAGE_ADVANCE_REWARD` (+250 per new world-stage,
new episode best only) and **non-farmable** target shaping — `TargetShaper`
pays `TARGET_PROGRESS_REWARD` (0.05) × improvement **only on a new episode-best
graph distance** to the known target; returning to an already-achieved distance
pays nothing, so `A→B→A→B` can never repeatedly earn positive progress. Backtrack
is a tiny flat `−0.01` only past a 3-tile margin. Tiles are a flat "keep moving"
trickle (Pallet 0.1 … Pewter 3.0, first 20/map/episode then 10 %, +1 fleet-once).
New route 50/run, city 300/run, **generic city buildings pay 0** (a random house
is no longer a story jackpot — Center/Mart/Gym keep their own dedicated rewards),
first global stage unlock 1000-once. Pokécenter enter 50/run, deeper heal 500/run
(wipe respawn anchor), 1000-once; Poké Mart 100/run + 1000-once; badge 3000/run +
5000-once. Pewter/Brock split into small episode-flagged milestones (reach Pewter
with Pikachu +300, gym enter +200, Brock battle start +500, first gym KO +300).
**In a battle only continuous signals pay** — dealt/taken damage, healing,
level-up, catching; no flat KO or win bonus. Trainer battles pay double on damage
and skip the wild decay (30 % after 6 wild wins). All edge/warp/corridor farm
rewards stay off. A `ShortCycleGuard` detects `A-B-A-B` / `A-B-C-A-B-C` loops,
suppresses positive shaping, applies an escalating `−0.05 … −0.25` penalty and
truncates the episode after sustained cycling. Persistent claim history
(`reward_events.json`) keeps every one-time bonus one-time across restarts.

Checkpoints are now two kinds (the old "north-most Y wins" replacement is gone —
invalid in forests, caves, buildings): `stage_<n>` is the **entry** checkpoint,
saved on first safe entry and **immutable** thereafter; `stage_frontier_<n>` is
an optional discovery anchor that advances only on a strictly higher exploration
score, never on Y.

The long-Full-probe horizon bug is fixed: `_episode_step_limit()` gives the
deeper half of the FULL ranks the real `LONG_FULL_PROBE_STEPS` (32768) instead of
silently capping every Full episode at `MAX_EPISODE_STEPS` (12000).

After a party wipe a **recovery mode** kicks in (no novelty-memory reset, so
dying is never a farm): wild-battle rewards are cut to 5 %, generic catches pay 0,
and graph-distance guidance back to the pre-wipe story front pulls at ±0.50 until
that front (or a deeper Center respawn, or a badge) is re-reached — then a
one-time +300. The −100 wipe penalty and the Center-respawn teleport are unchanged.

## 2026-09-06 — V20 CURRICULUM MODES

`BUILD_TAG = "V20_CURRICULUM_MODES"`. Goal: an architecture that can eventually
learn the **complete** game instead of another one-off reward patch. Deployed
with a **100 % clean reset** — fresh PPO net, `world_stage 0`, no checkpoints,
empty stats; only the `StartGame` master savegame is kept. Backup in
`brain_backups/V20_CLEAN_RESET_*`. 100 unit tests pass.

### The problem it fixes

After ~23 M steps scouts could reach later areas but full runners still failed
around Route 1: the policy learned isolated checkpoint skills without chaining
them. Recent target shaping also created Pallet / house / two-tile oscillation
loops.

### New modules

| File | Role |
|---|---|
| `src/curriculum_v20.py` | modes, `CurriculumState` (discovered/mastered stage, rolling per-transition stats, `current_bottleneck`), `allocate_modes`, generic story-`Objective` system |
| `src/nav_transitions_v20.py` | `KnownTransitions` — KNOWN/UNKNOWN state, real recorded crossings only |
| `src/target_shaper_v20.py` | `TargetShaper` — non-farmable best-distance shaping |
| `src/loop_guard.py` | added `ShortCycleGuard` (A-B-A-B / A-B-C detection, escalating penalty, truncate); `LocalLoopGuard` unchanged |

### Four modes, one PPO policy

`FULL` (real `StartGame`), `BRIDGE` (bottleneck stage entry checkpoint),
`FRONTIER` (deepest discovered frontier), `RETENTION` (rotates mastered
transitions) — allocated 12 / 12 / 6 / 3 scaled to `NUM_ENVS`
(`curriculum_v20.allocate_modes`). `POST_WIPE_RECOVERY` still overrides
dynamically on a wipe. `V20_CURRICULUM = True` is the master switch; set it
`False` to fall back to the V17–V19 scout-band behaviour.

### discovered vs mastered stage

- `discovered_stage` = deepest stage any episode reached.
- A transition is **mastered** only when: window (`TRANSITION_MASTERY_WINDOW`
  50) has ≥ `TRANSITION_MASTERY_MIN_ATTEMPTS` (20) attempts, `success_rate` ≥
  `TRANSITION_MASTERY_RATE` (0.80) **and** `full_chain_confirmations` ≥
  `FULL_CHAIN_CONFIRMATIONS` (5). `full_chain_confirmations` is incremented only
  by `FULL`-mode, `episode_start=="beginning"` episodes
  (`record_full_chain_result`). BRIDGE/scout successes build the rolling rate
  but never promote a stage alone.
- `mastered_stage` = 1 + the contiguous run of mastered transitions from stage 1.
- `current_bottleneck` = earliest discovered transition that is not mastered;
  BRIDGE trains it and walks forward automatically as each hop is mastered.
- State lives in `runtime/curriculum_v20/state.json` (env reads it with a
  256-step cache, writes under the fleet lock at episode end).

### Two checkpoint types (brief §4)

The "north-most Y wins" replacement heuristic is **removed** — smaller Y is not
more story progress in a forest / cave / building / gym.

- `stage_<n>` — **entry** checkpoint. Saved on first safe entry (trusted RAM,
  correct stage, starter present, not in battle / not wiping, stable position)
  and then **immutable**. `BRIDGE` / `RETENTION` resume from it.
- `stage_frontier_<n>` — optional discovery anchor; advances only on a strictly
  higher exploration score (tiles explored on that stage), never on Y.
  `FRONTIER` resumes from it.

### KNOWN vs UNKNOWN transitions (brief §5, §21)

`UNKNOWN_NEXT_TRANSITION` → exploration on, target shaping off, no fake
coordinate. When a real forward `stage N → stage N+1` crossing is observed the
exact source map, source exit coord, destination map and destination coord are
persisted (`runtime/curriculum_v20/known_transitions.json`); that exact
transition becomes the navigation objective and exploration reward on the solved
stage drops away. Pallet is a solved transit area — its only objective is the
real `(3,0)→(3,19)` transition; missing → an explicit diagnostic, never an
invented target.

### Non-farmable target shaping (brief §6, §22)

`TARGET_PROGRESS_REWARD` 0.20 → **0.05**, new `TARGET_BACKTRACK_PENALTY = −0.01`.
`TargetShaper` maintains `best_target_distance` per objective/episode and pays
`0.05 × improvement` **only** on a strict new best. Returning to an achieved
distance pays 0; moving past `best + 3` pays a flat `−0.01`. Combined with the
`ShortCycleGuard` (suppresses positive shaping while cycling, `−0.05 … −0.25`
escalating, truncate after ~600 cycle steps) the anti-loop invariants hold:
A/B target loop, house in/out loop, warp replay, tile revisit, wipe farm, endless
wild battles, Center farm and `5→4→5` stage farm can none of them be net
profitable.

### Other changes

- `BUILDING_FIRST_GLOBAL_REWARD` 500 → **0** (brief §8). Generic Viridian/Pewter
  houses are worth 0; real objectives keep their dedicated `POKECENTER_*`,
  `POKEMART_*`, `PEWTER_GYM_*` rewards. The Bank-4 `!= 4` guard still stands.
- **Long-Full-probe horizon bug fixed** (brief §16): new `_episode_step_limit()`
  — `scout` → `SCOUT_EPISODE_STEPS`; `full` + long probe → `LONG_FULL_PROBE_STEPS`
  (32768); other `full` → `MAX_EPISODE_STEPS` (12000). Previously every `full`
  episode was forced to 12000, making `_is_long_full_probe()` inert.
- Generic story-`Objective` representation (`reach_map`, `reach_transition`,
  `enter_required_building`, `heal_center`, `win_trainer`, `win_gym`,
  `obtain_badge`, `obtain_item`, `trigger_story_flag`) so the same architecture
  extends to Route 3 → Mt. Moon → Cerulean → Misty → … → Elite Four → Champion by
  adding objects, not code. `world_stage` (geography) and `story_objective` stay
  separate.
- Dashboard `info` fields: `training_mode`, `current_stage`, `discovered_stage`,
  `mastered_stage`, `current_bottleneck`, `objective`, `target_source`,
  `target_coordinate`, `best_target_distance`, `post_wipe_recovery`,
  `transition_attempt`, `transition_success`. Reward events distinguish
  `route_progress_best`, `loop_penalty`, `route_backtrack`, `stage_advance`,
  `post_wipe_front_recovered`.

### Reset / restart

`bash tools/v20_reset.sh --yes` — full clean wipe (keeps only the `StartGame`
master savegame), then `bash scripts/start_all.sh`. `train.py` seeds
`discovered_stage` from any valid `stage_*` metas + `global_progress.json` and,
if the champion / global record already shows Full depth ≥ Route 1, one-time
pre-confirms `Pallet→Route1` so the detected bottleneck starts at
`Route1→Viridian`. On the clean reset there is no such evidence, so the first
picture is `discovered = mastered = 1`, bottleneck `Pallet→Route1`, whole fleet
running `FULL` until the fresh net actually holds Route 1.

## 2026-09-06 — V19 BROCK RUSH + POST_WIPE_RECOVERY

`BUILD_TAG = "V19_BROCK_RUSH"`. Goal: faster real story progress to badge 1,
without re-introducing reward loops. Existing logic reused; every V17/V18
anti-farm fix kept. 72 unit tests pass. Deployed with a map/global reset (brain
kept — see below); all fleet-once bonuses re-fire against the new reward shape.

**Forward push moved off exploration novelty.** Tiles are now just a flat
"keep moving" trickle: `TILE_REWARD_BY_STAGE = {1:0.1, 2:1.5, 3:2.0, 4:2.5,
5:3.0, 6:3.0}`, first 20 per map per episode then 10 %, +1 fleet-once. The real
pull comes from:
- `STAGE_ADVANCE_REWARD = 250` per new world-stage — paid only on a new episode
  best (`_world_stage()` is monotone within an episode, so walking back pays 0).
- `TARGET_PROGRESS_REWARD = ±0.20` graph-distance shaping. New helper
  `_v19_forward_targets(bank, map_id)` returns the on-map transition coords that
  lead to a *higher* world-stage map (or the current city's Center while unhealed,
  or the Pewter gym while Brock is unfought). Pure graph distance, no compass;
  symmetric so a there-and-back nets zero. Wired as a third fallback in the
  existing `target_closer` / `target_farther` block (the older generic target
  sources deliberately return nothing for world roles — they used to prefer
  houses and dead ends).

**Milestones toward Brock** (each once per episode via an episode flag, never
farmable by re-entering or re-starting a battle):
- reach Pewter with Pikachu in the party: +300 (`PEWTER_WITH_PIKACHU_REWARD`)
- Pewter gym entered: +200 (inert until `PEWTER_GYM_MAPS` has a confirmed id)
- Brock/gym battle started (trainer flag + `bank == 6` or world-stage 6): +500
- first gym KO in such a battle: +300 (**approximate** — without a trainer-id RAM
  read this can't be told apart from Brock's first Pokémon if the gym trainer is
  skipped)
- badge itself: `BADGE_EARNED_REWARD` 3000/run + `BADGE_FIRST_GLOBAL_REWARD` 5000-once

**Other value changes:** `FRONTIER_SCOUT_SLOTS` 2→3 · `EPISODE_NEW_MAP_REWARD`
25→50 · `CITY_EPISODE_REWARD` 250→300 · `LEVEL_GAIN_REWARD` 10→15 ·
`POKECENTER_ENTER_REWARD` 100→50 · `POKECENTER_ADVANCE_HEAL_REWARD` 250→**500**
(the wipe-respawn anchor) · `SPECIES_CAUGHT_FIRST_REWARD` 120→50, level bonus 4→2 ·
`PIKACHU_FOREST_CAUGHT_REWARD` 1000→400 (Pikachu stays useful for Misty but is
not a Brock prerequisite) · `TILE_REWARD_AFTER_CAP_FACTOR` 0.2→0.1. All
edge / warp / corridor farm rewards stay at 0.

**`POST_WIPE_RECOVERY_MODE`.** After a wipe the episode keeps going and visited
tiles/maps rightly don't repay, so wild grass at the respawn can become the best
remaining reward stream and the policy just fights there instead of walking back
to the front. `_record_party_wipe()` now also sets `post_wipe_recovery = True`
and stores `pre_wipe_best_stage / _best_center_stage / _badges` — **no** reset of
`seen_coords` / `visited_maps` / any novelty memory (dying on purpose must never
be a farm). While recovering:
- wild-battle rewards (damage + level-up) are additionally ×`0.05`
  (`POST_WIPE_WILD_BATTLE_SCALE`); trainer / gym / Brock battles are untouched
- the generic catch reward is 0 (the Pikachu-forest bonus is a separate branch and stays)
- the graph-distance guidance back to the old front pulls at ±`0.50`
  (`POST_WIPE_TARGET_PROGRESS_REWARD`)

Recovery ends — checked out of battle on the outdoor-coordinate path — when the
agent's *current map* stage reaches `pre_wipe_best_stage`, or a deeper Center
respawn was activated, or a badge was won. Then a one-time
`post_wipe_front_recovered: +300` (`POST_WIPE_FRONT_RECOVERED_REWARD`) and
`post_wipe_recovery = False`. The `party_wiped: -100` charge and the
Center-respawn teleport are unchanged.

**Story priority the shaping encodes:** Route 1 → Viridian → Viridian Center →
Route 2 → Viridian Forest → (optional Pikachu) → Pewter → Pewter Center →
Pewter Gym → Brock → badge 1.

**Deploy:** full stop, backup + delete `exploration_memory/agent_*.json` +
`reward_events.json`, `global_progress.json` → `max_world_stage 0` (fleet and the
watcher's isolated dir), keep all model / skill zips, `champion_score.json`,
`model_version.json`, savestates and stage checkpoints; full start. Backups under
`brain_backups/V19_*`.

## 2026-09-06 — V18: per-run tile ladder, one-time fleet bonuses, battle rebalance, dashboard

Full trainer + watcher + web restart (brain kept: learner ~21.5M, champion v9).
67 unit tests pass. Live-verified in the watcher and dashboard.

**Tiles pay per run now.** The fleet-once tile bonus meant the watcher (and any
agent on long-known ground) saw no tile reward at all. First-find of a tile is
now rewarded every episode (`seen_coords`), on a hand-set ladder tied to the
tile's own map: `TILE_REWARD_BY_STAGE = {1:0.2, 2:3, 3:4, 4:5, 5:5, 6:6}` — Alabastia
is the spawn so exploring it barely pays; Route 1 onward is where it kicks in, so
the pile of tiles in Pallet never beats the higher rate ahead. Interior tiles by
city bank: Pallet houses 0.2, Vertania 2, Marmoria 3. Only the first 20 new tiles
per map per episode pay (`new_tile:…:capped` after). On top, `GLOBAL_NEW_TILE_BONUS = 1`
fleet-once for the very first agent ever to step on a tile — first foot into
Pewter = 6 + 1.

**One-time fleet-wide bonuses** (persisted in `reward_events.json`, cannot repay
after a restart), each on top of a per-run value:
- Pokécenter: +100/run to enter, +250/run for a heal at a center deeper than any
  used this run (the wipe respawn point advances), **+1000 once per center**.
- Poké Mart (Vertania `5,3`): +100/run to enter, **+1000 once** — so the brain
  learns the shop exists and can buy Poké Balls.
- First badge: +2000/run kept, **+5000 once** per badge number.

**Battle rebalance** (watcher was 45 % of steps fighting): enemy-damage reward
0.15 → 0.08 per HP; after 6 wild wins in an episode on a wild map, wild
damage/faint/win drop to 30 %; **trainer/gym battles pay ×2** and are exempt from
that decay. Catch reward 50 → 120 + `min(level,20)×4`.

**Scouts** back to 2 per checkpoint (from 5) — scouts were getting through,
full runners were not.

**Warp reward** now claims a coarse `(map_a↔map_b)` pair instead of exact
coordinates, and it is persisted (`claim_event` → `reward_events.json`) plus
pre-seeded from the navigation history, so the first global warp bonus truly
fires once ever and does not repay after a restart (the watcher was granting
+100 for Pallet↔Route 1 on every restart). Navigation data stays
coordinate-exact. No warp reward on the step a battle ends. A scout that drifts
back onto/behind its spawn stage now also earns no per-run tile reward there.

**Dashboard:** the watcher live image is gone from the Overworld-Map side column
(it only shows under the Watcher tab now); that column shows the clicked agent's
detail plus its **last 10 reward events**. Localisation: ~40 missing German→English
strings added, the `Beste`→`Best` substring bug that produced "Bestr" removed, and
the translation pass rewritten to run synchronously in the mutation observer
(`requestAnimationFrame` was paused in background tabs and silently stopped
translating) with a 700 ms safety interval.

**Deferred** to `docs/BIG_CHANGES_TODO.md`: a separate combat policy (FighterBrain)
beside the champion, and special handling for the house past Viridian Forest
(needs its map id first).

## 2026-09-06 — Geographic progression and battle/loop corrections

These changes supersede the older parcel-based stage descriptions below. No trainer,
watcher, web or mapper process was stopped or restarted during this change.

- Immutable master: the user-recorded `local/custom_integrations/PokemonFireRed-Gba/StartGame.state`
  after Oak's parcel, verified inside the lab at bank 4/map 3, x6/y4. Every full runner
  restores this exact original; only scouts load geographic checkpoints.
  The briefly prepared outdoor derivative was removed, per the latest instruction.
  Lab/indoor baseline counts as stage 1 and produces no geographic stage jump.
- Geography: Pallet 1, Route 1 2, Viridian 3, Route 2 4, Viridian Forest 5, Pewter 6.
  Parcel flags, stairs, buildings and badges do not manufacture geographic stages.
- Exactly five scouts per valid checkpoint on Route 1, Viridian, Route 2, Forest and Pewter.
  Fixed rank bands prevent reassignment when another checkpoint appears. Replacing
  a savestate never adds scouts. No Pallet scouts. Other agents continue full runs.
- Each stage independently prefers a further-north position (smaller Y), even
  with lower reward. At equal Y, strictly higher reward wins; south is rejected.
  This also applies to later full runners returning through earlier maps. Battle/wipe states are not captured.
  Legacy lab checkpoints are rejected by their actual map; old fleet depth is
  migrated from persisted visited maps at trainer startup, preserving model weights.
- Battle detection uses dynamically validated gMain.inBattle (not battle-type flags).
  An isolated real-ROM test detected a zero-type-flag wild encounter and cleared the
  battle after fleeing without overworld movement. Observations use this same signal.
- Fresh party HP on every decision; one wipe charge until recovery; wipe healing
  receives no heal bonus. EXP rewards require the same party and battle context.
- Global center-heal claim now persists in exploration_memory/reward_events.json.
  Historical claims made before this file existed cannot be reconstructed reliably;
  the first post-migration qualifying claim establishes the persistent baseline.
- A shared local-loop guard ends an episode after 900 overworld decisions on at most
  eight tiles without exploration/EXP/stage/badge progress. Battles pause counting.
  This also bounds short back-and-forth loops that reset the old stationary counter.
- Watcher remains policy evaluation without learning. Updated code takes effect only
  on the next explicitly approved restart; refreshing navigation is not code reload.

Validation: 58 unit tests, plus isolated emulator checks. Full environment smoke
validation: 1,000 isolated decisions. Two final resets restored the original master
location and stage 1; its SHA-256 remained unchanged.
No user savestate was modified.

## Previous release (historical)

**V17.4 — Reward-Rebalance gegen Farm-Loops, Frontier-Scouts pro Stage, kritische Checkpoint-/Warp-Bugs behoben** (2026-09-06, Nachtsession)

Ausgangspunkt war ein wiederkehrendes Live-Symptom: die Flotte lief lieber
zurück in längst bekanntes Gebiet (Alabastia, Starterhaus) statt weiter nach
Norden vorzustoßen, weil dort garantierter, risikofreier Reward wartete.
Mehrere Runden aus Live-Beobachtung → Root-Cause-Suche → Fix haben dabei vier
unabhängige, teils schon länger aktive Bugs zutage gefördert.

**Reward-Rebalance (Farm-Loops geschlossen):**
- Kanten-Reward komplett auf 0 (auch nicht mehr beim ersten Mal) — der
  Pro-Episode-Anteil resettete sich bei jedem Reset und machte bekannte
  Kurz-Loops am Spawn jede Episode neu profitabel.
- Kachel-Reward: Pro-Run-Anteil (`EPISODE_TILE_REWARD`) auf 0 — dieselbe
  Farm-Lücke wie bei Kanten, nur eine Stufe kleiner (sicheres Rumlaufen in
  bekannten Innenräumen gab bisher noch kleines Dauer-Einkommen). Nur der
  fleet-weite Einmal-Fund (`NEW_TILE_REWARD=2`) bleibt.
- Map/Stadt-Reward: kein fleet-weiter Einmal-Jackpot mehr (ging strukturell
  nur an den einen Agenten, der eine Map zuerst fand) — jetzt ein fester Wert
  pro Run für JEDEN Agenten (`EPISODE_NEW_MAP_REWARD=100`,
  `CITY_EPISODE_REWARD=250`). Gebäude-Erstfund (Reds Haus etc.) zahlt separat
  nur noch einmal für die ganze Flotte, nicht mehr pro Agent.
- Warp-Reward: kein Pro-Run-Bonus mehr für bekannte Türen
  (`EPISODE_TRANSITION_REWARD=0`), nur der fleet-weite Einmal-Fund
  (`NEW_TRANSITION_REWARD=100`) bleibt.
- Neuer, benannter `BADGE_EARNED_REWARD=2000` (vorher unbenannt inline 2500).
- Artenvielfalt-Fang und Pikachu-Wald-Bonus von fleet-weit-einmalig auf
  pro-Run umgestellt (`SPECIES_CAUGHT_FIRST_REWARD=50`,
  `PIKACHU_FOREST_CAUGHT_REWARD=1000`) — die Party resettet ja jede Episode,
  ein Lifetime-Claim hätte ab dem zweiten jemals gefangenen Exemplar nie
  wieder Anreiz gegeben, überhaupt zu fangen.
- `GAMEPLAY_STEP_COST` 5× verschärft (`-0.001` → `-0.005`), da die echten
  Ziele deutlich größer wurden, Herumstehen aber gleich billig blieb.

**Vier kritische Bugs live gefunden und behoben:**
1. **Stage-Checkpoints für Route 1/Vertania konnten nie entstehen.** Der
   feste Savestate startet mit bereits bestätigter Paketabgabe an Prof.
   Eich, wodurch `_world_stage()` (ein reiner Ratchet) ab Step 0 immer
   mindestens 5 zurückgibt — unabhängig vom tatsächlichen Standort. Der
   Checkpoint-Code nutzte diesen Wert sowohl (a) als zu speichernde
   Stufennummer als auch (b) als Baseline für `_saved_stage`
   (Anti-Doppel-Speicher-Schutz). Beide Stellen verglichen ihn gegen den rein
   standortbasierten `_stage_at_current_location()` (Route 1 = 2, Vertania =
   3) — 5 ≠ 2/3 schlug immer fehl. Beide Stellen jetzt auf den
   standortbasierten Wert umgestellt (`episode_best_stage`, das den
   Depth-Reward vor genau demselben Exploit schützt, bewusst NICHT
   angefasst).
2. **Alabastia (Pallet Town) gab jede Episode automatisch +250.** Der
   Spawnpunkt liegt in Eichs Labor, 1-2 Schritte vor Alabastia — da
   Alabastia selbst ein `CITY_MAPS`-Eintrag ist und längst bekannt, feuerte
   der Stadt-Wiederholungsbonus jede einzelne Episode automatisch, quasi nur
   fürs Rauslaufen. Alabastia explizit von diesem Bonus ausgenommen (andere
   Städte bleiben unverändert, da echte wiederholte Lauf-Leistung nötig).
3. **`shared_tiles` verlor seinen Fortschritt bei jedem Trainer-Neustart**
   (nicht nur bei einem vollen Reset) — anders als `shared_edges`/`shared_maps`/
   `shared_transitions` wurde es nie aus der persistierten Kanten-Historie
   neu geseedet. Jeder gezielte Trainer-Neustart öffnete dadurch kurzzeitig
   wieder das +2-Kachel-Fenster für längst bekanntes Terrain. Jetzt beim
   Start aus den Endpunkten der geladenen Kanten-Historie vorbefüllt
   (identische Logik im isolierten Watcher-Eval-Env, das zusätzlich alle 5
   Minuten automatisch nachlädt, ohne den Prozess neu zu starten).
4. **Party-Wipe-Teleport konnte einen Warp-Bonus auslösen.** Der
   Wipe-Cooldown schützte bisher nur Map-/Kachel-Rewards vor dem
   automatischen Pokecenter-Teleport nach einem Total-K.O. — der
   Warp-Reward-Block hatte nie eine solche Prüfung. Da die exakte
   Kampf-Position beim Fainten praktisch nie zweimal gleich ist, war das ein
   fast unerschöpflicher, immer "global neuer" Warp-Fund — jeder Wipe konnte
   zusätzlich zur -100-Strafe einen +100-Bonus einbringen. Jetzt ebenfalls
   während des Cooldowns unterdrückt (Buchführung/Claims laufen normal
   weiter).

**Frontier-Scout-System überarbeitet:** vorher wanderten alle
`FRONTIER_SCOUT_SLOTS` Scouts sofort zur tiefsten Front, sobald diese einen
Checkpoint bekam — die vorherige Front wurde komplett verwaist. Jetzt bekommt
jede validierte Stage ihre eigenen Scout-Slots dazu (`_scout_assigned_stage()`),
bestehende Paare bleiben stabil auf ihrer Stage. `FRONTIER_SCOUT_SLOTS` 2 → 5
(realistisch validiert die Flotte ohnehin nur 2-3 Stages pro Nacht). Die
Checkpoint-Haltezeit (`_hold_required`) von 25 auf 3 Lesezyklen gesenkt — die
eigentliche Qualitätssicherung passiert beim Ersetzen selbst (nur bei
höherem Episoden-Reward oder weiter nördlicher Position).

**Betrieb:** kompletter Reset aller Explorations-/Curriculum-/
Statistik-Daten (Backup in `runtime_reset_backup_*/`, nicht gelöscht) — das
trainierte Brain (`runtime/checkpoints/`), Champion-Metadaten und die
Web-Karten-Layout-Konstanten bleiben unangetastet. **Neue Standing-Regel:**
lief der Watcher gerade live (z.B. Stream), NIE über `stop_all.sh`/
`start_all.sh` mitneustarten — stattdessen den Trainer gezielt per PID
(`kill -INT`) stoppen und `start_all.sh` erneut aufrufen; das Skript erkennt
bereits laufende Prozesse (Watcher/Web/Status) automatisch und lässt sie in
Ruhe.

**V17.2 — Savestate-Start, Fleet-Rekorde, Artenvielfalt** (2026-09-05)

Seit V17 beginnt jede Episode nicht mehr am kalten Spielanfang, sondern an
einem festen, manuell erspielten Savestate (`StartGame.state`) kurz nach
Intro, Namensvergabe und Starterwahl (PWhiddy-Stil). Das Nachspielen der
~10-minütigen Introsequenz entfällt für jede der 60 parallelen Umgebungen bei
jedem Reset — der gesamte Trainingshorizont geht in echte Weltexploration statt
in wiederholtes Intro-Abspulen.

### Was sich seit V16 geändert hat

- **Savestate statt Kaltstart:** `env.load_state("StartGame", ...)` ersetzt den
  Kaltboot-Snapshot in `pokemon_env.py` und `watch.py`. Intro-/Treppen-/
  Hausausgangs- und Paket-Flags starten als bereits erledigt, damit kein
  Reset mehr fälschlich am alten Intro-Timeout (1800 Schritte) abbricht.
- **60 parallele Umgebungen** (`NUM_ENVS`), synchron mit `PPO_N_STEPS=512` →
  30.720 Samples/Update. `ACTION_HOLD_FRAMES=9` / `ACTION_RELEASE_FRAMES=5`
  (empirisch gegen 12/6 verifiziert: identische Bewegungszuverlässigkeit,
  +7–13 % Steps/Sekunde).
- **Persistenter, nicht farmbarer Kanten-Reward:** `NEW_EDGE_REWARD=1.0` für
  jede fleet-weit erstmals gelaufene Kachel-Kante (`_claim_shared`, genau
  einmal über alle Agenten und Episoden). Bereits bekannte Kanten geben dem
  einzelnen Agenten höchstens einmal `+0.20` (Imitationssignal), nie erneut.
- **Fleet-Rekord-Bonus (neu in V17.2):** `GLOBAL_STAGE_RECORD_REWARD=1000`.
  Wer als Erster im gesamten Brain einen neuen `world_stage`-Tiefenrekord
  erreicht (Route 2, Vertania-Wald, Marmoria, erster Orden), bekommt diesen
  Bonus einmalig fleet-weit — zusätzlich zum bestehenden, pro Episode und
  Agent wiederholbaren `NEW_GLOBAL_DEPTH_REWARD=1000 × Stufenanstieg`.
- **Artenvielfalt-Fangbonus (neu in V17.2):** erste je Spezies fleet-weit
  gefangene Pokémon geben `+1000` (`shared_species`, ebenfalls `_claim_shared`-
  geschützt); jeder weitere Fang derselben Art kostet `-500`. Verhindert
  Farmen häufiger Wildpokémon (Taubsi, Raupy) und belohnt seltene Funde wie
  Pikachu im Vertania-Wald.
- **Kampf-Rebalance:** `ENEMY_DAMAGE_REWARD_PER_HP=0.15`,
  `ENEMY_FAINT_REWARD=10`, `BATTLE_WIN_REWARD=15` (von 0,5/30/50 gekürzt,
  damit Dauerkämpfen im Wildgras Exploration nicht mehr strukturell schlägt).
  `FLED_BATTLE_PENALTY=-25`, kompletter Party-K.O. `-100` und sofortiges
  Episodenende (auch direkt beim Kampfende erkannt, nicht erst beim
  nächsten HP-Sample — verhindert einen Exploit, bei dem der automatische
  Pokémon-Center-Teleport nach einem Wipe als "neue Map" belohnt wurde).
- **Reproduzierbarkeits-Schwelle:** `STAGE_RELIABILITY_FRACTION=0.12` — ein
  neuer world_stage zählt für Champion-Aufstiege erst, wenn mindestens 12 %
  aller vollständigen Läufe ihn erreichen. Verhindert Champion-Beförderungen
  durch einen einzelnen glücklichen Ausreißer-Run.
- **Watcher-Anti-Loop korrigiert:** die alten, für teure Kaltstart-Resets
  gedachten Gnadenfristen (bis zu 8000 Schritte) ließen den sichtbaren
  Watcher-Lauf nach dem Savestate-Umbau unbegrenzt weiterlaufen, ohne je zu
  terminieren. Auf 900/1800 Schritte zurückgesetzt.
- **Dashboard-Weltkarte repariert:** eine zu klein bemessene feste
  `setMaxBounds`-Box zwang Leaflet, bis auf den Minimalzoom
  herauszuzoomen, egal welches Seitenverhältnis der Browser hatte — die
  Karte war dadurch faktisch unsichtbar ("links abgeschnitten"). Jetzt fester,
  nicht veränderbarer Zoom (nur noch verschiebbar, nicht mehr zoombar) und
  eine an die tatsächlich aufgedeckte Welt gekoppelte, automatisch
  mitwachsende Grenze mit 200 Kacheln unsichtbarem Rand — die Karte lässt
  sich dadurch nie mehr komplett aus dem Bild schieben.
- **Rollierendes Reward-Event-Log:** ein Klick auf einen Agenten im
  Dashboard zeigte bisher nur die Reward-Events des einen Schritts, in dem
  zufällig die Instanzdatei geschrieben wurde (alle 80 Schritte) — fast immer
  leer. Jetzt ein echtes Log der letzten ~40 tatsächlichen Ereignisse.
- `LEARNING_RATE=7.5e-05` (ein Experiment mit `0.0005` wurde nach über 2 Mio.
  Steps ohne Champion-Fortschritt als gescheitert bewertet und zurückgesetzt).

### Vorheriges V16 — Clean Full-Brain Generations (2026-09-04)

V16 beginnt nach einem vollständigen, gesicherten Reset aller alten Modelle,
Statistiken, Karten- und Curriculum-Daten. Der Stand unmittelbar vor dem Reset
wird als datiertes Backup erhalten. ROM, Quellcode und lokale Konfiguration
werden niemals gelöscht.

### Ziel und Grundprinzip

- Es existiert genau **ein gemeinsames PPO-Brain**. Keine Skill-Modelle, keine
  Progress-Agenten und keine gemischten Savestate-Starts.
- Alle **50 Trainings-Clients** beginnen jede Episode am echten Spielanfang und
  lernen die vollständige Kette Intro → Name → Haus → Labor → Schiggi → Welt.
- Die Clients laufen headless und ungebremst. Sie werden nicht auf 60 FPS
  reduziert und schreiben keine Screenshot-Karten.
- Der Mapper bleibt vollständig ausgeschaltet. Ein visueller Mapper kann später
  separat entworfen werden, beeinflusst aber weder Reward noch Champion.
- PPO sammelt pro Client 512 zusammenhängende Entscheidungen. Das ergibt
  25.600 Samples pro synchronem PPO-Update (`50 × 512`).
- `gamma=0,995` und `gae_lambda=0,98` lassen einen Erfolg weiter auf die
  vorherigen Entscheidungen zurückwirken als im alten 128er-Setup.
- Eine Episode hat zunächst höchstens 12.000 **Weg-Schritte**. Eindeutiger
  Stillstand beendet sie früher. Kampfentscheidungen werden separat gezählt:
  Sie gehören weiterhin zum PPO-Lernen, verbrauchen aber weder Intro- noch
  Routen-Horizont. Ein einzelner Kampf endet spätestens nach 2.000, alle Kämpfe
  einer Episode zusammen nach 6.000 Kampfentscheidungen als Sicherheitsgrenze.
- Stable-Retro erhält beim Erzeugen jedes Clients einen unveränderlichen
  Kaltstart-Snapshot. Jeder Episodenreset stellt genau diesen Snapshot wieder
  her. Damit können Party, Starter, Karte oder Story-RAM aus einem beendeten
  Lauf niemals in die nächste Episode durchsickern.

### V16 Reward-Vertrag

Normale Bewegung, einzelne Tiles und Tür-/Warp-Wechsel geben keinen Reward.
Alle positiven Ereignisse sind pro Episode oder projektweit einmalig geschützt.

| Ereignis | Reward |
| --- | ---: |
| normaler Schritt / bekanntes Tile | 0 |
| deutlich neuer Intro-/Dialogbildschirm | +2, insgesamt höchstens +20 |
| Intro abgeschlossen | +100 |
| Treppe erreicht | +150 |
| Haus bestätigt verlassen | +300 |
| bekannte Map erstmals in dieser Episode | +25 |
| projektweit wirklich neue Map | +500 |
| Warp / Tür / bekannte Transition | 0 |
| neue Weltstufe in dieser Episode | +1000 je Stufe |
| Schiggi gewählt | +1000 |
| Bisasam oder Glumanda gewählt | -500 und sofortiges Episodenende |
| Labor mit Schiggi verlassen | +500 |
| Gegner verliert neue HP | +0,5 je HP |
| Gegner K.O. | +30 |
| Kampf gewonnen / Erfahrung erhalten | +50 |
| Levelaufstieg | +25 je Level |
| eigene HP verloren | -0,1 je HP |
| Flucht aus einem begonnenen Kampf | -25 |
| komplette Party besiegt | -100 und Episodenende |
| Orden | +2500 |
| Heilung / Pokémon-Center | 0 |

Screenshot-Unterschiede geben außerhalb des Intros bewusst keinen Reward:
Menüs, Kampfanimationen, NPCs und Bildschirmeffekte wären leicht farmbar. Maps,
Positionen, Gegner-HP, Party und Story werden stattdessen aus bestätigten
RAM-Daten gelesen. Begegnungen sollen Gegner, eigenes Pokémon, gewählte Attacke,
Schaden, PP und Ergebnis als Telemetrie erfassen; proportionaler echter
HP-Schaden lehrt die Policy wirksame Attacken, ohne eine Kampftabelle
hartzukodieren.

Die Paket-Story wird nicht aus einem einzelnen RAM-Wert abgeleitet. „Paket
erhalten“ und „Paket abgegeben“ brauchen Schiggi, die jeweils richtige Karte,
die richtige Reihenfolge und drei aufeinanderfolgende bestätigende RAM-Lesungen.
Route 2, Wald, Marmoria und Orden dürfen die Weltstufe erst nach dieser
bestätigten Paketkette erhöhen. So kann ein kurzzeitig falsch gelesener Wert den
Webstatus und die Champion-Bewertung nicht mehr vorspulen.

### Brain-Pflege und Generationen

Alle Clients besitzen innerhalb eines Trainingsblocks dieselbe Policy. Ein
einzelner Agent besitzt daher kein eigenes Brain, das kopiert werden könnte.
Aus den synchron gesammelten Rollouts erzeugt PPO gemeinsam einen Candidate.

1. Alle 50 Clients sammeln mit derselben Ausgangspolicy Rollouts.
2. PPO aktualisiert daraus den Candidate in synchronen 25.600-Sample-Schritten.
3. Nach einem festen Trainingsblock wird der Candidate eingefroren.
4. Der Candidate wird in vollständigen Episoden vom Spielanfang ohne Lernen
   bewertet.
5. Vergleichsreihenfolge: Orden, Weltstufe, bestätigte Storykette, Schiggi plus
   Laborausgang, Maps, Reproduzierbarkeit und erst danach Reward/Tempo.
6. Nur ein nachweislich besserer Candidate wird neuer Champion und gemeinsame
   Basis der nächsten Generation. Alte und neue Messwerte dürfen niemals zu
   einer künstlichen Champion-Metrik vermischt werden.
7. Eine neue Tiefe darf das Intro nicht verdecken. Bei einem etablierten
   Champion muss der Candidate mindestens 85–90 % Intro-Retention halten.
8. Ein nicht besserer, aber stabiler Candidate darf begrenzt weiterlernen; bei
   klarer Regression wird wieder vom unveränderten Champion begonnen.

PPO kann Vergessen nicht mathematisch ausschließen. V16 verhindert das praktisch
durch wiederkehrende, gedeckelte Intro-Rewards, lange zusammenhängende Rollouts,
Full-from-Beginning-Episoden und eine unabhängige Retention-Prüfung vor jeder
Champion-Beförderung. Der Watcher zeigt ausschließlich den bestätigten
vollständigen Champion und verwendet keinerlei Skill-Snapshot.

### Vorheriger Stand

**V15.3 — All-Full, radikal vereinfachter Reward, kein Champion-Gate**
(2026-09-04). Aktueller Arbeitsstand und offene Punkte stehen in
`docs/AI_STATUS.md` (zuerst lesen). Laufende Zahlen aus
`python tools/pkmai_status.py`.

V15.3 ist eine bewusste Kurskorrektur weg vom Spezialisten-Curriculum
(intro/stairs/exit/starter-Bootcamp) hin zu einem einfacheren, PWhiddy-
artigen Modell: **fast die gesamte Flotte spielt jede Episode ab Spielanfang**
(inkl. Intro), der Reward ist auf Meilensteine + eine einzige un-farmbare
Erkundungsspur eingedampft, und der Champion wird nicht mehr durch einen
Schutzwall vor Verbesserung abgeschirmt.

### Lernlogik

- Eine gemeinsame PPO-Policy lernt alles; es gibt keine getrennten
  Spezialisten-Gehirne. `FULL_ONLY_MODE` (in `pokemon_env.py`) lässt **alle
  32 Trainings-Envs** die Rolle `full` spielen — jede Episode startet am
  Spielanfang, spielt Intro → Treppe → Haus → Labor → Starter → Welt, bis
  Blackout, falscher Starter oder ein Step-Cap.
- Die Observation enthält vier aufeinanderfolgende 64×64-Bilder sowie 31
  RAM-/Navigations-/Storywerte.
- 32 unsichtbare Trainingsumgebungen × 128 zusammenhängende Schritte ergeben
  4096 Samples pro PPO-Update. Nur der Watcher wird gerendert.
- Jede Agenten-Aktion hält die Taste **12 Emulator-Frames** und lässt sie
  dann 6 Frames los (`ACTION_HOLD_FRAMES` / `ACTION_RELEASE_FRAMES`,
  identisch in `pokemon_env.py` und `watch.py`). Kürzere Drücke drehen die
  Spielfigur in FireRed nur, statt eine Kachel zu laufen.
- **Reward, deutlich eingedampft:** Meilenstein-Einmalboni (Intro +100,
  Treppe +150, Haus +500, **Starter (Schiggy) +1000**, falscher Starter
  (Bisasam/Glumanda) −500 + Episodenende, Orden +500), ein pro Lauf
  eskalierender Tiefen-Bonus für jede neu erreichte Weltstufe
  (`NEW_GLOBAL_DEPTH_REWARD × Stufe`, gilt für **jeden** Agenten, nicht nur
  den Flotten-Ersten) und **eine einzige** Erkundungs-Spur: eine wirklich neue,
  global nie zuvor gelaufene Kachel-Kante zahlt `+0.10`. Alles Farmbare ist
  raus: die alten Korridor-/Nord-Richtungsreward, Routen-Imitation,
  Intro-Novelty-Screens und das rollenabhängige Step-Cost-Tuning sind
  deaktiviert. Eine winzige einheitliche Zeitgebühr (`-0.002`/Step) bleibt.
- Kein Champion-Schutzwall mehr: ein neuer Champion wird veröffentlicht,
  sobald ein Kandidat mit ≥4 abgeschlossenen Full-Läufen den aktuellen
  Champion-Score erreicht oder schlägt (`_score` = Orden, Weltstufe, Level,
  Maps, Tempo, Starter-Rate — reine Tiefe statt fragiler Endpositions-Raten).
- Der Watcher läuft im **Brain-Modus**: er lädt immer das aktuell trainierte
  Netz (`pokemon_model_resume.zip`) end-to-end, ohne Skill-Umschaltung, und
  zeigt Learner-Steps + Nachlade-Zähler live an — kein eingefrorener
  Skill-Snapshot mehr.

### Echte Feuerrot-Fortschrittskette

`world_stage` ist die einzige Fortschritts-Wahrheit:

0. Spielanfang/innen
1. Alabastia `(3,0)`
2. Route 1 `(3,19)`
3. Vertania City `(3,1)`
4. Eichs Paket im Vertania-Markt `(5,3)` erhalten
5. Paket bei Eich im Labor `(4,3)` abgegeben / Pokédex erhalten
6. Route 2 `(3,20)`
7. Vertania-Wald `(1,0)`
8. Marmoria City `(3,2)`
9. erster Orden

Die Paket-Stufen kommen aus den echten FireRed-SaveBlock-Variablen. Sie sind
nicht aus englischem Bildschirmtext abgeleitet und funktionieren daher auch
mit der deutschen `BPRD`-ROM. Jeder Checkpoint enthält die aktuelle Map und die
relevanten Storywerte als Sidecar; unpassende oder alte States werden ignoriert.

### Rollen bei 32 Umgebungen (V15.3)

Alle 32 Envs spielen `full` ab Spielanfang — keine Rollen-Aufteilung mehr.
`_agent_role()` ist auf `FULL_ONLY_MODE` kurzgeschlossen; die alte
sequentielle Bootcamp-Logik (intro → stairs → exit → starter →
free-world-Phasen, siehe Git-Historie) bleibt im Code als deaktivierter
Fallback erhalten, ist aber nicht aktiv. Ein Rest-Reservat für Deep-Resume-
Agenten (früher 2 Slots, resumten aus einem gespeicherten `stage_N`-Savestate)
wurde entfernt, weil der einzige verfügbare Checkpoint (Eichs Labor, ein
kleiner Innenraum) ein Datenrest der alten Reward-Ära war und praktisch
nichts beitrug.

Die Stage-, Checkpoint- und Rollenlogik wird durch Unit-Tests abgesichert.

---

Die vorherige Architektur war **V10.25 — Skill Vault + Full Chain**
(Spezialisten-Bootcamp je Story-Abschnitt, Champion-Schutzwall). Details und
warum V15.3 davon abgerückt ist: `docs/AI_STATUS.md`.

## Architecture

- Stable-Baselines3 PPO with `MultiInputPolicy`
- 32 parallel headless Stable-Retro environments
- four stacked 64×64 grayscale observations
- 31 RAM/navigation/story features
- 7 actions: A, B, START, UP, DOWN, LEFT, RIGHT
- identical action timing in training and watcher: 12 held frames + 6 release frames
- shared curriculum checkpoints and confirmed story transitions
- persistent exploration/navigation memory
- unprotected champion (best-score-wins, no regression shield) + live "brain mode" watcher
- browser dashboard with a clickable per-agent stats panel (Status and Watcher tabs)
- eigener sichtbarer Mapper (ID 121) mit einem Karten-Schritt pro Sekunde,
  separatem PPO-Modell und aus Screenshots zusammengesetzten 16×16-Tiles

### Mapper und Laufwege

`src/mapper.py` startet am tiefsten validierten Fortschritts-Checkpoint, lernt
aber in `pokemon_mapper_latest.zip` getrennt vom Haupt-Learner und Champion.
Er startet bevorzugt aus dem vorhandenen `battle_ready`-Savestate (Starter,
gesund, draußen im Gras). Der Emulator läuft dabei kontinuierlich mit 60 echten
Frames pro Sekunde. In der Welt wählt ein persistenter Frontier-Graph alle 1,5
Sekunden systematisch eine noch ungeprüfte Richtung und findet über bekannte
Kanten den kürzesten Rückweg zum nächsten offenen Feld. Das PPO-Modell wird nur
noch als Fallback in Kämpfen und Menüs verwendet. Nach jeder Aktion folgen 750
ms neutrale Beruhigungsframes; anschließend werden drei zeitlich versetzte,
ruhige Bilder desselben RAM-Standpunkts ausgewertet.
Jedes neue Feld der laufenden Episode gibt Reward;
ein projektweit erstmals kartiertes Feld, neue sichtbare Kartentiles und neue
Maps geben Extra-Reward. Dadurch bleibt der Übungsgradient erhalten, während
echte Entdeckung deutlich wertvoller ist. Story-, Kampf-, Level- und sonstige
Rewards des Haupttrainings werden vollständig verworfen. Sein 60-FPS-HD-Fenster zeigt zwischen den Entscheidungen ein ruhiges
Bild; exakt einmal pro Sekunde wird gehandelt und danach ein Mapping-Screenshot
ausgewertet. Während eines Kampfes werden weder Karten-Tiles noch Web-JPEGs
gespeichert; im Dashboard bleibt das letzte saubere Weltbild stehen. Die erzeugten Einzelkarten und der Atlas liegen unter
`runtime/mapper/` und bleiben über Neustarts erhalten.

Im Web ist `Alex (Watcher)` immer der erste Agent und sein Laufweg wird
standardmäßig angezeigt. Zusätzlich erscheinen nur Full-Journey-Agenten, deren
Episode wirklich mit `beginning` begonnen hat; Savestate-/Curriculum-Wege und
deren historische Navigationskanten sind standardmäßig aus. Linien werden ausschließlich zwischen zwei
benachbarten Koordinaten derselben Map gezogen. Savestate-Sprünge, Teleports und
Warps erscheinen deshalb nicht als Weg. Die Wege der übrigen Runner sind aus
Performance- und Lesbarkeitsgründen zunächst aus und lassen sich über
`👣 Weitere Wege` einblenden. Der Reiter `🧩 Mapper` zeigt Live-HD-Bild und den
aus echten Screenshots zusammengesetzten Atlas.

Die einzelnen Screenshot-Karten des Mappers werden außerdem alle zwei Sekunden
direkt als Bild-Overlays in der normalen `🗺️ Overworld Map` aktualisiert, sobald
ihre Kameraausrichtung durch einen echten Scrollschritt kalibriert wurde. Bis
dahin zeigt die Overworld eine exakte, animationsunabhängige RAM-Tilemap der vom
Mapper wirklich betretenen Felder. So können Wasser, Blumen, NPCs oder ein
geratener Kameraursprung den Watcher-Marker nicht mehr in Bäume verschieben.

Der Mapper besitzt eine harte Schreibgrenze: Curriculum-Savestates,
Story-Warps, Exit-/Journey-Routen und globale Weltstufe des Haupttrainings sind
für ihn schreibgeschützt. Sein Navigationsgedächtnis, seine Statistik, sein
PPO-Modell und sämtliche Bildkarten liegen ausschließlich unter
`runtime/mapper/` beziehungsweise im separaten Mapper-Checkpoint. Er liest die
validierten Trainings-Savestates nur für seinen Start und veröffentlicht außer
seiner Web-Telemetrie nichts in die Daten des Haupt-Learners.

Current PPO settings in V17.2 (`src/train.py`):

| Setting | Value |
| --- | ---: |
| Learning rate | `7.5e-05` |
| Environments | `60` |
| Steps per environment | `512` |
| Rollout size | `30720` |
| Batch size | `256` |
| Epochs | `4` |
| Gamma | `0.995` |
| GAE lambda | `0.98` |
| Entropy coefficient | `0.02` |

> Ältere Abschnitte unten (Mapper-Beschreibung, „Adaptive agent roles" mit 120
> Envs) stammen aus früheren Architekturversionen und sind teils veraltet —
> maßgeblich sind immer der aktuelle Code und die Angaben oben.

## Adaptive agent roles

The 120 environments change distribution automatically as the shared curriculum and Full Champion advance.

| Role | Starter breakthrough | Chain repair | Forest push |
| --- | ---: | ---: | ---: |
| Intro | 4 | 4 | 4 |
| Stairs | 12 | 20 | 10 |
| Exit | 20 | 20 | 12 |
| Starter | 52 | 18 | 12 |
| Battle | 0 | 4 | 8 |
| Level | 0 | 2 | 4 |
| Progress | 8 | 16 | 34 |
| Badge | 0 | 0 | 4 |
| Full journey | 24 | 36 | 32 |

All roles train the same Full-policy observation context. Their role only changes curriculum start, episode horizon and reward focus.

The current migration normally enters **Chain repair** when a shared Starter state exists but the protected Full Champion has not yet reached the Starter from the beginning. Once that happens, training switches automatically to **Forest push**.

## Checkpoints

Generated checkpoints live below `runtime/checkpoints/` and are intentionally ignored by Git.

| File | Purpose |
| --- | --- |
| `pokemon_model_resume.zip` | current learner, optimizer state and PPO step counter |
| `pokemon_model_best.zip` | protected end-to-end Full Champion |
| `pokemon_model_candidate.zip` | latest evaluated/final candidate |
| `pokemon_skill_intro_best.zip` | protected Intro policy |
| `pokemon_skill_stairs_best.zip` | protected stair policy |
| `pokemon_skill_exit_best.zip` | protected house-exit policy |
| `pokemon_skill_starter_best.zip` | protected Starter policy |
| `pokemon_skill_progress_best.zip` | protected post-Starter progress policy |

**V15.3 update:** the Skill Vault files above are historical — they were
written by the old specialist bootcamp and are no longer read by the watcher.
`WATCHER_BRAIN_MODE = True` in `watch.py` makes the watcher always load
`pokemon_model_resume.zip` (the live, continuously-training network) end to
end, with no per-stage skill switching. This is the honest "what you see is
what's training" view; see `docs/AI_STATUS.md` for why the switch was made.

## Repository layout

```text
src/        application, environment, training, watcher and web code
scripts/    start/stop utilities
tools/      development, RAM and map utilities
assets/     distributable static assets
docs/       architecture and AI handoff documentation
runtime/    generated models, telemetry, maps and curriculum (gitignored)
local/      private integration and game files (gitignored)
```

## Setup

Create a Python environment and install the project dependencies. Then create the local configuration:

```bash
cp .env.example .env
```

If ngrok is already configured globally, no token is required in `.env`.

Start all project processes:

```bash
./start_all.sh
```

Stop all project processes:

```bash
./stop_all.sh
```

During source-only or documentation changes, the trainer, watcher, webserver and ngrok do not need to be stopped. Changes to `train.py`, `pokemon_env.py` or `watch.py` require a controlled process restart before they become active.

## Runtime status

Useful local status files:

```text
runtime/trainer_status.json
runtime/champion_score.json
runtime/skill_vault_scores.json
runtime/instances_data/
```

Quick inspection:

```bash
cat runtime/trainer_status.json
echo
cat runtime/champion_score.json
echo
cat runtime/skill_vault_scores.json
```

## Development safety

- Never commit ROMs, private Stable-Retro integrations, `.env`, ngrok credentials, model checkpoints, runtime data or savestates.
- Do not reset `pokemon_model_resume.zip` merely because the Full Champion is older. The learner and champion intentionally advance on separate tracks.
- Do not reintroduce automatic hard rollback. It previously erased later-stage learning repeatedly.
- Keep watcher and training observation construction and action timing identical.
- Count only completed Full-from-beginning episodes for same-depth champion evaluation.
- Long Full probes must remain exempt from early house/stage caps; otherwise they terminate near 1,800 steps instead of their 32,768-step horizon.
- Avoid hard-coded map coordinates. Navigation should use RAM positions, discovered edges and confirmed transitions.

**Current work log: [docs/AI_STATUS.md](docs/AI_STATUS.md)** — read this first. It tracks
what changed, why, what is running, and what the next step is. Update it every session.

See [docs/AI_HANDOFF.md](docs/AI_HANDOFF.md) for the deeper technical background, invariants and tests.

Live runtime numbers: `python tools/pkmai_status.py`.

## Legal

No Pokémon ROM or proprietary game assets are included in this repository. Users must provide their own legally obtained game data and local Stable-Retro integration.

## Security

Never commit `.env`, ROMs, model checkpoints, save states, runtime data, backups or ngrok credentials. Review staged files with `git diff --cached` before every push.
