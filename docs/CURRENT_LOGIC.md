# Current training logic — 2026-09-06

This is the consolidated description of the active source defaults, replacing conflicting historical release notes. Runtime processes load Python changes at their next start. During this maintenance the trainer remains stopped and the watcher remains running.

## Roles and actual starts

All 60 environments train the same PPO policy. Roles are starting conditions and reward rules, not separate networks. IDs below are zero-based, as in Status (`A59`); names number agents 001–060.

| Mode | IDs | Start and purpose |
|---|---|---|
| FULL | 0–20 | Original `StartGame`: learn the complete route from the master save. |
| BRIDGE | 21–40 | Immutable `stage_<bottleneck>` entry: repeat the earliest discovered transition not reliably mastered. If no valid entry exists, start from master. |
| FRONTIER | 41–50 | Frontier/entry near the deepest discovered stage: extend the explored area and discover the next transition. Missing checkpoints fall back to a validated entry or master. |
| RETENTION | 51–55 | Rotate entry states for mastered transitions; until any are mastered, start from master. |
| FIGHTER | 56–59 | Healthy `stage_frontier_2`, then stage-2 entry, deepest available entry, or master: provide combat experience. No separate fighter network. |
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

`_battle_reward_scale` doubles enemy-damage/win/level/faint components for trainer battles in both cases. It does not double every penalty. On wild-training maps it reduces these components to ×0.3 after six recorded opponent faints for normal roles; Fighter alone skips that reduction. The counter is opponent faints, not necessarily six fully completed multi-opponent battles.

The existing post-wipe ×0.05 applies to damage (win and level are exempt) in both roles' scaling function. Fighter now ends its episode on a wipe, so it normally restarts healthy instead of spending its next battles in recovery.

Fighter receives **zero exploration, tile, route, stage, story, building, capture or recovery bonuses**. Its returned PPO reward, episode total, and reported reward events use the same exact numeric combat accumulator. Display strings are not parsed to reconstruct reward. Other roles retain their original total. No extra win/level multiplier or bigger base reward was introduced.

Shared policy means combat practice can benefit all roles, but removing navigation rewards from Fighter is not a guarantee of faster learning. The other 56 agents continue supplying navigation/story experience.

## Navigation and exploration rewards

These apply to non-Fighter roles; ordinary combat rewards above remain available to them too.

| Event | Actual rule |
|---|---|
| New episode stage | +250 per gained stage, above the reset baseline |
| Approach a known target | +0.05 per unit of new best distance; returning to an already achieved distance does not repay |
| Large backtrack | −0.005 beyond a 12-tile margin; the environment explicitly passes this margin |
| FULL/BRIDGE/RETENTION new tile | +0.02 on proven ground, +0.3 when forward navigation is unknown |
| FRONTIER new tile | +0.02; +1 for a fleet-first tile |
| Tile cap | After 20 tiles per outdoor map/episode, tile component ×0.1; separate interior cap; global-first component is separate |
| FRONTIER topological progress | +0.15 × improvement over anchored best frontier score, minimum improvement 0.5 |
| Newly confirmed forward transition | +40 if the crossing makes it known and objective is `scout` |
| Already known forward crossing | +25 once per episode for FULL/BRIDGE |
| Door/map back-and-forth | −0.10 |
| Short local cycle | −0.05, escalating to −0.25; persistent loops can truncate |

Town/route, story, badges and catching retain their existing conditional rewards (e.g. new route +50, city +300, badge +3000). They are not additive unconditional rewards on every visit. Scouts receive no tile reward below their starting stage, and map arrival guards suppress old-stage reward farming. `src/pokemon_env.py:step` contains the conditions; constants alone are not a complete reward specification.

FRONTIER score comes from walked graph depth plus unknown-neighbor openness minus revisits, not a hardcoded north direction. Unknown/unconnected positions do not invent a target or score. This also means a deep dead-end can look promising; the metric cannot guarantee the exit is found.

## Healthy and timely savestates

`src/checkpoint_health.py` defines a usable frontier team: all checksums valid, every Pokémon ≥80% HP, no status ailment, and at least one move with PP per Pokémon. Save and actual load both check this. A healthy candidate can replace a marked-unhealthy or legacy anchor without a new distance record. Otherwise it must improve the frontier score, or improve minimum team health by at least five percentage points at the same or better score.

Capture checks run on refreshed positions (every four agent actions), after three stable map readings, outside battle and wipe cooldown. There is no episode-end or thousands-of-steps wait. Entries stay immutable; only FRONTIER advances `stage_frontier_<n>`. The saved score is the **actual saved position's** value with fractional precision, not an old episode peak or truncated integer. Greater scores do not permit publishing an almost-dead team.

The local Route-1 anchor was repaired in an isolated emulator: Squirtle 31/31, Rattata 16/16, Pidgey 15/15, same Route 1 position (17,24), species, levels, PP and story. The previous files are backed up under `brain_backups/healthy_frontier_20260906_214032`. This is a local state repair, not a recurring heal during training, and is not a ROM/state payload shipped through Git.

FRONTIER/FIGHTER episodes end after a party wipe. FRONTIER additionally truncates after 120 trusted non-battle steps below its starting stage; returning to the start stage or farther resets the counter. This prevents Route-1 explorers spending thousands of steps in Pallet. The next reset selects the latest shared checkpoint. FIGHTER retains its 400-consecutive-non-battle-step leash. FULL/BRIDGE/RETENTION retain their recovery behavior.

Health gating can still delay a new frontier save after damage. This is intentional: faster overwriting with a dying team recreates the original failure. Newly discovered stage entries and later healthy frontier candidates provide subsequent starts.

## Episode length is not update frequency

- FULL/Watcher horizon: 32,768 **travel** actions. Scout-start roles: 12,000 travel actions. Battle actions are tracked separately.
- A single stuck battle is capped at 2,000 actions for everyone. Fighter is exempt from the total episode battle budget, not this single-battle cap.
- PPO: 60 environments × 512 actions = 30,720 samples per rollout/update; batch size 256, four epochs. Learning does not wait for an episode to finish.
- Resume publication: every 50,000 aggregate training steps. Champion checks: every 250,000 aggregate steps, subject to completed evaluation evidence. Longer episodes can delay completed-run evidence and champion promotion, not PPO updates.
- Restart chooses `pokemon_model_resume.zip`, then champion, then latest. Loaded policies keep their real counter. Current resume checked at 9,665,880 steps; all 36 policy tensors finite.

Trainer remains unrestricted on `TRAIN_DEVICE=auto` (MPS here). Input cadence remains 9 held + 5 released emulator frames. Only watcher display is paced at 59.7 FPS. No audio is required.

## Dashboard and verification

Status and Watcher have search by ID/name/map/start, role filters, health/battle/checkpoint filters, current team health, and reported start health. Actual `training_mode` takes priority, with canonical role in the agent name as a legacy fallback. Agent 59 is Fighter. UI changes are loaded through the existing dashboard asset route without restarting the web server. Reload the dashboard when convenient; do not restart the streaming watcher. While training is stopped, cards retain the last reported training observations and do not represent the newly repaired start state until those agents reset.

149 regression tests passed. Isolated real-emulator starts and 32 actions for each of the five roles passed; Fighter outdoor reward was zero, Fighter/Frontier starts had 62/62 HP. A further isolated 1,500-action Fighter run produced battle wins, damage and level-up rewards with the documented amounts and no navigation/story reward events. Browser Status search for 59 returned exactly one Fighter. These checks establish implementation behavior, not an overnight learning result. Next training evidence should be real Viridian arrivals, stage-3/4/5 safe anchors, combat win/wipe rates and repeatable transitions toward Viridian Forest.
