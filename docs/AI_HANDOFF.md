# PKMAI V10.25 AI handoff

This document is the technical continuation point for an AI or developer working on PKMAI. Read it together with `README.md` and inspect the current Git diff before changing source files.

## Current objective

The immediate target is a reliable end-to-end sequence:

1. finish Intro and name selection;
2. leave the upstairs starting room via the stairs;
3. leave the house;
4. obtain the Starter;
5. travel through Viridian City toward Viridian Forest;
6. later train battles, levels and the first badge.

At the V10.25 migration, the protected Full Champion was v000144 at 3,185,400 PPO steps. It had reached the house exit from the beginning but not the Starter. The learner had exceeded nine million aggregate PPO steps and had learned some Starter behavior through curriculum agents, but repeated hard rollbacks prevented consolidation.

Runtime values will continue changing and must be read from JSON rather than copied from this document.

## Source of truth

Primary files:

- `src/train.py`: PPO creation/resume, 120 subprocess environments, champion evaluation, Skill Vault and status files.
- `src/pokemon_env.py`: observation, actions, reward system, curriculum, role allocation, episode termination and agent telemetry.
- `src/watch.py`: visible emulator watcher, protected model routing, observation parity, mapping and watcher telemetry.
- `src/web_stream.py`: browser dashboard and stream aggregation. This file was modified locally before this handoff; inspect its Git diff before refactoring it.
- `src/firered_ram.py`: RAM location and party readers, if present in the repository.

Generated runtime state is not source and must remain Git-ignored.

## Non-negotiable compatibility invariants

### Observation

The policy uses a `Dict` observation:

- image: 64×64 grayscale;
- nav: 28 floating-point values.

The nav vector contains:

1. nine-value objective one-hot (currently all learning roles resolve to `full`);
2. gameplay-ready, battle, stairs-done, house-left and has-Starter flags;
3. normalized bank, map, x and y;
4. target-present, dx, dy and Manhattan distance;
5. level and badge count;
6. party size, maximum level, average level and average HP ratio.

Do not change its shape or ordering when loading existing checkpoints. `watch.py::build_v7_obs` must remain semantically identical to `pokemon_env.py::_build_nav_vector`.

### Actions

Action indices are fixed:

| Index | Action |
| ---: | --- |
| 0 | A |
| 1 | B |
| 2 | START |
| 3 | UP |
| 4 | DOWN |
| 5 | LEFT |
| 6 | RIGHT |

Both trainer and watcher must execute each decision as four held emulator frames followed by four neutral/release frames. A previous watcher implementation declared 4+4 but still used a hard-coded five-frame condition in its main loop; V10.25 removes that mismatch.

## Why V10.25 exists

The previous behavior had four important failure modes:

1. Automatic hard rollback restored v000144 repeatedly and erased later-stage learner improvements.
2. Full probes labelled as 32k were still caught by the global early-house cap and ended around 1,800 steps.
3. Watcher state differed from training: stair/house flags were inferred differently, party-based Starter detection was missing, and stage changes could use the wrong model temporarily.
4. Battle and level specialists did not resolve to the shared Full-policy objective one-hot even though startup output claimed policy unification.

V10.25 addresses all four without changing the observation or action space.

## Learner, Champion and Skill Vault

These are deliberately separate:

- The **learner** (`pokemon_model_resume.zip`) keeps training and retains optimizer state and global steps.
- The **Full Champion** (`pokemon_model_best.zip`) is protected from regression and is used for verified end-to-end progress.
- The **Skill Vault** stores best observed whole-policy snapshots for individual routed stages.

Skill files:

```text
pokemon_skill_intro_best.zip
pokemon_skill_stairs_best.zip
pokemon_skill_exit_best.zip
pokemon_skill_starter_best.zip
pokemon_skill_progress_best.zip
```

The Skill Vault is not weight surgery and does not merge layers. Each file is a full PPO policy captured when the corresponding role achieved a new local score. This avoids unsafe parameter splicing while allowing the watcher to preserve and execute a strong stage-specific routine.

On V10.25 installation:

- Resume learner and optimizer remain unchanged.
- Intro/Stairs/Exit vault files start from the protected Champion.
- Starter/Progress vault files start from the newer Resume learner.
- Later vault updates replace only the relevant best stage file.

## Adaptive 120-agent curriculum

### Phase 1 — Starter breakthrough

Used when no shared Starter state exists:

```text
4 intro | 12 stairs | 20 exit | 52 starter | 0 battle | 0 level
8 progress | 0 badge | 24 full
```

Eight Full-from-beginning slots are long probes.

### Phase 2 — Chain repair

Used when a shared Starter state exists but the Full Champion has no Starter:

```text
4 intro | 20 stairs | 20 exit | 18 starter | 4 battle | 2 level
16 progress | 0 badge | 36 full
```

Twenty-eight Full agents start from the beginning, eight use overlapping bridges, and sixteen beginning slots are long probes. This is the expected phase immediately after migration.

### Phase 3 — Forest push

Used once the protected Full Champion obtains the Starter from the beginning:

```text
4 intro | 10 stairs | 12 exit | 12 starter | 8 battle | 4 level
34 progress | 4 badge | 32 full
```

Twenty-four Full agents start from the beginning, eight use bridges, and twelve beginning slots are long probes.

The phase is refreshed when environments reset by reading `runtime/champion_score.json`.

## Episode horizons and evaluation

- Short specialists use dense, stage-focused episodes.
- Short Full probes retain stage caps for efficient failure recycling.
- Selected long Full-from-beginning probes are exempt from all early house and Full-stage caps.
- Long Full probes terminate at 32,768 agent decisions unless another legitimate termination occurs.
- Anti-loop termination remains active for truly stationary behavior.

Champion same-depth evaluation requires at least 24 completed episodes and eight completed Full-from-beginning runs. Active/incomplete Full snapshots must never be mixed into the denominator.

New Full-from-beginning stage depth is published immediately through the Frontier Champion logic. A Starter or badge milestone therefore does not need to wait for a full 32k episode to finish.

V10.25 detects regression but does not automatically restore weights. The existing Champion and Skill Vault remain untouched while the learner continues.

## Reward hierarchy

Important current rewards include:

| Event | Base reward |
| --- | ---: |
| Intro completed | +100 |
| Stairs reached | +150 |
| House exit confirmed | +500 |
| First Pokémon/Starter | +500 |
| New global episode-map depth | +300 |
| New outdoor map after start map | +150 |
| North-to-grass progress | +75 |
| Level gained | +25 per level |
| Badge gained | +500 per badge |

Specialist bonuses are additional. Navigation-target movement supplies dense symmetric shaping, while repeated edges, reversing the exit route, START spam and stationary loops receive penalties.

Do not reward the same persistent discovery every episode without a guard. It creates farmable reward loops.

## Watcher routing

The watcher selects a protected policy on every decision boundary:

```text
untrusted/non-gameplay -> intro
inside start room       -> stairs
after indoor room warp  -> exit
outside without Starter -> starter
Starter/battle/progress -> progress
```

Expected terminal messages include `SKILL-INTRO`, `SKILL-STAIRS`, `SKILL-EXIT`, `SKILL-STARTER` and `SKILL-PROGRESS`. The displayed PKMAI version may remain v000144 while stage skill files improve; the version identifies the Full Champion, not every vault update.

Watcher anti-loop reset at 1,800 room steps is a display/evaluation safeguard, not a training episode cap.

## Starter telemetry

Starter ownership is true when either:

- the primary RAM level is at least 5; or
- the decoded party contains a valid Pokémon with level at least 5 and positive maximum HP.

V10.25 uses both signals in training, baseline restore, watcher observation and telemetry. `has_starter` and `training_phase` are written to agent JSON. The Journey Starter milestone is also claimed from the party reader, fixing the dashboard showing 0% while a Pokémon was visible.

## Runtime diagnostics

Read these files before proposing another reset or learning-rate change:

```bash
cat runtime/trainer_status.json
echo
cat runtime/champion_score.json
echo
cat runtime/skill_vault_scores.json
```

Also inspect recent terminal lines for:

- `SKILL VAULT:` updates;
- `FULL-SLOT DONE` with slot, start, stage and episode steps;
- `FULL-MILESTONE` or `FRONTIER-CHAMPION`;
- watcher `SKILL-*` hot reloads;
- repeated timeouts or anti-loop resets.

Aggregate PPO steps are summed across all 120 environments. At roughly 920 aggregate steps/second, a 32,768-step episode per long slot spans about 3.93 million aggregate PPO steps, roughly 70 minutes. Story milestones may publish earlier.

## Safe validation after source changes

At minimum:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python \
  -m py_compile src/train.py src/pokemon_env.py src/watch.py src/web_stream.py
```

Verify important configuration:

```bash
grep -nE 'LEARNING_RATE|PPO_N_STEPS|PPO_ENT_COEF|V10\.25|ACTION_HOLD_FRAMES|ACTION_RELEASE_FRAMES' \
  src/train.py src/pokemon_env.py src/watch.py
```

Before committing:

```bash
git status --short
git diff --check
git diff --stat
```

Never use `git add .` until ignored runtime/private paths and all untracked files have been reviewed.

## Process control

Always tell the operator explicitly what must stop.

- Documentation or Git-only work: leave trainer, watcher, webserver and ngrok running.
- `train.py` or `pokemon_env.py` change: stop trainer cleanly with Ctrl+C and wait for the final Resume save.
- `watch.py` change: stop watcher with Ctrl+C.
- Dashboard-only change: normally leave trainer and watcher running; restart only the webserver if required.
- Patch installers may stop/restart only the processes stated in their output.

Do not launch `start_all.sh` when individual processes are already active; that can create duplicates.

## Recommended next work

1. Add browser-dashboard filters and sorting by role, map, party/Starter ownership, story stage and active skill.
2. Display learner versus Full Champion versus Skill Vault clearly so cumulative historical percentages are not mistaken for current-policy evaluation.
3. Add a recent-window chart alongside lifetime success rates.
4. Observe at least one long Full slot and confirm it exceeds 1,800 steps.
5. Confirm watcher transitions through `SKILL-STAIRS` and `SKILL-EXIT` before adjusting PPO again.
6. After the first Full Starter milestone, verify automatic transition to `forest_push`.

Do not make another broad hyperparameter or agent-allocation change until the V10.25 diagnostics above have been collected. Change one causal layer at a time.

## V10.25 source fingerprints

The released V10.25 payload was validated with these SHA-256 hashes:

```text
99fe1f36fbe70ef98cceaa6bb3fa98fc967d162c24fe8eeaaaccab22bcd1a1cb  src/train.py
260151623eaf27f118d47d8765ef5d4a697400e83d7b4b85104505ff66992248  src/pokemon_env.py
17ca8b6753b21156760a3f95e4033031481b38c0178987ca0607336f69b3950a  src/watch.py
```

If a file differs, inspect the diff rather than forcing an installer or overwriting it blindly.
