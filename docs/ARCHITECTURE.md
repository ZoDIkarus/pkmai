# PKMAI Architecture

Two training stacks live in the tree; the **2×2 stack is what runs today**.
Authoritative current behaviour: [`CURRENT_LOGIC.md`](CURRENT_LOGIC.md). Full 2×2
design: [`BATTLE_ARCHITECTURE.md`](BATTLE_ARCHITECTURE.md).

## 2×2 Navigation/Battle stack (`src/twoby2/` + `battle_*`) — LIVE

Cut over 2026-09-07 (`scripts/start_2x2_visible.sh`, `PKMAI_TWOBY2_LIVE=1`).
Two permanently separate PPO systems, coupled only by
`runtime/model_manifest.json`:

- **Navigation** — 1 learner + 1 champion, **40 workers, all FULL agents from the
  canonical master start `StartGame.state`** (no BRIDGE/FRONTIER/RETENTION/FIGHTER
  split — `_v20_mode()` forces `MODE_FULL` while the `nav_battle_wrapper` gate is
  on). Overworld observation (stacked `4×64×64` image + 31-value nav vector).
  `PPO_N_STEPS = 512`, independent of the episode horizon.
- **Battle** — 1 learner + 1 champion, 9 workers (8 headless + 1 frame-mirror),
  isolated `BattleEnv` scenario emulators. Compact RAM observation with
  unknown-masks, macro action space (`MOVE_1..4` / `SWITCH_*` / `RUN`),
  combat-only anti-farming reward, `BATTLE_PPO_N_STEPS = 256`. Battle champion is
  currently the verified rule controller (`battle_champion.rule.json`) — no PPO
  battle champion promoted yet.

Out of battle the navigation brain acts and only navigation transitions enter
its rollout. In battle `NavigationBattleWrapper` runs the fight via the real
`EmulatorBattleDriver`, stepping the raw emulator directly — `PokemonFireRedEnv.step`
and its combat-reward pipeline do not run, so the navigation PPO never sees
damage/KO/win/level. Battle steps never touch navigation counters. Hard isolation
is enforced by `tests/test_twoby2_ppo_isolation.py` (no shared param tensor /
optimizer). Rollback: unset `PKMAI_TWOBY2_LIVE`, restore from
`brain_backups/pre_2x2_activation_20260907_200507/`.

## Legacy single-PPO stack (`train.py` without `PKMAI_TWOBY2_LIVE`) — rollback target, not running

One shared PPO `MultiInputPolicy` over `NUM_ENVS = 46` Stable-Retro environments.
Roles (`FULL` / `BRIDGE` / `FRONTIER` / `RETENTION` / `FIGHTER`) are starting
conditions and reward rules on the **one** network, not separate policies. Same
`Dict` observation as the navigation stack. Intact but inactive.

## Curriculum

Runtime curriculum data is stored outside Git in `runtime/`. In the legacy
fallback stack, specialists learn short early-game skills while Progress/Frontier
agents resume from self-discovered checkpoints and Full-chain agents validate
end-to-end. The live 2×2 navigation stack drops the role split — every worker is
a full run from the canonical start (PWhiddy principle).

## Mapping

Training agents contribute coordinate / edge / warp metadata only.
The Watcher performs the visual live-map tiling.

## Secrets

Secrets are local-only. `.env` is ignored by Git.
ngrok may alternatively use its standard user-level config outside the repo.
