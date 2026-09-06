# PKMAI — BIG CHANGES TODO

Current implemented behavior: [CURRENT_LOGIC.md](CURRENT_LOGIC.md). Fighter now uses combat-only rewards on the shared network; a separate FighterBrain remains a future proposal.


Größere Umbauten, die eine eigene fokussierte Session + sauberen Neustart
brauchen (nicht mal eben zwischendurch). Kleinteiliges Reward-/Doku-Zeug
steht weiterhin in `docs/STATUS_TODO.md`.

> **2026-09-06 — erledigt: V20 CURRICULUM MODES.** Die „Architektur, die
> irgendwann das ganze Spiel lernen kann" ist gebaut: `FULL` / `BRIDGE` /
> `FRONTIER` / `RETENTION` auf einem PPO-Netz, `discovered_stage` vs
> `mastered_stage`, dynamischer `current_bottleneck`, generische
> `Objective`-Repräsentation für den Rest der Story. Details: `README.md` +
> `docs/STATUS_TODO.md`. Clean-Reset-Script: `tools/v20_reset.sh`
> (löst `tools/v11_reset.sh` ab).

---

## 1. FighterBrain — eigenes Kampf-Hirn neben dem ChampionBrain

**Stand:** 2026-09-06 — als **erster, leichter Schritt** wurde stattdessen eine
`FIGHTER`-Rolle auf dem GETEILTEN PPO-Netz gebaut (kein zweites Netz):
`curriculum_v20.MODE_FIGHTER`, 4 Ränge, resumen den FRONTIER-Route-1-Anker,
400-Step-Leash außerhalb Kampf, vom Wild-Decay + Post-Wipe-×0.05 ausgenommen.
Dazu `BATTLE_WIN_REWARD 0→10`, `LEVEL_GAIN_REWARD 15→10`, Billig-Flucht bei
HP ≤ 10 %. Siehe `README.md` + `docs/STATUS_TODO.md`. **Erst beobachten**, ob die
flottenweite Kampf-Sieg-Quote (aktuell ~33 %) damit steigt. Wenn nicht → der
volle FighterBrain unten (separates Netz, eigener Champion-Loop).

**Ziel:** Eine zweite, dauerhaft eigenständige PPO-Policy, die nur für Kämpfe
optimiert wird — unabhängig vom Fortschritts-Hirn und **nicht** von einem
Full-Reset betroffen.

### Architektur
- Eigene Modell-Dateien (`fighter_model_latest.zip`, `fighter_model_champion.zip`,
  `fighter_champion_score.json`, eigene `fighter_model_version.json`), abgelegt
  unter `runtime/fighter/` — dieser Ordner wird vom Reset-Script
  (`tools/v20_reset.sh`) **explizit ausgenommen**.
- Paralleler Promotion-Loop neben `ChampionManager` in `src/train.py`
  (eigener `FighterManager` o. Ä.).

### Reward-Modus „nur Kampf"
Für FighterBrain-Envs zählen ausschließlich:
- `enemy_damage`, `enemy_faint`, `battle_win` / EP-Anstieg im Kampf
- `team_level_up`
- `healed_partial` + Pokécenter-Heilung
- `party_wiped`-Strafe

Explizit **0**: `new_tile`, `new_map`/`replay_map_once`, `city`, `world_depth`/
`global_stage_record`, `new_warp`, `species_caught`/Pikachu, `starter_*`,
`pokecenter_enter`/`advance_heal`, `pokemart_*`. Orden: offen (ist das
Kampf-Ziel — vermutlich behalten).

### Upgrade-Kriterium (`fighter_champion_score.json`), in dieser Reihenfolge
1. mehr **abgeschlossene Kämpfe** pro Lauf → besser
2. bei Gleichstand: **weniger Kampf-Steps** → besser (schneller)
3. bei Gleichstand: höherer **Kampf-Reward-Total** → besser

Beispiele aus der Besprechung: 10 Kämpfe/1000 Steps → dann kommt einer mit
11 Kämpfen = upgrade. Oder 10 Kämpfe/800 Steps (schneller) = upgrade. Oder
10 Kämpfe/1000 Steps aber doppelter Reward = upgrade.

### Web (`src/web_stream.py`)
- Eigene Karte **„FIGHTER BRAIN"** neben „FRONTIER CHAMPION": Version + beste
  (Kämpfe / Kampf-Steps / Kampf-Reward).
- Anzeige-Reihenfolge: ChampionBrain → FighterBrain → **dann** erst der Learner.
- Als Upgrade-Metrik im Web „Kampf-Steps + Battles" zeigen.

### Offene Entscheidungen (vor dem Bau vom Nutzer holen)
1. Welche Agenten trainieren es? Fester Flotten-Anteil (z. B. 16/96 immer
   Rolle „battle") **oder** die bestehende dynamische `battle`-Rollenverteilung?
2. Startgewichte: frisch aus einem Skill **oder** Kopie des aktuellen Champions?
3. Eigene Savestate-Spawns mitten auf der Route (gesunde Party, sofort Kämpfe)
   **oder** komplette Läufe ab Alabastia?

---

## 2. „Haus nach dem Vertania-Wald" — Sonderbehandlung

Braucht zuerst die **Bank/Map-ID** dieses Hauses (aktuell unbekannt; ein Scout
muss es erreichen, oder im Watcher-Status ablesen, wenn ein Agent drinsteht).

Dann:
- Erstmals betreten pro Lauf: **+100** (wie eine neue Map)
- Allererster Fund fleet-weit: **+250 global einmalig** (wirklich nur der Erste)
- Innenraum-Kacheln dieses Hauses: **+5** pro neue Kachel/Lauf (statt des
  normalen Bank-Innenraumwerts), damit der Agent das Haus nicht als „schlechter
  als der Wald" wertet und zurückläuft.

Implementierung analog zu `POKECENTER_MAPS` / `POKEMART_MAPS` +
`INTERIOR_TILE_REWARD_BY_BANK`-Sonderfall.
