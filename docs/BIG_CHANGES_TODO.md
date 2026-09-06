# PKMAI — BIG CHANGES TODO

Größere Umbauten, die eine eigene fokussierte Session + sauberen Neustart
brauchen (nicht mal eben zwischendurch). Kleinteiliges Reward-/Doku-Zeug
steht weiterhin in `docs/STATUS_TODO.md`.

---

## 1. FighterBrain — eigenes Kampf-Hirn neben dem ChampionBrain

**Stand:** 2026-09-06 besprochen, noch nicht gebaut. Zuerst wird beobachtet,
wie die V18-Reward-Änderungen wirken.

**Ziel:** Eine zweite, dauerhaft eigenständige PPO-Policy, die nur für Kämpfe
optimiert wird — unabhängig vom Fortschritts-Hirn und **nicht** von einem
Full-Reset betroffen.

### Architektur
- Eigene Modell-Dateien (`fighter_model_latest.zip`, `fighter_model_champion.zip`,
  `fighter_champion_score.json`, eigene `fighter_model_version.json`), abgelegt
  unter `runtime/fighter/` — dieser Ordner wird von `tools/v11_reset.sh`
  **explizit ausgenommen**.
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
