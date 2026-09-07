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

## 0. MOVEMENT-REDESIGN + FRONTIER-KONSOLIDIERUNG  ← NAECHSTE grosse Session (geplant 2026-09-07 nachts)

**Braucht:** einen Brain-Reset (Action-Space ändert sich). Savestates BEHALTEN
(`curriculum_shared/*` + `curriculum_states/agent_*/`), wie beim Reset vom
2026-09-07 00:15. Script-Vorlage: `scratchpad/brain_reset_keep_savestates.sh`
bzw. `tools/v20_reset.sh` (Letzteres löscht Savestates — anpassen).

### Diagnose (Stand 2026-09-07 ~02:00, frisches Netz seit 00:15)

- `known_transitions`: 1→2 **KNOWN**, **2→3 (Route 1→Vertania) NICHT KNOWN**
  (7 obs, Top-Variante 1), 3→4 KNOWN.
- Curriculum: `discovered_stage 5`, `mastered_stage 2`, `bottleneck 2`.
- Übergang 2: **281 Versuche, 1 Erfolg, 0 % Fensterrate** — DAS ist die Wand.
- Die 12 FRONTIER-Agenten starten alle bei `stage_frontier_4` (Route 2) und
  erkunden Gebiet, das das frische Netz **nie erarbeitet hat** — die tiefen
  Savestates (`stage_3/4`, `stage_frontier_3/4`) stammen aus der Zeit VOR dem
  Reset und wurden behalten. FRONTIER „entdeckt" also geerbt, nicht verdient.
- FULL/BRIDGE hängen zwischen Alabastia und Route 1, mit nur einem verrauschten
  1-Beobachtungs-Ziel Richtung Vertania.
- Rückkopplung, die es festhält: `frontier_stage() = max(mastered, min(discovered,
  mastered+2))` = 4. FRONTIER meldet am Episodenende `record_discovery(4)` →
  `discovered` bleibt ≥4 → FRONTIER startet wieder bei 4. Reines Datei-Re-Baseline
  von `discovered_stage` hält NICHT.

### Änderungen (in einem Rutsch, ein Reset)

**1. Movement: variable Schrittweite, Policy wählt sie.**
   - Action-Space `7 → 15`: `A / B / START` + `↑ ↓ ← →` × Längen `{1, 2, 4}`.
     (Nicht `{1,2,3,4}` = 19 — die vielen fast-Duplikate bremsen das Lernen;
     `{1,2,4}` deckt fein / mittel / Sprint ab.)
   - `step()`-Umbau: ein N-Kachel-Zug läuft als **N interne 1-Kachel-Schritte**
     (Bewegung + Positionslesung + Tile-/Edge-/Blue-Line-Reward **pro Kachel**),
     dann EIN return. Sonst zählt nur der Endpunkt → `seen_coords`,
     `manhattan == 1`-Edge-Block, ShortCycleGuard, `route_approach` alle kaputt.
   - `A / B / START` bleiben 1× kurz (`ACTION_HOLD_FRAMES = 9`), sonst Menüs/Kämpfe
     brechen.
   - `src/watch.py` `ACTION_HOLD_FRAMES`/`ACTION_RELEASE_FRAMES` (Zeile 44-45)
     mitziehen / als Legacy markieren (Watcher fährt eh `env.step`).
   - **Warum:** ein Policy-Step = 1 Kachel → Labyrinthe mit Random-Walk fast
     unmöglich zu queren, fixe Schrittweite oszilliert weiter. Variabel +
     policy-gewählt: „offener Korridor ↑×4, Abzweig ↑×1".

**2. FRONTIER an den Bottleneck binden (kein Racing auf mastered+2).**
   - `frontier_stage` (bzw. der FRONTIER-Zweig in `_v20_choose_episode_start`):
     höchste Stufe N, für die Übergang N-1 die **Reproduktions-Schwelle** packt
     (Fenster-Erfolgsrate ≥ 0.8; die 5 Full-Chain-Bestätigungen NICHT verlangen —
     die kommen nur aus FULL-ab-Anfang, kann FRONTIER nicht liefern).
   - Effekt: solange 2→3 bei 0 % ist, arbeiten ALLE 12 FRONTIER auf Stufe 2
     (`stage_2` / `stage_frontier_2`) und pushen in den Wald. Zusammen mit den
     ~8 BRIDGE → **~20 Agenten auf der Wand statt ~8** (≈ 2,5× Lernsignal).
   - FRONTIER + BRIDGE rücken dann GEMEINSAM eine Stufe vor.

**3. `stage_frontier_N` nur speichern, wenn Übergang N-1 die Schwelle packt.**
   - Guard in `_save_stage_checkpoint(kind="frontier")` bzw. am Aufruf
     (`pokemon_env.py` ~5837): `if transition(stage-1).window_rate < 0.8: skip`.
   - Verhindert „Vorpreschen" durch einen Glückslauf — kein Checkpoint in der
     neuen Map, bevor der Weg dahin solide ist.
   - Feinheit: FRONTIER zählt aktuell nur ERFOLGE in die Übergangsstatistik
     (`_v20_record_episode_outcome`, FRONTIER-Zweig) — Fehlversuche zählt es
     nicht. Umlenken bringt also v. a. positives Signal. Ggf. auch Fehlversuche
     zählen lassen (FRONTIER startet Stufe N, `reached == N` → `record_transition_
     attempt(N, success=False)`), damit die Rate ehrlich ist.

**4. `discovered_stage` 5 → 3 re-baseline** (in `runtime/curriculum_v20/state.json`,
   Datei-Op) — damit das Curriculum die reale Reichweite des frischen Netzes zeigt.
   Mit Änderung 2/3 hält es diesmal (FRONTIER inflatiert es nicht mehr von Stufe 4).
   Tiefe Savestates bleiben liegen, ungenutzt bis verdient.

**5. (optional) `FRONTIER_PROGRESS_REWARD` 0.15 → 0.25** — der nicht-farmbare
   „geh tiefer"-Anreiz (strikt High-Watermark auf BFS-Graph-Tiefe). Wird erst
   nützlich, wenn FRONTIER durch Änderung 2 auf der richtigen Stufe steht.
   Gilt NUR für FRONTIER (`training_mode == "FRONTIER"`, [pokemon_env.py:5946]),
   nicht FULL/BRIDGE/Watcher.

**NICHT gemacht (bewusst):** Edge-Reward wieder an. Diskutiert, aber: Farm-Risiko
(A↔B / A→B→C→A-Loops), das das Projekt schon 2× gebissen hat. Erst 1-5 ausreizen.
Falls FULL/BRIDGE auf Route 1 dann noch zu wenig rumlaufen: `FULL_FRONTIER_TILE_
REWARD` 0.3 → 0.5 (trifft alle Navi-Rollen auf unbewiesenen Stufen) ODER
gecapptes Edge-Reward (~0.02 + harter Cap 40/Map, Cap MUSS auch auf unbewiesenen
Stufen aktiv bleiben).

### Ausgangslage für die Session (Stand 2026-09-07 nachts)

7 **lokale, ungepushte** Commits aus der Nacht (Details: `NIGHTFEHLER_KORREKTUR.md`):
`cd2b1b1` route_approach · `2a174ae` Split+Level · `33cd17d` Watcher-Revert ·
`17371ad` 46-Env-Fleet · `2234075` Champion-Gate 32→8 · `b0bc2ab` Dashboard-Extremwerte ·
`5264232` FIGHTER-Uncut. Vor der Session: `git diff origin/main` prüfen, pushen.

Fleet aktuell: 46 Envs — FULL 18 / BRIDGE 8 / FRONTIER 12 / RETENTION 4 / FIGHTER 4.
`NUM_ENVS = 46` in `src/train.py`. Movement aktuell: `ACTION_HOLD_FRAMES = 9` +
`ACTION_RELEASE_FRAMES = 5`, 7 Aktionen, Discrete.

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
