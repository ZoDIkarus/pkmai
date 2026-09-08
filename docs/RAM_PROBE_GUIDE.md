# Manuelle Battle-RAM-Probe — Anleitung

> **Abgeschlossen am 2026-09-07.** Der Datensatz
> `runtime/ram_probe/20260907_184350` enthält 20 Dumps aus einem Wild- und
> einem Trainerkampf. Alle Pflichtfelder wurden verifiziert:
> `gBattlerPartyIndexes=0x02023BCE`, `gBattleMons=0x02023BE4`,
> `gActionSelectionCursor=0x02023FF8`,
> `gMoveSelectionCursor=0x02023FFC`, `battle_menu_state=0x02022BC4`.
> Das optionale `gBattleWeather=0x02023F1C` wurde ebenfalls bestätigt.
> Diese Probe muss für dieselbe ROM nicht erneut durchgeführt werden.

Damit der 2×2-Battle-Pfad live gehen darf, müssen für die **deutsche FireRed
(BPRD)** fünf Adressen wirklich verifiziert sein:

`gBattleMons`, `gBattlerPartyIndexes`, `gActionSelectionCursor`,
`gMoveSelectionCursor`, `battle_menu_state`.

`gBattleWeather` ist **optional** (kein Aktivierungsblocker).

Das können nur **du** liefern, indem du kurz selbst spielst. Der laufende
Trainer/Watcher/Web wird dabei nicht angefasst (eigener Emulator).

---

## Ganz einfach: was du drückst und wo

Starte im Projektordner:

```bash
PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python tools/battle_dump_collect.py
```

Der Collector lädt automatisch eine eingefrorene Kopie des geeigneten
Route-1-Spielstands: Schiggy Level 11 mit vier Attacken sowie Rattfratz und
Taubsi. Das Original wird ausschließlich gelesen. Ein Fenster öffnet sich.
Laufen mit `w a s d`, `j` = A, `k` = B.

Neue Kämpfe und das Kampfende erkennt das Programm automatisch. Du darfst `e`
im normalen Ablauf **nicht** drücken; es ersetzt keine fehlende Kampferkennung.
Im Terminal erscheint beim Kampfbeginn `AUTO new encounter`. `W` ist nur ein
Notfallknopf für die Markierung außerhalb eines Kampfes.

`F`, `B`, `P`, `R` sind Großbuchstaben: auf der Tastatur also zusammen mit
Shift drücken. Diese Tasten speichern nur eine Messung; sie wählen im Spiel
nichts aus.

Die beiden Kampfmenüs sind **2×2**:

```
   Hauptmenü            Attacken-Menü
 ┌────────┬────────┐   ┌────────┬────────┐
 │ FIGHT  │  BAG   │   │ Move 1 │ Move 2 │
 ├────────┼────────┤   ├────────┼────────┤
 │POKEMON │  RUN   │   │ Move 3 │ Move 4 │
 └────────┴────────┘   └────────┴────────┘
```

### Teil A — Wildkampf

1. Ins hohe Gras laufen, bis ein Wildkampf startet. Warten, bis im Terminal
   `AUTO new encounter` und im Spiel das Hauptmenü erscheinen.
2. Du bist im **Hauptmenü**. `m` drücken.
3. Cursor steht oben links auf **FIGHT** → `Shift+F` drücken.
4. `d` (nach rechts) → Cursor auf **BAG** → `Shift+B` drücken.
5. `s` (nach unten) → Cursor auf **RUN** → `Shift+R` drücken.
6. `a` (nach links) → Cursor auf **POKEMON** → `Shift+P` drücken.
7. `w` (hoch) → zurück auf **FIGHT**, dann `j` drücken. Im Attacken-Menü
   anschließend `v` drücken.
8. Cursor steht oben links (Move 1) → `1` drücken.
9. `d` (nach rechts) → Move 2 → `2` drücken.
10. `s` (nach unten) → Move 4 → `4` drücken.
11. `a` (nach links) → Move 3 → `3` drücken.
12. `k` zurück, `s` auf POKEMON, `j` → du bist in der **Team-Liste**.
    `p` drücken, dann `SPACE`.
13. Ein anderes Pokémon einwechseln. Sofort danach die Slot-Taste drücken:
    `7` = 1. Pokémon, `8` = 2., `9` = 3., `0` = 4., `-` = 5., `=` = 6.
    Dann `m` und `SPACE`.
14. Jetzt sofort wieder auf Schiggi wechseln: im Hauptmenü `s`, `j`, Schiggi
    im ersten Team-Slot auswählen und den Wechsel bestätigen. Wenn Schiggi
    wieder aktiv und das Hauptmenü sichtbar ist: `7`, `m`, `SPACE`.
15. Den Wildkampf mit Schiggi beenden. Danach `l` drücken, um den unveränderten
    Probe-Spielstand neu zu laden. Bereits gespeicherte Dumps bleiben dabei
    erhalten und werden nicht überschrieben.

### Teil B — Trainerkampf

16. Zu einem noch nicht besiegten Trainer laufen. Den
    Trainerkampf starten und auf `AUTO new encounter` warten. Dann einmal `t`
    drücken; im HUD muss nun `trainer` stehen.
17. Hauptmenü → `m` → `Shift+F`, `d` `Shift+B`, `s` `Shift+R`,
    `a` `Shift+P`, `w`, `j`.
18. Attacken-Menü → `v` → `1`, `d` `2`, `s` `4`, `a` `3`.
19. `k`, `s` auf POKEMON, `j` → Team-Liste → `p` → `SPACE`. Im
    Trainerkampf **nicht** auf Rattfratz wechseln: mit `k` zurückgehen, FIGHT
    wählen und den Kampf vollständig mit Schiggi spielen.

### Teil C — Außerhalb

20. Kampf beenden. Sobald du wieder normal laufen kannst und im Terminal
    `AUTO OUT OF BATTLE` steht, einmal `SPACE` drücken.

`q` zum Beenden.

---

## Tastenübersicht

| Taste | Wirkung |
|---|---|
| `w a s d` | laufen · `j`=A · `k`=B · `n`=START · `l`=Start neu laden · `q`=beenden |
| `t` | Kampfart umschalten: wild ⇄ trainer |
| `W` | manueller Notfall: außerhalb markieren; normalerweise unnötig |
| `m` / `v` / `p` | Menü setzen: main / move / party |
| `7 8 9 0 - =` | aktiver Party-Slot 0 / 1 / 2 / 3 / 4 / 5 (nach einem Wechsel) |
| `SPACE` | Dump speichern |
| `1 2 3 4` | Dump **und** Attacken-Cursor-Position 0 / 1 / 2 / 3 |
| `F B P R` | Dump **und** Hauptmenü-Cursor FIGHT / BAG / POKEMON / RUN |

`1`–`4` sind **immer** Attacken-Cursor-Captures — nie ein Party-Slot.

---

## Auswerten

```bash
python tools/battle_dump_score.py runtime/ram_probe/<zeitstempel>/
```

- **Exitcode 0** + „ALL MANDATORY FIELDS VERIFIED" → bereit für den nächsten
  Schritt.
- **Exitcode 1** → noch blockiert; die Ausgabe sagt, was fehlt.

Ein Feld wird nur `VERIFIED`, wenn:
- Dumps aus **≥ 2 verschiedenen Kämpfen** kommen (Wild **und** Trainer),
- alle Werte im gültigen Bereich sind,
- positive **und** negative Kontexte da sind,
- nach einem Wechsel `gBattleMons[Spieler]` gegen den von
  `gBattlerPartyIndexes` genannten Slot passt — nie pauschal gegen Slot 0.

---

## Sicherheit

Eigener Emulator; Trainer/Watcher/Web unberührt. ROM nur gelesen. Keine
Savestates verändert. Dumps in `runtime/ram_probe/<zeitstempel>/`. Nichts wird
automatisch als Adresse übernommen — die Kandidaten in `battle_dump_score.py`
sind Vermutungen und gelten erst nach bestandener Prüfung.
