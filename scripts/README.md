# PKMai starten, stoppen und gezielt zurücksetzen

Alle Befehle werden im Projektordner ausgeführt:

```bash
cd /Users/alexnotabi/pokemon_ai_project
```

## Alles starten

Öffnet die Dienste in sichtbaren Terminalfenstern. Bereits laufende Dienste
werden nicht doppelt gestartet.

```bash
bash scripts/start_all.sh
```

## Alles sauber stoppen

Speichert Navigation- und Battle-Learner, beendet die Dienste und schließt
anschließend deren Terminalfenster.

```bash
bash scripts/stop_all.sh
```

## Nur das Navigation Brain zurücksetzen

Löscht Navigation-Champion, Learner und Kandidaten. Mapwissen, globale
Freischaltungen, Savestates, Battle Brain und Battle-Szenarien bleiben erhalten.

Zuerst nur anzeigen, was betroffen wäre:

```bash
bash scripts/reset_nav_brain.sh --dry-run
```

Danach wirklich ausführen:

```bash
bash scripts/reset_nav_brain.sh
```

## Map und globale Freischaltungen zurücksetzen

Löscht den gerichteten Bewegungsgraphen, Exploration Memory, per-Agent-Shaping,
Curriculum-v20-Freischaltungen, Live-Telemetrie und Lernkurven. Navigation- und
Battle-Modelle sowie sämtliche Savestates bleiben erhalten.

```bash
bash scripts/reset_map_global.sh --dry-run
bash scripts/reset_map_global.sh
```

## Nur das Battle Brain zurücksetzen

Löscht sämtliche gelernten Battle-PPOs für v1 und v2 sowie Battle-Statistiken.
Navigation, Maps, Savestates und Battle-Szenarien bleiben erhalten.

```bash
bash scripts/reset_battle_brain.sh --dry-run
bash scripts/reset_battle_brain.sh
```

## Vollständiger Reset

Kombiniert Navigation-Brain-, Battle-Brain- und Map/Global-Reset. Vor jeder
Änderung wird der komplette Ordner `runtime/` nach
`backups/component_resets/full_<ZEITSTEMPEL>/runtime_before_reset.tar.gz`
gesichert. Eine SHA-256-Prüfsumme und ein Dateimanifest liegen daneben. Die
laufenden Wild-/Shiny-Zähler werden beim vollständigen Reset ebenfalls geleert;
beim einzelnen Navigation-, Map- oder Battle-Reset bleiben sie erhalten.

Der echte Reset startet erst, nachdem exakt `KOMPLETT RESET` eingegeben wurde.

```bash
bash scripts/reset_all.sh --dry-run
bash scripts/reset_all.sh
```

Geschützt und bei allen Resets erhalten bleiben:

- die handgemachte `StartGame.state`;
- Shared- und Agent-Savestates;
- Battle-Szenarien und Seeds;
- bestehende Backup-Verzeichnisse.

Nach einem Reset wieder starten:

```bash
bash scripts/start_all.sh
```

Für bewusst nicht-interaktive einzelne Resets existiert `--yes`. Beim
vollständigen Reset wird die ausgeschriebene Bestätigung trotzdem immer
verlangt.
