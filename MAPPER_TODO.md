# Mapper — pausierter Stand und TODO

Stand: 2026-09-04. Der Mapper ist bewusst ausgeschaltet. Trainer, Watcher,
Webserver, Status und Cloudflare duerfen unabhaengig weiterlaufen.

## Bereits umgesetzt

- Eigenstaendiger Mapper (Instanz 121), der Hauptmodell, Champion,
  Curriculum-States und gemeinsame Journey-Daten nicht veraendern darf.
- Persistenter Frontier-Graph statt zufaelliger PPO-Bewegung auf der Welt.
- Exakte RAM-Tile-Abdeckung als verlaessliche Kartenbasis.
- 60-FPS-Anzeige, eine Aktion alle 1,5 Sekunden, 750 ms Beruhigungszeit und
  drei zeitlich getrennte ruhige Aufnahmen pro Position.
- Keine Mapper- oder Kartenbilder waehrend eines Kampfes.
- Screenshot-Mosaik wird erst nach einem nachgewiesenen Kamerasprung als
  absolut ausgerichtet freigegeben (`alignment_confident=true`).
- Alte verschobene Karten liegen wiederherstellbar unter
  `runtime/mapper/map_backups/pre_frontier_reset/`.
- Live-Tiles und sichere Screenshot-Overlays sind im Overworld-Tab vorhanden.
- 36 Mapper-/Projekt-Tests waren beim Pausieren erfolgreich.

## Vor dem naechsten Mapper-Start erledigen

1. **Direktes Kampfflag verwenden.** Die Basisklasse haelt `in_battle` wegen
   Gegner-RAM noch 96 Entscheidungen aktiv. Dadurch steuert der Mapper nach
   einem echten Kampf mehr als zwei Minuten faelschlich weiter das Kampfmenue.
   In `src/mapper.py` fuer Steuerung, Aufnahme-Sperre und Status direkt
   `firered_ram.read_battle_type_flags()` benutzen; bei Lesefehler konservativ
   keine Aufnahme schreiben.
2. Die vorhandene deterministische Fluchtfolge `B, B, DOWN, RIGHT, A`
   gegen einen echten Wildkampf der deutschen Feuerrot-ROM pruefen. Falls sie
   nicht flieht: anhand des direkten Kampfflags zeitlich begrenzen und den
   Mapper sauber auf seinen Start-Savestate zuruecksetzen.
3. Sicherstellen, dass Kampf-Richtungstasten niemals als Weltkanten oder
   blockierte Tiles im Frontier-Graph gespeichert werden.
4. Tests ausfuehren und danach genau **einen** Mapper starten. Noch keine drei
   Mapper: parallele Bildschreiber beschleunigen denselben Fehler und koennen
   das Mosaik gegenseitig ueberschreiben.
5. Im echten Lauf mindestens einen Zyklus nachweisen:
   `frontier -> battle_escape -> frontier`, waehrend `new_positions` und
   `frontier_edges` danach wieder wachsen.
6. Route-1-PNG visuell auf zusammenhaengende 16x16-Tiles pruefen und bestaetigen,
   dass unkalibrierte Karten nicht auf der Overworld eingeblendet werden.

## Sicheres Starten/Stoppen

- Start spaeter ueber `./start_all.sh --no-cloudflare`; die Dublettenpruefung
  muss genau einen `src/mapper.py` ergeben.
- Zum Stoppen Mapper per SIGINT beenden, auf die Meldung zum Speichern von
  Frontier-Gehirn und Bildkarten warten und erst danach sein Terminal mit
  `exit` schliessen.
- Reward-Balancing des Haupttrainings (Kampf/Level gegen Weltfortschritt und
  Heil-Loops) ist ein separates TODO und wird erst nach dem Mapper angefasst.
