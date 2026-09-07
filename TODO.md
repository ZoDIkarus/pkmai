# TODO — Offene, verifizierte Cluster-Contributions

> **Status:** Geplant. Erst umsetzen, nachdem die aktuelle Trainings-/Clusterarbeit stabil ist.

## Ziel

Das PKMAI-Cluster soll Contributions von beliebigen Trainern **ohne individuell verteilte Cluster-Keys** akzeptieren können, ohne dass ein einzelner böswilliger oder kaputter Trainer Training, Rewards oder die Best-Policy manipulieren kann.

Wichtig: „Jeder darf beitragen“ bedeutet **nicht**, dass jeder direkt Modellgewichte, Gradienten, Rewards oder PPO-Daten als wahr bestimmen darf.

Die zentrale Control Plane und das zentrale Gehirn bleiben vertrauenswürdig. Öffentliche Teilnehmer dürfen nur Vorschläge bzw. Action-Trajektorien liefern, die zentral reproduziert und verifiziert werden.

## Vor der Umsetzung prüfen

Relevante Dateien und ihre Aufrufer lesen:

- `src/cluster_master.py`
- `src/cluster_worker.py`
- `src/cluster_config.py`
- `src/rollout_protocol.py`
- `src/dynamic_brain.py`
- `src/dynamic_watcher.py`
- `src/pokemon_env.py`
- `src/curriculum.py`
- bestehende Tests unter `tests/`

Bestehende Bausteine:

- Cluster-Key-Auth im Master;
- Kompatibilitäts-Signatur für Observation `(64, 64, 1)`, `nav_features=28`, `action_count=7`, Protokollversion;
- Policy-Version-Fenster;
- atomare globale Reward-Claims;
- RAM-basierte Rewards, Anti-Loop-Logik und Curriculum-Savestates;
- Curriculum-Qualitätslogik mit 10 Mindestversuchen, 60-%-Erfolgsquote, Rolling Window und lexikografischem Qualitätsvergleich.

Der aktuelle Upload akzeptiert nur strukturelle Mindestfelder und gleiche Array-Längen. Das reicht nicht: Ein Worker mit gültigem Key könnte manipulierte Rewards, `log_probs`, Values oder unglaubwürdige Trajektorien einsenden.

Aktuell wird die Best-Policy in `src/dynamic_brain.py` nach Batch-`mean_reward` bestimmt. Das ist kein zulässiges Promotion-Kriterium.

## Zielarchitektur

### 1. Trust-Split: öffentliche Contribution-Queue, private Control Plane

Die bestehende Authentifizierung darf für administrative/private Funktionen bestehen bleiben, beispielsweise:

- Policy/Best-Policy veröffentlichen,
- Checkpoint-/Modellverwaltung,
- Learner und Evaluator steuern,
- interne Status- und Verwaltungsendpunkte.

Es soll einen **öffentlichen Contribution-Endpunkt ohne vorab ausgeteilten Cluster-Key** geben.

Dieser Endpunkt darf Contributions ausschließlich in eine Quarantäne-/Prüfqueue schreiben. Er darf niemals direkt:

- Rollouts in die Learner-Inbox legen,
- Modellgewichte entgegennehmen,
- Gradienten entgegennehmen,
- selbst gemeldete Rewards vertrauen,
- `best` oder aktuelle Policy verändern.

Kein neues Account-/Key-Verteilungssystem einführen. Stattdessen Missbrauch begrenzen über konfigurierbare, sichere Defaults für Payloadgrößen, maximale Episoden-/Action-Längen, Queue-Grenzen, Deduplizierung und gegebenenfalls einfache IP-/Zeit-basierte Rate Limits.

### 2. Untrusted Contributors senden keine fertigen PPO-Rollouts

Das öffentliche Format darf keine vom Client behaupteten Werte als Lernwahrheit akzeptieren:

- `rewards`,
- `images` / Observations,
- `nav`,
- `values`,
- `log_probs`,
- behauptete Meilensteine oder Positionen,
- fremde Modellgewichte.

Ein Beitrag soll mindestens enthalten:

- eine eindeutige, serverseitig bekannte `start_state_id` bzw. Stage-ID,
- deklarierte Basis-`policy_version`,
- eine begrenzte Folge diskreter Actions,
- optional harmlose Debug-/Timing-Metadaten, die nie Wahrheitsquelle sind.

Wenn vollständig deterministische Replay-Validierung noch nicht genug Infrastruktur hat, eine robuste, begrenzte erste Version implementieren und Sicherheits-/Determinismus-Annahmen präzise dokumentieren. Nichts vortäuschen.

### 3. Zentraler vertrauenswürdiger Replay-Evaluator

Einen Evaluator-Dienst bzw. -Worker bauen, der Quarantäne-Contributions verarbeitet.

Er muss:

1. den gewünschten Startzustand ausschließlich aus einem zentral verwalteten, bekannten Curriculum-State laden;
2. Actions in einer eigenen lokalen FireRed-/Emulator-Umgebung wiedergeben;
3. Observation, Navigation, RAM-Zustand, Reward, Done-Flags und Meilensteine selbst erzeugen;
4. die bestehende `PokemonFireRedEnv`- und Reward-Logik als Wahrheitsquelle nutzen statt Client-Werte zu übernehmen;
5. ungültige Startzustände, veraltete Policy-Versionen, unzulässige Actions, zu lange Sequenzen, Decoderfehler, nicht-endliche Werte und Replay-/Emulatorfehler sauber verwerfen;
6. nur erfolgreich reproduzierte und plausibel begrenzte Trajektorien als **verifizierte Rollouts** in den Learner-Pfad geben.

Öffentliche Requests dürfen keine beliebigen Dateipfade, Savestate-Dateinamen oder Server-Dateien lesen können. Nur Whitelist bekannter Stage-IDs.

Schlechte, aber formal gültige Actions sind kein Sicherheitsproblem; sie dürfen nur begrenzt Evaluator-Ressourcen kosten und keine gefälschten positiven Rewards erzeugen.

### 4. Strikte Validierung auch im privaten Rollout-Pfad

Den internen Rollout-Protokollpfad so härten, dass defekte Daten den Learner nicht zum Absturz bringen oder still vergiften:

- exakte erwartete Shapes;
- erwartete Dtypes bzw. sichere Konvertierung;
- alle Floatwerte endlich (`NaN`/`Inf` ablehnen);
- Actions strikt im gültigen Bereich;
- gleiche, positive Batchlänge;
- harte Größen-/Längenlimits;
- erlaubte Feldnamen, keine unerwarteten Object-/Pickle-Daten;
- Policy-Version und Environment-Signatur an den Beitrag binden, soweit passend;
- Fehler verwerfen nur den einzelnen Beitrag, nicht den Learner-Prozess.

### 5. Best-Policy nur durch unabhängige Qualitätsprüfung promoten

Die Promotion nach Batch-`mean_reward` in `src/dynamic_brain.py` ersetzen.

Die Best-Policy-Entscheidung muss mindestens die bestehende Curriculum-Qualitätslogik verwenden:

1. Anzahl zuverlässig bestätigter Stufen,
2. Rolling-Erfolgsquote,
3. dann niedrigere mediane erfolgreiche Schrittzahl.

Ein einzelner hoher Reward oder einzelner Glückstreffer darf nie genügen.

Zusätzlich soll ein zentraler, vertrauenswürdiger End-to-End-Evaluator die Promotion absichern:

- beginnt immer beim echten Spielanfang, niemals in einem späten Trainer-Savestate;
- prüft die höchste relevante Stage mehrfach;
- nutzt definierte, vergleichbare Prüfbedingungen;
- schreibt Status und Ergebnis atomar;
- promoted nur bei strikter Verbesserung gegenüber der gespeicherten Best-Policy;
- bei klarer Regression bleibt die bisherige Best-Policy erhalten bzw. wird vor dem nächsten Lernblock wiederhergestellt.

Unbedingt unterscheiden:

- `latest` = neueste trainierte Policy,
- `best` = unabhängig verifizierte, beste Policy.

Der Watcher bleibt sichtbare Demo, nicht einzige Vertrauensquelle. Er startet weiterhin beim echten Start und lädt keinen späten Curriculum-State als Start.

### 6. Beobachtbarkeit

Sichere, nicht-sensitive Status-/Dashboard-Zähler ergänzen:

- öffentliche Contributions empfangen,
- wegen Schema/Rate Limit verworfen,
- Replay erfolgreich,
- Replay fehlgeschlagen,
- verifizierte Schritte,
- verworfene alte Policy-Versionen,
- aktuelle Evaluator-Queue-Länge,
- letzte Evaluator-Ergebnisse,
- Best-Policy-Qualitätsvektor und Promotion-Grund.

Keine Cluster-Keys, vollständigen privaten Pfade oder andere Secrets in API/Dashboard ausgeben.

## Engineering-Vorgaben

- Keine große, ungetestete Komplett-Umschreibung.
- Erst bestehende Architektur und Tests nachvollziehen, dann klein und gezielt ändern.
- Reine Validierungs- und Vergleichslogik als gut testbare Funktionen extrahieren.
- Für öffentliche Payloads ein explizites, versionsfähiges Schema definieren.
- Atomare Dateioperationen für persistierte Queue-/Status-/Best-Policy-Metadaten verwenden.
- Bei Exceptions fail-closed: Ein nicht prüfbarer Beitrag darf keinen positiven Einfluss auf das Training erhalten.
- Public Queue und verified Learner Inbox strikt trennen.
- Bestehende lokale/vertrauenswürdige Trainingsabläufe nicht unnötig brechen.
- Keine zusätzlichen Abhängigkeiten einführen, wenn Standardbibliothek/FastAPI/Numpy genügen.
- Deutsche Kommentare und Stil des Projekts respektieren.

## Tests und Verifikation

Fokussierte Unit-Tests mindestens für:

1. öffentliche Contribution-Schema- und Größenvalidierung;
2. Whitelist bekannter Start-State-IDs;
3. Reject von `NaN`, `Inf`, falschen Shapes und Action-Out-of-Range;
4. Client-Rewards werden nie als verifizierte Rewards übernommen;
5. nur verifizierte Contributions landen im Learner-Inbox-Pfad;
6. ein kaputter/einzelner Contribution kann den Learner nicht zum Absturz bringen;
7. Best-Policy-Promotion bevorzugt bestätigte Tiefe vor allem anderen, Zuverlässigkeit vor Geschwindigkeit und Geschwindigkeit nur bei gleicher Zuverlässigkeit;
8. ein einzelner hoher Batch-Reward darf keine Best-Promotion erzwingen;
9. vorhandene Cluster-Kompatibilitäts- und Curriculum-Tests bleiben grün.

Wenn Emulator-Integration im CI nicht ausführbar ist, Replay-Evaluator über abstrahierte/fake Environment-Adapter testen und ehrlich dokumentieren, welche Live-Emulator-Prüfung noch erforderlich ist.

## Abschluss bei späterer Umsetzung

- Relevante Tests, Syntax- und Importchecks ausführen.
- `git status` prüfen.
- Änderungen gezielt committen.
- Vor Push nochmals `origin/sascha` integrieren.
- Nach Konfliktlösung fokussierte Tests wiederholen.
- Ohne Force-Push nach `origin/sascha` pushen.
- Abschließend kurz Sicherheitsgrenzen, öffentliche Endpunkte, erlaubte Contribution-Daten und reale Testergebnisse dokumentieren.
