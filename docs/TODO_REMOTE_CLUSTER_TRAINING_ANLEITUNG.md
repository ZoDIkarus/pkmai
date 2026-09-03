# PKMAI: Remote-Cluster-Training – Schritt-für-Schritt-Anleitung

Diese Anleitung beschreibt, wie mehrere Rechner später **ein gemeinsames PPO-Brain** weitertrainieren können. Sie ist bewusst als Umsetzungsleitfaden geschrieben: Der aktuelle Repository-Stand enthält noch keinen verteilten Learner.

> **Ziel:** Mehrere Remote-Rechner führen Stable-Retro-Emulatoren aus und liefern Erfahrungen an einen zentralen Trainer. Ausschließlich dieser Master aktualisiert und speichert das gemeinsame Brain.
>
> **Nicht tun:** Auf mehreren Rechnern gleichzeitig `src/train.py` gegen denselben `runtime/checkpoints/`-Ordner starten. Das sind unabhängige PPO-Läufe mit konkurrierenden Checkpoint-Schreibvorgängen, kein gemeinsames Training.

---

## 1. Ausgangslage im aktuellen Projekt

Der lokale Trainer in `src/train.py` verwendet Stable-Baselines3 PPO und lokale `SubprocVecEnv`-Prozesse. Der Modellpfad ist fest unter `runtime/checkpoints/pokemon_model_latest.zip` definiert und wird vom Trainer geschrieben.

Es gibt außerdem `src/cluster_master.py`. Dieser Dienst kann bereits Worker per Heartbeat erfassen und Statusdaten ausgeben, aber er trainiert oder verteilt derzeit **kein** Brain. Er ist daher nur eine brauchbare Grundlage für Registrierung, Health-Checks und Statusübersicht.

Der gemeinsame Curriculum-/Exploration-Zustand wird aktuell innerhalb eines lokalen `multiprocessing.Manager()` gehalten. Dieser Speicher muss für Remote-Worker durch eine zentrale, netzwerkfähige Zustandsquelle ersetzt oder in den gewählten Trainings-Backend-Mechanismus integriert werden.

---

## 2. Zielarchitektur festlegen

```text
                 VPN / privates LAN

+--------------------+       Rollout-Batches       +------------------------+
| Worker A           | --------------------------> | Master / zentraler     |
| Stable-Retro x N   |                             | PPO-Learner            |
| keine Model-Saves  | <-------------------------- | einzige Brain-Version  |
+--------------------+        Policy-Version       | Checkpoints, Curriculum|
                                                   +------------------------+
+--------------------+                                      |
| Worker B           | <------------------------------------+
| Stable-Retro x N   |
| keine Model-Saves  |
+--------------------+
```

### Rollen

| Rolle | Aufgabe | Darf Checkpoints schreiben? |
|---|---|---:|
| Master / Learner | sammelt Rollouts, führt PPO-Updates aus, verwaltet Versionen und Quality-Gate | Ja, ausschließlich er |
| Remote-Worker | startet Emulatoren, erzeugt Beobachtungen/Aktionen/Rewards, sendet Batches | Nein |
| Watcher | lädt nur eine bestätigte Policy zur Visualisierung | Nein |

### Backend-Entscheidung

Für die Umsetzung wird **Ray RLlib PPO** empfohlen. RLlib bringt die zentrale Learner-/verteilte-Rollout-Architektur bereits mit. Stable-Baselines3 ist für die aktuelle lokale Parallelisierung geeignet, bietet aber keinen fertigen Mehrrechner-PPO-Cluster.

Ein selbst implementierter HTTP-Parameter-Server ist möglich, aber erst sinnvoll, wenn RLlib die konkrete Stable-Retro-Umgebung nicht zuverlässig betreiben kann. Er müsste Policy-Synchronisation, Rollout-Serialisierung, PPO-Update-Zyklen, Versionskontrolle, Retry/Timeouts und Checkpoint-Konsistenz selbst korrekt lösen.

---

## 3. Sicherheits- und Betriebsregeln vorab

1. **Keinen Master-Port offen ins öffentliche Internet stellen.**
2. Remote-Verbindungen ausschließlich über ein privates Netz, vorzugsweise WireGuard, herstellen.
3. ROMs, Save-States, lokale `.env`, Checkpoints und `runtime/` bleiben lokal und werden weder übertragen noch committed.
4. Der Cluster-Key gehört in lokale Konfiguration/Umgebungsvariablen, nie in Git und nie in Logs. Vor der produktiven Nutzung muss die Key-Ausgabe in `src/cluster_master.py` entfernt werden.
5. Der Master akzeptiert nur Worker mit exakt passender Software-/Environment-Signatur.
6. Kein automatischer Brain-Reset: Der Master setzt vom letzten gültigen Checkpoint fort.
7. Zuerst nur LAN und zwei Rechner testen. VPN-/Internetbetrieb erst danach einschalten.

---

## 4. Phase A – Voraussetzungen vorbereiten

### Schritt A1: Master-Rechner auswählen

Wähle einen Rechner, der während des Trainings zuverlässig laufen kann. Er braucht:

- genügend CPU/RAM für PPO-Learner, lokale Emulatoren (falls gewünscht) und die Annahme der Remote-Rollouts;
- schnellen, stabilen Speicher für Checkpoints;
- einen festen Namen bzw. eine stabile Wireguard-/LAN-Adresse;
- möglichst eine USV oder zumindest einen kontrollierten Neustartplan.

Der Master bleibt die alleinige Quelle für:

- Policy-/Brain-Versionen;
- PPO-Optimiererzustand;
- bestätigte Bestmodelle;
- Curriculum- und globalen Exploration-Zustand;
- Cluster-Workerstatus.

### Schritt A2: Jeden Worker standardisieren

Auf **jedem** Worker müssen lokal vorhanden sein:

- eine legal beschaffte FireRed-ROM samt lokaler Stable-Retro-Integration;
- die gleiche Python-/Paketumgebung;
- dieselbe Version des Repositories;
- dieselbe Version von `pokemon_env.py`, Reward-Logik, Action-Space und Observation-Space;
- genügend CPU, RAM und Speicher für die lokale Zahl an Emulatoren.

Die ROM darf nicht über das Git-Repository, HTTP, Ray oder einen Shared Folder verteilt werden.

### Schritt A3: Kompatibilitäts-Signatur definieren

Vor dem Start erzeugt jeder Worker lokal eine Signatur, die der Master prüft. Sie soll mindestens enthalten:

```text
Git-Commit oder freigegebene Build-ID
Python-Version
Versionen: stable-retro, gymnasium, torch, ray/rllib
Observation-Space-Signatur
Action-Space-Signatur
Reward-/Environment-Version
SHA-256 der lokalen ROM (nur Hash, nie die ROM übertragen)
```

Der Master lehnt einen Worker bei Abweichung ab. Besonders die Observation-Signatur ist zwingend: Unterschiedliche Feature-Längen oder Bildformen machen PPO-Rollouts unbrauchbar.

### Schritt A4: VPN/LAN einrichten und testen

1. WireGuard auf Master und Test-Worker installieren.
2. Beide Rechner in dasselbe private Netz aufnehmen.
3. Nur die private VPN-Adresse des Masters verwenden.
4. Vom Worker aus den Health-Endpunkt testen:

```bash
curl http://<MASTER-VPN-IP>:8765/health
```

5. Firewall nur für die VPN-Schnittstelle bzw. private LAN-Zone freigeben.
6. Den Zugriff aus einem nicht autorisierten Rechner prüfen und blockieren.

Der derzeitige Master verwendet standardmäßig Port `8765`; der endgültige Port und die Bind-Adresse müssen in der späteren Cluster-Konfiguration explizit festgelegt werden.

---

## 5. Phase B – Lokalen Zwei-Prozess-Prototyp bauen

Diese Phase zuerst auf **einem** Rechner durchführen. Ziel: Der spätere Netzwerkweg muss funktionieren, ohne dass ein zweiter PC Fehlerursache sein kann.

### Schritt B1: Trainingsbackend in eine Schnittstelle trennen

Die bisherige direkte Trainingssteuerung aus `src/train.py` wird aufgeteilt in:

- Environment-Erzeugung und PKMAI-spezifische Callbacks;
- Konfiguration für PPO;
- Master-/Learner-Start;
- Worker-/Rollout-Start;
- Checkpoint- und Curriculum-Verwaltung.

`PokemonFireRedEnv` bleibt die Environment-Implementierung. Die Reward- und Emulator-Logik darf bei dieser Umstellung nicht verändert werden.

### Schritt B2: RLlib lokal mit einem Worker starten

1. `ray[rllib]` als klar dokumentierte Abhängigkeit ergänzen und im vorgesehenen WSL-Python installieren.
2. Eine RLlib-kompatible Factory für `PokemonFireRedEnv` ergänzen.
3. Einen lokalen Master starten.
4. Einen lokalen Rollout-Worker als separaten Prozess starten.
5. Beide mit einer kleinen, festen Zahl Emulatoren betreiben (zunächst z. B. ein bis zwei).
6. Einen kurzen Trainingslauf durchführen.

**Abnahmekriterien:**

- Der Learner erhält tatsächlich Samples vom Worker.
- Nur der Master schreibt einen Checkpoint.
- Der PPO-Trainingszähler steigt.
- Ein Restart des Masters lädt den letzten Checkpoint inklusive Optimiererzustand.
- Der Worker kann beendet und neu gestartet werden, ohne den Master zu beenden.

### Schritt B3: Checkpoint-Format festlegen

Ein Cluster-Checkpoint muss mindestens enthalten:

```text
PPO-Policy-Gewichte
PPO-Optimiererzustand
globale Timesteps / Iteration
Policy-Version
Environment-/Observation-Signatur
Git-Commit oder Build-ID
Curriculum-Zustand
Quality-/Bestmodel-Metadaten
```

Checkpoint schreiben:

1. in ein neues temporäres Verzeichnis oder eine temporäre Datei;
2. vollständig schreiben und prüfen;
3. atomar auf die neue Version umbenennen;
4. erst danach diese Version für Worker freigeben.

Dadurch lädt kein Worker einen halbfertigen Checkpoint.

---

## 6. Phase C – Curriculum und Exploration zentral machen

Der aktuelle lokale Manager-Speicher darf nicht hostübergreifend verwendet werden. Definiere daher pro Datenklasse klar Eigentümer, Format und Schreibrecht.

| Daten | Eigentümer | Worker-Zugriff | Bemerkung |
|---|---|---|---|
| PPO-Gewichte und Optimierer | Master | Lesen über Policy-Sync | nur Master schreibt |
| Curriculum-Freigaben/Stage-Statistik | Master | lesen, Ergebnisse senden | zentraler Konfliktlöser |
| globale Exploration (Kanten/Maps/Warps) | Master oder dedizierter Zustandsdienst | deltas senden, Snapshot lesen | Updates deduplizieren |
| Worker-Telemetrie | jeweiliger Worker | schreiben | für Dashboard/Health |
| Watcher-Bilder/Tiles | Watcher | schreiben | weiterhin nicht durch Trainingsworker |

### Schritt C1: Datenprotokoll klein halten

Worker senden keine kompletten `runtime/`-Ordner. Sie senden ausschließlich:

- Rollout-Daten für PPO;
- kompakte Exploration-Deltas;
- Stage-Ergebnisse;
- Telemetrie (FPS, aktive Emulatoren, CPU/RAM optional);
- ihre Kompatibilitäts-Signatur beim Anmelden.

### Schritt C2: Versionsgrenzen definieren

- Jeder Rollout erhält die Policy-Version, mit der er erzeugt wurde.
- Der Master akzeptiert nur Rollouts innerhalb eines konfigurierten Versionsfensters.
- Bei zu alter Policy fordert der Master einen Reload an.
- Ein Worker übernimmt eine neue Policy nur am Ende eines vollständigen Rollout-Batches bzw. an einer sicheren Episode-Grenze.

Damit werden extrem veraltete Erfahrungen nicht unbegrenzt in PPO-Updates gemischt.

---

## 7. Phase D – Einen echten zweiten Rechner hinzufügen

### Schritt D1: Worker installieren

Auf dem zweiten Rechner:

1. Repository auf den freigegebenen Commit bringen.
2. WSL2/Ubuntu und den vorgesehenen Linux-Python einrichten; Stable-Retro nicht nativ unter Windows erzwingen.
3. Abhängigkeiten inklusive des neuen Cluster-Backends installieren.
4. Die lokale ROM-/Stable-Retro-Integration einrichten.
5. Nur lokal den Environment-Smoke-Test ausführen.
6. VPN-Verbindung zum Master testen.

### Schritt D2: Worker registrieren und freigeben

Der Worker meldet sich mit:

- Worker-ID;
- Hostname;
- Build-/Commit-ID;
- Kompatibilitäts-Signatur;
- gewünschter und tatsächlich gestarteter Emulatorzahl;
- FPS/Status.

Der Master antwortet entweder mit `accepted` und der aktuellen Policy-Version oder mit einer präzisen Ablehnung, etwa `observation_signature_mismatch`.

### Schritt D3: Sehr klein starten

1. Master + lokaler Worker normal starten.
2. Remote-Worker zunächst mit **einem** Emulator starten.
3. Fünf bis zehn Minuten laufen lassen.
4. Prüfen:
   - Worker bleibt online;
   - Samples/s steigen;
   - Master lernt weiter;
   - kein Checkpoint wird vom Worker geschrieben;
   - Curriculum-/Exploration-Updates sind valide;
   - Netzwerk- und Master-CPU bleiben stabil.
5. Erst danach die Zahl der Remote-Emulatoren schrittweise erhöhen.

---

## 8. Bandbreite, Batch-Größe und Stabilität abstimmen

Die Bildobservationen sind relativ groß. Deshalb vor einem Ausbau messen statt raten.

Zu erfassen sind mindestens:

```text
Worker: aktive Emulatoren, FPS, Rollouts/s, Uploadrate, Queue-Länge
Master: Samples/s, PPO-Iterationsdauer, CPU/RAM, GPU-RAM falls genutzt
Netz: Latenz, Upload/Download, Retry-Rate
Training: Reward, bestätigte Stages, Regressionen, Checkpoint-Dauer
```

Empfohlene Reihenfolge:

1. kleine Rollout-Batches festlegen;
2. Übertragung komprimieren;
3. Batch-Größe erhöhen, bis Netzwerk oder Lernlatenz problematisch wird;
4. erst dann weitere Emulatoren/Worker aktivieren;
5. bei instabiler Verbindung Workerzahl reduzieren statt Timeouts zu ignorieren.

Ein Worker darf bei schlechter Verbindung pausieren oder nur wieder verbinden; er darf niemals eigenständig lokal weiteroptimieren und später ein zweites Brain hochladen.

---

## 9. Offline-Fallback: Lokales Training bei nicht erreichbarem Master

**Ja: Lokale Rechner sollen bei einem Ausfall oder einer nicht verfügbaren Remote-Verbindung weiter nutzbar bleiben.** Das muss jedoch als klar getrennter Betriebsmodus umgesetzt werden, weil sich zwei unabhängig per PPO weitertrainierte Modelle nicht sicher zu einem gemeinsamen Brain zusammenführen lassen.

### Betriebsmodi

| Modus | Wann | Verhalten | Checkpoint-Ziel |
|---|---|---|---|
| `cluster` | Master erreichbar und Worker akzeptiert | Worker erzeugt Rollouts für das zentrale Brain; nur Master lernt | ausschließlich Master-Checkpoint |
| `buffered` | kurze Unterbrechung nach zuvor gültiger Cluster-Verbindung | Worker arbeitet mit der zuletzt geladenen Master-Policy weiter und puffert nur begrenzte, versionierte Rollouts | keine lokalen Brain-Saves |
| `local` | Master beim Start nicht erreichbar oder Unterbrechung zu lang | der Rechner trainiert mit dem vorhandenen lokalen SB3-Trainer weiter | ausschließlich lokaler Fallback-Checkpoint |
| `paused` | weder Master noch erlaubter lokaler Fallback verfügbar | Emulatoren/Worker werden kontrolliert angehalten | keiner |

### Schritt F1: Lokale und Cluster-Checkpoint-Bereiche strikt trennen

Die vorhandenen lokalen Pfade unter `runtime/checkpoints/` dürfen nicht mit Cluster-Checkpoints oder Worker-Puffern geteilt werden. Bei der Implementierung werden mindestens diese getrennten Bereiche benötigt:

```text
runtime/checkpoints/                  # bestehendes lokales SB3-Brain
runtime/cluster/checkpoints/          # ausschließlich Master-Cluster-Brain
runtime/cluster/rollout_spool/        # begrenzter, temporärer Worker-Puffer
runtime/local_fallback/               # lokale Fallback-Logs und Metadaten
```

Ein lokaler Fallback darf nie `runtime/cluster/checkpoints/` beschreiben. Der Master darf nie einen lokalen Fallback-Checkpoint automatisch übernehmen.

### Schritt F2: Verhalten bei kurzer Unterbrechung

Ein zuvor akzeptierter Cluster-Worker darf bei einer kurzen Verbindungslücke weiter Emulator-Erfahrung sammeln, aber **keine PPO-Updates ausführen**. Jeder gepufferte Batch erhält:

```text
policy_version
environment_signature
worker_id
erzeugt_am
anzahl_steps
checksum
```

Nach einem reconnect entscheidet der Master:

1. Batch passt zu Build, Observation und noch zulässiger Policy-Version: annehmen.
2. Batch ist zu alt oder stammt aus einer unzulässigen Policy-Version: verwerfen und aktuelle Policy laden.
3. Puffer ist voll oder die Offline-Grenze überschritten: Worker wechselt kontrolliert in `local` oder `paused`.

Der Puffer muss eine feste Obergrenze für Alter, Größe und Batchzahl haben. PPO ist on-policy; alte Rollouts dürfen nicht unbegrenzt nachträglich gelernt werden.

### Schritt F3: Verhalten bei längerer Nichtverfügbarkeit

Ist der Master beim Start nicht erreichbar oder dauert die Unterbrechung länger als die konfigurierte Offline-Grenze, darf der Rechner den bestehenden lokalen Trainer starten. Dieser läuft weiter mit seinem **eigenen lokalen Brain** und dem vorhandenen lokalen Curriculum-/Exploration-Zustand.

Dabei gelten zwingend:

1. Der Start muss sichtbar als `LOCAL FALLBACK` geloggt werden.
2. Lokale Checkpoints bleiben ausschließlich lokal.
3. Der lokale Trainer versucht nicht, das zentrale Brain zu sperren, zu ersetzen oder hochzuladen.
4. Nach Rückkehr des Masters wird der lokale Trainingslauf nicht automatisch mit dem Cluster vermischt.
5. Für eine Rückkehr in den Cluster wird der lokale Trainer kontrolliert gestoppt, sein Final-Save abgeschlossen und anschließend ein Cluster-Worker mit der aktuellen Master-Policy gestartet.

### Schritt F4: Optionalen lokalen Kandidaten bewusst bewerten

Ein lokaler Fallback-Checkpoint kann später manuell als **Kandidat** geprüft werden. Er wird nicht gemergt. Der sichere Ablauf ist:

1. Kandidaten-Checkpoint als unveränderliche Kopie kennzeichnen.
2. Aktuelles Master-Brain und Kandidat mit derselben Evaluation-Suite, Seeds und Erfolgsmetriken ausführen.
3. Nur wenn der Kandidat nach definierten Kriterien besser ist, wird er durch eine explizite Operator-Entscheidung als neue Ausgangsbasis für einen neuen Master-Trainingslauf übernommen.
4. Optimiererzustand, Versionshistorie und Rückrollmöglichkeit des bisherigen Master-Brains bleiben erhalten.

Eine Mittelung oder ein automatisches "Zusammenführen" der PPO-Gewichte ist ausdrücklich verboten.

---

## 10. Fehlerfälle und erwartetes Verhalten

| Fehlerfall | Erwartetes Verhalten |
|---|---|
| Worker verliert Verbindung | Worker puffert nur kurz versionierte Rollouts, danach `local` oder `paused`; Master markiert ihn nach Timeout offline |
| Master beim Worker-Start nicht erreichbar | Worker startet klar markierten lokalen Fallback oder bleibt kontrolliert pausiert |
| Lokaler Fallback endet | Final-Save bleibt lokal; kein automatischer Upload oder Merge in das Cluster-Brain |
| Worker startet mit falschem Build/ROM/Observation | Master lehnt ihn vor dem ersten Rollout ab |
| Master startet neu | Master lädt den letzten vollständigen Cluster-Checkpoint und akzeptiert Worker erneut |
| Worker startet neu | Worker registriert sich erneut und lädt die aktuelle Policy |
| Checkpoint-Schreiben schlägt fehl | neue Policy wird nicht als aktiv markiert; vorheriger Checkpoint bleibt nutzbar |
| Quality-Gate erkennt Regression | Master entscheidet zentral über Bestmodell/Rollback |
| Watcher fällt aus | Training läuft weiter; Watcher trainiert nie |

---

## 11. Geplante Repository-Änderungen bei der Umsetzung

Die genauen Dateinamen können während des Designs angepasst werden. Voraussichtlich betroffen:

| Bereich | Wahrscheinliche Änderung |
|---|---|
| `src/train.py` | lokale SB3-Endlosschleife in Master-/Cluster-Einstieg überführen oder als lokaler Fallback erhalten |
| `src/pokemon_env.py` | nur falls eine RLlib-Adapter-Factory nötig ist; Reward-/Emulatorverhalten nicht verändern |
| `src/cluster_master.py` | Authentifizierung härten, keine Key-Ausgabe, Worker-Kompatibilität und Clusterstatus ergänzen |
| `src/cluster_worker.py` (neu) | Anmeldung, Compatibility-Check, Rollout-Worker-Start, Health/Telemetry |
| `src/cluster_config.py` (neu) | zentrale, testbare Cluster-Konfiguration und Signaturbildung |
| `scripts/start_cluster_master_wsl.sh` (neu) | idempotenter Master-Start mit PID/Log |
| `scripts/start_cluster_worker_wsl.sh` (neu) | idempotenter Worker-Start mit PID/Log und expliziter Fallback-Modusentscheidung |
| `scripts/start_local_fallback_wsl.sh` (neu oder bestehendes Script erweitert) | klar gekennzeichneter lokaler SB3-Fallback mit getrenntem Checkpoint-Pfad |
| `scripts/stop_cluster_*_wsl.sh` (neu) | kontrolliertes Stoppen ohne fremde Prozesse zu beenden |
| `tests/` | Kompatibilität, Versionierung, Worker-Ablehnungen, Checkpoint-/Recovery-Tests |
| `docs/WINDOWS_WSL_ANLEITUNG.md` | konkrete Start-, Status-, Log- und Stop-Kommandos nach erfolgreicher Umsetzung |
| `AGENTS.md` | dauerhafte Betriebsregeln für Cluster-Training ergänzen |

---

## 12. Go-Live-Checkliste

Vor dauerhaftem Clustertraining müssen alle Punkte erfüllt sein:

- [ ] Master ist der einzige Writer für Brain, Optimierer und Checkpoints.
- [ ] Ein lokaler Master/Worker-Prototyp hat erfolgreich gelernt und einen Restart überstanden.
- [ ] Ein zweiter LAN-Worker erhöht nachweislich die Samples/s.
- [ ] Falsche Environment-/ROM-/Build-Signaturen werden abgewiesen.
- [ ] Keine ROM, kein Save-State, kein Checkpoint und kein Secret wird übertragen oder committed.
- [ ] Master-Port ist nicht öffentlich erreichbar; Zugriff läuft nur über LAN/VPN.
- [ ] Cluster-Key erscheint weder in Logs noch Dashboard noch Git.
- [ ] Worker-Ausfall und Worker-Restart wurden getestet.
- [ ] Kurze Master-Unterbrechung mit begrenztem Rollout-Puffer wurde getestet.
- [ ] Längerer Master-Ausfall startet keinen konkurrierenden Cluster-Writer, sondern nur den getrennten lokalen Fallback oder pausiert kontrolliert.
- [ ] Ein lokaler Fallback-Checkpoint wird beim Reconnect nicht automatisch hochgeladen oder mit dem Master-Brain gemergt.
- [ ] Checkpoint-Ausfall und Master-Restart wurden getestet.
- [ ] Watcher bleibt optional und nimmt nicht am Training teil.
- [ ] Die Laufdokumentation enthält die tatsächlich geprüften Start-/Stop-/Status-Kommandos.

---

## 13. Empfohlener erster Umsetzungsschritt

Nicht sofort mehrere Rechner anbinden. Zuerst einen kleinen lokalen Prototyp erstellen:

1. RLlib als Backend evaluieren.
2. Einen Master und genau einen lokalen Worker starten.
3. Einen kurzen Lernlauf mit wenigen Emulatoren beweisen.
4. Checkpoint-Resume und Worker-Reconnect testen.
5. Erst danach den ersten Heimrechner über VPN/LAN hinzufügen.

So bleibt bei Problemen klar, ob sie aus PKMAI, dem Trainingsbackend, dem Netzwerk oder dem zweiten Rechner stammen.
