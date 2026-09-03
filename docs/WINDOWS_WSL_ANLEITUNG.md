# PKMAI unter Windows 11 ausführen (WSL2)

Diese Anleitung beschreibt die funktionierende lokale Windows-Konfiguration für dieses Projekt.

> **Kurzfassung:** PKMAI läuft hier nicht nativ mit Stable-Retro unter Windows, sondern innerhalb von **WSL2 mit Ubuntu 24.04**. Der sichtbare Watcher wird direkt als lokaler Browser-Stream bereitgestellt – nicht über VNC/noVNC.

## 1. Voraussetzungen

- Windows 11 mit aktivierter Virtualisierung
- WSL2 und Ubuntu 24.04
- Git
- Eine **rechtmäßig lokal vorhandene** Pokémon-Feuerrot-ROM

Die ROM wird nicht aus dem Internet geladen und gehört nicht in Git.

## 2. WSL2 installieren

In einer administrativen Windows-Konsole einmal ausführen:

```powershell
wsl --install --distribution Ubuntu-24.04
```

Nach dem ggf. nötigen Neustart Ubuntu einmal starten und den Linux-Benutzer anlegen.

Prüfen:

```powershell
wsl -l -v
```

`Ubuntu-24.04` muss mit Version `2` angezeigt werden.

## 3. Projekt auschecken

In Git Bash oder einer Windows-Konsole:

```bash
git clone https://github.com/ZoDIkarus/pkmai.git C:/zod/pkmai
```

Im weiteren Verlauf liegt das Projekt unter:

```text
Windows: C:\zod\pkmai
WSL:     /mnt/c/zod/pkmai
```

## 4. Linux-Abhängigkeiten installieren

In WSL ausführen:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  build-essential cmake swig git \
  libgl1 libglu1-mesa python3-opengl
```

`libgl1`, `libglu1-mesa` und `python3-opengl` sind relevant für Retro/OpenGL-nahe Abhängigkeiten. Eine VNC-/Xvfb-Installation ist für den aktuellen direkten Browser-Watcher **nicht erforderlich**.

## 5. Python-Umgebung und Stable-Retro einrichten

In WSL:

```bash
python3 -m venv /opt/pkmai-venv
/opt/pkmai-venv/bin/pip install --upgrade pip
cd /mnt/c/zod/pkmai
/opt/pkmai-venv/bin/pip install -r requirements.txt
/opt/pkmai-venv/bin/pip install stable-retro==1.0.1
```

Kurztest:

```bash
/opt/pkmai-venv/bin/python -c "import stable_retro; print(stable_retro.__version__)"
```

## 6. Lokale FireRed-ROM integrieren

Lege deine eigene ROM **lokal** in diesen Pfad:

```text
C:\zod\pkmai\local\custom_integrations\PokemonFireRed-Gba\rom.gba
```

Der zugehörige WSL-Pfad lautet:

```text
/mnt/c/zod/pkmai/local/custom_integrations/PokemonFireRed-Gba/rom.gba
```

Wichtig:

- Die Datei muss `rom.gba` heißen.
- Keine ROM oder Spielstände committen oder hochladen.
- Die lokale Integration wird von PKMAI über `CUSTOM_ONLY` verwendet.

## 7. Aktuelle lokale Performance-Anpassungen

Diese Installation nutzt bewusst folgende lokale Einstellungen:

- **10 Trainingsinstanzen** (`NUM_ENVS = 10` in `src/train.py`)
- Watcher mit direktem JPEG-Stream (`PKMAI_WATCHER_STREAM=1`)
- Watcher-Zielrate: `200 FPS`
- Browser-Ansicht: maximal ca. 10–15 visuelle Updates/s, um CPU und I/O zu sparen
- Party-Auswertung im Watcher weniger häufig; bei verfügbarem Umgebungscache wird dieser genutzt

Diese Änderungen sind lokale Arbeitsanpassungen und müssen bei einem zukünftigen `git pull` geprüft bzw. erneut übernommen werden.

## 8. Starten

Drei WSL-Terminals öffnen und jeweils einen Befehl starten.

### Dashboard / API

```bash
cd /mnt/c/zod/pkmai
exec env PYTHONPATH=src /opt/pkmai-venv/bin/python -m uvicorn web_stream:app --host 0.0.0.0 --port 8001 --log-level warning
```

### Training

```bash
cd /mnt/c/zod/pkmai
exec env PYTHONPATH=src /opt/pkmai-venv/bin/python src/train.py
```

### Sichtbarer Watcher

```bash
cd /mnt/c/zod/pkmai
exec env PKMAI_WATCHER_STREAM=1 PYTHONPATH=src /opt/pkmai-venv/bin/python src/watch.py
```

Im Windows-Browser öffnen:

| Zweck | Adresse |
|---|---|
| Dashboard | http://127.0.0.1:8001/ |
| Watcher | http://127.0.0.1:8001/watcher |
| JPEG-Frame | http://127.0.0.1:8001/watcher.jpg |
| API-Status | http://127.0.0.1:8001/api/state |

Die Dienste binden zwar innerhalb von WSL an `0.0.0.0`, werden hier aber nur über `127.0.0.1` auf dem lokalen Windows-Rechner genutzt.

## 9. Kontrolliertes Stoppen und Update

Den Trainer immer mit `Ctrl+C` beenden. Dadurch schreibt er den letzten Checkpoint.

Danach Watcher und Uvicorn ebenfalls mit `Ctrl+C` stoppen. Erst dann aktualisieren:

```bash
cd /mnt/c/zod/pkmai
git status --short
git pull --ff-only
```

Vor einem Pull lokale Änderungen sichern, beispielsweise über einen Git-Stash oder einen eigenen lokalen Patch. Nicht sichern/committen:

- `local/custom_integrations/.../rom.gba`
- `runtime/checkpoints/`
- `runtime/`-Laufzeitdaten
- lokale `.env`-Dateien oder Zugangsdaten

## 10. Typische Probleme

### `No romfiles found for game: PokemonFireRed-Gba`

Die ROM fehlt oder liegt nicht exakt unter:

```text
local/custom_integrations/PokemonFireRed-Gba/rom.gba
```

### Stable-Retro-Build schlägt unter Windows fehl

Das war der Grund für den WSL2-Weg. Stable-Retro nicht erneut nativ auf Windows erzwingen; in Ubuntu/WSL ausführen.

### Trainer oder Watcher melden verschiedene `nav`-Shapes (z. B. 20 vs. 28)

Der gespeicherte Modellcheckpoint passt nicht zur aktuellen `pokemon_env.py`-Observation. Den alten Checkpoint archivieren, statt ihn zu löschen, und mit einem frischen kompatiblen Modell starten – oder den passenden alten Code wiederherstellen.

### Dashboard zeigt zu viele Instanzen

Veraltete Telemetrie liegt unter:

```text
runtime/instances_data/inst_*.json
```

Nach einem Wechsel der Instanzzahl die nicht mehr aktiven `inst_XX.json`-Dateien entfernen. Die Watcher-Datei `inst_99.json` dabei behalten.

---

Erstellt für die lokale Einrichtung unter `C:\zod\pkmai`. Grundlage sind die tatsächlich durchgeführten Schritte der Hermes-Sitzung, einschließlich WSL2, Stable-Retro 1.0.1, lokaler ROM-Integration und direktem HTTP-Watcher.
