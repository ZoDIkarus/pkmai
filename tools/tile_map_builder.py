import os
import cv2
import json
import time
import hashlib
from collections import defaultdict

import numpy as np


class TileMapBuilder:
    """
    Baut fuer jede (map_bank, map_id)-Kombination eine Karte aus den
    16x16-Pixel-Tiles des aktuell sichtbaren GBA-Bildschirms.

    Pro Weltkoordinate werden mehrere beobachtete Varianten gezaehlt.
    Fuer die Ausgabe wird die am haeufigsten beobachtete Variante benutzt.
    So verschwinden Spieler/NPCs/Animationen mit zunehmender Beobachtung
    weitgehend aus der statischen Karte.
    """

    def __init__(self, base_dir, tile_size=16, save_interval=2.0):
        self.base_dir = os.path.expanduser(base_dir)
        self.tile_size = int(tile_size)
        self.save_interval = float(save_interval)
        self.global_save_interval = 12.0
        self.maps_dir = os.path.join(self.base_dir, "stitched_maps")
        os.makedirs(self.maps_dir, exist_ok=True)

        # key -> {(world_x, world_y): {hash: [count, tile_bgr]}}
        self.tiles = defaultdict(dict)

        self.last_save = 0.0
        self.last_global_save = 0.0
        self.global_dirty = False
        self.total_observations = 0

        # Schutz vor einzelnen falschen RAM-Koordinaten, die sonst eine
        # riesige 8k/16k Map erzeugen und den Watcher minutenlang blockieren.
        self.last_player_by_map = {}
        self.max_render_span_tiles = 160

        # Pro Map verfolgen wir den tatsaechlichen Kamera-Ursprung.
        # FireRed haelt die Kamera an Kartenraendern fest; deshalb darf die
        # Weltposition eines Screen-Tiles NICHT dauerhaft aus einer festen
        # Spielerposition (7,5) berechnet werden.
        self.camera_state = {}
        self.last_active_map_key = None
        self.registration_min_improvement = 0.12
        self.registration_max_score = 42.0

        # FireRed gMapGroup_TownsAndRoutes.
        # Nur dort wird die Kamera-/Stitching-Logik benutzt.
        # Indoor-Karten werden bewusst screen-fest gesammelt, damit kleine
        # Raeume nicht durch Kamerafehler mehrfach versetzt werden.
        self.overworld_bank = 3

        # FireRed: 240x160 -> 15x10 Bereiche a 16x16 Pixel
        self.view_w = 15
        self.view_h = 10

        # Nur FALLBACK fuer die allererste Beobachtung einer neuen Map.
        # Danach wird der Kamera-Ursprung per Frame-Registrierung verfolgt.
        self.player_screen_x = 7
        self.player_screen_y = 5

        # Ein eigener Mapper darf nach einem Neustart auf den bereits gebauten
        # Bildern weiterarbeiten. Die gerenderten PNGs werden als einfache
        # Ein-Stimmen-Basis geladen; neue Beobachtungen koennen sie anschliessend
        # wie gewohnt durch Mehrheitsentscheid verbessern.
        self._load_rendered_maps()

    def reset_tracking(self):
        """Vergisst nur Kamera-/Positionszustand, niemals gesammelte Tiles."""
        self.last_player_by_map.clear()
        self.camera_state.clear()
        self.last_active_map_key = None

    def _load_rendered_maps(self):
        try:
            names = os.listdir(self.maps_dir)
        except OSError:
            return

        for name in names:
            if not name.endswith(".json") or name == "index.json":
                continue
            meta_path = os.path.join(self.maps_dir, name)
            image_path = os.path.join(self.maps_dir, name[:-5] + ".png")
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None or meta.get("invalid_extent"):
                    continue
                bank = int(meta["bank"])
                map_id = int(meta["map_id"])
                min_x = int(meta["min_x"])
                min_y = int(meta["min_y"])
                width = int(meta["width_tiles"])
                height = int(meta["height_tiles"])
                if width > self.max_render_span_tiles or height > self.max_render_span_tiles:
                    continue
                key = self._map_key(bank, map_id)
                known = self.tiles[key]
                for ty in range(height):
                    for tx in range(width):
                        tile = image[
                            ty * self.tile_size:(ty + 1) * self.tile_size,
                            tx * self.tile_size:(tx + 1) * self.tile_size,
                        ]
                        if tile.shape[:2] != (self.tile_size, self.tile_size):
                            continue
                        # Schwarzer/leer gerenderter Hintergrund ist kein
                        # beobachtetes Spieltile.
                        empty = np.empty_like(tile)
                        empty[:] = (18, 20, 26)
                        if (
                            float(tile.mean()) < 3.0
                            or float(cv2.absdiff(tile, empty).mean()) < 1.0
                        ):
                            continue
                        coord = (min_x + tx, min_y + ty)
                        known[coord] = {
                            self._hash_tile(tile): [1, tile.copy()]
                        }
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    def _map_key(self, bank, map_id):
        return f"bank_{int(bank):03d}_map_{int(map_id):03d}"

    @staticmethod
    def _hash_tile(tile):
        return hashlib.blake2b(tile.tobytes(), digest_size=8).hexdigest()

    @staticmethod
    def _valid_ram(bank, map_id, x, y):
        bank, map_id, x, y = int(bank), int(map_id), int(x), int(y)

        if bank == 0 and map_id == 0 and x == 0 and y == 0:
            return False

        return 0 <= x < 512 and 0 <= y < 512

    @staticmethod
    def _looks_like_dialog(frame_bgr):
        """
        Grobe Erkennung grosser heller Textboxen im unteren Bildschirmbereich.
        """
        h, w = frame_bgr.shape[:2]
        strip = frame_bgr[int(h * 0.68):h, 4:w - 4]

        if strip.size == 0:
            return False

        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        bright_ratio = float((gray > 205).mean())

        # Dialogboxen in FireRed haben grosse helle Flaechen.
        return bright_ratio > 0.40

    def _registration_score(self, prev_gray, cur_gray, dx_tiles, dy_tiles):
        """
        Bewertet, wie gut der aktuelle Screen zu dem vorherigen passt, wenn
        die Kamera um dx/dy Welt-Tiles gescrollt ist.

        origin_cur = origin_prev + (dx, dy)
        => current(sx,sy) entspricht previous(sx+dx, sy+dy).

        Wir arbeiten auf halbierter Aufloesung, damit das im Watcher praktisch
        keine merkbare Last erzeugt.
        """
        shift_x = int(dx_tiles * self.tile_size // 2)
        shift_y = int(dy_tiles * self.tile_size // 2)
        h, w = cur_gray.shape[:2]

        x0c = max(0, -shift_x)
        x1c = min(w, w - shift_x)
        y0c = max(0, -shift_y)
        y1c = min(h, h - shift_y)

        if x1c - x0c < w // 2 or y1c - y0c < h // 2:
            return 999.0

        cur = cur_gray[y0c:y1c, x0c:x1c]
        prev = prev_gray[
            y0c + shift_y:y1c + shift_y,
            x0c + shift_x:x1c + shift_x
        ]

        if cur.shape != prev.shape or cur.size == 0:
            return 999.0

        diff = cv2.absdiff(cur, prev)

        # Median ist gegen Spieler/NPCs/kleine Animationen robuster als mean.
        return float(np.median(diff))

    def _estimate_camera_delta(self, prev_frame, cur_frame, player_delta):
        """
        Unterscheidet:
          - Spieler bewegt sich, Kamera bleibt stehen (Kartenrand)
          - Kamera scrollt tatsaechlich um ein Tile

        Nur ganze Tile-Shifts werden akzeptiert. Zwischenframes waehrend einer
        Scrollanimation werden bei unsicherer Registrierung nicht zum Verschieben
        der Karte benutzt.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (120, 80), interpolation=cv2.INTER_AREA)
        cur_gray = cv2.resize(cur_gray, (120, 80), interpolation=cv2.INTER_AREA)

        candidates = [(0, 0)]
        pdx, pdy = player_delta

        # Kamera kann pro Player-Step maximal um denselben Welt-Tile-Schritt
        # scrollen. Zusaetzlich testen wir die vier Cardinal-Shifts, falls ein
        # Frame/Read leicht versetzt ankommt.
        for cand in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if cand not in candidates:
                candidates.append(cand)

        scores = {
            cand: self._registration_score(
                prev_gray, cur_gray, cand[0], cand[1]
            )
            for cand in candidates
        }

        best = min(scores, key=scores.get)
        best_score = scores[best]
        zero_score = scores[(0, 0)]

        # Bewegt sich der Spieler gar nicht, darf die Kamera nicht spontan
        # springen. Das verhindert Drift durch Animationen/NPCs.
        if pdx == 0 and pdy == 0:
            return (0, 0), zero_score, True

        # Bei genau einem Player-Tile ist ein Kamera-Shift nur in derselben
        # Achse/Richtung plausibel; alles andere wird verworfen.
        plausible = {(0, 0)}
        if abs(pdx) + abs(pdy) == 1:
            plausible.add((pdx, pdy))
        else:
            # Mehr als ein Tile zwischen Mapper-Frames: konservativ bleiben.
            return (0, 0), zero_score, False

        plausible_best = min(plausible, key=lambda c: scores.get(c, 999.0))
        plausible_score = scores.get(plausible_best, 999.0)

        if plausible_best == (0, 0):
            return (0, 0), plausible_score, True

        improvement = (
            (zero_score - plausible_score) / max(zero_score, 1.0)
        )
        confident = (
            plausible_score <= self.registration_max_score
            and improvement >= self.registration_min_improvement
        )

        if confident:
            return plausible_best, plausible_score, True

        # Unsicherer Scrollframe: lieber Kamera nicht verschieben und diesen
        # Frame spaeter nicht aggressiv als neue Geometrie interpretieren.
        return (0, 0), plausible_score, False

    def add_frame(
        self,
        screen_rgb,
        bank,
        map_id,
        player_x,
        player_y,
        in_battle=0,
        allow_dialogs=False
    ):
        bank = int(bank)
        map_id = int(map_id)
        player_x = int(player_x)
        player_y = int(player_y)
        in_battle = int(in_battle)

        if in_battle:
            return 0

        if not self._valid_ram(bank, map_id, player_x, player_y):
            return 0

        frame = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR)
        frame = frame[:160, :240]

        if frame.shape[0] != 160 or frame.shape[1] != 240:
            return 0

        if (not allow_dialogs) and self._looks_like_dialog(frame):
            return 0

        key = self._map_key(bank, map_id)
        known = self.tiles[key]

        # Innerhalb derselben Map darf die Spielerposition zwischen zwei
        # Mapper-Frames nicht ploetzlich dutzende Tiles springen. Solche Reads
        # sind fast immer RAM-Fehlwerte und wuerden die Map-Ausdehnung sprengen.
        prev_player = self.last_player_by_map.get(key)
        if prev_player is not None:
            dx = abs(player_x - prev_player[0])
            dy = abs(player_y - prev_player[1])
            if dx > 8 or dy > 8:
                return 0

        state = self.camera_state.get(key)

        # INDOOR: absolut konservativ. Kleine Raeume wie das Spielerhaus werden
        # als fester 15x10-Screen gesammelt. Player-X/Y darf den Canvas NICHT
        # verschieben. Dadurch koennen Teppich, Treppe und Moebel nicht mehrfach
        # versetzt in dieselbe Map geschrieben werden.
        #
        # Fuer groessere Indoor-Maps ist das zunaechst absichtlich nur ein
        # stabiler Screen-Ausschnitt statt einer falschen "grossen" Karte.
        if bank != self.overworld_bank:
            origin_x = 0
            origin_y = 0
            registration_confident = True

            # Spielerposition fuer die Sprite-Maske ist indoor nicht verlaesslich
            # aus RAM->Screen ableitbar, wenn die Kamera geklemmt ist. Wir
            # blockieren deshalb zusaetzlich den klassischen Zentralbereich.
            player_screen_x = self.player_screen_x
            player_screen_y = self.player_screen_y

            self.camera_state[key] = {
                "origin_x": 0,
                "origin_y": 0,
                "frame": frame.copy(),
            }
            self.last_player_by_map[key] = (player_x, player_y)
            self.last_active_map_key = key

        else:
            # OVERWORLD: Kamera-Ursprung anhand aufeinanderfolgender Frames
            # verfolgen, weil grosse Aussenkarten tatsaechlich gescrollt werden.
            if state is None:
                origin_x = player_x - self.player_screen_x
                origin_y = player_y - self.player_screen_y
                # Noch NICHT aufnehmen: am Kartenrand steht der Spieler nicht
                # zwingend in der Bildschirmmitte. Erst ein nachgewiesener
                # Kameraschritt kalibriert den absoluten Screen-Ursprung.
                self.camera_state[key] = {
                    "origin_x": int(origin_x),
                    "origin_y": int(origin_y),
                    "frame": frame.copy(),
                    "absolute_confident": False,
                }
                self.last_player_by_map[key] = (player_x, player_y)
                self.last_active_map_key = key
                return 0
            else:
                origin_x = int(state["origin_x"])
                origin_y = int(state["origin_y"])
                absolute_confident = bool(
                    state.get("absolute_confident", False)
                )
                registration_confident = True

                if self.last_active_map_key != key or prev_player is None:
                    # Wiedereintritt in eine Map: alte relative Kamera darf
                    # nicht auf die neue Eintrittsposition uebertragen werden.
                    self.camera_state[key] = {
                        "origin_x": int(player_x - self.player_screen_x),
                        "origin_y": int(player_y - self.player_screen_y),
                        "frame": frame.copy(),
                        "absolute_confident": False,
                    }
                    self.last_player_by_map[key] = (player_x, player_y)
                    self.last_active_map_key = key
                    return 0
                else:
                    pdx = player_x - prev_player[0]
                    pdy = player_y - prev_player[1]
                    cam_delta, _score, registration_confident = (
                        self._estimate_camera_delta(
                            state["frame"], frame, (pdx, pdy)
                        )
                    )
                    # Ein unsicher registrierter Bewegungsframe darf auch
                    # bekannte Tiles nicht ueberschreiben. Vorher sammelte er
                    # dort Varianten mit falschem Ursprung und verschob die
                    # Karte schleichend. Kamera-/Player-Baseline unveraendert
                    # lassen, damit der naechste ruhige Frame erneut prueft.
                    if not registration_confident:
                        return 0
                    if not absolute_confident:
                        if cam_delta == (0, 0):
                            # Noch am geklemmten Kartenrand oder unbewegt.
                            # Baseline aktualisieren, aber nichts raten.
                            self.camera_state[key] = {
                                "origin_x": int(origin_x),
                                "origin_y": int(origin_y),
                                "frame": frame.copy(),
                                "absolute_confident": False,
                            }
                            self.last_player_by_map[key] = (player_x, player_y)
                            self.last_active_map_key = key
                            return 0
                        # Wenn die Kamera wirklich mit dem Spieler scrollt,
                        # sitzt dessen Fuss-Tile stabil bei (7,5). Damit ist
                        # erstmals ein absoluter Ursprung belegbar.
                        origin_x = player_x - self.player_screen_x
                        origin_y = player_y - self.player_screen_y
                        absolute_confident = True
                    else:
                        origin_x += cam_delta[0]
                        origin_y += cam_delta[1]

            player_screen_x = player_x - origin_x
            player_screen_y = player_y - origin_y

            self.camera_state[key] = {
                "origin_x": int(origin_x),
                "origin_y": int(origin_y),
                "frame": frame.copy(),
                "absolute_confident": bool(absolute_confident),
            }
            self.last_player_by_map[key] = (player_x, player_y)
            self.last_active_map_key = key

        blocked = set()
        if (
            0 <= player_screen_x < self.view_w
            and 0 <= player_screen_y < self.view_h
        ):
            blocked.add((player_screen_x, player_screen_y))
            blocked.add((player_screen_x, player_screen_y - 1))

        newly_seen_world_tiles = 0

        for sy in range(self.view_h):
            for sx in range(self.view_w):
                if (sx, sy) in blocked:
                    continue

                px = sx * self.tile_size
                py = sy * self.tile_size

                tile = frame[
                    py:py + self.tile_size,
                    px:px + self.tile_size
                ]

                if tile.shape[:2] != (self.tile_size, self.tile_size):
                    continue

                world_x = origin_x + sx
                world_y = origin_y + sy

                if bank == self.overworld_bank:
                    if not (0 <= world_x < 512 and 0 <= world_y < 512):
                        continue
                else:
                    # Indoor-Karten sind screen-verankert.
                    if not (
                        0 <= world_x < self.view_w
                        and 0 <= world_y < self.view_h
                    ):
                        continue

                coord = (world_x, world_y)

                # Wenn wir mitten in einer Scrollanimation sind und die
                # Registrierung nicht sicher war, duerfen bekannte Tiles noch
                # Stimmen sammeln, aber der Frame darf die Karte NICHT erweitern.
                if coord not in known and not registration_confident:
                    continue

                if coord not in known:
                    known[coord] = {}
                    newly_seen_world_tiles += 1
                    self.global_dirty = True

                tile_hash = self._hash_tile(tile)
                variants = known[coord]

                if tile_hash in variants:
                    variants[tile_hash][0] += 1
                else:
                    variants[tile_hash] = [1, tile.copy()]

                self.total_observations += 1

        now = time.time()

        # Kleine aktuelle Map darf relativ oft gespeichert werden.
        if newly_seen_world_tiles > 0 and now - self.last_save >= self.save_interval:
            self.save_map(bank, map_id)
            self.save_index()
            self.last_save = now

        # Die 3000x3000 Globalmap ist teuer: nur bei Aenderungen und hoechstens
        # alle 12 Sekunden neu schreiben.
        if (
            self.global_dirty
            and now - self.last_global_save >= self.global_save_interval
        ):
            self.save_global_atlas()
            self.last_global_save = now
            self.global_dirty = False

        return newly_seen_world_tiles

    @staticmethod
    def _best_tile(variants):
        if not variants:
            return None, 0

        _, (count, tile) = max(
            variants.items(),
            key=lambda kv: kv[1][0]
        )
        return tile, count

    def _render_map(self, bank, map_id):
        key = self._map_key(bank, map_id)
        known = self.tiles.get(key)

        if not known:
            return None, None

        xs = [coord[0] for coord in known]
        ys = [coord[1] for coord in known]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width_tiles = max_x - min_x + 1
        height_tiles = max_y - min_y + 1

        # Eine einzelne falsche Koordinate darf niemals hunderte MB Canvas
        # erzeugen und damit den Emulator-/Input-Loop blockieren.
        if (
            width_tiles > self.max_render_span_tiles
            or height_tiles > self.max_render_span_tiles
        ):
            return None, {
                "bank": int(bank),
                "map_id": int(map_id),
                "invalid_extent": True,
                "width_tiles": int(width_tiles),
                "height_tiles": int(height_tiles),
                "known_world_tiles": int(len(known)),
            }

        canvas = np.zeros(
            (
                height_tiles * self.tile_size,
                width_tiles * self.tile_size,
                3
            ),
            dtype=np.uint8
        )
        canvas[:] = (18, 20, 26)

        for (wx, wy), variants in known.items():
            tile, _count = self._best_tile(variants)
            if tile is None:
                continue

            px = (wx - min_x) * self.tile_size
            py = (wy - min_y) * self.tile_size

            canvas[
                py:py + self.tile_size,
                px:px + self.tile_size
            ] = tile

        meta = {
            "bank": int(bank),
            "map_id": int(map_id),
            "min_x": int(min_x),
            "max_x": int(max_x),
            "min_y": int(min_y),
            "max_y": int(max_y),
            "known_world_tiles": int(len(known)),
            "width_tiles": int(width_tiles),
            "height_tiles": int(height_tiles),
            "observations": int(sum(
                sum(v[0] for v in variants.values())
                for variants in known.values()
            )),
            "alignment_confident": bool(
                self.camera_state.get(key, {}).get(
                    "absolute_confident", bank != self.overworld_bank
                )
            ),
        }

        return canvas, meta

    def get_preview(self, bank, map_id, width=600, height=500, player_x=None, player_y=None):
        """
        Liefert eine zentrierte Vorschau fuer den Watcher.
        """
        rendered, meta = self._render_map(bank, map_id)

        preview = np.zeros((height, width, 3), dtype=np.uint8)
        preview[:] = (12, 15, 22)

        if rendered is None:
            cv2.putText(
                preview,
                "Noch keine Tiles fuer diese Map",
                (28, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )
            return preview, None

        rh, rw = rendered.shape[:2]
        scale = min(width / max(rw, 1), height / max(rh, 1), 4.0)

        nw = max(1, int(rw * scale))
        nh = max(1, int(rh * scale))

        interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
        resized = cv2.resize(rendered, (nw, nh), interpolation=interp)

        ox = (width - nw) // 2
        oy = (height - nh) // 2
        preview[oy:oy + nh, ox:ox + nw] = resized

        # Aktuelle Spielerposition als kleines Fadenkreuz nur in der Vorschau.
        if (
            player_x is not None and player_y is not None and
            meta is not None
        ):
            rel_x = (int(player_x) - meta["min_x"]) * self.tile_size + self.tile_size // 2
            rel_y = (int(player_y) - meta["min_y"]) * self.tile_size + self.tile_size // 2

            cx = ox + int(rel_x * scale)
            cy = oy + int(rel_y * scale)

            if 0 <= cx < width and 0 <= cy < height:
                cv2.drawMarker(
                    preview,
                    (cx, cy),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=2
                )

        return preview, meta

    def save_map(self, bank, map_id):
        key = self._map_key(bank, map_id)
        canvas, meta = self._render_map(bank, map_id)

        if canvas is None:
            return None

        image_path = os.path.join(self.maps_dir, key + ".png")
        meta_path = os.path.join(self.maps_dir, key + ".json")

        cv2.imwrite(image_path, canvas)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return image_path

    def save_index(self):
        maps = []

        for key, known in self.tiles.items():
            if not known:
                continue

            parts = key.split("_")
            maps.append({
                "key": key,
                "bank": int(parts[1]),
                "map_id": int(parts[3]),
                "known_world_tiles": int(len(known))
            })

        maps.sort(key=lambda item: (item["bank"], item["map_id"]))

        with open(os.path.join(self.maps_dir, "index.json"), "w") as f:
            json.dump({
                "maps": maps,
                "total_observations": int(self.total_observations)
            }, f, indent=2)

    def save_global_atlas(self, output_path=None, canvas_size=3000):
        """
        Schreibt die globale Live-Uebersicht der AUSSENWELT.

        Nur FireRed Bank 3 (gMapGroup_TownsAndRoutes) darf in kanto_map.png.
        Indoor-Raeume, Gebaeude und Dungeons werden weiterhin einzeln unter
        stitched_maps gespeichert, aber niemals in die Hauptkarte gemischt.

        FireRed-Maps besitzen jeweils lokale Koordinaten; echte
        Kanto-Nachbarschaften werden spaeter aus beobachteten
        Map-Uebergaengen zusammengesetzt.

        Der Webstream verwendet ~/pokemon_ai_project/kanto_map.png.
        """
        if output_path is None:
            output_path = os.path.join(self.base_dir, "kanto_map.png")

        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[:] = (12, 15, 22)

        # FireRed map groups:
        # Bank 3 = gMapGroup_TownsAndRoutes (echte Overworld).
        # Bank 4+ beginnt bereits mit IndoorPallet und weiteren
        # Innenraeumen/Dungeons. Diese bleiben als einzelne stitched_maps,
        # duerfen aber NICHT in kanto_map.png landen.
        OVERWORLD_BANK = 3
        keys = sorted(
            key for key in self.tiles.keys()
            if int(key.split("_")[1]) == OVERWORLD_BANK
        )

        if not keys:
            cv2.putText(
                canvas,
                "Kanto Overworld: waiting for first outdoor map (Bank 3) ...",
                (80, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (210, 215, 225),
                2,
                cv2.LINE_AA
            )
            cv2.imwrite(output_path, canvas)
            return output_path

        # 5x5 bis 25 Maps pro Seite; weitere Maps skalieren automatisch.
        cols = 5
        rows = max(1, (len(keys) + cols - 1) // cols)

        margin = 30
        gap = 20
        header_h = 42

        cell_w = (canvas_size - 2 * margin - (cols - 1) * gap) // cols
        cell_h = (canvas_size - 2 * margin - (rows - 1) * gap) // rows
        cell_h = max(140, cell_h)

        for index, key in enumerate(keys):
            parts = key.split("_")
            bank = int(parts[1])
            map_id = int(parts[3])

            rendered, meta = self._render_map(bank, map_id)
            if rendered is None:
                continue

            row = index // cols
            col = index % cols

            x0 = margin + col * (cell_w + gap)
            y0 = margin + row * (cell_h + gap)

            if y0 >= canvas_size - margin:
                break

            box_h = min(cell_h, canvas_size - margin - y0)
            map_area_h = max(1, box_h - header_h)

            rh, rw = rendered.shape[:2]
            scale = min(
                (cell_w - 12) / max(rw, 1),
                (map_area_h - 12) / max(rh, 1),
                4.0
            )

            nw = max(1, int(rw * scale))
            nh = max(1, int(rh * scale))

            interp = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
            resized = cv2.resize(rendered, (nw, nh), interpolation=interp)

            cv2.rectangle(
                canvas,
                (x0, y0),
                (x0 + cell_w - 1, y0 + box_h - 1),
                (55, 64, 82),
                1
            )

            label = (
                f"Bank {bank} / Map {map_id}  |  "
                f"{meta['known_world_tiles']} tiles"
            )
            cv2.putText(
                canvas,
                label,
                (x0 + 8, y0 + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 235, 245),
                1,
                cv2.LINE_AA
            )

            px = x0 + (cell_w - nw) // 2
            py = y0 + header_h + max(0, (map_area_h - nh) // 2)

            y1 = min(py + nh, canvas_size)
            x1 = min(px + nw, canvas_size)

            canvas[py:y1, px:x1] = resized[:y1 - py, :x1 - px]

        cv2.putText(
            canvas,
            f"Kanto Overworld - {len(keys)} discovered outdoor maps",
            (35, canvas_size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 118),
            1,
            cv2.LINE_AA
        )

        tmp_path = output_path + ".tmp.png"
        cv2.imwrite(tmp_path, canvas)
        os.replace(tmp_path, output_path)

        return output_path

    def save_all(self):
        for key in list(self.tiles.keys()):
            parts = key.split("_")
            self.save_map(int(parts[1]), int(parts[3]))

        self.save_index()
        self.save_global_atlas()
