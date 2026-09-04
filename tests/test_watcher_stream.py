import os
import tempfile
import unittest
from unittest.mock import patch

import firered_ram
import web_stream


class WatcherJpegEndpointTests(unittest.TestCase):
    def test_watcher_jpeg_returns_snapshot_with_no_store_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watcher.jpg")
            with open(path, "wb") as f:
                f.write(b"\xff\xd8fake-jpeg\xff\xd9")

            old_path = web_stream.WATCHER_FRAME_FILE
            try:
                web_stream.WATCHER_FRAME_FILE = path
                response = web_stream.get_watcher_jpeg()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.media_type, "image/jpeg")
                self.assertEqual(
                    response.headers["cache-control"],
                    "no-cache, no-store, must-revalidate",
                )
            finally:
                web_stream.WATCHER_FRAME_FILE = old_path

    def test_dashboard_has_a_dedicated_watcher_navigation_tab(self):
        html = web_stream.index()
        self.assertIn("showTab('watcher', event)", html)
        self.assertIn('id="watcher-view"', html)
        self.assertIn('id="watcher-stream"', html)
        self.assertIn('/watcher.jpg', html)

    def test_dashboard_has_a_separate_fleet_status_tab(self):
        html = web_stream.index()
        self.assertIn("showTab('status', event)", html)
        self.assertIn('id="status-view"', html)
        self.assertIn('id="status-role-grid"', html)
        self.assertIn('id="status-agent-grid"', html)
        self.assertIn('Champion: beste Full-Steps', html)
        self.assertIn('Ø Ep-Steps', html)

    def test_external_leaflet_script_is_closed_before_inline_code(self):
        html = web_stream.index()
        self.assertIn('leaflet.js"></script>\n    <script>', html)

    def test_selected_client_owns_the_party_header(self):
        html = web_stream.index()
        self.assertIn('const headInst = inst;', html)
        self.assertNotIn('const headInst = latestInstances.find', html)

    def test_web_version_falls_back_to_champion_before_first_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "model_version.json")
            champion = os.path.join(tmp, "champion.json")
            with open(champion, "w") as f:
                import json
                json.dump({"version": 3, "timesteps": 12}, f)
            with (
                patch.object(web_stream, "VERSION_FILE", missing),
                patch.object(web_stream, "CHAMPION_FILE", champion),
            ):
                self.assertEqual(web_stream._load_version_meta()["version"], 3)


class PartyTelemetryValidationTests(unittest.TestCase):
    def test_confirmed_battle_flags_are_read_as_uint32(self):
        class FakeEnv:
            def get_ram(self):
                ram = bytearray(firered_ram.BATTLE_TYPE_FLAGS_OFFSET + 4)
                ram[firered_ram.BATTLE_TYPE_FLAGS_OFFSET:
                    firered_ram.BATTLE_TYPE_FLAGS_OFFSET + 4] = (
                        0x12345678
                    ).to_bytes(4, "little")
                return ram

        self.assertEqual(
            firered_ram.read_battle_type_flags(FakeEnv()),
            0x12345678,
        )

    def test_invalid_checksum_cannot_appear_as_bulbasaur(self):
        class FakeEnv:
            def get_ram(self):
                return bytearray(
                    firered_ram.PLAYER_PARTY_OFFSET
                    + firered_ram.POKEMON_STRUCT_SIZE * 6
                )

        invalid = {
            "id": 1, "species_id": 1, "name": "Bulbasaur",
            "level": 5, "cur_hp": 20, "max_hp": 20,
            "checksum_ok": False,
        }
        valid = dict(invalid, id=4, species_id=4, name="Charmander",
                     checksum_ok=True)
        with patch(
            "firered_ram._decode_party_mon",
            side_effect=[invalid, valid, None, None, None, None],
        ):
            self.assertEqual(firered_ram.read_player_party(FakeEnv()), [valid])


if __name__ == "__main__":
    unittest.main()
