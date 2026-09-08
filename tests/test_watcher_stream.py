import os
import tempfile
import unittest
from unittest.mock import patch

import firered_ram
import web_stream


class WatcherJpegEndpointTests(unittest.TestCase):
    def test_confirmed_graph_summary_excludes_legacy_unknown_edges(self):
        graph = {
            "edges": {
                "3,19": [
                    [[1, 1], 4, {"kind": "walk", "confidence": 2, "legacy": False}],
                    [[1, 2], 4, {"kind": "unknown", "confidence": 0, "legacy": True}],
                ],
                "3,20": [
                    [[2, 2], 6, {"kind": "jump_or_ledge", "confidence": 1, "legacy": False}],
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "movement_graph_v1.json")
            with open(path, "w") as f:
                import json
                json.dump(graph, f)
            with patch.object(web_stream, "MOVEMENT_GRAPH_FILE", path):
                self.assertEqual(
                    web_stream._confirmed_movement_graph_summary(),
                    {"confirmed_edges": 2, "confirmed_maps": 2},
                )

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

    def test_emulator_endpoint_serves_only_the_separate_game_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "watcher_emulator.jpg")
            with open(path, "wb") as f:
                f.write(b"game-only-frame")
            with patch.object(web_stream, "RUNTIME_DIR", tmp):
                response = web_stream.get_watcher_emulator_jpeg()
            self.assertEqual(response.body, b"game-only-frame")
            self.assertIn("no-store", response.headers["cache-control"])

    def test_legacy_indoor_mapping_is_not_a_dashboard_tab(self):
        html = web_stream.index()
        self.assertNotIn("🏠 Indoor Mapping", html)
        self.assertNotIn('id="rooms-view"', html)
        self.assertIn("🗺️ Overview", html)

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
        self.assertIn('id="status-agent-grid"', html)
        self.assertIn('Navigation Brain · Battle Brain · 40 FULL Agents', html)
        self.assertNotIn('🧩 Kategorien – aktuelle Episoden', html)

    def test_dashboard_uses_live_twoby2_brain_names(self):
        html = web_stream.index()
        self.assertIn('NAVIGATION BRAIN · LIVE', html)
        self.assertIn('NAVIGATION CHAMPION', html)
        self.assertIn('BATTLE BRAIN · LIVE', html)
        self.assertNotIn('FRONTIER CHAMPION', html)
        self.assertNotIn('Full-Brain Retention', html)

    def test_dashboard_replaces_legacy_global_ai_with_compact_system_pulse(self):
        html = web_stream.index()
        self.assertIn('id="system-pulse"', html)
        self.assertIn('Nav-Prüfung', html)
        self.assertIn('Fluchten / Timeouts', html)
        self.assertNotIn('GLOBAL AI', html)
        self.assertNotIn('id="live-warps"', html)
        self.assertNotIn('id="live-edges"', html)
        self.assertNotIn('id="live-maps"', html)

    def test_learning_tab_uses_current_geographic_and_battle_metrics(self):
        html = web_stream.index()
        self.assertIn('Exploration Growth', html)
        self.assertIn('gerichteten Bewegungsgraphen', html)
        self.assertIn("label:'Bestätigte Kanten'", html)
        self.assertIn("label:'Graph-Maps'", html)
        self.assertIn('const rewardHist=hist.slice(runStart)', html)
        self.assertIn("label:'Stage'", html)
        self.assertNotIn('FULL Journey</div><div class="graph-sub">', html)
        self.assertNotIn("label:'Full Intro'", html)
        self.assertNotIn('Bestes Party-Level', html)

    def test_battle_agents_use_the_battler_asset_on_scenario_anchors(self):
        html = web_stream.index()
        self.assertIn('renderBattleFightersOnMap', html)
        self.assertIn('BATTLE_AREA_ANCHORS', html)
        self.assertIn('for(const area of areas)', html)
        self.assertIn('Battle-Training', html)
        self.assertNotIn('for(let i=0;i<count;i++)', html)
        self.assertIn('src="/battler.png"', html)
        self.assertIn('class="battle-map-icon"', html)

    def test_agent_detail_and_filters_have_no_legacy_mapping_or_roles(self):
        html = web_stream.index()
        self.assertNotIn('Known Edges', html)
        self.assertNotIn('Known Maps', html)
        self.assertNotIn('<div class="k">Transitions</div>', html)
        self.assertNotIn('id="af-role"', html)
        self.assertNotIn('id="af-starter"', html)
        self.assertIn('Geografische Stage', html)
        self.assertIn('Erkundete Tiles', html)

    def test_progress_card_uses_real_first_badge_sprite(self):
        html = web_stream.index()
        self.assertIn('Nächster Orden: Felsorden', html)
        self.assertIn('Kanto-Orden in richtiger Reihenfolge', html)

    def test_header_is_grouped_and_has_only_four_marker_buttons(self):
        html = web_stream.index()
        self.assertIn('Pokémon-Team<small>anklicken für Details</small>', html)
        self.assertIn('Kanto-Orden · 1 bis 8', html)
        self.assertIn('Agents anzeigen', html)
        self.assertIn('badge-group"><button id="language-toggle"', html)
        self.assertNotIn('<body>\n        <button id="language-toggle"', html)
        self.assertNotIn('id="agent-limit"', html)

    def test_status_agent_keys_live_in_a_quarter_height_scrollbox(self):
        html = web_stream.index()
        self.assertIn('max-height:25vh; overflow-y:auto', html)
        self.assertIn("const NAV_KEY_NAMES = ['A','B','START','↑','↓','←','→'];", html)
        self.assertIn("NAV_KEY_NAMES[Number(i.effective_action)]", html)

    def test_status_groups_navigation_before_battle_without_update_clocks(self):
        html = web_stream.index()
        self.assertIn('status-brain-group nav', html)
        self.assertIn('status-brain-group battle', html)
        self.assertLess(html.index('🧠 Navigation Brain <span>'),
                        html.index('⚔️ Battle Brain <span>'))
        self.assertNotIn('Battle zuletzt aktualisiert', html)

    def test_language_layer_cannot_restore_retired_role_ui(self):
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'assets', 'ui', 'dashboard-language.js',
        )
        with open(script_path, encoding='utf-8') as handle:
            script = handle.read()
        self.assertNotIn('Alle Rollen', script)
        self.assertNotIn("getElementById('af-role')", script)
        self.assertNotIn('Bridge: unsicheren Übergang', script)
        self.assertNotIn('Retention: gelernte Übergänge', script)
        self.assertNotIn("['GLOBAL AI'", script)
        self.assertNotIn("['Indoor Mapping'", script)

    def test_badges_are_versioned_and_not_served_with_stale_cache(self):
        html = web_stream.index()
        self.assertIn('/badges/1.png?v=kanto2', html)
        response = web_stream.get_badge_png(1)
        self.assertIn('no-store', response.headers['cache-control'])
        self.assertTrue(response.path.endswith('01_boulder_badge.png'))

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
