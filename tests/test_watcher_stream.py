import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
