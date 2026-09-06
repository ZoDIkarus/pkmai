import os
import tempfile
import unittest

import numpy as np

from watcher_stream import write_watcher_stream_frame


class WatcherStreamModeTests(unittest.TestCase):
    def test_writes_a_jpeg_frame_atomically_to_requested_path(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "watcher.jpg")
            self.assertTrue(write_watcher_stream_frame(frame, target))
            self.assertTrue(os.path.exists(target))
            self.assertGreater(os.path.getsize(target), 0)


if __name__ == "__main__":
    unittest.main()
