import ast
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from watcher_display import FramePacer, FramePublisher
import numpy as np
import cv2
import web_stream


class DisplayTests(unittest.TestCase):
    def test_pacing_deadlines_and_no_catchup_burst(self):
        now = [0.0]
        def sleep(delay):
            now[0] += delay
        pacer = FramePacer(60, lambda: now[0], sleep)
        pacer.wait()
        now[0] += .005
        pacer.wait()
        self.assertAlmostEqual(now[0], 1 / 60)
        now[0] = 2
        pacer.wait()
        self.assertEqual(now[0], 2)
        pacer.wait()
        self.assertAlmostEqual(now[0], 2 + 1 / 60)

    def test_training_and_watcher_keep_identical_nine_five_inputs(self):
        # Execute the actual action portion without ROM/reward dependencies.
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        cutoff = next(i for i, n in enumerate(step.body) if isinstance(n, ast.AugAssign))
        step.body = step.body[:cutoff]
        ns = {}
        exec(compile(ast.Module(body=[step], type_ignores=[]), '<action>', 'exec'), ns)
        for visible in (False, True):
            buttons, shown = [], []
            env = SimpleNamespace(ACTION_HOLD_FRAMES=9, ACTION_RELEASE_FRAMES=5,
                                  action_map={3: 'UP'}, btn_none='NONE',
                                  env=SimpleNamespace(step=buttons.append))
            if visible:
                env.frame_callback = lambda: shown.append(buttons[-1])
            ns['step'](env, 3)
            self.assertEqual(buttons, ['UP'] * 9 + ['NONE'] * 5)
            self.assertEqual(shown, buttons if visible else [])

    def test_background_publisher_and_multipart_payload(self):
        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                publisher = FramePublisher(tmp)
                try:
                    publisher.submit(np.zeros((570, 980, 3), dtype=np.uint8),
                                     np.zeros((160, 240, 3), dtype=np.uint8))
                    with patch.object(web_stream, 'WATCHER_FRAME_FILE', str(Path(tmp) / 'watcher.jpg')):
                        stream = web_stream.watcher_mjpeg_frames()
                        part = await asyncio.wait_for(anext(stream), timeout=3)
                        await stream.aclose()
                    header, jpeg = part.split(b'\r\n\r\n', 1)
                    self.assertIn(b'Content-Type: image/jpeg', header)
                    decoded = cv2.imdecode(np.frombuffer(jpeg[:-2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(decoded.shape, (570, 980, 3))
                finally:
                    publisher.close()
        asyncio.run(check())
