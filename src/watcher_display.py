"""Frame pacing and bounded background JPEG publication for the watcher."""
import logging
from pathlib import Path
import threading
import time

import cv2


class FramePacer:
    def __init__(self, fps, clock=time.perf_counter, sleep=time.sleep):
        self.period = 1.0 / fps
        self.clock, self.sleep = clock, sleep
        self.deadline = None

    def wait(self):
        now = self.clock()
        if self.deadline is None or now - self.deadline > self.period:
            # No burst of catch-up frames after model loading or a reset.
            self.deadline = now
        if self.deadline > now:
            self.sleep(self.deadline - now)
        self.deadline += self.period


class FramePublisher:
    """One pending frame only: a slow consumer never builds a video backlog."""
    def __init__(self, root):
        self.root = Path(root)
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='watcher-jpeg', daemon=True)
        self.thread.start()

    def submit(self, canvas, screen):
        with self.condition:
            self.pending = (canvas.copy(), screen.copy())
            self.condition.notify()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=2)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                canvas, screen = self.pending
                self.pending = None
            try:
                for name, frame in (
                    ('watcher', canvas),
                    ('watcher_emulator', cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)),
                ):
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        tmp = self.root / (name + '.stream.tmp')
                        tmp.write_bytes(encoded.tobytes())
                        tmp.replace(self.root / (name + '.jpg'))
            except OSError:
                logging.getLogger(__name__).exception('Watcher frame publication failed')
