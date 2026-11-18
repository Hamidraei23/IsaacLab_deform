"""A window showing a camera's frames as Depth-Anything-3 metric depth (``play_actor.py --camera d435i --depth``).

The model runs in the ``da3-d435i`` container (``camera_ws/da3_d435i_ros2/run_da3_sim_server.sh``), which serves
depth over TCP (see its ``da3_server.py`` for the byte protocol). This side sends the latest colour frame from a
background thread, one at a time, and drops frames while a request is in flight, so the simulation never waits for
the network: at about 60 ms a frame the depth window runs a few frames behind the colour one.

The depth is coloured as ``da3_d435i_view.py`` colours it: TURBO, inverted so near is warm, over the 2nd-98th
percentile of the frame, smoothed over time so the scale does not flicker. The window is the frame's own size.
"""

import socket
import struct
import threading

import cv2
import numpy as np

from camera_window import CameraWindow

REQUEST = struct.Struct("!IIIIff")
REPLY = struct.Struct("!IIf")


def _recv_exact(conn, size):
    chunks, remaining = [], size
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class DepthAnythingClient:
    """One request at a time to ``da3_server.py``: ``depth(rgb, fx, fy)`` returns float32 metres, the frame's size."""

    def __init__(self, address="127.0.0.1:5555", timeout=30.0):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.last_ms = 0.0

    def depth(self, rgb: np.ndarray, fx: float, fy: float) -> np.ndarray:
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        self.sock.sendall(REQUEST.pack(w, h, 3, rgb.nbytes, float(fx), float(fy)) + rgb.tobytes())
        rh, rw, self.last_ms = REPLY.unpack(_recv_exact(self.sock, REPLY.size))
        return np.frombuffer(_recv_exact(self.sock, rh * rw * 4), dtype=np.float32).reshape(rh, rw)

    def close(self):
        self.sock.close()


def colorize(depth: np.ndarray, smoothed: list) -> np.ndarray:
    """RGB uint8 image of a depth map, da3_d435i_view.py's scale: TURBO, near warm, 2-98 % percentiles smoothed."""
    sample = depth[::4, ::4]
    sample = sample[np.isfinite(sample) & (sample > 0.0)]
    low, high = (np.percentile(sample, (2.0, 98.0)) if sample.size else (0.0, 1.0))
    if smoothed:
        low, high = 0.8 * smoothed[0] + 0.2 * low, 0.8 * smoothed[1] + 0.2 * high
    smoothed[:] = [float(low), float(high)]
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bgr[~np.isfinite(depth) | (depth <= 0.0)] = (0, 0, 0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DepthAnythingWindow:
    """Feed it colour frames with ``submit``; call ``show`` every step to draw the latest depth that has come back.

    ``submit`` takes a ``meta`` object that travels with the frame (the tag pose and the true depth of that very
    frame, in ``play_actor.py``); ``take`` hands back each new (depth, meta) once, for the caller to correct and
    measure on its own thread; ``show`` with no image colours the latest raw depth, with an image shows that.
    """

    def __init__(self, title, width, height, fx, fy, address="127.0.0.1:5555"):
        self.client = DepthAnythingClient(address)
        self.window = CameraWindow(title, width, height)
        self.fx, self.fy = fx, fy
        self.smoothed = []
        self.pending = None  # the newest (frame, meta) not yet sent; older ones are dropped
        self.latest = None  # the newest (depth, meta) back from the server
        self.taken = 0  # serial of the result the caller last took
        self.image = None  # the last image shown
        self.error = None
        self.lock = threading.Condition()
        self.alive = True
        self.sent = self.received = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            with self.lock:
                while self.pending is None and self.alive:
                    self.lock.wait()
                if not self.alive:
                    return
                (frame, meta), self.pending = self.pending, None
            try:
                depth = self.client.depth(frame, self.fx, self.fy)
            except (OSError, ValueError, struct.error) as exc:
                with self.lock:
                    self.error = exc
                    self.alive = False
                return
            with self.lock:
                self.latest, self.received = (depth, meta), self.received + 1

    def submit(self, rgb: np.ndarray, meta=None) -> None:
        with self.lock:
            self.pending = (np.array(rgb[..., :3], copy=True), meta)
            self.sent += 1
            self.lock.notify()

    def take(self):
        """The newest (depth [m] float32, meta) not yet taken, or None. Raises the worker's error if the server went."""
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.latest is None or self.taken == self.received:
                return None
            self.taken = self.received
            return self.latest

    def show(self, image=None) -> bool:
        """Draw ``image``, or the latest raw depth coloured. False once the window is closed or the server is gone."""
        if image is None:
            try:
                result = self.take()
            except (OSError, ValueError, struct.error) as exc:
                print(f"[da3] depth server lost: {exc}", flush=True)
                return False
            if result is not None:
                image = colorize(result[0], self.smoothed)
        if image is not None:
            self.image = image
        if self.image is None:
            return self.window.open
        return self.window.show(self.image)

    def close(self):
        with self.lock:
            self.alive = False
            self.lock.notify()
        self.window.close()
        self.client.close()
