"""The camera window's frames, served to other programs over a local TCP socket (``play_actor.py --camera_port``).

``FrameServer`` sits in the playing process: ``publish`` hands it each frame the colour window shows, and every
connected client gets the newest frame as soon as it is free to take one (a slow client skips frames; nothing ever
waits for it). ``frames(address)`` is the client side, a generator of (H, W, 3) uint8 RGB arrays, used by
``scripts/qr_distance.py``, which so sees exactly what the RGB camera window shows and nothing else of the
simulation. Protocol: per frame a header ``!IIII`` = height, width, channels, payload bytes, then the raw bytes.
"""

import socket
import struct
import threading

import numpy as np

HEADER = struct.Struct("!IIII")


class FrameServer:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(4)
        self.address = f"{host}:{self.sock.getsockname()[1]}"
        self.lock = threading.Condition()
        self.frame, self.serial, self.alive = None, 0, True
        self.clients = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def publish(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frame, self.serial = np.ascontiguousarray(frame[..., :3], dtype=np.uint8), self.serial + 1
            self.lock.notify_all()

    def _accept(self):
        while self.alive:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        sent = 0
        with self.lock:
            self.clients += 1
        try:
            while self.alive:
                with self.lock:
                    while self.alive and self.serial == sent:
                        self.lock.wait()
                    if not self.alive:
                        return
                    frame, sent = self.frame, self.serial
                h, w, c = frame.shape
                conn.sendall(HEADER.pack(h, w, c, frame.nbytes) + frame.tobytes())
        except OSError:
            pass
        finally:
            conn.close()
            with self.lock:
                self.clients -= 1

    def close(self):
        with self.lock:
            self.alive = False
            self.lock.notify_all()
        self.sock.close()


def frames(address: str = "127.0.0.1:5556", timeout: float = 30.0):
    """Connect to a ``FrameServer`` and yield its frames as (H, W, 3) uint8 RGB arrays until it goes away."""
    host, port = address.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as conn:
        def read(size):
            chunks, remaining = [], size
            while remaining:
                chunk = conn.recv(min(remaining, 1 << 20))
                if not chunk:
                    raise ConnectionError("frame server closed")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        while True:
            try:
                h, w, c, nbytes = HEADER.unpack(read(HEADER.size))
                yield np.frombuffer(read(nbytes), dtype=np.uint8).reshape(h, w, c)
            except ConnectionError:
                return
