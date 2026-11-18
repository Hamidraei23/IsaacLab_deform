"""A coloured point cloud from the tag-corrected Depth-Anything-3 depth (``play_actor.py --depth --marker --point_cloud``).

Every pixel of the depth map (the corrected one when tags are visible, see ``depth_scale.py``) is back-projected
through the camera's pinhole intrinsics (``backproject``: x right, y down, z forward, metres) and coloured from the
RGB frame; points farther than ``max_range`` from the camera (2 m by default: the table and workspace, not the far
floor) and invalid pixels are dropped, and so are depth edges (``edge_mask``: pixels whose depth differs from a
neighbour by more than ``edge_m``), where a monocular model smears foreground and background together and would
otherwise draw streaks from the arm down to the table. When tag 0 is in the frame the cloud is moved into the tag's frame
(``to_tag_frame``): x, y along the tag's edges on the table, z its normal, so z is height above the table and the
cloud is a top-down map of the cell in table coordinates; without the tag it stays in the camera frame.

``PointCloudWindow`` shows the live cloud in an Open3D window (mouse: rotate, scroll: zoom; s saves a PLY to the
window's folder; closing it stops the play loop) and ``write_ply`` writes a cloud any viewer opens (Open3D,
MeshLab, CloudCompare). The window lives in its own process (``CloudProcess`` starts ``python point_cloud.py
--connect`` and streams clouds to it over a local socket, newest first, dropping the rest): an Open3D window in the
playing process took the GL context away from the Newton GL viewer, which then showed only its blue sky. The
child saves the last cloud when the play ends and exits; closing its window stops the play loop.
"""

import argparse
import socket
import struct
import subprocess
import sys
import threading

import os
import time

import cv2
import numpy as np

HEADER = struct.Struct("!IB")
"""Per cloud: number of points, frame (0 camera, 1 tag); then n * 3 float32 xyz and n * 3 uint8 rgb."""
FRAMES = ("camera", "tag")

DEFAULT_MAX_RANGE = 2.0
"""Points farther than this from the camera [m] are left out: the table and workspace lie within 2 m."""
DEFAULT_EDGE_M = 0.03
"""A pixel whose depth differs from a 4-neighbour by more than this [m] is a depth edge and is left out."""


def edge_mask(depth, edge_m=DEFAULT_EDGE_M):
    """Boolean (H, W) mask of the pixels that are NOT on a depth edge (a jump larger than ``edge_m`` to any of the
    four neighbours); the image border counts as no jump. 0 = no filtering."""
    if not edge_m or edge_m <= 0.0:
        return np.ones(depth.shape, dtype=bool)
    d = np.where(np.isfinite(depth), depth, 0.0).astype(np.float64)
    jump = np.zeros(d.shape, dtype=bool)
    dx, dy = np.abs(np.diff(d, axis=1)) > edge_m, np.abs(np.diff(d, axis=0)) > edge_m
    jump[:, 1:] |= dx
    jump[:, :-1] |= dx
    jump[1:, :] |= dy
    jump[:-1, :] |= dy
    return ~jump


def backproject(depth, fx, fy, cx, cy):
    """(H, W, 3) camera-frame points [m] of a depth map (z forward, x right, y down); invalid pixels give z = 0."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float64)
    return np.dstack(((u - cx) / fx * z, (v - cy) / fy * z, z))


def cloud_from_depth(depth, rgb, fx, fy, cx, cy, max_range=DEFAULT_MAX_RANGE, stride=1, edge_m=DEFAULT_EDGE_M):
    """(points (N, 3) [m] in the camera frame, colours (N, 3) in 0..1) of the valid pixels within ``max_range`` of
    the camera that are not depth edges (``edge_m``), every ``stride``-th row and column."""
    keep = edge_mask(depth, edge_m)[::stride, ::stride]
    depth = depth[::stride, ::stride]
    rgb = rgb[::stride, ::stride, :3]
    points = backproject(depth, fx / stride, fy / stride, cx / stride, cy / stride)
    keep &= (points[..., 2] > 0.0) & (np.linalg.norm(points, axis=-1) <= max_range)
    return points[keep], rgb[keep].astype(np.float64) / 255.0


def to_tag_frame(points, rvec, tvec):
    """Camera-frame points moved into the tag's frame: the solve gives p_cam = R p_tag + t, so p_tag = R^T (p_cam - t).
    For a tag flat on the table, z is then height above the table."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return (np.asarray(points, dtype=np.float64) - np.asarray(tvec, dtype=np.float64).reshape(3)) @ R


def write_ply(path, points, colors):
    """Binary little-endian PLY with xyz (float32) and rgb (uint8)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rows = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
    rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())
    return path


def read_ply(path):
    """(points (N, 3), colours (N, 3) in 0..1) of a PLY written by ``write_ply``."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        count = int([line for line in header.decode("ascii").splitlines() if line.startswith("element vertex")][0].split()[-1])
        rows = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1")], count=count)
    points = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    colors = np.stack([rows["red"], rows["green"], rows["blue"]], axis=1).astype(np.float64) / 255.0
    return points, colors


class PointCloudWindow:
    """A live Open3D view of a point cloud: ``show(points, colors, frame)`` replaces the cloud (frame: "tag" or
    "camera", written to the saved files' names) and returns False once the window is closed; s saves a PLY into
    ``save_dir``; ``close`` destroys the window."""

    def __init__(self, title, save_dir, width=960, height=720):
        import open3d as o3d

        self.o3d = o3d
        self.save_dir = save_dir
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(title, width, height):
            raise RuntimeError("Open3D could not open a window (is DISPLAY set?)")
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().background_color = np.array([0.1, 0.1, 0.12])
        self.cloud = o3d.geometry.PointCloud()
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.added = False
        self.frame = None
        self.open = True
        self.saved = []
        self.vis.register_key_callback(ord("S"), self._save)

    def _save(self, _vis=None):
        if not self.cloud.has_points():
            return False
        path = os.path.join(self.save_dir, f"cloud_{time.strftime('%Y%m%d_%H%M%S')}_{self.frame or 'camera'}_frame.ply")
        write_ply(path, np.asarray(self.cloud.points), np.asarray(self.cloud.colors))
        self.saved.append(path)
        print(f"[point_cloud] saved {len(self.cloud.points)} points to {path}", flush=True)
        return False

    def save(self, path):
        write_ply(path, np.asarray(self.cloud.points), np.asarray(self.cloud.colors))
        self.saved.append(path)
        return path

    def show(self, points, colors, frame="camera"):
        if not self.open:
            return False
        o3d = self.o3d
        self.cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        self.cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        if not self.added:
            self.vis.add_geometry(self.cloud)
            self.vis.add_geometry(self.axes)
            self.added = True
        else:
            self.vis.update_geometry(self.cloud)
        if frame != self.frame:  # the frame changed: look at the cloud afresh, z up in the tag frame
            self.frame = frame
            self.vis.reset_view_point(True)
            view = self.vis.get_view_control()
            if frame == "tag":
                view.set_up([0.0, 0.0, 1.0])
                view.set_front([1.0, -0.6, 0.8])
            else:
                view.set_up([0.0, -1.0, 0.0])
                view.set_front([0.0, 0.0, -1.0])
        if not self.vis.poll_events():
            self.open = False
            return False
        self.vis.update_renderer()
        return True

    def close(self):
        if self.open:
            self.open = False
        self.vis.destroy_window()


def encode_cloud(points, colors, frame):
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return HEADER.pack(len(points), FRAMES.index(frame)) + points.tobytes() + np.ascontiguousarray(rgb).tobytes()


def _recv_exact(conn, size):
    chunks, remaining = [], size
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_cloud(conn):
    """(points (N, 3), colours (N, 3) in 0..1, frame) from the socket, or None once it is closed."""
    head = _recv_exact(conn, HEADER.size)
    if head is None:
        return None
    n, frame = HEADER.unpack(head)
    xyz = _recv_exact(conn, n * 12)
    rgb = _recv_exact(conn, n * 3)
    if xyz is None or rgb is None:
        return None
    return (np.frombuffer(xyz, dtype=np.float32).reshape(n, 3).astype(np.float64),
            np.frombuffer(rgb, dtype=np.uint8).reshape(n, 3).astype(np.float64) / 255.0, FRAMES[frame])


def serve(conn, window, save_dir):
    """The child's loop: show every cloud the socket brings until it closes or the window is closed; then save the
    last cloud to ``save_dir/cloud_last_<frame>_frame.ply``. Returns the path, or None with no cloud."""
    shown = 0
    while True:
        cloud = recv_cloud(conn)
        if cloud is None:
            break
        shown += 1
        if not window.show(*cloud):
            break
    path = None
    if window.cloud.has_points():
        path = window.save(os.path.join(save_dir, f"cloud_last_{window.frame}_frame.ply"))
        print(f"[point_cloud] last cloud ({len(window.cloud.points)} points, {shown} shown) saved to {path}; "
              f"{len(window.saved) - 1} saved with s", flush=True)
    return path


class CloudProcess:
    """The play side: starts the window process and streams clouds to it. ``show(points, colors, frame)`` keeps the
    newest cloud for the sender thread (older ones are dropped while a send is in progress) and returns False once
    the child has gone (its window closed); ``close`` ends the stream, which makes the child save and exit."""

    def __init__(self, title, save_dir, host="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, 0))
        self.sock.listen(1)
        port = self.sock.getsockname()[1]
        self.child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--connect", f"{host}:{port}",
                                       "--save_dir", save_dir, "--title", title])
        self.sock.settimeout(30.0)
        self.conn, _ = self.sock.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Condition()
        self.pending = None
        self.alive = True
        self.sent = 0
        threading.Thread(target=self._sender, daemon=True).start()

    def _sender(self):
        while True:
            with self.lock:
                while self.pending is None and self.alive:
                    self.lock.wait()
                if not self.alive:
                    return
                payload, self.pending = self.pending, None
            try:
                self.conn.sendall(payload)
            except OSError:
                with self.lock:
                    self.alive = False
                return
            self.sent += 1

    def show(self, points, colors, frame="camera"):
        if self.child.poll() is not None:
            with self.lock:
                self.alive = False
        with self.lock:
            if not self.alive:
                return False
            self.pending = encode_cloud(points, colors, frame)
            self.lock.notify()
        return True

    def close(self, timeout=15.0):
        with self.lock:
            self.alive = False
            self.lock.notify()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()
        self.sock.close()
        try:
            self.child.wait(timeout)
        except subprocess.TimeoutExpired:
            self.child.kill()


def main():
    parser = argparse.ArgumentParser(description="The point cloud window process (started by CloudProcess).")
    parser.add_argument("--connect", required=True, help="host:port of the play process's cloud socket.")
    parser.add_argument("--save_dir", default="logs/point_clouds")
    parser.add_argument("--title", default="DA3 point cloud")
    args = parser.parse_args()
    host, port = args.connect.rsplit(":", 1)
    conn = socket.create_connection((host, int(port)))
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    window = PointCloudWindow(args.title, args.save_dir)
    try:
        serve(conn, window, args.save_dir)
    finally:
        conn.close()
        window.close()


if __name__ == "__main__":
    main()
