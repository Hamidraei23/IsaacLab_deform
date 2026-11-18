"""Place the floor tag (id 2) of the UR16e camera task where the arm never crosses the camera's view of it.

Run the camera task in one terminal (``play_actor.py --camera d435i ...`` serves its frames on 127.0.0.1:5556) and
this GUI in another. Over the live camera frame it draws where a tag of the chosen size at the chosen floor
position would appear, projected with the camera's pose and intrinsics (``d435i_camera.py``), together with the
table top's outline, and tells you whether the whole printed quad is inside the frame and clear of the table's
block, how big the black square is in pixels, how steeply it is seen, and, while episodes play, in what share of
frames that patch of the image stayed undisturbed by the arm (reset whenever you move the tag). Move it with the
sliders, the entries or the arrow keys (5 cm; with Shift 25 cm); Save writes ``config/floor_tag.json``
(``fiducial.FLOOR_TAG_FILE``), which the camera task reads when it is next built, so the tag itself moves on the next
start (the GUI prints the Hydra override that does the same for one run). Without a stream it draws on a plain
background. Pure Tk + Pillow, like the repo's other GUIs (the venv's OpenCV has no windows).

Camera model: pixel = principal point + f * (X / Z, Y / Z) with X along the camera's right (+y of the room for the
default heading), Y down and Z along the optical axis; the quad is ``quad_size(size)`` wide (black square plus the
white margin), flat on the floor slab.

    python scripts/floor_tag_gui.py                      # live, from the camera window's stream
    python scripts/floor_tag_gui.py --stream host:port   # elsewhere
"""

import argparse
import math
import os
import sys
import threading
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_stream import frames as stream_frames  # noqa: E402

# the room and camera, from the task's own definitions (imported lazily: they pull in isaaclab)
_SCENE = None


def scene_geometry():
    """dict: camera pos/yaw/pitch/fx/width/height, floor_z, table (centre, size), defaults (x, y, size), file."""
    global _SCENE
    if _SCENE is None:
        from sheet_rl.tasks.manager_based.sheet_twist import fiducial
        from sheet_rl.tasks.manager_based.sheet_twist.d435i_camera import (
            D435I_FOCAL_PX, D435I_HEIGHT, D435I_PITCH_DEG, D435I_POS, D435I_WIDTH, D435I_YAW_DEG)
        from sheet_rl.tasks.manager_based.sheet_twist.sheet_ur16e_camera_env_cfg import (
            SheetUR16eCameraEnvCfg, floor_top)

        cfg = SheetUR16eCameraEnvCfg()
        table = cfg.scene.table
        _SCENE = {
            "pos": tuple(D435I_POS), "yaw": D435I_YAW_DEG, "pitch": D435I_PITCH_DEG, "fx": D435I_FOCAL_PX,
            "width": D435I_WIDTH, "height": D435I_HEIGHT, "floor_z": floor_top(table) + .0005,
            "table_centre": tuple(table.init_state.pos), "table_size": tuple(table.spawn.size),
            "defaults": fiducial.floor_tag_placement(), "file": str(fiducial.FLOOR_TAG_FILE),
            "quad_size": fiducial.quad_size, "save": fiducial.save_floor_tag_placement,
            "tag_positions": {0: tuple(cfg.scene.tag.init_state.pos), 1: tuple(cfg.scene.tag2.init_state.pos)},
        }
    return _SCENE


class CameraModel:
    """Projects room points [m] to pixels for a pinhole camera at ``pos`` with heading ``yaw_deg`` and downward tilt
    ``pitch_deg`` (``d435i_camera.camera_rotation``'s convention), focal ``fx`` [px], principal point at the centre."""

    def __init__(self, pos, yaw_deg, pitch_deg, fx, width, height):
        self.pos = np.asarray(pos, dtype=np.float64)
        y, p = math.radians(yaw_deg), math.radians(pitch_deg)
        self.forward = np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), -math.sin(p)])
        self.up = np.array([math.sin(p) * math.cos(y), math.sin(p) * math.sin(y), math.cos(p)])
        self.right = np.cross(self.forward, self.up)
        self.fx = self.fy = float(fx)
        self.width, self.height = int(width), int(height)
        self.cx, self.cy = (width - 1) / 2., (height - 1) / 2.

    def camera_coords(self, points):
        d = np.atleast_2d(np.asarray(points, dtype=np.float64)) - self.pos
        return np.stack([d @ self.right, -(d @ self.up), d @ self.forward], axis=1)  # X right, Y down, Z forward

    def project(self, points):
        """(N, 2) pixels (u, v) and (N,) depths Z; points behind the camera get Z <= 0 and NaN pixels."""
        xyz = self.camera_coords(points)
        z = xyz[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z > 0, self.cx + self.fx * xyz[:, 0] / z, np.nan)
            v = np.where(z > 0, self.cy + self.fy * xyz[:, 1] / z, np.nan)
        return np.stack([u, v], axis=1), z

    def in_frame(self, pixels, depths, margin=0.0):
        return bool(np.all(depths > 0) and np.all(pixels[:, 0] >= margin) and np.all(pixels[:, 0] <= self.width - 1 - margin)
                    and np.all(pixels[:, 1] >= margin) and np.all(pixels[:, 1] <= self.height - 1 - margin))


def segment_hits_box(a, b, centre, size):
    """True when the segment a-b passes through the axis-aligned box (centre, size), slab test with t in (0, 1)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    lo, hi = np.asarray(centre) - np.asarray(size) / 2., np.asarray(centre) + np.asarray(size) / 2.
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
            continue
        ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return t1 > 1e-9 and t0 < 1.0 - 1e-9


def tag_corners(x, y, z, edge):
    """The 4 corners [m] of a square of ``edge`` flat on the floor, centred at (x, y, z)."""
    h = edge / 2.
    return np.array([[x - h, y - h, z], [x + h, y - h, z], [x + h, y + h, z], [x - h, y + h, z]])


def assess(camera, x, y, size, scene):
    """Everything the GUI shows for one placement: pixels of the quad and the black square, status flags, numbers."""
    z = scene["floor_z"]
    quad = tag_corners(x, y, z, scene["quad_size"](size))
    square = tag_corners(x, y, z, size)
    quad_px, quad_z = camera.project(quad)
    square_px, _ = camera.project(square)
    centre_px, centre_z = camera.project([[x, y, z]])
    in_frame = camera.in_frame(quad_px, quad_z)
    blocked = any(segment_hits_box(camera.pos, c, scene["table_centre"], scene["table_size"]) for c in quad)
    d = np.array([x, y, z]) - camera.pos
    distance = float(np.linalg.norm(d))
    elevation = math.degrees(math.asin(min(1.0, max(-1.0, -d[2] / max(distance, 1e-9)))))  # camera above the tag
    px_w = px_h = float("nan")
    if in_frame:
        px_w = float(square_px[:, 0].max() - square_px[:, 0].min())
        px_h = float(square_px[:, 1].max() - square_px[:, 1].min())
    return {"quad_px": quad_px, "square_px": square_px, "centre_px": centre_px[0], "depth": float(centre_z[0]),
            "distance": distance, "elevation": elevation, "in_frame": in_frame, "blocked": blocked,
            "px_w": px_w, "px_h": px_h, "ok": in_frame and not blocked}


class StreamThread(threading.Thread):
    """Keeps the latest frame from the camera window's stream, reconnecting while the play restarts."""

    def __init__(self, address):
        super().__init__(daemon=True)
        self.address, self.frame, self.serial, self.connected = address, None, 0, False
        self.start()

    def run(self):
        while True:
            try:
                for frame in stream_frames(self.address, timeout=5.0):
                    self.frame, self.serial, self.connected = frame, self.serial + 1, True
            except OSError:
                pass
            self.connected = False
            time.sleep(1.0)


class FloorTagGui:
    def __init__(self, address, scene):
        self.scene = scene
        self.camera = CameraModel(scene["pos"], scene["yaw"], scene["pitch"], scene["fx"], scene["width"], scene["height"])
        self.stream = StreamThread(address)
        x0, y0, s0 = scene["defaults"]
        self.root = tk.Tk()
        self.root.title("Floor tag placement (UR16e camera task)")
        self.canvas = tk.Canvas(self.root, width=scene["width"], height=scene["height"], bg="#404040", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=20, padx=4, pady=4)
        panel = tk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="n", padx=6, pady=6)
        self.x = tk.DoubleVar(value=x0)
        self.y = tk.DoubleVar(value=y0)
        self.size = tk.DoubleVar(value=s0)
        for row, (label, var, lo, hi, step) in enumerate((("x [m] (room, -: behind the robot)", self.x, -7.0, 1.2, .01),
                                                           ("y [m] (room, +: camera's right)", self.y, -3.5, 3.5, .01),
                                                           ("black square [m]", self.size, .2, 1.6, .01))):
            tk.Label(panel, text=label, anchor="w").grid(row=2 * row, column=0, columnspan=2, sticky="w")
            tk.Scale(panel, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal", length=260,
                     command=lambda _v: self.moved()).grid(row=2 * row + 1, column=0, sticky="w")
            entry = tk.Entry(panel, textvariable=var, width=8)
            entry.grid(row=2 * row + 1, column=1, sticky="w", padx=4)
            entry.bind("<Return>", lambda _e: self.moved())
        self.status = tk.Label(panel, text="", justify="left", anchor="w", font=("TkFixedFont", 10))
        self.status.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 4))
        buttons = tk.Frame(panel)
        buttons.grid(row=7, column=0, columnspan=2, sticky="w")
        tk.Button(buttons, text="Save placement", command=self.save).pack(side="left")
        tk.Button(buttons, text="Reset undisturbed count", command=self.reset_watch).pack(side="left", padx=6)
        tk.Button(buttons, text="Defaults", command=self.defaults).pack(side="left")
        self.saved_label = tk.Label(panel, text=f"file: {scene['file']}", anchor="w", wraplength=300, justify="left")
        self.saved_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        tk.Label(panel, text="arrows: move 5 cm (Shift: 25 cm); +/-: size; s: save; q: quit", anchor="w",
                 wraplength=300, justify="left").grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.root.bind("<Key>", self.key)
        self.photo = None
        self.reference = None  # the quad's pixels when the count started
        self.watched = self.disturbed = 0
        self.last_serial = -1
        self.result = None
        self.moved()
        self.root.after(40, self.tick)

    # --- interaction
    def key(self, event):
        step = .25 if event.state & 0x1 else .05
        moves = {"Left": (0, -step), "Right": (0, step), "Up": (-step, 0), "Down": (step, 0)}
        if event.keysym in moves:
            dx, dy = moves[event.keysym]
            self.x.set(round(self.x.get() + dx, 3))
            self.y.set(round(self.y.get() + dy, 3))
            self.moved()
        elif event.keysym in ("plus", "equal", "KP_Add"):
            self.size.set(round(self.size.get() + .05, 3))
            self.moved()
        elif event.keysym in ("minus", "KP_Subtract"):
            self.size.set(round(max(.1, self.size.get() - .05), 3))
            self.moved()
        elif event.keysym == "s":
            self.save()
        elif event.keysym == "q":
            self.root.destroy()

    def moved(self):
        try:
            x, y, size = float(self.x.get()), float(self.y.get()), float(self.size.get())
        except (tk.TclError, ValueError):
            return
        self.result = assess(self.camera, x, y, size, self.scene)
        self.reset_watch()

    def reset_watch(self):
        self.reference, self.watched, self.disturbed = None, 0, 0

    def defaults(self):
        from sheet_rl.tasks.manager_based.sheet_twist.fiducial import TAG3_SIZE, TAG3_XY

        self.x.set(TAG3_XY[0])
        self.y.set(TAG3_XY[1])
        self.size.set(TAG3_SIZE)
        self.moved()

    def save(self):
        x, y, size = float(self.x.get()), float(self.y.get()), float(self.size.get())
        path = self.scene["save"](x, y, size)
        override = (f"env.scene.tag3.init_state.pos=[{x:.3f},{y:.3f},{self.scene['floor_z']:.4f}] "
                    f"env.scene.tag3.spawn.size={size:.3f}")
        self.saved_label.configure(text=f"saved to {path}\nfor one run now: {override}")
        print(f"[floor_tag_gui] saved x {x:.3f} y {y:.3f} size {size:.3f} to {path}\n[floor_tag_gui] Hydra override: {override}",
              flush=True)

    # --- drawing
    def watch(self, frame):
        """Count frames in which the quad's patch of the image differs from its first appearance (the arm over it)."""
        r = self.result
        if r is None or not r["in_frame"]:
            return
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(r["quad_px"]).astype(np.int32).reshape(-1, 1, 2), 1)
        patch = frame[mask.astype(bool)].astype(np.int16)
        if patch.size == 0:
            return
        if self.reference is None or self.reference.shape != patch.shape:
            self.reference = patch
            return
        self.watched += 1
        if np.abs(patch - self.reference).mean() > 12.0:
            self.disturbed += 1

    def tick(self):
        frame = self.stream.frame
        if frame is not None and self.stream.serial != self.last_serial:
            self.last_serial = self.stream.serial
            self.watch(frame)
            image = Image.fromarray(frame[..., :3])
        else:
            image = None
        self.draw(image)
        self.root.after(40, self.tick)

    def draw(self, image):
        c = self.canvas
        c.delete("overlay")
        if image is not None:
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            c.delete("frame")
            c.create_image(0, 0, image=self.photo, anchor="nw", tags="frame")
        # the table top's outline
        tc, ts = self.scene["table_centre"], self.scene["table_size"]
        top = np.array([[tc[0] - ts[0] / 2., tc[1] - ts[1] / 2., 0.], [tc[0] + ts[0] / 2., tc[1] - ts[1] / 2., 0.],
                        [tc[0] + ts[0] / 2., tc[1] + ts[1] / 2., 0.], [tc[0] - ts[0] / 2., tc[1] + ts[1] / 2., 0.]])
        top[:, 2] = tc[2] + ts[2] / 2.
        px, z = self.camera.project(top)
        if np.all(z > 0):
            c.create_polygon(*px.ravel().tolist(), outline="#ff80a0", fill="", width=1, dash=(4, 3), tags="overlay")
        for tag_id, pos in self.scene["tag_positions"].items():
            p, zz = self.camera.project([pos])
            if zz[0] > 0:
                c.create_text(p[0, 0], p[0, 1] - 10, text=f"tag {tag_id}", fill="#80ff80", tags="overlay")
        r = self.result
        if r is None:
            return
        colour = "#40ff40" if r["ok"] else "#ff4040"
        if np.all(np.isfinite(r["quad_px"])):
            c.create_polygon(*r["quad_px"].ravel().tolist(), outline=colour, fill="", width=2, tags="overlay")
            c.create_polygon(*r["square_px"].ravel().tolist(), outline=colour, fill="", width=1, tags="overlay")
        if np.all(np.isfinite(r["centre_px"])):
            c.create_text(r["centre_px"][0], r["centre_px"][1], text="2", fill=colour, font=("TkDefaultFont", 11, "bold"),
                          tags="overlay")
        if self.result["in_frame"]:
            why = "clear of the table" if not r["blocked"] else "BLOCKED by the table"
        else:
            why = "NOT fully in the frame"
        share = f"{100. * (1. - self.disturbed / self.watched):.0f} % of {self.watched} frames" if self.watched else "waiting for frames"
        self.status.configure(
            fg="#107010" if r["ok"] else "#a01010",
            text=(f"{why}\n"
                  f"distance {r['distance']:.2f} m, depth {r['depth']:.2f} m\n"
                  f"seen {r['elevation']:.0f} deg above the floor\n"
                  f"black square {r['px_w']:.0f} x {r['px_h']:.0f} px\n"
                  f"undisturbed: {share}\n"
                  f"stream: {'live' if self.stream.connected else 'not connected'}"))

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="127.0.0.1:5556", help="The camera window's frame stream.")
    args = parser.parse_args()
    FloorTagGui(args.stream, scene_geometry()).run()


if __name__ == "__main__":
    main()
