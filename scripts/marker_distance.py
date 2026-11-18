"""Distance from the camera to a printed fiducial tag (AprilTag / ArUco), from the RGB camera stream alone.

Reads only colour frames: inside the playing process as ``play_actor.py --camera d435i --marker`` (its window and
printout come from the ``MarkerDistance`` class below), or standalone here, live from the camera window's stream
(``play_actor.py --camera d435i`` serves it on 127.0.0.1:5556, ``--camera_port``) or from image files. Nothing of
the simulation's state is used.

Detection is OpenCV's ArUco detector with the tag's dictionary and AprilTag-style corner refinement (edge fitting,
the most accurate of OpenCV's four on a small, foreshortened tag: see ``detector_for``); the tag is small in the
frame (a few pixels per cell), so detection is also tried on the frame enlarged. Pose: ``solvePnPGeneric`` with
``IPPE_SQUARE`` on the four corners and the tag's black-square size, keeping the solution with the lower
reprojection error, then Levenberg-Marquardt refinement (``solvePnPRefineLM``); the distance is the length of the
translation, camera optical centre to tag centre. Corner order is OpenCV's for the detector's dictionary; a square
is symmetric under every cyclic shift or reversal of its corners, so the translation, and with it the distance,
does not depend on that order.

Defaults are the simulated D435i in its 640 x 480 mode (fx = fy = 615 px, principal point at the image centre) and
the task's tag (AprilTag 36h11, 10 cm black square); for a real camera pass its calibrated values.

Usage (from the sheet_rl project, in the Isaac venv for Tk, or any Python with opencv and numpy with --no-window)::

    python scripts/marker_distance.py                                # live, from the camera window's stream
    python scripts/marker_distance.py --image frame.png other.png    # from files
    python scripts/marker_distance.py --family apriltag_16h5 --size 0.10 --fx 615 --fy 615 --cx 320 --cy 240

Prints, per frame: the tag id, the distance [m], the tag centre in the camera frame (x right, y down, z forward),
the camera's position and orientation in the tag's frame (x, y along the tag's edges, z its normal: for a tag flat
on the table, z is the camera's height above the table; roll/pitch/yaw of the camera axes in that frame, the
camera's elevation above the tag's plane as seen from the tag, and how far below the plane the camera looks), the
tag's size in pixels and the reprojection error. Which tag edge is "x" follows the detector's corner order for the
family; the height, the horizontal range and the angles do not depend on it. A window shows the frame with the tag outlined and the
distance written on it; q or closing it stops.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAMILIES = {"apriltag_36h11": "DICT_APRILTAG_36h11", "apriltag_16h5": "DICT_APRILTAG_16h5", "aruco_4x4": "DICT_4X4_50"}
DEFAULT_FX = 615.
"""The simulated D435i's focal length [px] at 640 x 480 (d435i_camera.py: the mode's factory intrinsics)."""
UPSCALES = (1, 2, 3)
"""Frame enlargements tried in turn (cubic) until the tag is found; small tags detect only enlarged."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--stream", default="127.0.0.1:5556", help="The camera window's frame stream (default).")
    source.add_argument("--image", nargs="+", help="Image file(s) instead of the stream.")
    parser.add_argument("--family", default="apriltag_36h11", choices=sorted(FAMILIES))
    parser.add_argument("--tag-id", type=int, default=None, help="Only this tag id (default: any).")
    parser.add_argument("--size", type=float, default=.10, help="Edge of the tag's black square [m].")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="Focal length x [px].")
    parser.add_argument("--fy", type=float, default=DEFAULT_FX, help="Focal length y [px].")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x [px] (default: image centre).")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y [px] (default: image centre).")
    parser.add_argument("--no-window", action="store_true", help="Print only.")
    return parser.parse_args()


def detector_for(family: str):
    params = cv2.aruco.DetectorParameters()
    # Measured on the simulated table (10 cm tag, 39 px across, 20-degree grazing angle, true 1.181 m): the AprilTag
    # refinement (edge fitting) gave 1.182 m, SUBPIX 1.200, CONTOUR 1.207, NONE 1.211. Edges beat corners on a
    # small, foreshortened tag.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    params.minMarkerPerimeterRate = .01  # tags a few pixels across still count
    params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax, params.adaptiveThreshWinSizeStep = 3, 23, 4
    params.perspectiveRemovePixelPerCell = 8
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, FAMILIES[family]))
    return cv2.aruco.ArucoDetector(dictionary, params)


def find_tags(detector, gray, tag_id=None, expected=None):
    """{id: corners (4, 2) float32 at the frame's scale} of the tags found. Each enlargement in ``UPSCALES`` is tried
    in turn and adds the ids not seen yet (a tag keeps the corners from the smallest scale that found it); the search
    stops once ``expected`` tags are found, or after an enlargement adds nothing to a non-empty set."""
    tags = {}
    for scale in UPSCALES:
        image = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(image)
        added = 0
        for quad, found in ([] if ids is None else zip(corners, ids.ravel())):
            found = int(found)
            if (tag_id is None or found == tag_id) and found not in tags:
                tags[found] = quad.reshape(4, 2).astype(np.float32) / scale
                added += 1
        if tags and (added == 0 or (expected is not None and len(tags) >= expected)):
            break
    return tags


def find_tag(detector, gray, tag_id=None):
    """(corners (4, 2) float32 at the frame's scale, id) of the lowest-id matching tag, or (None, None)."""
    tags = find_tags(detector, gray, tag_id, expected=1 if tag_id is not None else None)
    if not tags:
        return None, None
    found = min(tags)
    return tags[found], found


def camera_in_tag_frame(rvec, tvec):
    """The camera's pose in the tag's frame (x, y along the tag's edges, z its normal, origin its centre).

    The solve gives p_cam = R p_tag + t; inverting, the camera sits at -R^T t with orientation R^T. Returns
    (position [m], (roll, pitch, yaw) [deg] of the camera axes in the tag frame, elevation [deg] of the camera above
    the tag's plane as seen from the tag, tilt [deg] between the camera's optical axis and the tag's plane).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    position = -R.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    R_cam_in_tag = R.T
    # ZYX Euler angles of the camera axes in the tag frame
    pitch = -np.arcsin(np.clip(R_cam_in_tag[2, 0], -1., 1.))
    roll = np.arctan2(R_cam_in_tag[2, 1], R_cam_in_tag[2, 2])
    yaw = np.arctan2(R_cam_in_tag[1, 0], R_cam_in_tag[0, 0])
    horizontal = float(np.hypot(position[0], position[1]))
    elevation = float(np.degrees(np.arctan2(position[2], horizontal)))
    optical_axis = R_cam_in_tag[:, 2]  # the camera's z in the tag frame
    tilt = float(np.degrees(np.arcsin(np.clip(-optical_axis[2], -1., 1.))))  # how far below the plane it looks
    return position, tuple(float(np.degrees(a)) for a in (roll, pitch, yaw)), elevation, tilt


def tag_pose(corners, size, K):
    """(rvec, tvec [m], reprojection error [px]) of the tag centre in the camera frame, or (None, None, None)."""
    half = size / 2.
    objpts = np.array([[-half, half, 0.], [half, half, 0.], [half, -half, 0.], [-half, -half, 0.]], dtype=np.float32)
    ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(objpts, corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or len(rvecs) == 0:
        return None, None, None
    best = int(np.argmin(np.asarray(errors).ravel()))
    rvec, tvec = cv2.solvePnPRefineLM(objpts, corners, K, None, rvecs[best], tvecs[best])
    projected, _ = cv2.projectPoints(objpts, rvec, tvec, K, None)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
    return rvec, tvec.reshape(3), error


class MarkerDistance:
    """Measures the camera-to-tag distance in RGB frames; ``play_actor.py --marker`` uses it in the playing process."""

    def __init__(self, fx, fy, cx=None, cy=None, size=.10, family="apriltag_36h11", tag_id=None, expected=None,
                 sizes=None):
        """``size`` is every tag's black-square edge [m] unless ``sizes`` ({id: edge}) says otherwise for an id."""
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.size, self.family, self.tag_id, self.expected = size, family, tag_id, expected
        self.sizes = dict(sizes or {})
        self.detector = detector_for(family)
        self.frames = self.found = 0
        self.found_by_id = {}
        self.distances = []

    def intrinsics(self, width, height):
        return np.array([[self.fx, 0., self.cx if self.cx is not None else (width - 1) / 2.],
                         [0., self.fy, self.cy if self.cy is not None else (height - 1) / 2.], [0., 0., 1.]])

    def measure(self, rgb):
        """(result or None, annotated RGB frame). result is the lowest-id tag's: id, distance [m], centre (x, y, z)
        [m], px, error, camera_position, camera_rpy, elevation, tilt, corners (4 x 2 pixels), rvec (the tag's
        rotation in the camera frame, Rodrigues), and ``all``: the
        results of every tag found in the frame (the same keys, sorted by id). Every tag is outlined; the
        distance is written at each, the camera's pose in the lowest-id tag's frame at the bottom."""
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
        self.frames += 1
        K = self.intrinsics(w, h)
        results, shown = [], np.ascontiguousarray(rgb[..., :3]).copy()
        for found, corners in sorted(find_tags(self.detector, gray, self.tag_id, self.expected).items()):
            rvec, t, error = tag_pose(corners, self.sizes.get(found, self.size), K)
            if t is None:
                continue
            self.found_by_id[found] = self.found_by_id.get(found, 0) + 1
            side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
            distance = float(np.linalg.norm(t))
            cam_pos, cam_rpy, elevation, tilt = camera_in_tag_frame(rvec, t)
            results.append({"id": found, "distance": distance, "centre": tuple(float(v) for v in t), "px": side_px,
                            "error": error, "camera_position": tuple(float(v) for v in cam_pos),
                            "camera_rpy": cam_rpy, "elevation": elevation, "tilt": tilt, "corners": corners.copy(),
                            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).copy()})
            cv2.polylines(shown, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            x0, y0 = int(corners[:, 0].min()), int(corners[:, 1].min())
            cv2.putText(shown, f"{found}: {distance:.3f} m", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 0), 2)
        if not results:
            return None, rgb
        self.found += 1
        self.distances.append(results[0]["distance"])
        cam_pos, cam_rpy, elevation, tilt = (results[0][k] for k in ("camera_position", "camera_rpy", "elevation", "tilt"))
        cv2.putText(shown, f"camera in tag {results[0]['id']} frame: x {cam_pos[0]:+.3f} y {cam_pos[1]:+.3f} "
                           f"z {cam_pos[2]:+.3f} m", (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.putText(shown, f"roll {cam_rpy[0]:+.1f} pitch {cam_rpy[1]:+.1f} yaw {cam_rpy[2]:+.1f} deg; "
                           f"elevation {elevation:.1f}, looking {tilt:.1f} deg down", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        results[0]["all"] = results
        return results[0], shown

    @staticmethod
    def describe(name, result):
        if result is None:
            return f"{name}: no tag found"
        c, p, rpy = result["centre"], result["camera_position"], result["camera_rpy"]
        text = (f"{name}: tag {result['id']} distance {result['distance']:.3f} m | tag in camera frame: x {c[0]:+.3f} "
                f"y {c[1]:+.3f} z {c[2]:+.3f} m | camera in tag frame: x {p[0]:+.3f} y {p[1]:+.3f} z {p[2]:+.3f} m, "
                f"roll {rpy[0]:+.1f} pitch {rpy[1]:+.1f} yaw {rpy[2]:+.1f} deg, elevation {result['elevation']:.1f} deg, "
                f"looking {result['tilt']:.1f} deg down | {result['px']:.0f} px, reprojection {result['error']:.2f} px")
        for other in result.get("all", [])[1:]:
            text += (f" | tag {other['id']}: distance {other['distance']:.3f} m, z {other['centre'][2]:.3f} m, "
                     f"{other['px']:.0f} px, reprojection {other['error']:.2f} px")
        return text


def main():
    args = parse_args()
    marker = MarkerDistance(args.fx, args.fy, args.cx, args.cy, args.size, args.family, args.tag_id)
    window = None
    if args.image:
        def source():
            for path in args.image:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"cannot read {path}", file=sys.stderr)
                    continue
                yield path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        from camera_stream import frames

        def source():
            print(f"reading the camera stream at {args.stream} ...", flush=True)
            for k, frame in enumerate(frames(args.stream)):
                yield f"frame {k}", frame
    print(f"{args.family}, black square {args.size * 100:.1f} cm, fx {args.fx:g} fy {args.fy:g}", flush=True)
    last_print = 0.
    for name, rgb in source():
        result, shown = marker.measure(rgb)
        if args.image or time.time() - last_print > .5:
            print(MarkerDistance.describe(name, result), flush=True)
            last_print = time.time()
        if not args.no_window:
            if window is None:
                from camera_window import CameraWindow

                window = CameraWindow("Tag distance", rgb.shape[1], rgb.shape[0])
            if not window.show(shown):
                break
            if args.image:
                time.sleep(1.5)
    if window is not None:
        window.close()


if __name__ == "__main__":
    main()
