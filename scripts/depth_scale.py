"""Depth-Anything-3's metric scale, corrected per frame from the table's fiducial tags (``play_actor.py --depth --marker``).

A monocular "metric" model gets the shape of a scene well but its absolute scale only approximately, and the error
is mostly one multiplier per frame that drifts between frames. A tag's pose from the RGB frame alone
(``marker_distance.py``) is an exact metric anchor in the same image: its centre's depth along the camera axis
(the PnP translation's z, the same quantity as a depth map's values) against the median of DA3's depth inside the
tag's outline. With one tag the ratio is the factor the whole map is off by (``scale``). With two tags at different
depths (the camera task has id 0 at 1.14 m and id 1 at 0.72 m) a line through both anchors also gives an offset
(``affine``: truth = a * DA3 + b), or the same line in inverse depth (``inverse``: 1/truth = a / DA3 + b, the
space such models are trained in). ``TagFit`` does one of these per frame from the tags visible in that frame
only (the user's rule): one tag gives a scale, none gives no correction, and nothing is carried over from before a
tag was hidden unless ``hold=True`` (then the last offset, or the last fit, is held). ``scale_from_tag`` is the
single-tag measurement and ``TagScale`` the single-tag tracker from the first version. Holding the last fit
while no tag is visible is on by default again (the user asked for it once the floor tag was in). ``DepthErrors`` compares raw and corrected maps
with the simulator's own depth (``play_actor.py --depth_eval``) so each variant is judged by numbers: per frame,
the median of DA3 over truth (the scale error), the median absolute relative error (AbsRel), the RMSE and the share
of pixels within 5 % of the truth, over the pixels that see geometry (not the sky) and over the near band (truth up
to ``NEAR_M``, the table and workspace, where the error matters).

Measured on the camera task with one tag (3 episodes, 694 frames, 2026-11-04): raw DA3 was 1.0 to 2.1 times the
true depth from frame to frame (median x1.35, AbsRel 37 % in the near band); scaled per frame from the tag it was
x1.03 (0.995..1.08), AbsRel 4.4 %, 58 % of near pixels within 5 %. Smoothing the factor (alpha 0.3) was worse
(AbsRel 5.4 %, 0.88..1.25): DA3's scale jumps between frames, so the smoothed factor lags. The tag anchor itself
was off by 3 mm at 1.14 m. With the second tag (3 episodes, 679 frames, same day; every variant on the same
frames): single tag AbsRel 3.5 % in the near band (x1.02, 0.98..1.07, 68 % within 5 %), one scale from both tags
3.0 %, ``affine`` 1.9 % (x1.00, 0.976..1.020, 79 % within 5 %), ``inverse`` 1.9 % (80 %). The anchors: tag 0 PnP
z 1.142 m against a true 1.139, tag 1 0.698 against 0.693. The second tag was then moved to the camera's right at
the first tag's distance (the user's placement; 673 frames, tag 1 visible in 373): one scale from both tags 3.4 %,
the same as one tag; an offset from anchors only 9 cm apart in depth was worse (6.3 %, inverse 5.4 %), hence
``MIN_ANCHOR_SPREAD``, which makes ``affine`` (still the default) a scale in that layout.
"""

import math

import cv2
import numpy as np

NEAR_M = 2.0
"""The near band's far edge [m]: the table top and workspace lie within it from the fixed D435i."""
FAR_VALID_M = 9.0
"""Truth beyond this [m] is the renderer's far clip, sky: left out of every statistic."""
TAG_INSET_PX = 1.5
"""How far the tag's outline is shrunk [px] before taking DA3's median inside it: edge pixels mix tag and table."""
MIN_ANCHOR_SPREAD = 0.25
"""Anchors spanning less than this in true depth [m] cannot fix an offset (with 3-5 mm anchor errors a 9 cm lever
arm gave offsets of several cm: AbsRel 6.3 % against the scale's 3.4 %); the fit then falls back to a scale."""
MODES = ("scale", "affine", "inverse")
NEAR_ANCHOR_M = 2.5
"""Only tags nearer than this [m] (the table's) fit the correction for the near band: DA3's error is far from affine
over range (at the floor tag, 4.7 m, DA3 reads 1.7 times less relative to truth than at the table tags), so a far
anchor in the fit made near pixels wrong or negative (black). A far tag alone still gives a scale, through the
ratio between near and far factors last seen when both were visible (``TagFit.far_ratio``)."""
MAX_DEPTH_M = 10.0
"""Corrected depth beyond this [m] (the camera's far clip) is invalid: a bad inverse fit can send pixels to infinity."""


def tag_mask(shape, corners, inset_px=TAG_INSET_PX):
    """Boolean (H, W) mask of the tag's quad (4 x 2 pixel corners), shrunk towards its centre by ``inset_px``."""
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    centre = corners.mean(axis=0)
    radial = corners - centre
    lengths = np.linalg.norm(radial, axis=1, keepdims=True)
    shrunk = centre + radial * np.clip(1.0 - inset_px / np.maximum(lengths, 1e-6), 0.0, 1.0)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(shrunk).astype(np.int32).reshape(-1, 1, 2), 1)
    return mask.astype(bool)


def scale_from_tag(depth, corners, tag_depth):
    """(factor, DA3's median depth over the tag) with ``factor`` = tag_depth / median, or (None, None) when the tag
    covers no valid DA3 pixel. ``tag_depth`` is the tag centre's z in the camera frame [m]."""
    mask = tag_mask(depth.shape, corners)
    values = depth[mask]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 4 or not (tag_depth > 0.0):
        return None, None
    median = float(np.median(values))
    return tag_depth / median, median


def anchors_from_tags(depth, tag_result):
    """[(DA3 median over the tag, the tag's PnP depth, id)] for every tag in a ``MarkerDistance`` result (its
    ``all`` list, or the result itself), skipping tags without valid DA3 pixels."""
    anchors = []
    for tag in ([] if tag_result is None else tag_result.get("all", [tag_result])):
        factor, median = scale_from_tag(depth, tag["corners"], tag["centre"][2])
        if factor is not None:
            anchors.append((median, float(tag["centre"][2]), int(tag["id"])))
    return anchors


def fit_anchors(anchors, mode, held_offset=None):
    """(a, b) of truth = a * m + b (``affine``), 1/truth = a / m + b (``inverse``) or b = 0 (``scale``) from
    [(m, z, id)] anchors, least squares. ``affine``/``inverse`` with one anchor, or two too close in depth
    (``MIN_ANCHOR_SPREAD``), use ``held_offset`` (0 if none: a plain scale) and solve the slope; no anchors: None."""
    if not anchors:
        return None
    m = np.array([a[0] for a in anchors], dtype=np.float64)
    z = np.array([a[1] for a in anchors], dtype=np.float64)
    spread = z.max() - z.min()  # in true depth: DA3's own scale must not decide whether an offset is fitted
    if mode == "inverse":
        m, z = 1.0 / m, 1.0 / z
    if mode == "scale":
        return float(np.dot(z, m) / np.dot(m, m)), 0.0
    if len(anchors) >= 2 and spread >= MIN_ANCHOR_SPREAD:
        a, b = np.polyfit(m, z, 1)
        return float(a), float(b)
    b = 0.0 if held_offset is None else held_offset
    return float(np.dot(z - b, m) / np.dot(m, m)), float(b)


def apply_fit(depth, params, mode):
    """The corrected depth map for ``fit_anchors``' (a, b); pixels the fit makes non-positive become 0 (invalid)."""
    a, b = params
    if mode == "inverse":
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = a / depth + b
            out = np.where((inverse > 0.0) & (depth > 0.0), 1.0 / inverse, 0.0)
    else:
        out = a * depth + b
    return np.where(np.isfinite(out) & (out > 0.0) & (out <= MAX_DEPTH_M), out, 0.0).astype(np.float32)


class TagFit:
    """Per-frame correction of a DA3 map from the tags seen in it, in one of ``MODES``: ``update`` returns the
    corrected map, or None before any tag was seen. The fit uses the near tags (``NEAR_ANCHOR_M``); one gives a
    scale, two far enough apart an offset too. With only far tags visible (the floor tag) the scale comes from them
    through ``far_ratio``, the near-over-far factor ratio last seen with both in view (1 until then), and the
    offset is held. With ``hold`` (default) the last fit is kept while no tag is visible and the last offset while
    only one is; ``hold=False`` uses nothing from before. ``anchors`` is the last frame's list, ``params`` its fit."""

    def __init__(self, mode="scale", hold=True, near_m=NEAR_ANCHOR_M):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        self.mode, self.hold, self.near_m = mode, hold, near_m
        self.params = None
        self.anchors = []
        self.used = []  # the anchors the last fit came from
        self.via_far = False  # the last fit came from far tags through far_ratio
        self.far_ratio = None
        self.measured = 0  # frames that had at least one anchor
        self.frames_with = {}  # anchors per frame -> frames

    @staticmethod
    def _scale(anchors):
        return fit_anchors(anchors, "scale")[0]

    def update(self, depth, tag_result):
        self.anchors = anchors_from_tags(depth, tag_result)
        self.frames_with[len(self.anchors)] = self.frames_with.get(len(self.anchors), 0) + 1
        near = [a for a in self.anchors if a[1] <= self.near_m]
        far = [a for a in self.anchors if a[1] > self.near_m]
        if near and far:
            self.far_ratio = self._scale(near) / self._scale(far)
        held = self.params[1] if self.hold and self.params is not None else None
        self.via_far = False
        if near:
            self.used = near
            self.params = fit_anchors(near, self.mode, held)
            self.measured += 1
        elif far:
            self.used, self.via_far = far, True
            a = self._scale(far) * (self.far_ratio if self.far_ratio is not None else 1.0)
            b = 0.0 if held is None else held
            self.params = (a, b) if self.mode != "inverse" else (1.0 / a, b)
            self.measured += 1
        elif not self.hold:
            self.params = None
        if self.params is None:
            return None
        return apply_fit(depth, self.params, self.mode)

    def describe(self):
        if self.params is None:
            return f"{self.mode}: no tag visible"
        a, b = self.params
        ids = ",".join(str(anchor[2]) for anchor in self.used) if self.anchors else "none (held)"
        if self.via_far and self.anchors:
            ids += f" (far, ratio {self.far_ratio if self.far_ratio is not None else 1.0:.3f})"
        if self.mode == "scale":
            return f"{self.mode}: x{a:.3f} from tag {ids}"
        if self.mode == "affine":
            return f"{self.mode}: x{a:.3f} {b:+.3f} m from tag {ids}"
        return f"{self.mode}: 1/d = {a:.3f}/d {b:+.3f} from tag {ids}"


class TagScale:
    """The single-tag running factor: a new tag measurement moves it by ``alpha`` towards the frame's own factor (1:
    each frame its own, the default, see the module docstring). With no tag in the frame the factor is None, or
    the last value if ``hold``. ``update`` returns the factor to apply."""

    def __init__(self, alpha=1.0, hold=True):
        self.alpha, self.hold = alpha, hold
        self.scale = None
        self.frame_scale = None  # the latest single-frame factor, unsmoothed
        self.measured = 0

    def update(self, depth, tag_result):
        self.frame_scale = None
        if tag_result is not None:
            factor, _ = scale_from_tag(depth, tag_result["corners"], tag_result["centre"][2])
            if factor is not None:
                self.frame_scale = factor
                self.measured += 1
                self.scale = factor if self.scale is None else (1.0 - self.alpha) * self.scale + self.alpha * factor
                return self.scale
        if not self.hold:
            self.scale = None
        return self.scale


def compare(depth, truth, near_m=NEAR_M):
    """Per-frame statistics of a depth map against the truth: dict with, for ``all`` valid pixels and the ``near``
    band, ratio (median depth/truth), absrel (median |depth - truth| / truth), rmse [m] and within5 (share of
    pixels within 5 % of truth). None where a band has no pixels."""
    valid = np.isfinite(truth) & (truth > 0.0) & (truth < FAR_VALID_M) & np.isfinite(depth) & (depth > 0.0)
    out = {}
    for name, band in (("all", valid), ("near", valid & (truth <= near_m))):
        if not band.any():
            out[name] = None
            continue
        d, t = depth[band].astype(np.float64), truth[band].astype(np.float64)
        rel = np.abs(d - t) / t
        out[name] = {"ratio": float(np.median(d / t)), "absrel": float(np.median(rel)),
                     "rmse": float(math.sqrt(np.mean((d - t) ** 2))), "within5": float(np.mean(rel < 0.05)),
                     "pixels": int(band.sum())}
    return out


class DepthErrors:
    """Accumulates ``compare`` results for the raw map and any number of named corrected variants, frame by frame,
    and summarises them; also checks each tag's PnP depth against the truth at its centre."""

    KEYS = ("ratio", "absrel", "rmse", "within5")

    def __init__(self):
        self.frames = {"raw": []}
        self.tag_check = {}  # id -> [(tag z from PnP, truth depth at the tag centre)]

    def add(self, raw, variants, truth, tag_result=None):
        """``variants``: {name: corrected map or None}. Returns {name: compare result or None} with ``raw``."""
        stats = {"raw": compare(raw, truth)}
        self.frames["raw"].append(stats["raw"])
        for name, depth in variants.items():
            stats[name] = compare(depth, truth) if depth is not None else None
            if stats[name] is not None:
                self.frames.setdefault(name, []).append(stats[name])
        for tag in ([] if tag_result is None else tag_result.get("all", [tag_result])):
            u, v = np.asarray(tag["corners"]).reshape(4, 2).mean(axis=0)
            u, v = int(round(u)), int(round(v))
            if 0 <= v < truth.shape[0] and 0 <= u < truth.shape[1] and np.isfinite(truth[v, u]):
                self.tag_check.setdefault(int(tag["id"]), []).append((float(tag["centre"][2]), float(truth[v, u])))
        return stats

    @staticmethod
    def line(label, stats):
        """One line for one frame's ``compare`` result."""
        if stats is None:
            return f"{label}: n/a"
        parts = []
        for name in ("all", "near"):
            s = stats[name]
            if s is None:
                continue
            parts.append(f"{name}: x{s['ratio']:.3f}, AbsRel {100 * s['absrel']:.1f} %, RMSE {s['rmse']:.3f} m, "
                         f"within 5 % {100 * s['within5']:.0f} %")
        return f"{label} | " + " | ".join(parts)

    def summary(self):
        """Means over frames (ratio also min..max) of the raw and each variant's statistics, as text lines."""
        lines = []
        for which, frames in self.frames.items():
            if not frames:
                continue
            lines.append(f"{which} ({len(frames)} frames):")
            for name in ("all", "near"):
                rows = [f[name] for f in frames if f[name] is not None]
                if not rows:
                    continue
                ratios = [r["ratio"] for r in rows]
                lines.append(f"  {name:4s}: depth/truth x{np.mean(ratios):.3f} ({min(ratios):.3f}..{max(ratios):.3f}), "
                             f"AbsRel {100 * np.mean([r['absrel'] for r in rows]):.1f} %, "
                             f"RMSE {np.mean([r['rmse'] for r in rows]):.3f} m, "
                             f"within 5 % {100 * np.mean([r['within5'] for r in rows]):.0f} %")
        for tag_id, rows in sorted(self.tag_check.items()):
            z = np.array(rows)
            lines.append(f"tag {tag_id} anchor: PnP z {z[:, 0].mean():.3f} m vs truth at the tag centre "
                         f"{z[:, 1].mean():.3f} m ({len(z)} frames)")
        return lines
