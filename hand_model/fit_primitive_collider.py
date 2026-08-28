# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit a handful of analytic primitives to ``arm_and_hand.usd`` and score how well they cover it.

Newton evaluates cloth-vs-shape contact by branching on the shape's type: ``SPHERE``, ``BOX``,
``CAPSULE``, ``CYLINDER``, ``CONE`` and ``ELLIPSOID`` all resolve to a few dozen flops of closed-form
SDF, while ``MESH`` (and ``CONVEX_MESH``, which takes the same branch) runs a BVH closest-point
query. Replacing one 34k-triangle query with a short list of primitives is therefore the way to keep
the arm's shape as a collider without paying for it every particle, every iteration, every substep.

This script decides how short that list can be. It fits a capsule chain to the forearm by slicing
along the limb axis, an oriented box to the palm and one capsule per finger, then reports the
signed distance from every vertex of the real mesh to the union of those primitives.

The per-finger capsules are measured, not assumed. A single box over the splay looks defensible --
a finger is thinner than the sheet's node spacing, so the cloth cannot resolve one anyway -- but it
scores three times worse, because the error that matters is not the width of a finger, it is the
box filling the gaps *between* them.

Sign convention on the reported error: positive means the skin sits *outside* the primitives, so
cloth would float; negative means the primitives bulge *through* the skin, so cloth would visibly
intersect the sculpt.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(HERE, "arm_and_hand.usd")

ARM_FRAME_ORIGIN_FROM_WRIST = 0.12
"""Matches ``convert_arm_and_hand.py``, so the fit is reported in the frame the env spawns."""

FOREARM_SEGMENTS = 4
"""Capsules spent on the forearm.

The measured knee. The forearm's radius runs 0.027 m at the wrist to 0.059 m at the elbow, which a
single capsule cannot follow -- that is the collider in use today, and it misses the skin by 14.8 mm
on average across the band. Four capsules cut that to 6.7 mm; eight reach 5.6 mm and sixteen do no
better than eight, so the curve is flat well before the cost matters.

Four rather than eight because the remaining 1.1 mm is a twentieth of the sheet's node spacing --
the cloth has no way to represent it.
"""

FINGER_COUNT = 5
"""Capsules spent on the finger splay.

Not an aesthetic choice: one box over the whole splay reads as a mitten and misses by 14.3 mm,
while five capsules -- one per finger -- reach 5.9 mm. The gaps between fingers are the reason, and
a box has to fill them.
"""

# The sheet's nodes are 0.2 m / 8 = 0.025 m apart, so nothing finer than this is representable in
# the drape no matter how good the collider is. Used as the yardstick for "close enough".
NODE_SPACING = 0.025


def _load_points() -> np.ndarray:
    """Mesh vertices in the task frame the environment spawns."""
    stage = Usd.Stage.Open(USD_PATH)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/arm_and_hand/geom"))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    points[:, 0] += ARM_FRAME_ORIGIN_FROM_WRIST
    return points


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment ``ab``."""
    axis = b - a
    length_sq = axis @ axis
    t = np.clip((points - a) @ axis / length_sq, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * axis), axis=1)


class Capsule:
    """A capsule between two points, in the arm frame."""

    def __init__(self, a: np.ndarray, b: np.ndarray, radius: float):
        self.a, self.b, self.radius = a, b, float(radius)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return _segment_distance(points, self.a, self.b) - self.radius

    def describe(self) -> str:
        centre = 0.5 * (self.a + self.b)
        length = float(np.linalg.norm(self.b - self.a))
        return (
            f"Capsule r={self.radius:.4f} length={length:.4f} "
            f"centre=({centre[0]:+.4f},{centre[1]:+.4f},{centre[2]:+.4f})"
        )


class Box:
    """An oriented box: centre, orthonormal axes as rows, and half-extents."""

    def __init__(self, centre: np.ndarray, axes: np.ndarray, half: np.ndarray):
        self.centre, self.axes, self.half = centre, axes, half

    def sdf(self, points: np.ndarray) -> np.ndarray:
        local = (points - self.centre) @ self.axes.T
        q = np.abs(local) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def describe(self) -> str:
        return (
            f"Box half=({self.half[0]:.4f},{self.half[1]:.4f},{self.half[2]:.4f}) "
            f"centre=({self.centre[0]:+.4f},{self.centre[1]:+.4f},{self.centre[2]:+.4f})"
        )


def _fit_capsule_chain(points: np.ndarray, x_lo: float, x_hi: float, count: int) -> list[Capsule]:
    """Slice the limb along x and lay one capsule per slice, following the arm's own curve.

    Each slice contributes its centroid to a spine polyline and its mean radius about that
    centroid, so an arm that bends is tracked rather than averaged into a fatter straight tube.
    """
    edges = np.linspace(x_lo, x_hi, count + 1)
    spine, radii = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        slab = points[(points[:, 0] >= low) & (points[:, 0] < high)]
        if len(slab) < 20:
            continue
        centre = slab.mean(axis=0)
        spine.append(centre)
        radii.append(np.linalg.norm(slab[:, 1:] - centre[1:], axis=1).mean())

    capsules = []
    for index, (centre, radius) in enumerate(zip(spine, radii)):
        # each capsule spans its own slice, and is stretched to meet its neighbours so the chain
        # has no gaps for a cloth node to fall into
        a = centre.copy()
        b = centre.copy()
        a[0], b[0] = edges[index], edges[index + 1]
        if index > 0:
            a[1:] = 0.5 * (spine[index - 1][1:] + centre[1:])
        if index < len(spine) - 1:
            b[1:] = 0.5 * (spine[index + 1][1:] + centre[1:])
        capsules.append(Capsule(a, b, radius))
    return capsules


def _fit_box(points: np.ndarray, inflate: float = 0.0) -> Box:
    """Tightest oriented box around a point set, from its own principal axes."""
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    local = (points - centre) @ axes.T
    low, high = local.min(axis=0), local.max(axis=0)
    box_centre = centre + 0.5 * (low + high) @ axes
    return Box(box_centre, axes, 0.5 * (high - low) + inflate)


def _fit_finger_capsules(fingers: np.ndarray, count: int) -> list[Capsule]:
    """One capsule per finger, split by clustering the splay across the limb axis."""
    plane = fingers[:, 1:]
    # seeded on spread percentiles rather than at random, so the same fit comes out every run
    centres = np.percentile(plane, np.linspace(8, 92, count), axis=0)
    labels = np.zeros(len(plane), dtype=int)
    for _ in range(40):
        labels = ((plane[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        for k in range(count):
            if (labels == k).any():
                centres[k] = plane[labels == k].mean(axis=0)

    capsules = []
    for k in range(count):
        cluster = fingers[labels == k]
        if len(cluster) < 20:
            continue
        centre = cluster.mean(axis=0)
        _, _, axes = np.linalg.svd(cluster - centre, full_matrices=False)
        along = (cluster - centre) @ axes[0]
        radius = np.linalg.norm((cluster - centre) - np.outer(along, axes[0]), axis=1).mean()
        capsules.append(Capsule(centre + along.min() * axes[0], centre + along.max() * axes[0], radius))
    return capsules


def _report(name: str, shapes: list, points: np.ndarray, subset: np.ndarray) -> None:
    union = np.min(np.stack([shape.sdf(points[subset]) for shape in shapes]), axis=0)
    print(
        f"  {name:22s} {len(shapes):2d} shapes | mean {np.abs(union).mean():.4f}"
        f" p95 {np.percentile(np.abs(union), 95):.4f}"
        f" | worst out {union.max():+.4f} in {union.min():+.4f}"
        f" | {np.abs(union).mean() / NODE_SPACING:.2f} node spacings"
    )


def main() -> None:
    points = _load_points()
    wrist_x = ARM_FRAME_ORIGIN_FROM_WRIST
    elbow_x = points[:, 0].min()
    knuckle_x = wrist_x + 0.075

    is_forearm = points[:, 0] < wrist_x
    is_band = np.abs(points[:, 0]) <= 0.09
    everything = np.ones(len(points), dtype=bool)

    print("Forearm only -- capsule chain, error against segment count [m]:")
    print("  N   mean|e|   p95|e|   band mean|e|      (the collider in use today is N=1)")
    for count in (1, 2, 3, 4, 6, 8, 12, 16):
        chain = _fit_capsule_chain(points, elbow_x, wrist_x, count)
        forearm_err = np.min(np.stack([c.sdf(points[is_forearm]) for c in chain]), axis=0)
        band_err = np.min(np.stack([c.sdf(points[is_band]) for c in chain]), axis=0)
        print(
            f"  {count:2d}  {np.abs(forearm_err).mean():.4f}   "
            f"{np.percentile(np.abs(forearm_err), 95):.4f}   {np.abs(band_err).mean():.4f}"
        )

    forearm = _fit_capsule_chain(points, elbow_x, wrist_x, FOREARM_SEGMENTS)
    hand = points[points[:, 0] >= wrist_x]
    palm = [_fit_box(hand[hand[:, 0] < knuckle_x])]
    fingers = _fit_finger_capsules(hand[hand[:, 0] >= knuckle_x], FINGER_COUNT)
    mitten = [_fit_box(hand[hand[:, 0] >= knuckle_x])]

    print("\nWhole limb [m]:")
    _report("forearm only", forearm, points, is_forearm)
    _report("+ palm + finger box", forearm + palm + mitten, points, everything)
    _report("+ palm + 5 fingers", forearm + palm + fingers, points, everything)

    shapes = forearm + palm + fingers
    print(f"\nRecommended set -- {len(shapes)} primitives, no BVH, all closed-form:")
    for shape in shapes:
        print("  " + shape.describe())

    union = np.min(np.stack([shape.sdf(points[is_band]) for shape in shapes]), axis=0)
    print("\nOver the band's travel (|x| <= 0.09 m), where the drape is actually scored:")
    print(f"  mean |error| {np.abs(union).mean():.4f} m  ({np.abs(union).mean() / NODE_SPACING:.2f} node spacings)")
    print(f"  p95 |error|  {np.percentile(np.abs(union), 95):.4f} m")
    print("  (+ skin outside the primitives: cloth floats | - primitives through the skin)")
    print(f"  worst outward {union.max():+.4f} m   worst inward {union.min():+.4f} m")


if __name__ == "__main__":
    main()
