# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render orthographic previews of ``arm_and_hand.usd`` without launching Isaac Sim.

A z-buffered triangle rasteriser with flat Lambert shading, sampling the bound base-colour texture
through the mesh's own ``st`` primvar. Its job is verification rather than beauty: if the UVs were
flipped, the tangent frame lost or the canonical axes mis-derived, it shows up here in seconds
instead of after a full app launch.

Reads the USD rather than the GLB on purpose -- this checks the file that was actually written.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(HERE, "arm_and_hand.usd")

RESOLUTION = 1100
LIGHT_DIRECTION = np.array([0.4, -0.5, 0.75])


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Points, triangles, per-vertex UVs, and the base-colour image, from the USD."""
    stage = Usd.Stage.Open(USD_PATH)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/arm_and_hand/geom"))

    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    triangles = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    st = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(), dtype=np.float64)

    material, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    surface = UsdShade.Shader(material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim())
    texture_prim = surface.GetInput("diffuseColor").GetConnectedSource()[0].GetPrim()
    asset = UsdShade.Shader(texture_prim).GetInput("file").Get()
    # resolvedPath honours the layer-relative "./textures/..." the converter authored
    image = np.asarray(Image.open(asset.resolvedPath).convert("RGB"), dtype=np.float64) / 255.0

    return points, triangles, st, image


def _render(points: np.ndarray, triangles: np.ndarray, st: np.ndarray, image: np.ndarray,
            right: np.ndarray, up: np.ndarray, view: np.ndarray) -> Image.Image:
    """Orthographic z-buffered render looking along ``view``."""
    u = points @ right
    v = points @ up
    depth = points @ view

    low = np.array([u.min(), v.min()])
    high = np.array([u.max(), v.max()])
    scale = (RESOLUTION - 60) / (high - low).max()
    width = RESOLUTION
    height = int((high[1] - low[1]) * scale) + 60

    x = (u - low[0]) * scale + 30
    y = height - 1 - ((v - low[1]) * scale + 30)

    frame = np.zeros((height, width, 3))
    zbuffer = np.full((height, width), np.inf)

    # flat shading: one normal and one texel per triangle, which is plenty to see whether the UVs
    # land where they should
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    # abs(), because the mesh is double-sided and back faces would otherwise render black
    shade = 0.25 + 0.75 * np.abs(normals @ light)

    centroid_uv = st[triangles].mean(axis=1)
    rows = np.clip(((1.0 - centroid_uv[:, 1]) * (image.shape[0] - 1)).astype(int), 0, image.shape[0] - 1)
    cols = np.clip((centroid_uv[:, 0] * (image.shape[1] - 1)).astype(int), 0, image.shape[1] - 1)
    albedo = image[rows, cols] * shade[:, None]

    tri_x, tri_y, tri_z = x[triangles], y[triangles], depth[triangles]
    # far triangles first, so nearer ones overwrite them; the z-buffer then only has to resolve
    # the overlaps that ordering cannot
    for tri in np.argsort(-tri_z.mean(axis=1)):
        px, py, pz = tri_x[tri], tri_y[tri], tri_z[tri]
        x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max())) + 1
        y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max())) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x0 >= x1 or y0 >= y1:
            continue

        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0])
        if abs(area) < 1e-12:
            continue
        w0 = ((px[1] - gx) * (py[2] - gy) - (px[2] - gx) * (py[1] - gy)) / area
        w1 = ((px[2] - gx) * (py[0] - gy) - (px[0] - gx) * (py[2] - gy)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not inside.any():
            continue

        z = w0 * pz[0] + w1 * pz[1] + (1 - w0 - w1) * pz[2]
        window = zbuffer[y0:y1, x0:x1]
        visible = inside & (z < window)
        window[visible] = z[visible]
        frame[y0:y1, x0:x1][visible] = albedo[tri]

    return Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    points, triangles, st, image = _load()
    axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}

    # looking down -Z shows the back of the hand; looking down -Y shows the limb side-on
    for name, (right, up, view) in {
        "dorsal": (axes["x"], axes["y"], -axes["z"]),
        "side": (axes["x"], axes["z"], axes["y"]),
    }.items():
        out = os.path.join(HERE, f"arm_and_hand_preview_{name}.png")
        _render(points, triangles, st, image, right, up, view).save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
