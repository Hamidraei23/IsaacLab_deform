# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal glTF 2.0 / GLB reader, enough for a single static textured mesh.

Deliberately dependency-free apart from numpy. The alternative -- ``omni.kit.asset_converter`` --
needs a full Isaac Sim app launch to run, and its importer is the one that already mis-handled the
scale on ``hand.usd``; parsing the container directly is a few dozen lines and leaves every
conversion decision visible in :mod:`convert_arm_and_hand`.

Only what this asset uses is implemented: float32 vertex attributes, scalar indices, and images
stored in buffer views. Anything else raises rather than being silently skipped, so a future asset
that needs more cannot quietly lose data.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

# glTF componentType enum -> numpy dtype
_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

# glTF accessor type -> number of components
_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

_MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png"}


class Glb:
    """A parsed GLB file: the glTF JSON plus its binary chunk."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = f.read()

        magic, version, total_length = struct.unpack("<III", data[:12])
        if magic != 0x46546C67:  # b"glTF"
            raise ValueError(f"{path} is not a GLB file (bad magic {magic:#x})")
        if version != 2:
            raise ValueError(f"only glTF 2.0 is supported, got version {version}")

        self.json: dict[str, Any] = {}
        self.bin = b""
        offset = 12
        while offset < total_length:
            chunk_length, chunk_type = struct.unpack("<II", data[offset : offset + 8])
            payload = data[offset + 8 : offset + 8 + chunk_length]
            if chunk_type == 0x4E4F534A:  # b"JSON"
                self.json = json.loads(payload.decode("utf-8"))
            elif chunk_type == 0x004E4942:  # b"BIN\0"
                self.bin = payload
            offset += 8 + chunk_length

        if not self.json:
            raise ValueError(f"{path} has no JSON chunk")

    def buffer_view(self, index: int) -> bytes:
        """Raw bytes of one buffer view."""
        view = self.json["bufferViews"][index]
        if view.get("buffer", 0) != 0:
            raise NotImplementedError("only the embedded GLB buffer is supported")
        start = view.get("byteOffset", 0)
        return self.bin[start : start + view["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        """One accessor as an ``(count, components)`` array, de-interleaved if strided.

        Shape is squeezed to ``(count,)`` for SCALAR accessors, which is what index buffers want.
        """
        acc = self.json["accessors"][index]
        if acc.get("sparse") is not None:
            raise NotImplementedError("sparse accessors are not supported")

        dtype = _COMPONENT_DTYPES[acc["componentType"]]
        components = _TYPE_COUNTS[acc["type"]]
        count = acc["count"]
        item_size = np.dtype(dtype).itemsize * components

        view = self.json["bufferViews"][acc["bufferView"]]
        stride = view.get("byteStride") or item_size
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)

        if stride == item_size:
            raw = self.bin[base : base + count * item_size]
            out = np.frombuffer(raw, dtype=dtype, count=count * components)
        else:
            # interleaved: gather each element out of its stride slot
            raw = np.frombuffer(self.bin, dtype=np.uint8, count=count * stride, offset=base)
            rows = raw.reshape(count, stride)[:, :item_size]
            out = np.frombuffer(rows.tobytes(), dtype=dtype, count=count * components)

        out = out.reshape(count, components)
        return out[:, 0] if components == 1 else out

    def image_bytes(self, index: int) -> tuple[bytes, str]:
        """One image's encoded bytes and the file extension its MIME type implies."""
        image = self.json["images"][index]
        if "uri" in image:
            raise NotImplementedError("external image URIs are not supported; expected embedded images")
        mime = image.get("mimeType", "image/png")
        if mime not in _MIME_EXTENSIONS:
            raise NotImplementedError(f"unsupported image MIME type {mime!r}")
        return self.buffer_view(image["bufferView"]), _MIME_EXTENSIONS[mime]

    def node_world_matrices(self) -> dict[int, np.ndarray]:
        """World matrix of every node, composed down from the scene roots.

        Returned row-vector style (``v_world = v_local @ M``), matching how the points are laid out.
        """
        nodes = self.json["nodes"]
        scene = self.json["scenes"][self.json.get("scene", 0)]
        world: dict[int, np.ndarray] = {}

        def local_matrix(node: dict[str, Any]) -> np.ndarray:
            if "matrix" in node:
                # glTF matrices are column-major; transposing gives the row-vector form
                return np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F").T
            m = np.eye(4)
            if "scale" in node:
                m = np.diag([*node["scale"], 1.0]) @ m
            if "rotation" in node:
                x, y, z, w = node["rotation"]
                rot = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                    [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                    [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
                ])
                full = np.eye(4)
                full[:3, :3] = rot
                m = m @ full
            if "translation" in node:
                trans = np.eye(4)
                trans[3, :3] = node["translation"]
                m = m @ trans
            return m

        def walk(index: int, parent: np.ndarray) -> None:
            matrix = local_matrix(nodes[index]) @ parent
            world[index] = matrix
            for child in nodes[index].get("children", []):
                walk(child, matrix)

        for root in scene["nodes"]:
            walk(root, np.eye(4))
        return world
