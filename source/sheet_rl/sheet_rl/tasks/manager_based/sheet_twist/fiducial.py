"""A printed fiducial tag (AprilTag by default) lying flat on the table near the robot base, for the camera.

Replaces the QR code of the first version (user request): a QR code's 21-29 modules fall below a pixel each from the
camera a metre away, while an AprilTag 36h11 has 8 cells across (6 of data and the black border), an AprilTag 16h5
6, so each cell gets 3-4 times the pixels, and tag detectors are built for pose, not text. The tag's black square is
``TAG_SIZE`` on a side (the "10 x 10 cm code"), with the one-cell white margin the detectors need around it, so the
printed quad is a little larger (``quad_size``). Families (``TAG_FAMILIES``): ``apriltag_36h11`` (the robust general
choice), ``apriltag_16h5`` (fewer bits, about 25 % more range), ``aruco_4x4`` (OpenCV's own 4 x 4 family).

The quad is a UsdGeom.Mesh with ``st`` UVs and a UsdPreviewSurface whose diffuse colour is a UsdUVTexture reading a
PNG, which Newton's USD importer reads into the model and the Warp ray tracer samples. The PNG is OpenCV's
``generateImageMarker`` output with the margin added, written once to ``~/.cache/sheet_rl``.

``scripts/marker_distance.py`` (and ``play_actor.py --marker``) finds the tag in the camera's colour frames and,
from ``TAG_SIZE`` and the camera's intrinsics, the distance to it. Visual only: no collision, nothing in the task
reads it. Everything is in the environment frame: the table top is z = 0, the robot base at (-0.12, 0.02).
"""

import hashlib
import os
import pathlib

import numpy as np

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass

TAG_FAMILIES = {
    # name: (OpenCV predefined-dictionary attribute, cells across the black square including its 1-cell border)
    "apriltag_36h11": ("DICT_APRILTAG_36h11", 8),
    "apriltag_16h5": ("DICT_APRILTAG_16h5", 6),
    "aruco_4x4": ("DICT_4X4_50", 6),
}
TAG_FAMILY = "apriltag_36h11"
"""Default family."""
TAG_ID = 0
"""Which tag of the family is printed."""
TAG_SIZE = .10
"""Edge of the tag's black square [m]; the size the pose is solved against."""
TAG_MARGIN_CELLS = 1
"""White margin around the black square, in cells (what the detectors need)."""
TAG_POS = (.15, -.30, .0005)
"""Centre in the environment frame [m]: flat on the table, 27 cm in front of the robot base (-0.12, 0.02) and 30 cm
to its -y side, clear of the sheet slot (0.41, -0.08) and the platform (+y side); the user's placement. 1.18 m from
the default camera at a 20-degree grazing angle. Move it with ``env.scene.tag.init_state.pos=[x,y,0.0005]``."""
TAG2_ID = 1
"""The second tag's id (same family): two tags at different depths let a depth map's scale and offset be fitted."""
TAG2_POS = (.05, .40, .0005)
"""The second tag's centre [m]: flat on the table on the camera's right (+y; the camera looks along -x, so +y is
image right), a little further out and back than the first tag's mirror image (0.15, 0.30), where the arm hid it
in most frames on its way to the platform (still hidden in 45 % of frames here): 1.29 m from the default camera
(1.23 m deep against the first tag's 1.14 m), 17 cm in front of the robot base and 38 cm to its +y side; the platform lands in x 0.38..0.62, y
0.03..0.27 (footprint from x = 0.24). A tag on the camera's side, e.g. (0.62, -0.20) at 0.70 m, would give the
depth spread that fixes an offset as well as the scale (``depth_scale.MIN_ANCHOR_SPREAD``). Move it with
``env.scene.tag2.init_state.pos=[x,y,0.0005]``."""
TAG3_ID = 2
"""The floor tag's id (same family): a far anchor for the depth correction that nothing on the table can hide."""
TAG3_SIZE = .8
"""The floor tag's black square [m]: from the camera it is 4.9 m away and seen 22 degrees above the floor, so a
10 cm tag would be 15 px wide; 0.8 m (an A0 print) gives about 100 x 37 px."""
TAG3_XY = (-3.3, -1.6)
"""The floor tag's centre [m], on the floor (its z is the camera task's floor, ``sheet_ur16e_camera_env_cfg``): the
table is a solid block (x -0.35..1.15, y +-0.45, 1.05 m thick), so the only floor the camera sees is the strip
behind the robot beyond the far edge, from x = -2.5 on; at x = -3.3 the tag's near edge clears the table's edge
in the image by 2 degrees. Measured over 3 episodes each: at (-3.3, 1.0) the arm's elbow hid it in 55 % of frames,
at (-3.3, 1.6) in 42 %; on the camera's left (-y, the user's suggestion) at (-3.3, -1.6) it was seen in every one
of 702 frames (5.1 m away, 77 px wide). Nearer spots beside the table, e.g. (-1.5, +-1.5), leave the frame. Place
it yourself with ``scripts/floor_tag_gui.py`` (saved to ``FLOOR_TAG_FILE``, which overrides this)."""
FLOOR_TAG_FILE = pathlib.Path(os.environ.get(
    "SHEET_RL_FLOOR_TAG", pathlib.Path(__file__).resolve().parents[6] / "config" / "floor_tag.json"))
"""Where ``scripts/floor_tag_gui.py`` saves the floor tag's placement ({"x", "y", "size"} [m]); the camera task
reads it at build time (``floor_tag_placement``) and falls back to ``TAG3_XY`` / ``TAG3_SIZE`` without it."""


def floor_tag_placement(path=None):
    """(x, y, size) [m] of the floor tag: the saved placement if the file exists and parses, else the defaults."""
    import json

    path = pathlib.Path(path or FLOOR_TAG_FILE)
    try:
        saved = json.loads(path.read_text())
        return float(saved["x"]), float(saved["y"]), float(saved.get("size", TAG3_SIZE))
    except (OSError, ValueError, KeyError, TypeError):
        return TAG3_XY[0], TAG3_XY[1], TAG3_SIZE


def save_floor_tag_placement(x, y, size, path=None):
    """Write the floor tag's placement for the camera task; returns the path."""
    import json

    path = pathlib.Path(path or FLOOR_TAG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"x": round(float(x), 4), "y": round(float(y), 4), "size": round(float(size), 4),
                                "note": "floor tag (id 2) of the UR16e camera task, saved by scripts/floor_tag_gui.py"},
                               indent=2) + "\n")
    return path
TAG_PIXELS_PER_CELL = 32
"""Texture resolution: crisp edges when the ray tracer samples it."""
TAG_CACHE_DIR = pathlib.Path(os.environ.get("SHEET_RL_CACHE", pathlib.Path.home() / ".cache" / "sheet_rl"))


def family_cells(family: str = TAG_FAMILY) -> int:
    return TAG_FAMILIES[family][1]


def quad_size(size: float = TAG_SIZE, family: str = TAG_FAMILY) -> float:
    """Edge of the printed quad [m]: the black square plus the white margin on each side."""
    cells = family_cells(family)
    return size * (cells + 2 * TAG_MARGIN_CELLS) / cells


def dictionary(family: str = TAG_FAMILY):
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_FAMILIES[family][0]))


def tag_image(family: str = TAG_FAMILY, tag_id: int = TAG_ID) -> np.ndarray:
    """The tag as a uint8 grey image (0 black, 255 white) with its white margin, square."""
    import cv2

    cells = family_cells(family)
    side = cells * TAG_PIXELS_PER_CELL
    marker = cv2.aruco.generateImageMarker(dictionary(family), tag_id, side, borderBits=1)
    pad = TAG_MARGIN_CELLS * TAG_PIXELS_PER_CELL
    return cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def tag_png(family: str = TAG_FAMILY, tag_id: int = TAG_ID) -> pathlib.Path:
    """Path of the tag's PNG in the cache, written if missing."""
    import cv2

    TAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{family}:{tag_id}:{TAG_MARGIN_CELLS}:{TAG_PIXELS_PER_CELL}".encode()).hexdigest()[:10]
    path = TAG_CACHE_DIR / f"tag_{family}_{tag_id}_{key}.png"
    if not path.exists():
        cv2.imwrite(str(path), tag_image(family, tag_id))
    return path


@clone
def spawn_tag(prim_path, cfg, translation=None, orientation=None):
    """A textured square quad, ``quad_size(cfg.size, cfg.family)`` on a side, its +z face up."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt  # at call time: importing pxr before Kit starts crashes --viz kit

    from isaaclab.sim.utils import create_prim, get_current_stage

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    half = quad_size(cfg.size, cfg.family) / 2.
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/quad")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0.), Gf.Vec3f(half, -half, 0.),
                                         Gf.Vec3f(half, half, 0.), Gf.Vec3f(-half, half, 0.)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 0, 2, 3]))  # counter-clockwise seen from +z
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0., 0., 1.)] * 4))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0.), Gf.Vec3f(half, half, 0.)]))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(1., 1., 1.)])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                           UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0., 0.), Gf.Vec2f(1., 0.), Gf.Vec2f(1., 1.), Gf.Vec2f(0., 1.)]))

    material = UsdShade.Material.Define(stage, f"{prim_path}/material")
    surface = UsdShade.Shader.Define(stage, f"{prim_path}/material/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.9)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
    reader = UsdShade.Shader.Define(stage, f"{prim_path}/material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, f"{prim_path}/material/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tag_png(cfg.family, cfg.tag_id)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    material.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    prim = stage.GetPrimAtPath(prim_path)
    if not cfg.visible:
        UsdGeom.Imageable(prim).MakeInvisible()
    return prim


@configclass
class TagCfg(SpawnerCfg):
    """Spawner for the printed tag (see the module docstring)."""

    func = spawn_tag
    family: str = TAG_FAMILY
    tag_id: int = TAG_ID
    size: float = TAG_SIZE
