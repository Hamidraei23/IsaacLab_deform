# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Sketchfab arm-and-hand GLB into ``arm_and_hand.usd``.

Run with no arguments from this directory::

    python3 convert_arm_and_hand.py

Why this rather than ``omni.kit.asset_converter``: the kit importer needs a full Isaac Sim app
launch, and it is the same path that mis-handled the scale on ``hand.usd`` -- see ``HAND_SCALE`` in
``sheet_rl_hand_env_cfg.py``, where a spawner-side 0.001 was silently dropped by the kit renderer
and produced a 219-metre hand. Everything that matters here (units, up-axis, the pose of the limb
in its own frame) is a decision rather than a default, so it is made explicitly and in one place.

The output follows the same conventions ``hand.usd`` already established -- Z-up, metres baked into
the geometry, ``scale=1.0`` at the spawner -- so it drops into the existing workflow unchanged.

What is preserved from the GLB: positions, normals, UVs, tangents, the triangle topology, all three
PBR textures, and the Sketchfab attribution the CC-BY-4.0 licence requires.
"""

from __future__ import annotations

import json
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from glb_reader import Glb

HERE = os.path.dirname(os.path.abspath(__file__))
GLB_PATH = os.path.join(HERE, "hand_and_arm_-_zbrush_sculpt.glb")
USD_PATH = os.path.join(HERE, "arm_and_hand.usd")
ARM_FRAME_PATH = os.path.join(HERE, "arm_and_hand_arm_frame.usda")
TEXTURE_DIRNAME = "arm_and_hand_textures"

ROOT_PRIM = "/arm_and_hand"

ARM_MASS = 2.0
"""Mass authored on the limb [kg], carried over from the capsule it replaces.

Irrelevant to the dynamics -- the arm is spawned kinematic -- but its *presence* is not: a body
without mass is imported as static geometry and never registers with Newton, which leaves the
``RigidObject`` wrapper with nothing to find. The same trap the cosmetic wrist taper documented.
"""

MESH_APPROXIMATION = UsdPhysics.Tokens.none
"""Collision approximation authored on the mesh: ``none`` means the triangles themselves.

Newton's particle-vs-shape kernel branches on shape type, and both ``MESH`` and ``CONVEX_MESH``
resolve to the same ``mesh_query_point_sign`` BVH query -- so a convex hull would cost the same as
the real thing while throwing the shape away. Since the query is being paid for either way, it is
paid for the actual surface.

The mesh qualifies for the accurate branch: it is a closed manifold (51432 edges, every one shared
by exactly two faces) with outward winding, so ``resolve_mesh_sign_method`` selects ``PARITY``
rather than falling back to the pseudo-normal used for open surfaces.
"""

SDF_MAX_RESOLUTION = 128
"""Longest-axis voxel count for the signed-distance field built for this mesh.

Required, not optional. The sheet task couples rigid bodies into the soft solver with
``enable_rigid_soft_full_surface_contact=True``, and that path's edge and face passes *sample the
rigid shape's SDF*. A mesh without one is skipped silently at the contact stage, so Newton refuses
the whole model up front rather than letting cloth fall through::

    ValueError: enable_rigid_soft_full_surface_contact=True, but these participating rigid shapes
    have no signed-distance field

Analytic shapes are exempt, which is why the capsule this replaces never needed it.

128 rather than the schema's default of 64, because of the fingers. The limb is 0.607 m along its
longest axis, so 128 puts a voxel at about 4.7 mm and spans a ~17 mm finger with three or four of
them; at 64 a finger is under two voxels across and the SDF stops describing it. Newton requires the
value to be divisible by 8 -- SDF volumes are allocated in 8x8x8 tiles -- and it is the first knob to
turn down if memory becomes a problem.
"""


def _forearm_midpoint_offset(landmarks: dict) -> float:
    """Where the task-frame wrapper puts its origin, up the forearm from the wrist [m].

    The canonical asset's origin is the wrist, which is the right landmark for the asset but the
    wrong one for the task: the rewards model the arm as one analytic capsule centred on the body's
    root, and the red band slides along that root frame's +X. Left at the wrist, the capsule would
    straddle the hand and the band would be painted across the fingers.

    Half the forearm's length puts the root at the midpoint of the elbow-to-wrist segment, which is
    exactly the centre of the capsule the observation is meant to assume.
    """
    return 0.5 * landmarks["forearmLength_m"]

ADULT_HAND_LENGTH = 0.185
"""Wrist-to-fingertip length the mesh is scaled to [m].

The sculpt arrives unitless, so something has to set the scale, and hand length is the landmark
this mesh measures most cleanly -- the wrist is an unambiguous waist in the radius profile and the
fingertips are the extreme point along the limb axis. 0.185 m is an average adult hand.

Scaling to an anatomical size rather than to the collision capsule is deliberate: it makes the
asset correct on its own terms instead of encoding one task's geometry, and it is the same
convention ``hand.usd`` already follows. It costs nothing here, because the two agree anyway -- at
this scale the 0.28 m stretch of forearm just past the wrist has a mean radius of 0.043 m against
the capsule's ``MANNEQUIN_ARM_RADIUS`` of 0.04. The forearm does taper (0.027 m at the wrist to
0.059 m at the elbow), so no single radius can match a constant capsule everywhere; the printed
profile is there to pick a stretch from.
"""

# glTF texture-slot -> (USD shader name, colour space). Occlusion shares the metallic-roughness
# image, which is the usual ORM packing, so it is read as a channel of that texture rather than
# extracted twice.
_TEXTURE_ROLES = {
    "baseColor": "sRGB",
    "metallicRoughness": "raw",
    "normal": "raw",
}


def _canonical_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Derive the limb's own axes, its wrist, and the scale to metres, from the geometry.

    The sculpt arrives in an arbitrary pose -- its long axis is a diagonal in the source frame, and
    the two Sketchfab wrapper nodes cancel to identity, so nothing in the file says which way is
    along the arm. Everything below is measured off the mesh instead of guessed.

    Returns:
        ``(basis, origin, scale, landmarks)`` where ``basis`` has the new X, Y and Z axes as
        columns (so ``p_new = (p - origin) @ basis * scale``), ``origin`` is the wrist in source
        coordinates, and ``landmarks`` records the measurements for the USD's metadata.
    """
    centre = points.mean(axis=0)
    _, _, principal = np.linalg.svd(points - centre, full_matrices=False)
    # the limb is far longer than it is thick, so the first principal axis is along it. It points
    # towards the elbow: the hand sits at low ``t`` (verified by the radius profile below, which
    # finds the wrist waist on that side).
    long_axis = principal[0]
    t = (points - centre) @ long_axis

    # Wrist: the narrowest cross-section between the hand and the forearm. Radius is measured about
    # each slice's own centroid, so the arm's natural curve is not read as extra thickness.
    def slice_radius(mask: np.ndarray) -> float:
        local = points[mask] - points[mask].mean(axis=0)
        perpendicular = local - np.outer(local @ long_axis, long_axis)
        return float(np.linalg.norm(perpendicular, axis=1).mean())

    span = t.max() - t.min()
    edges = np.linspace(t.min(), t.max(), 65)
    waist = None
    for low, high in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (low + high)
        # search only the band where a wrist can be: past the fingers, short of mid-forearm
        if not (t.min() + 0.20 * span < mid < t.min() + 0.50 * span):
            continue
        mask = (t >= low) & (t < high)
        if mask.sum() < 20:
            continue
        radius = slice_radius(mask)
        if waist is None or radius < waist[1]:
            waist = (mid, radius)
    wrist_t, wrist_radius = waist
    scale = ADULT_HAND_LENGTH / (wrist_t - t.min())

    # Palm plane: within the hand alone the mesh is a flat slab, so its thinnest principal axis is
    # the palm normal. Fitted on the hand only -- including the forearm would drown the signal.
    hand = t < wrist_t - 0.02 * span
    hand_points = points[hand]
    hand_centre = hand_points.mean(axis=0)
    _, _, hand_axes = np.linalg.svd(hand_points - hand_centre, full_matrices=False)
    palm_normal = hand_axes[2]

    # Which side of that plane is the palm: the fingers curl towards it, so the fingertips sit
    # palm-side of the metacarpals.
    along = (hand_points - hand_centre) @ hand_axes[0]
    across = (hand_points - hand_centre) @ palm_normal
    tips = along < np.percentile(along, 4)
    metacarpals = (along > np.percentile(along, 55)) & (along < np.percentile(along, 85))
    if across[tips].mean() > across[metacarpals].mean():
        palm_normal = -palm_normal  # make it point dorsally (away from the palm)

    # X along the limb towards the fingertips, Z out of the back of the hand, Y completing a
    # right-handed frame. Z is re-orthogonalised against X because the palm plane and the limb axis
    # are not exactly perpendicular on a sculpt of a relaxed arm.
    x_axis = -long_axis
    z_axis = palm_normal - (palm_normal @ x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack([x_axis, y_axis, z_axis])
    assert np.linalg.det(basis) > 0, "basis must stay right-handed or the winding order flips"

    # The forearm tapers, so one radius cannot describe it. Sampled every 4 cm from the wrist so
    # the caller can pick the stretch that best stands in for a constant-radius capsule.
    profile = {}
    for start in np.arange(0.0, (t.max() - wrist_t) * scale, 0.04):
        mask = (t >= wrist_t + start / scale) & (t < wrist_t + (start + 0.04) / scale)
        if mask.sum() < 20:
            continue
        profile[f"{start:.2f}"] = round(slice_radius(mask) * scale, 5)

    # One radius standing for the whole forearm, which is what the analytic capsule in the rewards
    # has to be given. Averaged over the slices rather than over the points, so the densely
    # sculpted end does not outvote the rest.
    mean_forearm_radius = float(np.mean(list(profile.values())))

    origin = centre + wrist_t * long_axis
    landmarks = {
        "handLength_m": float((wrist_t - t.min()) * scale),
        "forearmLength_m": float((t.max() - wrist_t) * scale),
        "totalLength_m": float(span * scale),
        "wristRadius_m": float(wrist_radius * scale),
        "meanForearmRadius_m": mean_forearm_radius,
        "scaleFromSourceUnits": float(scale),
        "forearmRadiusByDistanceFromWrist_m": profile,
    }
    return basis, origin, scale, landmarks


def _write_textures(glb: Glb) -> dict[str, str]:
    """Extract the embedded images to files beside the USD, returning relative asset paths."""
    directory = os.path.join(HERE, TEXTURE_DIRNAME)
    os.makedirs(directory, exist_ok=True)

    material = glb.json["materials"][0]
    slots = {
        "baseColor": material["pbrMetallicRoughness"]["baseColorTexture"]["index"],
        "metallicRoughness": material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"],
        "normal": material["normalTexture"]["index"],
    }

    paths: dict[str, str] = {}
    for role, texture_index in slots.items():
        image_index = glb.json["textures"][texture_index]["source"]
        payload, extension = glb.image_bytes(image_index)
        name = f"arm_and_hand_{role}{extension}"
        with open(os.path.join(directory, name), "wb") as f:
            f.write(payload)
        # relative, so moving the folder keeps the material intact
        paths[role] = f"./{TEXTURE_DIRNAME}/{name}"
    return paths


def _build_material(stage: Usd.Stage, textures: dict[str, str]) -> UsdShade.Material:
    """A UsdPreviewSurface wired up the way the glTF metallic-roughness model specifies."""
    material = UsdShade.Material.Define(stage, f"{ROOT_PRIM}/Materials/Hand_and_arm_10")

    surface = UsdShade.Shader.Define(stage, f"{material.GetPath()}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(role: str) -> UsdShade.Shader:
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/{role}Texture")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(textures[role])
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
        # the GLB's single sampler is REPEAT/REPEAT (glTF enum 10497) on both axes
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(_TEXTURE_ROLES[role])
        for channel in ("r", "g", "b", "a"):
            shader.CreateOutput(channel, Sdf.ValueTypeNames.Float)
        shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        return shader

    base_colour = texture("baseColor")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_colour.ConnectableAPI(), "rgb"
    )

    # ORM packing: occlusion in red, roughness in green, metallic in blue -- one image, three
    # inputs, which is exactly how the glTF references it (occlusionTexture and
    # metallicRoughnessTexture point at the same source).
    orm = texture("metallicRoughness")
    surface.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(orm.ConnectableAPI(), "r")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(orm.ConnectableAPI(), "g")
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(orm.ConnectableAPI(), "b")

    # tangent-space normal map: the texture holds 0..1, the shader wants -1..1
    normal = texture("normal")
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(), "rgb")

    return material


def _author_physics(stage: Usd.Stage, mesh: UsdGeom.Mesh) -> None:
    """Bake a static triangle-mesh collider into the canonical asset.

    It has to be baked rather than left to the spawner. Isaac Lab's ``UsdFileCfg`` only ever calls
    the ``modify_*`` writers, and every one of them bails out early -- returning ``False`` in
    silence -- when the schema is not already applied::

        if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
            return False

    So ``rigid_props`` and ``collision_props`` on the spawner can *adjust* physics an asset already
    declares, but cannot bring any into existence. This mirrors the layout the mesh converter left
    on ``hand.usd`` and that Isaac Lab is known to load: body schemas on the root Xform, collision
    schemas on the mesh underneath.

    Nothing here decides *static*. Both the rigid-body and collision writers are decorated with
    ``apply_nested``, so the environment's ``kinematic_enabled`` reaches the root and its
    ``collision_enabled`` reaches the mesh, leaving the asset usable either way.
    """
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(ARM_MASS)

    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(MESH_APPROXIMATION)

    # ``NewtonSDFCollisionAPI`` is a Newton schema, unknown to the plain ``usd-core`` wheel this
    # script runs under, so ``Usd.Prim.ApplyAPI`` cannot resolve it. The applied-schema list is
    # ordinary prim metadata, though, and Newton's importer only ever asks
    # ``prim.HasAPI("NewtonSDFCollisionAPI")`` -- so appending the token by hand and authoring the
    # namespaced attribute produces exactly the prim the importer looks for, without needing the
    # schema registered here.
    prim = mesh.GetPrim()
    applied = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    existing = list(applied.prependedItems) + list(applied.explicitItems)
    if "NewtonSDFCollisionAPI" not in existing:
        listop = Sdf.TokenListOp()
        listop.prependedItems = existing + ["NewtonSDFCollisionAPI"]
        prim.SetMetadata("apiSchemas", listop)
    prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int).Set(SDF_MAX_RESOLUTION)


def _write_arm_frame_wrapper(landmarks: dict) -> dict:
    """Emit the task-frame layer: a reference to the canonical asset, shifted onto the forearm.

    A wrapper rather than a second copy -- it is a few hundred bytes, and the mesh, its textures and
    the baked physics schemas all resolve through the reference, so there is exactly one of each on
    disk and the canonical asset keeps its own wrist-origin frame.
    """
    offset = _forearm_midpoint_offset(landmarks)

    stage = Usd.Stage.CreateNew(ARM_FRAME_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/arm_and_hand_arm_frame")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(f"./{os.path.basename(USD_PATH)}")

    # The offset goes on the *mesh*, not on this layer's default prim, and that placement is
    # load-bearing. Isaac Lab's spawner authors ``xformOp:translate`` on the prim it references the
    # asset onto, in a stronger layer -- USD replaces the referenced opinion rather than composing
    # with it, so an offset parked on the default prim is silently overwritten by the spawn
    # position and the frame shift vanishes. Pushed one level down it is out of reach of that
    # write, and since the collider lives on the same prim the collision shape moves with it.
    #
    # +X runs wrist->fingertips, so sliding the mesh towards +X brings a forearm point back to the
    # origin.
    geom = UsdGeom.Xformable(stage.GetPrimAtPath("/arm_and_hand_arm_frame/geom"))
    geom.AddTranslateOp().Set(Gf.Vec3d(offset, 0.0, 0.0))

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
        root.GetPrim()
    ).ComputeAlignedRange()

    # The capsule the observation is asked to assume: elbow to wrist, centred on this frame's
    # origin. ``radius`` is the mean over the forearm because one number has to stand for a limb
    # that runs 0.026 m at the wrist to 0.059 m at the elbow; ``length`` is the cylindrical section,
    # so ``length + 2 * radius`` recovers the full elbow-to-wrist span exactly.
    # The limb's real centre line in this frame, sampled every 2 cm. A capsule's centre line is its
    # root axis; a sculpted arm's is not, and a band drawn on the axis hangs off the surface where
    # the two diverge. Consumed by ``band_centerline_offsets`` on the goal command.
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/arm_and_hand_arm_frame/geom"))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) + np.array([offset, 0.0, 0.0])
    centerline = []
    for station in np.arange(-0.20, 0.21, 0.02):
        slab = points[np.abs(points[:, 0] - station) < 0.01]
        if len(slab) < 30:
            continue
        centroid = slab[:, 1:].mean(axis=0)
        centerline.append([round(float(station), 3), round(float(centroid[0]), 4), round(float(centroid[1]), 4)])

    span = landmarks["forearmLength_m"]
    radius = landmarks["meanForearmRadius_m"]
    frame = {
        "centerlineOffsets": centerline,
        "centerlineMaxDrift_m": round(
            float(max(np.hypot(y, z) for _, y, z in centerline)), 5
        ),
        "originFromWrist_m": offset,
        # what the spawner needs so the limb rests on the table rather than through it
        "restHeight_m": float(-bounds.GetMin()[2]),
        "elbowX_m": float(bounds.GetMin()[0]),
        "fingertipX_m": float(bounds.GetMax()[0]),
        "capsuleRadius_m": radius,
        "capsuleLength_m": span - 2.0 * radius,
    }
    stage.SetMetadata(
        "customLayerData",
        {
            "generator": "convert_arm_and_hand.py",
            "frame": (
                f"Task frame for the sheet drape: origin {offset:.4f} m up the forearm from the "
                "wrist, which is the midpoint of the elbow-to-wrist segment, +X towards the "
                "fingertips, +Z out of the back of the hand. The equivalent analytic capsule is "
                "capsuleRadius_m / capsuleLength_m, centred here. Set the spawner's z to "
                "restHeight_m to seat the limb on the table."
            ),
            "landmarks": json.dumps({**landmarks, **frame}),
        },
    )
    stage.GetRootLayer().Save()
    return frame


def main() -> None:
    glb = Glb(GLB_PATH)
    primitive = glb.json["meshes"][0]["primitives"][0]
    if primitive.get("mode", 4) != 4:
        raise NotImplementedError("only triangle meshes are supported")

    attributes = primitive["attributes"]
    points = glb.accessor(attributes["POSITION"]).astype(np.float64)
    normals = glb.accessor(attributes["NORMAL"]).astype(np.float64)
    tangents = glb.accessor(attributes["TANGENT"]).astype(np.float64)
    uvs = glb.accessor(attributes["TEXCOORD_0"]).astype(np.float64)
    indices = glb.accessor(primitive["indices"]).astype(np.int32)

    # The two Sketchfab wrapper nodes are inverse rotations that cancel to identity, so the raw
    # positions are already the mesh's world coordinates. Asserted rather than assumed -- a
    # re-export from Sketchfab could easily bake a different pair.
    world = glb.node_world_matrices()
    mesh_node = next(i for i, n in enumerate(glb.json["nodes"]) if n.get("mesh") == 0)
    transform = world[mesh_node]
    if not np.allclose(transform, np.eye(4), atol=1e-6):
        points = points @ transform[:3, :3] + transform[3, :3]
        rotation = transform[:3, :3] / np.cbrt(abs(np.linalg.det(transform[:3, :3])))
        normals = normals @ rotation
        tangents[:, :3] = tangents[:, :3] @ rotation

    basis, origin, scale, landmarks = _canonical_frame(points)

    points = ((points - origin) @ basis) * scale
    normals = normals @ basis
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangents[:, :3] = tangents[:, :3] @ basis

    # glTF puts UV (0,0) at the top-left of the image, USD at the bottom-left
    st = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1)

    stage = Usd.Stage.CreateNew(USD_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, f"{ROOT_PRIM}/geom")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, np.int32)))
    mesh.CreateExtentAttr(
        Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)]).astype(np.float32))
    )

    # "none" rather than USD's catmullClark default. A subdivided 34k-triangle sculpt would be
    # re-tessellated on load in every environment, and the sculpt is already dense enough that the
    # smoothing buys nothing.
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    # the glTF material sets doubleSided, and a sculpt with open ends needs it
    mesh.CreateDoubleSidedAttr().Set(bool(glb.json["materials"][0].get("doubleSided", False)))

    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    primvars = UsdGeom.PrimvarsAPI(mesh)
    uv_primvar = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv_primvar.Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
    # UsdPreviewSurface does not consume tangents, but they are authored data from the sculpt and a
    # renderer or exporter that wants them should not have to re-derive them from the UVs
    tangent_primvar = primvars.CreatePrimvar(
        "tangents", Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex
    )
    tangent_primvar.Set(Vt.Vec4fArray.FromNumpy(tangents.astype(np.float32)))

    material = _build_material(stage, _write_textures(glb))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    _author_physics(stage, mesh)

    # Provenance: the model is CC-BY-4.0, so the attribution has to travel with the asset.
    extras = glb.json["asset"].get("extras", {})
    stage.SetMetadata(
        "customLayerData",
        {
            "source": os.path.basename(GLB_PATH),
            "title": extras.get("title", ""),
            "author": extras.get("author", ""),
            "license": extras.get("license", ""),
            "sourceUrl": extras.get("source", ""),
            "generator": "convert_arm_and_hand.py",
            "frame": (
                "+X wrist->fingertips, +Z out of the back of the hand, +Y right-handed; "
                "origin at the wrist on the limb axis. Metres, Z-up, scale baked in."
            ),
            "landmarks": json.dumps(landmarks),
        },
    )

    stage.GetRootLayer().Save()

    frame = _write_arm_frame_wrapper(landmarks)

    print(f"wrote {USD_PATH}")
    print(f"wrote {ARM_FRAME_PATH}")
    for key, value in frame.items():
        if isinstance(value, list):
            print(f"  {key:24s} {len(value)} samples")
        else:
            print(f"  {key:24s} {value:.4f}")
    print(f"  vertices {len(points)}  triangles {len(indices) // 3}")
    for key, value in landmarks.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for distance, radius in value.items():
                print(f"    {distance} m from wrist -> radius {radius:.4f} m")
        else:
            print(f"  {key:24s} {value:.4f}")


if __name__ == "__main__":
    main()
