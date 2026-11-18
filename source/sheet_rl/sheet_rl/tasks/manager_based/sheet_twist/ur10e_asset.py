"""Prepare the stock UR10e/2F-85 USD for Newton's full-surface cloth contacts and the task's elbow limit."""

from .self_collision import ELBOW_LIMIT_DEG

CLOTH_FINGERS = ("left_inner_finger", "right_inner_finger")
"""The 2F-85 bodies coupled to the cloth: the two inner fingers, which carry the pads -- as on the Franka,
whose cloth contact is its hand and two fingers with box pads. These are the robot entries of the coupler's
``proxy.bodies`` (see ``sheet_ur10e_env_cfg``); their colliders are boxes (:func:`box_fingers`)."""

SDF_SUBPATHS = ()
"""Robot parts that need an SDF for full-surface cloth contact. A mesh coupled to the cloth needs one or the
cloth passes straight through it; a box does not, so with box fingers this is empty. It used to be the whole
2F-85 and ``wrist_3_link`` (all meshes, the palm alone 31k vertices): 1.1 s per step at 256 environments,
against 0.33 s with only the two fingers' meshes coupled (2026-10-27)."""


def spawn_ur10e(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Sdf, Usd, UsdPhysics
    from isaaclab_newton.sim.schemas.schemas_cfg import NewtonSDFCollisionPropertiesCfg

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    # The stock collision meshes are instance proxies, so schemas cannot be authored
    # on them until their enclosing instances are made editable in this stage.
    sim_utils.make_uninstanceable(str(prim.GetPath()), stage=stage)
    # SDF goes on the coupled parts alone: an SDF on every arm link is wasted cooking.
    # ``modify_collision_properties`` applies to every collider beneath the path it is
    # given, so the paths have to be those parts, not the robot root.
    for subpath in SDF_SUBPATHS:
        path = f"{prim.GetPath()}/{subpath}"
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"No prim at {path} to author SDF collision on")
        sim_utils.modify_collision_properties(
            path, NewtonSDFCollisionPropertiesCfg(sdf_max_resolution=64), stage=stage,
        )
    # This asset's 2F-85 colliders are its visual meshes, and they carry
    # physics:approximation="convexHull" from the vendor. Newton queues anything asking for
    # a hull, then refuses it because a hull "replaces the mesh with non-mesh geometry" and
    # would drop the SDF we just authored -- which takes scene creation down. The importer
    # only runs that pass over shapes it is also drawing, so it is invisible headless and
    # fires the moment a viewer is attached. "none" is the token it reads as "leave the
    # mesh alone", which is what an SDF collider wants anyway.
    for descendant in Usd.PrimRange(prim):
        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not any(f"/{sub}/" in f"{descendant.GetPath()}/" for sub in SDF_SUBPATHS):
            continue
        approximation = descendant.GetAttribute("physics:approximation")
        if not approximation:
            approximation = descendant.CreateAttribute(
                "physics:approximation", Sdf.ValueTypeNames.Token,
            )
        approximation.Set("none")
    box_fingers(prim)
    limit_elbow(prim)
    return prim


def box_fingers(robot_prim, fingers=CLOTH_FINGERS):
    """Replace each finger's mesh colliders by boxes of their bounds, like the Franka's box finger pads.

    Each box is the mesh's bounding box in the finger's own frame, so the pad's inner contact face stays
    where it was (``finger_pad_geometry`` reads the pad gap from it) and the finger keeps its outline. The
    meshes stay as visuals with collision disabled; the boxes are guides, so they are not drawn. Masses are
    unchanged: the fingers carry explicit ``MassAPI`` values.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = robot_prim.GetStage()
    cache = UsdGeom.XformCache()
    boxes = []
    for finger in fingers:
        link = next((p for p in Usd.PrimRange(robot_prim)
                     if p.GetName() == finger and p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
        if link is None:
            raise RuntimeError(f"No {finger} body under {robot_prim.GetPath()} to box")
        to_link = cache.GetLocalToWorldTransform(link).GetInverse()
        meshes = [p for p in Usd.PrimRange(link) if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
        if not meshes:
            raise RuntimeError(f"No mesh colliders under {link.GetPath()} to replace")
        for mesh in meshes:
            to_frame = np.array(cache.GetLocalToWorldTransform(mesh) * to_link)  # row-vector convention
            points = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
            points = points @ to_frame[:3, :3] + to_frame[3, :3]
            low, high = points.min(0), points.max(0)
            UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr().Set(False)
            box = UsdGeom.Cube.Define(stage, link.GetPath().AppendChild(f"{mesh.GetName()}_box"))
            box.CreateSizeAttr().Set(1.0)
            box.AddTranslateOp().Set(Gf.Vec3d(*((low + high) / 2)))
            box.AddScaleOp().Set(Gf.Vec3f(*(high - low)))
            box.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            boxes.append(box)
    return boxes


def limit_elbow(robot_prim, limit_deg=ELBOW_LIMIT_DEG):
    """Author +/-``limit_deg`` on the robot's elbow joint, which the solver then enforces.

    The stock elbow allows +/-180 degrees, but the forearm folds into the upper arm from
    ``self_collision.ELBOW_COLLISION_DEG``, and with self-collision off nothing else stops it.
    """
    from pxr import Usd, UsdPhysics

    for descendant in Usd.PrimRange(robot_prim):
        if descendant.GetName() == "elbow_joint" and descendant.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(descendant)
            joint.CreateLowerLimitAttr().Set(-limit_deg)
            joint.CreateUpperLimitAttr().Set(limit_deg)
            return joint
    raise RuntimeError(f"No elbow_joint under {robot_prim.GetPath()} to limit")
