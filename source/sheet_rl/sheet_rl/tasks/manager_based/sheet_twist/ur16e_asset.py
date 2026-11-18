"""The stock UR16e with the UR10e task's Robotiq 2F-85, composed from the Isaac Sim assets.

Isaac Sim ships the UR16e without the UR10e's ``Gripper`` variant set, so :func:`compose_ur16e_2f85` writes a
small layer that references the stock UR16e and mounts the same 2F-85 payload the way the UR10e's
``Robotiq_2f_85`` variant does (``ur10e/configuration/ur10e_Gripper_2F_85.usd``): an ``ee_link`` holding
``Robotiq_2F_85`` (its own articulation root removed) and a fixed ``joints/ee_joint`` from ``wrist_3_link`` to
the gripper's ``base_link``. The joint's frames and the gripper's pose relative to ``wrist_3_link`` are read
off the composed UR10e, so the tool point (``ur10e_mdp.TOOL_POS``/``TOOL_ROT``), the finger bodies and their
prim paths are the UR10e task's.

The two arms share joint names, axes, limits (velocity 120/120/180/180/180/180 deg/s, effort 330/330/150/56/56/56
Nm, +/-360 deg, elbow +/-180) and the base and wrist geometry; only the upper arm (0.4784 against 0.6127 m) and
forearm (0.36 against 0.57155 m) differ, with their meshes and masses. The layer points at Isaac Lab's local
copies of the remote assets, so it is rewritten at every spawn (:func:`spawn_ur16e`) rather than shipped.
"""

import os

from .ur10e_asset import limit_elbow, spawn_ur10e
from .ur16e_self_collision import UR16E_ELBOW_LIMIT_DEG

UR16E_USD_PATH = os.path.join(os.path.expanduser("~"), ".cache", "sheet_rl", "ur16e_robotiq_2f_85.usda")
"""Where the composed UR16e + 2F-85 layer is written; the task's ``robot.spawn.usd_path``."""

UR16E_ASSET = "Robots/UniversalRobots/ur16e/ur16e.usd"
UR10E_ASSET = "Robots/UniversalRobots/ur10e/ur10e.usd"
GRIPPER_ASSET = "Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
UR10E_MOUNT_LAYER = "configuration/ur10e_Gripper_2F_85.usd"
"""The UR10e's 2F-85 variant payload, relative to ``ur10e.usd``: the overs it puts on the gripper are copied."""


def compose_ur16e_2f85(path=UR16E_USD_PATH):
    """Write the UR16e + 2F-85 layer to ``path`` (atomically) and return ``path``."""
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    arm = retrieve_file_path(f"{ISAAC_NUCLEUS_DIR}/{UR16E_ASSET}")
    ur10e = retrieve_file_path(f"{ISAAC_NUCLEUS_DIR}/{UR10E_ASSET}")
    gripper = retrieve_file_path(f"{ISAAC_NUCLEUS_DIR}/{GRIPPER_ASSET}")

    source = Usd.Stage.Open(ur10e)
    source.GetDefaultPrim().GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_85")
    target = Usd.Stage.Open(arm)
    cache = UsdGeom.XformCache()
    # The gripper's pose in wrist_3_link's frame on the UR10e, placed on the UR16e's wrist_3_link (row vectors).
    mount = (cache.GetLocalToWorldTransform(source.GetPrimAtPath("/ur10e/ee_link"))
             * cache.GetLocalToWorldTransform(source.GetPrimAtPath("/ur10e/wrist_3_link")).GetInverse())
    cache.Clear()
    ee_local = mount * cache.GetLocalToWorldTransform(target.GetPrimAtPath("/ur16e/wrist_3_link"))

    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.defaultPrim = "ur16e"
    root = Sdf.CreatePrimInLayer(layer, "/ur16e")
    root.specifier, root.typeName = Sdf.SpecifierDef, "Xform"
    root.referenceList.Prepend(Sdf.Reference(arm))
    # The variant payload's overs on the gripper: the deleted articulation root, joint states, joint APIs.
    mount_layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(ur10e), UR10E_MOUNT_LAYER))
    if not Sdf.CopySpec(mount_layer, "/ur10e/ee_link", layer, "/ur16e/ee_link"):
        raise RuntimeError(f"Could not copy the 2F-85 mount from {mount_layer.identifier}")
    ee = layer.GetPrimAtPath("/ur16e/ee_link")
    ee.specifier, ee.typeName = Sdf.SpecifierDef, "Xform"
    ee.payloadList.ClearEdits()
    ee.payloadList.Prepend(Sdf.Payload(gripper))
    for name in [p.name for p in ee.properties if p.name.startswith("xformOp")]:
        ee.RemoveProperty(ee.properties[name])

    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(target))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/ur16e/ee_link"))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(ee_local)
    joint = stage.DefinePrim("/ur16e/joints/ee_joint", "PhysicsFixedJoint")
    stage.GetPrimAtPath("/ur16e/joints").SetSpecifier(Sdf.SpecifierOver)
    template = source.GetPrimAtPath("/ur10e/joints/ee_joint")
    for attribute in template.GetAuthoredAttributes():
        if attribute.HasValue():
            joint.CreateAttribute(attribute.GetName(), attribute.GetTypeName()).Set(attribute.Get())
    for relationship in template.GetAuthoredRelationships():
        joint.CreateRelationship(relationship.GetName()).SetTargets(
            [Sdf.Path(str(t).replace("/ur10e/", "/ur16e/", 1)) for t in relationship.GetTargets()])
    # CopySpec keeps target paths as written; re-root the copied ones.
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/ur16e/ee_link")):
        for relationship in prim.GetAuthoredRelationships():
            targets = relationship.GetTargets()
            if any(str(t).startswith("/ur10e/") for t in targets):
                relationship.SetTargets([Sdf.Path(str(t).replace("/ur10e/", "/ur16e/", 1)) for t in targets])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    scratch = f"{path}.{os.getpid()}.tmp.usda"
    layer.Export(scratch)
    os.replace(scratch, path)
    return path


def spawn_ur16e(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Compose the layer at ``cfg.usd_path``, then spawn it as the UR10e task spawns its arm (``spawn_ur10e``:
    box finger pads, uninstanced colliders), with the UR16e's own elbow limit in place of the UR10e's."""
    compose_ur16e_2f85(cfg.usd_path)
    prim = spawn_ur10e(prim_path, cfg, translation, orientation, **kwargs)
    limit_elbow(prim, UR16E_ELBOW_LIMIT_DEG)
    return prim
