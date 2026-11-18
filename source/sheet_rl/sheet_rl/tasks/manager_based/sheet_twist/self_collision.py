"""Capsule self-proximity test for the UR10e.

The stock asset sets ``enabled_self_collisions=False``, so the solver never tests arm
links against each other and they pass through freely. The joint limits cannot stand in
for it either: five of the six are +/-360 degrees and the elbow is +/-180 in the stock
asset, well wide enough to fold the wrist into the column. This measures the geometry instead.

Each arm link is one capsule whose axis runs from its own joint anchor to the next
joint's anchor, both read off the USD (``shoulder_lift`` -> ``elbow`` is -0.6127 m along
the link's x, ``elbow`` -> ``wrist_1`` is -0.5716 m with the 0.1742 m d4 offset). Radii
are the collision meshes' cross-sections. Only pairs two or more joints apart are
tested; neighbours are always in contact by construction.

The one neighbour pair that can still collide is the upper arm and forearm, when the elbow
folds the forearm back along the upper arm. Their relative pose is the elbow angle alone, so
that pair is a joint-angle test (``ELBOW_COLLISION_DEG``) rather than a capsule pair, and the
task narrows the elbow's limit short of it (``ELBOW_LIMIT_DEG``, authored in ``ur10e_asset``).

The arm against the table (``arm_into_table``) cannot use these capsules: the joint housings
stand out up to 0.27 m from the link axes. It tests the collision meshes' hulls instead.
"""

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

ELBOW_COLLISION_DEG = 169.
"""|elbow| from which the forearm and upper arm interpenetrate. Measured on the USD collision meshes,
1 degree steps, points beyond 0.2 m of the elbow (the two housings interleave around the joint at every
angle): none inside the other link up to 168 degrees either way, 103+ from +169 and 166+ from -170."""

ELBOW_LIMIT_DEG = 160.
"""The elbow's position limit in the task. The stock asset allows +/-180; 160 keeps 9 degrees for the
solver's soft limit, which fast joints overshoot (2.7 degrees pushed slowly, about 19 at 13 rad/s), and
leaves 90 mm between the tubes."""

# name -> (body, axis start, axis end, radius), all in body coordinates [m].
CAPSULES = {
    # Base column and shoulder, which share one vertical axis up to the shoulder pan.
    "base": ("base_link", (0., 0., 0.), (0., 0., .22), .090),
    "upper_arm": ("upper_arm_link", (0., 0., 0.), (-.6127, 0., 0.), .078),
    "forearm": ("forearm_link", (0., 0., 0.), (-.5716, 0., .1742), .060),
    # Wrist 3 plus the 2F-85 out to the fingertips.
    "tool": ("wrist_3_link", (0., 0., 0.), (0., 0., .21), .055),
}
PAIRS = (("base", "forearm"), ("base", "tool"), ("upper_arm", "tool"))


def _segment_distance(p1, q1, p2, q2):
    """Shortest distance between two batched line segments (Ericson, clamped twice)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    s = ((b * f - c * e) / (a * e - b * b).clamp_min(1e-12)).clamp(0., 1.)
    t = ((b * s + f) / e.clamp_min(1e-12)).clamp(0., 1.)
    s = ((b * t - c) / a.clamp_min(1e-12)).clamp(0., 1.)
    return (p1 + s[..., None] * d1 - p2 - t[..., None] * d2).norm(dim=-1)


def _world_segment(robot, body, start, end):
    pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index(body)]
    position, rotation = pose[:, :3], pose[:, 3:7]
    offsets = torch.tensor([start, end], device=pose.device).expand(len(pose), 2, 3)
    return (position + quat_apply(rotation, offsets[:, 0]),
            position + quat_apply(rotation, offsets[:, 1]))


def self_collision_clearance(env, asset_cfg=SceneEntityCfg("robot")):
    """Signed gap [m] to the nearest tested pair; negative means the capsules overlap."""
    robot = env.scene[asset_cfg.name]
    segments = {name: _world_segment(robot, *spec[:3]) for name, spec in CAPSULES.items()}
    gaps = [_segment_distance(*segments[a], *segments[b]) - CAPSULES[a][3] - CAPSULES[b][3]
            for a, b in PAIRS]
    return torch.stack(gaps, -1).amin(-1)


def elbow_folded(env, asset_cfg=SceneEntityCfg("robot")):
    """Whether the forearm has folded into the upper arm: an elbow overshoot past its limit."""
    robot = env.scene[asset_cfg.name]
    elbow = robot.data.joint_pos.torch[:, robot.joint_names.index("elbow_joint")]
    return elbow.abs() > math.radians(ELBOW_COLLISION_DEG)


def arm_self_collision(env, margin=.01, asset_cfg=SceneEntityCfg("robot")):
    """Terminate when two non-adjacent arm capsules come within ``margin`` of touching, or the
    elbow folds the forearm into the upper arm."""
    return (self_collision_clearance(env, asset_cfg) < margin) | elbow_folded(env, asset_cfg)


TABLE_LINKS = ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
"""The arm links whose contact with the table ends an episode. The base and shoulder only turn about
the vertical pan axis above it; the gripper below wrist 3 is the ``ee_below_table`` termination's.

The upper arm is exempt (user request, 2026-10-28): its collision mesh includes the joint-2 housing,
whose lowest point sits at a fixed 0.104 m above the table top whatever the arm does, so raising the
table relative to the robot by more than ~9 cm would end every episode at its first step. The upper
arm can only reach the table elbow first, and the elbow end is the forearm's, which is still checked."""


def link_hull_points(usd_path, links, variants=None):
    """Vertices of each link's collision-mesh convex hull, in the link's frame: {link: (P, 3) array}.

    A link's lowest point in any pose is one of these vertices, so they are all a height test needs.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    from scipy.spatial import ConvexHull

    from isaaclab.utils.assets import retrieve_file_path

    stage = Usd.Stage.Open(retrieve_file_path(usd_path))
    for name, value in (variants or {}).items():
        stage.GetDefaultPrim().GetVariantSet(name).SetVariantSelection(value)
    cache = UsdGeom.XformCache()
    hulls = {}
    for link in links:
        link_prim = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild(link))
        if not link_prim.IsValid():
            raise ValueError(f"No link {link} in {usd_path}")
        to_link = cache.GetLocalToWorldTransform(link_prim).GetInverse()
        points = []
        for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh) and "/collisions/" in f"{prim.GetPath()}/":
                to_frame = np.array(cache.GetLocalToWorldTransform(prim) * to_link)  # row-vector convention
                vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                points.append(vertices @ to_frame[:3, :3] + to_frame[3, :3])
        if not points:
            raise ValueError(f"No collision mesh under {link} in {usd_path}")
        points = np.concatenate(points)
        hulls[link] = points[ConvexHull(points).vertices]
    return hulls


class arm_into_table(ManagerTermBase):
    """Terminate, as a failure, when any of the arm's moving links comes within ``margin`` of the
    table top or below it.

    The test is against the plane of the table top, not its footprint: an arm below the table's
    level has failed whether it went through the table or round its edge. Link geometry is the
    hull of each link's USD collision mesh (``link_hull_points``), posed by the simulator.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        links = cfg.params.get("links", TABLE_LINKS)
        self._robot = env.scene[asset_cfg.name]
        spawn = getattr(env.cfg.scene, asset_cfg.name).spawn
        hulls = link_hull_points(spawn.usd_path, links, getattr(spawn, "variants", None))
        self._body_ids = [self._robot.body_names.index(link) for link in links]
        # pad every hull to one length by repeating its first vertex, so all links pose in one batch
        size = max(len(points) for points in hulls.values())
        self._points = torch.stack([
            torch.tensor([*points, *[points[0]] * (size - len(points))], dtype=torch.float32)
            for points in (hulls[link] for link in links)
        ]).to(env.device)
        table = env.cfg.scene.table
        self.table_top = table.init_state.pos[2] + table.spawn.size[2] / 2

    def clearance(self, env):
        """Height [m] of the arm's lowest point above the table top; negative when below it."""
        pose = self._robot.data.body_link_pose_w.torch[:, self._body_ids]  # (n, links, 7)
        z_row = matrix_from_quat(pose[..., 3:7])[..., 2, :]  # each link's world-z row, (n, links, 3)
        z = pose[..., 2, None] + torch.einsum("nlj,lpj->nlp", z_row, self._points)
        return z.amin(dim=(1, 2)) - env.scene.env_origins[:, 2] - self.table_top

    def __call__(self, env, margin=.01, links=TABLE_LINKS, asset_cfg=SceneEntityCfg("robot")):
        return self.clearance(env) < margin
