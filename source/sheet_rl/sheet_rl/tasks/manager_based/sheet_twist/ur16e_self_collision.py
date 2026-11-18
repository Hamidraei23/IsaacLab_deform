"""The UR10e task's capsule self-proximity test (``self_collision``) on the UR16e.

The two arms share the base column, shoulder, wrists and the tube cross-sections -- measured on the collision
meshes, the upper arm's nearest surface to its anchor line is 0.0946 m against the UR10e's 0.0948, the housings
0.27 m on both, the forearm tube 0.058 against 0.061 -- so the capsules are the UR10e's (``CAPSULES``) with the
upper arm and forearm axes ending at the UR16e's joint anchors (``elbow`` -0.4784 m along x, ``wrist_1``
-0.36 m with the same 0.1742 m d4 offset). The pairs tested are the same (``PAIRS``).

The shorter forearm folds into the upper arm earlier: the same measurement that gives the UR10e's 169 degrees
(points beyond 0.2 m of the elbow inside the other link's hull, 1 degree steps; 103 points at +169 there) gives
162 degrees either way here, so the elbow limit keeps the UR10e's 9 degree margin below it.
"""

import math

import torch

from isaaclab.managers import SceneEntityCfg

from .self_collision import CAPSULES, PAIRS, _segment_distance, _world_segment

UR16E_ELBOW_COLLISION_DEG = 162.
"""|elbow| from which the UR16e's forearm and upper arm interpenetrate (the UR10e's is 169)."""

UR16E_ELBOW_LIMIT_DEG = 153.
"""The elbow's position limit on the UR16e, authored by ``ur16e_asset.spawn_ur16e``: the UR10e task's 9 degrees
below the fold for the solver's soft limit (the UR10e's is 160). The recorded UR10e tool paths need at most 139."""

UR16E_CAPSULES = {
    **CAPSULES,
    "upper_arm": ("upper_arm_link", (0., 0., 0.), (-.4784, 0., 0.), CAPSULES["upper_arm"][3]),
    "forearm": ("forearm_link", (0., 0., 0.), (-.36, 0., .1742), CAPSULES["forearm"][3]),
}
"""name -> (body, axis start, axis end, radius), all in body coordinates [m]."""


def ur16e_self_collision_clearance(env, asset_cfg=SceneEntityCfg("robot")):
    """Signed gap [m] to the nearest tested pair; negative means the capsules overlap."""
    robot = env.scene[asset_cfg.name]
    segments = {name: _world_segment(robot, *spec[:3]) for name, spec in UR16E_CAPSULES.items()}
    gaps = [_segment_distance(*segments[a], *segments[b]) - UR16E_CAPSULES[a][3] - UR16E_CAPSULES[b][3]
            for a, b in PAIRS]
    return torch.stack(gaps, -1).amin(-1)


def ur16e_elbow_folded(env, asset_cfg=SceneEntityCfg("robot")):
    """Whether the forearm has folded into the upper arm: an elbow overshoot past its limit."""
    robot = env.scene[asset_cfg.name]
    elbow = robot.data.joint_pos.torch[:, robot.joint_names.index("elbow_joint")]
    return elbow.abs() > math.radians(UR16E_ELBOW_COLLISION_DEG)


def ur16e_arm_self_collision(env, margin=.01, asset_cfg=SceneEntityCfg("robot")):
    """Terminate when two non-adjacent arm capsules come within ``margin`` of touching, or the
    elbow folds the forearm into the upper arm."""
    return (ur16e_self_collision_clearance(env, asset_cfg) < margin) | ur16e_elbow_folded(env, asset_cfg)
