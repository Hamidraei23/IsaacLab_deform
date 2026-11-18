# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations describing the sheet's shape to the policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from .gripper_geometry import finger_half_width

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def sheet_key_points(
    env: ManagerBasedRLEnv,
    resolution: tuple[int, int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Nine landmark nodes of the sheet, in the robot's root frame [m]. Shape ``(num_envs, 27)``.

    The landmarks are the four corners, the midpoint of each of the four edges, and the centre::

        0---1---2        0,2,6,8  corners
        |       |        1,3,5,7  edge midpoints
        3   4   5        4        centre
        |       |
        6---7---8

    Chosen instead of a random sample of nodes because they are *identified* points: index 0 is the
    same physical corner on every step and in every environment, so the policy can learn "this
    corner is the one in my gripper" and read the sheet's pose and fold state from how the nine
    move relative to one another. A random sample carries the same kind of information but
    scrambles which node is which between resets, so none of it is addressable.

    Corners and edge midpoints are also the parts that matter for this task: the grasp reward only
    accepts a pinch near an edge, and the coverage reward is driven by where the sheet's extremities
    fall on the band.

    Args:
        env: The environment.
        resolution: The sheet mesh's ``(x, y)`` element counts, as given to its spawner. The node
            grid is one larger in each direction.
        asset_cfg: The deformable sheet.
        robot_cfg: The robot whose root frame the points are expressed in.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cols = resolution[0] + 1
    rows = resolution[1] + 1
    # integer division: on an even-sided grid the "midpoint" is the node just below centre, which
    # is the closest thing to a midpoint that is an actual node
    mid_row, mid_col = rows // 2, cols // 2
    picks = [
        (0, 0), (0, mid_col), (0, cols - 1),
        (mid_row, 0), (mid_row, mid_col), (mid_row, cols - 1),
        (rows - 1, 0), (rows - 1, mid_col), (rows - 1, cols - 1),
    ]
    node_ids = torch.tensor([r * cols + c for r, c in picks], device=env.device)

    points_w = asset.data.nodal_pos_w.torch[:, node_ids, :]
    num_points = len(node_ids)
    root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
    root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
    points_b, _ = subtract_frame_transforms(
        root_pos_w.reshape(-1, 3), root_quat_w.reshape(-1, 4), points_w.reshape(-1, 3)
    )
    return points_b.view(env.num_envs, -1)


def tool_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.107),
) -> torch.Tensor:
    """The tool point -- ``tool_offset`` along the hand body, i.e. the point the IK action controls --
    in the robot's root frame [m]. Shape ``(num_envs, 3)``."""
    robot = env.scene[robot_cfg.name]
    pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index(hand_body_name)]
    offset = torch.tensor(tool_offset, device=pose.device, dtype=pose.dtype).expand(len(pose), 3)
    tool_w = pose[:, :3] + quat_apply(pose[:, 3:7], offset)
    tool_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, tool_w)
    return tool_b


def sheet_key_points_from_tool(
    env: ManagerBasedRLEnv,
    resolution: tuple[int, int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.107),
) -> torch.Tensor:
    """:func:`sheet_key_points` as offsets from the tool point: robot-frame axes, origin between the
    fingertips. Shape ``(num_envs, 27)``.

    The nine landmarks keep their identities and the robot frame's axes, but each is ``p - p_tool``,
    so the policy reads "this corner is 5 cm below my fingers" directly, and a corner it holds reads
    as roughly zero. The robot's own position cancels in the subtraction (user request: no distance
    to the base in the observation); its orientation still defines the axes.
    """
    points = sheet_key_points(env, resolution, asset_cfg, robot_cfg).view(env.num_envs, -1, 3)
    tool = tool_position_b(env, robot_cfg, hand_body_name, tool_offset)
    return (points - tool[:, None, :]).reshape(env.num_envs, -1)


def target_from_tool(
    env: ManagerBasedRLEnv,
    command_name: str = "deformable_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.107),
) -> torch.Tensor:
    """The goal command with its position taken from the tool point: ``(p_goal - p_tool, quat_goal)``,
    both in the robot frame's axes. Shape ``(num_envs, 7)``."""
    command = env.command_manager.get_command(command_name)
    tool = tool_position_b(env, robot_cfg, hand_body_name, tool_offset)
    return torch.cat([command[:, :3] - tool, command[:, 3:]], dim=-1)


def finger_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
) -> torch.Tensor:
    """The gripper opening: the sum of the finger joint positions [m]. Shape ``(num_envs, 1)``.

    The one piece of proprioception the ``action_history`` observation group carries. It tells a
    policy whether its fingers are open, closing or shut without giving it any pose relative to
    the robot base, which is what that group leaves out on purpose.

    .. warning::
        As declared in ``ActionHistoryObsCfg`` (``ObsTerm(func=finger_width)``, no ``params``) the
        default ``asset_cfg`` is never resolved by the manager, so ``joint_ids`` is ``slice(None)``
        and the term sums **all nine** joint positions, not the two fingers. Measured 2026-10-26
        at reset: 0.625 (arm joints) against a true finger sum of 0.069. Every ``action_history``
        checkpoint so far (DAgger v3, best_of_sac, D2, the yaw-curriculum runs) was trained on that
        value, so it is kept as is; :func:`gripper_opening` is the measured version.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos.torch[:, asset_cfg.joint_ids].sum(dim=-1, keepdim=True)


def gripper_opening(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
) -> torch.Tensor:
    """The physical gap between the finger pads [m]: twice
    :func:`~.gripper_geometry.finger_half_width`. Shape ``(num_envs, 1)``.

    For the Franka (``asset_cfg`` passed in ``params`` so it resolves) that is the sum of the two
    prismatic finger joints, 0.08 m open. For a gripper whose joints measure angles -- the 2F-85,
    whose task declares ``finger_pad_geometry`` -- it is the gap between the pad faces, so the
    channel has the same meaning and range on both arms. The UR10e task uses it in place of
    :func:`finger_width`.
    """
    return (2.0 * finger_half_width(env, asset_cfg)).unsqueeze(-1)
