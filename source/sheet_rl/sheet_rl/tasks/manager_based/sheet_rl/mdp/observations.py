# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations describing the sheet's shape to the policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

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
