# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations specific to lifting the sheet out of its slot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .rewards import _early_release, _high_release

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def released_before_extraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode if the gripper opens while holding a sheet still in the slot.

    The attempt is over at that point: the policy gets one scored closure per episode, so a sheet
    dropped back into the slot cannot be picked up again for credit. Ending immediately stops the
    remaining steps being spent on a run that can no longer achieve anything.

    The flag is set by :class:`~.rewards.grasp_stage_reward`, which also charges the penalty. The
    reward manager runs before the termination manager inside a step, so it is never stale.
    """
    return _early_release(env)


def released_too_high(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode if the gripper is opened above the release ceiling in phase two.

    Dropping the sheet from height is not a worse drape, it is a different episode: the cloth
    lands wherever it falls, the placement reward that follows is decided by luck rather than by
    anything the policy did, and nothing it does for the remaining steps can pick the sheet back
    up. Ending on the spot, with the charge :class:`~.rewards.release_stage_reward` levies, keeps
    that noise out of the return and spends the samples on a fresh episode instead.

    A release *below* the ceiling deliberately does not end anything. The sheet is meant to stay on
    the band, and leaving the episode running is what makes the placement shaping keep paying for
    it -- so a drape that slides off the arm afterwards is worth less than one that settles.

    The flag is set by :class:`~.rewards.release_stage_reward`, which also charges the penalty on
    the step the gripper opens. This fires one step later: terminations are computed *before*
    rewards inside a step, so the flag is read on the following one. A single extra frame of the
    sheet falling is immaterial, but the lag is real.
    """
    return _high_release(env)


def finger_in_slot(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    margin: float = 0.005,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    slot_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("slot_neg_y"),
        SceneEntityCfg("slot_pos_y"),
    ),
    finger_body_names: tuple[str, ...] = ("panda_leftfinger", "panda_rightfinger"),
) -> torch.Tensor:
    """End the episode if either fingertip enters the box enclosing the slot.

    The sheet has to be taken by the strip projecting above the walls, so a finger down inside the
    slot means the gripper is digging at the walls rather than picking the sheet off the top.

    Tested geometrically against the slot's own frame rather than through a contact sensor. The
    walls are kinematic and their pose is known exactly, so the box test is both cheaper and
    steadier than reading contacts through the rigid-soft coupler -- and it fires slightly *before*
    a real touch, which is the useful side to err on.

    Args:
        env: The environment.
        half_extents: Half the size of the box enclosing both walls, in the slot frame [m].
        margin: Extra clearance added to every side of that box [m].
        robot_cfg: The robot carrying the fingers.
        slot_cfgs: The two walls; their midpoint is taken as the slot's centre.
        finger_body_names: Bodies treated as fingertips.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    walls: list[RigidObject] = [env.scene[cfg.name] for cfg in slot_cfgs]

    centre = torch.stack([wall.data.root_pos_w.torch for wall in walls]).mean(dim=0)
    # both walls share an orientation, so either one gives the slot's frame
    quat = walls[0].data.root_quat_w.torch

    limits = torch.tensor(half_extents, device=centre.device) + margin
    inside = torch.zeros(env.num_envs, dtype=torch.bool, device=centre.device)
    for name in finger_body_names:
        body_id = robot.body_names.index(name)
        offset = robot.data.body_link_pose_w.torch[:, body_id, :3] - centre
        local = quat_apply_inverse(quat, offset)
        inside |= (local.abs() < limits).all(dim=-1)
    return inside
