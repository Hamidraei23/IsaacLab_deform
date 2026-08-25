# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the sheet pick-and-place environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_arm_and_sheet(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    arm_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    sheet_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    slot_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("slot_neg_y"),
        SceneEntityCfg("slot_pos_y"),
    ),
) -> None:
    """Place the arm, the sheet and its slot, mirroring which side of the table each occupies.

    All of them are reset in one term because they share draws. The mirror sign is shared by the
    arm and the sheet: mirroring them independently would put them on the same side half the time.
    The sheet and its slot walls additionally share an offset and a yaw, so the sheet always stands
    squarely in the gap instead of being left leaning against one wall or clipping through it.

    The arm gets its own independent offset and yaw.

    Args:
        env: The environment.
        env_ids: Environments being reset.
        position_range: Offset bounds [m] keyed by ``x``, ``y``, ``z``.
        yaw_range: Yaw bounds [rad].
        arm_cfg: The mannequin arm.
        sheet_cfg: The deformable sheet.
        slot_cfgs: The walls forming the slot the sheet stands in.
    """
    arm: RigidObject = env.scene[arm_cfg.name]
    sheet: DeformableObject = env.scene[sheet_cfg.name]
    device = sheet.device
    num_envs = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    # one draw shared by both assets: +1 keeps the spawn layout, -1 swaps the two sides
    side = torch.where(
        torch.rand(num_envs, device=device) < 0.5,
        -torch.ones(num_envs, device=device),
        torch.ones(num_envs, device=device),
    )

    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=device)

    def draw_offset() -> torch.Tensor:
        return sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 3), device=device)

    def draw_yaw() -> torch.Tensor:
        return sample_uniform(yaw_range[0], yaw_range[1], (num_envs,), device=device)

    # -- arm: mirror the default root position in y, then offset and yaw it
    arm_pose = arm.data.default_root_pose.torch[env_ids].clone()
    arm_pos = arm_pose[:, :3]
    arm_pos[:, 1] *= side
    arm_pos += draw_offset() * torch.tensor([1.0, 1.0, 0.0], device=device)
    arm_pos += origins
    zeros = torch.zeros(num_envs, device=device)
    arm_quat = quat_from_euler_xyz(zeros, zeros, draw_yaw())
    arm.write_root_pose_to_sim_index(root_pose=torch.cat([arm_pos, arm_quat], dim=-1), env_ids=env_ids)
    arm.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((num_envs, 6), device=device), env_ids=env_ids
    )

    # -- sheet and slot: one offset and one yaw shared between them, so the sheet cannot end up
    # leaning on a wall or clipping through one
    sheet_offset = draw_offset() * torch.tensor([1.0, 1.0, 0.0], device=device)
    sheet_yaw = draw_yaw()
    sheet_quat = quat_from_euler_xyz(zeros, zeros, sheet_yaw)
    cos_yaw, sin_yaw = torch.cos(sheet_yaw), torch.sin(sheet_yaw)

    # the rotation is done on the node positions directly, so it assumes no quaternion convention
    nodal_state = sheet.data.default_nodal_state_w.torch[env_ids].clone()
    pos = nodal_state[..., :3]
    centroid = pos.mean(dim=1, keepdim=True)
    rel = pos - centroid

    # the slot's default centre in env-local coordinates, used both to measure each wall's offset
    # and -- once mirrored and displaced exactly as the sheet's centroid is below -- as the centre
    # the walls are rebuilt around
    slot_center_default = centroid.squeeze(1)[:, :2] - origins[:, :2]
    slot_center_new = (
        slot_center_default * torch.stack([torch.ones_like(side), side], dim=-1)
        + sheet_offset[:, :2]
    )

    for slot_cfg in slot_cfgs:
        wall: RigidObject = env.scene[slot_cfg.name]
        wall_pose = wall.data.default_root_pose.torch[env_ids].clone()
        wall_pos = wall_pose[:, :3]

        # Each wall sits a fixed distance to one side of the slot's centre. Unlike a single
        # pedestal centred on the sheet, that offset is not invariant under the reset: it has to be
        # mirrored and then rotated with the sheet, or a yawed slot would end up with its walls
        # still facing along y while the sheet inside it has turned.
        local = wall_pos[:, :2] - slot_center_default
        local = local * torch.stack([torch.ones_like(side), side], dim=-1)
        rotated = torch.stack(
            [
                local[:, 0] * cos_yaw - local[:, 1] * sin_yaw,
                local[:, 0] * sin_yaw + local[:, 1] * cos_yaw,
            ],
            dim=-1,
        )

        wall_pos[:, :2] = slot_center_new + rotated
        wall_pos += origins
        wall.write_root_pose_to_sim_index(
            root_pose=torch.cat([wall_pos, sheet_quat], dim=-1), env_ids=env_ids
        )
        wall.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((num_envs, 6), device=device), env_ids=env_ids
        )

    cos_n, sin_n = cos_yaw.unsqueeze(1), sin_yaw.unsqueeze(1)
    rel = torch.stack(
        [
            rel[..., 0] * cos_n - rel[..., 1] * sin_n,
            rel[..., 0] * sin_n + rel[..., 1] * cos_n,
            rel[..., 2],
        ],
        dim=-1,
    )

    # default_nodal_state_w already carries the environment origin, so mirror the local part only
    local_centroid = centroid - origins.unsqueeze(1)
    local_centroid[..., 1] *= side.unsqueeze(1)
    new_centroid = local_centroid + origins.unsqueeze(1) + sheet_offset.unsqueeze(1)

    nodal_state[..., :3] = new_centroid + rel
    nodal_state[..., 3:] = 0.0
    sheet.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)
