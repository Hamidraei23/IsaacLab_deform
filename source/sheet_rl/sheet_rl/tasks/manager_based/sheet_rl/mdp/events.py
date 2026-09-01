# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and start-up events for the sheet pick-and-place environment."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


ARM_RADIUS_ATTR = "sheet_rl_arm_radius"
"""Attribute the per-environment arm radius is parked on, on the env object.

Not routed through term parameters because it is a ``(num_envs,)`` tensor and the config is
serialised through Hydra, which handles floats and not tensors. Consumers read it with
:func:`resolve_arm_radius`, which falls back to the scalar they were configured with, so every term
keeps working unchanged when the randomisation is not installed.
"""


def resolve_arm_radius(env: ManagerBasedEnv, fallback: float) -> torch.Tensor | float:
    """The arm's radius for each environment, or ``fallback`` if it was never randomised.

    Args:
        env: The environment.
        fallback: The nominal radius the caller was configured with [m].

    Returns:
        A ``(num_envs,)`` tensor when :func:`randomize_arm_radius` has run, otherwise ``fallback``
        unchanged. Callers must broadcast it themselves -- the trailing shape differs per term.
    """
    return getattr(env, ARM_RADIUS_ATTR, fallback)


def randomize_arm_radius(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    nominal_radius: float,
    jitter: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    verbose: bool = True,
) -> None:
    """Give each environment its own arm thickness, written straight into the Newton model.

    Domain randomisation over the dimension of the arm the sculpted evaluation asset disagrees with
    most: the capsule is a constant 0.04 m while the real forearm runs 0.026 m at the wrist to
    0.059 m at the elbow. A policy trained against a single radius has never had to generalise over
    limb thickness, which is exactly what the evaluation environment asks of it.

    **Why this edits Newton and not USD.** The obvious route -- Isaac Lab's ``prestartup`` mode plus
    ``replicate_physics=False`` -- cannot work in this project. ``cloner.replicate`` only registers
    its USD replication context when ``has_kit()`` is true, and this runs Newton kitless, so with
    replication off *nothing is cloned into USD before play*: only ``env_0`` exists when
    ``prestartup`` fires and Newton builds the rest from it afterwards. Authoring a radius there
    reaches one environment and is then overwritten for every other by the clone.

    So the write happens after the model is finalised instead. Newton stores geometry in a flat
    per-shape array, one entry per environment, and for a capsule ``shape_scale`` is
    ``(radius, half_height, _)``. Setting component 0 resizes the capsule and leaves its length
    alone. Registered in ``startup`` mode, which runs once after the simulation starts and before
    the first reset, so the reset below can rely on the radii already being drawn.

    **One documented gap.** The arm is in both solvers: the cloth reaches it through the soft
    solver, which reads these arrays directly and so sees the new radius, and MuJoCo-Warp holds the
    rigid side in its own compiled model, which does not. Cloth drape -- the thing being scored --
    is therefore randomised; a *physical* finger-vs-arm rigid contact would still use the nominal
    0.04. That mismatch is mostly cosmetic here because the arm is kinematic and
    ``finger_touches_arm`` scores contact analytically from the radius set here, not from contacts.

    Args:
        env: The environment.
        env_ids: Unused -- start-up terms run once over every environment.
        nominal_radius: The unrandomised radius the config is written around [m].
        jitter: Fractional half-range, so ``0.2`` draws uniformly over +/-20%.
        asset_cfg: The arm whose capsule is resized.
        verbose: Print the resulting spread once, so a run can be seen to have taken effect.
    """
    del env_ids  # start-up terms are called once for all environments

    import warp as wp
    from newton import GeoType

    from isaaclab_newton.physics.newton_manager import NewtonManager

    num_envs = env.scene.num_envs
    model = NewtonManager.get_model()

    # Shapes are matched through their *body*, not by their own label: ``body_label`` is documented
    # to hold prim paths, so the environment index can be read straight out of it, and every shape
    # points at its body through ``shape_body``.
    leaf = env.scene[asset_cfg.name].cfg.prim_path.removeprefix(env.scene.env_regex_ns).lstrip("/")
    pattern = re.compile(rf"/env_(\d+)/{re.escape(leaf)}(?:/|$)")

    env_by_body: dict[int, int] = {}
    for body_index, label in enumerate(model.body_label or []):
        match = pattern.search(str(label))
        if match is not None:
            env_by_body[body_index] = int(match.group(1))

    shape_body = wp.to_torch(model.shape_body).cpu()
    shape_type = wp.to_torch(model.shape_type).cpu()
    shape_scale = wp.to_torch(model.shape_scale)  # shares memory with the model

    shape_by_env: dict[int, int] = {}
    for shape_index in range(len(shape_body)):
        if int(shape_type[shape_index]) != int(GeoType.CAPSULE):
            continue
        env_index = env_by_body.get(int(shape_body[shape_index]))
        if env_index is not None:
            shape_by_env[env_index] = shape_index

    if len(shape_by_env) != num_envs:
        sample = [str(label) for label in list(model.body_label or [])[:8]]
        raise RuntimeError(
            f"found an arm capsule in {len(shape_by_env)} of {num_envs} environments while looking"
            f" for '{pattern.pattern}' among {len(model.body_label or [])} Newton bodies."
            f" First body labels: {sample}"
        )

    low, high = nominal_radius * (1.0 - jitter), nominal_radius * (1.0 + jitter)
    radii = sample_uniform(low, high, (num_envs,), device="cpu")

    for env_index, shape_index in shape_by_env.items():
        shape_scale[shape_index, 0] = float(radii[env_index])

    setattr(env, ARM_RADIUS_ATTR, radii.to(env.device))

    if verbose:
        print(
            f"[sheet_rl] arm radius randomised over {num_envs} envs:"
            f" {radii.min():.4f} .. {radii.max():.4f} m (nominal {nominal_radius:.4f}, +/-{jitter:.0%})"
        )


def reset_arm_and_sheet(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    arm_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    sheet_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    # Sequence rather than tuple: the config passes a list, because Hydra's slice-to-string pass
    # does not recurse into tuples and would leave a SceneEntityCfg's default slice in place.
    slot_cfgs: Sequence[SceneEntityCfg] = (
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

    # A capsule rests on the table when its centre sits exactly one radius up, so a thickness-
    # randomised arm has to be re-seated: the configured spawn height is the *nominal* radius, and
    # left alone a fattened capsule would be buried in the table and a thinned one would hover.
    # A no-op unless :func:`randomize_arm_radius` ran, which is what keeps this correct for the
    # sculpted asset too -- its height is a bounding-box measurement, not a radius.
    arm_radius = getattr(env, ARM_RADIUS_ATTR, None)
    if arm_radius is not None:
        arm_pos[:, 2] = arm_radius[env_ids].to(device)

    arm_pos += origins
    zeros = torch.zeros(num_envs, device=device)

    # The yaw is composed *onto* the configured spawn orientation rather than replacing it. Writing
    # a yaw-only quaternion here would silently drop whatever ``init_state.rot`` asks for, so a roll
    # baked into the config would survive the spawn and then vanish on the first reset.
    #
    # Order matters: the yaw is a world-frame turn about z and the default is a body-frame
    # orientation, so the yaw goes on the left and is applied *after* it. That keeps the body pose
    # equal to ``yaw * mount``, which is what the goal command inverts to recover its task frame.
    arm_quat = quat_mul(quat_from_euler_xyz(zeros, zeros, draw_yaw()), arm_pose[:, 3:])
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
