"""Terms of the rigid-plate version of the sheet platform twist task.

The cloth is replaced by a solid plate of the same 20 x 20 cm footprint, ``PLATE_THICKNESS`` thick,
standing in the same slot at the same fixed pose. Every term the cloth task gets from sheet nodes has a
plate counterpart here, written so the rest of the task -- and the expert and DAgger script -- is
unchanged:

- ``plate_key_points``: the nine landmarks at the same indices as the cloth's ``sheet_key_points``
  (bottom row 0-2, middle 3-5, top 6-8, columns along the slot), in the robot base frame;
- ``rigid_grasp_detector``: the hold and extraction flags the cloth task's ``grasp_stage`` publishes
  (``_is_holding`` / ``_phase_reached``), with the same one-off grasp and extraction payouts;
- ``rigid_coverage``: the share of a 16 x 16 grid on the platform top covered by the plate lying flat on
  it, logged as ``Coverage/*`` like the cloth version;
- ``PlatformTopCommand``: the platform top-centre pose in the robot frame, the goal observation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from ..sheet_rl.mdp.platform import FIXED_SHEET_POSITION, FIXED_SHEET_YAW
from ..sheet_rl.mdp.platform_geometry import surface_samples
from ..sheet_rl.mdp.rewards import _is_holding, _phase_reached
from .sheet_twist_mdp import PLATFORM_DEPTH, fingers_open

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PLATE_SIZE = (0.2, 0.2)
PLATE_THICKNESS = 0.006
PLATE_SPAWN_ROT = (0.70710678, 0.0, 0.0, 0.70710678)
"""(x, y, z, w): stands the plate on edge, its local y vertical, as the cloth's spawn rotation does."""
GRASP_OFFSET = (0.0, 0.0, 0.1034)
"""panda_hand to the point between the fingertips, as the cloth task's detectors use [m]."""


def _plate_local_points(device) -> torch.Tensor:
    """The nine landmarks in the plate frame, ordered as the cloth's: row (bottom to top) major."""
    hx, hy = PLATE_SIZE[0] / 2, PLATE_SIZE[1] / 2
    return torch.tensor([[x, y, 0.0] for y in (-hy, 0.0, hy) for x in (-hx, 0.0, hx)], device=device)


def plate_points_w(env: ManagerBasedRLEnv, plate_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    plate = env.scene[plate_cfg.name]
    local = _plate_local_points(env.device)
    n = env.num_envs
    quat = plate.data.root_quat_w.torch[:, None].expand(-1, 9, -1).reshape(-1, 4)
    return plate.data.root_pos_w.torch[:, None] + quat_apply(quat, local.repeat(n, 1)).reshape(n, 9, 3)


def plate_key_points(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The plate's nine landmarks in the robot base frame [m], shape (num_envs, 27)."""
    robot = env.scene[robot_cfg.name]
    points = plate_points_w(env, plate_cfg).reshape(-1, 3)
    root_pos = robot.data.root_pos_w.torch[:, None].expand(-1, 9, -1).reshape(-1, 3)
    root_quat = robot.data.root_quat_w.torch[:, None].expand(-1, 9, -1).reshape(-1, 4)
    return subtract_frame_transforms(root_pos, root_quat, points)[0].view(env.num_envs, -1)


def reset_plate_and_platform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict,
    yaw_range: tuple[float, float],
    platform_size: tuple[float, float],
    slot_radius: float,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    slot_cfgs=(SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")),
    wall_offset_y: float = 0.015,
):
    """The cloth task's layout with the plate: plate and slot at the fixed pose, platform randomised.

    Mirrors ``reset_platform_and_sheet``: the slot walls flank the plate at ``wall_offset_y`` along its
    normal, and the platform is drawn on the opposite (+y) side, pushed out until it clears the slot.
    """
    device = env.device
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=device)
    yaw = torch.full((n,), FIXED_SHEET_YAW, device=device)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    centre = torch.tensor(FIXED_SHEET_POSITION, device=device).expand(n, -1)

    plate = env.scene[plate_cfg.name]
    spawn = torch.tensor(PLATE_SPAWN_ROT, device=device).expand(n, -1)
    plate.write_root_pose_to_sim_index(root_pose=torch.cat((origins + centre, quat_mul(yaw_quat, spawn)), -1), env_ids=env_ids)
    plate.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids)

    normal = torch.stack((-torch.sin(yaw), torch.cos(yaw), zeros), -1)
    for side, cfg in zip((-1.0, 1.0), slot_cfgs):
        wall = env.scene[cfg.name]
        pose = wall.data.default_root_pose.torch[env_ids].clone()
        pose[:, :2] = origins[:, :2] + centre[:, :2] + side * wall_offset_y * normal[:, :2]
        pose[:, 2] += origins[:, 2]
        pose[:, 3:] = yaw_quat
        wall.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        wall.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids)

    target = env.scene[platform_cfg.name]
    ranges = torch.tensor([position_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=device)
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 3), device=device)
    platform_yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
    pose = target.data.default_root_pose.torch[env_ids].clone()
    pose[:, 1] *= -1.0
    pose[:, :3] += offset * torch.tensor([1.0, 1.0, 0.0], device=device)
    delta = pose[:, :2] - centre[:, :2]
    distance = delta.norm(dim=-1, keepdim=True)
    minimum = math.hypot(*platform_size) / 2 + slot_radius + 0.02
    fallback = torch.zeros_like(delta)
    fallback[:, 1] = 1.0
    direction = torch.where(distance > 1e-6, delta / distance.clamp_min(1e-6), fallback)
    pose[:, :2] = centre[:, :2] + direction * distance.clamp_min(minimum)
    pose[:, :3] += origins
    pose[:, 3:] = quat_mul(quat_from_euler_xyz(zeros, zeros, platform_yaw), target.data.default_root_pose.torch[env_ids, 3:])
    target.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    target.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids)


class PlatformTopCommand(CommandTerm):
    """The platform's top-centre position and orientation in the robot base frame (7 values)."""

    cfg: PlatformTopCommandCfg

    def __init__(self, cfg: PlatformTopCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.platform = env.scene[cfg.platform_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 6] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _resample_command(self, env_ids):
        quat = self.platform.data.root_quat_w.torch[env_ids]
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, 2] = self.cfg.top_offset
        pos = self.platform.data.root_pos_w.torch[env_ids] + quat_apply(quat, offset)
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[env_ids], self.robot.data.root_quat_w.torch[env_ids], pos, quat
        )
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class PlatformTopCommandCfg(CommandTermCfg):
    class_type: type = PlatformTopCommand
    asset_name: str = "robot"
    platform_name: str = "mannequin_arm"
    top_offset: float = PLATFORM_DEPTH / 2


def _tip_w(env, robot_cfg: SceneEntityCfg, hand_body_name: str = "panda_hand"):
    robot = env.scene[robot_cfg.name]
    hand = robot.data.body_link_pose_w.torch[:, robot.body_names.index(hand_body_name)]
    offset = torch.tensor(GRASP_OFFSET, device=env.device).expand(env.num_envs, 3)
    return hand[:, :3] + quat_apply(hand[:, 3:7], offset)


class rigid_grasp_detector(ManagerTermBase):
    """Publish the hold and extraction flags for the plate; pay the cloth task's one-off bonuses.

    Holding: the fingers are shut on something (width between ``min_width`` and the plate thickness plus
    ``width_slack``) with the point between the fingertips inside the plate's top band (within
    ``band_depth`` of the top edge, across its width, and within ``normal_slack`` of its mid-plane).
    Extraction latches the first time the plate's lowest landmark clears ``clear_height`` while held.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        _is_holding(self._env)[ids] = False
        _phase_reached(self._env)[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        clear_height: float,
        first_grasp_bonus: float = 10.0,
        extraction_bonus: float = 200.0,
        min_width: float = 0.001,
        width_slack: float = 0.004,
        band_depth: float = 0.05,
        normal_slack: float = 0.006,
        plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
    ) -> torch.Tensor:
        plate = env.scene[plate_cfg.name]
        local = quat_apply_inverse(plate.data.root_quat_w.torch, _tip_w(env, robot_cfg) - plate.data.root_pos_w.torch)
        hx, hy = PLATE_SIZE[0] / 2, PLATE_SIZE[1] / 2
        width = env.scene[finger_cfg.name].data.joint_pos.torch[:, finger_cfg.joint_ids].sum(-1)
        holding = (
            (width > min_width) & (width < PLATE_THICKNESS + width_slack)
            & (local[:, 0].abs() < hx) & (local[:, 1] > hy - band_depth) & (local[:, 1] < hy + 0.01)
            & (local[:, 2].abs() < PLATE_THICKNESS / 2 + normal_slack)
        )
        lowest = plate_points_w(env, plate_cfg)[..., 2].amin(-1) - env.scene.env_origins[:, 2]
        phase = _phase_reached(env)
        extracted_now = holding & ~phase & (lowest > clear_height)
        first = holding & ~self.grasped
        self.grasped |= holding
        _is_holding(env).copy_(holding)
        phase |= extracted_now
        return (first_grasp_bonus * first.float() + extraction_bonus * extracted_now.float()) / env.step_dt


class rigid_coverage(ManagerTermBase):
    """Share of the platform top covered by the plate lying flat on it, per step; ``Coverage/*`` at reset.

    A grid sample counts when it falls inside the plate's footprint while the plate is flat (its normal
    within ``max_tilt`` of vertical) and resting on the top (its centre ``PLATE_THICKNESS / 2`` above it,
    within ``height_tolerance``). Same logs as the cloth ``platform_coverage``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.samples = surface_samples(cfg.params["platform_size"], (16, 16), env.device)
        n = env.num_envs
        self.coverage = torch.zeros(n, device=env.device)
        self.best = torch.zeros(n, device=env.device)
        self.best_released = torch.zeros(n, device=env.device)
        self.released = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        log["Coverage/best"] = self.best[ids].mean()
        log["Coverage/best_released"] = self.best_released[ids].mean()
        log["Coverage/final"] = self.coverage[ids].mean()
        log["Coverage/released"] = self.released[ids].float().mean()
        for buffer in (self.coverage, self.best, self.best_released, self.released):
            buffer[ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        platform_size,
        top_offset: float,
        height_tolerance: float = 0.006,
        max_tilt: float = 0.15,
        plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
        finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
    ) -> torch.Tensor:
        plate, target = env.scene[plate_cfg.name], env.scene[target_cfg.name]
        n, k = env.num_envs, len(self.samples)
        top = torch.zeros(n, k, 3, device=env.device)
        top[..., :2] = self.samples[None, :, :2]
        top[..., 2] = top_offset
        t_quat = target.data.root_quat_w.torch[:, None].expand(-1, k, -1).reshape(-1, 4)
        samples_w = target.data.root_pos_w.torch[:, None] + quat_apply(t_quat, top.reshape(-1, 3)).reshape(n, k, 3)
        p_quat = plate.data.root_quat_w.torch[:, None].expand(-1, k, -1).reshape(-1, 4)
        in_plate = quat_apply_inverse(p_quat, (samples_w - plate.data.root_pos_w.torch[:, None]).reshape(-1, 3)).reshape(n, k, 3)
        inside = (in_plate[..., 0].abs() <= PLATE_SIZE[0] / 2) & (in_plate[..., 1].abs() <= PLATE_SIZE[1] / 2)
        normal = matrix_from_quat(plate.data.root_quat_w.torch)[:, :, 2]
        flat = normal[:, 2].abs() > math.cos(max_tilt)
        top_w_z = target.data.root_pos_w.torch[:, 2] + top_offset
        resting = (plate.data.root_pos_w.torch[:, 2] - top_w_z - PLATE_THICKNESS / 2).abs() < height_tolerance
        self.coverage = inside.float().mean(-1) * (flat & resting).float()
        free = _phase_reached(env) & ~_is_holding(env) & fingers_open(env, finger_cfg)
        self.released |= free
        self.best = torch.maximum(self.best, self.coverage)
        self.best_released = torch.where(free, torch.maximum(self.best_released, self.coverage), self.best_released)
        return self.coverage


def plate_out_of_bounds(
    env: ManagerBasedRLEnv, min_height: float = -0.05, plate_cfg: SceneEntityCfg = SceneEntityCfg("plate")
) -> torch.Tensor:
    """End the episode when the plate falls off the table."""
    return env.scene[plate_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] < min_height
