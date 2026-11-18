"""Commands, reset layout and placement rewards for the rectangular platform."""

import math

from .gripper_geometry import finger_half_width

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import DeformableUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import DeformableUniformPoseCommandCfg

from .platform_geometry import sheet_triangles, surface_coverage, surface_samples
from .rewards import _drape_success, _high_release, _is_holding, _phase_reached


# Selected from 100 random poses on the end-effector's world-right (-Y) side. With checkpoint
# model_920.pt, 75/100 candidates extracted and this pose had the highest successful return.
FIXED_SHEET_POSITION = (0.40946364, -0.07716751, 0.102)
FIXED_SHEET_YAW = -0.53996944


class PlatformPoseCommand(DeformableUniformPoseCommand):
    """Publish platform top-centre and orientation in the existing seven-value command."""

    def _resample_command(self, env_ids):
        target = self._env.scene[self.cfg.platform_name]
        quat = target.data.root_quat_w.torch[env_ids]
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, 2] = self.cfg.top_offset
        pos = target.data.root_pos_w.torch[env_ids] + quat_apply(quat, offset)
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[env_ids], self.robot.data.root_quat_w.torch[env_ids], pos, quat
        )
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b

    def _update_metrics(self):
        super()._update_metrics()
        # Green means settled coverage, rather than mere COM proximity.
        if self.success_vis_asset is not None:
            self.success_visualizer.visualize(
                self._get_success_vis_pos_w(), marker_indices=_drape_success(self._env).int()
            )


@configclass
class PlatformPoseCommandCfg(DeformableUniformPoseCommandCfg):
    class_type: type = PlatformPoseCommand
    platform_name: str = "mannequin_arm"
    top_offset: float = 0.01


def reset_platform_and_sheet(
    env, env_ids, position_range, yaw_range, platform_size, slot_radius,
    fixed_sheet_position=FIXED_SHEET_POSITION, fixed_sheet_yaw=FIXED_SHEET_YAW,
    sheet_xy_jitter: float = 0.0, sheet_yaw_jitter: float = 0.0,
    arm_cfg=SceneEntityCfg("mannequin_arm"), sheet_cfg=SceneEntityCfg("deformable"),
    slot_cfgs=(SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")),
):
    """Place the sheet and its slot at the fixed pose, optionally jittered per environment, and
    independently randomize the platform.

    ``sheet_xy_jitter`` adds U(-j, j) [m] to the sheet's x and y, ``sheet_yaw_jitter`` adds
    U(-j, j) [rad] to its yaw, drawn per environment at every reset (user request: +-5 cm and
    +-90 deg). The slot walls follow the jittered pose exactly, and the platform keeps its clearance
    from the jittered sheet centre. Both default to 0, which is the fixed pose as before.
    """
    sheet = env.scene[sheet_cfg.name]
    target = env.scene[arm_cfg.name]
    device = sheet.device
    num_envs = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    zeros = torch.zeros(num_envs, device=device)
    fixed_position = torch.tensor(fixed_sheet_position, device=device).repeat(num_envs, 1)
    fixed_yaw = torch.full((num_envs,), float(fixed_sheet_yaw), device=device)
    if sheet_xy_jitter > 0.0:
        fixed_position[:, :2] += sample_uniform(-sheet_xy_jitter, sheet_xy_jitter, (num_envs, 2), device=device)
    if sheet_yaw_jitter > 0.0:
        fixed_yaw += sample_uniform(-sheet_yaw_jitter, sheet_yaw_jitter, (num_envs,), device=device)
    fixed_quat = quat_from_euler_xyz(zeros, zeros, fixed_yaw)

    nodal_state = sheet.data.default_nodal_state_w.torch[env_ids].clone()
    pos = nodal_state[..., :3]
    centroid = pos.mean(dim=1, keepdim=True)
    rel = pos - centroid
    cos_yaw, sin_yaw = torch.cos(fixed_yaw).unsqueeze(1), torch.sin(fixed_yaw).unsqueeze(1)
    rel = torch.stack(
        [
            rel[..., 0] * cos_yaw - rel[..., 1] * sin_yaw,
            rel[..., 0] * sin_yaw + rel[..., 1] * cos_yaw,
            rel[..., 2],
        ],
        dim=-1,
    )
    nodal_state[..., :3] = origins.unsqueeze(1) + fixed_position.unsqueeze(1) + rel
    nodal_state[..., 3:] = 0.0
    sheet.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)

    # The rigid slot follows the fixed sheet pose exactly.
    default_centre = centroid.squeeze(1)[:, :2] - origins[:, :2]
    for slot_cfg in slot_cfgs:
        wall = env.scene[slot_cfg.name]
        wall_pose = wall.data.default_root_pose.torch[env_ids].clone()
        wall_pos = wall_pose[:, :3]
        local = wall_pos[:, :2] - default_centre
        rotated = torch.stack(
            [
                local[:, 0] * cos_yaw[:, 0] - local[:, 1] * sin_yaw[:, 0],
                local[:, 0] * sin_yaw[:, 0] + local[:, 1] * cos_yaw[:, 0],
            ],
            dim=-1,
        )
        wall_pos[:, :2] = fixed_position[:, :2] + rotated
        wall_pos += origins
        wall.write_root_pose_to_sim_index(
            root_pose=torch.cat([wall_pos, fixed_quat], dim=-1), env_ids=env_ids
        )
        wall.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((num_envs, 6), device=device), env_ids=env_ids
        )

    # Randomize only the rectangle, on the opposite (+Y) side from the fixed right-side sheet.
    ranges = torch.tensor(
        [position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=device
    )
    offset = sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 3), device=device)
    platform_yaw = sample_uniform(yaw_range[0], yaw_range[1], (num_envs,), device=device)
    pose = target.data.default_root_pose.torch[env_ids].clone()
    pose[:, 1] *= -1.0
    pose[:, :3] += offset * torch.tensor([1.0, 1.0, 0.0], device=device)

    centre = fixed_position[:, :2]
    delta = pose[:, :2] - centre
    distance = delta.norm(dim=-1, keepdim=True)
    minimum = math.hypot(*platform_size) / 2 + slot_radius + 0.02
    fallback = torch.zeros_like(delta)
    fallback[:, 1] = 1.0
    direction = torch.where(distance > 1e-6, delta / distance.clamp_min(1e-6), fallback)
    pose[:, :2] = centre + direction * distance.clamp_min(minimum)
    pose[:, :3] += origins
    pose[:, 3:] = quat_mul(
        quat_from_euler_xyz(zeros, zeros, platform_yaw),
        target.data.default_root_pose.torch[env_ids, 3:],
    )
    target.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    target.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((num_envs, 6), device=device), env_ids=env_ids
    )


class platform_opened_high(ManagerTermBase):
    """End phase two on a new applied open command above the table-relative ceiling.

    Runs before rewards, so an invalid opening cannot earn placement or success credit.
    Detect an opening edge rather than open fingers at height: a low release followed by
    retreat with the gripper remaining open is valid.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.was_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.was_physically_open = torch.ones_like(self.was_open)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.was_open[ids] = True
        self.was_physically_open[ids] = True
        _high_release(self._env)[ids] = False

    def __call__(self, env, max_height=0.10, top_offset=0.01,
                 gripper_action_name="gripper_action", hand_body_name="panda_hand",
                 grasp_offset=(0.0, 0.0, 0.1034), robot_cfg=SceneEntityCfg("robot"),
                 target_cfg=SceneEntityCfg("mannequin_arm"),
                 finger_cfg=SceneEntityCfg("robot", joint_names="panda_finger_joint.*")):
        gripper = env.action_manager.get_term(gripper_action_name)
        state = gripper.raw_actions[:, 0]
        opened = state > gripper.cfg.threshold if gripper.cfg.positive_threshold else state < gripper.cfg.threshold
        robot, target = env.scene[robot_cfg.name], env.scene[target_cfg.name]
        hand = robot.data.body_link_pose_w.torch[:, robot.body_names.index(hand_body_name)]
        offset = torch.tensor(grasp_offset, device=hand.device).expand(len(hand), 3)
        tip = hand[:, :3] + quat_apply(hand[:, 3:7], offset)
        local = quat_apply_inverse(target.data.root_quat_w.torch, tip - target.data.root_pos_w.torch)
        physically_open = finger_half_width(env, finger_cfg) > 0.030
        # Also reject a delayed physical opening above the ceiling after an earlier low command.
        opening = (opened & ~self.was_open) | (physically_open & ~self.was_physically_open)
        invalid = _phase_reached(env) & opening & (local[:, 2] + top_offset > max_height)
        _high_release(env).logical_or_(invalid)
        self.was_open.copy_(opened)
        self.was_physically_open.copy_(physically_open)
        return _high_release(env)


class platform_placement(ManagerTermBase):
    """Bounded placement shaping after extraction, with an off-target distance cost.

    Approach, lowering, supported coverage and opening pay only for new episode bests.
    Actual release pays once, smoothly graded by horizontal alignment, release height
    and coverage. After release, phase three moves the fingertips above the platform
    centre and holds there. Success requires both a settled sheet and a quiet end effector.
    Held cloth earns no standing reward. High opening is checked by the termination term;
    aborted episodes get no positive placement credit. Motion is a separate reward term.
    All components are literal per-control-step amounts; the manager's dt is undone.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.triangles = sheet_triangles(cfg.params["resolution"], dev)
        self.samples = surface_samples(cfg.params["platform_size"], (16, 16), dev)
        for name in ("best_coverage", "best_approach", "best_lower", "best_opening",
                     "best_released_coverage", "center_start", "center_best", "release_height",
                     "release_alignment", "high_count", "coverage", "elevation",
                     "best_retreat", "retreat_distance", "retreat_speed"):
            setattr(self, name, torch.zeros(n, device=dev))
        for name in ("active", "was_holding", "released", "hold_ended", "retreat_reached"):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=dev))
        self.dwell = torch.zeros(n, dtype=torch.long, device=dev)
        self.hand = None
        self.components = {}
        self.component_totals = {}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        log["Placement/best_coverage"] = self.best_coverage[ids].mean()
        log["Placement/best_released_coverage"] = self.best_released_coverage[ids].mean()
        log["Placement/success"] = _drape_success(self._env)[ids].float().mean()
        log["Placement/released"] = self.released[ids].float().mean()
        log["Placement/high_releases"] = self.high_count[ids].mean()
        log["Retreat/entered"] = self.released[ids].float().mean()
        log["Retreat/reached_target"] = self.retreat_reached[ids].float().mean()
        released_count = self.released[ids].sum().clamp_min(1)
        log["Retreat/final_distance"] = (self.retreat_distance[ids] * self.released[ids]).sum() / released_count
        start = self.center_start[ids]
        log["Placement/center_progress"] = torch.where(
            start > 1e-3, 1 - self.center_best[ids] / start.clamp_min(1e-3), torch.zeros_like(start)).mean()
        for name, total in self.component_totals.items():
            log[f"PlacementReward/{name}"] = total[ids].mean()
            total[ids] = 0
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value.shape == (self._env.num_envs,):
                value[ids] = 0
        for value in self.components.values():
            value[ids] = 0
        _drape_success(self._env)[ids] = False
        _high_release(self._env)[ids] = False

    def _progress(self, name, value, eligible):
        best = getattr(self, name)
        delta = (value - best).clamp_min(0) * eligible
        best.add_(delta)
        return delta

    def __call__(
        self, env, resolution, platform_size, top_offset=0.01, particle_clearance=0.002,
        coverage_threshold=0.50, settle_time=1.0, speed_threshold=0.03,
        approach_bonus=500.0, approach_std=0.2, lower_bonus=500.0, alignment_std=0.10,
        coverage_bonus=1500.0, opening_bonus=100.0,
        release_bonus=1500.0, release_height=0.03, release_height_std=0.08,
        released_coverage_bonus=1000.0, released_coverage_step=5.0, success_bonus=5000.0,
        distance_limit=0.30, distance_penalty=5.0, distance_penalty_span=0.10,
        retreat_height=0.20, retreat_bonus=1000.0, retreat_std=0.10, retreat_tolerance=0.03,
        retreat_speed_threshold=0.03, retreat_angular_threshold=0.20,
        retreat_distance_limit=0.30, retreat_distance_penalty=5.0, retreat_motion_penalty=2.0,
        gripper_action_name="gripper_action",
        hand_body_name="panda_hand", grasp_offset=(0.0, 0.0, 0.1034),
        asset_cfg=SceneEntityCfg("deformable"), target_cfg=SceneEntityCfg("mannequin_arm"),
        finger_cfg=SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
    ):
        sheet, target = env.scene[asset_cfg.name], env.scene[target_cfg.name]
        nodes = sheet.data.nodal_pos_w.torch
        quat = target.data.root_quat_w.torch[:, None, :].expand(-1, nodes.shape[1], -1)
        local = quat_apply_inverse(quat.reshape(-1, 4),
                                   (nodes - target.data.root_pos_w.torch[:, None]).reshape(-1, 3)).reshape_as(nodes)
        height = top_offset + particle_clearance
        coverage = surface_coverage(local, self.triangles, self.samples, height)
        center = local.mean(1)
        goal = torch.zeros_like(center)
        goal[:, 2] = height
        distance = (center - goal).norm(dim=-1)
        horizontal_distance = center[:, :2].norm(dim=-1)
        alignment = torch.exp(-horizontal_distance / alignment_std)
        phase, holding = _phase_reached(env), _is_holding(env)
        # Terminations are evaluated first. Never pay for a throw on its terminal step.
        valid = phase & ~_high_release(env)
        entering = phase & ~self.active
        self.center_start.copy_(torch.where(entering, distance, self.center_start))
        self.center_best.copy_(torch.where(entering, distance, self.center_best))
        self.center_best.copy_(torch.where(phase, torch.minimum(self.center_best, distance), self.center_best))

        robot = env.scene[finger_cfg.name]
        width = finger_half_width(env, finger_cfg)
        opened = width > 0.030
        if self.hand is None:
            self.hand = robot.body_names.index(hand_body_name)
        hand = robot.data.body_link_pose_w.torch[:, self.hand]
        offset = torch.tensor(grasp_offset, device=hand.device).expand(len(hand), 3)
        tip = hand[:, :3] + quat_apply(hand[:, 3:7], offset)
        tip_local = quat_apply_inverse(target.data.root_quat_w.torch, tip - target.data.root_pos_w.torch)
        # Relative table height, so translated environments have identical rewards.
        elevation = tip_local[:, 2] + top_offset
        low = torch.exp(-(elevation - release_height).clamp_min(0) / release_height_std)
        readiness = alignment * low
        proximity = torch.exp(-distance / approach_std)
        # No free approach/lowering payout just for crossing the extraction boundary.
        self.best_approach.copy_(torch.where(entering, proximity, self.best_approach))
        self.best_lower.copy_(torch.where(entering, readiness, self.best_lower))
        approach = approach_bonus * self._progress("best_approach", proximity, valid & holding)
        lower = lower_bonus * self._progress("best_lower", readiness, valid & holding)
        cover = coverage_bonus * self._progress("best_coverage", coverage, valid)

        # Credit opening the applied filter toward its binary threshold; a finite budget
        # prevents hovering just below the threshold from collecting reward forever.
        gripper = env.action_manager.get_term(gripper_action_name)
        sign = 1.0 if gripper.cfg.positive_threshold else -1.0
        margin = sign * (gripper.raw_actions[:, 0] - gripper.cfg.threshold)
        opening = (margin + 1.0).clamp(0, 1)
        open_pay = opening_bonus * self._progress(
            "best_opening", readiness * opening, valid & holding & ~self.released)

        # The hold detector usually turns off before the fingers are fully open.
        # Keep the height/alignment at hold loss, never the later lowered hand pose.
        ended = phase & self.was_holding & ~holding
        self.release_height.copy_(torch.where(ended, elevation, self.release_height))
        self.release_alignment.copy_(torch.where(ended, alignment, self.release_alignment))
        self.hold_ended.logical_or_(ended)
        self.hold_ended.logical_and_(~holding)
        release = valid & self.hold_ended & opened
        first_release = release & ~self.released
        release_low = torch.exp(-(self.release_height - release_height).clamp_min(0) / release_height_std)
        release_pay = release_bonus * self.release_alignment * release_low * (0.25 + 0.75 * coverage) * first_release
        self.released.logical_or_(release)
        self.high_count.add_((release & (self.release_height > release_height)).float())
        self.hold_ended.logical_and_(~release)

        free = valid & self.released & ~holding & opened
        # Phase three: absolute platform-centred goal, not an offset from the release point.
        retreat_goal = torch.zeros_like(tip_local)
        retreat_goal[:, 2] = top_offset + retreat_height
        retreat_distance = (tip_local - retreat_goal).norm(dim=-1)
        retreat_proximity = torch.exp(-retreat_distance / retreat_std)
        self.best_retreat.copy_(torch.where(first_release, retreat_proximity, self.best_retreat))
        retreat_pay = retreat_bonus * self._progress("best_retreat", retreat_proximity, free)
        velocity = robot.data.body_link_vel_w.torch[:, self.hand]
        # Account for wrist rotation sweeping the fingertip offset through space.
        tip_velocity = velocity[:, :3] + torch.cross(velocity[:, 3:], tip - hand[:, :3], dim=-1)
        tip_speed = tip_velocity.norm(dim=-1)
        angular_speed = velocity[:, 3:].norm(dim=-1)
        at_retreat = free & (retreat_distance <= retreat_tolerance)
        quiet = (tip_speed <= retreat_speed_threshold) & (angular_speed <= retreat_angular_threshold)
        self.retreat_reached.logical_or_(at_retreat)
        self.retreat_distance.copy_(retreat_distance)
        self.retreat_speed.copy_(tip_speed)
        platform_top = torch.zeros_like(tip_local)
        platform_top[:, 2] = top_offset
        hand_distance = (tip_local - platform_top).norm(dim=-1)
        phase_three = valid & self.released
        retreat_wander = retreat_distance_penalty * ((hand_distance - retreat_distance_limit)
                                                     / distance_penalty_span).clamp(0, 1) * phase_three
        near_retreat = phase_three & (retreat_distance <= 2 * retreat_tolerance)
        retreat_motion = retreat_motion_penalty * (
            (tip_speed - retreat_speed_threshold).clamp_min(0) / 0.10
            + (angular_speed - retreat_angular_threshold).clamp_min(0)).clamp(max=1) * near_retreat
        free_progress = released_coverage_bonus * self._progress("best_released_coverage", coverage, free)
        free_coverage = released_coverage_step * coverage * free
        speed = sheet.data.nodal_vel_w.torch.norm(dim=-1).amax(dim=-1)
        settled = at_retreat & quiet & (coverage >= coverage_threshold) & (speed < speed_threshold)
        self.dwell.copy_(torch.where(settled, self.dwell + 1, 0))
        success = valid & (self.dwell >= max(1, math.ceil(settle_time / env.step_dt))) & ~_drape_success(env)
        _drape_success(env).logical_or_(success)
        self.components = {
            "approach": approach, "lower": lower, "coverage_progress": cover,
            "opening_progress": open_pay, "release": release_pay,
            "released_coverage_progress": free_progress, "released_coverage": free_coverage,
            "success": success_bonus * coverage * success,
            "off_target": -distance_penalty * ((horizontal_distance - distance_limit)
                                               / distance_penalty_span).clamp(0, 1) * phase,
            "retreat_progress": retreat_pay,
            "retreat_distance": -retreat_wander,
            "retreat_motion": -retreat_motion,
        }
        for name, value in self.components.items():
            if name not in self.component_totals:
                self.component_totals[name] = torch.zeros_like(value)
            self.component_totals[name].add_(value)
        self.coverage.copy_(coverage)
        self.elevation.copy_(elevation)
        self.active.copy_(phase)
        self.was_holding.copy_(holding)
        return sum(self.components.values()) / env.step_dt
