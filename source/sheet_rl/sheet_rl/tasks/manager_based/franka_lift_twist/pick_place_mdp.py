"""Pick from one platform, then place on a second: a goal that switches when the first phase succeeds.

Phase 1 is the lift task: carry the cube to the commanded goal in the air. When the cube comes within
``pick_threshold`` of that goal while lifted, the episode's goal command switches, for good, to the
cube's resting pose on the centre of a second platform. Phase 2 is to set the cube down there and let
go. The goal is the same observation term as before, so the policy sees the switch as its target
dropping onto the second platform, and the observation layout (35) is unchanged.

The reward is staged so that finishing phase 1 never costs anything. Once the goal has switched, the
phase-1 terms (lift and both goal-tracking terms) pay their maximum. The phase-2 terms are paid on top
of that: tracking of the placement pose, a bonus while the cube rests there, and another while it rests
there with the gripper open.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.lift.mdp.rewards import object_goal_distance

from .platform_mdp import PLATFORM_DEPTH, _hand_speed, _lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .pick_place_command import PickPlacePoseCommand


@configclass
class PickPlacePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration of ``pick_place_command.PickPlacePoseCommand``."""

    # by name: the command module imports USD-backed markers, which must not load before the simulator
    class_type: type | str = f"{__name__.rpartition('.')[0]}.pick_place_command:PickPlacePoseCommand"

    object_name: str = "object"
    place_platform_name: str = "place_platform"
    gripper_action_name: str = "gripper_action"
    pick_threshold: float = 0.10
    """Phase 1 succeeds when the lifted cube is this close to the air goal [m] (the task's success radius)."""
    lift_height: float = 0.04
    """Height above its support at which the cube counts as lifted [m]."""
    place_tolerance: float = 0.025
    """Distance from the placement pose (cube centre resting on the platform centre) that counts as placed [m]."""
    settle_speed: float = 0.05
    """Cube speed below which it counts as resting [m/s]."""
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Never resample mid-episode: a resample would undo the switch to the placement goal."""


class place_target_platform(ManagerTermBase):
    """Reset event: put the second platform on the far side of the cube (in y), with a random height.

    Runs after ``place_cube_on_platform``. The second platform's centre is drawn at
    x ~ U(``x_range``) and at |y| ~ U(``y_range``) on the opposite side of the robot's x axis from the
    cube, so the two 10 cm platforms are at least ``y_range[0]`` apart and cannot overlap.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        height_range: tuple[float, float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ):
        n = len(env_ids)
        origins = env.scene.env_origins[env_ids]
        cube_y = env.scene["object"].data.root_pos_w.torch[env_ids, 1] - origins[:, 1]
        side = torch.where(cube_y >= 0.0, -1.0, 1.0)
        platform = env.scene["place_platform"]
        pose = platform.data.default_root_pose.torch[env_ids].clone()
        pose[:, 0] = origins[:, 0] + torch.empty(n, device=env.device).uniform_(*x_range)
        pose[:, 1] = origins[:, 1] + side * torch.empty(n, device=env.device).uniform_(*y_range)
        pose[:, 2] = origins[:, 2] + torch.empty(n, device=env.device).uniform_(*height_range) - PLATFORM_DEPTH / 2
        platform.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def _command(env: ManagerBasedRLEnv, command_name: str) -> PickPlacePoseCommand:
    return env.command_manager.get_term(command_name)


def object_lifted_or_picked(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Phase 1: lifted above its support. Phase 2: always 1, the phase-1 result kept."""
    return (_lifted(env, minimal_height, object_cfg) | (_command(env, command_name).phase == 1)).float()


class pick_goal_tracking(object_goal_distance):
    """Phase 1: ``object_goal_distance`` gated on height above the support. Phase 2: always 1.

    Its sticky success (``Metrics/success_rate``) is phase 1 reached, i.e. the pick succeeded.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        distance = torch.linalg.norm(env.scene[object_cfg.name].data.root_pos_w.torch - command.goal_w(), dim=-1)
        picked = command.phase == 1
        if success_threshold is not None:
            self._succeeded |= picked
        tracking = _lifted(env, minimal_height, object_cfg).float() * (1 - torch.tanh(distance / std))
        return torch.where(picked, 1.0, tracking)


def place_tracking(
    env: ManagerBasedRLEnv, std: float, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Phase 2: ``1 - tanh(d / std)`` of the cube to the placement pose. Not lift-gated: placing lowers it."""
    command = _command(env, command_name)
    distance = torch.linalg.norm(env.scene[object_cfg.name].data.root_pos_w.torch - command.goal_w(), dim=-1)
    return (command.phase == 1).float() * (1 - torch.tanh(distance / std))


def cube_placed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 while the cube rests at the placement pose."""
    return _command(env, command_name).placed().float()


def cube_released(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 while the cube rests at the placement pose with the gripper open."""
    return _command(env, command_name).released().float()


def ee_speed_approach_before_pick(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    hand_body_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The platform task's approach charge, phase 1 only: ``|v_hand|^2 (1 - tanh(d_hand_cube / std))``
    until the cube is lifted.

    Zero in phase 2. A phase-2 version weighted by nearness to the placement pose was tried: it made
    every move near the target dearer than the same move far from it, and the policy hovered 12-17 cm
    out, where the charge fades. Phase 2 is still charged by ``ee_speed`` and ``yaw_rate``.
    The phase test matters: a cube being lowered onto a platform lower than the first one is "not
    lifted" again, which would otherwise re-arm the charge during the placement.
    """
    command = _command(env, command_name)
    cube_w = env.scene[object_cfg.name].data.root_pos_w.torch
    tool_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    near_cube = 1 - torch.tanh(torch.linalg.norm(cube_w - tool_w, dim=1) / std)
    approaching = (~_lifted(env, minimal_height, object_cfg)) & (command.phase == 0)
    return approaching.float() * near_cube * _hand_speed(env, robot_cfg, hand_body_name).square()


def place_height_tracking(
    env: ManagerBasedRLEnv, std: float, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Phase 2: ``1 - tanh(|dz| / std)``, the cube's height error to the placement pose alone.

    The 3D tracking terms let the policy trade height for sideways distance; this one pays only for
    coming down to the platform, which is the part a policy trained on goals 25-50 cm up never did.
    """
    command = _command(env, command_name)
    dz = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - command.goal_w()[:, 2]
    return (command.phase == 1).float() * (1 - torch.tanh(dz.abs() / std))


def release_shaping(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gripper_action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Phase 2: ``(1 - tanh(d / std)) * (clip(a_gripper, -6, 0) + 6) / 6``, a ramp toward opening near the target.

    The binary gripper opens only when its raw action reaches 0, and after phase 1 the policy holds it
    near -2.7, about five spreads away, so ``released`` and the completion bonus are never sampled.
    This pays for every step of raising the raw action, as long as the cube is close to the placement
    pose (``d`` is cube to placement pose), so the policy can climb to the opening instead of having
    to land on it by chance. Far from the target it pays nothing, so it does not teach dropping.

    The ramp is linear on purpose. ``(1 + tanh(a)) / 2`` was tried first: at a = -2.7 its slope is 0.009,
    so it paid nothing to climb exactly where the gripper sits. It reaches down to -6 because a ramp
    clipped at -3 was escaped: under a wider gripper spread the policy pushed the action below -3 to
    stop chance openings dropping the cube in transit, and there the ramp is flat.
    """
    command = _command(env, command_name)
    distance = torch.linalg.norm(env.scene[object_cfg.name].data.root_pos_w.torch - command.goal_w(), dim=1)
    gripper = env.action_manager.get_term(gripper_action_name).raw_actions[:, 0]
    near = 1 - torch.tanh(distance / std)
    return (command.phase == 1).float() * near * (gripper.clamp(-6.0, 0.0) + 6.0) / 6.0


def hand_yaw_rate_after_pick(
    env: ManagerBasedRLEnv,
    command_name: str,
    hand_body_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phase 2: the hand's yaw rate squared, ``w_z^2`` about the world's vertical [rad^2/s^2].

    The yaw rate, not the yaw angle: the cube may be carried at any yaw (it is square), but nothing in
    phase 2 needs the wrist to turn. Before the switch it is zero.
    """
    robot = env.scene[robot_cfg.name]
    yaw_rate = robot.data.body_link_vel_w.torch[:, robot.body_names.index(hand_body_name), 5]
    return (_command(env, command_name).phase == 1).float() * yaw_rate.square()


class place_completion_bonus(ManagerTermBase):
    """A one-off bonus, paid on the first step of an episode that the cube is placed and released.

    Divided by ``step_dt`` so the term's weight is the literal amount added to the return. Paid once
    per episode: staying placed afterwards is paid by ``placed`` and ``released``, so the episode does
    not end on success, and finishing never forfeits the reward still to come.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        released = _command(env, command_name).released()
        first = released & ~self._paid
        self._paid |= released
        return first.float() / env.step_dt
