"""A platform of random height under the lift cube, and the reward terms that have to know about it.

The cube can stand on a square kinematic box whose top is re-drawn at every reset. A box cannot
change size at run time, so it is one ``PLATFORM_DEPTH``-tall box sunk into the table until its top is
at the drawn height. A kinematic body does not collide with the static table, so the buried part
does nothing. The task's original lift test is ``z > 0.04`` in absolute terms. On anything taller
than 1.6 cm that is true of a cube just standing on the platform, so the lift and goal terms here
measure height above the support instead: the platform top, or the table where there is none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
import isaaclab.sim as sim_utils

from isaaclab_tasks.contrib.lift.mdp.rewards import object_goal_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PLATFORM_DEPTH = 0.20
"""Height of the box geometry [m]; must exceed the tallest platform drawn."""


def platform_cfg(
    size: float,
    position_xy: tuple[float, float],
    name: str = "Platform",
    color: tuple[float, float, float] = (0.25, 0.45, 0.8),
) -> RigidObjectCfg:
    """A ``size`` x ``size`` kinematic box, ``PLATFORM_DEPTH`` tall, starting fully below the table."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=(size, size, PLATFORM_DEPTH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*position_xy, -PLATFORM_DEPTH / 2)),
    )


def support_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height of the platform top above the env origin [m] (zero where the scene has no platform)."""
    if "platform" not in env.scene.keys():
        return torch.zeros(env.num_envs, device=env.device)
    platform_z = env.scene["platform"].data.root_pos_w.torch[:, 2]
    return platform_z + PLATFORM_DEPTH / 2 - env.scene.env_origins[:, 2]


class place_cube_on_platform(ManagerTermBase):
    """Reset event: draw a platform height, move the platform under the cube, and stand the cube on it.

    It must run after the cube's own reset event, which leaves the cube at its table pose. That holds
    when it is the last event term defined.
    """

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, height_range: tuple[float, float]):
        low, high = height_range
        if not 0.0 <= low <= high < PLATFORM_DEPTH:
            raise ValueError(f"Platform heights {height_range} must lie in [0, {PLATFORM_DEPTH}).")
        height = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
        cube, platform = env.scene["object"], env.scene["platform"]
        cube_pose = torch.cat((cube.data.root_pos_w.torch[env_ids], cube.data.root_quat_w.torch[env_ids]), dim=-1)
        platform_pose = platform.data.default_root_pose.torch[env_ids].clone()
        platform_pose[:, :2] = cube_pose[:, :2]
        platform_pose[:, 2] = env.scene.env_origins[env_ids, 2] + height - PLATFORM_DEPTH / 2
        platform.write_root_pose_to_sim_index(root_pose=platform_pose, env_ids=env_ids)
        cube_pose[:, 2] += height
        cube.write_root_pose_to_sim_index(root_pose=cube_pose, env_ids=env_ids)


def platform_event(height_range: tuple[float, float]) -> EventTermCfg:
    return EventTermCfg(func=place_cube_on_platform, mode="reset", params={"height_range": height_range})


def _lifted(env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    cube_z = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return cube_z - support_height(env) > minimal_height


def object_is_lifted_above_support(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """``object_is_lifted`` measured from the support rather than from z = 0."""
    return _lifted(env, minimal_height, object_cfg).float()


class object_goal_distance_above_support(object_goal_distance):
    """``object_goal_distance`` (with its success tracking) gated on height above the support."""

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
        robot, cube = env.scene[robot_cfg.name], env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
        distance = torch.linalg.norm(goal_w - cube.data.root_pos_w.torch, dim=1)
        lifted = _lifted(env, minimal_height, object_cfg)
        if success_threshold is not None:
            self._succeeded |= lifted & (distance < success_threshold)
        return lifted.float() * (1 - torch.tanh(distance / std))


def _hand_speed(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, hand_body_name: str) -> torch.Tensor:
    # the hand link, not the tool point, so a wrist turn is not billed as travel
    robot = env.scene[robot_cfg.name]
    return robot.data.body_link_vel_w.torch[:, robot.body_names.index(hand_body_name), :3].norm(dim=-1)


def ee_speed_above(
    env: ManagerBasedRLEnv,
    free_speed: float,
    hand_body_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``max(0, |v_hand| - free_speed)^2`` [m^2/s^2]: a charge on hand speed past an allowance."""
    return (_hand_speed(env, robot_cfg, hand_body_name) - free_speed).clamp(min=0.0).square()


def ee_speed_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    hand_body_name: str = "panda_hand",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """``|v_hand|^2`` weighted by ``1 - tanh(d / std)`` (the reach kernel), until the cube is lifted.

    This charges the approach and the grasp: it grows as the fingers close in on the cube, and switches
    off once the cube is above the support, so carrying it to the goal is charged only by
    ``ee_speed_above``.
    """
    cube_w = env.scene[object_cfg.name].data.root_pos_w.torch
    tool_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    nearness = 1 - torch.tanh(torch.linalg.norm(cube_w - tool_w, dim=1) / std)
    approaching = ~_lifted(env, minimal_height, object_cfg)
    return approaching.float() * nearness * _hand_speed(env, robot_cfg, hand_body_name).square()
