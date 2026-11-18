"""The goal command of the pick-and-place task (see ``pick_place_mdp``).

Kept apart from the configuration because ``UniformPoseCommand`` imports USD-backed visualisation
markers, and the configuration is parsed before the simulator starts; ``PickPlacePoseCommandCfg``
names this class by string so it is imported only when the environment is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from .platform_mdp import PLATFORM_DEPTH, _lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .pick_place_mdp import PickPlacePoseCommandCfg

CUBE_HALF_HEIGHT = 0.024
"""Half the side of the lift task's cube (DexCube scaled 0.8) [m]: its centre's height when resting."""


class PickPlacePoseCommand(UniformPoseCommand):
    """``UniformPoseCommand`` whose goal moves onto the second platform once phase 1 succeeds.

    ``phase`` is 0 while picking and 1 after the switch; it resets to 0 with the episode. The metrics
    ``pick_success`` and ``place_success`` are logged per episode as ``Metrics/object_pose/<name>``.
    ``place_success`` is sticky: the cube rested at the placement pose with the gripper open at least once.
    """

    cfg: PickPlacePoseCommandCfg

    def __init__(self, cfg: PickPlacePoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cube = env.scene[cfg.object_name]
        self.place_platform = env.scene[cfg.place_platform_name]
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["pick_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["place_success"] = torch.zeros(self.num_envs, device=self.device)

    def goal_w(self) -> torch.Tensor:
        """The current goal position in the world, computed fresh from the base-frame command."""
        goal_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, self.pose_command_b[:, :3]
        )
        return goal_w

    def placed(self) -> torch.Tensor:
        """Phase 2, with the cube resting at the placement pose."""
        error = torch.linalg.norm(self.cube.data.root_pos_w.torch - self.goal_w(), dim=-1)
        speed = torch.linalg.norm(self.cube.data.root_lin_vel_w.torch, dim=-1)
        return (self.phase == 1) & (error < self.cfg.place_tolerance) & (speed < self.cfg.settle_speed)

    def released(self) -> torch.Tensor:
        """Placed, with the gripper commanded open (the binary gripper closes on negative actions)."""
        gripper = self._env.action_manager.get_term(self.cfg.gripper_action_name)
        return self.placed() & (gripper.raw_actions[:, 0] >= 0.0)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.phase[env_ids] = 0

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["pick_success"] = (self.phase == 1).float()
        self.metrics["place_success"] = torch.maximum(self.metrics["place_success"], self.released().float())

    def _update_command(self):
        picking = self.phase == 0
        distance = torch.linalg.norm(self.cube.data.root_pos_w.torch - self.goal_w(), dim=-1)
        lifted = _lifted(self._env, self.cfg.lift_height, SceneEntityCfg(self.cfg.object_name))
        ids = (picking & lifted & (distance < self.cfg.pick_threshold)).nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        self.phase[ids] = 1
        top_w = self.place_platform.data.root_pos_w.torch[ids].clone()
        top_w[:, 2] += PLATFORM_DEPTH / 2 + CUBE_HALF_HEIGHT
        place_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[ids], self.robot.data.root_quat_w.torch[ids], top_w
        )
        self.pose_command_b[ids, :3] = place_b
