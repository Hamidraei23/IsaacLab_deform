# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal command that targets the mannequin arm instead of a free point in the workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from isaaclab_tasks.core.lift.mdp.commands.pose_commands import DeformableUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import DeformableUniformPoseCommandCfg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


class ArmDrapePoseCommand(DeformableUniformPoseCommand):
    """Goal position sampled on the mannequin arm rather than uniformly in the workspace.

    Identical to :class:`DeformableUniformPoseCommand` except for where the target comes from: the
    ranges are read as an offset in the *arm's* frame, so the goal rides with the arm wherever the
    reset event puts it. Because the offset is applied in the arm frame, a yaw-randomised arm
    carries its goal around with it.

    The command is still published in the robot root frame, so every downstream consumer -- the
    tracking reward, the success bonus, the goal marker, and the 7-dim ``target_position``
    observation -- is unchanged.

    Note:
        Resampling reads the arm's live pose, and the event manager repositions the arm before the
        command manager resamples on reset, so the goal is never a step behind. The arm does not
        move during an episode, so the periodic resample simply re-draws the offset jitter.
    """

    cfg: ArmDrapePoseCommandCfg

    def __init__(self, cfg: ArmDrapePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.arm: RigidObject = env.scene[cfg.arm_name]
        # where along the arm's axis the target band sits, in the arm's frame
        self.band_offset_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.band_visualizer = VisualizationMarkers(self.cfg.band_visualizer_cfg)
        self.band_visualizer.set_visibility(True)
        # The marker prototype's own "axis" attribute is not honoured by every visualizer backend,
        # which leaves the sleeve standing upright through the arm instead of wrapping it. Rotate
        # the marker's default +Z onto the arm's +X here instead, so the alignment does not depend
        # on that attribute being respected.
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._band_align_quat = quat_from_euler_xyz(zeros, zeros + torch.pi / 2, zeros)

    def _resample_command(self, env_ids: Sequence[int]):
        arm_pos_w = self.arm.data.root_pos_w.torch[env_ids]
        arm_quat_w = self.arm.data.root_quat_w.torch[env_ids]

        # ranges are an offset in the arm frame: x runs along the arm, z lifts off its surface
        local = torch.zeros(len(env_ids), 3, device=self.device)
        local[:, 0].uniform_(*self.cfg.ranges.pos_x)
        local[:, 1].uniform_(*self.cfg.ranges.pos_y)
        local[:, 2].uniform_(*self.cfg.ranges.pos_z)
        goal_pos_w = arm_pos_w + quat_apply(arm_quat_w, local)

        # the band is the same point projected onto the arm's axis, so the goal sits directly
        # above the middle of the red region
        self.band_offset_b[env_ids] = 0.0
        self.band_offset_b[env_ids, 0] = local[:, 0]

        # express in the robot root frame with the inverse of the transform the base class'
        # _update_metrics applies, so the two cancel and no quaternion convention is assumed here
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w.torch[env_ids],
            self.robot.data.root_quat_w.torch[env_ids],
            goal_pos_w,
            arm_quat_w,
        )
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b

    def _update_metrics(self):
        super()._update_metrics()
        # redraw the band where the arm currently is. Driven from here rather than from the debug
        # visualiser callback so the region is shown whether or not debug_vis is enabled.
        arm_pos_w = self.arm.data.root_pos_w.torch
        arm_quat_w = self.arm.data.root_quat_w.torch
        band_pos_w = arm_pos_w + quat_apply(arm_quat_w, self.band_offset_b)
        band_quat_w = quat_mul(arm_quat_w, self._band_align_quat)
        self.band_visualizer.visualize(translations=band_pos_w, orientations=band_quat_w)


@configclass
class ArmDrapePoseCommandCfg(DeformableUniformPoseCommandCfg):
    """Configuration for :class:`ArmDrapePoseCommand`.

    The inherited :attr:`ranges` are reinterpreted as an offset in the arm's frame rather than an
    absolute region of the robot's workspace; the roll/pitch/yaw entries are unused.
    """

    class_type: type[ArmDrapePoseCommand] = ArmDrapePoseCommand

    arm_name: str = "mannequin_arm"
    """Scene entity of the arm the goal is placed on."""

    band_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_band",
        markers={
            "band": sim_utils.CylinderCfg(
                radius=0.042,
                height=0.15,
                axis="X",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            )
        },
    )
    """Red region of interest painted around the arm -- the patch the sheet has to cover.

    Drawn as a per-instance overlay hugging the arm's surface rather than baked into the arm's own
    material, because the scene clones env_0's prims: a material or vertex colour would put the
    band at the same place in every environment, and per-environment prims would cost one arm per
    environment instead of one instanced arm.

    Purely visual -- no rigid or collision properties -- so the sheet drapes over the arm itself
    and the band never perturbs the physics. Keep the radius a millimetre or so above the arm's so
    it reads as paint; :attr:`height` is the region's extent along the arm.
    """
