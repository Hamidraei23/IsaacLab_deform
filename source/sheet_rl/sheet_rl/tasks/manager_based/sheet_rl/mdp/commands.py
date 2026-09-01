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
from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

from isaaclab_tasks.core.lift.mdp.commands.pose_commands import DeformableUniformPoseCommand
from isaaclab_tasks.core.lift.mdp.commands.pose_commands_cfg import DeformableUniformPoseCommandCfg

from .events import resolve_arm_radius

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
        # the goal point itself in the arm's frame: the band offset plus the lift off the surface.
        # Published for the same reason ``band_offset_b`` is -- reward terms run before the command
        # manager within a step, so ``pose_command_w`` is one step stale when they read it, and
        # simply wrong on the first step after a reset. Combining this with the arm's live pose is
        # always current, and gives the same point the success visualiser measures against.
        self.goal_offset_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.band_visualizer = VisualizationMarkers(self.cfg.band_visualizer_cfg)
        self.band_visualizer.set_visibility(True)
        # The marker prototype's own "axis" attribute is not honoured by every visualizer backend,
        # which leaves the sleeve standing upright through the arm instead of wrapping it. Rotate
        # the marker's default +Z onto the arm's +X here instead, so the alignment does not depend
        # on that attribute being respected.
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._band_align_quat = quat_from_euler_xyz(zeros, zeros + torch.pi / 2, zeros)

        # Optional correction for an arm whose centre line is not its root frame's x-axis. A capsule
        # is straight, so for the base task this stays ``None`` and the band rides the axis exactly
        # as before; a sculpted arm curves, and drawing the sleeve on the axis leaves it visibly
        # hanging off one side of the limb.
        # Fixed body-frame rotation between the arm's *task* frame and the pose its rigid body
        # actually carries. ``None`` -- the default -- means the two coincide and the body's own
        # orientation is the task frame, which is the case for every asset mounted as authored.
        self._arm_mount_conj: torch.Tensor | None = None
        if cfg.arm_mount_quat is not None:
            mount = torch.tensor(cfg.arm_mount_quat, dtype=torch.float32, device=self.device)
            self._arm_mount_conj = (
                quat_conjugate(mount.unsqueeze(0)).expand(self.num_envs, 4).contiguous()
            )

        self._centerline_x: torch.Tensor | None = None
        self._centerline_yz: torch.Tensor | None = None
        if cfg.band_centerline_offsets is not None:
            samples = torch.tensor(
                cfg.band_centerline_offsets, dtype=torch.float32, device=self.device
            )
            order = torch.argsort(samples[:, 0])
            self._centerline_x = samples[order, 0].contiguous()
            self._centerline_yz = samples[order, 1:].contiguous()

    def arm_task_quat_w(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor:
        """The arm's *task* frame in world coordinates, with any mounting rotation stripped off.

        Everything the policy sees about the arm comes through this frame, so it must depend only on
        where the arm has been placed and not on how the asset happens to be mounted in its body.
        The body pose is ``task * mount``: the reset composes its world yaw onto the configured spawn
        orientation, so right-multiplying by the mount's conjugate recovers the yaw alone.

        Without this, turning the asset over in its body would rewrite the goal command -- both the
        published orientation and the lift that puts the goal *above* the arm rather than under it --
        and a policy trained against the old frame reads a target it has never seen.

        Args:
            env_ids: Environments to return the frame for. Defaults to all of them.

        Returns:
            Orientation of the arm's task frame as ``(w, x, y, z)``. Shape ``(len(env_ids), 4)``.
        """
        body_quat_w = self.arm.data.root_quat_w.torch[env_ids]
        if self._arm_mount_conj is None:
            return body_quat_w
        return quat_mul(body_quat_w, self._arm_mount_conj[env_ids])

    def _centerline_offset(self, along: torch.Tensor) -> torch.Tensor:
        """Lateral offset of the arm's centre line at each band position, linearly interpolated.

        Args:
            along: Distance of each environment's band centre from the arm's origin, along the
                arm's own axis [m]. Shape ``(num_envs,)``.

        Returns:
            The ``(y, z)`` offset to add in the arm's frame. Shape ``(num_envs, 2)``.
        """
        table_x, table_yz = self._centerline_x, self._centerline_yz
        # clamp into the sampled span, so a band beyond the ends holds the last measured offset
        # rather than extrapolating a curve that was never measured
        index = torch.searchsorted(table_x, along.contiguous()).clamp(1, len(table_x) - 1)
        low, high = table_x[index - 1], table_x[index]
        weight = ((along - low) / (high - low)).clamp(0.0, 1.0).unsqueeze(-1)
        return torch.lerp(table_yz[index - 1], table_yz[index], weight)

    def _resample_command(self, env_ids: Sequence[int]):
        arm_pos_w = self.arm.data.root_pos_w.torch[env_ids]
        # the task frame, not the body's own orientation -- see :meth:`arm_task_quat_w`. For an
        # as-authored asset the two are the same tensor and nothing below changes.
        arm_quat_w = self.arm_task_quat_w(env_ids)

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
        self.goal_offset_b[env_ids] = local

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
        band_offset_b = self.band_offset_b
        if self._centerline_x is not None:
            # slide the sleeve sideways onto the limb's real centre line. Only the marker moves --
            # ``goal_offset_b``, and the capsule the reward terms build about the arm's root axis,
            # are deliberately left alone, so this changes what is drawn and nothing that is scored.
            band_offset_b = band_offset_b.clone()
            band_offset_b[:, 1:] = self._centerline_offset(band_offset_b[:, 0])
        band_pos_w = arm_pos_w + quat_apply(arm_quat_w, band_offset_b)
        band_quat_w = quat_mul(arm_quat_w, self._band_align_quat)

        # Track a randomised arm thickness, so the sleeve keeps hugging the surface instead of being
        # swallowed by a fattened arm or left hanging off a thinned one. Radial only: the marker's
        # local z was rotated onto the arm's axis above and carries the band's width, which is a
        # task constant and must not scale with the limb.
        scales = None
        if self.cfg.arm_nominal_radius is not None:
            radius = resolve_arm_radius(self._env, self.cfg.arm_nominal_radius)
            if isinstance(radius, torch.Tensor):
                ratio = (radius / self.cfg.arm_nominal_radius).unsqueeze(-1)
                scales = torch.cat([ratio, ratio, torch.ones_like(ratio)], dim=-1)

        self.band_visualizer.visualize(
            translations=band_pos_w, orientations=band_quat_w, scales=scales
        )


@configclass
class ArmDrapePoseCommandCfg(DeformableUniformPoseCommandCfg):
    """Configuration for :class:`ArmDrapePoseCommand`.

    The inherited :attr:`ranges` are reinterpreted as an offset in the arm's frame rather than an
    absolute region of the robot's workspace; the roll/pitch/yaw entries are unused.
    """

    class_type: type[ArmDrapePoseCommand] = ArmDrapePoseCommand

    arm_name: str = "mannequin_arm"
    """Scene entity of the arm the goal is placed on."""

    arm_nominal_radius: float | None = None
    """Arm radius the band marker was sized against [m]. ``None`` disables marker scaling.

    Only consulted when the arm's radius has been randomised per environment. The marker is one
    instanced prototype at a fixed radius, so without this a thickness-randomised arm would show the
    same sleeve everywhere -- buried in the fattened environments, standing clear of the thinned
    ones. Given the nominal, each instance is scaled radially by its own arm's ratio to it.

    Purely cosmetic: it scales what is drawn and nothing that is scored, since the reward terms read
    the per-environment radius directly.
    """

    arm_mount_quat: tuple[float, float, float, float] | None = None
    """Fixed rotation baked into the arm's spawn orientation, as ``(w, x, y, z)``. Body frame.

    ``None`` -- the default -- means the body *is* the task frame, which is the case whenever the
    asset is spawned as authored. Set it to whatever ``init_state.rot`` the arm carries and the
    command works in the frame the arm would have had without it.

    This exists to keep the goal command invariant under a purely presentational change to the arm.
    The command publishes the arm's orientation as four of its seven dimensions, and derives the
    goal point by lifting off the arm along that frame's +z, so re-mounting the asset -- turning it
    palm-up, say -- would both hand the policy a quaternion it never saw in training and push the
    goal through the arm to the underside. Declaring the mount here leaves both untouched.

    Only what the policy is scored and steered by is corrected. The band marker is still drawn from
    the body's true pose, because :attr:`band_centerline_offsets` are measured in the asset's own
    frame and have to turn with it to stay on the limb.
    """

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

    band_centerline_offsets: tuple[tuple[float, float, float], ...] | None = None
    """Samples of the arm's true centre line in its own frame, as ``(x, y, z)`` triples [m].

    ``None`` -- the default -- draws the band on the root frame's x-axis, which is exactly right for
    a capsule, since a capsule's centre line *is* that axis.

    A sculpted arm is not straight. Its cross-section centroid wanders off the root axis as the limb
    curves, so a sleeve pinned to the axis hangs off one side of the surface. Give the measured
    centroid at a few stations along the arm and the sleeve is slid sideways onto the real centre
    line, interpolating linearly between them and holding the end values beyond the sampled span.

    Only the marker is moved. The goal point and the analytic capsule the reward terms build are
    still taken about the root axis, so this changes what is drawn and nothing that is scored --
    which is also the caveat: on a markedly curved arm the drawn band and the scored region drift
    apart by however far the centre line strays.
    """
