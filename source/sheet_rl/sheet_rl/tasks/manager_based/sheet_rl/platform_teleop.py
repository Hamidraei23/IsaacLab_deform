# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the platform from the keyboard while the policy runs (user request, 2026-11-18).

Hold ``W``/``S`` to move the platform along world +x/-x, ``A``/``D`` along +y/-y, and ``Q``/``E`` to
turn it about the vertical axis (``Q`` counter-clockwise seen from above, ``E`` clockwise). Its
height is never changed, so the per-episode random height stays as drawn.

**Two things move, not one.** The platform is the kinematic ``mannequin_arm`` box, written
directly with ``write_root_pose_to_sim_index`` as the reset event does. But the policy never sees
that box: it sees the ``deformable_pose`` command, the platform top-centre in the robot frame,
which the task resamples only at reset. Moving the box alone would leave the policy aiming at the
old spot. So after every move the command is re-read from the box (``PlatformPoseCommand``'s
``_resample_command`` draws nothing random; it only measures the box), before the step computes its
observation -- the policy sees the new goal on the same step the box moves.

**Off by default.** Nothing is bound and nothing is written unless ``env.platform_teleop.enabled``
is true (``scripts/play.py --platform_teleop``), so training and ordinary playback are unchanged.

In the Newton GL viewer ``WASDQE`` also fly the camera. While teleop is on, the presses of those six
keys are consumed by the teleop handler, so the camera stays put; the arrow keys and the mouse still
move the camera. The episode's reset still re-randomises the platform as usual.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class PlatformTeleopCfg:
    """Keyboard control of the platform. Every field is reachable through Hydra, e.g.
    ``env.platform_teleop.speed=0.2``."""

    enabled: bool = False
    """Bind the keys and move the platform. Off leaves the environment exactly as it was."""

    speed: float = 0.10
    """Translation speed while a key is held [m/s] (2 mm per 50 Hz control step)."""

    yaw_rate: float = math.radians(30.0)
    """Turn rate while ``Q`` or ``E`` is held [rad/s]."""

    platform_name: str = "mannequin_arm"
    """Scene entity of the platform box."""

    command_name: str = "deformable_pose"
    """Command that publishes the platform pose to the policy; refreshed after every move."""


KEYS = {
    # key: (dx, dy, dyaw) direction while held
    "W": (1.0, 0.0, 0.0),
    "S": (-1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "D": (0.0, -1.0, 0.0),
    "Q": (0.0, 0.0, 1.0),
    "E": (0.0, 0.0, -1.0),
}
"""Held key -> (x, y, yaw) direction, world axes (x away from the robot base, y to its left)."""


class PlatformTeleop:
    """Moves the platform by whichever of :data:`KEYS` are held; call :meth:`apply` before each step."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: PlatformTeleopCfg) -> None:
        self.env = env
        self.cfg = cfg
        self.keys = tuple(KEYS)
        self.held: set[str] = set()
        self._moving = False
        self._platform = env.scene[cfg.platform_name]
        self._command = env.command_manager.get_term(cfg.command_name)
        self._all_ids = torch.arange(env.num_envs, device=env.device)
        print(
            f"[sheet_rl] platform teleop on: W/S = +x/-x, A/D = +y/-y, Q/E = yaw +/- "
            f"({cfg.speed:.2f} m/s, {math.degrees(cfg.yaw_rate):.0f} deg/s)"
        )

    """
    Key callbacks (viewer thread: only touch the held set).
    """

    def press(self, key: str) -> None:
        self.held.add(key.upper())

    def release(self, key: str) -> None:
        self.held.discard(key.upper())

    def release_all(self) -> None:
        self.held.clear()

    """
    Operations.
    """

    def apply(self) -> None:
        """Move the platform one control step's worth for the held keys and refresh the goal command."""
        held = [KEYS[k] for k in tuple(self.held) if k in KEYS]
        if not held:
            if self._moving:
                self._moving = False
                self._report()
            return
        self._moving = True

        dt = float(self.env.step_dt)
        dx = sum(h[0] for h in held) * self.cfg.speed * dt
        dy = sum(h[1] for h in held) * self.cfg.speed * dt
        dyaw = sum(h[2] for h in held) * self.cfg.yaw_rate * dt

        pose = self._platform.data.root_pose_w.torch.clone()
        pose[:, 0] += dx
        pose[:, 1] += dy
        if dyaw != 0.0:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            turn = quat_from_euler_xyz(zeros, zeros, torch.full_like(zeros, dyaw))
            # world-frame turn about +z, applied on the left
            pose[:, 3:7] = quat_mul(turn, pose[:, 3:7])
        self._platform.write_root_pose_to_sim_index(root_pose=pose, env_ids=self._all_ids)
        self._platform.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((self.env.num_envs, 6), device=self.env.device), env_ids=self._all_ids
        )
        # re-measure the goal from the moved box (no randomness in PlatformPoseCommand's resample)
        self._command._resample_command(self._all_ids)

    """
    Internal helpers.
    """

    def _report(self) -> None:
        pose = self._platform.data.root_pose_w.torch[0]
        local = pose[:3] - self.env.scene.env_origins[0]
        _, _, yaw = euler_xyz_from_quat(pose[3:7].unsqueeze(0))
        yaw_deg = math.degrees(math.remainder(float(yaw[0]), 2 * math.pi))
        print(f"[teleop] platform (env 0): x={local[0]:.3f} m  y={local[1]:.3f} m  yaw={yaw_deg:.1f} deg", flush=True)
