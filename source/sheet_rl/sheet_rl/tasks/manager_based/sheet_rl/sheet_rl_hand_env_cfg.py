# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation sheet task using a mesh-collided sculpted arm and hand."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import NewtonCfg

from .mdp import simulation_diverged

from .sheet_rl_env_cfg import (
    TARGET_BAND_WIDTH,
    SheetRlEnvCfg,
    SheetSceneCfg,
    SheetScenePresetCfg,
)

# Arm asset

ARM_AND_HAND_USD_PATH = str(
    Path(__file__).resolve().parents[6] / "hand_model" / "arm_and_hand_arm_frame.usda"
)
"""Task-frame wrapper for the converted sculpted arm asset."""

ARM_ROLL_QUAT = (0.0, 1.0, 0.0, 0.0)
"""Palm-up 180-degree roll about the limb's +X axis, as ``(w, x, y, z)``."""

ARM_REST_HEIGHT = 0.0560
"""Root height that seats the rolled arm on the table [m]."""

# Analytical reward geometry

ARM_CAPSULE_RADIUS = 0.025
"""Radius of the elbow-to-wrist capsule used by reward terms [m]."""

ARM_CAPSULE_LENGTH = 0.3368
"""Cylindrical length of the reward capsule [m]."""

ARM_BAND_TRAVEL = 0.5 * (ARM_CAPSULE_LENGTH - TARGET_BAND_WIDTH)
"""Half-range of the target band's travel along the forearm [m]."""

BAND_MARKER_RADIUS = 0.025
"""Radius of the visible target sleeve [m]."""

ARM_CENTERLINE_OFFSETS = (
    (-0.200, +0.0184, +0.0039),
    (-0.180, +0.0151, -0.0009),
    (-0.160, +0.0155, -0.0054),
    (-0.140, +0.0170, -0.0031),
    (-0.120, +0.0121, -0.0005),
    (-0.100, +0.0080, +0.0032),
    (-0.080, +0.0007, +0.0047),
    (-0.060, -0.0014, +0.0054),
    (-0.040, -0.0047, +0.0016),
    (-0.020, -0.0056, -0.0009),
    (+0.000, -0.0101, -0.0014),
    (+0.020, -0.0177, +0.0004),
    (+0.040, -0.0196, +0.0007),
    (+0.060, -0.0226, +0.0023),
    (+0.080, -0.0273, +0.0028),
    (+0.100, -0.0281, +0.0005),
    (+0.120, -0.0255, -0.0032),
    (+0.140, -0.0240, -0.0034),
    (+0.160, -0.0238, -0.0063),
    (+0.180, -0.0216, -0.0083),
    (+0.200, -0.0166, -0.0106),
)
"""Measured arm-frame centre-line samples ``(x, y, z)`` at 2 cm intervals [m]."""

# Push

PUSH_FORCE = 3.0
"""Force applied to the sheet's latched lower-edge node [N]."""

PUSH_DURATION_S = 0.15
"""Push duration [s]."""

PARTICLE_CONTACT_BUFFER = 8192
"""Rigid-body-to-particle contact capacity for mesh collision."""

TRIANGLE_PAIR_BUFFER = 8_000_000
"""Narrow-phase triangle-pair capacity for rigid-soft collision."""


# Scene


@configclass
class SheetArmMeshSceneCfg(SheetSceneCfg):
    """Base scene with the capsule replaced by the sculpted arm."""

    mannequin_arm: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MannequinArm",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.15, ARM_REST_HEIGHT), rot=ARM_ROLL_QUAT
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=ARM_AND_HAND_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            # Mass ensures Newton registers the kinematic body.
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # Extra grip keeps the sheet from sliding off.
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
        ),
    )


@configclass
class SheetArmMeshScenePresetCfg(SheetScenePresetCfg):
    # Mesh contact favors a smaller evaluation batch.
    newton_mjwarp_vbd_proxy: SheetArmMeshSceneCfg = SheetArmMeshSceneCfg(
        num_envs=64, env_spacing=2.0, replicate_physics=True
    )

    default = newton_mjwarp_vbd_proxy


# Environment


@configclass
class SheetRlHandEnvCfg(SheetRlEnvCfg):
    """The sheet task with a sculpted, mesh-collided arm and hand. Evaluation only."""

    scene: SheetArmMeshScenePresetCfg = SheetArmMeshScenePresetCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.push.force = PUSH_FORCE
        self.push.duration_s = PUSH_DURATION_S

        # Reset unrecoverable solver states, including NaNs.
        self.terminations.diverged = DoneTerm(func=simulation_diverged)

        # A triangle mesh has no capsule radius to randomize.
        self.events.randomize_arm_radius = None

        # Rewards use an analytical capsule instead of the collider.
        self.rewards.drape_coverage.params["band_radius"] = ARM_CAPSULE_RADIUS
        for term in (self.rewards.finger_touches_arm, self.rewards.dropped_on_table):
            term.params["arm_radius"] = ARM_CAPSULE_RADIUS
            term.params["arm_length"] = ARM_CAPSULE_LENGTH

        command = self.commands.deformable_pose
        # Preserve the unrolled task frame; must match ``init_state.rot``.
        command.arm_mount_quat = ARM_ROLL_QUAT
        command.ranges.pos_x = (-ARM_BAND_TRAVEL, ARM_BAND_TRAVEL)
        command.band_visualizer_cfg.markers["band"].radius = BAND_MARKER_RADIUS
        command.band_visualizer_cfg.markers["band"].height = TARGET_BAND_WIDTH
        command.band_centerline_offsets = ARM_CENTERLINE_OFFSETS
        # Draw the sleeve at the goal rather than the body's centre line.
        command.band_follows_goal = True

        for value in vars(self.sim.physics).values():
            if not isinstance(value, NewtonCfg):
                continue
            # Visit every preset because ``default`` may be a distinct copy.
            for entry in getattr(value.solver_cfg, "entries", []) or []:
                solver = entry.solver_cfg
                if hasattr(solver, "rigid_body_particle_contact_buffer_size"):
                    solver.rigid_body_particle_contact_buffer_size = PARTICLE_CONTACT_BUFFER
            # The narrow-phase buffer belongs to each proxy, not its solver entry.
            for proxy in getattr(value.solver_cfg, "proxies", []) or []:
                pipeline = getattr(proxy, "collision_pipeline", None)
                if pipeline is not None and hasattr(pipeline, "max_triangle_pairs"):
                    pipeline.max_triangle_pairs = TRIANGLE_PAIR_BUFFER
