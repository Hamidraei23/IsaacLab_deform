# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda picks a deformable sheet from a slot and drapes it over an arm."""

from __future__ import annotations

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    TABLE_SPAWN_CFG,
    TerminationsCfg,
    _FrankaSoftSceneCfg,
)
from isaaclab_tasks.utils import PresetCfg

from .mdp import (
    ArmDrapePoseCommandCfg,
    SquashedRelativeJointPositionActionCfg,
    applied_actions,
    randomize_arm_radius,
    band_approach_progress,
    closed_over_finished_drape,
    timeout_still_gripping,
    drape_complete,
    drape_failure_penalty,
    drape_milestones,
    extraction_bonus,
    gripper_closed_near_band,
    ee_speed_penalty,
    ee_table_clearance,
    finger_arm_contact_penalty,
    finger_in_slot,
    phase_one_finished,
    grasp_alignment,
    band_center_distance,
    band_coverage,
    goal_com_proximity,
    grasp_stage_reward,

    release_retreat_progress,
    release_stage_reward,
    released_before_extraction,
    released_too_high,
    reset_arm_and_sheet,
    sheet_key_points,
    sheet_lift_progress,
    table_drop_penalty,
    top_edge_progress,
)

# Reset randomization

POSE_XY_RANGE = 0.12
"""Planar offset bound applied to both the arm and the sheet at reset [m]."""

YAW_RANGE = (-math.pi / 2, math.pi / 2)
"""Yaw bound applied to both the arm and the sheet at reset [rad]."""

ARM_RADIUS_JITTER = 0.20
"""Fractional radius jitter, sampled once per environment at startup."""

EPISODE_STEPS = 500
"""Episode length in environment steps, at the 30 Hz control rate."""

FRANKA_JOINT_ACC_LIMITS = {
    "panda_joint1": 15.0,
    "panda_joint2": 7.5,
    "panda_joint3": 10.0,
    "panda_joint4": 12.5,
    "panda_joint5": 15.0,
    "panda_joint6": 20.0,
    "panda_joint7": 20.0,
}
"""Published Franka FCI joint acceleration limits [rad/s^2]."""

ACTION_VELOCITY_GAIN = 0.222
"""Steady joint speed per unit of held action [rad/s]. Measured for ``arm_action.scale``."""

DRAPE_SETTLE_STEPS = 60
"""Required stable dwell for a successful drape [steps]."""

# Scene geometry

SHEET_SIZE = (0.2, 0.2)
SHEET_RESOLUTION = (8, 8)

SLOT_WALL_SIZE = (0.10, 0.02, 0.15)
"""Slot wall dimensions: length, thickness, and height [m]."""

SLOT_GAP = 0.01
"""Clear space between the two walls [m]. The sheet stands in this gap."""

SLOT_WALL_OFFSET_Y = 0.5 * (SLOT_GAP + SLOT_WALL_SIZE[1])
"""Distance from the slot centre to each wall's centre [m]."""

SLOT_WALL_SPAWN_CFG = sim_utils.CuboidCfg(
    size=SLOT_WALL_SIZE,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # Low friction prevents the sheet snagging during extraction.
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
)

SHEET_SPAWN_POS = (0.5, 0.18, 0.102)
"""Initial sheet centre; the sheet stands in the slot on the arm's opposite side."""

SHEET_SPAWN_ROT = (0.70710678, 0.0, 0.0, 0.70710678)
"""Quaternion that places the sheet upright in the slot."""

MANNEQUIN_ARM_RADIUS = 0.04
MANNEQUIN_ARM_LENGTH = 0.28
"""Length of the capsule's cylindrical section [m]; total span is length + 2 * radius."""

MANNEQUIN_ARM_SPAWN_POS = (0.5, -0.15, MANNEQUIN_ARM_RADIUS)
"""Arm centre at spawn [m]. z equals the radius, so the capsule rests on the table."""

TARGET_BAND_WIDTH = 0.10
"""Extent of the red target band along the arm's axis [m]."""

SLOT_CLEAR_HEIGHT = SLOT_WALL_SIZE[2] + 0.02
"""Minimum sheet height for extraction from the slot [m]."""

TARGET_BAND_TRAVEL = (MANNEQUIN_ARM_LENGTH - TARGET_BAND_WIDTH) / 2.0
"""Half-range for band placement along the arm's cylindrical section [m]."""

MANNEQUIN_ARM_SPAWN_CFG = sim_utils.CapsuleCfg(
    radius=MANNEQUIN_ARM_RADIUS,
    height=MANNEQUIN_ARM_LENGTH,
    axis="X",
    # Keep the target fixed against gravity and sheet contact.
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # Higher friction than the table helps the drape settle.
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.72, 0.62)),
)


# Scene definition


@configclass
class PhysicsCfg(PresetCfg):
    """Cloth physics, with every rigid body the sheet has to touch coupled into the soft solver."""

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                    ),
                    bodies=[
                        r"/World/envs/env_.*/Robot",
                        r"/World/envs/env_.*/MannequinArm",
                        r"/World/envs/env_.*/Slot(Neg|Pos)Y",
                    ],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=1024),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_(left|right)finger",
                        # Newton couples only bodies explicitly proxied into the soft solver.
                        r"/World/envs/env_.*/MannequinArm",
                        r"/World/envs/env_.*/Slot(Neg|Pos)Y",
                    ],
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                )
            ],
            iterations=1,
            model_cfg=NewtonModelCfg(soft_contact_ke=8e3, soft_contact_mu=10.0),
        ),
        num_substeps=2,
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class SheetDeformableCfg(PresetCfg):
    """The sheet, lying flat on the table."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        # Spawn upright in the slot.
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=SHEET_SPAWN_POS,
            rot=SHEET_SPAWN_ROT,
        ),
        spawn=sim_utils.MeshRectangleCfg(
            size=SHEET_SIZE,
            resolution=SHEET_RESOLUTION,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=10.0,
                particle_radius=0.002,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=0.5,
                edge_kd=1e-3,
            ),
        ),
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class SheetSceneCfg(_FrankaSoftSceneCfg):
    """Franka, sheet, and table. No support blocks: nothing stands the sheet up."""

    deformable: SheetDeformableCfg = SheetDeformableCfg()

    # A RigidObject supports per-environment reset poses.
    mannequin_arm: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MannequinArm",
        init_state=RigidObjectCfg.InitialStateCfg(pos=MANNEQUIN_ARM_SPAWN_POS),
        spawn=MANNEQUIN_ARM_SPAWN_CFG,
    )

    # Reset both slot walls with the sheet.
    slot_neg_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SlotNegY",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                SHEET_SPAWN_POS[0],
                SHEET_SPAWN_POS[1] - SLOT_WALL_OFFSET_Y,
                0.5 * SLOT_WALL_SIZE[2],
            )
        ),
        spawn=SLOT_WALL_SPAWN_CFG,
    )
    slot_pos_y: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SlotPosY",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                SHEET_SPAWN_POS[0],
                SHEET_SPAWN_POS[1] + SLOT_WALL_OFFSET_Y,
                0.5 * SLOT_WALL_SIZE[2],
            )
        ),
        spawn=SLOT_WALL_SPAWN_CFG,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Drive both fingers separately; the upstream passive finger cannot hold the sheet.
        gripper = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            # Extra effort maintains a firm position-controlled pinch.
            effort_limit_sim=200.0,
            velocity_limit=0.2,
            velocity_limit_sim=2.0,
            # High stiffness produces useful force from the small pinch error.
            stiffness=2000.0,
            damping=200.0,
            armature=0.1,
        )
        self.robot.actuators["panda_hand"] = gripper
        self.robot.actuators["panda_finger2_passive"] = gripper.replace(
            joint_names_expr=["panda_finger_joint2"]
        )

    # Lower only table friction; global soft friction also controls the gripper.
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG.replace(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
        ),
    )


@configclass
class SheetScenePresetCfg(PresetCfg):
    newton_mjwarp_vbd_proxy: SheetSceneCfg = SheetSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)

    default = newton_mjwarp_vbd_proxy


# MDP settings


@configclass
class SheetEventCfg:
    """Reset events: robot to its default configuration, sheet flat on the table.

    Plus one start-up term. ``randomize_arm_radius`` runs in ``startup`` mode -- once, after the
    Newton model is finalised and before the first reset -- because a collider's size can only be
    changed by writing the model directly, and it must be in place before ``reset_layout`` seats
    each arm at its own radius.
    """

    randomize_arm_radius = EventTerm(
        func=randomize_arm_radius,
        mode="startup",
        params={
            "nominal_radius": MANNEQUIN_ARM_RADIUS,
            "jitter": ARM_RADIUS_JITTER,
            "asset_cfg": SceneEntityCfg("mannequin_arm"),
        },
    )

    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
        },
    )

    reset_robot_gripper_joints = EventTerm(
        func=mdp.reset_joints_shared_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
        },
    )

    # Reset together before the command manager derives the new goal.
    reset_layout = EventTerm(
        func=reset_arm_and_sheet,
        mode="reset",
        params={
            "position_range": {
                "x": (-POSE_XY_RANGE, POSE_XY_RANGE),
                "y": (-POSE_XY_RANGE, POSE_XY_RANGE),
                "z": (0.0, 0.0),
            },
            "yaw_range": YAW_RANGE,
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
            "sheet_cfg": SceneEntityCfg("deformable"),
            # Must be a list: Hydra does not sanitize SceneEntityCfg slices inside tuples.
            "slot_cfgs": [SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")],
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
            "operation": "abs",
        },
    )


@configclass
class SheetRewardsCfg:
    """Two phases, switched by one latch, with no reward paid in both.

    **Phase one -- pick.** Come down over the sheet, take the middle of its top edge, and draw it
    straight up out of the slot. ``approach``, ``alignment`` and ``lift_progress`` shape it;
    ``grasp_stage`` pays for the grasp and the extraction.

    **Phase two -- drape.** Latched the instant the sheet clears the walls, which is where phase
    one's terms stop paying and where the episode used to end. ``drape_closeness`` pulls the
    sheet's centre onto the centre of the red band, ``release_stage`` pays for opening the gripper
    over it -- or charges for dropping it from height -- and ``release_retreat`` pays for taking the
    hand back up out of the way afterwards, so the drape settles without the fingers in it.

    Every phase-one term carries a ``phase_one_only`` gate rather than being left to run.
    ``alignment`` would actively fight the drape by paying for the phase-one grasp posture, and the
    other two would pay a near-constant offset for the rest of the episode, diluting the only
    signal phase two has.

    ``table_clearance`` and ``ee_speed`` are ungated on purpose: they are safety and
    regularisation, not task shaping, and they mean the same thing in both phases.
    """

    # -- getting into position
    approach = RewTerm(
        func=top_edge_progress,
        params={
            "resolution": SHEET_RESOLUTION,
            "coarse_std": 0.4,
            "fine_std": 0.05,
            # Ratcheting caps the total approach payout and prevents hovering income.
            "max_reward": 1000.0,
            # Stop when the first grasp is scored.
            "until_grasp": True,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        # max_reward is the literal payout after internal dt normalization.
        weight=1.0,
    )

    alignment = RewTerm(
        func=grasp_alignment,
        params={
            "resolution": SHEET_RESOLUTION,
            # Keep the alignment signal active throughout the approach.
            "std": 0.2,
            # Score 50 when aligned and 5 at 20 degrees.
            "max_reward": 50.0,
            "reference_error_deg": 20.0,
            "reward_at_reference": 5.0,
            # Reward pad alignment only; the downward axis is already solved.
            "downward_frac": 0.0,
            # Disable the phase-one wrist target after extraction.
            "phase_one_only": True,
            "asset_cfg": SceneEntityCfg("deformable"),
            "slot_cfg": SceneEntityCfg("slot_neg_y"),
        },
        weight=1.0,
    )

    table_clearance = RewTerm(
        func=ee_table_clearance,
        params={"floor": 0.06, "robot_cfg": SceneEntityCfg("robot")},
        # Make table dives cost more than their shortcut is worth.
        weight=-2.0,
    )

    # Optional Cartesian-speed charge: free below 0.2 m/s and capped above 1 m/s.
    # ``--slow`` sets weight -4, making the ceiling 190.4 per step.
    ee_speed = RewTerm(
        func=ee_speed_penalty,
        params={
            "free_speed": 0.2,
            "reference_speed": 1.0,
            "reference_penalty": 47.6,
            # Level off at the reference speed.
            "max_penalty": 47.6,
            "hand_body_name": "panda_hand",
            "robot_cfg": SceneEntityCfg("robot"),
        },
        # The weight supplies the sign; magnitudes are literal per-step charges.
        weight=0.0,
    )

    # -- the outcome
    grasp_stage = RewTerm(
        func=grasp_stage_reward,
        params={
            "asset_cfg": SceneEntityCfg("deformable"),
            "robot_cfg": SceneEntityCfg("robot"),
            "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
            "slot_cfg": SceneEntityCfg("slot_neg_y"),
            "resolution": SHEET_RESOLUTION,
            "hand_body_name": "panda_hand",
            "left_finger_body_name": "panda_leftfinger",
            "right_finger_body_name": "panda_rightfinger",
            # Judge capture by cloth between the pads; alignment must not veto a valid grasp.
            # Bounds how far off the closing axis the cloth may sit, i.e. the pad footprint.
            "capture_radius": 0.04,
            # Slack past a pad face, for the sheet's 25 mm node spacing. Raise it if grasps that
            # plainly worked stop registering, lower it if closing beside the sheet still pays.
            "pinch_margin": 0.010,
            "open_threshold": 0.030,
            "closed_threshold": 0.012,
            "first_grasp_bonus": 500.0,
            # Keep failed closures cheap enough that the policy still attempts grasps.
            "bad_closure_penalty": 20.0,
            "premature_release_penalty": 1000.0,
            "slot_clear_height": SLOT_CLEAR_HEIGHT,
            # Make extraction worthwhile before phase-two rewards become reachable.
            "extraction_bonus": 4000.0,
            "alignment_max_reward": 50.0,
            "alignment_offset": 1.0,
        },
        # Extraction is scaled by alignment; other event values are literal.
        weight=1.0,
    )

    # Shape the holding phase toward extraction, scaled by alignment.
    lift_progress = RewTerm(
        func=sheet_lift_progress,
        params={
            "lift_span": 0.15,
            # Total before the alignment multiplier. 3.5x the 500 it was.
            "max_reward": 2500.0,
            "alignment_max_reward": 50.0,
            "alignment_offset": 45.0,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=1.0,
    )

    # -- phase two: drape the sheet on the red band
    # Ratchet closeness so approaching pays once and hovering pays nothing.
    drape_closeness = RewTerm(
        func=band_approach_progress,
        params={
            "command_name": "deformable_pose",
            "std": 0.2,
            # Total bounded payout for the approach.
            "max_reward": 2000.0,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        # max_reward is the literal total after internal dt normalization.
        weight=1.0,
    )

    # Reward spread over the band, which centre distance cannot distinguish from bunching.
    # This farmable level reward pays 40 per step at full coverage.
    drape_coverage = RewTerm(
        func=band_coverage,
        params={
            "command_name": "deformable_pose",
            "band_length": TARGET_BAND_WIDTH,
            "band_radius": MANNEQUIN_ARM_RADIUS,
            # Exceed the 0.025 m node spacing.
            "cover_threshold": 0.03,
            "num_axial": 5,
            "num_angular": 7,
            "coverage_arc": math.pi / 2,
            "gate_on_phase": True,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        # 40 per step at full coverage.
        weight=1200.0,
    )

    # Must follow ``drape_coverage`` because it reads that term's published fraction.
    # Pays 15 at the goal, tapering to 1 at 15 cm; monitor it for hovering income.
    goal_proximity = RewTerm(
        func=goal_com_proximity,
        params={
            "command_name": "deformable_pose",
            "max_reward": 15.0,
            "min_reward": 1.0,
            # Shared reach and failure radius.
            "max_distance": 0.15,
            # Do not reward favorable reset placement before extraction.
            "gate_on_phase": True,
            # Hand off to drape rewards at the success threshold.
            "stop_coverage": 0.2,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        # Values are literal per-step amounts after internal dt normalization.
        weight=1.0,
    )

    # One-shot milestones; must follow the coverage term they read.
    drape_stage = RewTerm(
        func=drape_milestones,
        params={
            "command_name": "deformable_pose",
            # Use the shared arrival radius.
            "reach_radius": 0.15,
            "reach_bonus": 80.0,
            "touch_coverage": 0.0,
            "touch_bonus": 40.0,
            # Require a released sheet to cover 20% of the reachable band.
            "success_coverage": 0.2,
            # Require two seconds of stable coverage instead of scoring a passing sheet.
            "dwell_steps": DRAPE_SETTLE_STEPS,
            # Compare this bonus with the coverage income that termination cuts off.
            "success_bonus": 3200.0,
            # Scale milestones by the highest-corner symmetry score; 5000/4000 gives 1.25x max.
            "resolution": SHEET_RESOLUTION,
            "symmetry_bonus": 5000.0,
            "symmetry_divisor": 4000.0,
            "symmetry_span": 0.1,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        # Milestone values are literal one-shots after dt normalization.
        weight=1.0,
    )

    # Charge the commanded closed state near the band until the policy releases.
    hold_over_band = RewTerm(
        func=gripper_closed_near_band,
        params={
            "command_name": "deformable_pose",
            "radius": 0.15,
            "penalty": 1.0,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        # The negative weight makes penalty a literal per-step charge.
        weight=-1.0,
    )

    # Penalize holding an otherwise finished drape more than its coverage income.
    # Must follow ``drape_coverage``, whose published fraction it reads.
    blocked_by_gripper = RewTerm(
        func=closed_over_finished_drape,
        params={
            # Match the success threshold.
            "success_coverage": 0.2,
            # Exceed the observed ~17-per-step coverage income.
            "penalty": 40.0,
            # Ignore a closed command after the cloth has left the pads.
            "require_holding": True,
            "action_term_name": "gripper_action",
        },
        # The negative weight makes penalty a literal per-step charge.
        weight=-1.0,
    )

    # Penalize each new finger-arm contact; the kinematic target otherwise hides collisions.
    finger_touches_arm = RewTerm(
        func=finger_arm_contact_penalty,
        params={
            "arm_radius": MANNEQUIN_ARM_RADIUS,
            "arm_length": MANNEQUIN_ARM_LENGTH,
            # Match the slot collision margin.
            "margin": 0.005,
            "penalty": 800.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
            "finger_body_names": ("panda_leftfinger", "panda_rightfinger"),
        },
        # The negative weight makes penalty the literal charge per contact.
        weight=-1.0,
    )

    # Reward release below 0.1 m, grade penalties to 0.25 m, and abort above it.
    release_stage = RewTerm(
        func=release_stage_reward,
        params={
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
            "robot_cfg": SceneEntityCfg("robot"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
            "hand_body_name": "panda_hand",
            "std": 0.2,
            # Release near the arm surface without a height penalty.
            "release_height": 0.1,
            # Reach the full graded penalty here.
            "high_release_limit": 0.25,
            # Scale the release payout by placement quality.
            "release_bonus": 500.0,
            "high_release_penalty": 1000.0,
            # Abort releases above the limit.
            "abort_penalty": 1500.0,
        },
        # Values are literal one-shot returns after internal dt normalization.
        weight=1.0,
    )

    # Ratchet progress toward a retreat point latched 0.2 m above the first release.
    # Latching once prevents repeated close-open cycles from reselling the same climb.
    release_retreat = RewTerm(
        func=release_retreat_progress,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "hand_body_name": "panda_hand",
            "action_term_name": "gripper_action",
            # Use a point overhead so sideways motion does not count.
            "retreat_height": 0.2,
            "std": 0.1,
            # Total bounded payout for the retreat.
            "max_reward": 4000.0,
        },
        # max_reward is literal after internal dt normalization.
        weight=1.0,
    )

    # Charge a settled table drop once without ending the episode.
    # Keep it moderate so fear of dropping does not suppress all releases.
    dropped_on_table = RewTerm(
        func=table_drop_penalty,
        params={
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
            "arm_radius": MANNEQUIN_ARM_RADIUS,
            "arm_length": MANNEQUIN_ARM_LENGTH,
            # Separate a flat drop from a correct drape around 0.08 m.
            "table_height": 0.03,
            "touch_margin": 0.01,
            # Ignore brief passes near the table.
            "settle_steps": 5,
            "penalty": 500.0,
        },
        # penalty is literal after internal dt normalization.
        weight=1.0,
    )

    # Disabled terminal marker; per-step charges currently handle failure to release.
    timeout_gripping = RewTerm(
        func=timeout_still_gripping,
        params={
            "penalty": 0.0,
            "action_term_name": "gripper_action",
        },
        weight=1.0,
    )

    # -- smoothness.
    # Penalize both command reversals and resulting arm acceleration.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    # Exclude binary fingers, whose snap is intentional.
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        weight=-0.1,
    )

    # Enabled only by ``--phase_one`` to avoid duplicating the full-task extraction payout.
    extraction_complete = RewTerm(
        func=extraction_bonus,
        params={"bonus": 1000.0},
        weight=0.0,
    )

    # Charge any terminal state outside the shared success radius.
    drape_failure = RewTerm(
        func=drape_failure_penalty,
        params={
            "command_name": "deformable_pose",
            "success_radius": 0.15,
            "penalty": 1100.0,
            "asset_cfg": SceneEntityCfg("deformable"),
            "arm_cfg": SceneEntityCfg("mannequin_arm"),
        },
        weight=1.0,
    )


@configclass
class SheetTerminationsCfg(TerminationsCfg):
    """The inherited bounds, plus the three ways an episode ends early.

    Two phase-two failures that could be here are deliberately *not*: a release from above the
    ceiling and a sheet dropped flat on the table are both charged -- 300 and 500 respectively --
    and both leave the episode running. Ending on them would decide the outcome at the moment the
    sheet leaves the hand, which is the one moment the physics has not yet resolved: a sheet let go
    high over the band can still land on the arm, settle and be scored a success, and a sheet on
    the table still has to be paid for by ``drape_failure`` at whatever ending does arrive. Charging
    without ending keeps the price of the mistake and lets the outcome decide the rest.

    Extraction is conspicuously not here either. Pulling the sheet clear of the slot used to be the
    task and ended the episode on the spot; it is now the boundary between the two phases, and the
    episode carries straight on into the drape.
    """

    released_early = DoneTerm(func=released_before_extraction)

    # End only releases above ``high_release_limit``.
    released_high = DoneTerm(func=released_too_high)

    # End after two seconds of stable band coverage.
    draped = DoneTerm(func=drape_complete)

    # Declared disabled so ``--phase_one`` can enable it through Hydra.
    extracted = DoneTerm(func=phase_one_finished, params={"enabled": False})

    finger_touched_slot = DoneTerm(
        func=finger_in_slot,
        params={
            # Half the box enclosing both walls in slot coordinates.
            "half_extents": (
                0.5 * SLOT_WALL_SIZE[0],
                SLOT_WALL_OFFSET_Y + 0.5 * SLOT_WALL_SIZE[1],
                0.5 * SLOT_WALL_SIZE[2],
            ),
            "margin": 0.005,
            "robot_cfg": SceneEntityCfg("robot"),
            # Must be a list for Hydra slice serialization.
            "slot_cfgs": [SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")],
            "finger_body_names": ("panda_leftfinger", "panda_rightfinger"),
        },
    )


@configclass
class SheetCurriculumCfg:
    """Empty, replacing the inherited curriculum rather than switching its terms off one by one.

    The parent ramps gravity up from nothing and ramps in an action-rate penalty. Neither applies:
    this task wants full gravity from the first frame so the sheet hangs the way it looks like it
    should, and the action-rate reward the second one scales does not exist here.
    """


@configclass
class SheetCommandsCfg:
    """Goal pose for the sheet, anchored to the mannequin arm."""

    deformable_pose = ArmDrapePoseCommandCfg(
        asset_name="robot",
        object_name="deformable",
        arm_name="mannequin_arm",
        # Keep the visual sleeve on a radius-randomized arm.
        arm_nominal_radius=MANNEQUIN_ARM_RADIUS,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # Arm-frame band offset and draped centre-of-mass height.
        ranges=ArmDrapePoseCommandCfg.Ranges(
            pos_x=(-TARGET_BAND_TRAVEL, TARGET_BAND_TRAVEL),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.03),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        band_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/target_band",
            markers={
                "band": sim_utils.CylinderCfg(
                    # Add 1 mm to avoid z-fighting while keeping the sleeve flush.
                    radius=MANNEQUIN_ARM_RADIUS + 0.001,
                    height=TARGET_BAND_WIDTH,
                    # The command rotates this marker; Newton ignores prototype axes.
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                )
            },
        ),
        # Tint the marker-rendered table when the goal is reached.
        success_vis_asset_name="table",
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                ),
                "success": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                ),
            },
        ),
    )


@configclass
class _SheetJointActionsCfg:
    """7-dim relative joint-position arm targets + 1-dim binary gripper."""

    # max_delta is filled in from FRANKA_JOINT_ACC_LIMITS in SheetRlEnvCfg.__post_init__, where the
    # real control period is known.
    arm_action = SquashedRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.03
    )

    # Binary control preserves the open-close transitions used by grasp rewards.
    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        # One binary action drives both fingers symmetrically.
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
        close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        # Split an initial Gaussian policy evenly between open and closed.
        threshold=0.0,
        positive_threshold=True,
    )


@configclass
class _SheetIkActionsCfg:
    """7-dim absolute end-effector pose via differential IK + the same 1-dim binary gripper.

    For teleoperation, not for training: a keyboard produces an end-effector motion, not seven
    joint deltas. The gripper term is deliberately identical to the joint preset's, so what a human
    tries by hand drives exactly the gripper channel the policy will be trained on.

    Absolute rather than relative on purpose. In relative mode the controller re-derives its target
    from the *current* end-effector pose on every step, so a zero command means "target wherever
    you are now" -- and since these actuators do not compensate gravity, that re-latches the
    arm's own droop as the setpoint each step and it sinks to the floor. An absolute target is
    latched once and held, so the controller keeps pulling back against gravity. The operator
    integrates their twist into that target instead, which is the teleop script's job.
    """

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.6},
        ),
        # Match the grasp frame used by the reward.
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        # One binary action drives both fingers symmetrically.
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
        close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        threshold=0.0,
        positive_threshold=True,
    )


@configclass
class SheetActionsCfg(PresetCfg):
    joint: _SheetJointActionsCfg = _SheetJointActionsCfg()

    ik_rel: _SheetIkActionsCfg = _SheetIkActionsCfg()

    default = joint


# Environment configuration


@configclass
class SheetRlEnvCfg(FrankaClothEnvCfg):
    """Franka Panda picking a flat sheet off the table and placing it at the commanded pose.

    Rewards, terminations, commands and observations are inherited from the cloth task and are
    meaningful again here: the sheet starts on the table, so the reach, lift and goal-tracking
    terms all have something to reward.
    """

    scene: SheetScenePresetCfg = SheetScenePresetCfg()
    actions: SheetActionsCfg = SheetActionsCfg()
    events: SheetEventCfg = SheetEventCfg()
    commands: SheetCommandsCfg = SheetCommandsCfg()
    rewards: SheetRewardsCfg = SheetRewardsCfg()
    terminations: SheetTerminationsCfg = SheetTerminationsCfg()
    curriculum: SheetCurriculumCfg = SheetCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Replace the inherited support-block physics preset.
        self.sim.physics = PhysicsCfg()
        # Half-step slack prevents floating-point ceil from producing 501 steps.
        self.episode_length_s = (EPISODE_STEPS - 0.5) * self.decimation * self.sim.dt
        # Resample the band only at episode boundaries.
        self.commands.deformable_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # Use nine stable landmarks instead of randomly changing node identities.
        self.observations.policy.deformable_sampled_points = ObsTerm(
            func=sheet_key_points,
            params={"resolution": SHEET_RESOLUTION, "asset_cfg": SceneEntityCfg("deformable")},
        )
        # Hard per-joint bound on commanded acceleration, derived here because the control period is
        # only known once decimation and the physics timestep are resolved.
        control_dt = self.decimation * self.sim.dt
        self.actions.joint.arm_action.max_delta = {
            name: limit * control_dt / ACTION_VELOCITY_GAIN
            for name, limit in FRANKA_JOINT_ACC_LIMITS.items()
        }
        # The squashed term measures each increment against its own applied action, so that is what
        # the policy has to see; ``last_action`` reports the request, which differs by the squash.
        self.observations.policy.actions = ObsTerm(func=applied_actions)
