# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda picking a deformable sheet out of a slot and draping it over a target region.

Derived from the Isaac Lab cloth-lifting task, and like that task the sheet stands upright in a
slot formed by two thin walls, projecting above them so the robot has a free edge to pinch. What is
new here is where it has to go: pulling the sheet clear of the slot opens up the placement
objective -- draping it over a red band on a mannequin arm lying on the table.
Both the slot and the arm are randomly placed at reset, on randomly swapped sides.
"""

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
    ee_speed_penalty,
    ee_table_clearance,
    finger_in_slot,
    grasp_alignment,
    band_center_distance,
    band_coverage,
    grasp_stage_reward,
    gripper_recommit_penalty,
    released_before_extraction,
    reset_arm_and_sheet,
    sheet_extracted,
    sheet_key_points,
    sheet_lift_progress,
    squareness_progress,
    top_edge_distance,
)

##
# Reset randomization
##

POSE_XY_RANGE = 0.12
"""Planar offset bound applied to both the arm and the sheet at reset [m]."""

YAW_RANGE = (-math.pi / 2, math.pi / 2)
"""Yaw bound applied to both the arm and the sheet at reset [rad]."""

EPISODE_STEPS = 500
"""Episode length in environment steps, at the 30 Hz control rate."""

##
# Scene geometry
##

SHEET_SIZE = (0.2, 0.2)
SHEET_RESOLUTION = (8, 8)

SLOT_WALL_SIZE = (0.10, 0.02, 0.15)
"""One wall of the slot the sheet stands in: length along x, thickness in y, height [m].

Shorter than the 0.2 m sheet on purpose, so roughly 5 cm of sheet projects above the walls. That
projecting strip is the whole point of the slot: it is a free edge held upright and presented
side-on to the gripper, which is a far easier thing to pinch than a sheet lying flat with nothing
to get a finger under.
"""

SLOT_GAP = 0.01
"""Clear space between the two walls [m]. The sheet stands in this gap."""

SLOT_WALL_OFFSET_Y = 0.5 * (SLOT_GAP + SLOT_WALL_SIZE[1])
"""Distance from the slot centre to each wall's centre [m]."""

SLOT_WALL_SPAWN_CFG = sim_utils.CuboidCfg(
    size=SLOT_WALL_SIZE,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # slippery, so the sheet slides out of the slot when lifted instead of snagging on the walls
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
)

SHEET_SPAWN_POS = (0.5, 0.18, 0.102)
"""Sheet centre at spawn [m].

The sheet stands vertically in the slot with its lower edge just clear of the table, so the centre
of a 0.2 m sheet sits at about half its height. The sheet and slot sit on the +y side and the arm
on the -y side so the two never spawn inside one another -- separating them by construction is
cheaper and more reliable than rejection-sampling resets until they happen not to overlap.
"""

SHEET_SPAWN_ROT = (0.70710678, 0.0, 0.0, 0.70710678)
"""Quaternion that stands the sheet on edge in the slot.

Taken verbatim from the upstream cloth task, which used exactly this value to hold a 0.2 m sheet
upright between two supports. The rectangle mesh is generated lying in a plane, so some rotation is
required to stand it up; copying the value that is known to work avoids depending on a reading of
the quaternion convention.
"""

MANNEQUIN_ARM_RADIUS = 0.04
MANNEQUIN_ARM_LENGTH = 0.28
"""Length of the capsule's cylindrical section [m]; total span is length + 2 * radius."""

MANNEQUIN_ARM_SPAWN_POS = (0.5, -0.15, MANNEQUIN_ARM_RADIUS)
"""Arm centre at spawn [m]. z equals the radius, so the capsule rests on the table."""

TARGET_BAND_WIDTH = 0.10
"""Extent of the red target band along the arm's axis [m]."""

SLOT_CLEAR_HEIGHT = SLOT_WALL_SIZE[2] + 0.02
"""Height the sheet's lowest point must exceed to count as out of the slot [m].

Two centimetres above the walls, so the sheet is unambiguously clear rather than hovering at the
lip where a millimetre of wobble would toggle the phase back and forth. Measured on the lowest
node, which means the whole sheet is above the slot, not merely tilted out of it.
"""

TARGET_BAND_TRAVEL = (MANNEQUIN_ARM_LENGTH - TARGET_BAND_WIDTH) / 2.0
"""Half-range the band centre may slide along the arm [m].

Bounded so the band stays inside the capsule's cylindrical section instead of running onto the
rounded end caps, where a cylindrical sleeve would visibly float off the surface.
"""

MANNEQUIN_ARM_SPAWN_CFG = sim_utils.CapsuleCfg(
    radius=MANNEQUIN_ARM_RADIUS,
    height=MANNEQUIN_ARM_LENGTH,
    axis="X",
    # kinematic and gravity-free: the sheet must not be able to shove the arm around, and it must
    # not fall. Same treatment the upstream task gave its support blocks.
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    # grippier than the table (0.1) so a draped sheet settles instead of sliding straight off
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.72, 0.62)),
)


##
# Scene definition
##


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
                        # without these the sheet passes straight through: Newton only couples the
                        # rigid bodies that are explicitly proxied into the soft solver
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
        # stood on edge in the slot, rather than left flat -- see SHEET_SPAWN_ROT
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

    # the placement target. A RigidObject rather than an AssetBase because it has to be
    # repositioned per environment at reset, which only rigid objects support.
    mannequin_arm: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MannequinArm",
        init_state=RigidObjectCfg.InitialStateCfg(pos=MANNEQUIN_ARM_SPAWN_POS),
        spawn=MANNEQUIN_ARM_SPAWN_CFG,
    )

    # the slot holding the sheet upright: two thin walls a centimetre apart. Repositioned with the
    # sheet at reset, so the sheet always stands squarely between them.
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
        # -- grip strength.
        #
        # Upstream drives only ``panda_finger_joint1`` and leaves ``panda_finger_joint2`` passive
        # with a 1 N effort limit, mimicking the real Franka's single actuator and mimic coupling.
        # A pinch is an action-reaction pair, so that caps the whole grasp at about 1 N however
        # hard the driven finger pushes -- not enough to hold a sheet against gravity, and the
        # cloth slides out from between the pads.
        #
        # Both fingers are driven here instead, and harder. The two entries are kept separate
        # rather than merged into one regex so each finger keeps its own actuator and neither can
        # be left behind by a joint-name change.
        gripper = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint1"],
            # 70 N is the real gripper's limit; the sheet needs a firm pinch more than it needs
            # fidelity to the hardware, and the fingers are position-controlled onto a closed
            # target, so the excess is only ever spent squeezing
            effort_limit_sim=200.0,
            velocity_limit=0.2,
            velocity_limit_sim=2.0,
            # grip force is stiffness times position error, and a sheet only holds the fingers a
            # couple of millimetres apart, so a soft gain converts that small error into almost no
            # force at all
            stiffness=2000.0,
            damping=200.0,
            armature=0.1,
        )
        self.robot.actuators["panda_hand"] = gripper
        self.robot.actuators["panda_finger2_passive"] = gripper.replace(
            joint_names_expr=["panda_finger_joint2"]
        )

    # low-friction table, inherited from the cloth task so the sheet slides rather than sticking.
    # Effective sheet-table friction is sqrt(soft_contact_mu * shape_mu); this sets shape_mu
    # without touching the global soft_contact_mu that governs the gripper's hold on the sheet.
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


##
# MDP settings
##


@configclass
class SheetEventCfg:
    """Reset events: robot to its default configuration, sheet flat on the table."""

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

    # arm and sheet are placed together so they can share one mirror draw, and this runs before
    # the command manager resamples so the goal is derived from the arm's new pose
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
            "slot_cfgs": (SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")),
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
    """Pick the sheet out of the slot. Nothing else is scored.

    Cut back to the single skill the policy has to acquire first: come down over the sheet, take
    the middle of its top edge, and draw it straight up out of the slot. The draping objective and
    its shaping are gone -- they were unreachable while the grasp itself was unsolved, and their
    presence made every diagnostic harder to read.

    Two dense terms shape the approach, two one-shot terms pay for the outcome.
    """

    # -- getting into position
    approach = RewTerm(
        func=top_edge_distance,
        params={
            "resolution": SHEET_RESOLUTION,
            # comparable to the ~0.4 m the gripper starts from, so the pull is felt from the start
            "coarse_std": 0.4,
            # comparable to the grasp radius, for positioning once it has arrived
            "fine_std": 0.05,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=3.0,
    )

    alignment = RewTerm(
        func=grasp_alignment,
        params={
            "resolution": SHEET_RESOLUTION,
            # loose enough that turning the wrist starts paying during the approach. At 0.05 the
            # gate was ~1e-9 half a metre out, so the term was numerically dead and the gripper
            # arrived in an arbitrary posture having never been paid to point down.
            "std": 0.2,
            # All of it goes to squareness. Pointing down sits at 0.90 and is not what the policy
            # still has to learn, so an even split was spending half the term on a solved axis.
            "downward_frac": 0.0,
            "asset_cfg": SceneEntityCfg("deformable"),
            "slot_cfg": SceneEntityCfg("slot_neg_y"),
        },
        # Raised from 3.0, where the term was 0.16% of the episode return and squareness inside it
        # 0.07% -- four orders of magnitude under the value function's own error, so nothing about
        # the wrist could be learned from it. The ceiling is the hover attractor: this term pays
        # per step whether or not the sheet is ever picked up, so at weight w loitering near the
        # edge in a good pose for a whole episode is worth 16.7w against the ~1200 the task pays.
        # 20 leaves that at roughly a quarter, which is a margin; past about 60 it inverts.
        weight=20.0,
    )

    # Pays the *change* in squareness, so the reward lands on the step the wrist turns rather than
    # inside a bonus collected at the end of the episode. Telescopes, so it cannot be farmed.
    square_progress = RewTerm(
        func=squareness_progress,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "slot_cfg": SceneEntityCfg("slot_neg_y"),
            # the whole payout for going from fully skew to fully square, not a per-step rate
            "max_reward": 300.0,
        },
        # the term divides by dt internally, so the magnitude above is the literal payout
        weight=1.0,
    )

    table_clearance = RewTerm(
        func=ee_table_clearance,
        params={"floor": 0.06, "robot_cfg": SceneEntityCfg("robot")},
        # comparable in size to the approach reward, so diving at the table is not worth whatever
        # shortcut it might otherwise buy
        weight=-2.0,
    )

    # Charged for how fast the hand travels: nothing up to 0.2 m/s, then exponential, reaching
    # 9.52 per step at 1 m/s and held flat above that. Off at weight zero unless
    # ``scripts/train.py --slow`` turns it on, which arms it at -4 -- so the figures below are the
    # shape of the curve and the weight is what scales it, to 38.08 per step at 1 m/s.
    #
    # Flat past the reference point rather than left to compound. An uncapped exponential makes a
    # momentary overshoot worth more than the whole task, which is a return the critic cannot fit;
    # levelling off keeps a hard deterrent without handing the value function a spike. Everything
    # the charge has to say is said between 0.2 and 1 m/s anyway.
    #
    # Cartesian rather than joint-space on purpose. A joint-speed charge bills every wrist rotation
    # at the same rate as useful travel, which would pull directly against ``square_progress`` --
    # a term that pays for turning the wrist. Hand speed does not care how the arm reaches a pose,
    # so the two can both be on at once.
    #
    # A charge rather than a cap on ``arm_action.scale``: a cap forbids speed everywhere, while a
    # charge lets the policy spend it where it is worth paying for -- crossing the table quickly
    # and slowing down over the sheet.
    ee_speed = RewTerm(
        func=ee_speed_penalty,
        params={
            "free_speed": 0.2,
            "reference_speed": 1.0,
            "reference_penalty": 9.52,
            # equal to the reference penalty, so the curve levels off exactly where it reaches it
            "max_penalty": 9.52,
            "hand_body_name": "panda_hand",
            "robot_cfg": SceneEntityCfg("robot"),
        },
        # the term returns a positive magnitude and divides by dt, so the weight carries the sign
        # and the magnitudes above are the literal per-step charges
        weight=0.0,
    )

    # Every closure after the first, counted on the commanded bit rather than on the measured
    # finger width. The gripper is one binary channel driven by the sign of a Gaussian sample, so
    # the command toggles several times more often than the joints do; this charges the decision
    # to take a second bite rather than the flicker underneath it. The first closure is free, so a
    # policy that arrives lined up and shuts once pays nothing at all.
    gripper_recommit = RewTerm(
        func=gripper_recommit_penalty,
        params={"penalty": 120.0},
        # the term divides by dt internally, so the magnitude above is the literal charge
        weight=1.0,
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
            # A grasp is judged by whether cloth is actually between the pads, not by the pose the
            # gripper struck: this is how close the nearest sheet node has to be to the point
            # between the fingertips. The pose gates that used to sit here rejected a hand-flown
            # grasp that lifted the sheet clean out of the slot, so they are gone -- ``alignment``
            # still encourages the posture, it just cannot veto a grasp that worked.
            "capture_radius": 0.04,
            "open_threshold": 0.030,
            "closed_threshold": 0.012,
            "first_grasp_bonus": 200.0,
            # Raised from 20 now that aiming is solved. At 200 an early run produced exactly zero
            # closures in twenty minutes -- with a miss near-certain, a charge that size made never
            # closing the gripper the better bet. That reasoning was sound for a policy that could
            # not yet aim; this one grasps on essentially every episode, so a missed closure is a
            # mistake rather than the expected outcome and can be charged like one. ``Events/
            # closures`` collapsing towards zero is the sign that the old failure has returned.
            "bad_closure_penalty": 200.0,
            "premature_release_penalty": 500.0,
            "slot_clear_height": SLOT_CLEAR_HEIGHT,
            "extraction_bonus": 1000.0,
        },
        # the term divides by dt internally, so every magnitude above is the literal reward
        weight=1.0,
    )

    # potential-based, holding-gated slope from the grasp up to the extraction bonus, so getting
    # the sheet out is a climb rather than a cliff to be stumbled over
    lift_progress = RewTerm(
        func=sheet_lift_progress,
        params={
            "lift_span": 0.15,
            # the TOTAL paid for a full 15 cm lift, spread over however long the climb takes
            "max_reward": 500.0,
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=1.0,
    )


@configclass
class SheetTerminationsCfg(TerminationsCfg):
    """The inherited bounds, plus the two ways of botching the pick.

    Both end the episode rather than only costing reward: once the single scored grasp attempt has
    been thrown away, nothing the policy does for the remaining steps can retrieve the episode, and
    the samples are better spent on a fresh one.
    """

    # success: the sheet is out of the slot, which is the whole task now
    extracted = DoneTerm(func=sheet_extracted)

    released_early = DoneTerm(func=released_before_extraction)

    finger_touched_slot = DoneTerm(
        func=finger_in_slot,
        params={
            # half the box enclosing both walls, in the slot's own frame
            "half_extents": (
                0.5 * SLOT_WALL_SIZE[0],
                SLOT_WALL_OFFSET_Y + 0.5 * SLOT_WALL_SIZE[1],
                0.5 * SLOT_WALL_SIZE[2],
            ),
            "margin": 0.005,
            "robot_cfg": SceneEntityCfg("robot"),
            "slot_cfgs": (SceneEntityCfg("slot_neg_y"), SceneEntityCfg("slot_pos_y")),
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
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # offsets in the arm's own frame: pos_x slides the red band along the arm and the goal
        # rides above its centre, pos_z lifts the goal off the surface to where a draped sheet's
        # centre of mass would end up
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
                    # 1 mm proud of a 40 mm arm: enough to beat z-fighting, small enough that the
                    # sleeve reads as paint on the surface rather than a ring around it. The flat
                    # end caps are then buried inside the arm and never show.
                    radius=MANNEQUIN_ARM_RADIUS + 0.001,
                    height=TARGET_BAND_WIDTH,
                    # left on the default Z axis on purpose: the command term rotates the marker
                    # onto the arm rather than relying on this attribute, which the Newton
                    # visualizer does not apply to marker prototypes
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                )
            },
        ),
        # the invisible table is drawn by these markers, tinted green once the goal is reached
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

    arm_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.03)

    # binary gripper: above the threshold commands fully open, below it fully closed, with nothing
    # in between. The grasp reward keys on open-then-close transitions, and a continuous channel
    # lets the policy loiter at intermediate widths that are neither fully open nor fully closed,
    # so the transitions it is being paid for never cleanly occur.
    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        # both fingers, now that both are driven -- a target for only one would leave the other
        # holding its last position and the pinch would be one-sided again. The action stays
        # 1-dimensional: a binary term sends the same open/close decision to every joint it lists.
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
        close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        # zero splits a freshly-initialised Gaussian policy evenly between open and closed
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
        # same 0.1034 m hand-to-grasp-frame offset the reward term uses, so the frame being driven
        # is the frame being scored
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        # both fingers, now that both are driven -- a target for only one would leave the other
        # holding its last position and the pinch would be one-sided again. The action stays
        # 1-dimensional: a binary term sends the same open/close decision to every joint it lists.
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


##
# Environment configuration
##


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
        # the cloth task's physics preset couples the support blocks into the solver
        self.sim.physics = PhysicsCfg()
        # the inherited curriculum -- a gravity ramp starting near 0 g, and a ramp on an
        # action-rate reward this task does not have -- is replaced wholesale by
        # ``SheetCurriculumCfg`` rather than switched off term by term. Full gravity applies from
        # the first frame so the sheet rests on the table as it looks like it should.
        # 500 environment steps. Derived from the control rate rather than hard-coded in seconds so
        # it stays 500 steps if the decimation or physics timestep is ever retuned. The half-step
        # slack matters: max_episode_length is a ceil(), and the exact product lands a hair above
        # 500 in floating point, which rounds up to 501.
        self.episode_length_s = (EPISODE_STEPS - 0.5) * self.decimation * self.sim.dt
        # the goal must not be redrawn mid-episode: resampling re-rolls where the red band sits
        # along the arm, so with the inherited 5 s range the target would teleport twice during a
        # 500-step episode. Tying it to the episode length means it only ever fires on reset.
        self.commands.deformable_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # Nine identified landmarks instead of twenty randomly resampled nodes. The inherited term
        # picks a fresh random subset every reset, so the meaning of each slot changes from episode
        # to episode and the policy cannot learn what any of them refers to; these nine are always
        # the same physical points. Also cuts the term from 60 values to 27.
        self.observations.policy.deformable_sampled_points = ObsTerm(
            func=sheet_key_points,
            params={"resolution": SHEET_RESOLUTION, "asset_cfg": SceneEntityCfg("deformable")},
        )
