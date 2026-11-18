"""Extract the existing sheet, then cover a matching platform 2 cm above the table."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm, ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from .mdp.platform import (
    FIXED_SHEET_POSITION, FIXED_SHEET_YAW, PlatformPoseCommandCfg, platform_opened_high, platform_placement,
    reset_platform_and_sheet,
)
from .mdp.terminations import deformable_outside_layout_bounds
from .platform_teleop import PlatformTeleopCfg
from .mdp.rewards import sheet_extracted
from .mdp.grasp_control import (
    FilteredBinaryGripperActionCfg, SmoothRelativeJointPositionActionCfg, applied_policy_actions,
)
from .mdp.grasp_progress import (
    grasp_lift_milestones, grasp_progress, grasp_motion_cost, gripper_closure_penalty,
)
from .sheet_rl_env_cfg import (
    SHEET_RESOLUTION, SHEET_SIZE, SLOT_CLEAR_HEIGHT, SLOT_WALL_OFFSET_Y, SLOT_WALL_SIZE,
    SheetRlEnvCfg, SheetSceneCfg, SheetScenePresetCfg,
)

PLATFORM_HEIGHT = 0.02
# Control steps per episode for both platform tasks (33.3 s at 30 Hz); the capsule/hand tasks keep 500.
PLATFORM_EPISODE_STEPS = 1000
# Fingers pinching the cloth stop at about 2 mm (recorded minimum 1.7 mm); shut on nothing, 0.
MIN_HOLD_WIDTH = 0.001


@configclass
class SheetPlatformSceneCfg(SheetSceneCfg):
    # Retain the internal entity/path so the existing rigid-to-cloth physics proxy
    # includes the collider. The scene contains a cuboid, with no capsule or hand.
    mannequin_arm: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MannequinArm",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.15, PLATFORM_HEIGHT / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(*SHEET_SIZE, PLATFORM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
    )


@configclass
class SheetPlatformScenePresetCfg(SheetScenePresetCfg):
    newton_mjwarp_vbd_proxy: SheetPlatformSceneCfg = SheetPlatformSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class SheetRlPlatformEnvCfg(SheetRlEnvCfg):
    scene: SheetPlatformScenePresetCfg = SheetPlatformScenePresetCfg()
    # Keyboard teleop of the platform while the policy runs (W/S x, A/D y, Q/E yaw); off unless
    # env.platform_teleop.enabled=true (scripts/play.py --platform_teleop). See platform_teleop.py.
    platform_teleop: PlatformTeleopCfg = PlatformTeleopCfg()

    def __post_init__(self):
        super().__post_init__()
        self.push.enabled = False
        self.push.on_extraction = False
        self.events.randomize_arm_radius = None
        self.events.reset_layout.func = reset_platform_and_sheet
        self.events.reset_layout.params.update(
            platform_size=SHEET_SIZE,
            # Include the sheet's overhang as well as both slot walls.
            slot_radius=math.hypot(max(SHEET_SIZE) / 2, SLOT_WALL_OFFSET_Y + SLOT_WALL_SIZE[1] / 2),
            # The function's own defaults, listed so a run can move the sheet and its slot from the command line
            # (env.events.reset_layout.params.fixed_sheet_position=[x,y,z]; the jitter is added on top).
            fixed_sheet_position=FIXED_SHEET_POSITION,
            fixed_sheet_yaw=FIXED_SHEET_YAW,
        )
        # The workspace box follows the layout: the same box as before when the sheet is at its default position,
        # shifted with it when a run moves the sheet (see deformable_outside_layout_bounds).
        if getattr(self.terminations, "deformable_out_of_bounds", None) is not None:
            self.terminations.deformable_out_of_bounds.func = deformable_outside_layout_bounds
        # Half-step slack as in the base config: max_episode_length is a ceil() of this.
        self.episode_length_s = (PLATFORM_EPISODE_STEPS - 0.5) * self.decimation * self.sim.dt
        previous = self.commands.deformable_pose
        self.commands.deformable_pose = PlatformPoseCommandCfg(
            asset_name="robot", object_name="deformable", top_offset=PLATFORM_HEIGHT / 2,
            # Only resampled at reset: the platform never moves mid-episode.
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=False, ranges=previous.ranges,
            success_vis_asset_name=previous.success_vis_asset_name,
            success_visualizer_cfg=previous.success_visualizer_cfg,
        )
        # Same seven arm joints and binary gripper, with an observable filtered command.
        self.actions.joint.arm_action = SmoothRelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.02,
        )
        gripper = self.actions.joint.gripper_action
        self.actions.joint.gripper_action = FilteredBinaryGripperActionCfg(
            asset_name=gripper.asset_name, joint_names=gripper.joint_names,
            open_command_expr=gripper.open_command_expr, close_command_expr=gripper.close_command_expr,
            threshold=gripper.threshold, positive_threshold=gripper.positive_threshold,
        )
        self.actions.default = self.actions.joint
        self.observations.policy.actions = ObsTerm(func=applied_policy_actions)
        # Keep the established grasp/extraction detector, replace all first-phase shaping.
        keep = {"table_clearance", "ee_speed", "grasp_stage"}
        for name, term in vars(self.rewards).items():
            if isinstance(term, RewTerm) and name not in keep:
                setattr(self.rewards, name, None)
        # The grasp state reward is at most 0.5 per step, so holding forever is worth 50 at gamma
        # 0.99. Extraction pays 200 on top of up to 100 of lift shaping.
        self.rewards.grasp_stage.params.update(
            first_grasp_bonus=10.0, extraction_bonus=200.0, min_hold_width=MIN_HOLD_WIDTH,
            bad_closure_penalty=0.5, premature_release_penalty=5.0, phase_one_only=True,
        )
        # Must run after grasp_stage: its detector publishes this step's grasp geometry.
        self.rewards.grasp_progress = RewTerm(
            func=grasp_progress, weight=1.0,
            # gamma must match the grasp runner's for the lift shaping to be potential-based.
            params={"clear_height": SLOT_CLEAR_HEIGHT, "lift_scale": 100.0, "gamma": 0.99,
                    "min_hold_width": MIN_HOLD_WIDTH, "edge_travel": 0.20, "edge_share": 0.5})
        # Milestones on top of the above: +400 first real grasp, lift pay 500 for the first cm of
        # top-edge rise and 5000 by extraction, -0.5 per step holding still, -400 per release
        # before lifting 1 cm. Must also run after grasp_stage.
        self.rewards.lift_milestones = RewTerm(
            func=grasp_lift_milestones, weight=1.0,
            params={"grasp_bonus": 400.0, "first_pay": 500.0, "first_height": 0.01, "total_pay": 5000.0,
                    "edge_travel": 0.20, "still_penalty": 0.5, "still_rise": 0.0005,
                    "release_penalty": 400.0, "release_lift": 0.01, "release_steps": 3})
        # Every commanded gripper closure after the first costs 50 (raw policy action, before the
        # gripper filter; before extraction only).
        self.rewards.gripper_closures = RewTerm(
            func=gripper_closure_penalty, weight=1.0, params={"penalty": 50.0, "free_closures": 1})
        # Preserve extraction shaping; phase two has a smaller, capped motion cost.
        self.rewards.table_clearance.params["phase_one_only"] = True
        self.rewards.ee_speed.params["phase_one_only"] = True
        self.rewards.motion_cost = RewTerm(
            func=grasp_motion_cost, weight=-1.0,
            params={"min_scale": 0.05, "phase_one_only": False,
                    "phase_two_scale": 0.2, "phase_two_max_cost": 5.0, "phase_two_speed_limit": 0.15})
        self.rewards.placement = RewTerm(
            func=platform_placement, weight=1.0,
            params={"resolution": SHEET_RESOLUTION, "platform_size": SHEET_SIZE,
                    "top_offset": PLATFORM_HEIGHT / 2,
                    "coverage_threshold": 0.50, "settle_time": 1.0,
                    # Sheet-centre horizontal distance: 0 at 30 cm, -5/step by 40 cm.
                    "distance_limit": 0.30, "distance_penalty": 5.0, "distance_penalty_span": 0.10,
                    # Finite progress budgets, in literal reward units per episode.
                    "approach_bonus": 500.0, "lower_bonus": 500.0,
                    "coverage_bonus": 1500.0, "opening_bonus": 100.0,
                    # Smoothly prefer releases close to the platform and close to its surface.
                    "release_bonus": 1500.0, "release_height": 0.03, "release_height_std": 0.08,
                    # Only unheld cloth with physically open fingers gets ongoing coverage pay.
                    "released_coverage_bonus": 1000.0, "released_coverage_step": 5.0,
                    # Phase 3: fingertips 20 cm above platform top-centre, within 3 cm,
                    # with low translation/rotation for the same continuous success window.
                    "retreat_height": 0.20, "retreat_bonus": 1000.0, "retreat_tolerance": 0.03,
                    "retreat_speed_threshold": 0.03, "retreat_angular_threshold": 0.20,
                    "retreat_distance_limit": 0.30, "retreat_distance_penalty": 5.0,
                    "retreat_motion_penalty": 2.0,
                    "success_bonus": 5000.0,
                    "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        self.terminations.released_high = DoneTerm(
            func=platform_opened_high, params={"max_height": 0.10, "top_offset": PLATFORM_HEIGHT / 2})
        self.terminations.released_early = None



@configclass
class SheetRlPlatformGraspEnvCfg(SheetRlPlatformEnvCfg):
    """Extraction-only task sharing the placement task's progress and motion costs."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.placement = None
        self.terminations.draped = None
        self.terminations.released_high = None  # Placement only.
        self.terminations.extracted = DoneTerm(func=sheet_extracted)
