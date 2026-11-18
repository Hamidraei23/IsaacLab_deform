"""Extract the existing sheet, then cover a matching platform 2 cm above the table."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as common_mdp
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .mdp.platform import PlatformPoseCommandCfg, platform_placement, reset_platform_and_sheet
from .mdp.rewards import ee_motion_while_closing, sheet_extracted
from .sheet_rl_env_cfg import (
    SHEET_RESOLUTION, SHEET_SIZE, SLOT_WALL_OFFSET_Y, SLOT_WALL_SIZE,
    SheetRlEnvCfg, SheetSceneCfg, SheetScenePresetCfg,
)

PLATFORM_HEIGHT = 0.02


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
        )
        previous = self.commands.deformable_pose
        self.commands.deformable_pose = PlatformPoseCommandCfg(
            asset_name="robot", object_name="deformable", top_offset=PLATFORM_HEIGHT / 2,
            resampling_time_range=previous.resampling_time_range,
            debug_vis=False, ranges=previous.ranges,
            success_vis_asset_name=previous.success_vis_asset_name,
            success_visualizer_cfg=previous.success_visualizer_cfg,
        )
        # Preserve phase one's exact rewards, weights, action space and observations.
        keep = {"approach", "alignment", "square_progress", "table_clearance", "ee_speed",
                "gripper_recommit", "grasp_stage", "lift_progress"}
        for name, term in vars(self.rewards).items():
            if isinstance(term, RewTerm) and name not in keep:
                setattr(self.rewards, name, None)
        self.rewards.placement = RewTerm(
            func=platform_placement, weight=1.0,
            params={"resolution": SHEET_RESOLUTION, "platform_size": SHEET_SIZE,
                    "top_offset": PLATFORM_HEIGHT / 2,
                    "coverage_threshold": 0.90, "settle_time": 2.0,
                    "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        self.rewards.joint_velocity = RewTerm(
            func=common_mdp.joint_vel_l2, weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
        )
        self.rewards.joint_acceleration = RewTerm(
            func=common_mdp.joint_acc_l2, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-7]")},
        )
        self.rewards.action_rate = RewTerm(func=common_mdp.action_rate_l2, weight=-1.0)
        self.rewards.ee_motion_during_closure = RewTerm(
            func=ee_motion_while_closing,
            # L1 equivalent speed and a literal per-step scale: 0.1 m/s translation, or 1 rad/s
            # rotation at the 10 cm angular scale, costs 50 reward for every closing step.
            weight=-500.0,
            params={
                "closed_threshold": 0.012,
                "linear_scale": 1.0,
                "angular_scale": 0.1,
                "hand_body_name": "panda_hand",
                "action_term_name": "gripper_action",
                "robot_cfg": SceneEntityCfg("robot"),
                "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
            },
        )
        # Release height and capsule collision heuristics do not apply to the platform.
        self.terminations.released_high = None


@configclass
class SheetRlPlatformGraspEnvCfg(SheetRlPlatformEnvCfg):
    """Extraction-only task with strong staged rewards for discovering the complete grasp."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.placement = None
        self.terminations.draped = None
        self.terminations.extracted = DoneTerm(func=sheet_extracted)

        # The grasp-only curriculum needs each prerequisite to be visible before the sparse
        # extraction event is ever discovered. Approach and first grasp are ten times their shared
        # placement-task values. Extraction is an explicit 5,000-point outcome, and the 5,000-point
        # lift budget uses a normalized exponential potential: later upward progress is worth more,
        # but the increments telescope and therefore cannot be farmed by hovering or bobbing.
        self.rewards.approach.weight = 30.0
        self.rewards.grasp_stage.params["first_grasp_bonus"] = 2000.0
        self.rewards.grasp_stage.params["extraction_bonus"] = 5000.0
        self.rewards.lift_progress.params["max_reward"] = 5000.0
        self.rewards.lift_progress.params["exponential_sharpness"] = 2.0
