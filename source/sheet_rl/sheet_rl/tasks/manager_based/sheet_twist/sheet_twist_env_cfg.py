"""The sheet platform task on the cube twist teacher's terms: 50 Hz twist control of a stiff weightless arm.

Built for distilling ``Lift-Cube-Franka-Twist-PickPlace-v0`` policies into a sheet student. The
teacher's commands only mean the same motion here if the arm behaves as the cube task's does, so:

- **50 Hz control** (physics 1/150 s, decimation 3), as the cube task's 0.01 s x 2. The sheet task ran
  at 30 Hz (1/120 s x 4); the finer physics step here keeps the cloth solver at least as stable.
- **The cube task's arm drives**: stiffness 400, damping 80, effort 87/12 N m, no joint velocity cap,
  armature 1e-3, and the arm's weight cancelled by MuJoCo gravity compensation (the cloth keeps gravity). Arm link masses are scaled by
  U(0.8, 1.2) at startup, so the student does not learn one exact inertia.
- **The cube task's twist action**: a 6D delta pose of the tool point (0.107 m on ``panda_hand``) in
  the robot base frame, scale 0.5, damped least-squares IK. The sheet task's filtered binary gripper
  is kept.
- **A platform of random height**: its top is drawn from ``PLATFORM_HEIGHTS`` at every reset (see
  ``sheet_twist_mdp``). The goal observation is still the platform's top-centre pose.
- **Reward**: the staged pick-and-place reward of ``sheet_twist_reward`` (reach, lift to extraction,
  place tracking, coverage once released, a completion bonus, motion charges), on top of the extraction
  detector (``grasp_stage``, the source of the hold and extraction flags) and ``coverage``. The sheet
  task's placement reward and its 2 cm-platform release ceiling are removed.

Observations are the sheet task's: joints, the nine sheet landmarks, the goal, and the applied
actions (now 6 + 1 = 7 values instead of 7 + 1).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyPropertiesCfg

from ..sheet_rl.sheet_rl_env_cfg import SHEET_RESOLUTION, SHEET_SIZE
from ..sheet_rl.sheet_rl_platform_env_cfg import (
    PLATFORM_HEIGHT,
    SheetPlatformSceneCfg,
    SheetPlatformScenePresetCfg,
    SheetRlPlatformEnvCfg,
)
from .sheet_twist_mdp import PLATFORM_DEPTH, platform_coverage, set_platform_height
from .sheet_twist_reward import apply_staged_reward

PLATFORM_HEIGHTS = (0.015, 0.15)
"""Range the platform top is drawn from at every reset [m], as in the cube pick-and-place task."""

GRAVITY_COMPENSATION = 1.0
"""Share of each arm link's weight cancelled by the solver (``mjc:gravcomp``): 1.0 is weightless."""

EPISODE_STEPS = 900
"""Control steps per episode: 18 s at 50 Hz, room for a slow extraction and a slow lay."""


@configclass
class SheetTwistSceneCfg(SheetPlatformSceneCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.mannequin_arm = self.mannequin_arm.replace(
            init_state=self.mannequin_arm.init_state.replace(pos=(0.5, -0.15, PLATFORM_HEIGHT - PLATFORM_DEPTH / 2)),
            spawn=self.mannequin_arm.spawn.replace(size=(*SHEET_SIZE, PLATFORM_DEPTH)),
        )
        # Weightless arm. ``disable_gravity`` is not honoured by Newton's MuJoCo solver; its body-level
        # gravity compensation is (``mjc:gravcomp``, 1.0 = full). It replaces the asset's rigid-body
        # fragment, whose settings (depenetration velocity, gravity on) are PhysX-only or unwanted.
        self.robot = self.robot.replace(
            spawn=self.robot.spawn.replace(rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=GRAVITY_COMPENSATION))
        )
        self.robot.actuators["panda_arm"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            effort_limit_sim={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
            stiffness=400.0,
            damping=80.0,
            armature=1e-3,
        )


@configclass
class SheetTwistScenePresetCfg(SheetPlatformScenePresetCfg):
    newton_mjwarp_vbd_proxy: SheetTwistSceneCfg = SheetTwistSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    default = newton_mjwarp_vbd_proxy


def apply_twist_plant(cfg, episode_steps: int = EPISODE_STEPS) -> None:
    """The twist students' plant, applied to any cfg built on the platform scene: 50 Hz control, the
    relative-pose IK action at the cube tasks' scale, the random-height platform event and the arm-mass
    randomization. Rewards, observations and terminations are left untouched."""
    cfg.sim.dt = 1.0 / 150
    cfg.decimation = 3
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = (episode_steps - 0.5) * cfg.decimation * cfg.sim.dt
    cfg.commands.deformable_pose.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s)
    cfg.commands.deformable_pose.top_offset = PLATFORM_DEPTH / 2
    cfg.actions.joint.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    cfg.actions.default = cfg.actions.joint
    # after reset_layout, which places the platform at its table height
    cfg.events.set_platform_height = EventTerm(
        func=set_platform_height, mode="reset", params={"height_range": PLATFORM_HEIGHTS}
    )
    cfg.events.randomize_arm_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_link.*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )


@configclass
class SheetRlPlatformTwistEnvCfg(SheetRlPlatformEnvCfg):
    scene: SheetTwistScenePresetCfg = SheetTwistScenePresetCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_twist_plant(self)

        keep = {"grasp_stage"}
        for name, term in vars(self.rewards).items():
            if isinstance(term, RewTerm) and name not in keep:
                setattr(self.rewards, name, None)
        self.rewards.coverage = RewTerm(
            func=platform_coverage,
            weight=1.0,
            # finger_cfg passed explicitly: the manager only resolves joint names of configs in params
            params={"resolution": SHEET_RESOLUTION, "platform_size": SHEET_SIZE, "top_offset": PLATFORM_DEPTH / 2,
                    "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
        self.terminations.released_high = None
        self.terminations.draped = None
        apply_staged_reward(self)


@configclass
class SheetRlPlatformTwistOriginalRewardEnvCfg(SheetRlPlatformEnvCfg):
    """The ORIGINAL platform task's reward and layout, on the twist students' plant.

    An evaluation bridge: the three-phase reward of ``SheetRlPlatformEnvCfg`` (grasp_progress,
    lift_milestones, gripper_closures, motion_cost, the budgeted placement/retreat term and its
    released-high termination) is kept in full, while the plant is the one the twist students were
    trained on -- 50 Hz, weightless 400/80 arm, relative-pose IK action, filtered gripper. The
    platform is fixed at the original 2 cm top; episodes are 1500 steps (30 s, near the original's
    33 s). Terms that measure from the platform root get the tall sunk box's half-depth as their
    top offset. A zero-weight-adjacent ``coverage`` term is added only so the twist evaluation
    tooling (which reads it) works here unchanged.
    """

    scene: SheetTwistScenePresetCfg = SheetTwistScenePresetCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_twist_plant(self, episode_steps=1500)
        # the ORIGINAL 2 cm slab platform, not the twist tasks' sunk 20 cm box. The original terms use
        # one ``top_offset`` both as root-to-top and as root-above-table -- true for the slab (both
        # 0.01), 18 cm wrong for the sunk box, whose geometry made every release read as above the
        # 10 cm ceiling and terminate as released_high.
        self.scene.mannequin_arm = RigidObjectCfg(
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
        self.events.set_platform_height = None
        self.commands.deformable_pose.top_offset = PLATFORM_HEIGHT / 2
        # the original motion_cost reads the smooth joint action's filter internals and crashes the
        # process against the IK action; the bridge reports the task reward without motion charges
        self.rewards.motion_cost = None
        # the original leaves released_high's finger_cfg as an unresolved default, so its "fingers
        # physically open" check averages all nine joints (arm included) and fires on arm posture:
        # every carry died ~0.3 s after extraction. Passing it in params makes the manager resolve it.
        self.terminations.released_high.params["finger_cfg"] = SceneEntityCfg(
            "robot", joint_names="panda_finger_joint.*"
        )
        self.rewards.coverage = RewTerm(
            func=platform_coverage,
            weight=1e-3,
            params={"resolution": SHEET_RESOLUTION, "platform_size": SHEET_SIZE, "top_offset": PLATFORM_HEIGHT / 2,
                    "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")},
        )
