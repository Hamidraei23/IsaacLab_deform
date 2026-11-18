"""The sheet platform twist task with a solid plate instead of the cloth (see ``rigid_mdp``).

Everything else is ``SheetRlPlatformTwistEnvCfg``: 50 Hz twist control of the stiff weightless arm, the
same slot at the same fixed pose, the platform of random height, and the same 59 observations (the
plate's nine landmarks in place of the cloth's, at the same indices). Physics is Newton's MuJoCo-Warp
solver on its own: with no cloth there is nothing for the VBD solver or the rigid-to-soft coupler to do.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..sheet_rl.mdp.platform import FIXED_SHEET_POSITION, FIXED_SHEET_YAW
from ..sheet_rl.sheet_rl_env_cfg import SHEET_SIZE, SLOT_CLEAR_HEIGHT
import math
from .rigid_mdp import (
    PLATE_SIZE,
    PLATE_THICKNESS,
    PlatformTopCommandCfg,
    plate_key_points,
    plate_out_of_bounds,
    reset_plate_and_platform,
    rigid_coverage,
    rigid_grasp_detector,
)
from .sheet_twist_env_cfg import SheetRlPlatformTwistEnvCfg, SheetTwistSceneCfg, SheetTwistScenePresetCfg
from .sheet_twist_mdp import PLATFORM_DEPTH
from .sheet_twist_reward import apply_staged_reward

FINGERS = SceneEntityCfg("robot", joint_names="panda_finger_joint.*")
"""Passed explicitly in params: the manager only resolves joint names of configs it is given."""


@configclass
class SheetTwistRigidSceneCfg(SheetTwistSceneCfg):
    """The cloth twist scene with a plate instead of the cloth.

    The table's collision is switched off. The platform is a tall kinematic box sunk through the table
    (``sheet_twist_mdp``), and with every body in one MuJoCo-Warp model that overlap made a plate resting
    on the platform bounce off it (the cloth task never collided the two: its table was not in the rigid
    solver). A plate on a box standing on the table, or on the table itself, settles to 0.0 m/s. In the
    table's place under the slot is ``slot_floor``, a thin static box the plate stands on, small enough
    never to reach a platform. A plate that misses the platform falls through and ends the episode.
    """

    deformable = None
    slot_floor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SlotFloor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(FIXED_SHEET_POSITION[0], FIXED_SHEET_POSITION[1], -0.01),
            rot=(0.0, 0.0, math.sin(FIXED_SHEET_YAW / 2), math.cos(FIXED_SHEET_YAW / 2)),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.22, 0.04, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        ),
    )
    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CuboidCfg(
            size=(*PLATE_SIZE, PLATE_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.18, 0.102)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.table = self.table.replace(
            spawn=self.table.spawn.replace(collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False))
        )


@configclass
class SheetTwistRigidScenePresetCfg(SheetTwistScenePresetCfg):
    newton_mjwarp_vbd_proxy: SheetTwistRigidSceneCfg = SheetTwistRigidSceneCfg(
        num_envs=2048, env_spacing=2.0, replicate_physics=True
    )
    default = newton_mjwarp_vbd_proxy


@configclass
class SheetRlPlatformTwistRigidEnvCfg(SheetRlPlatformTwistEnvCfg):
    scene: SheetTwistRigidScenePresetCfg = SheetTwistRigidScenePresetCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(cone="elliptic", ls_iterations=20, integrator="implicitfast"),
            num_substeps=2,
        )
        # the cloth-specific parts, each replaced by its plate counterpart
        self.observations.policy.deformable_sampled_points = ObsTerm(func=plate_key_points)
        layout = self.events.reset_layout.params
        self.events.reset_layout = EventTerm(
            func=reset_plate_and_platform,
            mode="reset",
            params={
                "position_range": layout["position_range"],
                "yaw_range": layout["yaw_range"],
                "platform_size": SHEET_SIZE,
                "slot_radius": layout["slot_radius"],
            },
        )
        self.commands.deformable_pose = PlatformTopCommandCfg(
            resampling_time_range=(self.episode_length_s, self.episode_length_s), top_offset=PLATFORM_DEPTH / 2
        )
        self.rewards.grasp_stage = RewTerm(
            func=rigid_grasp_detector, weight=1.0, params={"clear_height": SLOT_CLEAR_HEIGHT, "finger_cfg": FINGERS}
        )
        self.rewards.coverage = RewTerm(
            func=rigid_coverage, weight=1.0,
            params={"platform_size": SHEET_SIZE, "top_offset": PLATFORM_DEPTH / 2, "finger_cfg": FINGERS},
        )
        self.terminations.deformable_out_of_bounds = None
        self.terminations.plate_dropped = DoneTerm(func=plate_out_of_bounds)
        apply_staged_reward(self)  # again: the detector and coverage terms were just replaced
