"""Twist (IK-Rel) lift configurations for the DAgger student and its PPO continuation."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab_tasks.contrib.lift.config.franka import ik_rel_env_cfg

from .pick_place_mdp import (
    PickPlacePoseCommandCfg,
    cube_placed,
    cube_released,
    ee_speed_approach_before_pick,
    hand_yaw_rate_after_pick,
    object_lifted_or_picked,
    pick_goal_tracking,
    place_completion_bonus,
    place_height_tracking,
    place_target_platform,
    place_tracking,
    release_shaping,
)
from .platform_mdp import (
    ee_speed_above,
    ee_speed_near_object,
    object_goal_distance_above_support,
    object_is_lifted_above_support,
    platform_cfg,
    platform_event,
)

MOTION_COST = -0.2
"""Weight of the ``action_rate`` and ``joint_vel`` charges: twice the joint teacher's -0.1."""


@configclass
class FrankaCubeLiftTwistEnvCfg(ik_rel_env_cfg.FrankaCubeLiftEnvCfg):
    """Isaac Lab's IK-Rel lift task with the motion costs at ``MOTION_COST``.

    The curriculum terms are set to the same value: they assign their weight on every step once past
    10000 env steps, and at the inherited -0.1 they would undo the increase a few iterations in.
    """

    def __post_init__(self):
        super().__post_init__()
        for term in ("action_rate", "joint_vel"):
            getattr(self.rewards, term).weight = MOTION_COST
            getattr(self.curriculum, term).params["weight"] = MOTION_COST


@configclass
class FrankaCubeLiftTwistTeacherGainsEnvCfg(FrankaCubeLiftTwistEnvCfg):
    """The twist task with ``FRANKA_PANDA_CFG`` (stiffness 80, gravity on) instead of the high-PD arm.

    The joint teacher learned its targets on this plant: they lead the arm and carry a gravity offset.
    On the stiff, weightless arm of the IK-Rel task the same targets are reached at once and the offset
    becomes motion, so a converted teacher label means something different there.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


PLATFORM_HEIGHTS = (0.015, 0.15)
"""Range the platform top is drawn from at every reset [m]."""

PLATFORM_SIZE = 0.10
"""Side of the square platform [m]; the cube is 4.8 cm."""

PLACE_BONUS = 300.0
"""One-off return for completing the placement [reward]. A whole pick-and-place episode returns about 400."""


@configclass
class FrankaCubeLiftTwistPlatformEnvCfg(FrankaCubeLiftTwistEnvCfg):
    """The twist task with the cube on a platform of random height, and a slower hand.

    - The platform top is drawn from ``PLATFORM_HEIGHTS`` at every reset, under wherever the cube's
      own reset put it (see ``platform_mdp``).
    - Lift and goal terms measure height above the platform, so a cube merely standing on a tall
      platform is not paid as lifted.
    - ``ee_speed`` charges hand speed above 0.2 m/s everywhere.
    - ``approach_speed`` charges hand speed on the way in, growing as the hand nears the cube with
      the reach kernel, until the cube is lifted.

    Observations and actions are those of ``FrankaCubeLiftTwistEnvCfg``, so its checkpoints resume
    here unchanged. The penalty weights are Hydra-tunable, e.g. ``env.rewards.approach_speed.weight=-80``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.platform = platform_cfg(PLATFORM_SIZE, self.scene.object.init_state.pos[:2])
        # defined last, so it runs after reset_object_position has placed the cube on the table
        self.events.place_cube_on_platform = platform_event(PLATFORM_HEIGHTS)
        self.rewards.lifting_object.func = object_is_lifted_above_support
        self.rewards.object_goal_tracking.func = object_goal_distance_above_support
        self.rewards.object_goal_tracking_fine_grained.func = object_goal_distance_above_support
        # quadratic in the excess: 0.5 m/s costs 0.9 a second, 1 m/s costs 6.4 (lifting pays 15)
        self.rewards.ee_speed = RewTerm(func=ee_speed_above, weight=-10.0, params={"free_speed": 0.2})
        # 0.5 m/s at 1 cm from the cube costs about 9 a second, 0.1 m/s about 0.4 (reaching pays up to 1)
        self.rewards.approach_speed = RewTerm(
            func=ee_speed_near_object, weight=-40.0, params={"std": 0.1, "minimal_height": 0.04}
        )


@configclass
class FrankaCubeLiftTwistPickPlaceEnvCfg(FrankaCubeLiftTwistPlatformEnvCfg):
    """The platform task, then a second phase: set the cube down on a second platform (see ``pick_place_mdp``).

    - Phase 1 is unchanged: lift the cube off its platform and carry it to the goal in the air. Within
      10 cm of it (the success radius) the goal switches to the cube resting on the second platform.
    - The second platform is 10 cm square too, with its own height drawn from ``PLATFORM_HEIGHTS``, on
      the far side of the cube in y (x 0.4-0.6 m, |y| 0.15-0.30 m).
    - Phase-1 terms pay their maximum once the goal has switched. Phase 2 adds ``place_tracking``
      (coarse std 0.15 m and fine 0.05 m), ``placed`` (cube resting within 2.5 cm of the placement pose) and ``released``
      (placed, with the gripper open).
    - ``approach_speed`` is the phase-1 approach charge only (see ``ee_speed_approach_before_pick``).
    - ``place_height`` pays for bringing the cube down to the placement height, and ``release_shaping``
      for raising the gripper action toward open while the cube is at the placement pose.
    - ``yaw_rate`` charges the hand turning about the vertical once the goal has switched.
    - ``place_bonus`` adds a one-off ``PLACE_BONUS`` to the return the first time the cube is placed and
      released in an episode.
    - Episodes are 10 s instead of 5, for both phases.

    Observations and actions are unchanged, so platform-task checkpoints resume here.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        old = self.commands.object_pose
        self.commands.object_pose = PickPlacePoseCommandCfg(
            asset_name=old.asset_name, body_name=old.body_name, debug_vis=old.debug_vis, ranges=old.ranges
        )
        self.scene.place_platform = platform_cfg(PLATFORM_SIZE, (0.5, 0.25), name="PlacePlatform", color=(0.3, 0.7, 0.35))
        # after place_cube_on_platform, which the parent defined, so the cube's side is known
        self.events.place_target_platform = EventTerm(
            func=place_target_platform,
            mode="reset",
            params={"height_range": PLATFORM_HEIGHTS, "x_range": (0.4, 0.6), "y_range": (0.15, 0.30)},
        )
        self.rewards.lifting_object.func = object_lifted_or_picked
        self.rewards.lifting_object.params["command_name"] = "object_pose"
        self.rewards.object_goal_tracking.func = pick_goal_tracking
        self.rewards.object_goal_tracking_fine_grained.func = pick_goal_tracking
        self.rewards.approach_speed.func = ee_speed_approach_before_pick
        self.rewards.approach_speed.params["command_name"] = "object_pose"
        # the same weights as the phase-1 goal terms, so the switch keeps the scale of the task
        self.rewards.place_tracking = RewTerm(
            # std 0.15, not the phase-1 goal's 0.3: closing the last 10 cm must be clearly worth it
            func=place_tracking, weight=16.0, params={"std": 0.15, "command_name": "object_pose"}
        )
        self.rewards.place_tracking_fine_grained = RewTerm(
            func=place_tracking, weight=5.0, params={"std": 0.05, "command_name": "object_pose"}
        )
        # 10 cm too high pays 0.3 a second, 5 cm 1.9, 2 cm 4.9, on the height 8
        self.rewards.place_height = RewTerm(
            func=place_height_tracking, weight=8.0, params={"std": 0.05, "command_name": "object_pose"}
        )
        # at 1 cm from the placement pose: raw gripper -2.7 pays 3.7 a second, -1.5 pays 5.1, 0 (open) 6.8
        self.rewards.release_shaping = RewTerm(
            func=release_shaping, weight=10.0, params={"std": 0.03, "command_name": "object_pose"}
        )
        self.rewards.placed = RewTerm(func=cube_placed, weight=10.0, params={"command_name": "object_pose"})
        self.rewards.released = RewTerm(func=cube_released, weight=10.0, params={"command_name": "object_pose"})
        # 1 rad/s of yaw costs 2 a second, 3 rad/s costs 18 (phase 2 pays 36-77 a second)
        self.rewards.yaw_rate = RewTerm(
            func=hand_yaw_rate_after_pick, weight=-2.0, params={"command_name": "object_pose"}
        )
        self.rewards.place_bonus = RewTerm(
            func=place_completion_bonus, weight=PLACE_BONUS, params={"command_name": "object_pose"}
        )
