"""The sheet platform task, whole, under the twist student's control. One env, one class.

Everything is ``SheetRlPlatformEnvCfg`` -- its layout randomization, its goal command, every reward
term (grasp_progress, lift_milestones, grasp_stage, gripper_closures, table_clearance, ee_speed,
the budgeted placement/retreat term) and every termination (released_high, draped,
finger_touched_slot, the bounds) -- except for the control and the platform, which are the twist
tasks':

- **Scene**: ``SheetTwistSceneCfg`` -- the sunk ``PLATFORM_DEPTH`` platform box in place of the
  original 2 cm slab, and the arm the student was trained on: stiffness 400 / damping 80 drives,
  weight cancelled by MuJoCo gravity compensation (the cloth keeps gravity), link masses scaled by
  U(0.8, 1.2) at startup as in training.
- **Platform height**: drawn from ``PLATFORM_HEIGHTS`` = U(0.02, 0.14) at every reset (the layout
  event places the box, ``set_platform_height`` then sinks it). Every term that measures from the
  platform root gets the box's half-depth as its top offset; the placement reward's release grading
  and the released_high ceiling are re-anchored to the (moving) platform top.
- **Control**: 50 Hz (physics 1/150 s, decimation 3). Episodes are 1500 steps = 30 s, matching the
  original's 33 s of wall time rather than its step count.
- **Action**: the student's -- a 6D relative tool pose through damped-least-squares IK (scale 0.1,
  tool 0.107 m along ``panda_hand``), plus the task's own filtered binary gripper. Observations are
  untouched (59 values: the applied-action term shrinks from 7+1 to 6+1 on its own).
- **Second observation group** ``action_history`` (49 values, user request): the last two applied
  actions, the sheet key points and the target measured from the tool point, and the finger width,
  without the joint state -- see
  ``ActionHistoryObsCfg``. The ``policy`` group is unchanged, so existing checkpoints run as before;
  a policy is trained on the new group only through ``SheetTwistActionHistoryPPORunnerCfg``.

Episode outcomes (user request): one gripper cycle per episode -- closed once, opened once. The
first closing command is the grasp; the first opening command after it is the release, and the
release ends the episode on that step either way, judged on the platform coverage it was taken
with (``sheet_twist_mdp._commanded_release``):

- **placed** (success): at least ``RELEASE_COVERAGE`` of the platform covered when the opening
  command is applied. On its terminal step the placement term's success component pays its full
  5000 flat (user request; +143 in the return) -- see ``placement_success_on_placed``.
- **dropped** (failure): less than ``RELEASE_COVERAGE`` covered at that same moment.

The two are complements on one step, so every release resolves to exactly one outcome and nothing
follows it: there is no settling window, and where the sheet lands after the fingers part does not
count. Opening commands before any closure are the approach with an open hand and are ignored; a
second closure cannot occur because the first opening ends the episode.

No alive cost (user request: the earlier win-or-bleed charge made returns too negative; its terms
``alive_cost``/``unfinished_cost`` remain available in ``sheet_twist_mdp``). What keeps farming
out without it: the episode cannot outlive its one release. The placement term's three coverage
reward components are scaled to 15 times the original balance (user request).

Two deliberate deviations, both unavoidable or bug fixes:

- ``motion_cost`` is removed: it reads the smooth joint action's filter internals, which do not
  exist on the IK action, and takes the process down. It is the one original term that cannot apply.
- ``released_high`` gets its finger joints resolved. The original leaves the term's ``finger_cfg``
  as an unresolved default, so its "fingers physically open" test averages all nine joints, arm
  included, and fires on arm posture; under twist control that killed every carry ~0.3 s after
  extraction with the gripper shut.

A logging-only ``coverage`` term (weight 0.001, so it never matters to the return) is included
because the evaluation tooling in ``scripts/train_dagger_sheet_twist.py`` reads it for its
``Coverage/*`` columns -- and the placed/dropped terminations read its state.

Registered as ``Template-Sheet-Rl-Platform-TwistStudent-v0``.
"""

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from ..sheet_rl.mdp.grasp_control import applied_policy_actions
from ..sheet_rl.mdp.observations import finger_width, gripper_opening, sheet_key_points_from_tool, target_from_tool
from ..sheet_rl.sheet_rl_env_cfg import SHEET_RESOLUTION, SHEET_SIZE
from ..sheet_rl.sheet_rl_platform_env_cfg import SheetRlPlatformEnvCfg
from .sheet_twist_env_cfg import SheetTwistScenePresetCfg
from .sheet_twist_mdp import (
    PLATFORM_DEPTH,
    lowered_without_release,
    placed_coverage_bonus,
    placement_success_on_placed,
    randomize_action_scale,
    release_edge_angle,
    platform_coverage,
    set_platform_height,
    sheet_dropped,
    sheet_placed,
)

EPISODE_STEPS = 1500
"""Control steps per episode: 30 s at 50 Hz (the original: 33 s at 30 Hz)."""

PLATFORM_HEIGHTS = (0.02, 0.14)
"""Range the platform top is drawn from at every reset [m] (user request)."""

RELEASE_COVERAGE = 0.5
"""Platform coverage the sheet must already hold when the gripper is commanded open for the
release to count as a success (user request: more than half the platform)."""

RELEASE_MAX_ANGLE = 20.0
"""How far the sheet's marked edge may sit from a platform edge and still count as a placement
[deg] (user request). Past it the release is a failure whatever it covers -- and the geometry
agrees: a 30 degree yaw caps coverage at 0.844 even perfectly centred."""

SUCCESS_COVERAGE_BONUS = 300000.0
"""What a release at full coverage pays, in literal reward units (user request: very big). It
reaches the return at 1/REWARD_DIVISOR, so 8571 of it at coverage 1.0 and 2582 at 0.85."""

SUCCESS_COVERAGE_GAIN = 8.0
"""Exponent of the success pay: every extra 0.1 of coverage at release multiplies it by e^0.8."""

RELEASE_FLOOR = 0.01
"""Height above the platform top at which a still-shut gripper fails the episode [m] (user
request): the sheet must be let go of on the way down, not carried to the platform and held."""

REWARD_DIVISOR = 35.0
"""Every reward term's weight is divided by this (user request), so the documented literal amounts
(budgets, bonuses, penalties) all reach the return at 1/35 of their stated value."""



class sheet_center_progress(ManagerTermBase):
    """A progressive proximity budget: ``bonus x progress of exp(-d / std)``, where ``d`` is the
    3-D distance from the deformable sheet's centre to the centre of the platform's top surface.

    Progress is the placement term's currency: only a new episode-best proximity pays, so the
    budget (at most ``bonus``, reached as d goes to zero) can be collected once per episode and
    neither hovering nor retreat-and-return collects it again. The baseline is captured on the
    episode's first step, so the starting distance itself earns nothing. Returned divided by the
    step dt: with weight 1.0 the ``bonus`` is the literal episode total.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = 0.0
        self.fresh[ids] = True

    def __call__(self, env, bonus: float = 1500.0, std: float = 0.2,
                 sheet_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
                 platform_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
                 top_offset: float = PLATFORM_DEPTH / 2) -> torch.Tensor:
        platform = env.scene[platform_cfg.name]
        offset = torch.zeros(env.num_envs, 3, device=env.device)
        offset[:, 2] = top_offset
        top_centre = platform.data.root_pos_w.torch + quat_apply(platform.data.root_quat_w.torch, offset)
        centre = env.scene[sheet_cfg.name].data.nodal_pos_w.torch.mean(dim=1)
        score = torch.exp(-(centre - top_centre).norm(dim=-1) / std)
        self.best = torch.where(self.fresh, score, self.best)
        self.fresh &= False
        delta = (score - self.best).clamp_min(0.0)
        self.best += delta
        return bonus * delta / env.step_dt


@configclass
class ActionHistoryObsCfg(ObsGroup):
    """A second observation group: the last two actions, the sheet and the goal; no joint state.

    49 values, concatenated in this order:

    - ``actions`` (2 x 7): the applied action -- 6 raw IK pose-step values plus the filtered gripper
      state -- at t-2 then t-1. After a reset both slots hold the reset value.
    - ``deformable_sampled_points`` (27): the nine sheet landmarks as offsets from the tool point
      (the point between the fingertips that the IK action controls), robot-frame axes.
    - ``target_position`` (7): the platform top-centre as an offset from the tool point, robot-frame
      axes, plus the platform's orientation quaternion in the robot frame.
    - ``finger_width`` (1): the gripper opening -- the distance between the fingertip pads [m],
      0.08 open, a few millimetres shut on the sheet (``gripper_opening``; repaired 2026-10-26, see
      ``ActionHistoryLegacyObsCfg`` for what the slot held before).

    User request: everything positional is measured from the tool, never from the robot base, so
    the base-to-hand distance appears nowhere; the ``policy`` group (the teacher's) keeps the
    base-frame terms.

    It sits beside ``policy``, never in place of it: rsl_rl fills its actor and critic from
    ``policy`` unless ``obs_groups`` names another group, so checkpoints trained on the 59-value
    group are unaffected by this one existing.
    """

    actions = ObsTerm(func=applied_policy_actions, history_length=2)
    # the tool point is the IK action's: 0.107 m along panda_hand (its body_offset)
    deformable_sampled_points = ObsTerm(
        func=sheet_key_points_from_tool,
        params={"resolution": SHEET_RESOLUTION, "asset_cfg": SceneEntityCfg("deformable"),
                "hand_body_name": "panda_hand", "tool_offset": (0.0, 0.0, 0.107)},
    )
    target_position = ObsTerm(
        func=target_from_tool,
        params={"command_name": "deformable_pose", "hand_body_name": "panda_hand", "tool_offset": (0.0, 0.0, 0.107)},
    )
    # the asset cfg is passed in params so the manager resolves it to the two finger joints
    finger_width = ObsTerm(
        func=gripper_opening, params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")}
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ActionHistoryLegacyObsCfg(ActionHistoryObsCfg):
    """``action_history`` as every actor trained before 2026-10-26 read it: slot 49 is the SUM OF ALL
    NINE JOINT POSITIONS, not the finger width.

    ``ObsTerm(func=finger_width)`` with no ``params`` leaves the function's default
    ``SceneEntityCfg`` unresolved, so its ``joint_ids`` is ``slice(None)`` and the term sums the
    seven arm joints as well as the two fingers (0.2-0.6 at reset, swinging between -0.9 and +1.0
    through a lift-and-turn, while the true finger width is 0-0.08). Those actors depend on it
    (0/3 placed with the true width in the slot), so the group is kept under this name for loading
    them -- as the teacher of the repaired student, or with ``play_sac.py --actor_obs_groups
    action_history_legacy``. New actors read ``action_history``.
    """

    finger_width = ObsTerm(func=finger_width)


@configclass
class SheetPlatformTwistStudentEnvCfg(SheetRlPlatformEnvCfg):
    """See the module docstring: the original platform task, twist-student control only."""

    # the twist tasks' scene: the sunk platform box, and the student's weightless 400/80 arm
    scene: SheetTwistScenePresetCfg = SheetTwistScenePresetCfg()

    def __post_init__(self):
        super().__post_init__()
        # -- control rate and episode length
        self.sim.dt = 1.0 / 150
        self.decimation = 3
        self.sim.render_interval = self.decimation
        self.episode_length_s = (EPISODE_STEPS - 0.5) * self.decimation * self.sim.dt
        self.commands.deformable_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # -- the sunk box's true top is half its depth above its root, not the slab's 1 cm
        self.commands.deformable_pose.top_offset = PLATFORM_DEPTH / 2
        # -- random platform height (user request): after reset_layout, which places the box at its
        # table position, sink it until its top is at a height drawn from PLATFORM_HEIGHTS
        self.events.set_platform_height = EventTerm(
            func=set_platform_height, mode="reset", params={"height_range": PLATFORM_HEIGHTS}
        )
        # -- the student's action in place of the smooth joint action (gripper stays the task's own)
        self.actions.joint.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.default = self.actions.joint
        # -- per-episode IK scale randomisation (user request); off until a range is given, e.g.
        # env.events.randomize_arm_scale.params.scale_range=[0.04,0.1]. Declared after the layout
        # events; the order among reset events does not matter to it.
        self.events.randomize_arm_scale = EventTerm(
            func=randomize_action_scale, mode="reset", params={"scale_range": None, "action_name": "arm_action"}
        )
        # -- the second observation group (user request); ``policy`` stays the 59-value original.
        # ``action_history_legacy`` is the same group with the pre-repair slot 49, for old actors.
        self.observations.action_history = ActionHistoryObsCfg()
        self.observations.action_history_legacy = ActionHistoryLegacyObsCfg()
        # -- the training plant's mass variation
        self.events.randomize_arm_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_link.*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        # -- the two deviations documented in the module docstring
        self.rewards.motion_cost = None
        self.rewards.grasp_progress.params["terminal_potential_zero"] = True
        self.terminations.released_high.params["finger_cfg"] = SceneEntityCfg(
            "robot", joint_names="panda_finger_joint.*"
        )
        # -- released_high measured from the sunk box: local z - depth/2 is height above the
        # (moving) platform top, so the release ceiling is top-relative -- 10 cm above the top
        # (the original: 10 cm above the table, 8 cm above its fixed 2 cm slab). Left as the
        # literal slab offsets, the box's geometry reads every release as high (see the
        # OrigReward bridge's note in sheet_twist_env_cfg).
        self.terminations.released_high.params.update(top_offset=-PLATFORM_DEPTH / 2, max_height=0.10)
        # -- episode outcomes (user request): the episode's one commanded opening ends it, as a
        # success if the sheet already covers RELEASE_COVERAGE of the platform at that moment AND
        # lies within RELEASE_MAX_ANGLE of square with it, as a failure otherwise. Complements on
        # the same step, so exactly one of them fires.
        self.terminations.placed = DoneTerm(
            func=sheet_placed,
            params={"coverage_threshold": RELEASE_COVERAGE, "max_angle": RELEASE_MAX_ANGLE},
        )
        self.terminations.dropped = DoneTerm(
            func=sheet_dropped,
            params={"coverage_threshold": RELEASE_COVERAGE, "max_angle": RELEASE_MAX_ANGLE},
        )
        # -- the release floor (user request): coming down to RELEASE_FLOOR above the platform top
        # still holding the sheet, with no opening command applied, ends the episode as a failure.
        # It is what stops "lower it on and never let go", which costs nothing otherwise.
        self.terminations.held_too_low = DoneTerm(
            func=lowered_without_release,
            params={"min_height": RELEASE_FLOOR, "top_offset": -PLATFORM_DEPTH / 2},
        )
        # -- placement reward re-anchored to the sunk box. top_offset is root-to-top; its
        # "elevation" then reads height-above-top + PLATFORM_DEPTH, so release_height moves by the
        # same amount to keep the release grading's meaning: full pay within ~1 cm of the top.
        self.rewards.placement.params["top_offset"] = PLATFORM_DEPTH / 2
        self.rewards.placement.params["release_height"] = PLATFORM_DEPTH + 0.01
        # -- coverage pay: 15 times the original amounts (user request).
        self.rewards.placement.params["coverage_bonus"] = 22500.0
        self.rewards.placement.params["released_coverage_bonus"] = 15000.0
        self.rewards.placement.params["released_coverage_step"] = 75.0
        # -- the placement term's 5000 success bonus, paid by the real outcome (user request): the
        # parent's own success test (retreat-and-hold) never fires, so this subclass pays it flat
        # on the placed termination's terminal step; PlacementReward/success logs 5000 per success
        # (+143 in the return). It replaces the earlier flat +120 placed_bonus term, so success is
        # paid through one channel. The alive/unfinished bleed stays removed (too negative).
        self.rewards.placement.func = placement_success_on_placed
        # -- progressive proximity budget (user request): up to 1500 as the sheet centre closes on
        # the platform-top centre, paid only on new episode bests; see sheet_center_progress above.
        # The placement term's own approach component (500, holding-gated) remains active beside it.
        self.rewards.center_progress = RewTerm(func=sheet_center_progress, weight=1.0)
        # -- the success pay, exponential in the coverage the release was taken with (user
        # request: push for coverage, and make it very big). 0.5 pays about what the placement
        # term's flat 5000 does; 0.85 pays 16 times that, 1.0 pays 55 times. Declared before the
        # ``coverage`` term below so it reads the same value the placed termination judged.
        self.rewards.placed_coverage = RewTerm(
            func=placed_coverage_bonus, weight=1.0,
            params={"bonus": SUCCESS_COVERAGE_BONUS, "gain": SUCCESS_COVERAGE_GAIN},
        )
        # -- diagnostic only: how square the sheet lands. On the release step it
        # measures the angle between the sheet's red-dot bottom edge and the platform's four top
        # edges, logs the smallest (the placement's yaw error, 0-45 deg) as
        # Placement/release_angle_min, and prints a line per release when few enough environments
        # are running to watch -- playback prints, training does not.
        # The weight is 1e-3 rather than 0 for the same reason the coverage term's is: the reward
        # manager skips zero-weight terms outright, and a skipped term never measures anything.
        # The term returns zeros, so what reaches the return is exactly nothing either way.
        self.rewards.release_angle = RewTerm(
            func=release_edge_angle, weight=1e-3, params={"resolution": SHEET_RESOLUTION},
        )
        # -- logging-only, for the evaluation tooling's Coverage/* columns; the placed/dropped
        # terminations read its coverage and released-latch state
        self.rewards.coverage = RewTerm(
            func=platform_coverage,
            weight=1e-3,
            params={
                "resolution": SHEET_RESOLUTION,
                "platform_size": SHEET_SIZE,
                "top_offset": PLATFORM_DEPTH / 2,
                "finger_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
            },
        )
        # -- global reward scale (user request): every term's weight divided by REWARD_DIVISOR.
        # Relative balance and the terms' internal "literal amount" bookkeeping are untouched;
        # only what reaches the return shrinks. Keep this last, after every term is final.
        for term in vars(self.rewards).values():
            if isinstance(term, RewTerm):
                term.weight /= REWARD_DIVISOR
