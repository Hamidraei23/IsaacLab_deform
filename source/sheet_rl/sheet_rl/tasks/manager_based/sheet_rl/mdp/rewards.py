# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


_PHASE_ATTR = "_sheet_placement_phase"
_COVERAGE_ATTR = "_sheet_band_coverage"
_HOLDING_ATTR = "_sheet_holding"
_EARLY_RELEASE_ATTR = "_sheet_early_release"
_DEBUG_ATTR = "_sheet_debug"


def grasp_debug(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Every quantity the grasp test looked at this step, for offline inspection.

    Populated by :class:`grasp_stage_reward` on each call and overwritten in place, so it always
    describes the current step. Nothing in the environment reads it -- it exists so a recorded
    teleop episode can be replayed against the exact numbers the reward saw, rather than against a
    reconstruction of them.
    """
    state = getattr(env, _DEBUG_ATTR, None)
    if state is None:
        state = {}
        setattr(env, _DEBUG_ATTR, state)
    return state


def _early_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment flag: was the sheet let go of before it left the slot?

    Set by :class:`grasp_stage_reward` and read by the matching termination term, which is what
    actually ends the episode. Kept on the environment because the reward manager runs before the
    termination manager within a step, so the flag is always current when it is read.
    """
    state = getattr(env, _EARLY_RELEASE_ATTR, None)
    if state is None:
        state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, _EARLY_RELEASE_ATTR, state)
    return state


def _is_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment flag: is the gripper currently holding the sheet?

    Owned by :class:`grasp_stage_reward`, which is the term that tracks the grasp state machine,
    and republished here so the lift shaping can require a grasp without duplicating that logic.
    Created lazily so read order does not matter.
    """
    state = getattr(env, _HOLDING_ATTR, None)
    if state is None:
        state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, _HOLDING_ATTR, state)
    return state


def _phase_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-environment latch: has the sheet been pulled clear of the slot this episode?

    The task is split in two. Phase one is "get the sheet out of the slot"; phase two is "drape it
    on the red band". This flag is what separates them: it is set by :class:`grasp_stage_reward`
    the first time the sheet clears the walls, and read by the placement terms, which pay nothing
    until it is set.

    Kept on the environment rather than inside any one term because writer and readers are
    different reward terms, and created lazily so it does not matter which of them runs first.
    """
    state = getattr(env, _PHASE_ATTR, None)
    if state is None:
        state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, _PHASE_ATTR, state)
    return state


def _last_coverage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Coverage fraction measured by :func:`band_coverage` earlier in this same step.

    Republished here so the release bonus can be scaled by how well the sheet actually sits on the
    band at the instant the gripper opens, without paying for a second ``cdist`` over every node.
    Reward terms run in declaration order and ``coverage`` is declared before ``grasp_stage``, so
    this is the current step's value, not the previous one.
    """
    state = getattr(env, _COVERAGE_ATTR, None)
    if state is None:
        state = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _COVERAGE_ATTR, state)
    return state


def _closing_squareness(
    robot: Articulation, slot: RigidObject, left_id: int, right_id: int
) -> torch.Tensor:
    """How square the fingers' closing axis is to the slot, in [0, 1].

    One when the axis lies across the gap -- one pad ending up on each face of the sheet, so they
    close through its thickness -- and zero when the fingers would close along the gap instead.

    Absolute because the gripper is symmetric: a grasp made with the hand rotated by 180 degrees
    puts the same two pads on the same two faces, so the two must score alike.

    Defined once and shared by everything that reads it. It used to be written out separately in
    the alignment shaping and in the grasp term's diagnostics, which meant the quantity being
    rewarded and the number logged to judge it could drift apart without anyone noticing.
    """
    unit_y = torch.tensor([0.0, 1.0, 0.0], device=robot.device).expand(robot.num_instances, 3)
    lateral = quat_apply(slot.data.root_quat_w.torch, unit_y)
    bodies = robot.data.body_link_pose_w.torch
    closing = bodies[:, right_id, :3] - bodies[:, left_id, :3]
    closing = closing / closing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (closing * lateral).sum(-1).abs()


def _band_frame(
    env: ManagerBasedRLEnv, command_name: str, arm_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the red band's world centre and unit axis.

    Derived from the arm's live pose rather than from a value cached by the command term: within a
    step the reward manager runs *before* the command manager, so anything the command caches is
    one step stale and simply wrong on the first step after a reset. ``band_offset_b`` is written
    during the reset itself, so combining it with the current arm pose is always current.
    """
    command = env.command_manager.get_term(command_name)
    arm: RigidObject = env.scene[arm_cfg.name]
    arm_pos = arm.data.root_pos_w.torch
    arm_quat = arm.data.root_quat_w.torch
    axis_local = torch.tensor([1.0, 0.0, 0.0], device=arm_pos.device).expand(len(arm_pos), 3)
    center = arm_pos + quat_apply(arm_quat, command.band_offset_b)
    axis = quat_apply(arm_quat, axis_local)
    return center, axis


def band_coverage(
    env: ManagerBasedRLEnv,
    command_name: str,
    band_length: float,
    band_radius: float,
    cover_threshold: float = 0.03,
    num_axial: int = 5,
    num_angular: int = 7,
    coverage_arc: float = math.pi / 2,
    gate_on_phase: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    arm_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
) -> torch.Tensor:
    """Fraction of the red region's surface that the sheet covers, in [0, 1].

    The band's surface is sampled on a grid of :paramref:`num_axial` points along the arm by
    :paramref:`num_angular` points around it, and a sample counts as covered when some sheet node
    lies within :paramref:`cover_threshold` of it. The reward is the fraction covered, so it grows
    smoothly as the sheet is spread over the region rather than only paying out at the end.

    Only the arc reachable from above is sampled: the arm rests on the table, so its underside can
    never be covered and including it would cap the achievable reward below one.

    Args:
        env: The environment.
        command_name: Command term carrying the band's offset along the arm.
        band_length: Extent of the region along the arm [m].
        band_radius: Radius the samples sit at, i.e. the arm's radius [m].
        cover_threshold: Distance within which a sheet node counts as covering a sample [m].
            Should exceed the sheet's node spacing, or a correctly draped sheet still reads as
            covering nothing.
        num_axial: Samples along the arm.
        num_angular: Samples around the arm.
        coverage_arc: Half-angle of the sampled arc, measured from straight up [rad].
        gate_on_phase: Pay nothing until the sheet has been pulled clear of the slot. This is what
            makes the placement objective a second phase: brushing the band with a sheet still
            standing in its slot earns nothing, so the policy cannot short-circuit the pick.
        asset_cfg: The deformable sheet.
        arm_cfg: The mannequin arm.
    """
    sheet: DeformableObject = env.scene[asset_cfg.name]
    nodes = sheet.data.nodal_pos_w.torch
    center, axis = _band_frame(env, command_name, arm_cfg)
    device = nodes.device
    num_envs = len(center)

    # build an orthonormal frame around the arm: "up" is world +z projected perpendicular to the
    # axis, which is well defined because the arm always lies flat (roll and pitch stay zero)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, 3)
    side = torch.cross(axis, world_up, dim=-1)
    side = side / side.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    radial_up = torch.cross(side, axis, dim=-1)

    t = torch.linspace(-0.5 * band_length, 0.5 * band_length, num_axial, device=device)
    theta = torch.linspace(-coverage_arc, coverage_arc, num_angular, device=device)

    # (num_envs, num_axial, num_angular, 3)
    along = t.view(1, num_axial, 1, 1) * axis.view(num_envs, 1, 1, 3)
    radial = torch.cos(theta).view(1, 1, num_angular, 1) * radial_up.view(
        num_envs, 1, 1, 3
    ) + torch.sin(theta).view(1, 1, num_angular, 1) * side.view(num_envs, 1, 1, 3)
    samples = center.view(num_envs, 1, 1, 3) + along + band_radius * radial
    samples = samples.reshape(num_envs, num_axial * num_angular, 3)

    nearest = torch.cdist(samples, nodes).min(dim=-1).values
    covered = (nearest < cover_threshold).float().mean(dim=-1)
    # publish the ungated fraction for the release bonus, which needs the raw quality of the drape
    _last_coverage(env).copy_(covered)
    if gate_on_phase:
        covered = covered * _phase_reached(env).float()
    return covered


def band_center_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.1,
    gate_on_phase: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    arm_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
) -> torch.Tensor:
    """Shaped reward for the sheet's centre sitting on the centre of the red region.

    Complements :func:`band_coverage`, which is indifferent to *which* part of the sheet does the
    covering: a sheet draped by its corner can cover the band as well as one draped by its middle.
    This term asks for the sheet to be centred on the region, which is what leaves margin on all
    sides.

    Args:
        env: The environment.
        command_name: Command term carrying the band's offset along the arm.
        std: Distance scale of the shaping [m].
        gate_on_phase: Pay nothing until the sheet has been pulled clear of the slot. Without the
            gate this term is largest at the *start* of the episode, when the sheet stands a short
            hop from the arm, and would reward never picking it up at all.
        asset_cfg: The deformable sheet.
        arm_cfg: The mannequin arm.
    """
    sheet: DeformableObject = env.scene[asset_cfg.name]
    center, _ = _band_frame(env, command_name, arm_cfg)
    distance = torch.norm(center - sheet.data.root_pos_w.torch, dim=-1)
    reward = 1.0 - torch.tanh(distance / std)
    if gate_on_phase:
        reward = reward * _phase_reached(env).float()
    return reward


def _border_ids(resolution: tuple[int, int], device) -> torch.Tensor:
    """Index rows for the four borders of the sheet's node grid."""
    cols, rows = resolution[0] + 1, resolution[1] + 1
    grid = torch.arange(rows * cols, device=device).view(rows, cols)
    return torch.stack([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])


def _top_edge_nodes(nodes: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """Positions along whichever border currently sits highest. Shape ``(N, k, 3)``.

    Chosen per step by height rather than fixed at build time: which grid row ends up on top
    depends on how the sheet was stood upright, and it can change as the cloth deforms.
    """
    borders = _border_ids(resolution, nodes.device)
    candidates = nodes[:, borders.reshape(-1), :].view(len(nodes), *borders.shape, 3)
    best = candidates[..., 2].mean(dim=-1).argmax(dim=1)
    return candidates[torch.arange(len(nodes), device=nodes.device), best]


def _grasp_site(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    hand_body_name: str,
    grasp_offset: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """World position of the point between the fingertips, and the hand's orientation."""
    robot: Articulation = env.scene[robot_cfg.name]
    pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index(hand_body_name)]
    offset = torch.tensor(grasp_offset, device=pose.device).expand(len(pose), 3)
    return pose[:, :3] + quat_apply(pose[:, 3:7], offset), pose[:, 3:7]


def top_edge_distance(
    env: ManagerBasedRLEnv,
    resolution: tuple[int, int],
    coarse_std: float = 0.4,
    fine_std: float = 0.05,
    hand_body_name: str = "panda_hand",
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shaped reward for bringing the fingertips to the middle of the sheet's top edge.

    The target is the one place the sheet may be taken from, so this is the whole approach: it
    pulls the gripper across the table and down onto the strip standing above the slot walls.

    Aimed at the *centre* of that edge rather than the nearest point on it, because a grasp near a
    corner peels the sheet sideways out of the slot instead of drawing it straight up.

    Two ``tanh`` scales are averaged rather than one, because a single scale cannot do both jobs.
    ``tanh`` saturates hard: at the half-metre the gripper starts from, a 0.1 m scale has a slope of
    about 0.002 per metre, so closing a centimetre earns ~1e-5 and the signal is buried under the
    policy's own action noise -- a run with exactly that setup sat 0.5 m from the sheet for 78
    iterations without ever approaching. The coarse scale carries the gripper across the workspace,
    the fine one positions it once it arrives.

    Args:
        env: The environment.
        resolution: The sheet mesh's ``(x, y)`` element counts.
        coarse_std: Distance scale of the travel term [m]. Should be comparable to how far the
            gripper starts from the sheet, or it saturates before it can pull.
        fine_std: Distance scale of the placement term [m]. Comparable to the grasp radius.
        hand_body_name: Body the grasp frame hangs off.
        grasp_offset: Grasp frame offset in the hand's frame [m].
        asset_cfg: The deformable sheet.
        robot_cfg: The robot.
    """
    sheet: DeformableObject = env.scene[asset_cfg.name]
    tip, _ = _grasp_site(env, robot_cfg, hand_body_name, grasp_offset)
    center = _top_edge_nodes(sheet.data.nodal_pos_w.torch, resolution).mean(dim=1)
    distance = (tip - center).norm(dim=-1)
    return 0.5 * (1.0 - torch.tanh(distance / coarse_std)) + 0.5 * (
        1.0 - torch.tanh(distance / fine_std)
    )


def ee_table_clearance(
    env: ManagerBasedRLEnv,
    floor: float = 0.06,
    hand_body_name: str = "panda_hand",
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty, in [0, 1], that grows as the fingertips drop towards the table.

    Zero above :paramref:`floor` and rising linearly to one at table level, so it is a slope the
    policy can feel on the way down rather than a wall it only discovers by hitting.

    The ``ee_below_table`` termination already ends the episode when the gripper goes under, but a
    termination on its own teaches nothing: by the time it fires the episode is over and the only
    lesson available is that the return was short. With no reward anywhere pointing away from the
    floor, exploration drifts down and stays there -- one run lost 61% of its episodes this way.
    This gives the drift something to push back against; the termination stays as the backstop.

    Measured from the grasp frame rather than the hand, since the fingertips reach lowest.
    """
    tip, _ = _grasp_site(env, robot_cfg, hand_body_name, grasp_offset)
    return ((floor - tip[:, 2]) / floor).clamp(min=0.0)


def ee_speed_penalty(
    env: ManagerBasedRLEnv,
    free_speed: float = 0.1,
    reference_speed: float = 1.0,
    reference_penalty: float = 25.0,
    hand_body_name: str = "panda_hand",
    max_penalty: float | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge for how fast the hand travels, rising exponentially past a free allowance.

    Zero at or below :paramref:`free_speed`, then growing as
    ``exp(rate * (speed - free_speed)) - 1`` with ``rate`` fixed by requiring that
    :paramref:`reference_speed` costs exactly :paramref:`reference_penalty`, and finally held flat
    at :paramref:`max_penalty`. Setting that ceiling equal to :paramref:`reference_penalty` makes
    the curve level off exactly where it reaches its reference point.

    Measured in Cartesian space rather than on the joints, and that choice matters here: a
    joint-space charge bills every wrist rotation at the same rate as useful travel, which puts it
    in direct opposition to ``square_progress``, a term paid *for* turning the wrist. Hand speed is
    indifferent to how the arm arrives at a pose, so the two do not fight.

    Taken at the hand link rather than at the grasp frame for the same reason. The fingertips sweep
    an arc when the wrist turns, so a grasp-frame measurement carries the wrist's angular rate into
    the charge; the hand link does not.

    Warning:
        Left uncapped the growth keeps compounding, and it compounds fast enough that a brief
        excursion outweighs an entire episode of ordinary reward -- which is a good way to hand the
        value function a target it cannot fit. :paramref:`max_penalty` is what keeps the curve
        bounded; leave it set unless the tail is wanted.

    Note:
        Divided by ``step_dt`` so :paramref:`reference_penalty` is the literal charge per step,
        matching the convention the other explicit magnitudes in this file use.

    Args:
        env: The environment.
        free_speed: Hand speed below which nothing is charged [m/s].
        reference_speed: Speed at which the charge equals :paramref:`reference_penalty` [m/s].
        reference_penalty: Charge per step at :paramref:`reference_speed` [reward].
        hand_body_name: Body whose linear velocity is measured.
        max_penalty: Optional ceiling on the per-step charge [reward]. ``None`` leaves it unbounded.
        robot_cfg: The robot.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    velocity = robot.data.body_link_vel_w.torch[:, robot.body_names.index(hand_body_name), :3]
    excess = (velocity.norm(dim=-1) - free_speed).clamp(min=0.0)
    # log1p/expm1 rather than log/exp: the curve is pinned through (free_speed, 0), and these keep
    # that exact instead of leaving a rounding step at the allowance boundary
    rate = math.log1p(reference_penalty) / max(reference_speed - free_speed, 1e-6)
    penalty = torch.expm1(rate * excess)
    if max_penalty is not None:
        penalty = penalty.clamp(max=max_penalty)
    return penalty / env.step_dt


def grasp_alignment(
    env: ManagerBasedRLEnv,
    resolution: tuple[int, int],
    std: float = 0.1,
    downward_frac: float = 0.5,
    hand_body_name: str = "panda_hand",
    left_finger_body_name: str = "panda_leftfinger",
    right_finger_body_name: str = "panda_rightfinger",
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    slot_cfg: SceneEntityCfg = SceneEntityCfg("slot_neg_y"),
) -> torch.Tensor:
    """Shaped reward for holding the hand in the one posture that can take the top edge.

    Two things are asked for at once, averaged:

    * **pointing down** -- the hand's approach axis faces the table, so the fingers hang below the
      wrist and can be lowered onto the edge from directly above.
    * **square to the slot** -- the fingers' closing axis lies across the gap, so one pad ends up on
      each face of the sheet and they close through its thickness.

    Scaled by how close the fingertips are to the edge, so striking the pose on the far side of the
    table is worth little. The scale is deliberately loose: gated tightly the term is numerically
    dead during the approach -- at 0.05 m a gripper half a metre away sees a factor of about 1e-9 --
    and the policy arrives with its wrist in an arbitrary posture having never been paid to turn it.
    Pointing the hand down is easy to learn and useful the whole way in, so it is better rewarded
    early and weakly than late and sharply.

    :paramref:`downward_frac` splits the term between the two. It used to be a hardcoded even
    split, which is wasteful once one of the two is solved: with the hand already pointing down on
    nine steps in ten, half of every unit of this term was being paid for something the policy was
    no longer going to change, and the half that could still buy something got diluted for it.

    Args:
        env: The environment.
        resolution: The sheet mesh's ``(x, y)`` element counts.
        std: Distance scale of the proximity gate [m].
        downward_frac: Share of the term given to pointing down, in [0, 1]. The remainder goes to
            being square to the slot. At 0 the term pays for squareness alone.
        hand_body_name: Body the grasp frame hangs off.
        left_finger_body_name: Left fingertip body.
        right_finger_body_name: Right fingertip body.
        grasp_offset: Grasp frame offset in the hand's frame [m].
        asset_cfg: The deformable sheet.
        robot_cfg: The robot.
        slot_cfg: The slot wall whose lateral axis defines "square".
    """
    sheet: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    slot: RigidObject = env.scene[slot_cfg.name]

    tip, hand_quat = _grasp_site(env, robot_cfg, hand_body_name, grasp_offset)
    center = _top_edge_nodes(sheet.data.nodal_pos_w.torch, resolution).mean(dim=1)
    proximity = 1.0 - torch.tanh((tip - center).norm(dim=-1) / std)

    unit_z = torch.tensor([0.0, 0.0, 1.0], device=tip.device).expand(len(tip), 3)
    # the hand's +z runs from the wrist out towards the fingers, so pointing down means its world
    # z component is -1
    approach_z = quat_apply(hand_quat, unit_z)[:, 2]
    downward = 0.5 * (1.0 - approach_z)

    square = _closing_squareness(
        robot,
        slot,
        robot.body_names.index(left_finger_body_name),
        robot.body_names.index(right_finger_body_name),
    )

    return proximity * (downward_frac * downward + (1.0 - downward_frac) * square)


class squareness_progress(ManagerTermBase):
    """Potential-based shaping that pays for *turning* the wrist square to the slot.

    Pays the change in squareness rather than its level, so the reward arrives on the step the
    wrist actually turns instead of forty steps later folded into an extraction bonus. Credit
    assignment is the entire point of the term. Squareness already influences the return -- a
    crooked grasp is a worse grasp -- but only through a payout at the end of a long episode,
    which has to be propagated back through a critic simultaneously fitting returns that span
    three orders of magnitude. The gradient that reaches joint 7 by that route is negligible.

    Since the increments telescope, an episode collects ``max_reward * (square_end -
    square_start)`` regardless of the route the wrist took, so there is nothing to farm: turning
    square and then back again nets exactly zero. That is what separates this from scaling some
    other term *by* squareness, which is path-dependent and pays for the same turn over and over
    -- a version of this that multiplied the lift shaping's delta could be cycled for +450 per
    bob, while this one nets 0 by construction.

    Deliberately ungated. The lift shaping has to require a grasp because it pays for the sheet
    rising by any means, and batting it up is easier to find than lifting it; this pays for an
    orientation the gripper is free to correct at any moment, and a gate would only introduce an
    edge for the potential to jump across.

    The first step of an episode pays nothing. There is no previous value to difference against,
    and treating the reset value as one would hand out up to ``max_reward`` of reward decided
    entirely by the reset draw -- noise the policy cannot act on, straight into the critic.

    Args:
        env: The environment.
        max_reward: Total paid for going from fully skew to fully square [reward]. Signed: losing
            the alignment again refunds it.
        left_finger_body_name: Left fingertip body.
        right_finger_body_name: Right fingertip body.
        robot_cfg: The robot.
        slot_cfg: The slot wall whose lateral axis defines "square".
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._robot: Articulation = env.scene[params["robot_cfg"].name]
        self._slot: RigidObject = env.scene[params["slot_cfg"].name]
        names = self._robot.body_names
        self._left_id = names.index(params.get("left_finger_body_name", "panda_leftfinger"))
        self._right_id = names.index(params.get("right_finger_body_name", "panda_rightfinger"))
        self._prev = torch.zeros(env.num_envs, device=env.device)
        # nothing to difference against on the first step of an episode
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._fresh[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        slot_cfg: SceneEntityCfg,
        left_finger_body_name: str = "panda_leftfinger",
        right_finger_body_name: str = "panda_rightfinger",
        max_reward: float = 300.0,
    ) -> torch.Tensor:
        square = _closing_squareness(self._robot, self._slot, self._left_id, self._right_id)
        delta = torch.where(self._fresh, torch.zeros_like(square), square - self._prev)
        self._prev = square
        self._fresh[:] = False
        # divided by step_dt so max_reward is the literal total for a full correction, matching the
        # convention the other payouts in this file use
        return delta * max_reward / env.step_dt


class gripper_recommit_penalty(ManagerTermBase):
    """Charge for every closure of the gripper after the first in an episode.

    Counted on the *commanded* bit rather than on the measured finger width. The gripper channel is
    one bit -- the sign of a sample from the policy's Gaussian -- so the command flickers open and
    shut far faster than the fingers can travel, and the width smooths most of that away: at an
    open fraction of about a fifth the command changes state on roughly a third of all steps, while
    the joints complete two or three closures in a whole episode. Charging the width bills the
    policy for the flicker it happened to be caught mid-way through; charging the command bills the
    decision, which is the thing it can actually control.

    The first closure is free, so lining the gripper up and shutting it once costs nothing. What is
    charged is *re-committing* -- opening again to take a second bite. That is what makes the first
    attempt worth aiming rather than something to be spammed until it happens to catch.

    A grasp needs no toggle at all if the policy simply keeps the reset width and closes when it
    arrives, so the floor really is zero rather than one closure's worth.

    Note:
        Divided by ``step_dt`` so :paramref:`penalty` is the literal charge per re-closure, rather
        than being scaled down by the reward manager's ``weight * dt``.

    Args:
        env: The environment.
        penalty: Charge for each closure after the first in an episode [reward].
        action_term_name: Action term carrying the binary gripper channel.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        name = cfg.params.get("action_term_name", "gripper_action")
        self._action_term = env.action_manager.get_term(name)
        # read off the action term rather than restated here, so the two cannot come to disagree
        # about which sign of the channel means open
        self._threshold = float(self._action_term.cfg.threshold)
        self._positive = bool(self._action_term.cfg.positive_threshold)
        # starts closed: whatever the episode opens with, the closure ending the first open-close
        # cycle is the free one
        self._was_open = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._closures = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # logged so the charge can be priced against what the policy actually does: the commanded
        # closure count is several times the width-based ``Events/closures`` and is the one this
        # term bills
        extras = self._env.extras.setdefault("log", {})
        counts = self._closures[env_ids].float()
        extras["Events/gripper_closures_raw"] = counts.mean()
        extras["Events/gripper_recommits"] = (counts - 1.0).clamp(min=0.0).mean()
        self._closures[env_ids] = 0
        self._was_open[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        penalty: float = 120.0,
        action_term_name: str = "gripper_action",
    ) -> torch.Tensor:
        raw = self._action_term.raw_actions[:, 0]
        is_open = raw > self._threshold if self._positive else raw < self._threshold
        closure = self._was_open & ~is_open
        self._closures += closure
        # the count already includes this step, so ">1" is exactly "not the first of the episode"
        charged = closure & (self._closures > 1)
        self._was_open = is_open
        return -penalty * charged.float() / env.step_dt


def sheet_extracted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The task's success condition: the sheet has been pulled clear of the slot.

    Reads the latch :class:`grasp_stage_reward` sets, so the reward and the termination cannot
    disagree about what counts as success. The reward manager runs before the termination manager
    inside a step, so the episode ends on the very step the sheet comes out.
    """
    return _phase_reached(env)


class sheet_lift_progress(ManagerTermBase):
    """Potential-based shaping that leads the policy up to the extraction bonus.

    Pays the *change* in a height potential rather than the height itself. The potential ramps
    linearly from 0 at the sheet's spawn height to 1 once its centre has risen
    :paramref:`lift_span` metres, and the reward each step is :paramref:`max_reward` times how much
    that potential grew. Since the increments telescope, an episode collects
    ``max_reward * potential`` in total no matter how slowly it climbs -- so
    :paramref:`max_reward` is the *whole* payout for a full lift, not a per-step rate.

    Why the difference and not the level: a reward for *being* high pays every step the sheet is
    held up, which makes hovering below the extraction threshold indefinitely worth far more than
    extracting and moving on. Rewarding progress removes any gain from lingering, and it is
    potential-based shaping in the standard sense, so it provably does not change which policy is
    optimal -- it only makes the extraction bonus easier to find.

    The change is signed, so letting the sheet fall back refunds what the climb paid. That is what
    stops the policy from farming the term by bobbing the sheet up and down.

    Once the sheet is extracted the term goes quiet. It is *not* driven to zero by the latch, which
    would claw back the whole climb in a single -``max_reward`` spike and teach the policy to avoid
    extracting; the potential simply stops being paid out, and everything already earned is kept.

    Args:
        env: The environment.
        lift_span: Rise at which the potential reaches 1 [m].
        max_reward: Total paid over a full lift from the spawn height to :paramref:`lift_span`.
        asset_cfg: The deformable sheet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sheet: DeformableObject = env.scene[cfg.params["asset_cfg"].name]
        self._prev = torch.zeros(env.num_envs, device=env.device)
        # highest the sheet got in the current episode, logged on reset. The lift is phase one's
        # whole objective now, and without this there is no way to tell "never lifted it at all"
        # apart from "lifted it almost high enough" -- both show up as a zero extraction count.
        self._best_rise = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Events/max_lift"] = self._best_rise[env_ids].mean()
        self._best_rise[env_ids] = 0.0
        # the reset puts the sheet back at its spawn height, where the potential is zero
        self._prev[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_span: float = 0.15,
        max_reward: float = 1000.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ) -> torch.Tensor:
        # measured against the sheet's own spawn height rather than a hardcoded number, so it stays
        # correct if the slot moves. Reset only displaces the sheet in x and y, never in z.
        spawn_height = self._sheet.data.default_nodal_state_w.torch[..., 2].mean(dim=1)
        rise = (self._sheet.data.root_pos_w.torch[:, 2] - spawn_height).clamp(min=0.0)
        # only upward motion counts: raw distance from the spawn pose would pay just as well for
        # dragging the sheet sideways out of the slot
        potential = (rise / lift_span).clamp(max=1.0)
        self._best_rise = torch.maximum(self._best_rise, rise)

        delta = potential - self._prev
        self._prev = potential
        # Only pays while the sheet is actually held, and only before it is extracted.
        #
        # The holding requirement is not optional. Without it the term pays for the sheet going up
        # by any means, and batting it upwards with the arm is far easier to discover than grasping
        # it -- a 554-iteration run collected 92% of its total reward that way, jostling the sheet
        # 3.5 cm without ever completing a grasp, and learned to thrash the arm to do it.
        delta = delta * (_is_holding(env) & ~_phase_reached(env)).float()

        # divided by step_dt so max_reward is the literal total for a full lift, matching the
        # convention grasp_stage_reward uses for its one-shot bonuses
        return delta * max_reward / env.step_dt


class grasp_stage_reward(ManagerTermBase):
    """Grasp the sheet by its top edge and hold it until it is out of the slot.

    Holding is *observed*, not inferred from a state machine: on every step the sheet counts as held
    when the fingers are shut and there is a sheet node within :paramref:`capture_radius` of the
    point between them. That is what a grasp physically is, and it is the only definition that
    cannot disagree with what the simulation actually did.

    The earlier version tested a pose instead -- distance to the edge's centre, fingertips above the
    edge, hand within 32 degrees of straight down, closing axis within 32 degrees of the slot's
    lateral axis -- and every one of those had to hold at the exact instant of closure. A hand-flown
    grasp that took the sheet and lifted it 27 cm out of the slot was rejected by that test, which
    then blocked everything downstream: no hold, so no lift shaping, no extraction bonus, and no
    success termination for an episode that had plainly succeeded. Judging the outcome instead of
    the posture removes that whole class of failure. The posture is still *encouraged*, by the
    ``alignment`` shaping term -- it just no longer has a veto.

    Three things are paid:

    * :paramref:`first_grasp_bonus`, once, the first time the sheet is held **by its top edge** --
      a node on the highest border is the one between the fingers.
    * :paramref:`bad_closure_penalty`, whenever the gripper completes a closure with no cloth
      between the pads at all, so snapping shut in mid-air is never free.
    * :paramref:`extraction_bonus`, once the whole sheet is above :paramref:`slot_clear_height`
      while held. Letting go before then costs :paramref:`premature_release_penalty`.

    Note:
        Bonuses are divided by ``step_dt`` so the configured magnitudes are the literal one-shot
        payouts rather than being scaled down by the reward manager's ``weight * dt``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._sheet: DeformableObject = env.scene[params["asset_cfg"].name]
        self._robot: Articulation = env.scene[params["robot_cfg"].name]

        names = self._robot.body_names
        self._hand_id = names.index(params["hand_body_name"])
        self._left_id = names.index(params["left_finger_body_name"])
        self._right_id = names.index(params["right_finger_body_name"])

        # the four borders of the node grid, as index rows. Which one is "the top" depends on how
        # the sheet was stood up and on how it has since deformed, so it is chosen per step by
        # height rather than fixed here.
        zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._was_open = zeros.clone()
        self._holding = zeros.clone()
        self._grasped = zeros.clone()

        # per-episode tallies of each discrete event, logged on reset. Integer counters on the GPU:
        # incrementing them is a handful of elementwise adds per step and never syncs to the host,
        # so the cost is invisible next to the cloth solver.
        self._counts = {
            name: torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
            for name in ("grasp", "bad_closure", "early_release", "extraction", "closures")
        }
        # Steps spent with the gripper open, over steps lived. Zero attempts is ambiguous on its
        # own -- a gripper welded shut never completes an open-then-close cycle either -- and the
        # two cases need opposite fixes, so the fraction is logged to tell them apart.
        self._steps = torch.zeros(env.num_envs, device=env.device)
        self._open_steps = torch.zeros(env.num_envs, device=env.device)
        # Running means of the three things a valid grasp needs, logged on reset. The reward terms
        # bundle them, so a low ``alignment`` cannot say whether the gripper is in the wrong place,
        # tilted wrong, or rotated wrong -- these separate the three.
        self._gauges = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in ("dist_to_edge", "facing_down", "square_to_slot")
        }
        # The same four quantities, sampled only at the instant a closure completes. Episode means
        # cannot diagnose a rejected grasp: they average the whole approach, so a closure that was
        # perfect in three conditions and off in one looks like a mediocre episode throughout.
        # These say exactly which condition failed on the closure that mattered.
        self._at_closure = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in ("dist", "height_offset", "facing_down", "square_to_slot")
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._was_open[env_ids] = False
        self._holding[env_ids] = False
        self._grasped[env_ids] = False
        _is_holding(self._env)[env_ids] = False
        _early_release(self._env)[env_ids] = False
        # drop the environment back to phase one
        _phase_reached(self._env)[env_ids] = False

        # Publish how often each event fired in the episode that just ended, then clear the tally.
        # Written straight into the environment's log dict: the reward manager discards whatever a
        # term's reset() returns, and this runs after the environment has recreated that dict, so
        # the keys survive to the end of the step.
        extras = self._env.extras.setdefault("log", {})
        for name, counter in self._counts.items():
            extras[f"Events/{name}"] = counter[env_ids].float().mean()
            counter[env_ids] = 0
        lived = self._steps[env_ids].clamp(min=1.0)
        extras["Events/open_frac"] = (self._open_steps[env_ids] / lived).mean()
        for name, gauge in self._gauges.items():
            extras[f"Events/{name}"] = (gauge[env_ids] / lived).mean()
            gauge[env_ids] = 0.0
        # averaged over closures rather than steps, since that is when they were sampled
        closed = self._counts["closures"][env_ids].float().clamp(min=1.0)
        for name, gauge in self._at_closure.items():
            extras[f"AtClosure/{name}"] = (gauge[env_ids] / closed).mean()
            gauge[env_ids] = 0.0
        self._steps[env_ids] = 0.0
        self._open_steps[env_ids] = 0.0

    def _grasp_frame(
        self, grasp_offset: tuple[float, float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The point between the fingertips, and the hand's orientation."""
        pose = self._robot.data.body_link_pose_w.torch[:, self._hand_id]
        offset = torch.tensor(grasp_offset, device=pose.device).expand(len(pose), 3)
        return pose[:, :3] + quat_apply(pose[:, 3:7], offset), pose[:, 3:7]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        finger_cfg: SceneEntityCfg,
        slot_cfg: SceneEntityCfg,
        resolution: tuple[int, int],
        hand_body_name: str = "panda_hand",
        left_finger_body_name: str = "panda_leftfinger",
        right_finger_body_name: str = "panda_rightfinger",
        grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
        capture_radius: float = 0.04,
        open_threshold: float = 0.030,
        closed_threshold: float = 0.012,
        first_grasp_bonus: float = 200.0,
        bad_closure_penalty: float = 20.0,
        premature_release_penalty: float = 500.0,
        slot_clear_height: float = 0.17,
        extraction_bonus: float = 1000.0,
    ) -> torch.Tensor:
        nodes = self._sheet.data.nodal_pos_w.torch
        tip, hand_quat = self._grasp_frame(grasp_offset)

        # how close the nearest bit of cloth is to the point between the fingertips -- anywhere on
        # the sheet, and restricted to its highest border
        top_nodes = _top_edge_nodes(nodes, resolution)
        center = top_nodes.mean(dim=1)
        sheet_dist = (nodes - tip.unsqueeze(1)).norm(dim=-1).min(dim=1).values
        top_dist = (top_nodes - tip.unsqueeze(1)).norm(dim=-1).min(dim=1).values

        # read the joint ids at call time: the manager resolves SceneEntityCfg indices after the
        # term is constructed, so anything captured in __init__ is still the unresolved slice(None)
        width = self._robot.data.joint_pos.torch[:, finger_cfg.joint_ids].mean(dim=1)
        is_open = width > open_threshold
        is_closed = width < closed_threshold

        # -- the whole definition of a grasp: shut, with cloth between the pads
        holding = is_closed & (sheet_dist < capture_radius)
        hit = holding & (top_dist < capture_radius) & ~self._grasped

        closed_now = self._was_open & is_closed
        # closing on nothing at all. Judged on the same capture test, so a closure that caught the
        # sheet is never charged, whatever posture it was made in.
        bad_closure = closed_now & (sheet_dist >= capture_radius)
        reward = first_grasp_bonus * hit.float() - bad_closure_penalty * bad_closure.float()

        phase = _phase_reached(env)
        # letting go before the sheet is out of the slot; the matching termination reads this flag
        early = self._holding & is_open & ~phase
        reward = reward - premature_release_penalty * early.float()
        _early_release(env).copy_(early)

        # -- pose diagnostics. Logged only: these describe how the grasp was made, and after a
        # hand-flown grasp was rejected for being two centimetres low they no longer gate anything.
        unit_z = torch.tensor([0.0, 0.0, 1.0], device=nodes.device).expand(len(nodes), 3)
        approach_z = quat_apply(hand_quat, unit_z)[:, 2]
        downward = 0.5 * (1.0 - approach_z)
        slot = env.scene[slot_cfg.name]
        squareness = _closing_squareness(self._robot, slot, self._left_id, self._right_id)
        dist_to_edge = (tip - center).norm(dim=-1)

        # -- latch state for the next step
        self._grasped |= hit
        self._holding = holding
        self._was_open = torch.where(is_open, True, torch.where(is_closed, False, self._was_open))
        _is_holding(env).copy_(holding)

        # -- phase one: pull the sheet clear of the slot.
        #
        # Measured on the sheet's *lowest* node, so the whole sheet has to be above the walls
        # rather than merely tilted or part-way up. Heights are taken relative to the environment
        # origin, which is where the slot geometry is defined.
        lowest = nodes[..., 2].min(dim=1).values - env.scene.env_origins[:, 2]
        # keyed on the observed hold, so a sheet that is demonstrably out of the slot and in the
        # gripper always counts, whether or not the grasp bonus happened to be awarded
        extracted = holding & ~phase & (lowest > slot_clear_height)
        reward = reward + extraction_bonus * extracted.float()
        phase |= extracted

        self._steps += 1.0
        self._open_steps += is_open.float()
        self._gauges["dist_to_edge"] += dist_to_edge
        # rescaled to 0..1 the same way the alignment reward does, so the two read alike
        downward = 0.5 * (1.0 - approach_z)
        self._gauges["facing_down"] += downward
        self._gauges["square_to_slot"] += squareness

        sampled = closed_now.float()
        self._at_closure["dist"] += sheet_dist * sampled
        # signed: negative means the fingertips were below the top of the edge, which is exactly
        # what a real grasp looks like and what the old height gate wrongly rejected
        self._at_closure["height_offset"] += (tip[:, 2] - center[:, 2]) * sampled
        self._at_closure["facing_down"] += downward * sampled
        self._at_closure["square_to_slot"] += squareness * sampled
        grasp_debug(env).update(
            tip_x=tip[:, 0],
            tip_y=tip[:, 1],
            tip_z=tip[:, 2],
            edge_z=center[:, 2],
            height_offset=tip[:, 2] - center[:, 2],
            width=width,
            is_open=is_open,
            is_closed=is_closed,
            closed_now=closed_now,
            sheet_dist=sheet_dist,
            top_dist=top_dist,
            dist_to_edge=dist_to_edge,
            facing_down=downward,
            square_to_slot=squareness,
            holding=holding,
            hit=hit,
            bad_closure=bad_closure,
            early_release=early,
            extracted=extracted,
            lowest_z=lowest,
        )

        self._counts["closures"] += closed_now
        self._counts["grasp"] += hit
        self._counts["bad_closure"] += bad_closure
        self._counts["early_release"] += early
        self._counts["extraction"] += extracted

        # undo the manager's dt scaling so the configured magnitudes are the real one-shot returns
        # rather than being silently divided by the 30 Hz timestep
        return reward / env.step_dt
