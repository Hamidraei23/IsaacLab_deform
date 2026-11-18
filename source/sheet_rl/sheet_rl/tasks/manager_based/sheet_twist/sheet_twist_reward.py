"""The sheet platform twist reward: pays, stage by stage, for behaving as the scripted expert does.

Everything is computed from what the student observes of the task -- the sheet's nine landmarks, the
hand, the gripper and the platform pose -- plus the task's own hold and extraction flags. The stages
follow ``ScriptedSheetExpert`` in ``scripts/train_dagger_sheet_twist.py``, cloth and rigid plate alike:

=========  ===============================================================================  ======
stage      term                                                                             weight
=========  ===============================================================================  ======
approach   ``reach``: 1 - tanh(d / 0.1), tool to the grasp point 2 cm under the top edge        1
           ``align``: hand pointing down x fingers closing across the sheet, near the edge       1
hold       ``hold``: the task's detector reports a hold (before extraction)                      2
lift       ``lift_goal`` / ``_fine``: held, the top edge to ``GOAL_POINT`` above the slot       16 / 5
           (1 - tanh(d / 0.3) and / 0.05, as the cube task's goal terms)
           ``extraction``: the detector's one-off +10 first grasp, +200 extraction               1
place      ``layout`` / ``_fine``: the nine landmarks to their places lying flat on the        20 / 10
           platform, top edge on ``PLACE_EDGE``, bottom edge on the opposite one
           (1 - tanh(mean distance / 0.15) and / 0.03); the drape is rewarded on the way down
release    ``released_coverage``: coverage, only with the sheet let go                          30
           ``completion``: once, released, coverage >= 0.5, sheet still: 300 x coverage          1
always     ``terminated``: an episode ending early (hand below the table, sheet out of           -600
           bounds or dropped) -- literal, so ending early never beats carrying on
           ``gripper_closures``: 25 per commanded re-closure before extraction (first is free)     -25
           ``ee_speed`` (above 0.2 m/s), ``approach_speed`` (near the edge, before the hold),  -10 / -40
           ``yaw_rate`` (after extraction), ``joint_vel``, ``action_rate``                  -2/-0.2/-0.2
=========  ===============================================================================  ======

Weights are per second (the reward manager multiplies by the 0.02 s step) except the one-off terms,
which are literal. Phase-1 terms (approach, hold, lift) pay their maximum once the sheet is extracted
**for as long as the sheet stays held or on the platform** (``_phase_one_ceiling``): finishing phase 1
never costs anything, but dropping the sheet afterwards forfeits the kept pay. Phase 2 adds up to 30
for the layout and 30 x coverage once the sheet is let go. Ending the episode early costs 600, more
than extraction can still guarantee.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.math import matrix_from_quat, quat_apply

from ..franka_lift_twist.platform_mdp import _hand_speed, ee_speed_above
from ..sheet_rl.mdp.grasp_progress import gripper_closure_penalty
from ..sheet_rl.mdp.rewards import _is_holding, _phase_reached
from ..sheet_rl.sheet_rl_env_cfg import SHEET_RESOLUTION, SHEET_SIZE
from .sheet_twist_mdp import PLATFORM_DEPTH, fingers_open

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GRASP_OFFSET = (0.0, 0.0, 0.1034)
PICK_DEPTH = 0.02
GOAL_HEIGHT = 0.42
"""Phase-1 goal for the top edge: right above the slot, high enough that the sheet clears the walls [m]."""
PLACE_EDGE = "right"
"""Platform edge the top edge is laid on (named from the slot, as in the expert); the best for coverage."""
EDGES = ("near", "far", "left", "right")
LAYOUT_CLEARANCE = 0.003
FINGERS = SceneEntityCfg("robot", joint_names="panda_finger_joint.*")


# -- what the reward sees
def landmarks_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The nine sheet landmarks in the world frame, (num_envs, 9, 3), same indices for cloth and plate:
    bottom row 0-2, middle 3-5, top 6-8 (7 = middle of the top edge, 4 = centre)."""
    if "plate" in env.scene.keys():
        from .rigid_mdp import plate_points_w

        return plate_points_w(env)
    cols, rows = SHEET_RESOLUTION[0] + 1, SHEET_RESOLUTION[1] + 1
    picks = [(r, c) for r in (0, rows // 2, rows - 1) for c in (0, cols // 2, cols - 1)]
    ids = torch.tensor([r * cols + c for r, c in picks], device=env.device)
    return env.scene["deformable"].data.nodal_pos_w.torch[:, ids]


def _tool_w(env) -> torch.Tensor:
    robot = env.scene["robot"]
    hand = robot.data.body_link_pose_w.torch[:, robot.body_names.index("panda_hand")]
    return hand[:, :3] + quat_apply(hand[:, 3:7], torch.tensor(GRASP_OFFSET, device=env.device).expand(env.num_envs, 3))


def _grasp_point_w(env) -> torch.Tensor:
    point = landmarks_w(env)[:, 7].clone()
    point[:, 2] -= PICK_DEPTH
    return point


def _slot_centre_w(env) -> torch.Tensor:
    return 0.5 * (env.scene["slot_neg_y"].data.root_pos_w.torch + env.scene["slot_pos_y"].data.root_pos_w.torch)


def _slot_normal_w(env) -> torch.Tensor:
    normal = env.scene["slot_pos_y"].data.root_pos_w.torch - env.scene["slot_neg_y"].data.root_pos_w.torch
    normal[:, 2] = 0.0
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def goal_point_w(env) -> torch.Tensor:
    goal = _slot_centre_w(env).clone()
    goal[:, 2] = env.scene.env_origins[:, 2] + GOAL_HEIGHT
    return goal


def place_frames_w(env, edge: str = PLACE_EDGE):
    """Platform top: midpoints of the target edge and of the opposite edge, and the unit direction along
    the target edge. Edges are named from the slot (near/far: closest/farthest; left/right: the other two
    as seen from the slot), exactly as the expert names them."""
    platform = env.scene["mannequin_arm"]
    n = env.num_envs
    centre, quat = platform.data.root_pos_w.torch, platform.data.root_quat_w.torch
    hx, hy = SHEET_SIZE[0] / 2, SHEET_SIZE[1] / 2
    local = torch.tensor([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], device=env.device)
    offsets = quat_apply(quat[:, None].expand(-1, 4, -1).reshape(-1, 4), local.repeat(n, 1)).reshape(n, 4, 3)
    top_z = centre[:, 2] + PLATFORM_DEPTH / 2
    mids = centre[:, None] + offsets
    mids[..., 2] = top_z[:, None]
    slot = _slot_centre_w(env)
    distance = (mids[..., :2] - slot[:, None, :2]).norm(dim=-1)
    order = distance.argsort(-1)  # near ... far
    near, far = order[:, 0], order[:, 3]
    towards = centre[:, :2] - slot[:, :2]
    cross = towards[:, None, 0] * offsets[..., 1] - towards[:, None, 1] * offsets[..., 0]
    middle = order[:, 1:3]
    first_left = cross.gather(1, middle[:, :1]).squeeze(-1) > 0
    left = torch.where(first_left, middle[:, 0], middle[:, 1])
    right = torch.where(first_left, middle[:, 1], middle[:, 0])
    choice = {"near": near, "far": far, "left": left, "right": right}[edge]
    opposite = {"near": far, "far": near, "left": right, "right": left}[edge]
    rows = torch.arange(n, device=env.device)
    target, back = mids[rows, choice], mids[rows, opposite]
    along = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3), target - back, dim=-1)
    along = along / along.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return target, back, along


def layout_targets_w(env, edge: str = PLACE_EDGE) -> torch.Tensor:
    """Where the nine landmarks are with the sheet lying flat on the platform, its top edge (6-8) on the
    target edge and bottom edge (0-2) on the opposite one. Shape (num_envs, 2, 9, 3): both mirror images
    (columns reversed along the edge), since either is a correct lay."""
    target, back, along = place_frames_w(env, edge)
    across = (target - back) / 2  # half the sheet, from a row to the next
    points = []
    for sign in (1.0, -1.0):
        rows = []
        for r in range(3):
            for c in range(3):
                p = back + r * across + sign * (c - 1) * along * (SHEET_SIZE[0] / 2)
                rows.append(p)
        points.append(torch.stack(rows, dim=1))
    layout = torch.stack(points, dim=1)
    layout[..., 2] += LAYOUT_CLEARANCE
    return layout


def _phase_one_ceiling(env, value):
    """After extraction, hold a phase-1 term at its maximum -- but only while the job is still intact:
    the sheet held, or lying on the platform (coverage above 5%). PPO farmed the earlier unconditional
    version: extraction alone guaranteed ~25/s for the rest of the episode, so dropping the sheet cost
    almost nothing and placing it well earned little more. Now a drop forfeits the annuity."""
    coverage = env.reward_manager.get_term_cfg("coverage").func.coverage
    engaged = _is_holding(env) | (coverage > 0.05)
    return torch.where(_phase_reached(env), engaged.float(), value)


# -- approach and hold
def reach(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    value = 1 - torch.tanh((_tool_w(env) - _grasp_point_w(env)).norm(dim=-1) / std)
    return _phase_one_ceiling(env, value)


def align(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Hand pointing down and fingers closing across the sheet, weighted by nearness to the grasp point."""
    robot = env.scene["robot"]
    axes = matrix_from_quat(robot.data.body_link_pose_w.torch[:, robot.body_names.index("panda_hand"), 3:7])
    down = (-axes[:, 2, 2]).clamp(0.0, 1.0)
    across = (axes[:, :, 1] * _slot_normal_w(env)).sum(-1).abs()
    near = 1 - torch.tanh((_tool_w(env) - _grasp_point_w(env)).norm(dim=-1) / std)
    value = holding_or(env, near * down * across)
    return _phase_one_ceiling(env, value)


def holding_or(env, value):
    return torch.where(_is_holding(env), 1.0, value)


def hold(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _phase_one_ceiling(env, _is_holding(env).float())


# -- lift to the goal above the slot
def lift_goal(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    distance = (landmarks_w(env)[:, 7] - goal_point_w(env)).norm(dim=-1)
    value = _is_holding(env).float() * (1 - torch.tanh(distance / std))
    return _phase_one_ceiling(env, value)


# -- place: the landmarks into their lying-flat layout on the platform
def layout_match(env: ManagerBasedRLEnv, std: float, edge: str = PLACE_EDGE) -> torch.Tensor:
    error = (landmarks_w(env)[:, None] - layout_targets_w(env, edge)).norm(dim=-1).mean(-1).amin(-1)
    return _phase_reached(env).float() * (1 - torch.tanh(error / std))


# -- release
def _released(env, finger_cfg) -> torch.Tensor:
    return _phase_reached(env) & ~_is_holding(env) & fingers_open(env, finger_cfg)


def released_coverage(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg = FINGERS) -> torch.Tensor:
    """Coverage, paid only with the sheet let go. Reads the ``coverage`` term computed earlier this step."""
    return _released(env, finger_cfg).float() * env.reward_manager.get_term_cfg("coverage").func.coverage


class completion_bonus(ManagerTermBase):
    """Once per episode: released, covering at least ``min_coverage``, no landmark faster than
    ``still_speed``. Pays ``bonus x coverage`` (literal)."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous = None

    def reset(self, env_ids=None):
        self.paid[slice(None) if env_ids is None else env_ids] = False
        self.previous = None

    def __call__(self, env, bonus: float, min_coverage: float = 0.5, still_speed: float = 0.03,
                 finger_cfg: SceneEntityCfg = FINGERS) -> torch.Tensor:
        points = landmarks_w(env)
        speed = torch.zeros(env.num_envs, device=env.device)
        if self.previous is not None:
            speed = ((points - self.previous).norm(dim=-1) / env.step_dt).amax(-1)
        self.previous = points.clone()
        coverage = env.reward_manager.get_term_cfg("coverage").func.coverage
        done = _released(env, finger_cfg) & (coverage >= min_coverage) & (speed < still_speed) & ~self.paid
        self.paid |= done
        return done.float() * bonus * coverage / env.step_dt


# -- charges
def terminated_early(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 / dt on a step that ends the episode before its time limit, so the weight is literal."""
    return mdp.is_terminated(env) / env.step_dt


def approach_speed(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    approaching = ~_is_holding(env) & ~_phase_reached(env)
    near = 1 - torch.tanh((_tool_w(env) - _grasp_point_w(env)).norm(dim=-1) / std)
    return approaching.float() * near * _hand_speed(env, SceneEntityCfg("robot"), "panda_hand").square()


def yaw_rate_after_extraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    yaw_rate = robot.data.body_link_vel_w.torch[:, robot.body_names.index("panda_hand"), 5]
    return _phase_reached(env).float() * yaw_rate.square()


def apply_staged_reward(cfg) -> None:
    """Install the reward on a twist env config whose ``grasp_stage`` (detector) and ``coverage`` terms are
    set. Coverage stays as a term for its value and ``Coverage/*`` logs, at a weight too small to matter
    (a zero weight would stop the manager computing it). Terms of an earlier version are cleared."""
    for name in ("lift", "place_tracking"):
        setattr(cfg.rewards, name, None)
    cfg.rewards.grasp_stage.weight = 1.0
    # +50, not the sheet task's +200: extraction opens the gate to ~35/s more of dense pay, which is
    # incentive enough. A +200 spike at a mid-episode moment the critic cannot pinpoint dominated the
    # normalized advantages under PPO -- a few outlier samples per batch drove the updates while the
    # adaptive KL schedule pinned the learning rate, which is how v3 collapsed at lr ~1e-5.
    cfg.rewards.grasp_stage.params["extraction_bonus"] = 50.0
    cfg.rewards.coverage.weight = 1e-3
    cfg.rewards.reach = RewTerm(func=reach, weight=1.0, params={"std": 0.1})
    cfg.rewards.align = RewTerm(func=align, weight=1.0, params={"std": 0.1})
    cfg.rewards.hold = RewTerm(func=hold, weight=2.0)
    cfg.rewards.lift_goal = RewTerm(func=lift_goal, weight=16.0, params={"std": 0.3})
    cfg.rewards.lift_goal_fine = RewTerm(func=lift_goal, weight=5.0, params={"std": 0.05})
    cfg.rewards.layout = RewTerm(func=layout_match, weight=20.0, params={"std": 0.15})
    cfg.rewards.layout_fine = RewTerm(func=layout_match, weight=10.0, params={"std": 0.03})
    cfg.rewards.released_coverage = RewTerm(func=released_coverage, weight=30.0, params={"finger_cfg": FINGERS})
    cfg.rewards.completion = RewTerm(func=completion_bonus, weight=1.0, params={"bonus": 300.0, "finger_cfg": FINGERS})
    # above everything extraction can still guarantee, so ending the episode early is never the cheap way out
    cfg.rewards.terminated = RewTerm(func=terminated_early, weight=-600.0)
    cfg.rewards.ee_speed = RewTerm(func=ee_speed_above, weight=-10.0, params={"free_speed": 0.2})
    cfg.rewards.approach_speed = RewTerm(func=approach_speed, weight=-40.0, params={"std": 0.1})
    cfg.rewards.yaw_rate = RewTerm(func=yaw_rate_after_extraction, weight=-2.0)
    # The one unpriced channel was the binary gripper, and PPO degraded through it twice: advantage
    # noise walked the gripper mean toward the threshold, closures rose to 3-5 per episode at up to
    # 23 cm from the edge, and the grasp rate fell (v3 and v4 both show closures and closure distance
    # rising before the reward slid). 25 per re-closure prices decisiveness; one closure stays free.
    cfg.rewards.gripper_closures = RewTerm(
        func=gripper_closure_penalty, weight=1.0, params={"penalty": 25.0, "free_closures": 1}
    )
    cfg.rewards.joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot")})
    cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
