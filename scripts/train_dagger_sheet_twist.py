"""Distil the cube twist pick-and-place teacher into a sheet twist student, by DAgger.

The student acts in ``Template-Sheet-Rl-Platform-Twist-v0`` (see ``sheet_twist_env_cfg``) with the sheet
task's own observations (joints, nine sheet landmarks, the platform-top goal, applied actions) and the
cube task's twist action. The teacher is a ``Lift-Cube-Franka-Twist-PickPlace-v0`` checkpoint. It is run
on a cube-task view of the sheet scene, and its answer is completed by two scripted parts:

- **The teacher's view** (35 values, its training layout): the joints relative to the cube task's
  default pose; as the "cube", the middle node of the sheet's top edge, ``PICK_DEPTH`` below it so the
  pads close on the top 2 cm of cloth; as the goal, ``GOAL_HEIGHT`` above the slot until the sheet is
  extracted (the task's own detector), then the midpoint of one platform edge, ``PLACE_HEIGHT`` above
  the platform top; and its own previous action.
- **Translation and, during phase 1, the gripper**: the teacher's.
- **Rotation**: an orientation controller, because the teacher never observed orientation and a sheet
  edge can only be pinched with the fingers closing across it. It keeps the hand pointing down with the
  closing axis along the slot normal, and after extraction turns the held edge parallel to the target
  platform edge, so the sheet is laid along it.
- **Release**: scripted. In phase 2 the gripper is held shut until the held point is within
  ``RELEASE_RADIUS`` of its target, then opened for good, and the hand retreats ``RETREAT_HEIGHT`` up.

``--eval_edges`` runs that combined teacher alone (no student) with the four placement edges spread
across the environments, and reports extraction, release and coverage per edge. The edges are named
relative to the slot: ``near`` and ``far`` are the platform edges closest to and farthest from it,
``left`` and ``right`` the other two as seen from the slot.

Without it, DAgger trains a student on the combined teacher's actions for ``--edge``: the arm runs the
teacher's action with probability beta, else the student's, beta falling from 1 to 0 over
``--beta_iters``; every (observation, action) pair is kept in an aggregated buffer; the actor is
regressed on it with a per-dimension normalised MSE, and the critic on the rollouts' lambda-returns of
the task's coverage reward. Checkpoints are RSL-RL ``model_<it>.pt`` files under
``logs/rsl_rl/sheet_platform_twist/<run_name>``.

Examples (from the sheet_rl project)::

    python scripts/train_dagger_sheet_twist.py --eval_edges --num_envs 256 \\
        --teacher logs/rsl_rl/lift_cube_franka_twist/twist_pick_place_v4/model_5900.pt
    python scripts/train_dagger_sheet_twist.py --edge far --run_name sheet_dagger_v1 \\
        --teacher logs/rsl_rl/lift_cube_franka_twist/twist_pick_place_v4/model_5900.pt
"""

import argparse
import importlib.metadata as metadata
import json
import math
import sys
import time
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env, write_run_manifest
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import sheet_rl.tasks  # noqa: F401

EDGES = ("near", "far", "left", "right")
PICK_DEPTH = 0.02
"""How far below the top edge the teacher's "cube" is [m]: the pads close on the top 2 cm of cloth."""
GOAL_HEIGHT = 0.42
"""Phase-1 goal height above the table [m]: the held point there lifts the sheet's lowest node past the
17 cm clearance (15 cm walls + 2 cm) with margin. Inside the cube teacher's 25-50 cm goal range."""
PLACE_HEIGHT = 0.02
"""Phase-2 goal for the held point above the platform top [m], as a resting cube's centre was 2.4 cm."""
RELEASE_RADIUS = 0.03
"""Distance of the held point from its placement target at which the gripper is opened [m]."""
RETREAT_HEIGHT = 0.15
"""How far above the release point the hand retreats after letting go [m]."""
ROTATION_GAIN = 0.5
"""Share of the orientation error commanded per step (the damped IK arm covers a fraction of it)."""
MAX_ROTATION_STEP = 0.3
MAX_RETREAT_STEP = 0.05

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Template-Sheet-Rl-Platform-Twist-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--expert", choices=("scripted", "teacher"), default="scripted",
                    help="scripted: the waypoint twist expert (ScriptedSheetExpert). teacher: the cube twist network.")
parser.add_argument("--teacher", type=Path, help="Cube twist pick-and-place checkpoint (.pt), for --expert teacher.")
parser.add_argument("--edge", choices=EDGES, default="far", help="Platform edge the held edge is placed on.")
parser.add_argument("--eval_edges", action="store_true", help="Run the teacher alone on all four edges and report.")
parser.add_argument("--eval_student", type=Path, default=None,
                    help="Evaluate this student checkpoint (deterministic) on the same measures as --eval_edges.")
parser.add_argument("--stochastic", action="store_true",
                    help="With --eval_student: sample actions from the policy's own distribution, as PPO does.")
parser.add_argument("--eval_episodes", type=int, default=1, help="Episodes per environment with --eval_edges.")
parser.add_argument(
    "--rotation",
    choices=("teacher", "yaw", "controller"),
    default="yaw",
    help="teacher: the teacher's rotation. yaw: the teacher's, plus a yaw correction turning the closing axis "
    "across the sheet (then along the target edge). controller: hand straight down, closing axis set (breaks it).",
)
parser.add_argument("--governor", type=float, default=1.0,
                    help="Cap the teacher's translation step at this multiple of the distance left to its current "
                    "target (pick point, air goal, placement point); <=0 off. The cube stopped its pushes; the sheet does not.")
parser.add_argument("--approach_floor", type=float, default=0.01,
                    help="Until the sheet is held, never command the tool more than this below the pick point [m]; <0 off.")
parser.add_argument("--exact_reset", action="store_true", help="Start every episode at the default joint pose.")
parser.add_argument("--virtual_cube", type=str, default=None,
                    help="Diagnostic: 'x,y,z' in the robot base frame, shown to the teacher as the cube instead of the sheet.")
parser.add_argument("--plot", action="store_true",
                    help="With --eval_edges/--eval_student: live plot of env 0's reward terms; one PNG per episode.")
parser.add_argument("--plot_every", type=int, default=10, help="Control steps between plot refreshes.")
parser.add_argument("--trace", type=int, default=0, help="With --eval_edges: print env 0's state every N steps.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--beta_iters", type=int, default=100)
parser.add_argument("--buffer_size", type=int, default=1_000_000)
parser.add_argument("--updates_per_iter", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--student_std", type=float, default=0.05)
parser.add_argument("--gru", default=None, metavar="HIDDEN,LAYERS",
                    help="Recurrent actor: a GRU of this size feeding the MLP head. Trained by truncated "
                    "BPTT over each rollout window (the flat replay buffer would break recurrence); the "
                    "critic stays an MLP. Pass the same flag to --eval_student for such a checkpoint.")
parser.add_argument("--gru_epochs", type=int, default=6, help="BPTT passes over each window (GRU only).")
parser.add_argument("--init_student", type=Path, default=None,
                    help="Start the student (actor and critic) from this checkpoint, e.g. a rigid-plate student.")
parser.add_argument("--beta_start", type=float, default=1.0, help="Beta at iteration 0 (falls linearly to 0).")
parser.add_argument("--beta_floor", type=float, default=0.0,
                    help="Beta never falls below this. Required for a recurrent student: its BPTT trains on "
                    "fresh windows only (no replay buffer), so without a steady share of expert-driven "
                    "episodes it forgets the demonstrations once it drives alone and collapses.")
parser.add_argument("--mix", choices=("step", "episode"), default="step",
                    help="Who drives, drawn with probability beta: per step, or per episode (the student then drives "
                    "whole episodes and sees its own drift).")
parser.add_argument("--expert_noise", type=float, default=0.0,
                    help="Std of Gaussian noise added to the expert's executed arm action (raw units); the label stays clean.")
parser.add_argument("--expert_variant", choices=("normal", "hover", "drop"), default="normal",
                    help="Reward checks: hover never lets go; drop lets go as soon as the sheet is extracted.")
parser.add_argument("--tilt_angle", type=float, default=60.0, help="Rigid plate: wrist tilt at release [deg].")
parser.add_argument("--pivot_clearance", type=float, default=0.005,
                    help="Rigid plate: keep its bottom edge this far above the platform while tilting [m]; "
                    "negative presses it in (which flung plates off the table).")
parser.add_argument("--pivot_inset", type=float, default=0.0,
                    help="Rigid plate: put the bottom-edge pivot this far inside the opposite platform edge [m].")
parser.add_argument("--student_noise", type=float, default=0.0,
                    help="Std of Gaussian noise on the student's executed arm action (raw units), so it practises "
                    "recovering from imperfect motion; the expert still labels the state it lands in.")
parser.add_argument("--expert_gain", type=float, default=1.0,
                    help="Proportional gain of the scripted expert's step toward its waypoint (before the cap).")
parser.add_argument("--save_interval", type=int, default=50)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining


def load_teacher(path: Path, device: str) -> torch.nn.Module:
    """Rebuild the teacher's actor MLP from its checkpoint (no observation normalisation expected)."""
    from rsl_rl.modules import MLP

    state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    if any(key.startswith("obs_normalizer.") for key in state):
        raise ValueError(f"Teacher {path} normalises its observations; this script expects it not to.")
    weights = [state[k] for k in sorted((k for k in state if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))]
    mlp = MLP(weights[0].shape[1], weights[-1].shape[0], [w.shape[0] for w in weights[:-1]], "elu").to(device)
    mlp.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")})
    mlp.eval()
    for parameter in mlp.parameters():
        parameter.requires_grad_(False)
    return mlp


class SheetTeacher:
    """The cube teacher on a cube-task view of the sheet scene, plus scripted rotation and release."""

    def __init__(self, env, teacher: torch.nn.Module, edges: torch.Tensor):
        from isaaclab.utils.string import resolve_matching_names_values

        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

        from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _border_ids
        from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env_cfg import SHEET_RESOLUTION, SHEET_SIZE
        from sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_mdp import PLATFORM_DEPTH

        u = env.unwrapped
        self.u, self.teacher, self.device = u, teacher, u.device
        self.rigid = "plate" in u.scene.keys()
        self.robot = u.scene["robot"]
        self.sheet = u.scene["plate"] if self.rigid else u.scene["deformable"]
        self.platform = u.scene["mannequin_arm"]
        self.walls = (u.scene["slot_neg_y"], u.scene["slot_pos_y"])
        self.arm = u.action_manager.get_term("arm_action")
        self.ik_scale = self.arm._scale[:, :3]
        self.platform_half = (SHEET_SIZE[0] / 2, SHEET_SIZE[1] / 2)
        # from the scene, not a constant: the platform is a 20 cm sunk box in the twist tasks but the
        # original 2 cm slab in the OrigReward bridge, and a hardcoded half-depth put the expert's lay
        # 9 cm in the air there (it never met its release condition)
        self.platform_top_offset = float(u.cfg.scene.mannequin_arm.spawn.size[2]) / 2
        ids, _, values = resolve_matching_names_values(FRANKA_PANDA_CFG.init_state.joint_pos, self.robot.joint_names)
        self.teacher_default = torch.zeros(len(self.robot.joint_names), device=self.device)
        self.teacher_default[ids] = torch.tensor(values, device=self.device)
        self.borders = _border_ids(SHEET_RESOLUTION, self.device)  # (4, nodes per border)
        n = u.num_envs
        self.edges = edges.to(self.device)  # index into EDGES per env
        self.top_border = torch.zeros(n, dtype=torch.long, device=self.device)
        self.previous = torch.zeros(n, 7, device=self.device)
        self.released = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.retreat_target = torch.zeros(n, 3, device=self.device)
        self.extracted_ever = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.held_ever = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.raw = torch.zeros(n, 7, device=self.device)
        self.reset(torch.arange(n, device=self.device))

    # -- frames
    def _to_base(self, points_w):
        from isaaclab.utils.math import subtract_frame_transforms

        return subtract_frame_transforms(self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, points_w)[0]

    def _dir_to_base(self, vectors_w):
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.robot.data.root_quat_w.torch, vectors_w)

    # -- scene quantities, world frame
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self.rigid:
            nodes = self.sheet.data.nodal_pos_w.torch[env_ids]
            heights = nodes[:, self.borders, 2].mean(-1)  # (k, 4)
            self.top_border[env_ids] = heights.argmax(-1)
        self.previous[env_ids] = 0.0
        self.released[env_ids] = False
        self.extracted_ever[env_ids] = False
        self.held_ever[env_ids] = False

    def pick_point_w(self):
        if args.virtual_cube:
            from isaaclab.utils.math import combine_frame_transforms

            point_b = torch.tensor([float(v) for v in args.virtual_cube.split(",")], device=self.device).expand(self.u.num_envs, 3)
            return combine_frame_transforms(self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch, point_b)[0]
        if self.rigid:
            from sheet_rl.tasks.manager_based.sheet_twist.rigid_mdp import plate_points_w

            point = plate_points_w(self.u)[:, 7].clone()  # middle of the top edge, as cloth landmark 7
            point[:, 2] -= PICK_DEPTH
            return point
        nodes = self.sheet.data.nodal_pos_w.torch
        border = self.borders[self.top_border]  # (n, k)
        middle = border[:, border.shape[1] // 2]
        point = nodes[torch.arange(len(nodes), device=self.device), middle].clone()
        point[:, 2] -= PICK_DEPTH
        return point

    def slot_centre_w(self):
        return 0.5 * (self.walls[0].data.root_pos_w.torch + self.walls[1].data.root_pos_w.torch)

    def slot_normal_w(self):
        normal = self.walls[1].data.root_pos_w.torch - self.walls[0].data.root_pos_w.torch
        normal[:, 2] = 0.0
        return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def edge_frames_w(self):
        """Midpoint (top surface) and outward normal of the chosen platform edge, per env."""
        from isaaclab.utils.math import quat_apply

        n = self.u.num_envs
        centre = self.platform.data.root_pos_w.torch
        quat = self.platform.data.root_quat_w.torch
        hx, hy = self.platform_half
        local = torch.tensor([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], device=self.device)
        offsets = quat_apply(quat[:, None].expand(-1, 4, -1).reshape(-1, 4), local.repeat(n, 1)).reshape(n, 4, 3)
        mids = centre[:, None] + offsets
        mids[..., 2] = centre[:, None, 2] + self.platform_top_offset
        slot = self.slot_centre_w()[:, None]
        distance = (mids[..., :2] - slot[..., :2]).norm(dim=-1)
        near, far = distance.argmin(-1), distance.argmax(-1)
        towards = (centre[:, :2] - slot[:, 0, :2])[:, None]
        cross = towards[..., 0] * offsets[..., 1] - towards[..., 1] * offsets[..., 0]
        side = (distance - distance.gather(1, near[:, None])).abs() > 1e-4
        side &= (distance - distance.gather(1, far[:, None])).abs() > 1e-4
        left = torch.where(side & (cross > 0), torch.arange(4, device=self.device), 99).amin(-1).clamp(max=3)
        right = torch.where(side & (cross <= 0), torch.arange(4, device=self.device), 99).amin(-1).clamp(max=3)
        choice = torch.stack((near, far, left, right), dim=-1).gather(1, self.edges[:, None]).squeeze(-1)
        rows = torch.arange(n, device=self.device)
        normal = offsets[rows, choice]
        normal[:, 2] = 0.0
        return mids[rows, choice], normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # -- teacher
    def teacher_obs(self, goal_w):
        q = self.robot.data.joint_pos.torch - self.teacher_default
        qd = self.robot.data.joint_vel.torch
        quat = torch.zeros(len(q), 4, device=self.device)
        quat[:, 3] = 1.0  # the cube task's goal orientation, identity in (x, y, z, w)
        return torch.cat((q, qd, self._to_base(self.pick_point_w()), self._to_base(goal_w), quat, self.previous), dim=-1)

    def _rotation(self, closing_axis_w, hand_quat_b):
        """Raw IK rotation command turning the hand to point down with its y axis along ``closing_axis``."""
        from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix

        current = matrix_from_quat(hand_quat_b)
        y_des = self._dir_to_base(closing_axis_w)
        flip = (current[:, :, 1] * y_des).sum(-1, keepdim=True) < 0
        y_des = torch.where(flip, -y_des, y_des)
        z_des = torch.zeros_like(y_des)
        z_des[:, 2] = -1.0
        x_des = torch.linalg.cross(y_des, z_des, dim=-1)
        desired = torch.stack((x_des, y_des, z_des), dim=-1)
        error = axis_angle_from_quat(quat_from_matrix(desired @ current.transpose(1, 2)))
        step = ROTATION_GAIN * error
        step = step * (MAX_ROTATION_STEP / step.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        return step / self.arm._scale[:, 3:]

    def _rotation_frame(self, z_des, y_hint, hand_quat_b):
        """Raw IK rotation turning the hand to z axis ``z_des`` with its y axis nearest ``y_hint`` (base frame)."""
        from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix

        current = matrix_from_quat(hand_quat_b)
        z_des = z_des / z_des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        y_des = y_hint - (y_hint * z_des).sum(-1, keepdim=True) * z_des
        y_des = y_des / y_des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        flip = (current[:, :, 1] * y_des).sum(-1, keepdim=True) < 0
        y_des = torch.where(flip, -y_des, y_des)
        x_des = torch.linalg.cross(y_des, z_des, dim=-1)
        desired = torch.stack((x_des, y_des, z_des), dim=-1)
        error = axis_angle_from_quat(quat_from_matrix(desired @ current.transpose(1, 2)))
        step = ROTATION_GAIN * error
        step = step * (MAX_ROTATION_STEP / step.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        return step / self.arm._scale[:, 3:]

    def _yaw(self, closing_axis_w, hand_quat_b):
        """Raw IK rotation about the vertical turning the hand's closing axis onto ``closing_axis`` (mod 180)."""
        from isaaclab.utils.math import matrix_from_quat

        y_cur = matrix_from_quat(hand_quat_b)[:, :, 1]
        y_des = self._dir_to_base(closing_axis_w)
        yaw_cur = torch.atan2(y_cur[:, 1], y_cur[:, 0])
        yaw_des = torch.atan2(y_des[:, 1], y_des[:, 0])
        error = torch.remainder(yaw_des - yaw_cur + math.pi / 2, math.pi) - math.pi / 2
        step = torch.zeros(len(error), 3, device=self.device)
        step[:, 2] = (ROTATION_GAIN * error).clamp(-MAX_ROTATION_STEP, MAX_ROTATION_STEP)
        return step / self.arm._scale[:, 3:]

    def action(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached

        extracted = _phase_reached(self.u).clone()
        holding = _is_holding(self.u).clone()
        self.extracted_ever |= extracted
        self.held_ever |= holding
        place_w, edge_normal_w = self.edge_frames_w()
        place_w[:, 2] += PLACE_HEIGHT
        air_w = self.slot_centre_w()
        air_w[:, 2] = self.u.scene.env_origins[:, 2] + GOAL_HEIGHT
        goal_w = torch.where(extracted[:, None], place_w, air_w)

        self.raw = self.teacher(self.teacher_obs(goal_w))
        action = self.raw.clone()
        tool_b, hand_quat_b = self.arm._compute_frame_pose()
        axis = torch.where(extracted[:, None], edge_normal_w, self.slot_normal_w())
        if args.rotation == "controller":
            action[:, 3:6] = self._rotation(axis, hand_quat_b)
        elif args.rotation == "yaw":
            action[:, 3:6] = action[:, 3:6] + self._yaw(axis, hand_quat_b)
        if args.governor > 0:
            pick_b = self._to_base(self.pick_point_w())
            held_target_b = self._to_base(goal_w)
            # approaching: the tool goes to the pick point; holding: the held point goes to the goal
            remaining = torch.where((holding | extracted)[:, None], held_target_b - pick_b, pick_b - tool_b).norm(dim=-1)
            step = action[:, :3] * self.ik_scale
            cap = (args.governor * remaining).clamp_min(0.005)
            action[:, :3] = action[:, :3] * (cap / step.norm(dim=-1).clamp_min(1e-9)).clamp(max=1.0)[:, None]
        if args.approach_floor >= 0:
            # the cube stopped the teacher's plunge; a thin sheet does not
            floor = self._to_base(self.pick_point_w())[:, 2] - args.approach_floor
            lowest = (floor - tool_b[:, 2]) / self.ik_scale[:, 2]
            approaching = ~extracted & ~holding
            action[:, 2] = torch.where(approaching, torch.maximum(action[:, 2], lowest), action[:, 2])

        held_error = (self.pick_point_w() - place_w).norm(dim=-1)
        release = extracted & holding & ~self.released & (held_error < RELEASE_RADIUS)
        self.retreat_target = torch.where(release[:, None], tool_b + torch.tensor([0, 0, RETREAT_HEIGHT], device=self.device), self.retreat_target)
        self.released |= release
        action[:, 6] = torch.where(extracted, -1.0, action[:, 6])
        action[:, 6] = torch.where(self.released, 1.0, action[:, 6])
        retreat = self.retreat_target - tool_b
        retreat = retreat * (MAX_RETREAT_STEP / retreat.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        action[:, :3] = torch.where(self.released[:, None], retreat / self.ik_scale, action[:, :3])
        return action

    def after_step(self, dones):
        self.previous = self.raw.clone()
        self.reset(dones.nonzero(as_tuple=False).squeeze(-1))


class ScriptedSheetExpert(SheetTeacher):
    """A waypoint twist expert for the sheet: grasp the top edge, extract it straight up, lay it on the platform.

    Every step the twist is a capped proportional step toward the current waypoint (the damped IK arm
    covers a fraction of it, so the cap sets the speed), plus a rotation step keeping the hand pointing
    down with its closing axis set. Per environment, a state machine:

    ABOVE    6 cm above the middle of the top edge, gripper open, closing axis along the slot normal
    DESCEND  slowly down until the tool is ``PICK_DEPTH`` below the edge
    CLOSE    close and hold still until the task's detector reports a hold; reopen and retry if none
    LIFT     straight up at the grasp position to ``GOAL_HEIGHT``, slowly, until the task reports extraction
    CARRY    above the platform edge opposite the target edge, the hanging sheet's bottom just above the
             platform top, closing axis along the target edge's normal (held edge parallel to it)
    LAY      diagonally down to just above the target edge: the cloth lays onto the platform from the
             opposite edge as the hand travels, and the held edge ends on the target edge
    RELEASE  open, hold still; then RETREAT straight up
    """

    ABOVE, DESCEND, CLOSE, LIFT, CARRY, LAY, RELEASE, RETREAT = range(8)
    NAMES = ("above", "descend", "close", "lift", "carry", "lay", "release", "retreat")
    ABOVE_HEIGHT = 0.06
    HANG = 0.18 + 0.02  # sheet below the held point (20 cm - PICK_DEPTH), plus 2 cm clearance
    LAY_END_HEIGHT = 0.01
    LAY_SPEED = 0.06  # m/s along the lay path (rigid branch only)
    LAY_LOOKAHEAD = 0.03  # cloth lay: command this far ahead of measured progress [m]; speed emerges from the cap
    # rigid plate: lower onto the pivot, then tilt about it
    PIVOT_RADIUS = 0.18 + args.pivot_clearance  # grip to the plate's bottom edge, plus its clearance
    LOWER_TIME = 1.0
    TILT_ANGLE = math.radians(args.tilt_angle)
    TILT_RATE = 0.4  # rad/s
    SLIDE_OUT = 0.06  # after letting go, back the fingers out along themselves before lifting [m]
    SLIDE_STEPS = 40
    CAPS = {"above": 0.04, "descend": 0.01, "close": 0.01, "lift": 0.02, "carry": 0.04, "lay": 0.03,
            "release": 0.01, "retreat": 0.02}

    def __init__(self, env, edges):
        n = env.unwrapped.num_envs
        device = env.unwrapped.device
        self.state = torch.zeros(n, dtype=torch.long, device=device)
        self.timer = torch.zeros(n, dtype=torch.long, device=device)
        self.grasp_xy = torch.zeros(n, 2, device=device)
        self.lay_s = torch.zeros(n, device=device)
        self.release_point = torch.zeros(n, 3, device=device)
        self.release_z = torch.zeros(n, 3, device=device)
        self.release_z[:, 2] = -1.0
        super().__init__(env, None, edges)

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids):
            self.state[env_ids] = self.ABOVE
            self.timer[env_ids] = 0
            self.lay_s[env_ids] = 0.0

    def _aligned(self, axis_w, quat_b):
        from isaaclab.utils.math import matrix_from_quat

        axes = matrix_from_quat(quat_b)
        across = (axes[:, :, 1] * self._dir_to_base(axis_w)).sum(-1).abs()
        return (across > 0.98) & (-axes[:, 2, 2] > 0.98)

    def action(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached

        n, dev = self.u.num_envs, self.device
        extracted, holding = _phase_reached(self.u).clone(), _is_holding(self.u).clone()
        self.extracted_ever |= extracted
        self.held_ever |= holding
        tool_b, quat_b = self.arm._compute_frame_pose()
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        grasp_b = self._to_base(self.pick_point_w())
        edge_b = grasp_b + PICK_DEPTH * up
        target_mid_w, edge_normal_w = self.edge_frames_w()
        target_mid_b = self._to_base(target_mid_w)
        centre_w = self.platform.data.root_pos_w.torch.clone()
        centre_w[:, 2] += self.platform_top_offset
        opposite_b = 2 * self._to_base(centre_w) - target_mid_b
        lay_start = opposite_b + self.HANG * up
        lay_end = target_mid_b + self.LAY_END_HEIGHT * up
        lift_target = torch.cat((self.grasp_xy, torch.full((n, 1), GOAL_HEIGHT, device=dev)), dim=-1)

        s = self.state
        self.timer += 1
        # transitions, on the state at the start of this step
        def go(mask, new):
            self.state = torch.where(mask, new, self.state)
            self.timer = torch.where(mask, 0, self.timer)

        slot_aligned = self._aligned(self.slot_normal_w(), quat_b)
        edge_aligned = self._aligned(edge_normal_w, quat_b)
        # Transitions read where the hand is, not how it got there: under DAgger the student drives, and a
        # history-bound expert (e.g. one that insists on passing the above-waypoint first) keeps
        # labelling "go back up" to a student that went straight down, which teaches it to hover.
        lateral = (tool_b[:, :2] - grasp_b[:, :2]).norm(dim=-1)
        below_waypoint = tool_b[:, 2] < edge_b[:, 2] + self.ABOVE_HEIGHT + 0.01
        go((s == self.ABOVE) & (lateral < 0.01) & below_waypoint & slot_aligned, self.DESCEND)
        go((s == self.DESCEND) & (lateral > 0.02), self.ABOVE)
        go((s == self.DESCEND) & ((tool_b - grasp_b).norm(dim=-1) < 0.01), self.CLOSE)
        # a student that has closed on the plate on its own is past the grasp, whatever the expert thought
        lifting = (((s == self.CLOSE) & (self.timer >= 10)) | (s == self.ABOVE) | (s == self.DESCEND)) & holding
        self.grasp_xy = torch.where(lifting[:, None], tool_b[:, :2], self.grasp_xy)
        go(lifting, self.LIFT)
        go((s == self.CLOSE) & ~holding & (self.timer > 40), self.ABOVE)
        go((s == self.LIFT) & extracted, self.RELEASE if args.expert_variant == "drop" else self.CARRY)
        go((s == self.LIFT) & ~holding & (self.timer > 20), self.ABOVE)
        go((s == self.CARRY) & ((tool_b - lay_start).norm(dim=-1) < 0.015) & edge_aligned, self.LAY)
        dropped = ((s == self.CARRY) | (s == self.LAY)) & ~holding & (self.timer > 20)
        self.release_point = torch.where(dropped[:, None], tool_b, self.release_point)
        go(dropped, self.RETREAT)
        if self.rigid:
            # pivot-and-tilt: the plate's bottom edge comes down on the opposite edge, then the hand
            # swings over it toward the target edge, tilting the wrist so the plate rotates with it
            towards = target_mid_b - opposite_b
            towards[:, 2] = 0.0
            towards = towards / towards.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            opposite_b = opposite_b + args.pivot_inset * towards
            duration = self.TILT_ANGLE / self.TILT_RATE + self.LOWER_TIME
            self.lay_s = torch.where(self.state == self.LAY, (self.lay_s + self.u.step_dt / duration).clamp(max=1.0), 0.0)
            lower = (self.lay_s * duration / self.LOWER_TIME).clamp(max=1.0)
            theta = self.TILT_ANGLE * ((self.lay_s * duration - self.LOWER_TIME) / (duration - self.LOWER_TIME)).clamp(0.0, 1.0)
            radius = self.HANG - lower * (self.HANG - self.PIVOT_RADIUS)
            arm_dir = towards * torch.sin(theta)[:, None] + up * torch.cos(theta)[:, None]
            lay_target = opposite_b + radius[:, None] * arm_dir
            lay_end = lay_target
            done = self.lay_s >= 1.0
        else:
            # Pure pursuit, not a clock: the commanded point sits LAY_LOOKAHEAD ahead of the hand's
            # measured progress along the lay line. A time-driven target (the old lay_s schedule) made
            # the expert's action depend on hidden state, so no obs->action student could fit it --
            # both net sizes plateaued at imitation loss ~0.23 and coverage ~0.16 because of it.
            path_vec = lay_end - lay_start
            path = path_vec.norm(dim=-1).clamp_min(1e-3)
            direction = path_vec / path[:, None]
            s_meas = (((tool_b - lay_start) * direction).sum(-1) / path).clamp(0.0, 1.0)
            s_cmd = (s_meas + self.LAY_LOOKAHEAD / path).clamp(max=1.0)
            lay_target = lay_start + s_cmd[:, None] * path_vec
            done = (s_meas >= 0.98) & ((tool_b - lay_end).norm(dim=-1) < 0.01)
        self.release_point = torch.where((self.state == self.LAY)[:, None], lay_target, self.release_point)
        self.release_point = torch.where((self.state == self.LIFT)[:, None], tool_b, self.release_point)
        go((self.state == self.LAY) & done & (args.expert_variant != "hover"), self.RELEASE)
        go((self.state == self.RELEASE) & (self.timer >= 15), self.RETREAT)
        s = self.state

        targets = {
            self.ABOVE: edge_b + self.ABOVE_HEIGHT * up, self.DESCEND: grasp_b, self.CLOSE: grasp_b,
            self.LIFT: lift_target, self.CARRY: lay_start, self.LAY: lay_target,
            self.RELEASE: self.release_point, self.RETREAT: self.release_point + RETREAT_HEIGHT * up,
        }
        if self.rigid:
            # the lower finger is under the plate's edge: slide out along the fingers first, then go up
            out = self.release_point - self.SLIDE_OUT * self.release_z
            sliding = self.timer < self.SLIDE_STEPS
            targets[self.RETREAT] = torch.where(sliding[:, None], out, out + RETREAT_HEIGHT * up)
        target = tool_b.clone()
        cap = torch.zeros(n, device=dev)
        for state, point in targets.items():
            here = s == state
            target = torch.where(here[:, None], point, target)
            cap = torch.where(here, self.CAPS[self.NAMES[state]], cap)
        delta = args.expert_gain * (target - tool_b)
        delta = delta * (cap / delta.norm(dim=-1).clamp_min(1e-9)).clamp(max=1.0)[:, None]
        action = torch.zeros(n, 7, device=dev)
        action[:, :3] = delta / self.ik_scale
        axis = torch.where((s >= self.CARRY)[:, None], edge_normal_w, self.slot_normal_w())
        action[:, 3:6] = self._rotation(axis, quat_b)
        if self.rigid:
            # while laying and letting go, the hand follows the plate's tilt (z along grip-to-pivot)
            tilting = ((s == self.LAY) | (s == self.RELEASE) | ((s == self.RETREAT) & (self.timer < self.SLIDE_STEPS)))[:, None]
            z_tilt = -arm_dir if self.rigid else None
            self.release_z = torch.where((s == self.LAY)[:, None], z_tilt, self.release_z)
            y_hint = self._dir_to_base(edge_normal_w)
            tilt = self._rotation_frame(self.release_z, y_hint, quat_b)
            action[:, 3:6] = torch.where(tilting, tilt, action[:, 3:6])
        action[:, 6] = torch.where((s == self.ABOVE) | (s == self.DESCEND) | (s >= self.RELEASE), 1.0, -1.0)
        self.released = s >= self.RELEASE
        self.raw = action
        return action


def trace(driver, action, step):
    from isaaclab.utils.math import matrix_from_quat

    from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached

    u, i = driver.u, 0
    tool_b, quat_b = driver.arm._compute_frame_pose()
    pick_b = driver._to_base(driver.pick_point_w())
    axes = matrix_from_quat(quat_b)[i]
    normal = driver._dir_to_base(driver.slot_normal_w())[i]
    fingers = driver.robot.data.joint_pos.torch[i, -2:].sum().item()
    gripper = u.action_manager.get_term("gripper_action").raw_actions[i, 0].item()
    print(f"[trace] step {step:3d} tool {[round(v, 3) for v in tool_b[i].tolist()]} pick {[round(v, 3) for v in pick_b[i].tolist()]} "
          f"dist {100 * (tool_b[i] - pick_b[i]).norm():5.1f} cm | hand z.down {-axes[2, 2]:+.2f} y.normal {abs((axes[:, 1] * normal).sum()):.2f} "
          f"| teacher grip {driver.raw[i, 6]:+.2f} cmd {action[i, 6]:+.2f} filter {gripper:+.2f} width {1000 * fingers:4.1f} mm "
          f"| hold {bool(_is_holding(u)[i])} extracted {bool(_phase_reached(u)[i])} | teacher |v| {driver.raw[i, :3].norm():.2f}"
          + (f" | state {ScriptedSheetExpert.NAMES[int(driver.state[i])]}" if hasattr(driver, "state") else "")
          + (plate_trace(driver, i) if driver.rigid else ""),
          flush=True)


def plate_trace(driver, i):
    from isaaclab.utils.math import matrix_from_quat

    plate = driver.sheet
    pos = driver._to_base(plate.data.root_pos_w.torch)[i]
    normal_z = matrix_from_quat(plate.data.root_quat_w.torch)[i, 2, 2].abs()
    speed = plate.data.root_lin_vel_w.torch[i].norm()
    spin = plate.data.root_ang_vel_w.torch[i].norm()
    top = driver.platform.data.root_pos_w.torch[i, 2] + driver.platform_top_offset - driver.u.scene.env_origins[i, 2]
    centre = driver._to_base(driver.platform.data.root_pos_w.torch)[i]
    return (f" | plate {[round(v, 3) for v in pos.tolist()]} |n_z| {normal_z:.2f} v {speed:.2f} w {spin:.1f} "
            f"| platform top {top:.3f} centre {[round(v, 3) for v in centre[:2].tolist()]}")


class RewardPlot:
    """Live plot of environment 0: cumulative reward per term, return, coverage and the task's flags.

    Refreshed every ``--plot_every`` steps; when env 0's episode ends the figure is saved under
    ``evaluations/sheet_twist_plots/<name>/episode_<k>.png`` and cleared for the next episode.
    """

    def __init__(self, u, name):
        import matplotlib

        import os

        matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
        import matplotlib.pyplot as plt

        self.plt, self.u = plt, u
        self.names = list(u.reward_manager.active_terms)
        self.out = Path("evaluations", "sheet_twist_plots", name)
        self.out.mkdir(parents=True, exist_ok=True)
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        self.episode = 0
        self.clear()

    def clear(self):
        self.t, self.terms, self.total, self.coverage, self.flags = [], [], [], [], []

    def record(self, driver):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached

        u = self.u
        self.t.append(len(self.t) * u.step_dt)
        self.terms.append((u.reward_manager._step_reward[0] * u.step_dt).cpu().numpy())
        self.total.append(float(self.terms[-1].sum()))
        self.coverage.append(float(coverage_term(u).coverage[0]))
        self.flags.append((float(_is_holding(u)[0]), float(_phase_reached(u)[0]), float(coverage_term(u).released[0])))

    def draw(self, final=False):
        import numpy as np

        if not self.t:
            return
        t = np.array(self.t)
        terms = np.cumsum(np.array(self.terms), axis=0)
        ax_terms, ax_total, ax_state = self.axes
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=0.3)
        order = np.argsort(-np.abs(terms[-1]))
        for k in order[:10]:
            ax_terms.plot(t, terms[:, k], label=f"{self.names[k]} ({terms[-1, k]:+.1f})", linewidth=1.4)
        ax_terms.set_ylabel("cumulative reward per term")
        ax_terms.legend(fontsize=7, loc="upper left", ncol=2)
        ax_total.plot(t, np.cumsum(self.total), color="k", linewidth=1.8, label=f"return ({sum(self.total):+.1f})")
        ax_total.plot(t, np.array(self.total) * 10, color="#888", linewidth=0.8, label="per-step reward x 10")
        ax_total.set_ylabel("reward")
        ax_total.legend(fontsize=8, loc="upper left")
        flags = np.array(self.flags)
        ax_state.plot(t, self.coverage, color="#1baf7a", linewidth=1.8, label="coverage")
        ax_state.step(t, flags[:, 0] * 0.9, color="#2a78d6", label="holding", where="post")
        ax_state.step(t, flags[:, 1] * 0.8, color="#eb6834", label="extracted", where="post")
        ax_state.step(t, flags[:, 2] * 0.7, color="#e34948", label="released", where="post")
        ax_state.set_ylim(-0.05, 1.05)
        ax_state.set_xlabel("time [s]")
        ax_state.legend(fontsize=8, loc="upper left")
        self.fig.suptitle(f"env 0, episode {self.episode + 1}")
        self.fig.tight_layout()
        self.plt.pause(0.001)
        if final:
            path = self.out / f"episode_{self.episode + 1}.png"
            self.fig.savefig(path, dpi=110)
            print(f"[plot] saved {path}", flush=True)
            self.episode += 1
            self.clear()


def _clone_hidden(hidden):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(h.detach().clone() for h in hidden)
    return hidden.detach().clone()


def coverage_term(u):
    return u.reward_manager.get_term_cfg("coverage").func


def evaluate_edges(env, teacher_net, student=None):
    """Run the expert on every edge at once (or ``student`` on ``--edge``); record each episode as it ends.

    With a student the expert still runs alongside, but only to track the hold and extraction flags;
    "released" is then the coverage term's own flag (sheet unheld and fingers open after extraction).
    """
    u = env.unwrapped
    n = u.num_envs
    edges = torch.full((n,), EDGES.index(args.edge)) if student is not None else torch.arange(n) % len(EDGES)
    driver = ScriptedSheetExpert(env, edges) if args.expert == "scripted" else SheetTeacher(env, teacher_net, edges)
    term = coverage_term(u)
    keys = ("held", "extracted", "released", "early_end", "success", "best_released_coverage", "final_coverage", "return")
    results = {e: {k: [] for k in keys} for e in EDGES}
    ended = {}
    returns = torch.zeros(n, device=u.device)
    completed = torch.zeros(n, dtype=torch.long, device=u.device)
    plot = None
    if args.plot:
        name = (args.eval_student.parent.name + "_" + args.eval_student.stem) if student is not None else "expert"
        plot = RewardPlot(u, name + ("_stochastic" if args.stochastic else ""))
    env.reset()
    driver.reset(torch.arange(n, device=u.device))
    start = time.time()
    for step in range(args.eval_episodes * int(u.max_episode_length)):
        with torch.no_grad():
            action = driver.action()
            if student is not None:
                action = student({"policy": u.obs_buf["policy"]}, stochastic_output=args.stochastic)
                if args.student_noise > 0:  # robustness check: the same noise the student trained with
                    action = action.clone()
                    action[:, :6] += args.student_noise * torch.randn_like(action[:, :6])
        if args.trace and step % args.trace == 0:
            trace(driver, action, step)
        best, final = term.best_released.clone(), term.coverage.clone()
        released = term.released.clone() if student is not None else driver.released.clone()
        flags = (driver.held_ever.clone(), driver.extracted_ever.clone(), released)
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        if student is not None:
            student.reset(done)  # no-op for an MLP; clears a recurrent student's memory of ended episodes
        returns += reward
        if plot is not None:
            plot.record(driver)
            if bool(done[0]):
                plot.draw(final=True)
            elif step % args.plot_every == 0:
                plot.draw()
        eligible = done & (completed < args.eval_episodes)
        success_name = "placed" if "placed" in u.termination_manager.active_terms else "draped"
        success = (u.termination_manager.get_term(success_name) if success_name in u.termination_manager.active_terms
                   else torch.zeros_like(done))
        for name in u.termination_manager.active_terms:
            ended[name] = ended.get(name, 0) + int((u.termination_manager.get_term(name) & eligible).sum())
        for row in eligible.nonzero(as_tuple=False).squeeze(-1).tolist():
            record = results[EDGES[int(edges[row])]]
            for key, flag in zip(("held", "extracted", "released"), flags):
                record[key].append(float(flag[row]))
            record["early_end"].append(float(terminated[row] & ~truncated[row]))
            record["success"].append(float(success[row]))
            record["best_released_coverage"].append(float(best[row]))
            record["final_coverage"].append(float(final[row]))
            record["return"].append(float(returns[row]))
        completed += eligible.long()
        returns[done] = 0.0
        driver.after_step(done)
        if bool((completed >= args.eval_episodes).all()):
            break
    print(f"[edges] {step + 1} steps in {time.time() - start:.0f}s; terminations: {ended}", flush=True)
    summary = {}
    for edge, r in results.items():
        mean = {k: sum(v) / max(1, len(v)) for k, v in r.items()}
        mean["episodes"] = len(r["held"])
        values = r["best_released_coverage"]
        mean["best_released_coverage_sem"] = (
            (sum((v - mean["best_released_coverage"]) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
            / max(1, len(values)) ** 0.5
        )
        summary[edge] = mean
        print(f"[edges] {edge:5s}: episodes {mean['episodes']:4d}  early end {mean['early_end']:.2f}  held {mean['held']:.2f}  "
              f"extracted {mean['extracted']:.2f}  released {mean['released']:.2f}  "
              f"best released coverage {mean['best_released_coverage']:.3f} +- {mean['best_released_coverage_sem']:.3f}  "
              f"final coverage {mean['final_coverage']:.3f}  success {mean['success']:.3f}  return {mean['return']:.1f}",
              flush=True)
    best = max(summary, key=lambda e: summary[e]["best_released_coverage"])
    print(f"[edges] highest best-released coverage: {best}", flush=True)
    return summary, best


@hydra_task_config(args.task, args.agent, play_mode=False)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    from torch.utils.tensorboard import SummaryWriter

    if args.expert == "teacher" and (args.teacher is None or not args.teacher.is_file()):
        raise FileNotFoundError(f"--expert teacher needs --teacher <checkpoint>; got {args.teacher}")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.exact_reset:
        env_cfg.events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_gripper_joints.params["position_range"] = (0.0, 0.0)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    if args.gru:
        from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

        hidden, layers = (int(v) for v in args.gru.split(","))
        agent_cfg.actor = RslRlRNNModelCfg(
            hidden_dims=[128, 64], activation="elu", obs_normalization=False,
            distribution_cfg=agent_cfg.actor.distribution_cfg,
            rnn_type="gru", rnn_hidden_dim=hidden, rnn_num_layers=layers,
        )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    agent_cfg.actor.distribution_cfg.init_std = args.student_std
    agent_cfg.save_interval = args.save_interval
    if not args.eval_edges and args.eval_student is None:
        if not args.run_name:
            raise ValueError("--run_name is required for DAgger.")
        agent_cfg.run_name = args.run_name
        log_dir = Path("logs", "rsl_rl", agent_cfg.experiment_name, args.run_name).resolve()
        if log_dir.exists() and any(log_dir.iterdir()):
            raise FileExistsError(f"Run folder already exists: {log_dir}.")

    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(
            create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
            clip_actions=agent_cfg.clip_actions,
        )
        u = env.unwrapped
        device = agent_cfg.device
        teacher_net = load_teacher(args.teacher, device) if args.expert == "teacher" else None
        if args.eval_student is not None:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
            runner.load(str(args.eval_student), load_cfg={"actor": True})
            student = runner.get_inference_policy(device=device)
            summary, _ = evaluate_edges(env.env, teacher_net, student)
            mode = "_stochastic" if args.stochastic else ""
            out = Path("evaluations", "sheet_twist_student", args.eval_student.parent.name + "_" + args.eval_student.stem
                       + mode + f"_seed{args.seed}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"student": str(args.eval_student), "edge": args.edge,
                                      "task": args.task, "seed": args.seed, "stochastic": args.stochastic,
                                      "result": summary[args.edge]}, indent=2))
            print(f"[edges] wrote {out}")
            env.close()
            return
        if args.eval_edges:
            summary, best = evaluate_edges(env.env, teacher_net)
            name = "scripted" if args.expert == "scripted" else args.teacher.parent.name + "_" + args.teacher.stem
            out = Path("evaluations", "sheet_twist_edges", name + ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"expert": args.expert, "teacher": str(args.teacher), "best": best, "edges": summary}, indent=2))
            print(f"[edges] wrote {out}")
            env.close()
            return

        write_run_manifest(str(log_dir), library="rsl_rl", task=args.task, metadata={"agent": args.agent})
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        invocation = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (log_dir / "dagger.json").write_text(json.dumps(invocation, indent=2))

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        if args.init_student is not None:
            runner.load(str(args.init_student), load_cfg={"actor": True, "critic": True})
            print(f"[DAgger] student initialised from {args.init_student}", flush=True)
        actor, critic = runner.alg.actor, runner.alg.critic
        critic_only = args.updates_per_iter == 0 and not args.gru
        if critic_only:
            if args.beta_start != 0 or args.beta_floor != 0 or args.student_noise != 0:
                raise ValueError("Critic-only collection requires beta_start=beta_floor=student_noise=0; "
                                 "it samples the checkpoint's policy distribution directly.")
            actor.requires_grad_(False)
        edges = torch.full((u.num_envs,), EDGES.index(args.edge))
        driver = ScriptedSheetExpert(env, edges) if args.expert == "scripted" else SheetTeacher(env, teacher_net, edges)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        writer = SummaryWriter(str(log_dir))
        gamma, lam = agent_cfg.algorithm.gamma, agent_cfg.algorithm.lam
        term = coverage_term(u)

        obs = env.get_observations().to(device)
        obs_dim, act_dim, n, steps = obs["policy"].shape[-1], env.num_actions, env.num_envs, args.steps_per_iter
        buf_obs = torch.zeros(args.buffer_size, obs_dim, device=device)
        buf_act = torch.zeros(args.buffer_size, act_dim, device=device)
        count = head = 0
        roll_obs = torch.zeros(steps, n, obs_dim, device=device)
        roll_lab = torch.zeros(steps, n, act_dim, device=device)
        roll_val, roll_rew, roll_done = (torch.zeros(steps, n, device=device) for _ in range(3))
        # desynchronise the episodes, as RSL-RL does for PPO, so every rollout mixes all stages of the task
        u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
        episode_teacher = torch.rand(n, device=device) < args.beta_start
        print(f"[DAgger] sheet student, edge {args.edge}, teacher {args.teacher}; logging to {log_dir}", flush=True)
        for it in range(args.iterations):
            start = time.time()
            beta = max(args.beta_floor, args.beta_start * (1.0 - it / args.beta_iters))
            logs = {}
            actor.eval()
            start_hidden = _clone_hidden(actor.get_hidden_state())
            with torch.no_grad():
                for step in range(steps):
                    label = driver.action()
                    roll_lab[step] = label
                    if args.mix == "step":
                        use_teacher = torch.rand(n, 1, device=device) < beta
                    else:
                        use_teacher = episode_teacher[:, None]
                    executed = label.clone()
                    if args.expert_noise > 0:
                        executed[:, :6] += args.expert_noise * torch.randn_like(executed[:, :6])
                    # Frozen-actor preparation must value the same stochastic policy PPO executes.
                    student = actor(obs, stochastic_output=critic_only)
                    if args.student_noise > 0:
                        student = student.clone()
                        student[:, :6] += args.student_noise * torch.randn_like(student[:, :6])
                    action = torch.where(use_teacher, executed, student)
                    rows = (head + torch.arange(n, device=device)) % args.buffer_size
                    buf_obs[rows], buf_act[rows] = obs["policy"], label
                    head, count = (head + n) % args.buffer_size, min(count + n, args.buffer_size)
                    roll_obs[step], roll_val[step] = obs["policy"], critic(obs).squeeze(-1)
                    obs, rew, dones, extras = env.step(action.to(env.device))
                    obs, rew, dones = obs.to(device), rew.to(device), dones.to(device)
                    driver.after_step(dones)
                    actor.reset(dones)  # no-op for an MLP; zeroes a recurrent actor's ended episodes
                    redraw = dones.bool()
                    episode_teacher = torch.where(redraw, torch.rand(n, device=device) < beta, episode_teacher)
                    rew = rew.clone()
                    if "time_outs" in extras:
                        rew += gamma * roll_val[step] * extras["time_outs"].to(device).float()
                    roll_rew[step], roll_done[step] = rew, dones.float()
                    for key, value in extras.get("log", {}).items():
                        if key.startswith("Coverage/"):
                            logs.setdefault(key, []).append(float(value))
                end_hidden = _clone_hidden(actor.get_hidden_state())
                last = critic(obs).squeeze(-1)
                returns, adv = torch.zeros_like(roll_rew), torch.zeros(n, device=device)
                for step in reversed(range(steps)):
                    nxt = last if step == steps - 1 else roll_val[step + 1]
                    keep = 1.0 - roll_done[step]
                    adv = roll_rew[step] + gamma * nxt * keep - roll_val[step] + gamma * lam * keep * adv
                    returns[step] = adv + roll_val[step]
            collect = time.time() - start

            actor.train()
            actor_loss = 0.0
            if args.gru:
                # truncated BPTT over this rollout window: replay it sequentially from the stored
                # initial hidden state, resetting at episode ends, gradients flowing through time
                std = roll_lab.reshape(-1, act_dim).std(dim=0).clamp_min(1e-3)
                for _ in range(args.gru_epochs):
                    actor.reset(hidden_state=_clone_hidden(start_hidden))
                    loss = 0.0
                    for t in range(steps):
                        loss = loss + ((actor({"policy": roll_obs[t]}) - roll_lab[t]) / std).square().mean()
                        actor.reset(roll_done[t])
                    loss = loss / steps
                    actor_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()
                    actor.detach_hidden_state()
                    actor_loss += loss.item() / args.gru_epochs
                actor.reset(hidden_state=_clone_hidden(end_hidden))  # resume rollout where it left off
            else:
                std = buf_act[:count].std(dim=0).clamp_min(1e-3)
                for _ in range(args.updates_per_iter):
                    rows = torch.randint(0, count, (min(args.batch_size, count),), device=device)
                    loss = ((actor({"policy": buf_obs[rows]}) - buf_act[rows]) / std).square().mean()
                    actor_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()
                    actor_loss += loss.item() / args.updates_per_iter
            critic.train()
            flat_obs, flat_ret = roll_obs.reshape(-1, obs_dim), returns.reshape(-1)
            critic_loss = 0.0
            for _ in range(args.critic_epochs):
                for rows in torch.randperm(len(flat_ret), device=device).split(max(1, len(flat_ret) // 4)):
                    loss = (critic({"policy": flat_obs[rows]}).squeeze(-1) - flat_ret[rows]).square().mean()
                    critic_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_opt.step()
                    critic_loss += loss.item() / (4 * args.critic_epochs)

            stats = {"DAgger/beta": beta, "DAgger/actor_loss_normalised": actor_loss, "DAgger/critic_loss": critic_loss,
                     "DAgger/extracted_now": driver.extracted_ever.float().mean().item(),
                     "DAgger/released_now": driver.released.float().mean().item(),
                     "DAgger/coverage_now": term.coverage.mean().item()}
            stats.update({k: sum(v) / len(v) for k, v in logs.items()})
            for key, value in stats.items():
                writer.add_scalar(key, value, it)
            print(f"[DAgger] it {it:4d} beta {beta:.2f} actor {actor_loss:.3f} critic {critic_loss:.3f} "
                  f"extracted {stats['DAgger/extracted_now']:.2f} released {stats['DAgger/released_now']:.2f} "
                  f"coverage now {stats['DAgger/coverage_now']:.3f} best released "
                  f"{stats.get('Coverage/best_released', math.nan):.3f}  {collect:.1f}s+{time.time() - start - collect:.1f}s",
                  flush=True)
            if it % args.save_interval == 0 or it == args.iterations - 1:
                saved = runner.alg.save()
                saved.update(iter=it, infos={"dagger": invocation})
                torch.save(saved, log_dir / f"model_{it}.pt")
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
