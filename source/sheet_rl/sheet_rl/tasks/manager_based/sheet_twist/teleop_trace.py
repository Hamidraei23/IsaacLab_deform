"""Per-step trace of a hand-flown UR10e episode, for replaying and picking apart offline.

One row per control step, holding the command that was sent and everything the arm, the
gripper and the grasp logic did in response. ``scripts/teleop_ur10e.py`` writes it;
``scripts/replay_teleop_ur10e.py`` feeds the same commands back into a fresh environment
built from the sidecar metadata (seed, flags, action scale) and records the same columns,
so the two files line up step for step and any divergence is visible by subtraction.

Columns are captured by :func:`capture_row` for both writers on purpose: a replay that
measured things its own way could differ in exactly the detail being investigated.
"""

import csv
import json
import os

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from ..sheet_rl.mdp.gripper_geometry import finger_half_width, pad_positions
from ..sheet_rl.mdp.rewards import _is_holding, _phase_reached, grasp_debug
from .ur10e_mdp import TOOL_POS, TOOL_ROT

ACTION_COLUMNS = [f"a{i}" for i in range(7)]
"""The seven values handed to ``env.step``: six scaled twist channels, then the gripper."""

TWIST_COLUMNS = ["dx", "dy", "dz", "drx", "dry", "drz"]
"""The same twist in metres and radians, before the action scale was divided out."""


def tool_pose_in_base(env, robot):
    """Tool position and xyzw orientation in the robot base frame, as the IK action sees it."""
    pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index("wrist_3_link")]
    n = len(pose)
    offset = torch.tensor(TOOL_POS, device=pose.device).expand(n, -1)
    rotation = torch.tensor(TOOL_ROT, device=pose.device).expand(n, -1)
    return subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
        pose[:, :3] + quat_apply(pose[:, 3:7], offset), quat_mul(pose[:, 3:7], rotation),
    )


def capture_row(env, *, step, episode, action, twist, reset_before, reward, ended):
    """Everything worth knowing about environment 0 on this step, as flat floats."""
    robot = env.scene["robot"]
    row = {"step": step, "episode": episode, "t": step * env.step_dt, "reset_before": int(reset_before)}
    row.update(zip(ACTION_COLUMNS, (float(v) for v in action)))
    row.update(zip(TWIST_COLUMNS, (float(v) for v in twist)))
    for name, value in zip(robot.joint_names, robot.data.joint_pos.torch[0].tolist()):
        row[f"q_{name}"] = value
    for name, value in zip(robot.joint_names, robot.data.joint_vel.torch[0].tolist()):
        row[f"qd_{name}"] = value
    # What the drives were asked for, if this build exposes it: the thing to compare against
    # the twist when the arm moves without a command.
    targets = getattr(robot.data, "joint_pos_target", None)
    if targets is not None:
        for name, value in zip(robot.joint_names, targets.torch[0].tolist()):
            row[f"qt_{name}"] = value
    position, orientation = tool_pose_in_base(env, robot)
    row.update(zip(("tool_x", "tool_y", "tool_z"), position[0].tolist()))
    row.update(zip(("tool_qx", "tool_qy", "tool_qz", "tool_qw"), orientation[0].tolist()))
    row["half_width"] = float(finger_half_width(env, SceneEntityCfg("robot"))[0])
    # Where the pads actually are, and where the slot walls are: a pinch made on the wall's
    # top edge sandwiches cloth between a pad and a kinematic box, which the coupler resolves
    # violently. Walls are static, but recording them keeps the trace self-contained.
    pads = pad_positions(env, robot)
    if pads is not None:
        for side, point in zip(("padL", "padR"), pads[0].tolist()):
            row.update(zip((f"{side}_x", f"{side}_y", f"{side}_z"), point))
    for name in ("slot_neg_y", "slot_pos_y"):
        if name in env.scene.keys():
            data = env.scene[name].data
            pos = data.root_pos_w.torch[0] if hasattr(data.root_pos_w, "torch") else data.root_pos_w[0]
            quat = data.root_quat_w.torch[0] if hasattr(data.root_quat_w, "torch") else data.root_quat_w[0]
            row.update(zip((f"{name}_x", f"{name}_y", f"{name}_z"), pos.tolist()))
            row.update(zip((f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"), quat.tolist()))
    row["gripper_filter"] = float(env.action_manager.get_term("gripper_action").raw_actions[0, 0])
    row["is_holding"] = int(bool(_is_holding(env)[0]))
    row["phase_reached"] = int(bool(_phase_reached(env)[0]))
    for name, value in grasp_debug(env).items():
        row[f"g_{name}"] = float(value[0])
    row["reward_total"] = float(reward)
    for name, value in env.reward_manager.get_active_iterable_terms(0):
        # The manager reports per-second values; scale to what the term actually paid this step.
        row[f"r_{name}"] = float(value[0]) * env.step_dt
    row["ended_by"] = "+".join(ended)
    return row


class TraceWriter:
    """Accumulates rows and rewrites the CSV whole on each flush, so a crash loses at most a burst."""

    def __init__(self, path, meta):
        self.path = path
        self.meta = dict(meta)
        self.rows = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(meta_path(path), "w") as handle:
            json.dump(self.meta, handle, indent=2)

    def append(self, row):
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        columns = list(dict.fromkeys(key for row in self.rows for key in row))
        with open(self.path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


def meta_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def load_trace(path):
    """The sidecar metadata and the rows, numeric columns parsed, ``ended_by`` kept as text."""
    with open(meta_path(path)) as handle:
        meta = json.load(handle)
    with open(path, newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key == "ended_by":
                    row[key] = value
                else:
                    row[key] = float(value) if value != "" else float("nan")
            rows.append(row)
    return meta, rows
