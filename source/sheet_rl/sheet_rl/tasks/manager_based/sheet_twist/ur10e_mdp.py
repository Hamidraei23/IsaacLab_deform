"""UR10e reset and Franka proprioception adapter for the existing twist policy."""

import json
import math
import pathlib
import time

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_inv, quat_mul, sample_uniform

from ..sheet_rl.mdp.grasp_control import FilteredBinaryGripperAction, FilteredBinaryGripperActionCfg
from ..sheet_rl.mdp.gripper_geometry import finger_half_width
from . import ur10e_gripper as _gripper
from .self_collision import ELBOW_LIMIT_DEG
from .serial_kinematics import SerialKinematics

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# The gripper's force and pose control is tuned in ur10e_gripper.py; these are its names.
GRIPPER_SIGNS = _gripper.SIGNS
GRIPPER_LEADERS = _gripper.LEADERS
GRIPPER_KNUCKLES = _gripper.KNUCKLES
GRIPPER_OPEN_ANGLE = _gripper.OPEN_ANGLE
GRIPPER_CLOSE_ANGLE = _gripper.CLOSE_ANGLE
OPEN_HALF_WIDTH = _gripper.OPEN_HALF_WIDTH_MAX
GRIPPER_OPEN_HALF_WIDTH = _gripper.open_half_width(_gripper.OPEN_ANGLE)
TOOL_POS = (0., 0., 0.145)
# ee_joint connects wrist_3_link to the gripper with a -90 degree rotation about z.
TOOL_ROT = (0., 0., -math.sqrt(.5), math.sqrt(.5))
PANDA_DEFAULT = (0., -.569, 0., -2.810, 0., 3.037, .741)
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


class FrankaObservationAdapter:
    """Represent the measured UR tool pose using a continuous virtual Panda IK solution.

The checkpoint's 9+9 proprioceptive inputs retain their meanings: seven Panda
angles/velocities and two equivalent finger travels. This is an approximate
cross-robot transfer, not a claim that UR joint angles are interchangeable.
"""

    def __init__(self, env):
        from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

        self.env = env
        self.robot = env.scene["robot"]
        self.arm_ids, _ = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(list(GRIPPER_SIGNS), preserve_order=True)
        self.ur = SerialKinematics.from_usd(
            env.cfg.scene.robot.spawn.usd_path, ARM_JOINTS, "wrist_3_link", TOOL_POS, TOOL_ROT,
            variants=env.cfg.scene.robot.spawn.variants, device=env.device,
        )
        # the stock USD's elbow allows +/-180; the task narrows it (ur10e_asset.limit_elbow), so the reset IK must too
        self.ur.limits[:, ARM_JOINTS.index("elbow_joint")] = torch.tensor([-1., 1.], device=env.device) * math.radians(ELBOW_LIMIT_DEG)
        self.panda = SerialKinematics.from_usd(
            FRANKA_PANDA_MENAGERIE_CFG.spawn.usd_path, PANDA_JOINTS, "panda_hand", (0., 0., .107),
            device=env.device,
        )
        self.home = torch.tensor(PANDA_DEFAULT, device=env.device).expand(env.num_envs, -1)
        self.q = self.home.clone()
        self.qvel = torch.zeros_like(self.q)
        self.width = torch.full((env.num_envs,), .04, device=env.device)
        self.width_vel = torch.zeros_like(self.width)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.position_error = torch.zeros_like(self.width)
        self.rotation_error = torch.zeros_like(self.width)
        self.last_step = -1
        self.warned = False

    def update(self):
        step = self.env.common_step_counter
        if self.last_step == step:
            return
        self.last_step = step
        urq = self.robot.data.joint_pos.torch[:, self.arm_ids]
        urqd = self.robot.data.joint_vel.torch[:, self.arm_ids]
        pos, rot, urjac = self.ur.forward(urq)
        # A blown-up simulation (the cloth coupling can put NaN into the arm's state) must not take the
        # whole environment down: in an environment whose arm state is not finite the virtual Panda
        # holds its last posture with zero velocity until the episode ends and the reset restores it.
        self.q = torch.where(torch.isfinite(self.q).all(-1, keepdim=True), self.q, self.home)
        ok = (torch.isfinite(pos).all(-1) & torch.isfinite(rot).all(-1) & torch.isfinite(urqd).all(-1)
              & torch.isfinite(urjac).flatten(1).all(-1))
        if not bool(ok.all()):
            if not self.warned:
                print(f"[ur10e adapter] non-finite arm state in env(s) {(~ok).nonzero().flatten().tolist()[:8]} at step "
                      f"{step}: their Franka joint observations are held until the reset (shown once)", flush=True)
                self.warned = True
            held_pos, held_rot, _ = self.panda.forward(self.q)
            pos = torch.where(ok[:, None], pos, held_pos)
            rot = torch.where(ok[:, None], rot, held_rot)
            urqd = torch.where(ok[:, None], urqd, 0.)
            urjac = torch.where(ok[:, None, None], urjac, 0.)
        self.q = self.panda.solve(pos, rot, self.q, iterations=5)
        p, r, jac = self.panda.forward(self.q)
        dp, dr = compute_pose_error(p, r, pos, rot, rot_error_type="axis_angle")
        self.position_error = dp.norm(dim=-1)
        self.rotation_error = dr.norm(dim=-1)
        velocity = (urjac @ urqd[..., None]).squeeze(-1)
        self.qvel = self.panda.inverse_velocity(jac, velocity)
        width = (finger_half_width(self.env, SceneEntityCfg("robot")) * (.04 / OPEN_HALF_WIDTH)).clamp(0., .04)
        self.width_vel = torch.where(self.fresh, 0., (width - self.width) / self.env.step_dt)
        self.width = width
        self.fresh[:] = False


def adapter(env):
    if not hasattr(env, "_franka_observation_adapter"):
        env._franka_observation_adapter = FrankaObservationAdapter(env)
    return env._franka_observation_adapter


def franka_joint_pos_rel(env):
    state = adapter(env)
    state.update()
    return torch.cat((state.q - state.home, (state.width - .04)[:, None].expand(-1, 2)), -1)


def franka_joint_vel(env):
    state = adapter(env)
    state.update()
    return torch.cat((state.qvel, state.width_vel[:, None].expand(-1, 2)), -1)


START_POSE_FILE = pathlib.Path(__file__).with_name("ur10e_start_pose.json")
"""Where ``scripts/ur10e_start_pose_gui.py`` saves a hand-set start pose. When the file exists the UR10e
task starts from that posture instead of from the Franka's start (see ``reset_ur10e``)."""


def load_start_pose(path=START_POSE_FILE):
    """The saved arm joint positions [rad] in ``ARM_JOINTS`` order, or ``None`` if none is saved."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    joints = json.loads(path.read_text())["joint_pos"]
    return [float(joints[name]) for name in ARM_JOINTS]


def save_start_pose(joint_pos, path=START_POSE_FILE, **info):
    """Write arm joint positions [rad], in ``ARM_JOINTS`` order, as the task's start pose."""
    record = {"joint_pos": {name: float(q) for name, q in zip(ARM_JOINTS, joint_pos)},
              "saved": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    pathlib.Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return pathlib.Path(path)


ARM_GROUPS = {"shoulder": "shoulder_.*", "elbow": "elbow_joint", "wrist": "wrist_.*"}
"""The arm's joints in the three groups its drives are tuned in, as joint-name expressions."""
ARM_CONTROL_FILE = pathlib.Path(__file__).with_name("ur10e_arm_control.json")
"""Where ``scripts/ur10e_arm_tuning_gui.py`` saves the arm's tuned control; used by the task when present."""
DEFAULT_ARM_CONTROL = {
    # the Franka task's drive gains (400/80) and effort limits, which match its per-step response
    "stiffness": {"shoulder": 400., "elbow": 400., "wrist": 400.},  # Nm/rad
    "damping": {"shoulder": 80., "elbow": 80., "wrist": 80.},  # Nm s/rad
    "armature": {"shoulder": .01, "elbow": .01, "wrist": .01},  # kg m^2
    "effort_limit": {"shoulder": 87., "elbow": 87., "wrist": 12.},  # Nm
    "velocity_limit": {"shoulder": None, "elbow": None, "wrist": None},  # rad/s; None: the asset's own (2.09 / 3.14 / 3.14)
    "ik_lambda": .01,  # damping of the damped-least-squares IK (Isaac Lab's default)
}
"""The arm control the UR10e task uses when no tuned file is saved."""


def load_arm_control(path=ARM_CONTROL_FILE):
    """The tuned arm control (``DEFAULT_ARM_CONTROL`` overlaid with the saved file, if any)."""
    control = json.loads(json.dumps(DEFAULT_ARM_CONTROL))
    path = pathlib.Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        for key, value in saved.items():
            if key in control and isinstance(control[key], dict):
                control[key].update({g: v for g, v in value.items() if g in ARM_GROUPS})
            elif key in control:
                control[key] = value
    return control


def save_arm_control(control, path=ARM_CONTROL_FILE, **info):
    """Write a tuned arm control (``DEFAULT_ARM_CONTROL``'s layout) for the task to use."""
    record = {key: control[key] for key in DEFAULT_ARM_CONTROL}
    record.update(saved=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
    pathlib.Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return pathlib.Path(path)


def arm_actuator(control):
    """The arm's implicit actuator for an arm control, one value per joint group."""
    def per_joint(key):
        return {ARM_GROUPS[group]: float(value) for group, value in control[key].items() if value is not None}

    return ImplicitActuatorCfg(
        joint_names_expr=ARM_JOINTS, stiffness=per_joint("stiffness"), damping=per_joint("damping"),
        armature=per_joint("armature"), effort_limit_sim=per_joint("effort_limit"),
        velocity_limit_sim=per_joint("velocity_limit") or None,
    )


FRANKA_FINGER_OPEN = .04
"""The Franka's default finger joint position, half its open jaw gap [m]; its reset offsets from it."""


def reset_ur10e(env, env_ids, position_range=(.9, 1.1), gripper_offset_range=(-.02, 0.), start_pose="panda"):
    """The Franka task's randomized start -- Panda TCP pose and jaw opening -- reached by UR inverse kinematics.

    The random draws mirror the Franka's two reset events one for one, with the same ranges, shapes
    and order: ``reset_joints_by_scale`` on the arm (a joint-position scale in ``position_range``,
    then a joint-velocity scale, zero for both arms) and ``reset_joints_shared_offset`` on the
    fingers (one opening offset in ``gripper_offset_range`` from the open 0.04 m per finger). So a
    given seed starts both arms at the same tool pose with the same jaw gap, and -- since every
    later reset event then draws from the same point of the random stream -- with the same layout.
    ``position_range=(1, 1)`` and ``gripper_offset_range=(0, 0)`` give the unrandomized default start.

    ``start_pose`` picks the nominal start the variation is applied around:

    - ``"panda"``: the Panda's start tool pose, so both arms start in the same place (observations
      match the Franka's exactly for a given seed);
    - ``"home"``: this arm's configured home posture (``robot.init_state.joint_pos``, e.g. a pose
      saved with ``scripts/ur10e_start_pose_gui.py``). The Franka's random variation of its start --
      the perturbed Panda tool pose against the unperturbed one -- is applied to the home tool pose
      as an offset, so the spread of starts is the Franka's but centred on the hand-set pose.
    """
    state = adapter(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    scale = sample_uniform(position_range[0], position_range[1], (n, 7), device=env.device)
    sample_uniform(0., 0., (n, 7), device=env.device)  # the Franka event's joint-velocity scale: keeps the streams in step
    offset = sample_uniform(gripper_offset_range[0], gripper_offset_range[1], (n, 1), device=env.device)
    reference = (state.home[env_ids] * scale).clamp(state.panda.limits[0], state.panda.limits[1])
    pos, rot, _ = state.panda.forward(reference)
    # The configured home posture seeds the IK, so the branch the arm reaches the pose in (elbow up or
    # down, wrist flipped or not) is chosen by ``robot.init_state.joint_pos`` in the task config.
    seed = state.robot.data.default_joint_pos.torch[env_ids][:, state.arm_ids]
    if start_pose == "home":
        # the Panda's random variation (perturbed start against the unperturbed one), applied to the
        # home posture's tool pose; with position_range (1, 1) the arm starts exactly at home
        panda_pos, panda_rot, _ = state.panda.forward(state.home[env_ids])
        home_pos, home_rot, _ = state.ur.forward(seed)
        pos = home_pos + (pos - panda_pos)
        rot = quat_mul(quat_mul(rot, quat_inv(panda_rot)), home_rot)
    elif start_pose == "panda":
        # The Panda's start pose is a point in the workspace (its base sits at the environment
        # origin); if this arm's base is placed elsewhere, express that pose relative to it, so the
        # tool starts at the same place over the table whatever the base position. The base is
        # translated only, never yawed, so the rotation and the robot-frame axes are unchanged.
        base = state.robot.data.root_pos_w.torch[env_ids] - env.scene.env_origins[env_ids]
        pos = pos - base
    else:
        raise ValueError(f"start_pose must be 'panda' or 'home', got {start_pose!r}")
    q = state.ur.solve(pos, rot, seed, iterations=100)
    p, r, _ = state.ur.forward(q)
    dp, dr = compute_pose_error(p, r, pos, rot, rot_error_type="axis_angle")
    # written so that NaN fails it too (a NaN distance compares False against any threshold)
    if not bool(((dp.norm(dim=-1) <= .002) & (dr.norm(dim=-1) <= .02)).all()):
        raise RuntimeError(f"UR10e reset IK did not converge: position {dp.norm(dim=-1).max().item():.4f} m")
    positions = state.robot.data.default_joint_pos.torch[env_ids].clone()
    positions[:, state.arm_ids] = q
    # The Franka's fingers start at 0.04 m + offset each, i.e. that half jaw gap; the 2F-85's pads at the
    # same half gap, capped at what its open command reaches (0.03996 m at 0.08 rad).
    half_width = (FRANKA_FINGER_OPEN + offset[:, 0]).clamp(max=GRIPPER_OPEN_HALF_WIDTH)
    # Exact planar linkage: half_gap = .0435484 - .03716*(1-cos(q)) - .04343*sin(q).
    lo, hi = torch.zeros_like(half_width), torch.full_like(half_width, .82)
    for _ in range(20):
        mid = (lo + hi) / 2
        gap = OPEN_HALF_WIDTH - .03716 * (1 - mid.cos()) - .04343 * mid.sin()
        lo = torch.where(gap > half_width, mid, lo)
        hi = torch.where(gap > half_width, hi, mid)
    signs = torch.tensor(list(GRIPPER_SIGNS.values()), device=env.device)
    positions[:, state.finger_ids] = ((lo + hi) / 2)[:, None] * signs
    state.robot.write_joint_state_to_sim_index(position=positions, velocity=torch.zeros_like(positions), env_ids=env_ids)
    state.robot.set_joint_position_target_index(target=positions, env_ids=env_ids)
    state.q[env_ids] = reference
    state.qvel[env_ids] = 0.
    state.fresh[env_ids] = True
    state.last_step = -1


class MimicGripperAction(FilteredBinaryGripperAction):
    """The task's filtered binary gripper, with the linkage's followers slaved in software."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)  # column order of the processed targets
        self._follower_cols, leader_ids, self._follower_gain = [], [], []
        for follower, leader in GRIPPER_LEADERS.items():
            if follower not in names:
                continue
            self._follower_cols.append(names.index(follower))
            leader_ids.append(self._asset.find_joints([leader], preserve_order=True)[0][0])
            self._follower_gain.append(GRIPPER_SIGNS[follower] * GRIPPER_SIGNS[leader])
        self._leader_ids = leader_ids
        self._follower_gain = torch.tensor(self._follower_gain, device=self.device)

    def mimic_targets(self):
        """The processed targets with each follower replaced by its knuckle's measured angle."""
        targets = self._processed_actions.clone()
        if self._follower_cols:
            leaders = self._asset.data.joint_pos.torch[:, self._leader_ids]
            targets[:, self._follower_cols] = leaders * self._follower_gain
        return targets

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self.mimic_targets(), joint_ids=self._joint_ids)


@configclass
class MimicGripperActionCfg(FilteredBinaryGripperActionCfg):
    class_type: type = MimicGripperAction


FAULT_EXITS = ("arm_into_table", "arm_self_collision", "finger_touched_slot", "ee_below_table")
"""Terminations the arm causes, as opposed to the task's own outcomes (placed, dropped, ...)."""


def fault_exit(env, term_keys=FAULT_EXITS, grace_steps=2):
    """``1 / step_dt`` on the step an episode ends by one of ``term_keys``, else 0: with weight ``-P`` a
    one-off penalty of ``P`` (the reward manager multiplies by ``step_dt``).

    Without it these exits cost nothing, and a fine-tune learned to end episodes by driving the arm into
    the table (2026-10-27). Exits within the first ``grace_steps`` steps are not charged: only the reset can
    have put the arm there (the saved start pose touches the slot in about 5 % of resets, at step 1-2).
    Terminations are computed before rewards, so the term reads this step's flags.
    """
    manager = env.termination_manager
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_keys:
        if name in manager.active_terms:
            hit |= manager.get_term(name)
    return (hit & (env.episode_length_buf > grace_steps)).float() / env.step_dt


def tool_speed_over(env, max_speed=0.5, base_penalty=15.0, reference_speed=0.75, reference_penalty=35.0,
                    hand_body_name="wrist_3_link", tool_offset=TOOL_POS, robot_cfg=SceneEntityCfg("robot")):
    """Per-step charge while the tool point moves faster than ``max_speed`` [m/s]: ``base_penalty`` just above it,
    rising linearly to ``reference_penalty`` at ``reference_speed`` and on at that slope beyond (user request,
    2026-10-30: 15 per step from 0.5 m/s, 35 at 0.75 m/s, so 80 per m/s); 0 at or below ``max_speed``. Returned
    divided by ``step_dt``, so weight -1 charges the literal amounts per step.

    The tool point is ``hand_body_name`` + ``tool_offset`` (the 2F-85 pad midpoint on the UR arms), so the speed
    includes what the wrist's rotation adds there (``v + w x r``) -- the speed the real arm's Cartesian controller
    caps (``max_linear_velocity`` 0.45 m/s), which the joint-speed limit bounds only indirectly: at 45 % of the joint
    limits the tool still peaked at ~0.75 m/s (2026-10-30, scale 0.03).
    """
    robot = env.scene[robot_cfg.name]
    hand = robot.body_names.index(hand_body_name)
    pose = robot.data.body_link_pose_w.torch[:, hand]
    velocity = robot.data.body_link_vel_w.torch[:, hand]
    lever = quat_apply(pose[:, 3:7], torch.tensor(tool_offset, device=env.device).expand(env.num_envs, -1))
    speed = (velocity[:, :3] + torch.cross(velocity[:, 3:], lever, dim=-1)).norm(dim=-1)
    slope = (reference_penalty - base_penalty) / (reference_speed - max_speed)
    charge = base_penalty + slope * (speed - max_speed)
    return torch.where(speed > max_speed, charge, torch.zeros_like(charge)) / env.step_dt


def joint_vel_out_of_scaled_sim_limit(env, limit_scale=1.0, asset_cfg=SceneEntityCfg("robot")):
    """Isaac Lab's ``joint_vel_out_of_sim_limit`` with every limit multiplied by ``limit_scale``: ends the episode
    when a selected joint's speed exceeds ``limit_scale`` times its actuator's ``velocity_limit_sim``.

    1.0 is the stock term. Below 1 it trains a slower policy without touching the drives, whose own
    velocity limits stay the arm's (``env.terminations.joint_vel_out_of_limit.params.limit_scale=0.7``: 84 deg/s
    shoulders, 126 deg/s elbow and wrists on the UR).
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = torch.full_like(asset.data.joint_vel.torch, torch.inf)
    for actuator in asset.actuators.values():
        limits[:, actuator.joint_indices] = actuator.velocity_limit_sim
    return torch.any(asset.data.joint_vel.torch[:, joint_ids].abs() > limit_scale * limits[:, joint_ids], dim=1)


def never(env, coverage_threshold=None, max_angle=None):
    """A termination that does not fire: stands in for ``placed``/``dropped`` in hand-flown sessions.

    Takes the two terms' own parameters by name: the manager checks a term's signature against
    its params and reads ``**kwargs`` as a mandatory argument called ``params``.

    The task ends the episode on the first opening after a closure (``placed`` on good coverage,
    ``dropped`` otherwise). ``teleop_ur10e --free_run`` swaps this in so a person can open and
    close freely; the terms have to stay registered under their names because
    ``placed_coverage_bonus``, ``placement_success_on_placed`` and ``release_edge_angle`` read
    them by name -- with this function they simply never pay or log.
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
