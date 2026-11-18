"""The UR10e task (``SheetUR10eTwistStudentEnvCfg``) on a UR16e: the same scene, table, layout, base position,
2F-85 gripper and tool point, observations, actions, rewards, events and terminations; only the arm changes.

What the UR16e brings and what is set for it:

- the arm: the stock Isaac Sim UR16e with the UR10e task's 2F-85 mounted as on the UR10e (``ur16e_asset``). The
  arms share joint names, axes, velocity and effort limits and the base and wrist geometry; the upper arm is
  0.4784 m (UR10e 0.6127) and the forearm 0.36 m (0.57155), with their own meshes and masses;
- the base stays at the UR10e's (-0.12, 0.02): sequential IK over 5000 recorded UR10e tool poses (model_1924, 20
  episodes, randomised layouts) tracks every one with no error from there, elbow 45-139 degrees; a base nearer
  the layout folds the elbow further (to 148 at x = -0.06) and lowers the manipulability;
- the start pose: the UR10e's saved start tool pose solved on the UR16e (``ur16e_start_pose.json``, elbow -117
  degrees; the reset's random spread keeps it within -133..-100);
- self-collision: the UR10e's capsules and pairs with the UR16e's link lengths, and its earlier elbow fold
  (162 degrees, limit 153; ``ur16e_self_collision``). ``arm_into_table`` reads the arm's own collision meshes;
- the arm drives: the UR10e task's (``DEFAULT_ARM_CONTROL``, 400/80, 87/12 Nm -- the setting matched to the
  Franka's per-step response), read from ``ur16e_arm_control.json`` when one is saved, so tuning the UR10e does
  not change this arm.
"""

import pathlib

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from .sheet_ur10e_env_cfg import SheetUR10eTwistStudentEnvCfg
from .ur10e_mdp import ARM_JOINTS, GRIPPER_SIGNS, arm_actuator, load_arm_control, load_start_pose
from .ur16e_asset import UR16E_USD_PATH, spawn_ur16e
from .ur16e_self_collision import ur16e_arm_self_collision

UR16E_START_POSE_FILE = pathlib.Path(__file__).with_name("ur16e_start_pose.json")
"""The UR16e's start posture (``save_start_pose`` format): IK of the UR10e task's saved start tool pose."""

UR16E_ARM_CONTROL_FILE = pathlib.Path(__file__).with_name("ur16e_arm_control.json")
"""A tuned arm control for the UR16e (``save_arm_control`` format); without one the task's defaults apply."""


@configclass
class SheetUR16eTwistStudentEnvCfg(SheetUR10eTwistStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        robot = self.scene.robot
        robot.spawn = robot.spawn.replace(func=spawn_ur16e, usd_path=UR16E_USD_PATH, variants=None)
        start = load_start_pose(UR16E_START_POSE_FILE)
        if start is None:
            raise FileNotFoundError(f"No UR16e start pose at {UR16E_START_POSE_FILE}")
        robot.init_state.joint_pos = dict(zip(ARM_JOINTS, start))
        robot.init_state.joint_pos.update({name: 0. for name in GRIPPER_SIGNS})
        # the start variation is applied around this arm's own start posture, which seeds the reset IK
        self.events.reset_robot_arm_joints.params["start_pose"] = "home"
        arm_control = load_arm_control(UR16E_ARM_CONTROL_FILE)
        robot.actuators["arm"] = arm_actuator(arm_control)
        self.actions.joint.arm_action.controller.ik_params = {"lambda_val": arm_control["ik_lambda"]}
        self.terminations.arm_self_collision = TerminationTermCfg(func=ur16e_arm_self_collision, params={"margin": .01})
