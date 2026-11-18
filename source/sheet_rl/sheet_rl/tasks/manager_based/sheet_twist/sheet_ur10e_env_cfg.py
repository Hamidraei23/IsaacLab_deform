"""The twist-student cloth task with UR10e + Robotiq 2F-85 and checkpoint adaptation.

Everything the Franka task (``SheetPlatformTwistStudentEnvCfg``) defines is inherited: the scene,
the sheet-pose and IK-scale randomisation events, every reward term with its weight, and the
episode outcomes (one gripper cycle; ``placed``/``dropped`` on the commanded release). This file
only swaps the arm and re-points the robot-specific parameters at it, so that a policy trained on
the Franka's ``action_history`` group reads the same 49 values here:

- the sheet points and the goal are measured from this arm's tool point (``wrist_3_link`` +
  ``TOOL_POS``), robot-frame axes, as they are from the Franka's ``panda_hand`` + 0.107 m;
- the ``finger_width`` slot holds the physical pad gap (``gripper_opening`` over
  ``finger_pad_geometry``), 0.08 m open, not the 2F-85's knuckle angle -- see the note in
  ``observations.finger_width`` on what the Franka checkpoints actually read there;
- the IK action keeps the Franka's scale (0.1, randomised per episode by the inherited
  ``randomize_arm_scale`` event when a range is given), so the applied-action channels mean the
  same tool displacement per unit.

The ``policy`` group (the teacher's 59 values, read by the asymmetric SAC critic) keeps its 9+9
joint channels through a virtual-Panda IK adapter (``ur10e_mdp.FrankaObservationAdapter``); that
is an approximation, not a measurement of this arm.
"""

from copy import deepcopy

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG
from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyPropertiesCfg

from ..sheet_rl.mdp import simulation_diverged
from .sheet_platform_twist_student_env_cfg import SheetPlatformTwistStudentEnvCfg
from .self_collision import arm_into_table, arm_self_collision
from . import ur10e_gripper
from .ur10e_asset import CLOTH_FINGERS, spawn_ur10e
from .ur10e_mdp import (
    ARM_JOINTS, GRIPPER_CLOSE_ANGLE, GRIPPER_OPEN_ANGLE, GRIPPER_SIGNS, TOOL_POS, TOOL_ROT,
    MimicGripperActionCfg, arm_actuator, fault_exit, franka_joint_pos_rel, franka_joint_vel,
    joint_vel_out_of_scaled_sim_limit, load_arm_control,
    load_start_pose, reset_ur10e, tool_speed_over,
)


TABLE_BACK = -0.35
"""Back edge of the table in the UR10e task [m, robot-root x of the Franka task]: 13 cm clear of this
arm's base, whose rear edge is at x = -0.215."""


@configclass
class SheetUR10eTwistStudentEnvCfg(SheetPlatformTwistStudentEnvCfg):
    # Inner contact faces of the 6.0 Isaac Sim 2F-85 fingertip meshes, in body coordinates [m].
    finger_pad_geometry: dict = {
        "left_inner_finger": (0., -.0435484, .14),
        "right_inner_finger": (0., .0435484, .14),
    }

    def __post_init__(self):
        super().__post_init__()
        self.scene = self.scene.newton_mjwarp_vbd_proxy.copy()
        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot = self.scene.robot
        # Keep the reference base frame: sheet points, goals and twist actions all use it.
        robot.init_state.rot = (0., 0., 0., 1.)
        # Base placement and home posture (user choice 2026-10-26, from open-loop replays of a
        # recorded Franka episode on this arm). The tool starts at the Panda's start pose over the
        # table whatever the base position (reset_ur10e expresses that pose relative to the base),
        # so the base position is the knob that sets the arm's posture at a given tool pose:
        # 12 cm behind and 2 cm to the +y side of the Franka's base: the minimum of a 9-point sweep
        # (x -0.12..-0.24, y -0.06..0.02) of the replay's tool-tracking error against the Franka,
        # confirmed over three runs each -- grasp phase 120 mm, carry 176 mm, the sheet held for
        # 83 steps and lifted for the Franka's full lift reward (5.14) every time; at (-0.16,-0.02)
        # it was 141 / 198 mm and 4.8, from the base at the origin the series held for 4 steps,
        # and the old elbow-down home (-0.35, -1.8, 2.1, -1.9, -1.57, 0) put the elbow 0.40 m
        # under the table and folded the arm by step 44. The approach phase (first 30 steps) is
        # ~17.5 mm wherever the base is. The joint values are the IK branch the reset should land
        # on -- elbow-up on the far shoulder side, eight branches reach the start pose -- taken
        # from the reset of the confirming runs; they seed the solver.
        robot.init_state.pos = (-0.12, 0.02, 0.0)
        # The table's back edge is at x = -0.15 (a 1.3 m box centred at x = 0.5), sized for the Franka,
        # whose base sits at x = 0. This base (radius 0.095 m) at x = -0.12 stood a third over that edge,
        # so the table is extended backwards to TABLE_BACK for this arm: the whole base stands on it and
        # nothing else moves (the sheet, slot and platform are all at x > 0.2). The visible copy of the
        # table (the goal-success markers, drawn at the table's pose) gets the same size.
        front = self.scene.table.init_state.pos[0] + self.scene.table.spawn.size[0] / 2
        size = (front - TABLE_BACK, *self.scene.table.spawn.size[1:])
        self.scene.table = self.scene.table.replace(
            spawn=self.scene.table.spawn.replace(size=size),
            init_state=self.scene.table.init_state.replace(pos=((front + TABLE_BACK) / 2, *self.scene.table.init_state.pos[1:])),
        )
        markers = self.commands.deformable_pose.success_visualizer_cfg.markers
        for name in markers:
            markers[name] = markers[name].replace(size=size)
        robot.init_state.joint_pos = dict(zip(ARM_JOINTS, [-2.723, -1.415, -2.076, 2.516, -1.326, -2.829]))
        # A start pose set by hand in scripts/ur10e_start_pose_gui.py (ur10e_start_pose.json beside
        # ur10e_mdp.py) replaces the home posture and becomes the nominal start; without one the arm
        # starts where the Franka does. Delete the file to go back.
        saved_start = load_start_pose()
        if saved_start is not None:
            robot.init_state.joint_pos = dict(zip(ARM_JOINTS, saved_start))
        robot.init_state.joint_pos.update({name: 0. for name in GRIPPER_SIGNS})
        robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.)
        robot.spawn.func = spawn_ur10e
        # Explicitly drive the linkage's six hinges. The PhysX mimic APIs on this asset
        # are not a portable actuator contract for the Newton rigid/cloth coupler.
        # The Franka task's drives, gains AND effort limits (400/80; 87 Nm shoulders/elbow, 12 Nm
        # wrists), so one IK step moves this tool as it moves the Franka's (user request, 2026-10-26:
        # keep the setting with the smallest tracking error). Measured with
        # scripts/ik_axis_response.py and compare_first_steps.py at scale 0.05 / 0.03:
        # - gains: a +1 step gives 4.7 mm / 0.3 deg here against the Franka's 4.8 mm / 0.3 deg
        #   (the stock 1200/100 gave 10.9 mm / 0.7 deg and joint speeds of 6-18 rad/s under the
        #   policy);
        # - effort limits: the policy commands ~23 deg/step rotations, which the Franka's 12 Nm
        #   wrists saturate and ramp (0.2 -> 2.1 deg/step); with the UR10e's real 56 Nm the ramp is
        #   immediate (1.6 -> 2.2) and the open-loop tracking error over 15 Franka actions is
        #   19/0/14 mm; with the Franka's limits it is 16/1/8 mm and the rotation ramp matches
        #   (0.7 -> 2.0). Lower wrist limits (8, 6 Nm) slow the ramp further but add 1-2 deg/step of
        #   off-axis rotation and 10-25 mm of cross-coupled translation, so 12 Nm is the minimum.
        # The real UR10e wrists deliver 56 Nm; this is a transfer setting, not a model of the arm.
        # The policy was trained on the Franka's response and relies on it filtering its large
        # commands, so the arm is matched to the policy rather than the policy to the arm.
        # The arm's drives, per joint group, from scripts/ur10e_arm_tuning_gui.py's saved file
        # (ur10e_arm_control.json beside ur10e_mdp.py) or DEFAULT_ARM_CONTROL without one.
        arm_control = load_arm_control()
        robot.actuators = {
            "arm": arm_actuator(arm_control),
            # Force and pose control of the 2F-85: tune it in ur10e_gripper.py, which also
            # states the restrictions and validates them when this config is built.
            **ur10e_gripper.actuators(),
        }
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/base_link", debug_vis=False,
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/wrist_3_link", name="end_effector",
                offset=OffsetCfg(pos=TOOL_POS, rot=TOOL_ROT),
            )],
        )
        # Same scale as the Franka task, so one unit of action is the same tool displacement
        # (teleop_ur10e divides the scale out, so its key step sizes do not depend on it).
        # The body offset carries the tool POSITION only. The relative rotation command is applied
        # about the robot frame's axes whatever the tool frame's orientation, but the IK action
        # rotates the angular rows of its Jacobian by the offset's rotation, so with TOOL_ROT
        # (-90 deg about z) a commanded rotation about x came out about y and y about -x (measured
        # 2026-10-26 with scripts/ik_axis_response.py). With no offset rotation all six channels
        # act on the UR10e as they do on the Franka. TOOL_ROT stays on ee_frame and the reset IK.
        self.actions.joint.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls",
                                                   ik_params={"lambda_val": arm_control["ik_lambda"]}),
            scale=self.actions.joint.arm_action.scale,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TOOL_POS),
        )
        # Knuckles are commanded; the four followers copy them (see GRIPPER_LEADERS).
        self.actions.joint.gripper_action = MimicGripperActionCfg(
            asset_name="robot", joint_names=list(GRIPPER_SIGNS),
            open_command_expr={name: GRIPPER_OPEN_ANGLE * sign for name, sign in GRIPPER_SIGNS.items()},
            close_command_expr={name: GRIPPER_CLOSE_ANGLE * sign for name, sign in GRIPPER_SIGNS.items()},
            threshold=0., positive_threshold=True,
        )
        self.actions.default = self.actions.joint
        # One event in place of the Franka's arm and finger resets, drawing the same random numbers
        # with the same ranges (taken from them), so a seed gives both arms the same start
        self.events.reset_robot_arm_joints = EventTermCfg(func=reset_ur10e, mode="reset", params={
            "position_range": self.events.reset_robot_arm_joints.params["position_range"],
            "gripper_offset_range": self.events.reset_robot_gripper_joints.params["position_range"],
            "start_pose": "panda" if saved_start is None else "home",
        })
        self.events.reset_robot_gripper_joints = None
        self.events.randomize_arm_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["shoulder_link", "upper_arm_link", "forearm_link", "wrist_[123]_link"],
        )
        # Preserve the trained network's 59 inputs, including meaningful 9+9 joint-state channels.
        self.observations.policy.joint_pos = ObservationTermCfg(func=franka_joint_pos_rel)
        self.observations.policy.joint_vel = ObservationTermCfg(func=franka_joint_vel)
        # A critic trained on this arm (FastSAC: agent.critic_obs_groups=[ur_critic]) reads its own joint
        # state -- the six arm joints and the 2F-85's driven knuckle -- instead of the virtual Panda's,
        # beside the policy group's sheet points, target and last action: 7 + 7 + 27 + 7 + 7 = 55.
        ur_critic = deepcopy(self.observations.policy)
        joints = SceneEntityCfg("robot", joint_names=[*ARM_JOINTS, "finger_joint"], preserve_order=True)
        ur_critic.joint_pos = ObservationTermCfg(func=isaaclab_mdp.joint_pos_rel, params={"asset_cfg": joints})
        ur_critic.joint_vel = ObservationTermCfg(func=isaaclab_mdp.joint_vel_rel, params={"asset_cfg": joints})
        self.observations.ur_critic = ur_critic
        # finger_width is the pad gap on both arms (gripper_opening); here the asset resolves to the
        # 2F-85's driven knuckle and the gap comes from finger_pad_geometry. The legacy group (sum
        # of the Franka's nine joints) has no meaning on this arm.
        self.observations.action_history.finger_width.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="finger_joint")
        self.observations.action_history_legacy = None
        # the tool-relative terms measure from this arm's tool point
        for name in ("deformable_sampled_points", "target_position"):
            getattr(self.observations.action_history, name).params.update(
                hand_body_name="wrist_3_link", tool_offset=TOOL_POS
            )
        for name in ("table_clearance", "ee_speed", "grasp_stage", "placement"):
            term = getattr(self.rewards, name)
            term.params["hand_body_name"] = "wrist_3_link"
            if name != "ee_speed":
                term.params["grasp_offset"] = TOOL_POS
        for name in ("grasp_stage", "placement", "coverage"):
            getattr(self.rewards, name).params["finger_cfg"] = SceneEntityCfg("robot", joint_names="finger_joint")
        self.rewards.grasp_stage.params.update(
            left_finger_body_name="left_inner_finger", right_finger_body_name="right_inner_finger",
        )
        # Extraction paid 10x the Franka task's on this arm (user request, 2026-10-27): 2000 x weight 1/35 = 57
        # on the extraction step instead of 5.7, against ~160 of lift shaping by then and 2840 for placing.
        self.rewards.grasp_stage.params["extraction_bonus"] = 10 * self.rewards.grasp_stage.params["extraction_bonus"]
        for name in ("released_high", "held_too_low"):
            getattr(self.terminations, name).params.update(hand_body_name="wrist_3_link", grasp_offset=TOOL_POS)
        self.terminations.released_high.params["finger_cfg"] = SceneEntityCfg("robot", joint_names="finger_joint")
        self.terminations.finger_touched_slot.params["finger_body_names"] = tuple(self.finger_pad_geometry)
        # The joint-speed termination binds the arm only: its caps are the real UR10e limits (120 deg/s shoulders,
        # 180 deg/s elbow and wrists). The 2F-85 joints overshoot their 6 rad/s cap whenever the fingers snap shut
        # or open (up to 1.8x, best_ur10e_ppo/model_925) -- set by the gripper drive, not by the policy, which only
        # commands open or close -- so counting them would end grasps rather than fast arm motion.
        self.terminations.joint_vel_out_of_limit.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
        # The same test with a factor on the limits (1.0: the stock term), so a run can train within a fraction of them
        # (env.terminations.joint_vel_out_of_limit.params.limit_scale=0.7, user request 2026-10-29).
        self.terminations.joint_vel_out_of_limit.func = joint_vel_out_of_scaled_sim_limit
        self.terminations.joint_vel_out_of_limit.params["limit_scale"] = 1.0
        # The one-gripper-cycle outcomes (placed/dropped on the commanded release) are inherited
        # as they are: they pay the success and are what the SAC runner's metrics, probes and
        # curriculum gate on. teleop_ur10e --free_run switches them off for hand-flown sessions.
        # The asset ships with self-collision off and joint limits of +/-360 deg (elbow +/-180,
        # narrowed to +/-160 by spawn_ur10e), so nothing in the plant stops the arm folding
        # through itself. See self_collision.py.
        self.terminations.arm_self_collision = TerminationTermCfg(func=arm_self_collision, params={"margin": .01})
        # Any arm link touching the table, or below the level of its top, ends the episode as a failure.
        self.terminations.arm_into_table = TerminationTermCfg(func=arm_into_table, params={"margin": .01})
        # A blown-up cloth (non-finite state, about 2 episodes in 50 on this arm) ends its episode; the
        # task's environment class (DivergenceTolerantEnv) zeroes that step's non-finite rewards.
        self.terminations.diverged = TerminationTermCfg(func=simulation_diverged)
        # Ending an episode by an exit the arm caused costs 100 -- above what 90 % of failed episodes earn
        # (82-84, model_4000 on this arm, 512 episodes), far below a success (700-5800). Without it a
        # fine-tune learned to end episodes in the table. Dropping the sheet stays free, as on the Franka.
        self.rewards.fault_exit = RewardTermCfg(func=fault_exit, weight=-100.0)
        # A per-step charge while the tool point moves faster than 0.5 m/s: 15 just above, 35 at 0.75 m/s, linear
        # (user request 2026-10-30). Off by default, so the runs before it are unchanged; weight -1 turns it on at
        # those literal amounts (env.rewards.tool_speed_over.weight=-1.0).
        self.rewards.tool_speed_over = RewardTermCfg(
            func=tool_speed_over, weight=0.0,
            params={"max_speed": 0.5, "base_penalty": 15.0, "reference_speed": 0.75, "reference_penalty": 35.0})
        # Couple only the two inner fingers (box pads, see ur10e_asset.box_fingers) to the cloth, as the
        # Franka couples its hand and fingers -- not every 2F-85 body and the wrist, which cost 3x the step time.
        self.sim.physics = deepcopy(self.sim.physics.newton_mjwarp_vbd_proxy)
        proxy = self.sim.physics.solver_cfg.proxies[0]
        proxy.bodies = [rf"/World/envs/env_.*/Robot/ee_link/Robotiq_2F_85/({'|'.join(CLOTH_FINGERS)})",
                        r"/World/envs/env_.*/MannequinArm", r"/World/envs/env_.*/Slot(Neg|Pos)Y"]
