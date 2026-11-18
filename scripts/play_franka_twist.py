"""Compare cached Franka RSL-RL weights under joint and converted-twist control.

Examples (from the sheet_rl project, after active_isaac):
    PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control joint --real-time
    PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control twist --real-time

Both runs use the same 36 observations and 8 raw policy outputs. Only the arm
actuation changes: q_target -> J(q)(q_target-q)/dt -> damped IK -> joint targets.
The 6D twist is expressed at the observed tool point in the robot base frame.
Joint-position PD gains, the gripper, rewards and episode endings are preserved.
Twist limits default to disabled so the first comparison isolates conversion.
Use --max-linear-speed / --max-angular-speed to test limits separately.

Each run prints episode results and writes JSON including tool-motion metrics.
--episodes 0 watches continuously; Ctrl-C saves completed episodes.
The joint and twist modes are an inference adapter, not a policy trained with Cartesian actions.

--control student plays a policy that *is* trained with Cartesian actions: the DAgger twist student
(scripts/train_dagger_twist.py) or its PPO continuation, in Lift-Cube-Franka-Twist-v0 (35 observations,
6D delta pose + gripper). Its raw action is the twist; the speed limits apply to it the same way.
Without --checkpoint it plays the newest checkpoint of the newest lift_cube_franka_twist run:
    PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control student --real-time
"""

import argparse
import contextlib
import functools
import json
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", choices=("joint", "twist", "student"), default="twist")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, choices=(1,), default=1)
    parser.add_argument("--rl_library", choices=("rsl_rl",), default="rsl_rl")
    parser.add_argument(
        "--task",
        choices=(
            "IsaacContrib-Lift-Cube-Franka",
            "Lift-Cube-Franka-Twist-v0",
            "Lift-Cube-Franka-Twist-TeacherGains-v0",
            "Lift-Cube-Franka-Twist-Platform-v0",
            "Lift-Cube-Franka-Twist-PickPlace-v0",
        ),
        help="Default: IsaacContrib-Lift-Cube-Franka (joint, twist) or Lift-Cube-Franka-Twist-v0 (student).",
    )
    parser.add_argument(
        "--checkpoint",
        help="Default: the cached pretrained weights (joint, twist) or the newest twist-student checkpoint (student).",
    )
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-linear-speed", type=float, default=0.0, help="Twist norm limit [m/s]; 0 disables.")
    parser.add_argument("--max-angular-speed", type=float, default=0.0, help="Twist norm limit [rad/s]; 0 disables.")
    parser.add_argument(
        "--platform",
        type=float,
        default=0.0,
        help="Stand the cube on a platform exactly this tall [m]. In the platform task this fixes its random "
        "height; 0 keeps the task as it is.",
    )
    parser.add_argument(
        "--platform_size", type=float, default=0.10, help="Side of an added platform [m] (the platform task keeps its own)."
    )
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    if args.episodes < 0 or args.damping <= 0 or min(args.max_linear_speed, args.max_angular_speed) < 0:
        parser.error("episodes and speed limits must be nonnegative; damping must be positive")
    if args.platform < 0 or args.platform_size <= 0:
        parser.error("--platform must be nonnegative and --platform_size positive")
    student = args.control == "student"
    if args.task is None:
        args.task = "Lift-Cube-Franka-Twist-v0" if student else "IsaacContrib-Lift-Cube-Franka"
    if student != args.task.startswith("Lift-Cube-Franka-Twist"):
        parser.error("--control student needs a twist task; joint and twist need IsaacContrib-Lift-Cube-Franka")
    if args.checkpoint is None and student:
        args.checkpoint = str(newest_checkpoint(root / "logs/rsl_rl/lift_cube_franka_twist", parser))
    elif args.checkpoint is None:
        args.checkpoint = str(root / ".pretrained_checkpoints/rsl_rl/Isaac-Lift-Cube-Franka-v0/checkpoint.pt")
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = root / f"evaluations/franka_twist/{args.control}_{stamp}.json"
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output path")
    if not any(value == "--viz" or value.startswith("--viz=") for value in remaining):
        remaining += ["--viz", "newton_gl"]
    return args, remaining


def newest_checkpoint(experiment: Path, parser) -> Path:
    """Highest-numbered model_<it>.pt in the most recently modified run folder of an experiment."""
    def models(run):
        return [path for path in run.glob("model_*.pt") if path.stem.split("_")[1].isdigit()]

    runs = sorted((run for run in experiment.glob("*/") if models(run)), key=lambda run: run.stat().st_mtime)
    if not runs:
        parser.error(f"No twist-student checkpoints under {experiment}; pass --checkpoint")
    return max(models(runs[-1]), key=lambda path: int(path.stem.split("_")[1]))


def set_platform(env_cfg, height: float, size: float) -> None:
    """Evaluation: stand the cube on a platform exactly ``height`` tall, in any of the lift tasks.

    Uses the platform of ``Lift-Cube-Franka-Twist-Platform-v0`` (see ``platform_mdp``): a kinematic box
    moved under the cube, wherever its reset put it, by an event that runs after the cube's own. In that
    task this fixes the otherwise random height (``size`` is then the task's own); elsewhere it adds the
    platform. In the pick-and-place task the second platform gets the same height. Rewards are not
    touched by this function.
    """
    from sheet_rl.tasks.manager_based.franka_lift_twist.platform_mdp import platform_cfg, platform_event

    if getattr(env_cfg.events, "place_cube_on_platform", None) is None:
        env_cfg.scene.platform = platform_cfg(size, env_cfg.scene.object.init_state.pos[:2])
    # replacing an existing term keeps its place in the event order; a new one is appended last
    env_cfg.events.place_cube_on_platform = platform_event((height, height))
    if getattr(env_cfg.events, "place_target_platform", None) is not None:
        env_cfg.events.place_target_platform.params["height_range"] = (height, height)


@contextlib.contextmanager
def install_adapter(env_cfg, args):
    # Import simulation-dependent modules only after launch_simulation entered.
    import torch
    from franka_twist_math import joint_target_to_twist, limit_twist, shift_jacobian, twist_to_joint_target
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, quat_apply, quat_inv

    from isaaclab_tasks.contrib.lift.mdp.rewards import object_goal_distance
    from sheet_rl.tasks.manager_based.franka_lift_twist.platform_mdp import (
        object_goal_distance_above_support,
        support_height,
    )
    from sheet_rl.tasks.manager_based.franka_lift_twist.pick_place_mdp import pick_goal_tracking

    completed = []
    current = None
    previous_position = previous_velocity = None
    dt = env_cfg.sim.dt * env_cfg.decimation
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(
        policy_dt_s=dt,
        physics=type(env_cfg.sim.physics).__name__,
        twist_frame="robot base",
        twist_point="ee_frame/end_effector",
        conversion="J(q) @ (q_target - q) / policy_dt; damped inverse; no null-space command",
    )
    if args.control == "student":
        metadata.update(
            twist_point="IK-Rel tool offset on panda_hand",
            conversion="policy outputs the twist: delta pose = ik_scale * action; Isaac Lab DLS IK each physics step",
        )

    class ConvertedArmAction(JointPositionAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            if self.action_dim != 7:
                raise ValueError("Expected seven Franka arm joints")
            body_ids, _ = self._asset.find_bodies("panda_hand")
            self.body_id = body_ids[0]
            self.jacobian_body_id = self.body_id - 1 if self._asset.is_fixed_base else self.body_id
            self.jacobian_joint_ids = self._joint_ids + self._asset.num_base_dofs
            # Use exactly the same TCP as the task's frame sensor (0.1034 m).
            tcp = env.cfg.scene.ee_frame.target_frames[0].offset.pos
            self.tool_offset = torch.tensor(tcp, device=self.device).expand(self.num_envs, -1)
            self.requested_twist = torch.zeros(self.num_envs, 6, device=self.device)
            self.commanded_twist = self.requested_twist.clone()
            self.applied_target = self.processed_actions.clone()

        def process_actions(self, actions):
            super().process_actions(actions)
            robot = self._asset
            joint_position = robot.data.joint_pos.torch[:, self._joint_ids]
            jacobian = robot.data.body_link_jacobian_w.torch[:, self.jacobian_body_id, :, :]
            jacobian = jacobian[:, :, self.jacobian_joint_ids].clone()
            offset_world = quat_apply(robot.data.body_quat_w.torch[:, self.body_id], self.tool_offset)
            jacobian = shift_jacobian(jacobian, offset_world)
            rotation = matrix_from_quat(quat_inv(robot.data.root_quat_w.torch))
            jacobian = torch.cat((rotation @ jacobian[:, :3, :], rotation @ jacobian[:, 3:, :]), dim=1)
            self.requested_twist = joint_target_to_twist(jacobian, self.processed_actions, joint_position, dt)
            self.commanded_twist = limit_twist(self.requested_twist, args.max_linear_speed, args.max_angular_speed)
            if args.control == "twist":
                self.applied_target = twist_to_joint_target(
                    jacobian, self.commanded_twist, joint_position, dt, args.damping
                )
            else:
                self.applied_target = self.processed_actions

        def apply_actions(self):
            self._asset.set_joint_position_target_index(target=self.applied_target, joint_ids=self._joint_ids)

    class StudentTwistAction(DifferentialInverseKinematicsAction):
        """The student's own twist action, recorded and optionally speed-limited like the converted one."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.requested_twist = torch.zeros(self.num_envs, 6, device=self.device)
            self.commanded_twist = self.requested_twist.clone()
            self.applied_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        def process_actions(self, actions):
            super().process_actions(actions)
            self.requested_twist = self.processed_actions / dt
            self.commanded_twist = limit_twist(self.requested_twist, args.max_linear_speed, args.max_angular_speed)
            if args.max_linear_speed > 0 or args.max_angular_speed > 0:
                ee_position, ee_quat = self._compute_frame_pose()
                self._ik_controller.set_command(self.commanded_twist * dt, ee_position, ee_quat)

        def apply_actions(self):
            super().apply_actions()
            self.applied_target = self._asset.data.joint_pos_target.torch[:, self._joint_ids].clone()

    old_action_class = env_cfg.actions.arm_action.class_type
    env_cfg.actions.arm_action.class_type = StudentTwistAction if args.control == "student" else ConvertedArmAction
    if args.platform > 0:
        set_platform(env_cfg, args.platform, args.platform_size)
        metadata.update(platform_height_m=args.platform, platform_size_m=args.platform_size)

    def save():
        count = len(completed)
        summary = {
            "episodes": count,
            "success_10cm": sum(ep["success_10cm"] for ep in completed),
            "success_5cm": sum(ep["success_5cm"] for ep in completed),
            "timeouts": sum(ep["timeout"] for ep in completed),
            "drops": sum(ep["terminated"] for ep in completed),
            "final_within_10cm": sum(ep["final_success_10cm"] for ep in completed),
        }
        if completed and "place_success" in completed[0]:
            summary.update(
                pick_success=sum(ep["pick_success"] for ep in completed),
                place_success=sum(ep["place_success"] for ep in completed),
                placed_at_end=sum(ep["placed_at_end"] for ep in completed),
            )
        if count:
            for key in ("min_distance_m", "final_distance_m", "ee_speed_rms_m_s", "ee_accel_rms_m_s2"):
                summary["mean_" + key] = sum(ep[key] for ep in completed) / count
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"config": metadata, "summary": summary, "episodes": completed}, indent=2) + "\n"
        )
        return summary

    # The platform tasks' goal terms override __call__, so each class gets its own wrapper.
    goal_terms = (object_goal_distance, object_goal_distance_above_support, pick_goal_tracking)
    original_rewards = {term: term.__dict__["__call__"] for term in goal_terms}

    def measured_reward(self, env, *reward_args, **kwargs):
        result = original_rewards[type(self)](self, env, *reward_args, **kwargs)
        if self._track_success:
            measure(env)
        return result

    def measure(env):
        nonlocal current, previous_position, previous_velocity
        robot = env.scene["robot"]
        cube = env.scene["object"].data.root_pos_w.torch[0]
        command = env.command_manager.get_command("object_pose")
        target, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
        distance = float(torch.linalg.vector_norm(target[0] - cube))
        position = env.scene["ee_frame"].data.target_pos_w.torch[0, 0].clone()
        arm = env.action_manager.get_term("arm_action")
        if current is None:
            current = dict(
                episode=len(completed) + 1,
                steps=0,
                min_distance_m=float("inf"),
                success_10cm=False,
                success_5cm=False,
                steps_within_10cm=0,
                target=target[0].tolist(),
                start_cube=cube.tolist(),
                speed_squared_sum=0.0,
                accel_squared_sum=0.0,
                trace=[],
            )
            current["support_height_m"] = float(support_height(env)[0])
            if "platform" in env.scene.keys():
                platform = env.scene["platform"].data.root_pos_w.torch[0]
                current["platform_xy_offset_m"] = float(torch.linalg.vector_norm(platform[:2] - cube[:2]))
        current["steps"] += 1
        goal_term = env.command_manager.get_term("object_pose")
        placing = hasattr(goal_term, "phase") and bool(goal_term.phase[0] == 1)
        if hasattr(goal_term, "phase"):
            # pick and place: the goal has moved to the second platform once the pick succeeded, so the
            # air-goal metrics stop at the switch and the placement is tracked on its own
            current.setdefault("pick_success", False)
            current.setdefault("place_success", False)
            current.setdefault("min_place_distance_m", float("inf"))
            if placing and not current["pick_success"]:
                current.update(pick_success=True, pick_step=current["steps"], success_10cm=True)
            if placing:
                current["min_place_distance_m"] = min(current["min_place_distance_m"], distance)
                current["place_success"] |= bool(goal_term.released()[0])
            current["placed_at_end"] = bool(goal_term.placed()[0])
        if not placing:
            current["min_distance_m"] = min(current["min_distance_m"], distance)
            # the task's 4 cm, measured from what the cube stands on (platform top, or the table)
            lifted = float(cube[2]) > 0.04 + current["support_height_m"]
            current["success_10cm"] |= lifted and distance < 0.10
            current["success_5cm"] |= lifted and distance < 0.05
            current["steps_within_10cm"] += int(lifted and distance < 0.10)
            current["final_success_10cm"] = lifted and distance < 0.10
            current["final_distance_m"] = distance
        else:
            current["final_success_10cm"] = True  # the air goal was reached; it is gone now
        if previous_position is not None:
            velocity = (position - previous_position) / dt
            current["speed_squared_sum"] += float(velocity.square().sum())
            if previous_velocity is not None:
                current["accel_squared_sum"] += float(((velocity - previous_velocity) / dt).square().sum())
            previous_velocity = velocity
        previous_position = position
        if current["steps"] % 5 == 0:
            current["trace"].append(
                dict(
                    step=current["steps"],
                    cube=cube.tolist(),
                    ee=position.tolist(),
                    distance_m=distance,
                    raw_policy_action=env.action_manager.action[0].tolist(),
                    requested_twist=arm.requested_twist[0].tolist(),
                    commanded_twist=arm.commanded_twist[0].tolist(),
                    applied_joint_target=arm.applied_target[0].tolist(),
                    joint_pos=robot.data.joint_pos.torch[0].tolist(),
                    hand_quat_w=robot.data.body_quat_w.torch[0, robot.body_names.index("panda_hand")].tolist(),
                    cube_quat_w=env.scene["object"].data.root_quat_w.torch[0].tolist(),
                    goal_phase=int(goal_term.phase[0]) if hasattr(goal_term, "phase") else 0,
                    goal=target[0].tolist(),
                )
            )

    original_step = ManagerBasedRLEnv.step

    def measured_step(self, action):
        nonlocal current, previous_position, previous_velocity
        result = original_step(self, action)
        if bool(result[2][0]) or bool(result[3][0]):
            current["terminated"] = bool(result[2][0])
            current["timeout"] = bool(result[3][0])
            current["logged_success"] = float(result[4].get("log", {}).get("Metrics/success_rate", -1))
            current["ee_speed_rms_m_s"] = (current.pop("speed_squared_sum") / max(1, current["steps"] - 1)) ** 0.5
            current["ee_accel_rms_m_s2"] = (current.pop("accel_squared_sum") / max(1, current["steps"] - 2)) ** 0.5
            completed.append(current)
            print(
                f"[{args.control}] episode {len(completed)}: success={current['success_10cm']} "
                f"closest={100 * current['min_distance_m']:.1f} cm "
                f"final={100 * current['final_distance_m']:.1f} cm "
                f"EE speed RMS={current['ee_speed_rms_m_s']:.3f} m/s"
                + (
                    f" | placed={current['place_success']} closest to place="
                    f"{100 * current['min_place_distance_m']:.1f} cm"
                    if "place_success" in current
                    else ""
                ),
                flush=True,
            )
            current = previous_position = previous_velocity = None
            save()
            if args.episodes and len(completed) >= args.episodes:
                self.close()
                raise KeyboardInterrupt  # The stock playback loop handles this and exits.
        return result

    old_jit, old_onnx = OnPolicyRunner.export_policy_to_jit, OnPolicyRunner.export_policy_to_onnx
    for term in goal_terms:
        term.__call__ = functools.wraps(original_rewards[term])(measured_reward)
    ManagerBasedRLEnv.step = measured_step
    OnPolicyRunner.export_policy_to_jit = lambda *a, **kw: None
    OnPolicyRunner.export_policy_to_onnx = lambda *a, **kw: None
    try:
        yield
    finally:
        for term, original in original_rewards.items():
            term.__call__ = original
        ManagerBasedRLEnv.step = original_step
        OnPolicyRunner.export_policy_to_jit, OnPolicyRunner.export_policy_to_onnx = old_jit, old_onnx
        env_cfg.actions.arm_action.class_type = old_action_class
        print("Comparison results:", json.dumps(save()), flush=True)
        print(f"Saved {args.output}", flush=True)


def main():
    args, remaining = parse_args()
    if args.control == "student":
        import sheet_rl.tasks  # noqa: F401  (registers the twist tasks)
    import warp as wp

    wp.config.enable_backward = False
    import isaaclab.app

    from isaaclab_rl.entrypoints import run_play_cli

    original_launch = isaaclab.app.launch_simulation

    @contextlib.contextmanager
    def launch_with_adapter(env_cfg, *launch_args, **kwargs):
        with original_launch(env_cfg, *launch_args, **kwargs) as simulation:
            with install_adapter(env_cfg, args):
                yield simulation

    isaaclab.app.launch_simulation = launch_with_adapter
    try:
        return run_play_cli(
            [
                "--rl_library",
                args.rl_library,
                "--task",
                args.task,
                "--checkpoint",
                args.checkpoint,
                "--num_envs",
                "1",
                "--seed",
                str(args.seed),
                *remaining,
            ]
        )
    finally:
        isaaclab.app.launch_simulation = original_launch


if __name__ == "__main__":
    sys.exit(main())
