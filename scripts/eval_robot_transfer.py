"""Bounded, deterministic playback of a twist checkpoint, with transfer diagnostics.

Run once with the UR10e task and once with the original task for a baseline.
Counts the first completed episode per environment and saves a JSON report.

Per environment it also records, over that first episode, the peak arm joint speed as a fraction of the joint's
velocity limit and the tool point's peak and mean speed. With the joint-speed termination off
(``env.terminations.joint_vel_out_of_limit=null``) one rollout then scores a checkpoint against any speed limit:
an episode placed with a peak fraction below ``L`` is one that would also have placed with the termination at ``L``.
Two checkpoints run with the same seed and overrides start from the same layouts.
"""

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import sheet_rl.tasks  # noqa: F401
from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _phase_reached, _is_holding

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="evaluations/ur10e_transfer.json")
parser.add_argument("--tool_speed_threshold", type=float, default=0.4,
                    help="Report the time each first episode spends with the tool point faster than this [m/s].")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args.task, args.agent, play_mode=False)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms
    from sheet_rl.tasks.manager_based.sheet_twist.ur10e_mdp import TOOL_POS, TOOL_ROT

    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
    else:
        from isaaclab_tasks.utils import get_checkpoint_path
        checkpoint = Path(get_checkpoint_path(
            str((Path("logs/rsl_rl") / agent_cfg.experiment_name).resolve()), "^best_deformable_twist$", agent_cfg.load_checkpoint,
        )).resolve()
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
                                 clip_actions=agent_cfg.clip_actions)
        inner = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=inner.device)
        n = inner.num_envs
        done = torch.zeros(n, dtype=torch.bool, device=inner.device)
        extracted, grasped = done.clone(), done.clone()
        returns = torch.zeros(n, device=inner.device)
        steps = torch.zeros(n, dtype=torch.long, device=inner.device)
        endings = [None] * n
        robot = inner.scene["robot"]
        max_joint_speed = torch.zeros(robot.num_joints, device=inner.device)
        max_adapter_pos_error = max_adapter_rot_error = max_fk_error = 0.
        compute_reward = inner.reward_manager.compute
        arm_ids = inner.action_manager.get_term("arm_action")._joint_ids
        limits = torch.full((robot.num_joints,), torch.inf, device=inner.device)
        for actuator in robot.actuators.values():
            ids = actuator.joint_indices
            count = robot.num_joints if isinstance(ids, slice) else len(ids)
            limits[ids] = torch.as_tensor(actuator.velocity_limit_sim, device=inner.device).reshape(-1, count)[0]
        hand_id = robot.body_names.index("wrist_3_link")
        tool_offset = torch.tensor(TOOL_POS, device=inner.device).expand(n, -1)
        peak_speed_fraction = torch.zeros(n, device=inner.device)
        peak_tool_speed, tool_path = torch.zeros(n, device=inner.device), torch.zeros(n, device=inner.device)
        tool_over = torch.zeros(n, device=inner.device)
        tool_prev = []

        def capture(dt):
            reward = compute_reward(dt)
            max_joint_speed.copy_(torch.maximum(max_joint_speed, robot.data.joint_vel.torch.abs().amax(0)))
            extracted.logical_or_(_phase_reached(inner) & ~done)
            grasped.logical_or_(_is_holding(inner) & ~done)
            fraction = (robot.data.joint_vel.torch[:, arm_ids].abs() / limits[arm_ids]).amax(1)
            peak_speed_fraction.copy_(torch.where(done, peak_speed_fraction, torch.maximum(peak_speed_fraction, fraction)))
            pose = robot.data.body_link_pose_w.torch[:, hand_id]
            tool = pose[:, :3] + quat_apply(pose[:, 3:7], tool_offset)
            if tool_prev:
                step = (tool - tool_prev[0]).norm(dim=-1)
                peak_tool_speed.copy_(torch.where(done, peak_tool_speed, torch.maximum(peak_tool_speed, step / inner.step_dt)))
                tool_path.add_(step * ~done)
                tool_over.add_(((step / inner.step_dt > args.tool_speed_threshold) & ~done).float())
            tool_prev[:] = [tool]
            return reward

        inner.reward_manager.compute = capture
        with torch.inference_mode():
            obs, _ = env.reset()
            for _ in range(int(inner.max_episode_length) + 2):
                action = policy(obs)
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Non-finite policy action")
                obs, reward, terminated, _ = env.step(action)
                if not bool(torch.isfinite(obs["policy"]).all()):
                    raise RuntimeError("Non-finite policy observation")
                returns += reward * ~done
                steps += ~done
                state = getattr(inner, "_franka_observation_adapter", None)
                if state is not None:
                    max_adapter_pos_error = max(max_adapter_pos_error, state.position_error.max().item())
                    max_adapter_rot_error = max(max_adapter_rot_error, state.rotation_error.max().item())
                    robot = inner.scene["robot"]
                    pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index("wrist_3_link")]
                    off = torch.tensor(TOOL_POS, device=inner.device).expand(n, -1)
                    p, r = subtract_frame_transforms(
                        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                        pose[:, :3] + quat_apply(pose[:, 3:7], off),
                        quat_mul(pose[:, 3:7], torch.tensor(TOOL_ROT, device=inner.device).expand(n, -1)),
                    )
                    fk, _, _ = state.ur.forward(robot.data.joint_pos.torch[:, state.arm_ids])
                    max_fk_error = max(max_fk_error, (fk - p).norm(dim=-1).max().item())
                new = terminated.bool() & ~done
                for i in new.nonzero().flatten().tolist():
                    endings[i] = [name for name in inner.termination_manager.active_terms
                                  if bool(inner.termination_manager.get_term(name)[i])]
                done.logical_or_(terminated.bool())
                if bool(done.all()):
                    break
        report = {
            "task": args.task, "checkpoint": str(checkpoint), "seed": args.seed, "num_envs": n,
            "observation_shape": list(obs["policy"].shape), "action_shape": list(action.shape),
            "grasped": grasped.cpu().tolist(), "extracted": extracted.cpu().tolist(),
            "endings": endings, "returns": returns.cpu().tolist(), "steps": steps.cpu().tolist(),
            "successes": sum("placed" in (terms or []) for terms in endings),
            "max_adapter_position_error_m": max_adapter_pos_error,
            "max_adapter_orientation_error_rad": max_adapter_rot_error,
            "max_ur_fk_vs_sim_position_error_m": max_fk_error,
            "max_joint_speed": dict(zip(robot.joint_names, max_joint_speed.cpu().tolist())),
            "step_dt": inner.step_dt,
            "peak_joint_speed_fraction": peak_speed_fraction.cpu().tolist(),
            "peak_tool_speed": peak_tool_speed.cpu().tolist(),
            "mean_tool_speed": (tool_path / (steps.clamp(min=1) * inner.step_dt)).cpu().tolist(),
            "tool_speed_threshold": args.tool_speed_threshold,
            "time_over_tool_speed_threshold": (tool_over * inner.step_dt).cpu().tolist(),
        }
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("TRANSFER_REPORT " + json.dumps(report), flush=True)
        env.close()


if __name__ == "__main__":
    main()
