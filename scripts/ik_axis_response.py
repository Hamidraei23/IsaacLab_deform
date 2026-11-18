"""Unit response of the IK action on one environment: for each of the six pose channels, hold +1 on
that channel alone (gripper open) for a few steps after a reset and report how the tool point moves and
how the hand rotates, in the robot frame. Run on the Franka and the UR10e tasks and compare::

    python scripts/ik_axis_response.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 env.actions.arm_action.scale=0.05
"""

import argparse
import math
import sys

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--amplitude", type=float, default=1.0, help="Action value held on the channel (the policy commands up to ~8 on rotations).")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args

CHANNELS = ("+x", "+y", "+z", "+rot_x", "+rot_y", "+rot_z")


def fmt(values):
    return "[" + ", ".join(f"{v:+.1f}" for v in values) + "]"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.observations import tool_position_b
    from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul, subtract_frame_transforms

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.joint_vel_out_of_limit = None
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        term = env_cfg.observations.action_history.deformable_sampled_points
        hand, offset = term.params["hand_body_name"], tuple(term.params["tool_offset"])
        robot = inner.scene["robot"]
        scale = inner.action_manager.get_term("arm_action")._scale[0].cpu()

        def tool():
            return tool_position_b(inner, hand_body_name=hand, tool_offset=offset)[0].cpu()

        def hand_quat():
            pose = robot.data.body_link_pose_w.torch[:1, robot.body_names.index(hand)]
            _, q = subtract_frame_transforms(robot.data.root_pos_w.torch[:1], robot.data.root_quat_w.torch[:1], pose[:, :3], pose[:, 3:7])
            return q

        print("=" * 100)
        print(f"task {args_cli.task}  scale {scale.tolist()}  {args_cli.steps} steps of {args_cli.amplitude:+g} on one channel")
        print(f"{'channel':8s} | {'commanded per step':28s} | {'tool d per step [mm]':34s} | {'hand rotation per step [deg, robot-frame axis-angle]':50s}")
        with torch.inference_mode():
            for c, name in enumerate(CHANNELS):
                env.reset()
                for _ in range(3):  # settle the reset
                    env.step(torch.zeros(1, inner.action_manager.total_action_dim, device=inner.device) + torch.tensor([0, 0, 0, 0, 0, 0, 1.0], device=inner.device))
                action = torch.zeros(1, inner.action_manager.total_action_dim, device=inner.device)
                action[0, c], action[0, 6] = args_cli.amplitude, 1.0
                p0, q0 = tool(), hand_quat()
                moves, rots = [], []
                for _ in range(args_cli.steps):
                    env.step(action)
                    p1, q1 = tool(), hand_quat()
                    moves.append(((p1 - p0) * 1000.0).tolist())
                    rots.append(torch.rad2deg(axis_angle_from_quat(quat_mul(q1, quat_inv(q0))))[0].cpu().tolist())
                    p0, q0 = p1, q1
                cmd = f"{args_cli.amplitude * scale[c].item() * 1000:.0f} mm" if c < 3 else f"{math.degrees(args_cli.amplitude * scale[c].item()):.1f} deg"
                print(f"{name:8s} | {cmd:28s} | {' '.join(fmt(m) for m in moves):34s} | {' '.join(fmt(r) for r in rots)}")
        print("=" * 100)
        env.close()


if __name__ == "__main__":
    main()
