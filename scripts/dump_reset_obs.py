"""Print the ``action_history`` observation of one environment at reset, labelled, plus the
geometry behind it (tool point, pad faces, sheet points in the robot frame), as a cross-robot check.

Run once per task and compare::

    python scripts/dump_reset_obs.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --headless
    python scripts/dump_reset_obs.py --task Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0 --headless

``--out`` writes the same numbers as JSON.
"""

import argparse
import json
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
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default=None, help="Write the labelled numbers to this JSON file.")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args

POINT_NAMES = ["corner0", "edge1", "corner2", "edge3", "centre4", "edge5", "corner6", "edge7", "corner8"]


def fmt(values):
    return "[" + ", ".join(f"{v:+.4f}" for v in values) + "]"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.gripper_geometry import pad_positions
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.observations import sheet_key_points, tool_position_b
    from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env_cfg import SHEET_RESOLUTION
    from isaaclab.utils.math import subtract_frame_transforms

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    torch.manual_seed(args_cli.seed)
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        obs, _ = env.reset()
        ah = obs["action_history"][0].cpu()
        pol = obs["policy"][0].cpu()
        term = env_cfg.observations.action_history.deformable_sampled_points
        hand, offset = term.params["hand_body_name"], tuple(term.params["tool_offset"])
        robot = inner.scene["robot"]
        tool_b = tool_position_b(inner, hand_body_name=hand, tool_offset=offset)[0].cpu()
        hand_pose = robot.data.body_link_pose_w.torch[0, robot.body_names.index(hand)]
        hand_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch[:1], robot.data.root_quat_w.torch[:1], hand_pose[None, :3])
        points_b = sheet_key_points(inner, SHEET_RESOLUTION)[0].cpu().view(9, 3)
        command = inner.command_manager.get_command("deformable_pose")[0].cpu()
        pads = pad_positions(inner, robot)
        if pads is None:  # Franka: fingertip bodies stand in for the pad faces
            ids = [robot.body_names.index(n) for n in ("panda_leftfinger", "panda_rightfinger")]
            pads = robot.data.body_link_pose_w.torch[:, ids, :3]
        pads_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w.torch[:1].expand(2, -1), robot.data.root_quat_w.torch[:1].expand(2, -1), pads[0]
        )
        pads_b = pads_b.cpu()
        root_quat = robot.data.root_quat_w.torch[0].cpu()

        report = {
            "task": args_cli.task, "seed": args_cli.seed,
            "root_quat_w (w,x,y,z)": root_quat.tolist(),
            "hand_body": hand, "tool_offset": list(offset),
            "hand_body_pos_b": hand_b[0].cpu().tolist(),
            "tool_point_b": tool_b.tolist(),
            "pad_faces_b": pads_b.tolist(),
            "pad_midpoint_b": pads_b.mean(0).tolist(),
            "pad_midpoint_minus_tool": (pads_b.mean(0) - tool_b).tolist(),
            "pad_gap": (pads_b[0] - pads_b[1]).norm().item(),
            "sheet_points_b": {n: p.tolist() for n, p in zip(POINT_NAMES, points_b)},
            "command (pos_b, quat)": command.tolist(),
            "action_history_raw": ah.tolist(),
            "action_history": {
                "actions_t-2": ah[0:7].tolist(),
                "actions_t-1": ah[7:14].tolist(),
                "sheet_from_tool": {n: ah[14 + 3 * i: 17 + 3 * i].tolist() for i, n in enumerate(POINT_NAMES)},
                "target_from_tool_pos": ah[41:44].tolist(),
                "target_quat": ah[44:48].tolist(),
                "finger_width": ah[48].item(),
            },
            "policy_group": pol.tolist(),
        }
        print("=" * 100)
        print(f"task {args_cli.task}  seed {args_cli.seed}")
        print(f"root_quat_w (w,x,y,z) {fmt(root_quat)}")
        print(f"hand body {hand} at {fmt(hand_b[0])} (robot frame); tool offset {offset}")
        print(f"tool point (robot frame)      {fmt(tool_b)}")
        print(f"pad faces (robot frame)       {fmt(pads_b[0])}  {fmt(pads_b[1])}   gap {report['pad_gap']:.4f} m")
        print(f"pad midpoint - tool point     {fmt(report['pad_midpoint_minus_tool'])}")
        print("sheet key points (robot frame):")
        for n, p in zip(POINT_NAMES, points_b):
            print(f"   {n:8s} {fmt(p)}")
        print(f"command (robot frame)         {fmt(command)}")
        print("-" * 100)
        print("action_history (49):")
        print(f"   actions t-2      {fmt(ah[0:7])}")
        print(f"   actions t-1      {fmt(ah[7:14])}")
        for i, n in enumerate(POINT_NAMES):
            print(f"   {n:8s} from tool {fmt(ah[14 + 3 * i: 17 + 3 * i])}")
        print(f"   target from tool {fmt(ah[41:44])}  quat {fmt(ah[44:48])}")
        print(f"   finger width     {ah[48].item():+.4f}")
        print(f"policy (59): {fmt(pol)}")
        print("=" * 100)
        if args_cli.out:
            with open(args_cli.out, "w") as f:
                json.dump(report, f, indent=1)
        env.close()


if __name__ == "__main__":
    main()
