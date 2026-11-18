"""The first control steps of a FastSAC actor on one environment: observation at reset, the action
at each step and how the tool point moves in response. A cross-robot diagnostic.

Run on the Franka and the UR10e tasks with the same seed and fixed layout, then compare; with
``--replay`` the actions come from another run's JSON instead of the actor, which isolates the
plant's response from the policy's decisions::

    python scripts/compare_first_steps.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 \\
        --checkpoint model.pt --out franka.json env.events.reset_layout.params.yaw_range=[0,0] ...
    python scripts/compare_first_steps.py --task ...-UR10e-v0 --checkpoint model.pt --replay franka.json
"""

import argparse
import json
import sys

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    add_frontend_args, apply_video_recording, create_isaaclab_env, pre_launch_video_config,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.algorithms.fast_sac import FastSAC, FastSACCfg, IsaacLabSacEnv  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--actor_obs_groups", type=str, nargs="+", default=None,
                    help="Override the checkpoint's actor groups (actors from before 2026-10-26: action_history_legacy).")
parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of the mean.")
parser.add_argument("--replay", type=str, default=None, help="JSON from another run: apply its actions open-loop.")
parser.add_argument("--out", type=str, default=None)
# the recorder flags of the training entry points (add_common_train_args would also add --task/--num_envs)
parser.add_argument("--video", action="store_true", default=False, help="Record the run with Isaac Lab's recorder (--viz newton_gl).")
parser.add_argument("--video_length", type=int, default=None, help="Clip length in env steps.")
parser.add_argument("--video_interval", type=int, default=None, help="Steps between clips.")
parser.add_argument("--clip_dir", type=str, default="videos", help="With --video: where the mp4 goes (<clip_dir>/videos/replay/clip_*.mp4).")
parser.add_argument("--camera", type=float, nargs=6, default=None, metavar=("EX", "EY", "EZ", "LX", "LY", "LZ"),
                    help="With --video: record from a headless Newton GL view with this eye and look-at point (robot frame) instead of --viz.")
parser.add_argument("--show_collision", action="store_true", default=False,
                    help="With --camera: also draw the collision shapes (the table is an invisible collider).")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args


def fmt(values):
    return "[" + ", ".join(f"{v:+.4f}" for v in values) + "]"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.gripper_geometry import finger_half_width, pad_positions
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.observations import tool_position_b
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import subtract_frame_transforms

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.compute_final_obs = True
    torch.manual_seed(args_cli.seed)
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    replay = json.load(open(args_cli.replay))["steps"] if args_cli.replay else None
    if args_cli.video and args_cli.camera is not None:
        # a concrete, headless GL visualizer with a chosen camera; apply_video_recording records from it
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=tuple(args_cli.camera[:3]), lookat=tuple(args_cli.camera[3:]), headless=True,
            window_width=1280, window_height=720, show_collision=args_cli.show_collision,
        )]
    if args_cli.video:  # --video --viz newton_gl: the env's own recorder writes <clip_dir>/videos/replay/clip_*.mp4
        pre_launch_video_config(env_cfg, None, args_cli)
    with launch_simulation(env_cfg, args_cli):
        if args_cli.video:
            apply_video_recording(env_cfg, args_cli.clip_dir, args_cli, subdir="replay")
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        # the pre-repair group too, where the task has it, so slot 49 can be logged both ways
        legacy = [g for g in ("action_history_legacy",) if g in inner.observation_manager.group_obs_dim]
        sac_env = IsaacLabSacEnv(env, args_cli.actor_obs_groups or ckpt["actor_obs_groups"], ckpt["critic_obs_groups"],
                                 extra_obs_groups=legacy)
        device = str(sac_env.device)
        cfg = FastSACCfg(experiment_name="compare")
        for key, value in ckpt["cfg"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.device, cfg.compile = device, False
        alg = FastSAC(cfg, sac_env.n_obs, sac_env.n_critic_obs, sac_env.n_act, sac_env.num_envs, device)
        alg.load(args_cli.checkpoint, load_optimizers=False)
        alg.actor.eval()

        term = env_cfg.observations.action_history.deformable_sampled_points
        hand, offset = term.params["hand_body_name"], tuple(term.params["tool_offset"])
        robot = inner.scene["robot"]
        finger_cfg = env_cfg.observations.action_history.finger_width.params.get("asset_cfg")
        if finger_cfg is None:
            finger_cfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*")
        finger_cfg.resolve(inner.scene)

        def tool():
            return tool_position_b(inner, hand_body_name=hand, tool_offset=offset)[0].cpu()

        def hand_quat_b():
            pose = robot.data.body_link_pose_w.torch[:1, robot.body_names.index(hand)]
            _, q = subtract_frame_transforms(robot.data.root_pos_w.torch[:1], robot.data.root_quat_w.torch[:1], pose[:, :3], pose[:, 3:7])
            return q[0].cpu()

        # arm links whose height matters (the elbow and the wrist): the lowest origin each step
        arm_bodies = [n for n in robot.body_names if n in ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
                                                             "panda_link4", "panda_link5", "panda_link6", "panda_link7")]
        arm_ids = [robot.body_names.index(n) for n in arm_bodies]

        # the fingertip distance from the simulator's body poses, independent of every observation
        # term: the pad contact faces on a gripper with finger_pad_geometry, the two fingertip bodies
        # on the Franka (their origins sit on the jaw axis and coincide when the jaws are shut)
        tip_ids = [robot.body_names.index(n) for n in ("panda_leftfinger", "panda_rightfinger") if n in robot.body_names]

        def fingertip_distance():
            pads = pad_positions(inner, robot)
            if pads is not None:
                return (pads[0, 0] - pads[0, 1]).norm().item()
            if len(tip_ids) == 2:
                tips = robot.data.body_link_pose_w.torch[0, tip_ids, :3]
                return (tips[0] - tips[1]).norm().item()
            return float("nan")

        def legacy_slot49():
            return sac_env.extra_obs[0, 48].item() if legacy else float("nan")

        def lowest_arm_link():
            z = robot.data.body_link_pose_w.torch[0, arm_ids, 2] - robot.data.root_pos_w.torch[0, 2]
            i = int(z.argmin())
            return arm_bodies[i], z[i].item()

        obs, _ = sac_env.reset()
        obs0 = obs[0].cpu()
        print(f"slot 49 at reset: actor input {obs0[48].item():.4f} | fingertip distance from body poses "
              f"{fingertip_distance():.4f} | pre-repair group {legacy_slot49():.4f}")
        print(f"arm joints at reset: {fmt(robot.data.joint_pos.torch[0, :6].cpu())}  lowest arm link {lowest_arm_link()}")
        min_link = ("", 9.9)
        # peak |joint velocity| over the episode against the actuators' sim limits (what the task's
        # joint_vel_out_of_limit termination compares; the Franka arm's limit is effectively infinite)
        vel_limits = torch.full((robot.num_joints,), float("inf"), device=inner.device)
        for actuator in robot.actuators.values():
            vel_limits[actuator.joint_indices] = torch.as_tensor(actuator.velocity_limit_sim, device=inner.device)[0]
        peak_vel = torch.zeros(robot.num_joints, device=inner.device)
        first_over = None  # (step, joint): where joint_vel_out_of_limit would have ended the episode
        p0, q0 = tool(), hand_quat_b()
        print("=" * 100)
        print(f"task {args_cli.task}  seed {args_cli.seed}  {'replay ' + args_cli.replay if replay else ('stochastic' if args_cli.stochastic else 'mean action')}")
        print(f"obs at reset (49): {fmt(obs0)}")
        print(f"tool at reset (robot frame) {fmt(p0)}  hand quat {fmt(q0)}  pad gap {2 * finger_half_width(inner, finger_cfg)[0].item():.4f}")
        print("-" * 100)
        print(f"{'step':>4} | {'action (6 IK + gripper)':60s} | {'tool pos after step':30s} | {'d tool [mm]':24s} | |d| mm | gap")
        record = {"task": args_cli.task, "seed": args_cli.seed, "replay": args_cli.replay, "obs0": obs0.tolist(),
                  "tool0": p0.tolist(), "hand_quat0": q0.tolist(), "steps": []}
        prev = p0
        with torch.inference_mode():
            for k in range(args_cli.steps):
                if replay is not None:
                    action = torch.tensor(replay[k]["action"], device=device).view(1, -1)
                else:
                    action = alg.act(obs, deterministic=not args_cli.stochastic)
                step = sac_env.step(action)
                obs = step["obs"]
                if not bool(step["dones"][0]):  # a terminal step's state is already the reset one
                    speed = robot.data.joint_vel.torch[0].abs()
                    peak_vel = torch.maximum(peak_vel, speed)
                    if first_over is None and bool((speed > vel_limits).any()):
                        first_over = (k + 1, robot.joint_names[int((speed / vel_limits).argmax())])
                p = tool()
                d = (p - prev) * 1000.0
                gap = 2 * finger_half_width(inner, finger_cfg)[0].item()
                holding, extracted = bool(_is_holding(inner)[0]), bool(_phase_reached(inner)[0])
                low = lowest_arm_link()
                if low[1] < min_link[1]:
                    min_link = low
                a = action[0].cpu()
                print(f"{k + 1:4d} | {fmt(a):60s} | {fmt(p):30s} | {fmt(d):24s} | {d.norm():6.1f} | {gap:.4f}"
                      f"{' HOLD' if holding else ''}{' EXTRACTED' if extracted else ''}")
                record["steps"].append({"action": a.tolist(), "tool": p.tolist(), "d_tool_mm": d.tolist(),
                                        "hand_quat": hand_quat_b().tolist(), "pad_gap": gap,
                                        "holding": holding, "extracted": extracted,
                                        "fingertip_distance": fingertip_distance(), "legacy_slot49": legacy_slot49(),
                                        "reward": step["rewards"][0].item(), "done": bool(step["dones"][0]),
                                        "obs_after": obs[0].cpu().tolist()})
                prev = p
                if bool(step["dones"][0]):
                    flags = sac_env.termination_flags(tuple(inner.termination_manager.active_terms))
                    outcome = next((n for n, f in flags.items() if bool(f[0])), "other")
                    record["outcome"] = outcome
                    record["return"] = sum(r["reward"] for r in record["steps"])
                    log = step["extras"].get("log", {})
                    record["episode_log"] = {k2: float(v) for k2, v in log.items()
                                             if k2.startswith(("Episode_Reward/", "Coverage/", "Placement/"))}
                    record["lowest_arm_link"] = {"body": min_link[0], "z": min_link[1]}
                    record["peak_joint_vel"] = {n: [round(v, 3), round(l, 3)] for n, v, l in
                                                zip(robot.joint_names, peak_vel.tolist(), vel_limits.tolist())}
                    over = [f"{n} {v:.2f}>{l:.2f}" for n, (v, l) in record["peak_joint_vel"].items() if v > l]
                    print(f"peak joint speed: max {peak_vel.max().item():.2f} rad/s; over the sim limit: {over or 'none'}")
                    record["first_over_speed_limit"] = first_over
                    print(f"first step over a joint speed limit: {first_over or 'never'}")
                    print(f"episode ended at step {k + 1}: {outcome}, return {record['return']:.1f}; "
                          f"lowest arm link over the episode: {min_link[0]} at z={min_link[1]:+.3f} m (table top at 0)")
                    for prefix in ("Episode_Reward/", "Coverage/", "Placement/"):
                        items = [(k2, v) for k2, v in record["episode_log"].items() if k2.startswith(prefix)]
                        if items:
                            print("   " + "  ".join(f"{k2.split('/', 1)[1]}={v:.4g}" for k2, v in items))
                    break
        total = (torch.tensor(record["steps"][-1]["tool"]) - p0) * 1000.0
        print(f"net tool displacement over {len(record['steps'])} steps [mm]: {fmt(total)}  |{total.norm():.1f}|")
        print("=" * 100)
        if args_cli.out:
            with open(args_cli.out, "w") as f:
                json.dump(record, f, indent=1)
        env.close()


if __name__ == "__main__":
    main()
