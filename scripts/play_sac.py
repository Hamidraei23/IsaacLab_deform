"""Watch a FastSAC actor (``train_sac.py`` / ``train_dagger_sac_student.py`` checkpoint) in the viewer.

The checkpoint records its observation groups, so the task only has to provide them. The actor
acts stochastically by default (the policy as trained and evaluated); ``--mean_action`` plays
its mean action. ``--lite`` is one environment at wall-clock speed, as in ``play.py``.

Usage (inside ``active_isaac``)::

    python scripts/play_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 \\
        --checkpoint logs/fast_sac/sheet_platform_twist_sac/<run>/model_66500.pt --viz kit --lite --seed 42

Each finished episode prints its outcome, return and length.
"""

import argparse
import sys
import time

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.algorithms.fast_sac import FastSAC, FastSACCfg, IsaacLabSacEnv  # noqa: E402

OUTCOMES = ("placed", "dropped", "held_too_low", "time_out")

parser = argparse.ArgumentParser(description="Play a FastSAC actor.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True, help="FastSAC checkpoint (model_*.pt).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--lite", action="store_true", default=False, help="One environment at wall-clock speed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Sleep out the rest of each control period.")
parser.add_argument("--mean_action", action="store_true", default=False, help="Play the mean action instead of a sample.")
parser.add_argument("--steps", type=int, default=10**9, help="Control steps to run before exiting.")
parser.add_argument("--episodes", type=int, default=None, help="Stop after this many episodes have ended.")
parser.add_argument("--first_episode_only", action="store_true", default=False,
                    help="Count only each environment's first episode and stop when every environment has finished one "
                         "(an unbiased success rate; stopping at N ended episodes over-counts short ones).")
parser.add_argument("--true_finger_width", action="store_true", default=False,
                    help="Diagnostic: feed the actor the measured gripper opening (gripper_opening) in the finger_width slot.")
parser.add_argument("--obs_override", type=str, default=None,
                    help="Diagnostic: 'IDX=VALUE[,IDX=VALUE]' constants written into the actor observation every step.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--actor_obs_groups", type=str, nargs="+", default=None,
                    help="Override the checkpoint's actor groups (actors from before 2026-10-26: action_history_legacy).")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args
if args_cli.lite:
    args_cli.num_envs, args_cli.real_time = 1, True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.compute_final_obs = True
    if "gamma" in getattr(env_cfg.rewards.grasp_progress, "params", {}):
        env_cfg.rewards.grasp_progress.params["gamma"] = agent_cfg.gamma
    torch.manual_seed(args_cli.seed)
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    if args_cli.true_finger_width:
        from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.observations import gripper_opening
        term = env_cfg.observations.action_history.finger_width
        asset_cfg = term.params.get("asset_cfg", SceneEntityCfg("robot", joint_names="panda_finger_joint.*"))
        env_cfg.observations.action_history.finger_width = ObservationTermCfg(func=gripper_opening, params={"asset_cfg": asset_cfg})
    overrides = {}
    if args_cli.obs_override:
        for item in args_cli.obs_override.split(","):
            idx, value = item.split("=")
            overrides[int(idx)] = float(value)
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        sac_env = IsaacLabSacEnv(env, args_cli.actor_obs_groups or ckpt["actor_obs_groups"], ckpt["critic_obs_groups"])

        def apply_overrides(obs):
            for idx, value in overrides.items():
                obs[:, idx] = value
            return obs
        device = str(sac_env.device)
        cfg = FastSACCfg(experiment_name="play")
        for key, value in ckpt["cfg"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.device, cfg.compile = device, False
        alg = FastSAC(cfg, sac_env.n_obs, sac_env.n_critic_obs, sac_env.n_act, sac_env.num_envs, device)
        alg.load(args_cli.checkpoint, load_optimizers=False)
        alg.actor.eval()
        print(f"[play_sac] {args_cli.checkpoint} | actor {ckpt['actor_obs_groups']} = {sac_env.n_obs} | "
              f"{'deterministic' if args_cli.mean_action else 'stochastic'} | {sac_env.num_envs} env(s)"
              f"{' | real time' if args_cli.real_time else ''}")
        step_dt = float(env.unwrapped.step_dt)
        obs, _ = sac_env.reset()
        obs = apply_overrides(obs)
        ep_return = torch.zeros(sac_env.num_envs, device=sac_env.device)
        ep_length = torch.zeros(sac_env.num_envs, device=sac_env.device)
        counts = {name: 0 for name in (*OUTCOMES, "other")}
        counted = torch.zeros(sac_env.num_envs, dtype=torch.bool, device=sac_env.device)
        returns, outcomes = [], []
        # each episode's drawn conditions, read at its start: the IK scale from the action term and the
        # sheet yaw from the bottom edge (corner0 -> corner2, actor slots 15-16 and 21-22), as an offset
        # from the default slot yaw (-30.94 deg); only when the actor reads action_history
        arm_term = env.unwrapped.action_manager.get_term("arm_action")
        reads_sheet = ckpt["actor_obs_groups"] == ["action_history"] or (args_cli.actor_obs_groups or [None])[0] in ("action_history", "action_history_legacy")

        def conditions(o):
            scale = arm_term._scale[:, 0].clone()
            yaw = torch.rad2deg(torch.atan2(o[:, 21] - o[:, 15], o[:, 20] - o[:, 14])) + 30.94 if reads_sheet else torch.zeros_like(scale)
            return scale, yaw

        ep_scale, ep_yaw = conditions(obs)
        # every termination the task has, so an ending outside OUTCOMES is named rather than "other"
        all_terms = tuple(dict.fromkeys((*OUTCOMES, *env.unwrapped.termination_manager.active_terms)))
        # peak |joint velocity| per joint over the episode (env 0), against the actuators' sim limits:
        # what the task's joint_vel_out_of_limit termination compares
        robot = env.unwrapped.scene["robot"]
        vel_limits = torch.full((robot.num_joints,), float("inf"), device=sac_env.device)
        for actuator in robot.actuators.values():
            vel_limits[actuator.joint_indices] = torch.as_tensor(actuator.velocity_limit_sim, device=sac_env.device)[0]
        peak_vel = torch.zeros(robot.num_joints, device=sac_env.device)
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                t0 = time.time()
                step = sac_env.step(alg.act(obs, deterministic=args_cli.mean_action))
                ep_return += step["rewards"]
                ep_length += 1
                peak_vel = torch.maximum(peak_vel, robot.data.joint_vel.torch[0].abs())
                if step["dones"].any():
                    flags = sac_env.termination_flags(all_terms)
                    for i in step["dones"].nonzero(as_tuple=False).flatten().tolist():
                        if args_cli.first_episode_only and bool(counted[i]):
                            continue
                        counted[i] = True
                        outcome = next((n for n in all_terms if n in flags and bool(flags[n][i])), "other")
                        counts[outcome] = counts.get(outcome, 0) + 1
                        returns.append(ep_return[i].item()); outcomes.append(outcome)
                        total = sum(counts.values())
                        print(f"[play_sac] env {i}: {outcome:12s} return {ep_return[i].item():8.1f} length {int(ep_length[i])}"
                              f" | scale {ep_scale[i].item():.3f} yaw {ep_yaw[i].item():+5.1f} deg"
                              f" | success so far {counts['placed']}/{total}", flush=True)
                    # the env's episode log (reward terms, coverage, placement) is rewritten on reset steps;
                    # with one environment it describes exactly the episode that just ended
                    log = step["extras"].get("log", {})
                    for prefix in ("Episode_Reward/", "Coverage/", "Placement/", "Episode_Termination/"):
                        items = [(k, float(v)) for k, v in log.items() if k.startswith(prefix)]
                        if items:
                            print("[play_sac]   " + "  ".join(f"{k.split('/', 1)[1]}={v:.4g}" for k, v in items), flush=True)
                    if bool(step["dones"][0]):
                        print("[play_sac]   peak |joint vel| / sim limit (env 0): " + "  ".join(
                            f"{n}={v:.2f}/{l:.2f}" for n, v, l in zip(robot.joint_names, peak_vel.tolist(), vel_limits.tolist())),
                            flush=True)
                        peak_vel.zero_()
                    ep_return[step["dones"]] = 0.0
                    ep_length[step["dones"]] = 0.0
                    new_scale, new_yaw = conditions(step["obs"])  # the reset environments' new draws
                    ep_scale = torch.where(step["dones"], new_scale, ep_scale)
                    ep_yaw = torch.where(step["dones"], new_yaw, ep_yaw)
                    if args_cli.episodes is not None and sum(counts.values()) >= args_cli.episodes:
                        break
                    if args_cli.first_episode_only and bool(counted.all()):
                        break
                obs = apply_overrides(step["obs"])
                if args_cli.real_time:
                    time.sleep(max(0.0, step_dt - (time.time() - t0)))
        if returns:
            n = len(returns)
            placed = [r for r, o in zip(returns, outcomes) if o == "placed"]
            mix = "  ".join(f"{k} {v / n:.3f}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v)
            print(f"[play_sac] SUMMARY {n} episodes{' (first per env)' if args_cli.first_episode_only else ''}: "
                  f"success {len(placed) / n:.3f} | {mix} | mean return {sum(returns) / n:.1f}"
                  f"{f' | mean return when placed {sum(placed) / len(placed):.1f}' if placed else ''}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
