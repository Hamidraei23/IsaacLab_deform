# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a checkpoint and measure how fast the arm actually moves.

Capping ``arm_action.scale`` sets how fast the arm *may* move; it says nothing about how fast it
chooses to. This runs the deterministic policy and records what the joints actually did, so the two
can be told apart -- an arm sitting at 40% of its cap needs no further capping, and one pinned at
100% of it is being held back by the cap rather than by the policy.

Written as its own script because ``scripts/play.py`` hands straight off to Isaac Lab's playback
backend, whose rollout loop has no per-step hook to record from.

Reports, per arm joint, the median, 95th percentile and maximum angular speed, alongside the
end-effector's linear speed, and writes a three-panel plot next to the checkpoint.

The same rollout also records what every reward term paid on every step, and writes a second figure
beside the first. The two answer different questions about one run -- how the arm moved, and what it
was paid for -- and recording both from a single rollout means they describe the *same* episodes
rather than two runs that happened to be configured alike.

Usage::

    python3 scripts/eval_joint_speed.py --task Template-Sheet-Rl-v0 --num_envs 16 --headless \
        --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_2380.pt \
        --steps 600 agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
        agent.actor.obs_normalization=true agent.critic.obs_normalization=true agent.clip_actions=2.4
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.string import list_intersection

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import (
    CHECKPOINT_SELECTORS,
    add_frontend_args,
    create_isaaclab_env,
    resolve_checkpoint_selector,
    resolve_play_task_name,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import sheet_rl.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# Real Franka Emika Panda joint velocity limits [rad/s], drawn on the plot for reference. The
# simulated actuators are configured an order of magnitude above these, so "within limits" in
# simulation says nothing about whether the motion is achievable on hardware.
FRANKA_JOINT_VEL_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610)

parser = argparse.ArgumentParser(description="Measure arm joint speeds for a trained checkpoint.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=600, help="Environment steps to record.")
parser.add_argument("--warmup", type=int, default=30, help="Steps to discard before recording.")
parser.add_argument("--out", type=str, default=None, help="Plot path. Defaults to beside the checkpoint.")
parser.add_argument(
    "--reward_out", type=str, default=None, help="Reward-term plot path. Defaults to beside the checkpoint."
)
parser.add_argument(
    "--no_rewards", action="store_true", default=False, help="Skip recording and plotting the reward terms."
)
parser.add_argument(
    "--train_env_cfg", action="store_true", default=False, help="Use the training env config rather than play mode."
)
# brings --checkpoint along with the rest of the group; ``update_rsl_rl_cfg`` reads every one of
# them off the namespace unguarded, so the whole group has to be present even though playback only
# uses --checkpoint
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="rsl_rl")
args_cli.task = resolve_play_task_name(args_cli.task)
sys.argv = [sys.argv[0]] + list_intersection(remaining_args, None)


def _summarize(speeds: torch.Tensor, names: list[str], limits) -> list[tuple[str, float, float, float, float]]:
    """Median, 95th percentile, max and limit fraction per column of ``speeds``."""
    rows = []
    for index, name in enumerate(names):
        column = speeds[..., index].flatten()
        peak = column.max().item()
        limit = limits[index] if index < len(limits) else float("nan")
        rows.append((name, column.median().item(), column.quantile(0.95).item(), peak, peak / limit))
    return rows


def _plot(path, joint_speeds, ee_speed, names, dt, commanded_cap):
    """Write the three-panel figure. Imported lazily so the sim can run without a display stack."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = torch.arange(joint_speeds.shape[0]) * dt
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    axes[0].set_title("Per-joint angular speed (mean over environments)")
    for index, name in enumerate(names):
        axes[0].plot(time_axis, joint_speeds[:, :, index].mean(dim=1), linewidth=1.0, label=name)
    if commanded_cap is not None:
        axes[0].axhline(commanded_cap, color="k", linestyle="--", linewidth=1.0, label=f"cap {commanded_cap:.2f}")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("|omega| [rad/s]")
    axes[0].legend(fontsize=7, ncol=4)

    axes[1].set_title("End-effector linear speed")
    # one sample shorter than the joint trace: the first step has no predecessor to difference
    ee_axis = torch.arange(ee_speed.shape[0]) * dt
    axes[1].plot(ee_axis, ee_speed.mean(dim=1), linewidth=1.0, label="mean")
    axes[1].fill_between(
        ee_axis, ee_speed.quantile(0.05, dim=1), ee_speed.quantile(0.95, dim=1), alpha=0.25, label="5-95%"
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].legend(fontsize=8)

    axes[2].set_title("Angular speed distribution vs. real Franka limits")
    positions = range(len(names))
    axes[2].bar(
        positions,
        [joint_speeds[..., i].flatten().quantile(0.95).item() for i in positions],
        label="95th percentile",
    )
    axes[2].plot(
        positions, [FRANKA_JOINT_VEL_LIMITS[i] for i in positions], "kv", markersize=9, label="hardware limit"
    )
    axes[2].set_xticks(list(positions), names, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("|omega| [rad/s]")
    axes[2].legend(fontsize=8)

    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_rewards(path, step_rewards, term_names, episode_ends, end_labels, dt):
    """Write the reward-term figure.

    Args:
        path: Where to write the PNG.
        step_rewards: ``(steps, num_envs, num_terms)`` of what each term actually added to the
            return on each step -- already weighted and dt-scaled, so the numbers sum to the return.
        term_names: One name per column of ``step_rewards``.
        episode_ends: Step indices at which an episode ended.
        end_labels: The termination term that fired at each of those indices.
        dt: Control step [s].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # averaged over environments: with one environment this is the episode itself, and with many it
    # is the mean episode, which is the quantity the training curves are also drawn from
    per_step = step_rewards.mean(dim=1)
    cumulative = per_step.cumsum(dim=0)
    totals = cumulative[-1]
    time_axis = torch.arange(per_step.shape[0]) * dt

    # A term that never paid anything is noise in a legend of sixteen. Ordered by how much they
    # moved the return, so the terms that decide the run are the ones named first.
    order = sorted(range(len(term_names)), key=lambda i: -abs(totals[i].item()))
    shown = [i for i in order if abs(totals[i].item()) > 1e-6]
    colors = plt.get_cmap("tab20")(torch.linspace(0, 1, max(len(shown), 1)).numpy())

    figure, axes = plt.subplots(3, 1, figsize=(13, 15), constrained_layout=True)

    axes[0].set_title("Cumulative reward by term (mean over environments)")
    for color, index in zip(colors, shown):
        axes[0].plot(time_axis, cumulative[:, index], linewidth=1.3, color=color, label=term_names[index])
    axes[0].plot(time_axis, cumulative.sum(dim=1), linewidth=2.2, color="k", label="total")
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("cumulative reward")
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")

    # Symmetric log: the one-shots are thousands and the dense terms are single digits, so a linear
    # axis shows the spikes and a flat line along zero. Symlog keeps the sign and stays linear
    # within +/-1, which is where the per-step terms live.
    axes[1].set_title("Per-step reward by term (symlog)")
    for color, index in zip(colors, shown):
        axes[1].plot(time_axis, per_step[:, index], linewidth=1.0, color=color, label=term_names[index])
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("reward this step")
    axes[1].legend(fontsize=7, ncol=3)

    # episode boundaries on both time-series panels, labelled with what ended them: a reward trace
    # is unreadable without knowing where one attempt stopped and the next began
    for axis in axes[:2]:
        for step, label in zip(episode_ends, end_labels):
            axis.axvline(step * dt, color="0.4", linestyle=":", linewidth=1.0)
            axis.annotate(
                label,
                xy=(step * dt, axis.get_ylim()[1]),
                fontsize=6,
                rotation=90,
                va="top",
                ha="right",
                color="0.3",
            )

    axes[2].set_title("Total contribution over the rollout")
    # same filter the traces use: a row of zeros for a term that never fired reads as data
    bar_order = sorted(shown, key=lambda i: totals[i].item())
    values = [totals[i].item() for i in bar_order]
    axes[2].barh(
        range(len(bar_order)),
        values,
        color=["#c0504d" if value < 0 else "#4f81bd" for value in values],
    )
    axes[2].set_yticks(range(len(bar_order)), [term_names[i] for i in bar_order], fontsize=8)
    axes[2].axvline(0.0, color="k", linewidth=0.8)
    axes[2].set_xlabel("total reward over the rollout")
    for position, value in enumerate(values):
        axes[2].annotate(
            f"{value:,.0f}",
            xy=(value, position),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
        )
    axes[2].margins(x=0.15)

    figure.savefig(path, dpi=130)
    plt.close(figure)


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Record joint speeds while the deterministic policy runs."""
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed

    with launch_simulation(env_cfg, args_cli):
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint in CHECKPOINT_SELECTORS:
            resume_path = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="rsl_rl",
                task=args_cli.task,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": args_cli.agent},
            )
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot = env.unwrapped.scene["robot"]
        arm_names = [name for name in robot.joint_names if name.startswith("panda_joint")]
        arm_ids = [robot.joint_names.index(name) for name in arm_names]
        hand_id = robot.body_names.index("panda_hand")
        dt = env.unwrapped.step_dt

        # the cap the action term imposes: scale * clip_actions, once per control step
        scale = getattr(env.unwrapped.action_manager.get_term("arm_action"), "_scale", None)
        cap = None
        if isinstance(scale, float) and agent_cfg.clip_actions:
            cap = scale * float(agent_cfg.clip_actions) / dt
            print(f"[INFO]: commanded cap = {scale:.4f} rad/step x {agent_cfg.clip_actions} / {dt:.4f} s"
                  f" = {cap:.3f} rad/s")

        reward_manager = env.unwrapped.reward_manager
        termination_manager = env.unwrapped.termination_manager
        reward_names = list(reward_manager.active_terms)
        done_names = list(termination_manager.active_terms)

        joint_speeds, ee_speeds = [], []
        step_rewards, term_dones = [], []
        previous_hand = None
        obs = env.get_observations()
        with torch.inference_mode():
            for step in range(args_cli.warmup + args_cli.steps):
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                hand = robot.data.body_link_pose_w.torch[:, hand_id, :3].clone()
                if step >= args_cli.warmup:
                    joint_speeds.append(robot.data.joint_vel.torch[:, arm_ids].abs().clone())
                    # finite-differenced rather than read off a velocity buffer, so it means the
                    # same thing whichever physics backend produced it
                    if previous_hand is not None:
                        ee_speeds.append(((hand - previous_hand).norm(dim=-1) / dt).clone())
                    if not args_cli.no_rewards:
                        # the manager stores each term as a *rate*; multiplying by dt undoes that
                        # and gives what the term actually added to the return this step, so the
                        # columns sum to the return rather than to 30 times it
                        step_rewards.append(reward_manager._step_reward.clone() * dt)
                        term_dones.append(termination_manager._term_dones.clone())
                previous_hand = hand

        joint_speeds = torch.stack(joint_speeds).cpu()
        ee_speeds = torch.stack(ee_speeds).cpu()

        print(f"\n{'joint':>16} {'median':>9} {'p95':>9} {'max':>9} {'max/limit':>10}")
        for name, median, p95, peak, fraction in _summarize(joint_speeds, arm_names, FRANKA_JOINT_VEL_LIMITS):
            print(f"{name:>16} {median:>9.3f} {p95:>9.3f} {peak:>9.3f} {fraction:>9.0%}")
        flat = ee_speeds.flatten()
        print(f"\n{'end-effector':>16} {flat.median():>9.3f} {flat.quantile(0.95):>9.3f} {flat.max():>9.3f}   [m/s]")
        if cap is not None:
            worst = joint_speeds.max().item()
            print(f"\nfastest joint reached {worst:.3f} rad/s = {worst / cap:.0%} of the {cap:.3f} rad/s cap")

        out = args_cli.out or os.path.join(os.path.dirname(resume_path), "joint_speed.png")
        _plot(out, joint_speeds, ee_speeds, arm_names, dt, cap)
        print(f"[INFO]: wrote {out}")

        if not args_cli.no_rewards:
            step_rewards = torch.stack(step_rewards).cpu()
            term_dones = torch.stack(term_dones).cpu()

            # which term ended each episode. Taken from environment 0: with several environments
            # the endings interleave and a single timeline cannot honestly label them all.
            ends, labels = [], []
            for step_index, row in enumerate(term_dones[:, 0, :]):
                fired = row.nonzero(as_tuple=True)[0]
                if len(fired):
                    ends.append(step_index)
                    labels.append("+".join(done_names[i] for i in fired.tolist()))

            totals = step_rewards.mean(dim=1).sum(dim=0)
            episodes = max(len(ends), 1)
            print(f"\n{'reward term':>22} {'total':>12} {'per episode':>12} {'per step':>10}")
            for index in sorted(range(len(reward_names)), key=lambda i: -abs(totals[i].item())):
                total = totals[index].item()
                if abs(total) < 1e-6:
                    continue
                print(
                    f"{reward_names[index]:>22} {total:>12,.1f} {total / episodes:>12,.1f}"
                    f" {total / len(step_rewards):>10,.2f}"
                )
            net = totals.sum().item()
            print(f"{'NET':>22} {net:>12,.1f} {net / episodes:>12,.1f} {net / len(step_rewards):>10,.2f}")
            print(f"\n{len(ends)} episode(s) ended in env 0: {', '.join(labels) if labels else 'none'}")

            reward_out = args_cli.reward_out or os.path.join(os.path.dirname(resume_path), "reward_terms.png")
            _plot_rewards(reward_out, step_rewards, reward_names, ends, labels, dt)
            print(f"[INFO]: wrote {reward_out}")

        env.close()


if __name__ == "__main__":
    main()
