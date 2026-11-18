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
end-effector's linear speed. The same rollout also records what every reward term paid on every
step. The two answer different questions about one run -- how the arm moved, and what it was paid
for -- and recording both from a single rollout means they describe the *same* episodes rather than
two runs that happened to be configured alike.

**Everything drawn is drawn per episode.** A rollout is several attempts one after another, and a
figure spanning all of them averages a good episode with a failed one and shows the mean of two
things that never happened. Each episode therefore gets its own files, written into ``episodes``
beside the checkpoint, and the script asks whether its plots are worth keeping *as each episode
ends* -- answer ``a`` to stop asking and keep them all, ``q`` to stop asking and keep none. The
rollout pauses while it waits, viewer included, which is the point: it is a chance to look at what
just happened before deciding. Every answer then carries on to the next episode, ``n`` included --
declining the pictures declines the pictures, it does not end the run. Pass ``--save_plots
all``/``none`` to decide up front, or ``--rollout_plots`` to also get the two whole-rollout figures
this script used to write on its own.

Per episode, four figures and two tables:

* ``ep<n>_corners.csv`` / ``.png`` -- the sheet's four corner nodes in world coordinates, one row
  per control step.
* ``ep<n>_ee_pose.csv`` / ``.png`` -- the hand's 6-D pose, ``x, y, z`` and roll/pitch/yaw, in metres
  and radians in the file and degrees on the plot. All six are *relative to where the episode
  started*: the first sample is subtracted, so every episode opens at ``0, 0, 0`` and ``0, 0, 0``
  and two of them can be laid over each other whatever reset poses they were dealt. The angles are
  unwrapped before that, so a roll through the +/-180 boundary reads as a continuous turn rather
  than a jump. The absolute start is printed with the file and named on the figure, so nothing is
  lost by it.
* ``ep<n>_joint_speed.png`` and ``ep<n>_rewards.png`` -- the two original figures, over that
  episode alone.

Every figure washes the same interval over itself: from the moment the gripper first holds the sheet
to the moment it comes clear of the slot. Both moments are read off the reward terms' own flags --
the observed grasp ``grasp_stage_reward`` publishes and the phase latch it sets on extraction -- so
the shaded region is the one the policy was actually paid for, and the same instants line up across
all four plots. An episode that grasped but never extracted is shaded from the grasp to its end and
labelled as such; one that never grasped is not shaded at all.

The CSVs are written whatever the answer to the prompt: the numbers are the record, the pictures
are a convenience.

**The run does not stop on its own.** Episodes keep coming until Ctrl-C: each runs to its own
ending or to ``--steps``, whichever arrives first, and one cut short by that limit is written,
marked ``_partial``, and followed by an explicit reset so the next record starts from a fresh sheet
rather than from one already halfway draped. ``--steps`` is therefore the longest episode to record,
not a budget for the run. Ctrl-C stops after the episode in progress, which is left unwritten, and
the summary tables are still printed.

The records come from one environment, ``--env_index``, because an episode is a property of an
environment.

Usage::

    python3 scripts/eval_joint_speed.py --task Template-Sheet-Rl-v0 --num_envs 16 --headless \
        --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_2380.pt \
        --steps 600 agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
        agent.actor.obs_normalization=true agent.critic.obs_normalization=true agent.clip_actions=2.4
"""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import itertools
import os
import sys

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat
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
from sheet_rl.tasks.manager_based.sheet_rl.mdp import grasp_debug, sheet_extracted

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
parser.add_argument(
    "--steps",
    type=int,
    default=600,
    help="Longest episode to record, in steps. Not a budget for the run: episodes keep coming"
    " until Ctrl-C.",
)
parser.add_argument("--warmup", type=int, default=30, help="Steps to discard before recording.")
parser.add_argument("--out", type=str, default=None, help="Plot path. Defaults to beside the checkpoint.")
parser.add_argument(
    "--reward_out", type=str, default=None, help="Reward-term plot path. Defaults to beside the checkpoint."
)
parser.add_argument(
    "--no_rewards", action="store_true", default=False, help="Skip recording and plotting the reward terms."
)
parser.add_argument(
    "--env_index", type=int, default=0, help="Which environment the per-episode records describe."
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Where the per-episode files go. Defaults to 'episodes' beside the checkpoint.",
)
parser.add_argument(
    "--save_plots",
    type=str,
    default="ask",
    choices=("ask", "all", "none"),
    help="Per-episode plots: ask after each episode, save them all, or save none. The CSVs are"
    " written either way. Falls back to 'all' when stdin is not a terminal.",
)
parser.add_argument(
    "--rollout_plots",
    action="store_true",
    default=False,
    help="Also write the two whole-rollout figures this script used to write on its own.",
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


def _ask_save(prompt: str, mode: str) -> tuple[bool, str]:
    """Ask whether this episode's plots should be written.

    Blocks the run while it waits, viewer included -- which is the point: it is a chance to look at
    what just happened before deciding to keep it.

    Args:
        prompt: What to show, already describing the episode that just ended.
        mode: ``ask`` to ask, ``all`` to save without asking, ``none`` to skip without asking.

    Returns:
        Whether to save this episode, and the mode to carry into the next one -- ``a`` and ``q``
        answer for the whole rest of the run.
    """
    if mode != "ask":
        return mode == "all", mode
    try:
        # every answer carries on with the rollout; ``n`` and ``q`` decline the pictures, they do
        # not stop the recording
        answer = (
            input(f"{prompt} save plots? [y]es / [n]o / [a]ll / [q]uit asking, then keep recording (y): ")
            .strip()
            .lower()
        )
    except EOFError:
        # stdin closed mid-run: keep everything rather than silently dropping it
        return True, "all"
    if answer.startswith("a"):
        return True, "all"
    if answer.startswith("q"):
        return False, "none"
    return not answer.startswith("n"), "ask"


HIGHLIGHT_COLOR = "#4f81bd"
"""Wash used for the grasp-to-extraction interval. The same blue the reward bars use."""


def _grasp_span(times, holding, extracted):
    """When the sheet was taken hold of and when it came clear of the slot.

    Both flags are the reward terms' own: ``holding`` is the observed grasp -- gripper shut with
    cloth between the pads -- that :class:`grasp_stage_reward` publishes each step, and ``extracted``
    is the phase latch it sets once the whole sheet is above the slot walls. Reading them rather
    than re-deriving the conditions here means the shaded interval is the one the rewards were paid
    for, not an approximation of it.

    Args:
        times: Time of each sample [s]. Shape ``(steps,)``.
        holding: Whether the sheet was held on each step. Shape ``(steps,)``.
        extracted: Whether the extraction latch was set on each step. Same shape.

    Returns:
        ``(start, stop, label)`` in seconds, or ``None`` when the episode never grasped, or when it
        was already past both by the time recording began -- a partial first episode can start with
        the latch already set, and there is no interval to draw for something that happened before
        the first sample.
    """
    grasps = holding.nonzero(as_tuple=True)[0]
    if not len(grasps):
        return None
    start = int(grasps[0])

    # an episode already holding on its first sample was grasped before recording began -- during
    # the warmup, or in the part of a partial episode that was never seen -- so the shading marks
    # where the record starts rather than where the grasp happened, and says so
    opening = "already holding" if bool(holding[0]) else "grasp"

    extractions = extracted.nonzero(as_tuple=True)[0]
    if len(extractions) and int(extractions[0]) > start:
        stop = int(extractions[0])
        return float(times[start]), float(times[stop]), f"{opening} to extraction"
    if len(extractions):
        # the latch was already set when the recording started, so the grasp being seen here is a
        # hold during the drape rather than the one that pulled the sheet out
        return None
    # held, but never got it clear of the slot: the interval is still the interesting part of the
    # episode, and saying where it stops matters
    return float(times[start]), float(times[-1]), f"{opening} onwards (never extracted)"


def _shade(axes, highlight) -> None:
    """Wash the grasp-to-extraction interval across every time-series panel given.

    Call it before the legends are built, so the label lands in them.
    """
    if highlight is None:
        return
    start, stop, label = highlight
    for index, axis in enumerate(axes):
        axis.axvspan(
            start, stop, color=HIGHLIGHT_COLOR, alpha=0.13, zorder=0, label=label if index == 0 else None
        )


def _write_csv(path: str, header: list[str], rows) -> None:
    """Write one table, one row per control step."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_corners(path, times, corners, title, highlight=None):
    """Plot the sheet's four corners, one panel per world axis.

    Args:
        path: Where to write the PNG.
        times: Time of each sample [s]. Shape ``(steps,)``.
        corners: World positions of the four corners. Shape ``(steps, 4, 3)``.
        title: Episode label for the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    figure.suptitle(f"Sheet corners -- {title}")
    _shade(axes, highlight)
    names = ("corner 0 (0,0)", "corner 1 (0,n)", "corner 2 (n,0)", "corner 3 (n,n)")
    for axis_index, axis_name in enumerate("xyz"):
        for corner_index, name in enumerate(names):
            axes[axis_index].plot(times, corners[:, corner_index, axis_index], linewidth=1.2, label=name)
        axes[axis_index].set_ylabel(f"{axis_name} [m]")
        axes[axis_index].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("time [s]")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _relative_angles(eulers: torch.Tensor) -> torch.Tensor:
    """Turn absolute euler angles into a continuous trace starting at zero.

    Two things happen here, and both are needed for the answer to mean anything:

    * **Unwrapping.** ``euler_xyz_from_quat`` returns angles in ``(-pi, pi]``, so a hand rolling
      smoothly through half a turn is reported as jumping from ``+3.14`` to ``-3.14``. Each step's
      difference is folded back into ``(-pi, pi]`` and re-accumulated, which is what
      :func:`numpy.unwrap` does, so the trace runs continuously and a rotation of 200 degrees reads
      as 200 rather than as -160.
    * **Zeroing.** The first sample is then subtracted from all of them, so every episode starts at
      0 and what is plotted is how far the hand has turned since it began -- comparable between
      episodes that started from different reset poses, which the absolute angles are not.

    Args:
        eulers: Roll, pitch and yaw over one episode [rad]. Shape ``(steps, 3)``.

    Returns:
        The same angles, unwrapped and offset to start at zero. Shape ``(steps, 3)``.
    """
    two_pi = 2.0 * torch.pi
    steps = eulers[1:] - eulers[:-1]
    folded = steps - two_pi * torch.round(steps / two_pi)
    unwrapped = torch.cat([eulers[:1], eulers[:1] + torch.cumsum(folded, dim=0)], dim=0)
    return unwrapped - unwrapped[:1]


def _plot_ee_pose(path, times, positions, eulers, title, reference, highlight=None):
    """Plot the end-effector's motion away from the pose it started the episode in.

    Args:
        path: Where to write the PNG.
        times: Time of each sample [s]. Shape ``(steps,)``.
        positions: Hand position relative to the episode's first sample [m]. Shape ``(steps, 3)``.
        eulers: Roll, pitch and yaw relative to that same sample [rad], as returned by
            :func:`_relative_angles`. Shape ``(steps, 3)``.
        title: Episode label for the figure.
        reference: The absolute pose the episode started from -- ``x, y, z`` [m] then roll, pitch
            and yaw [rad] -- named on the plot so the zeroed traces can still be tied back to a real
            pose. Shape ``(6,)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    figure.suptitle(f"End-effector pose -- {title}")
    _shade(axes, highlight)

    for index, name in enumerate("xyz"):
        axes[0].plot(times, positions[:, index], linewidth=1.2, label=name)
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel("moved since start [m]")
    axes[0].set_title(
        f"relative to the point it started at: {reference[0]:.3f}, {reference[1]:.3f},"
        f" {reference[2]:.3f} m",
        fontsize=9,
    )
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.3)

    # degrees on the plot, radians in the CSV: the file is for arithmetic and the picture is for
    # reading, and 90 is easier to recognise on an axis than 1.5708
    start = torch.rad2deg(reference[3:])
    for index, name in enumerate(("roll", "pitch", "yaw")):
        axes[1].plot(times, torch.rad2deg(eulers[:, index]), linewidth=1.2, label=name)
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].set_ylabel("turned since start [deg]")
    axes[1].set_title(
        f"relative to the pose it started in: roll {start[0]:.1f}, pitch {start[1]:.1f},"
        f" yaw {start[2]:.1f} deg",
        fontsize=9,
    )
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(alpha=0.3)

    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot(path, joint_speeds, ee_speed, names, dt, commanded_cap, highlight=None):
    """Write the three-panel figure. Imported lazily so the sim can run without a display stack."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = torch.arange(joint_speeds.shape[0]) * dt
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    # only the first two panels are time series; the third is a bar chart over joints
    _shade(axes[:2], highlight)

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


def _plot_rewards(path, step_rewards, term_names, episode_ends, end_labels, dt, highlight=None):
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
    _shade(axes[:2], highlight)

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

        # -- what the per-episode records are taken from.
        #
        # One environment, because an episode is a property of an environment: with several running
        # they start and end at different steps, and a file holding all of them would be a table of
        # unrelated attempts. ``--env_index`` picks which one.
        tracked = args_cli.env_index
        sheet = env.unwrapped.scene["deformable"]
        marker_cfg = getattr(env.unwrapped.cfg, "edge_marker", None)
        resolution = getattr(marker_cfg, "resolution", (8, 8))
        cols, rows = resolution[0] + 1, resolution[1] + 1
        # the same four nodes ``drape_milestones`` calls the corners, in the same order
        corner_ids = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]

        out_dir = args_cli.out_dir or os.path.join(os.path.dirname(resume_path), "episodes")
        os.makedirs(out_dir, exist_ok=True)
        save_mode = args_cli.save_plots
        if save_mode == "ask" and not sys.stdin.isatty():
            print("[INFO]: stdin is not a terminal, so every episode's plots are saved without asking.")
            save_mode = "all"

        episode_index = 0
        episode_start = 0
        first_is_partial = True

        def flush_episode(stop: int, ending: str) -> None:
            """Write one episode's CSVs, then its plots if they are wanted.

            Args:
                stop: Index one past the episode's last recorded step.
                ending: What ended it, for the prompt and the figure titles.
            """
            nonlocal episode_index, episode_start, save_mode, first_is_partial

            span = slice(episode_start, stop)
            length = stop - episode_start
            episode_start = stop
            if length < 2:
                # a single sample is not an episode worth writing; happens when a done lands on the
                # first recorded step, i.e. the warmup ended right at an episode boundary
                episode_index += 1
                first_is_partial = False
                return

            partial = first_is_partial or ending in ("still running", "step limit")
            first_is_partial = False
            label = f"ep{episode_index:02d}" + ("_partial" if partial else "")
            title = f"{label}, {length} steps, ended by {ending}"
            times = torch.arange(length) * dt

            corner_track = torch.stack(corners_log[span])
            pose_track = torch.stack(ee_pose_log[span])
            roll, pitch, yaw = euler_xyz_from_quat(pose_track[:, 3:7])
            # The whole 6-D pose is written relative to the episode's first sample, so every episode
            # starts at (0, 0, 0) and (0, 0, 0) and two of them can be laid over each other whatever
            # reset poses they happened to be dealt. The angles are unwrapped first, so a roll
            # through the +/-180 boundary reads as a turn rather than a jump -- see
            # ``_relative_angles``. The absolute start is kept in ``reference`` and reported.
            eulers = _relative_angles(torch.stack([roll, pitch, yaw], dim=-1))
            positions = pose_track[:, :3] - pose_track[:1, :3]
            reference = torch.cat([pose_track[0, :3], torch.stack([roll[0], pitch[0], yaw[0]])])

            # the interval every figure washes over: the sheet in the gripper, on its way out of the
            # slot. Shared by all four so the same moment lines up across them.
            highlight = _grasp_span(
                times, torch.stack(holding_log[span]), torch.stack(extracted_log[span])
            )

            corners_csv = os.path.join(out_dir, f"{label}_corners.csv")
            _write_csv(
                corners_csv,
                ["step", "time_s"]
                + [f"corner{c}_{axis}" for c in range(4) for axis in "xyz"],
                (
                    [step, round(float(times[step]), 6)]
                    + [round(float(value), 6) for value in corner_track[step].flatten()]
                    for step in range(length)
                ),
            )
            pose_csv = os.path.join(out_dir, f"{label}_ee_pose.csv")
            _write_csv(
                pose_csv,
                ["step", "time_s", "x", "y", "z", "roll_rad", "pitch_rad", "yaw_rad"],
                (
                    [step, round(float(times[step]), 6)]
                    + [round(float(value), 6) for value in positions[step]]
                    + [round(float(value), 6) for value in eulers[step]]
                    for step in range(length)
                ),
            )
            span_note = (
                f"{highlight[2]}: {highlight[0]:.2f} to {highlight[1]:.2f} s"
                if highlight is not None
                else "no grasp to highlight"
            )
            start_deg = torch.rad2deg(reference[3:])
            print(
                f"[INFO]: wrote {corners_csv} and {pose_csv}\n"
                f"        the pose is relative to the one the episode started in:"
                f" {reference[0]:.3f}, {reference[1]:.3f}, {reference[2]:.3f} m and"
                f" roll {start_deg[0]:.1f}, pitch {start_deg[1]:.1f}, yaw {start_deg[2]:.1f} deg\n"
                f"        {span_note}"
            )

            wanted, save_mode = _ask_save(
                f"\n[{label}] {length} steps, ended by {ending}, {span_note}.", save_mode
            )
            episode_index += 1
            if not wanted:
                return

            _plot_corners(
                os.path.join(out_dir, f"{label}_corners.png"), times, corner_track, title, highlight
            )
            _plot_ee_pose(
                os.path.join(out_dir, f"{label}_ee_pose.png"),
                times,
                positions,
                eulers,
                title,
                reference,
                highlight,
            )
            _plot(
                os.path.join(out_dir, f"{label}_joint_speed.png"),
                torch.stack(joint_speeds[span])[:, tracked : tracked + 1].cpu(),
                torch.stack(ee_speeds[span])[:, tracked : tracked + 1].cpu(),
                arm_names,
                dt,
                cap,
                highlight,
            )
            written = 3
            if not args_cli.no_rewards:
                _plot_rewards(
                    os.path.join(out_dir, f"{label}_rewards.png"),
                    torch.stack(step_rewards[span])[:, tracked : tracked + 1].cpu(),
                    reward_names,
                    [],
                    [],
                    dt,
                    highlight,
                )
                written += 1
            print(f"[INFO]: wrote {written} plots for {label} in {out_dir}")

        joint_speeds, ee_speeds = [], []
        step_rewards, term_dones = [], []
        corners_log, ee_pose_log = [], []
        holding_log, extracted_log = [], []
        previous_hand = None
        obs = env.get_observations()
        # Episodes keep coming until Ctrl-C: each one runs to its own ending or to ``--steps``,
        # whichever comes first, and is then written, asked about, and followed by a fresh one.
        # There is no total budget -- stopping is the operator's call, not the script's.
        print(
            f"[INFO]: recording episodes of up to {args_cli.steps} steps each."
            " Press Ctrl-C to stop after the one in progress."
        )
        interrupted = False
        try:
            with torch.inference_mode():
                for step in itertools.count():
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    hand = robot.data.body_link_pose_w.torch[:, hand_id, :3].clone()
                    if step >= args_cli.warmup:
                        joint_speeds.append(robot.data.joint_vel.torch[:, arm_ids].abs().clone())
                        # finite-differenced rather than read off a velocity buffer, so it means the
                        # same thing whichever physics backend produced it. Zero on the first sample,
                        # which has no predecessor, so the trace stays aligned with everything else.
                        ee_speeds.append(
                            ((hand - previous_hand).norm(dim=-1) / dt).clone()
                            if previous_hand is not None
                            else torch.zeros(env.unwrapped.num_envs, device=hand.device)
                        )
                        # the tracked environment's own record, kept per step and cut into episodes below
                        corners_log.append(sheet.data.nodal_pos_w.torch[tracked, corner_ids].clone().cpu())
                        ee_pose_log.append(robot.data.body_link_pose_w.torch[tracked, hand_id].clone().cpu())
                        # the grasp state machine's own flags, for the highlighted interval. ``holding``
                        # is absent for one step if the reward term has not run yet, which reads as
                        # "not holding" and is right.
                        debug = grasp_debug(env.unwrapped)
                        holding = debug.get("holding")
                        holding_log.append(
                            holding[tracked].clone().cpu()
                            if holding is not None
                            else torch.zeros((), dtype=torch.bool)
                        )
                        extracted_log.append(sheet_extracted(env.unwrapped)[tracked].clone().cpu())
                        if not args_cli.no_rewards:
                            # the manager stores each term as a *rate*; multiplying by dt undoes
                            # that and gives what the term actually added to the return this step,
                            # so the columns sum to the return rather than to 30 times it
                            step_rewards.append(reward_manager._step_reward.clone() * dt)
                            term_dones.append(termination_manager._term_dones.clone())
                        fired = termination_manager._term_dones[tracked].nonzero(as_tuple=True)[0]
                        if len(fired):
                            # The episode ends *inside* this step: the environment has already been
                            # reset by the time ``step`` returns, so everything sampled above
                            # describes the fresh episode, not the one that just finished. The
                            # boundary therefore excludes this sample, and the next episode starts
                            # on it -- no reset is needed here, the environment has done it.
                            flush_episode(
                                len(corners_log) - 1, "+".join(done_names[i] for i in fired.tolist())
                            )
                        elif len(corners_log) - episode_start >= args_cli.steps:
                            # Long enough. Nothing ended it, so the environment is still mid-episode
                            # and has to be sent back to the start explicitly, or the next record
                            # would carry straight on from a sheet already halfway draped.
                            flush_episode(len(corners_log), "step limit")
                            env.reset()
                            obs = env.get_observations()
                            previous_hand = None
                            continue
                    previous_hand = hand

        except KeyboardInterrupt:
            # Wherever it lands -- mid-step, mid-plot, or at the prompt -- Ctrl-C ends the run
            # and leaves the episode in progress unwritten, which is what declining to answer
            # means.
            interrupted = True
        if interrupted:
            print("\n[INFO]: stopped. Anything recorded since the last episode was written is dropped.")

        joint_speeds = torch.stack(joint_speeds).cpu()
        ee_speeds = torch.stack(ee_speeds).cpu()
        if step_rewards:
            step_rewards = torch.stack(step_rewards).cpu()
            term_dones = torch.stack(term_dones).cpu()

        print(f"\n{'joint':>16} {'median':>9} {'p95':>9} {'max':>9} {'max/limit':>10}")
        for name, median, p95, peak, fraction in _summarize(joint_speeds, arm_names, FRANKA_JOINT_VEL_LIMITS):
            print(f"{name:>16} {median:>9.3f} {p95:>9.3f} {peak:>9.3f} {fraction:>9.0%}")
        flat = ee_speeds.flatten()
        print(f"\n{'end-effector':>16} {flat.median():>9.3f} {flat.quantile(0.95):>9.3f} {flat.max():>9.3f}   [m/s]")
        if cap is not None:
            worst = joint_speeds.max().item()
            print(f"\nfastest joint reached {worst:.3f} rad/s = {worst / cap:.0%} of the {cap:.3f} rad/s cap")

        if args_cli.rollout_plots:
            out = args_cli.out or os.path.join(os.path.dirname(resume_path), "joint_speed.png")
            _plot(out, joint_speeds, ee_speeds, arm_names, dt, cap)
            print(f"[INFO]: wrote {out}")

        if not args_cli.no_rewards:
            # which term ended each episode. Taken from the tracked environment: with several
            # environments the endings interleave and a single timeline cannot honestly label them.
            ends, labels = [], []
            for step_index, row in enumerate(term_dones[:, tracked, :]):
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
            print(
                f"\n{len(ends)} episode(s) ended in env {tracked}:"
                f" {', '.join(labels) if labels else 'none'}"
            )

            if args_cli.rollout_plots:
                reward_out = args_cli.reward_out or os.path.join(
                    os.path.dirname(resume_path), "reward_terms.png"
                )
                _plot_rewards(reward_out, step_rewards, reward_names, ends, labels, dt)
                print(f"[INFO]: wrote {reward_out}")

        env.close()


if __name__ == "__main__":
    main()
