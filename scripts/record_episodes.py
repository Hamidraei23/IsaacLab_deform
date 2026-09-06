# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a checkpoint run and save the episodes worth keeping as action traces.

The same rollout ``scripts/play.py`` shows, with a recorder attached. Every episode is buffered as
it runs; the moment one ends -- for any reason, success or failure -- the ending is named and the
episode is offered for saving. Answer yes and the whole thing is written as a CSV, named after the
arm joint pose it started from::

    1J(0.000),2J(-0.535),3J(0.000),4J(-2.938),5J(0.000),6J(3.145),7J(0.702).csv

Each row is one control step: the step index, its time in seconds, the eight action dimensions
(``act_*`` -- seven relative arm joint targets and the binary gripper channel), and the measured
arm joint positions those actions were issued from (``pos_*``).

The positions are what make the trace usable as a reference trajectory. The arm actions are
*relative* -- the environment turns each one into ``measured position + action * scale`` -- so on
their own they carry no absolute pose, and integrating them open-loop does not reproduce the
motion: the joint lags its target every step, and the next action is issued from wherever it
actually got to. Recording the measured position alongside is what closes that gap.

Written as its own script rather than a flag on ``play.py``, which hands straight off to Isaac Lab's
playback backend and has no per-step hook to record from. The rollout loop here is the one from
``scripts/eval_joint_speed.py``, which already had to be written out for the same reason.

Usage::

    PYOPENGL_PLATFORM=glx python3 scripts/record_episodes.py --rl_library rsl_rl \
        --task Template-Sheet-Rl-Hand-v0 --num_envs 1 --viz kit \
        --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_10500.pt \
        agent.experiment_name=sheet_grasp \
        agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
        agent.actor.obs_normalization=true agent.critic.obs_normalization=true \
        agent.clip_actions=2.4
"""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import json
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

parser = argparse.ArgumentParser(description="Record and save episodes from a trained checkpoint.")
# Accepted and ignored, so a working ``play.py`` command can be pasted here with only the script
# name changed. ``play.py`` needs it to pick a backend; this script, like eval_joint_speed.py, is
# rsl-rl only and has nothing to dispatch on. Left out entirely it would fall through to Hydra as
# a stray override and fail there instead, which is a worse way to say the same thing.
parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl"], help="Ignored; rsl-rl only.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--episodes", type=int, default=0, help="Episode endings to offer before quitting. 0 runs until interrupted."
)
parser.add_argument("--env_id", type=int, default=0, help="Environment index to record, if more than one is running.")
parser.add_argument(
    "--out_dir", type=str, default=None,
    help="Where to write traces. Defaults to logs/rsl_rl/<experiment>/saved_trajectories/new_control_ref."
)
parser.add_argument(
    "--precision", type=int, default=3, help="Decimal places for the joint angles in the filename. Defaults to 3."
)
parser.add_argument(
    "--yes", action="store_true", default=False, help="Save every episode without asking. For unattended runs."
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


def _action_columns(action_manager) -> list[str]:
    """One column name per action dimension, in the order the manager slices them.

    Derived from the terms rather than hard-coded as seven joints plus a gripper, so the header
    keeps telling the truth if a term is ever added, reordered or resized. ``active_terms`` and
    ``action_term_dim`` are both in construction order, which is the order the action vector is cut
    into, so walking them together recovers the layout.
    """
    names = []
    for term_name, width in zip(action_manager.active_terms, action_manager.action_term_dim):
        term = action_manager.get_term(term_name)
        joint_names = list(getattr(term, "_joint_names", []))
        # a binary gripper drives two joints from one channel, so its joint names do not line up
        # with its width; only label per joint when the two actually match
        if len(joint_names) == width:
            names.extend(joint_names)
        elif width == 1:
            names.append(term_name)
        else:
            names.extend(f"{term_name}_{index}" for index in range(width))
    return names


def _pose_name(values: torch.Tensor, precision: int) -> str:
    """The filename stem for a trajectory that started from arm joint pose ``values``.

    The format asked for: one ``<n>J(<angle>)`` group per arm joint, numbered from one, comma
    separated. Commas are legal in a filename but do need quoting in a shell, which is the price of
    the pose being readable at a glance in a directory listing.
    """
    return ",".join(f"{index + 1}J({value:.{precision}f})" for index, value in enumerate(values.tolist()))


def _unique_path(directory: str, stem: str, suffix: str) -> str:
    """``stem + suffix`` inside ``directory``, with a counter appended if that name is taken.

    Two episodes can start from the same pose to within the printed precision. Three of the seven
    joints have a default of zero and ``reset_joints_by_scale`` multiplies, so those three read
    zero in every episode and only four digits actually vary. Silently overwriting the earlier
    trace would be the worst way to handle that.
    """
    candidate = os.path.join(directory, stem + suffix)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


def _ask(prompt: str) -> bool:
    """Yes/no on the terminal, defaulting to no on anything unexpected.

    Returns ``False`` rather than raising when stdin is closed or the user interrupts, so a piped
    or backgrounded run degrades to "record but keep nothing" instead of dying mid-rollout with the
    simulator still up.
    """
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run the deterministic policy, buffering each episode's actions and offering them for saving."""
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

        # One collection point rather than a folder per run. Traces are compared against each other
        # and fed to the replay controller as a set, so scattering them beside whichever checkpoint
        # produced them meant hunting through run directories to assemble a batch.
        #
        # The cost is that provenance is no longer implied by location: two traces sitting side by
        # side here can come from different policies. The sidecar's ``checkpoint`` field is the only
        # thing that distinguishes them, and it is worth reading before trusting a batch -- an
        # earlier set that looked like new-policy output turned out to be ten byte-identical copies
        # of a baseline recording, and that field was what caught it.
        out_dir = args_cli.out_dir or os.path.join(log_root_path, "saved_trajectories", "new_control_ref")
        os.makedirs(out_dir, exist_ok=True)

        robot = env.unwrapped.scene["robot"]
        dt = env.unwrapped.step_dt
        env_id = args_cli.env_id
        clip = agent_cfg.clip_actions

        action_manager = env.unwrapped.action_manager

        # the filename's arm pose, the joints whose measured position is recorded, and the scale
        # that turns a recorded action back into a joint target -- all read off the action term
        # rather than restated here
        arm_term = action_manager.get_term("arm_action")
        arm_names = list(arm_term._joint_names)
        arm_ids = arm_term._joint_ids
        arm_scale = arm_term._scale

        # ``act_`` and ``pos_`` rather than bare joint names. The two groups are different
        # quantities in different units -- a relative command in action units against an absolute
        # angle in radians -- and a header that did not separate them would invite exactly the
        # mistake of reading a command as a pose.
        columns = (
            ["step", "time_s"]
            + [f"act_{name}" for name in _action_columns(action_manager)]
            + [f"pos_{name}" for name in arm_names]
        )

        termination_manager = env.unwrapped.termination_manager
        done_names = list(termination_manager.active_terms)

        print(
            f"[INFO]: recording env {env_id} at {1.0 / dt:.0f} Hz into {out_dir}\n"
            f"[INFO]: policy: {os.path.basename(resume_path)} from {os.path.basename(os.path.dirname(resume_path))}\n"
            f"[INFO]: {action_manager.total_action_dim} action dims + {len(arm_names)} joint"
            f" positions = {len(columns)} columns, clip_actions={clip}, arm scale={arm_scale}"
        )

        rows: list[list[float]] = []
        start_pose: torch.Tensor | None = None
        episodes = saved = 0
        step_in_episode = 0
        episode_return = 0.0

        obs = env.get_observations()
        with torch.inference_mode():
            while args_cli.episodes == 0 or episodes < args_cli.episodes:
                # Read *before* stepping, which matters twice over. ``ManagerBasedRLEnv.step``
                # resets a finished environment inside the same call, so on a terminal step
                # anything read afterwards already belongs to the next episode -- the trace would
                # be filed under the wrong trajectory's pose, and its last row would carry a
                # position from an episode that had not started yet. It is also the honest pairing:
                # this is the position the policy saw and issued its action from, which is what
                # ``measured position + action * scale`` is evaluated against.
                joint_pos = robot.data.joint_pos.torch[env_id, arm_ids].clone()
                if step_in_episode == 0:
                    start_pose = joint_pos.clone()

                actions = policy(obs)
                if clip is not None:
                    # clipped here as well as inside the wrapper, so the row records the action
                    # that was applied rather than the one the network asked for
                    actions = torch.clamp(actions, -clip, clip)

                rows.append(
                    [step_in_episode, step_in_episode * dt]
                    + actions[env_id].tolist()
                    + joint_pos.tolist()
                )
                step_in_episode += 1

                obs, reward, dones, _ = env.step(actions)
                policy.reset(dones)
                episode_return += reward[env_id].item()

                if not bool(dones[env_id]):
                    continue

                # ``_term_dones`` is written by the compute that produced this step's dones and is
                # not cleared by the reset that follows, so it still names what ended the episode
                fired = termination_manager._term_dones[env_id].nonzero(as_tuple=True)[0]
                ending = [done_names[index] for index in fired.tolist()]

                episodes += 1
                stem = _pose_name(start_pose, args_cli.precision)
                print(
                    f"\n[episode {episodes}] ended after {step_in_episode} steps"
                    f" ({step_in_episode * dt:.1f} s) on: {'+'.join(ending) or 'unknown'}"
                    f"\n            return {episode_return:,.1f}   start pose {stem}"
                )

                if args_cli.yes or _ask("            save this episode? [y/N] "):
                    path = _unique_path(out_dir, stem, ".csv")
                    with open(path, "w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(columns)
                        # formatted rather than left to repr: nine significant figures is past what
                        # the simulation's float32 actions carry, and it keeps the file diffable
                        writer.writerows([f"{value:.9g}" for value in row] for row in rows)

                    # Everything needed to interpret the trace, kept out of the CSV so that stays a
                    # plain action time series. ``arm_action_scale`` is the one that matters for a
                    # replay: the arm columns are relative targets, and a joint command is
                    # ``measured position + action * scale``.
                    with open(os.path.splitext(path)[0] + ".json", "w") as handle:
                        json.dump(
                            {
                                "task": args_cli.task,
                                "checkpoint": resume_path,
                                "seed": env_cfg.seed,
                                "env_id": env_id,
                                "control_hz": round(1.0 / dt, 4),
                                "step_dt": dt,
                                "clip_actions": clip,
                                "arm_action_scale": arm_scale if isinstance(arm_scale, float) else None,
                                "columns": columns,
                                "steps": step_in_episode,
                                "duration_s": round(step_in_episode * dt, 4),
                                "ended_by": ending,
                                "success": "draped" in ending,
                                "return": episode_return,
                                "initial_arm_joint_pos": dict(zip(arm_names, start_pose.tolist())),
                            },
                            handle,
                            indent=2,
                        )
                    saved += 1
                    print(f"            saved {path}")
                else:
                    print("            discarded")

                rows = []
                step_in_episode = 0
                episode_return = 0.0

        print(f"\n[INFO]: {saved} of {episodes} episode(s) saved into {out_dir}")
        env.close()


if __name__ == "__main__":
    main()
