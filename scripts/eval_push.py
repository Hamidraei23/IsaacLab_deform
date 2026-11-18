# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score a checkpoint on episodes whose extraction was disturbed by the push.

Runs the deterministic policy for a fixed number of episodes per environment with the automatic
push armed -- by default on *every* extraction -- and reports how many of the pushed episodes still
ended in a finished drape, and how evenly that drape fell.

Written as its own script rather than as a flag on ``scripts/play.py`` for the same reason
``eval_joint_speed.py`` is: playback hands straight off to Isaac Lab's rollout loop, which has no
per-step hook, and the figures wanted here are per-*episode* and have to be paired with whether that
particular episode was pushed. ``play.py --push`` still exists for watching one.

The three figures reported per model:

* **pushed** -- episodes that extracted and were therefore pushed. With the default probability of
  1.0 this is exactly the episodes that got the sheet out of the slot.
* **success rate** -- of those, the share that went on to a finished drape, i.e. the ``draped``
  termination: the sheet out of the gripper and covering the band past its threshold for two
  unbroken seconds.
* **symmetry** -- ``success_symmetry`` averaged over those successes: 1.0 when every corner of the
  sheet reached the table at the moment it scored, tapering to 0 when the highest corner is
  ``symmetry_span`` above it. It is a measure of drape quality, not of success, so it is only
  meaningful over the episodes that succeeded.

The unpushed episodes are reported alongside as the control. At probability 1.0 they are the
episodes that never extracted, so their success rate is zero by construction; run with
``--push_probability 0.5`` to get a like-for-like comparison of disturbed against undisturbed
extractions within one rollout.

Usage::

    python3 scripts/eval_push.py --task Template-Sheet-Rl-v0 --num_envs 16 --episodes 10 \
        --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_10500.pt \
        agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
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

SUCCESS_TERMINATION = "draped"
"""The one ending that is not a failure: the drape held for its full dwell."""

DRAPE_TERM = "drape_stage"
"""Reward term owning the symmetry recorded at the instant an episode scored."""

parser = argparse.ArgumentParser(description="Score a checkpoint on pushed extractions.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--episodes", type=int, default=10, help="Episodes to score per environment.")
parser.add_argument(
    "--push_probability",
    type=float,
    default=1.0,
    help="Share of extractions to push. 1.0 pushes every one, which is what makes the pushed sample"
    " as large as the rollout allows.",
)
parser.add_argument(
    "--no_push", action="store_true", default=False, help="Score without any push, as the baseline."
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


def _report(name: str, pushed: torch.Tensor, success: torch.Tensor, symmetry: torch.Tensor) -> None:
    """Print the success rate and drape symmetry for one slice of the episodes.

    Args:
        name: What the slice is.
        pushed: Whether each episode was pushed. Shape ``(episodes,)``.
        success: Whether each episode ended in a finished drape. Same shape.
        symmetry: ``success_symmetry`` for each episode; meaningless where it did not succeed.
    """
    count = int(pushed.sum())
    if count == 0:
        print(f"{name:>22} {0:>9} {'--':>14} {'--':>12}")
        return
    scored = success & pushed
    rate = int(scored.sum()) / count
    even = symmetry[scored].mean().item() if bool(scored.any()) else float("nan")
    print(f"{name:>22} {count:>9} {rate:>13.1%} {even:>11.1%}")


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out the deterministic policy and score the episodes it pushed."""
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed

    # -- arm the push. Set here rather than left to a Hydra override so the script cannot be run
    #    with the very thing it measures switched off by accident.
    env_cfg.push.on_extraction = not args_cli.no_push
    env_cfg.push.on_extraction_probability = args_cli.push_probability

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

        inner = env.unwrapped
        device = inner.device
        drape = inner.reward_manager.get_term_cfg(DRAPE_TERM).func
        terminations = inner.termination_manager

        # -- per-episode accumulators.
        #
        # Every one of these is cleared inside ``step`` for an environment that reset on that step
        # -- the push latch by the environment, the symmetry by the reward term's ``reset`` -- so
        # they are accumulated *during* the episode and read out when the done arrives, rather than
        # sampled on the done step itself, where they would all read zero.
        pushed = torch.zeros(inner.num_envs, dtype=torch.bool, device=device)
        symmetry = torch.zeros(inner.num_envs, device=device)
        finished = torch.zeros(inner.num_envs, dtype=torch.long, device=device)
        records: list[tuple[bool, bool, float]] = []

        target = args_cli.episodes
        # every episode ends by the timeout at the latest, so this cannot loop forever; the slack
        # covers the partial episode each environment starts in
        step_budget = (target + 1) * int(inner.max_episode_length)

        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(step_budget):
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                # the RL wrapper hands dones back as an integer tensor
                dones = dones.bool()

                if inner._auto_push:
                    pushed |= inner._pushed_episode
                symmetry = torch.maximum(symmetry, drape._success_symmetry)

                if bool(dones.any()):
                    # read the success off the termination that fired, which is current on this
                    # step for exactly the environments that are done
                    scored = terminations.get_term(SUCCESS_TERMINATION)
                    for env_id in dones.nonzero(as_tuple=False).flatten().tolist():
                        # environments run at different speeds, so the fast ones are left to run on
                        # rather than being frozen; their extra episodes are simply not scored
                        if int(finished[env_id]) < target:
                            records.append(
                                (bool(pushed[env_id]), bool(scored[env_id]), float(symmetry[env_id]))
                            )
                        finished[env_id] += 1
                    keep = ~dones
                    pushed &= keep
                    symmetry = symmetry * keep.float()

                if int(finished.min()) >= target:
                    break

        was_pushed = torch.tensor([record[0] for record in records])
        success = torch.tensor([record[1] for record in records])
        even = torch.tensor([record[2] for record in records])

        print(f"\n[INFO]: {resume_path}")
        print(
            f"[INFO]: {len(records)} episodes over {inner.num_envs} environments,"
            f" push {'off' if args_cli.no_push else f'on {args_cli.push_probability:.0%} of extractions'}"
        )
        print(f"\n{'episodes':>22} {'count':>9} {'success rate':>13} {'symmetry':>12}")
        _report("pushed", was_pushed, success, even)
        _report("not pushed", ~was_pushed, success, even)
        _report("all", torch.ones_like(was_pushed), success, even)

        env.close()


if __name__ == "__main__":
    main()
