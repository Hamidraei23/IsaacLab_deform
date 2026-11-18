"""Train FastSAC (holosoma's soft actor-critic) on an Isaac Lab task, in parallel or in one environment.

The algorithm is ``sheet_rl.algorithms.fast_sac``. Its config comes from the task's
``fast_sac_cfg_entry_point`` (for the twist student: ``SheetTwistFastSACCfg``, actor on
``action_history`` and critic on ``policy``). Runs are written to
``logs/fast_sac/<experiment_name>/<date>_<run_name>/``.

Usage (inside ``active_isaac``)::

    # parallel: holosoma's FastSAC settings (8192 batch, 8 updates per step, 1024 slots per env)
    python scripts/train_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --num_envs 1024 --viz none

    # one environment: classic SAC settings (1M buffer, 256 batch, 1 update per step)
    python scripts/train_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --single_env --viz kit

    # critic only (actor and alpha frozen), resuming from a checkpoint
    python scripts/train_sac.py --task ... --checkpoint logs/fast_sac/.../model_5000.pt agent.train_actor=false

    # critic warm-up on a frozen rsl_rl teacher acting stochastically: 20% of the environments are
    # held out, the critic is scored against their Monte Carlo returns every 100 iterations, and
    # the run stops once the gate passes (see algorithms/fast_sac/validation.py); writes validation.json
    python scripts/train_sac.py --task ... --num_envs 1024 --viz none \
        --teacher logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt --run_name teacher_cwarm

Hydra overrides work as in ``train.py``: ``agent.batch_size=4096``, ``env.rewards.<term>.weight=...``.
"""

import argparse
import os
import sys
from datetime import datetime

import sheet_rl.tasks  # noqa: F401

# Warp captures ``enable_backward`` when a module is created, so it is set before anything that
# defines kernels is imported (see zero_agent.py).
import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.algorithms.fast_sac import CriticGate, FastSACRunner, FrozenTeacher, IsaacLabSacEnv  # noqa: E402
from sheet_rl.algorithms.fast_sac.curriculum import ParamCurriculum  # noqa: E402
from sheet_rl.algorithms.fast_sac.probes import ActionScaleProbe  # noqa: E402

parser = argparse.ArgumentParser(description="Train FastSAC on an Isaac Lab task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point", help="FastSAC config entry point.")
parser.add_argument("--num_envs", type=int, default=None, help="Parallel environments (default: the task's).")
parser.add_argument(
    "--single_env", action="store_true", default=False,
    help="One environment with classic SAC settings (FastSACCfg.single_env_preset); explicit agent.* overrides win.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="Iterations to run (one step of every env each).")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment and the networks.")
parser.add_argument("--run_name", type=str, default=None, help="Suffix of the run directory.")
parser.add_argument("--checkpoint", type=str, default=None, help="FastSAC checkpoint to resume from.")
parser.add_argument(
    "--actor_from", type=str, default=None,
    help="FastSAC checkpoint whose actor and observation normaliser replace the ones in --checkpoint (e.g. a DAgger"
    " student on top of warmed-up critics). The actor optimiser starts fresh.",
)
parser.add_argument(
    "--teacher", type=str, default=None,
    help="rsl_rl checkpoint of a frozen teacher that drives the environments (stochastically) while only the"
    " critic trains: policy evaluation of the teacher. Implies agent.train_actor=false, agent.policy_evaluation=true.",
)
parser.add_argument(
    "--holdout_fraction", type=float, default=None,
    help="Fraction of environments held out to score the critic against Monte Carlo returns (default 0.2 with"
    " --teacher, else 0).",
)
parser.add_argument("--eval_interval", type=int, default=100, help="Iterations between held-out evaluations.")
parser.add_argument(
    "--probe_scales", type=float, nargs="*", default=None,
    help="Every --probe_interval iterations, run --probe_episodes episode(s) per environment at each of these fixed"
    " IK action scales (e.g. 0.05 0.1) and log Probe/*; the transitions are not trained on.",
)
parser.add_argument("--probe_interval", type=int, default=5000)
parser.add_argument("--probe_episodes", type=int, default=1)
parser.add_argument("--probe_mean_action", action="store_true", default=False, help="Probe with the mean action.")
parser.add_argument("--probe_max_steps", type=int, default=400, help="Steps per probed scale; episodes still running count as failures.")
parser.add_argument(
    "--yaw_curriculum", type=float, nargs=3, metavar=("START_DEG", "STEP_DEG", "MAX_DEG"), default=None,
    help="Sheet yaw jitter curriculum: start at +-START_DEG and widen by STEP_DEG (up to MAX_DEG) whenever the"
    " training success has stayed >= --curriculum_threshold for --curriculum_windows consecutive logging windows.",
)
parser.add_argument("--curriculum_threshold", type=float, default=0.75)
parser.add_argument("--curriculum_windows", type=int, default=3)
parser.add_argument("--curriculum_cooldown", type=int, default=300, help="Minimum iterations between two widenings.")
parser.add_argument(
    "--reanchor_interval", type=int, default=0,
    help="Move the anchor's reference to the current actor every N iterations (trust region); 0 = fixed reference.",
)
parser.add_argument(
    "--reanchor", action="store_true", default=False,
    help="Make the loaded actor the anchor's reference (otherwise the anchor saved in the checkpoint, if any, is kept).",
)
parser.add_argument("--no_gate", action="store_true", default=False, help="Do not stop when the validation gate passes.")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args
# top-level agent fields set on the command line (agent.batch_size=..., ...): the single-env preset leaves them
AGENT_OVERRIDES = {arg.split("=", 1)[0].removeprefix("agent.").split(".")[0]
                   for arg in remaining_args if arg.startswith("agent.") and "=" in arg}


def _sync_potential_gamma(env_cfg, gamma: float) -> None:
    """Potential-based shaping (``grasp_progress``) must discount with the learner's gamma, as ReferencePPO enforces."""
    term = getattr(getattr(env_cfg, "rewards", None), "grasp_progress", None)
    if term is not None and "gamma" in term.params and term.params["gamma"] != gamma:
        print(f"[FastSAC] grasp_progress gamma {term.params['gamma']} -> {gamma} (the learner's)")
        term.params["gamma"] = gamma


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.single_env:
        if args_cli.num_envs not in (None, 1):
            raise ValueError("--single_env runs exactly one environment; drop --num_envs")
        agent_cfg.single_env_preset(keep=AGENT_OVERRIDES)
        env_cfg.scene.num_envs = 1
    elif args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.teacher:
        agent_cfg.train_actor = False
        agent_cfg.policy_evaluation = True
    holdout = args_cli.holdout_fraction if args_cli.holdout_fraction is not None else (0.2 if args_cli.teacher else 0.0)
    env_cfg.seed = agent_cfg.seed
    env_cfg.compute_final_obs = True  # true next observations for the replay buffer
    _sync_potential_gamma(env_cfg, agent_cfg.gamma)
    torch.manual_seed(agent_cfg.seed)

    with launch_simulation(env_cfg, args_cli):
        log_root = os.path.abspath(os.path.join("logs", "fast_sac", agent_cfg.experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root, log_dir)
        env_cfg.log_dir = log_dir
        print(f"[FastSAC] logging to {log_dir}")

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        sac_env = IsaacLabSacEnv(env, agent_cfg.actor_obs_groups, agent_cfg.critic_obs_groups)
        agent_cfg.device = str(sac_env.device)
        print(
            f"[FastSAC] {sac_env.num_envs} env(s) | actor {agent_cfg.actor_obs_groups} = {sac_env.n_obs} | "
            f"critic {agent_cfg.critic_obs_groups} = {sac_env.n_critic_obs} (+{sac_env.n_act} action) | "
            f"batch {agent_cfg.batch_size}, {agent_cfg.num_updates} updates/step, buffer "
            f"{agent_cfg.buffer_size}/env | train_actor={agent_cfg.train_actor}"
        )
        runner = FastSACRunner(sac_env, agent_cfg, log_dir, agent_cfg.device)
        if args_cli.checkpoint:
            runner.load(args_cli.checkpoint)
        if args_cli.actor_from:
            runner.alg.load_actor(args_cli.actor_from)
            print(f"[FastSAC] actor and its normaliser taken from {args_cli.actor_from}")
        if args_cli.reanchor:
            runner.alg.set_anchor()
            print("[FastSAC] anchor reference = the actor as loaded")
        print(f"[FastSAC] learning rates in use: {runner.alg.learning_rates()}")
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        behavior = reference_actions = reference_std = None
        if args_cli.teacher:
            if agent_cfg.critic_obs_groups != ["policy"]:
                raise ValueError("--teacher reads the 'policy' group, so critic_obs_groups must be ['policy']")
            teacher = FrozenTeacher(args_cli.teacher, sac_env.n_critic_obs, "policy",
                                    runner.alg.action_low, runner.alg.action_high, agent_cfg.device)
            behavior = _TeacherBehavior(teacher)
            reference_actions, reference_std = teacher.mean, teacher.std
            print(f"[FastSAC] teacher {args_cli.teacher} drives (stochastic, std {teacher.std.tolist()})")
        validator = runner.hold_out(holdout) if holdout > 0 else None
        gate = CriticGate() if (validator is not None and not args_cli.no_gate) else None
        if validator is not None:
            print(f"[FastSAC] {len(validator.env_ids)} environments held out; evaluating every {args_cli.eval_interval}")
        probe = None
        if args_cli.probe_scales:
            probe = ActionScaleProbe(args_cli.probe_scales, args_cli.probe_episodes, args_cli.probe_mean_action,
                                     max_steps=args_cli.probe_max_steps)
            print(f"[FastSAC] probing scales {args_cli.probe_scales} every {args_cli.probe_interval} iterations")
        curriculum = None
        if args_cli.yaw_curriculum:
            import math

            start, step, maximum = (math.radians(v) for v in args_cli.yaw_curriculum)
            curriculum = ParamCurriculum("reset_layout", "sheet_yaw_jitter", start, step, maximum,
                                         threshold=args_cli.curriculum_threshold, windows=args_cli.curriculum_windows,
                                         min_iterations_between=args_cli.curriculum_cooldown, unit=180.0 / math.pi)
            curriculum.attach(env)
            print(f"[FastSAC] yaw curriculum: start +-{args_cli.yaw_curriculum[0]:g} deg, +{args_cli.yaw_curriculum[1]:g} deg "
                  f"whenever placed >= {args_cli.curriculum_threshold} for {args_cli.curriculum_windows} windows, "
                  f"max +-{args_cli.yaw_curriculum[2]:g} deg")
        runner.learn(agent_cfg.max_iterations, behavior=behavior, validator=validator, gate=gate,
                     eval_interval=args_cli.eval_interval, reference_actions=reference_actions,
                     reference_std=reference_std, probe=probe, probe_interval=args_cli.probe_interval,
                     reanchor_interval=args_cli.reanchor_interval, curriculum=curriculum)
        env.close()


class _TeacherBehavior:
    """The frozen teacher as a ``BehaviorPolicy``: it reads the critic's ``policy`` observations."""

    def __init__(self, teacher: FrozenTeacher):
        self.teacher = teacher

    def act(self, actor_obs, critic_obs):
        return self.teacher.act(critic_obs)

    def next_actions(self, actor_obs, critic_obs):
        return self.teacher.act(critic_obs)


if __name__ == "__main__":
    main()
