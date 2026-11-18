"""Train redesigned grasp rewards; optionally warm-start only the actor from a working grasp.

The critic and optimizer start fresh because the old reward scale was radically different.
Use train_platform.py --grasp --resume for subsequent full-state resumes of these new runs.
"""
import argparse
import importlib.metadata as metadata
import json
import math
import shutil
from pathlib import Path
import sys

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import dump_yaml
import sheet_rl.tasks  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Template-Sheet-Rl-Platform-Grasp-v0')
parser.add_argument('--agent', default='rsl_rl_cfg_entry_point')
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--warm_start', help='Load only actor weights and actor observation normalization.')
parser.add_argument('--warm_start_std', type=float, default=0.25)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--out', required=True)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library='rsl_rl')
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args.task, args.agent, play_mode=False)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    if args.resume or args.checkpoint:
        raise ValueError('Use --warm_start here; use train_platform.py --grasp --resume for full-state resume.')
    if args.warm_start_std <= 0 or args.iterations < 1:
        raise ValueError('Standard deviation and iteration count must be positive.')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Refusing to overwrite an existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version('rsl-rl-lib'))
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    if args.warm_start:
        agent_cfg.actor.distribution_cfg.init_std = args.warm_start_std
    dump_yaml(str(out / 'params' / 'env.yaml'), env_cfg)
    dump_yaml(str(out / 'params' / 'agent.yaml'), agent_cfg)
    source_root = Path(__file__).resolve().parents[1] / 'source/sheet_rl/sheet_rl/tasks/manager_based/sheet_rl'
    for relative in ('mdp/grasp_control.py', 'mdp/grasp_progress.py', 'sheet_rl_platform_env_cfg.py', 'agents/platform_ppo_cfg.py'):
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    (out / 'invocation.json').write_text(json.dumps(vars(args), indent=2, default=str))
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
                                clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(out), device=agent_cfg.device)
        if args.warm_start:
            runner.load(args.warm_start, load_cfg={'actor': True})
            # Loading the actor also loads its learned exploration scale. Reset it deliberately.
            actor = runner.alg.get_policy()
            with torch.no_grad():
                for name, parameter in actor.named_parameters():
                    if name.endswith('log_std_param'):
                        parameter.fill_(math.log(args.warm_start_std))
                    elif name.endswith('std_param'):
                        parameter.fill_(args.warm_start_std)
            print(f'Warm-started actor only: {args.warm_start}; fresh critic and optimizer.', flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(out / 'model_final.pt'))
        env.close()


if __name__ == '__main__':
    main()
