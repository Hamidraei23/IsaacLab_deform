"""Record deterministic grasp rollouts, including terminal states before automatic resets."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
import sheet_rl.tasks  # noqa: F401
from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import grasp_debug, _is_holding, _phase_reached
from sheet_rl.tasks.manager_based.sheet_rl.mdp.grasp_progress import MOTION_NAMES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Template-Sheet-Rl-Platform-Grasp-v0')
parser.add_argument('--agent', default='rsl_rl_cfg_entry_point')
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--episodes', type=int, default=2)
parser.add_argument('--out', required=True)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library='rsl_rl')
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args.task, args.agent, play_mode=False)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(out / "env.yaml"), env_cfg)
    dump_yaml(str(out / "agent.yaml"), agent_cfg)
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        inner = env.unwrapped
        robot = inner.scene['robot']
        hand = robot.body_names.index('panda_hand')
        joints = [i for i, name in enumerate(robot.joint_names) if name.startswith('panda_joint')]
        print(f'ARM CONTROL: {type(inner.action_manager.get_term("arm_action")).__name__}', flush=True)
        reward_names = list(inner.reward_manager.active_terms)
        records = []
        finished = torch.zeros(inner.num_envs, device=inner.device, dtype=torch.long)
        # Capture at reward computation, after physics and before _reset_idx replaces terminal states.
        original_compute = inner.reward_manager.compute
        def capture(dt):
            reward = original_compute(dt)
            debug = grasp_debug(inner)
            def field(name):
                return debug.get(name, torch.zeros(inner.num_envs, device=inner.device)).float()
            record = {
                'episode': finished.clone(), 'valid': finished < args.episodes,
                'q': robot.data.joint_pos.torch[:, joints].clone(),
                'qd': robot.data.joint_vel.torch[:, joints].clone(),
                'qdd': robot.data.joint_acc.torch[:, joints].clone(),
                'hand_pose': robot.data.body_link_pose_w.torch[:, hand].clone(),
                'hand_velocity': robot.data.body_link_vel_w.torch[:, hand].clone(),
                'actions': inner.action_manager.action.clone(),
                'applied_actions': inner.action_manager.get_term('arm_action').raw_actions.clone(),
                'reward_terms': inner.reward_manager._step_reward.clone() * dt,
                'holding': _is_holding(inner).clone(), 'extracted': _phase_reached(inner).clone(),
                'dones': inner.termination_manager.dones.clone(),
                'termination': torch.stack([inner.termination_manager.get_term(n) for n in inner.termination_manager.active_terms], -1),
            }
            for name in ('dist_to_edge', 'sheet_dist', 'top_dist', 'width', 'lowest_z', 'facing_down', 'square_to_slot', 'hit', 'extracted', 'bad_closure'):
                if name not in record:
                    record[name] = field(name).clone()
            if 'motion_cost' in inner.reward_manager.active_terms:
                term = inner.reward_manager.get_term_cfg('motion_cost').func
                record['motion_components'] = torch.stack(list(term.components.values()), -1).clone()
            records.append({k: v.cpu().numpy() for k, v in record.items()})
            return reward
        inner.reward_manager.compute = capture
        obs = env.get_observations()
        with torch.inference_mode():
            for step in range((int(inner.max_episode_length) + 1) * args.episodes):
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                finished += dones.long()
                if step % 100 == 0:
                    print(f'rollout step={step} complete={int(finished.clamp(max=args.episodes).sum())}', flush=True)
                if bool((finished >= args.episodes).all()):
                    break
        data = {k: np.stack([r[k] for r in records]) for k in records[0]}
        np.savez_compressed(out / 'traces.npz', **data)
        mask = data['valid']
        episodes = []
        for i in range(inner.num_envs):
            for ep in range(args.episodes):
                keep = mask[:, i] & (data['episode'][:, i] == ep)
                if not keep.any():
                    continue
                episodes.append({'env': i, 'episode': ep, 'success': bool(data['extracted'][keep, i].any()),
                                 'steps': int(keep.sum()), 'return': float(data['reward_terms'][keep, i].sum())})
        same_episode = mask[1:] & mask[:-1] & (data['episode'][1:] == data['episode'][:-1])
        jerk = np.diff(data['qdd'], axis=0) / inner.step_dt
        first_hold = data['holding'] & ~np.concatenate((np.zeros((1, inner.num_envs), dtype=bool), data['holding'][:-1])) & mask
        metrics = {
            'checkpoint': str(Path(args.checkpoint).resolve()), 'seed': args.seed, 'dt': inner.step_dt,
            'episodes': episodes, 'success_rate': float(np.mean([e['success'] for e in episodes])),
            'joint_velocity_p95': float(np.quantile(np.abs(data['qd'][mask]), .95)),
            'joint_acceleration_p95': float(np.quantile(np.abs(data['qdd'][mask]), .95)),
            'joint_acceleration_max': float(np.abs(data['qdd'][mask]).max()),
            'joint_jerk_p95': float(np.quantile(np.abs(jerk[same_episode]), .95)),
            'first_hold_linear_speed_median': float(np.median(np.linalg.norm(data['hand_velocity'][first_hold, :3], axis=-1))) if first_hold.any() else None,
            'mean_episode_seconds': float(np.mean([e['steps'] for e in episodes]) * inner.step_dt),
            'completed_episodes': int(finished.clamp(max=args.episodes).sum()),
            'motion_component_names': list(MOTION_NAMES),
            'reward_names': reward_names, 'termination_names': list(inner.termination_manager.active_terms),
            'reward_mean_per_episode': dict(zip(reward_names, (data['reward_terms'] * mask[..., None]).sum((0, 1)).tolist())),
        }
        metrics['reward_mean_per_episode'] = {k: v / len(episodes) for k, v in metrics['reward_mean_per_episode'].items()}
        (out / 'summary.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in metrics.items() if k != 'episodes'}, indent=2), flush=True)
        env.close()


if __name__ == '__main__':
    main()
