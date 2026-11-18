"""Play a checkpoint, record arm joint trajectories at the physics rate, and check joint limits.

Like ``scripts/play.py`` (deterministic policy, optional viewer), but every physics substep
(120 Hz here) the seven arm joints' position, velocity and acceleration are recorded, jerk is
computed as the difference of acceleration, and every quantity is checked against the Franka
Emika Panda joint limits (FCI datasheet; see ``FRANKA_LIMITS``).

Example (viewer on)::

    PYOPENGL_PLATFORM=glx python scripts/check_joint_limits.py \\
        --checkpoint $PWD/logs/rsl_rl/sheet_platform_grasp/<run>/model_360.pt \\
        --viz newton_gl --num_envs 1 --episodes 3

Headless: ``--viz none``. Writes, into ``--out`` (default beside ``evaluations/joint_limits``):

* ``traces.npz``: per-sample env, episode, time, q, qd, qdd, jerk (jerk NaN where undefined).
* ``summary.json`` and ``report.md``: per episode and per joint, max and p99 of each quantity,
  its fraction of the limit, and how many samples exceeded it; an overall PASS/FAIL.
* ``env<e>_ep<k>.png``: the four quantities over time (left) and as a fraction of the limit
  (right, the red line is the limit). ``limit_summary.png``: worst fraction per joint.

Notes on measurement. Acceleration is the simulator's finite difference of joint velocity at the
physics step; jerk differences it once more, so single-sample contact spikes show up strongly.
Both max (strict) and p99 (ignores the worst 1% of samples) are reported. The first samples after
each reset are skipped for acceleration and jerk (``--skip_after_reset``), because the reset
teleports the arm. The limits are the manufacturer's; ``--margin 0.8`` checks against 80% of them.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
import sheet_rl.tasks  # noqa: F401

# Franka Emika Panda joint limits, from the FCI "Robot and interface specifications".
FRANKA_LIMITS = {
    'position_min': [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],   # rad
    'position_max': [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],         # rad
    'velocity': [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],              # rad/s
    'acceleration': [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],                         # rad/s^2
    'jerk': [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],                # rad/s^3
}
QUANTITIES = ('position', 'velocity', 'acceleration', 'jerk')
UNITS = {'position': 'rad', 'velocity': 'rad/s', 'acceleration': 'rad/s²', 'jerk': 'rad/s³'}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--task', default='Template-Sheet-Rl-Platform-Grasp-v0')
parser.add_argument('--agent', default='rsl_rl_cfg_entry_point')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--episodes', type=int, default=3, help='Episodes to record per environment.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', help='Output directory (default: evaluations/joint_limits/<run>__<model>).')
parser.add_argument('--margin', type=float, default=1.0, help='Check against this fraction of each limit.')
parser.add_argument('--skip_after_reset', type=int, default=3,
                    help='Physics samples after each reset excluded from acceleration and jerk.')
parser.add_argument('--max_plots', type=int, default=6, help='Maximum number of per-episode figures.')
parser.add_argument('--real-time', dest='real_time', action='store_true', help='Play at wall-clock speed.')
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library='rsl_rl')
sys.argv = [sys.argv[0]] + remaining


def limit_fraction(quantity, values, limits):
    """|value| / limit per joint; for position, distance from the range centre / half range."""
    if quantity == 'position':
        low, high = np.array(limits['position_min']), np.array(limits['position_max'])
        return np.abs(values - (low + high) / 2) / ((high - low) / 2)
    return np.abs(values) / np.array(limits[quantity])


def episode_metrics(series, limits, margin, joint_names):
    result = {}
    for quantity in QUANTITIES:
        values = series[quantity]
        values = values[~np.isnan(values).any(-1)]
        if not len(values):
            continue
        fraction = limit_fraction(quantity, values, limits)
        over = fraction > margin
        result[quantity] = {
            'unit': UNITS[quantity], 'samples': int(len(values)),
            'max_abs': dict(zip(joint_names, np.abs(values).max(0).round(4).tolist())),
            'p99_abs': dict(zip(joint_names, np.quantile(np.abs(values), .99, axis=0).round(4).tolist())),
            'max_fraction_of_limit': dict(zip(joint_names, fraction.max(0).round(3).tolist())),
            'p99_fraction_of_limit': dict(zip(joint_names, np.quantile(fraction, .99, axis=0).round(3).tolist())),
            'samples_over_limit': dict(zip(joint_names, over.sum(0).astype(int).tolist())),
            'violated': bool(over.any()),
        }
    return result


def plot_episode(path, t, series, limits, margin, joint_names, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
    ink, muted, grid, surface, critical = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#d03b3b'
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, constrained_layout=True, facecolor=surface)
    for row, quantity in enumerate(QUANTITIES):
        values = series[quantity]
        fraction = limit_fraction(quantity, values, limits)
        for col, data, label in ((0, values, f'{quantity} [{UNITS[quantity]}]'),
                                 (1, fraction, f'{quantity}: fraction of limit')):
            ax = axes[row, col]
            ax.set_facecolor(surface)
            for j, name in enumerate(joint_names):
                ax.plot(t, data[:, j], color=colors[j], linewidth=1.6, label=name)
            if col == 1:
                ax.axhline(margin, color=critical, linewidth=1.6, linestyle='--')
                ax.text(t[0], margin, f' limit ({margin:g}×)', color=ink, va='bottom', fontsize=9)
                ax.set_ylim(0, max(1.15 * margin, np.nanmax(fraction) * 1.08 if np.isfinite(fraction).any() else 1))
            ax.set_ylabel(label, color=ink, fontsize=10)
            ax.grid(color=grid, linewidth=0.8)
            ax.tick_params(colors=muted, labelsize=9)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(grid)
    for ax in axes[-1]:
        ax.set_xlabel('time [s]', color=ink)
    fig.suptitle(title, color=ink, fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, fontsize=9, frameon=False, loc='outside lower center', labelcolor=ink)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def plot_summary(path, worst, margin, joint_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
    ink, muted, grid, surface, critical = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#d03b3b'
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True, facecolor=surface)
    for ax, quantity in zip(axes, QUANTITIES):
        ax.set_facecolor(surface)
        values = [worst[quantity][name] for name in joint_names]
        ax.bar(range(len(joint_names)), values, color=colors, width=0.7)
        ax.axhline(margin, color=critical, linewidth=1.6, linestyle='--')
        ax.set_axisbelow(True)
        for x, value in enumerate(values):
            ax.text(x, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8, color=muted)
        ax.set_xticks(range(len(joint_names)), [n.replace('panda_joint', 'J') for n in joint_names])
        ax.set_title(f'{quantity}: worst fraction of limit\n(red dashed line = {margin:g}× limit)', color=ink, fontsize=10)
        ax.set_ylim(0, max(1.3 * margin, max(values) * 1.15))
        ax.grid(axis='y', color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


@hydra_task_config(args.task, args.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version('rsl-rl-lib'))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    out = Path(args.out) if args.out else Path('evaluations/joint_limits') / f'{checkpoint.parent.name}__{checkpoint.stem}'
    out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
                                 clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        inner = env.unwrapped
        robot = inner.scene['robot']
        joints = [i for i, name in enumerate(robot.joint_names) if name.startswith('panda_joint')]
        joint_names = [robot.joint_names[i] for i in joints]
        termination_names = list(inner.termination_manager.active_terms)

        episode = torch.zeros(inner.num_envs, dtype=torch.long, device=inner.device)
        samples = {'q': [], 'qd': [], 'qdd': [], 'episode': [], 'dt': []}
        # Record after every physics substep: scene.update runs once per substep with the physics dt
        # (or once per control step if the physics backend folds the decimation loop).
        original_update = inner.scene.update

        def recording_update(dt):
            original_update(dt)
            samples['q'].append(robot.data.joint_pos.torch[:, joints].clone())
            samples['qd'].append(robot.data.joint_vel.torch[:, joints].clone())
            samples['qdd'].append(robot.data.joint_acc.torch[:, joints].clone())
            samples['episode'].append(episode.clone())
            samples['dt'].append(float(dt))

        inner.scene.update = recording_update
        outcomes = {}
        obs = env.get_observations()
        print(f'Recording {args.episodes} episode(s) x {inner.num_envs} env(s) from {checkpoint}', flush=True)
        with torch.inference_mode():
            for _ in range((int(inner.max_episode_length) + 1) * args.episodes):
                start = time.time()
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                for e in torch.nonzero(dones).flatten().tolist():
                    ended = [n for n in termination_names if bool(inner.termination_manager.get_term(n)[e])]
                    outcomes[(e, int(episode[e]))] = ended
                episode += dones.long()
                if bool((episode >= args.episodes).all()):
                    break
                if args.real_time:
                    time.sleep(max(0.0, inner.step_dt - (time.time() - start)))
        inner.scene.update = original_update
        env.close()

    q = torch.stack(samples['q']).cpu().numpy()          # [T, N, 7]
    qd = torch.stack(samples['qd']).cpu().numpy()
    qdd = torch.stack(samples['qdd']).cpu().numpy()
    ep = torch.stack(samples['episode']).cpu().numpy()   # [T, N]
    dt = np.array(samples['dt'])
    limits = FRANKA_LIMITS
    report, worst = {'episodes': []}, {k: {n: 0.0 for n in joint_names} for k in QUANTITIES}
    flat = {k: [] for k in ('env', 'episode', 't', 'q', 'qd', 'qdd', 'jerk')}
    plotted = 0
    for e in range(q.shape[1]):
        for k in range(args.episodes):
            idx = np.where(ep[:, e] == k)[0]
            if len(idx) < args.skip_after_reset + 2:
                continue
            t = np.cumsum(dt[idx]) - dt[idx][0]
            acc = qdd[idx, e].copy()
            acc[:args.skip_after_reset] = np.nan
            jerk = np.full_like(acc, np.nan)
            jerk[1:] = np.diff(acc, axis=0) / dt[idx][1:, None]
            series = {'position': q[idx, e], 'velocity': qd[idx, e], 'acceleration': acc, 'jerk': jerk}
            metrics = episode_metrics(series, limits, args.margin, joint_names)
            for quantity, m in metrics.items():
                for name, value in m['max_fraction_of_limit'].items():
                    worst[quantity][name] = max(worst[quantity][name], value)
            ended = outcomes.get((e, k), ['not finished'])
            report['episodes'].append({'env': e, 'episode': k, 'seconds': round(float(t[-1]), 3),
                                       'ended_by': ended, 'violated': any(m['violated'] for m in metrics.values()),
                                       'metrics': metrics})
            for key, value in (('env', np.full(len(idx), e)), ('episode', np.full(len(idx), k)), ('t', t),
                               ('q', q[idx, e]), ('qd', qd[idx, e]), ('qdd', acc), ('jerk', jerk)):
                flat[key].append(value)
            if plotted < args.max_plots:
                verdict = 'VIOLATES LIMITS' if report['episodes'][-1]['violated'] else 'within limits'
                plot_episode(out / f'env{e}_ep{k}.png', t, series, limits, args.margin, joint_names,
                             f'{checkpoint.parent.name}/{checkpoint.name}  env {e} episode {k}  '
                             f'({", ".join(ended)}; {verdict})')
                plotted += 1
    if not report['episodes']:
        raise RuntimeError('No complete episode was recorded.')
    np.savez_compressed(out / 'traces.npz', joint_names=np.array(joint_names),
                        **{k: np.concatenate(v) for k, v in flat.items()})
    violated_any = {qn: any(ep_['metrics'].get(qn, {}).get('violated', False) for ep_ in report['episodes'])
                    for qn in QUANTITIES}
    report.update(checkpoint=str(checkpoint), task=args.task, margin=args.margin, limits=limits,
                  sample_dt=float(np.median(dt)), joint_names=joint_names,
                  worst_fraction_of_limit=worst, violated=violated_any, passed=not any(violated_any.values()))
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    plot_summary(out / 'limit_summary.png', worst, args.margin, joint_names)

    short = [n.replace('panda_joint', 'J') for n in joint_names]
    lines = [f'# Joint limit check: {checkpoint.parent.name}/{checkpoint.name}', '',
             f'{len(report["episodes"])} episode(s), sampled every {np.median(dt) * 1000:.2f} ms, '
             f'limits x{args.margin:g} (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):', '',
             '| Quantity | ' + ' | '.join(short) + ' | Verdict |', '|---|' + '---:|' * len(short) + '---|']
    for quantity in QUANTITIES:
        cells = [f'{worst[quantity][n] / args.margin:.2f}' for n in joint_names]
        lines.append(f'| {quantity} | ' + ' | '.join(cells) + f' | {"VIOLATED" if violated_any[quantity] else "ok"} |')
    lines += ['', 'Per episode:', '', '| Env | Episode | Seconds | Ended by | Violations |', '|---:|---:|---:|---|---|']
    for item in report['episodes']:
        bad = [f'{qn} ({sum(m["samples_over_limit"].values())} samples)'
               for qn, m in item['metrics'].items() if m['violated']]
        lines.append(f'| {item["env"]} | {item["episode"]} | {item["seconds"]:.2f} | {", ".join(item["ended_by"])} | '
                     f'{"; ".join(bad) or "none"} |')
    lines += ['', f'**Overall: {"PASS" if report["passed"] else "FAIL"}**']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'\nSaved to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
