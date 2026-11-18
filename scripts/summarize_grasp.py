"""Compare saved grasp trajectories without starting the simulator."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    metadata = json.loads((path / 'summary.json').read_text())
    with np.load(path / 'traces.npz') as traces:
        data = dict(traces)
    mask = data['valid']
    same = mask[1:] & mask[:-1] & (data['episode'][1:] == data['episode'][:-1])
    jerk = np.diff(data['qdd'], axis=0) / metadata['dt']
    first_hold = data['holding'] & ~np.concatenate((np.zeros((1, mask.shape[1]), dtype=bool), data['holding'][:-1])) & mask
    result = {
        'checkpoint': metadata['checkpoint'], 'seed': metadata['seed'],
        'episodes': len(metadata['episodes']), 'success_rate': metadata['success_rate'],
        'episode_seconds_mean': np.mean([e['steps'] for e in metadata['episodes']]) * metadata['dt'],
        'joint_velocity_p95': np.quantile(abs(data['qd'][mask]), .95),
        'joint_acceleration_p95': np.quantile(abs(data['qdd'][mask]), .95),
        'joint_acceleration_max': abs(data['qdd'][mask]).max(),
        'joint_jerk_p95': np.quantile(abs(jerk[same]), .95),
        'first_hold_linear_speed_median': np.median(np.linalg.norm(data['hand_velocity'][first_hold, :3], axis=-1)) if first_hold.any() else None,
        'first_hold_angular_speed_median': np.median(np.linalg.norm(data['hand_velocity'][first_hold, 3:], axis=-1)) if first_hold.any() else None,
    }
    result = {k: v.item() if isinstance(v, np.generic) else v for k, v in result.items()}
    return result, metadata, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {path.name: summarize(path) for path in args.recordings}
    (args.out / 'comparison.json').write_text(json.dumps({k: v[0] for k,v in results.items()}, indent=2))
    lines = ['| Recording | Success | Seconds | Joint speed p95 (rad/s) | Acceleration p95 (rad/s²) | Jerk p95 (rad/s³) | Hand speed at first hold (m/s) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, (row, _, _) in results.items():
        speed = row['first_hold_linear_speed_median']
        lines.append(f"| {name} | {row['success_rate']:.1%} | {row['episode_seconds_mean']:.2f} | {row['joint_velocity_p95']:.3f} | {row['joint_acceleration_p95']:.2f} | {row['joint_jerk_p95']:.1f} | {speed:.3f} |" if speed is not None else f'| {name} | No grasp |')
    lines += ['', 'Percentiles pool absolute measurements over valid steps and seven arm joints. Jerk is the control-step difference of recorded acceleration; episode reset boundaries are excluded. First-hold speed is the median over observed hold onsets. These are simulator metrics, not physical hardware limits.',
              '', 'Rewards changed between recordings, so raw episode returns are not comparable. All sheet poses are fixed; target poses and robot resets follow the recorded seeds. The plot uses environment 0, episode 0 as an example, not an aggregate trajectory.']
    (args.out / 'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for name, (row, metadata, data) in results.items():
        valid = data['valid'][:, 0] & (data['episode'][:, 0] == 0)
        time = np.arange(valid.sum()) * metadata['dt']
        for ax, key, label in zip(axes[0], ['dist_to_edge', 'width', 'lowest_z'], ['Distance to edge (m)', 'Finger position (m)', 'Lowest sheet node (m)']):
            ax.plot(time, data[key][valid, 0], label=name)
            ax.set(xlabel='Time (s)', ylabel=label)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax, key, label in zip(axes[1], ['joint_velocity_p95','joint_acceleration_p95','joint_jerk_p95'], ['Joint speed p95 (rad/s)','Joint acceleration p95 (rad/s²)','Joint jerk p95 (rad/s³)']):
        ax.bar(list(results), [result[0][key] for result in results.values()])
        ax.set(ylabel=label)
        ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    fig.suptitle('Grasp/extraction trajectories and measured motion')
    fig.savefig(args.out / 'comparison.png', dpi=160)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
