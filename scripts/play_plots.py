"""Playback that plots each finished episode: motion and phase-2/3 rewards.

Used by ``scripts/play.py --plot``. Runs the deterministic policy like ``play.py`` (viewer and
Hydra overrides included), records environment 0 every control step, and when its episode ends
opens two figures. Close both windows to start the next episode; Ctrl-C quits. The figures are
also saved as PNGs under ``evaluations/play_plots/<run>__<checkpoint>/``.

* Motion: hand speed against the smoothing envelope, hand angular speed, the seven arm joint
  velocities and accelerations, and each ``motion_cost`` part as the reward it cost per step
  (motion_cost weight and actual curriculum scaling included), plus its running total. Pass the
  same ``env.rewards.motion_cost.weight=...`` override used in training to match its scale.
* Phase-2 rewards: each ``placement`` part per step and cumulatively (approach, lowering, coverage progress,
  opening progress, release, released coverage and success),
  platform coverage, fingertip height against the full-credit release height, and the gripper filter
  state with finger width.

Phase 2 (after extraction) is shaded; the release step is marked.
"""

import argparse
import importlib.metadata as metadata
import sys
import time
from pathlib import Path

import numpy as np
import torch

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e"]
INK, MUTED, GRID, SURFACE, CRITICAL, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d03b3b", "#eeeeea"


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _mark_phases(ax, t, phase, release_step):
    if phase.any():
        ax.axvspan(t[np.argmax(phase)], t[-1], color=SHADE, zorder=0)
    if release_step is not None:
        ax.axvline(t[release_step], color=MUTED, linewidth=1.2, linestyle=":")


def _lines(ax, t, series, labels):
    for i, (values, label) in enumerate(zip(series, labels)):
        ax.plot(t, values, color=COLORS[i % len(COLORS)], linewidth=1.5, label=label)


def plot_episode(data, title, out_dir, episode, plt):
    t = np.arange(len(data["phase"])) * data["dt"]
    phase = np.array(data["phase"], dtype=bool)
    released = np.array(data["released"], dtype=bool)
    release_step = int(np.argmax(released)) if released.any() else None
    figures = []

    # ---- Motion
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, constrained_layout=True, facecolor=SURFACE)
    fig.canvas.manager.set_window_title(f"Motion - episode {episode}")
    ax = axes[0, 0]
    _lines(ax, t, [data["hand_speed"], data["speed_limit"]], ["hand speed", "allowed (envelope)"])
    ax.get_lines()[1].set_linestyle("--")
    _style(ax, "hand speed [m/s]"); ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    _lines(ax, t, [data["hand_angular"]], ["hand angular speed"]); ax.axhline(1.0, color=CRITICAL, linestyle="--", linewidth=1.2)
    _style(ax, "hand angular speed [rad/s]")
    q_names = [f"J{j + 1}" for j in range(7)]
    ax = axes[1, 0]
    _lines(ax, t, np.array(data["qd"]).T, q_names); _style(ax, "joint velocity [rad/s]")
    ax.legend(ncol=7, fontsize=7, frameon=False)
    ax = axes[1, 1]
    _lines(ax, t, np.array(data["qdd"]).T, q_names); _style(ax, "joint acceleration [rad/s²]")
    motion = {k: np.array(v) for k, v in data["motion"].items()}
    active = [k for k, v in motion.items() if np.abs(v).max() > 1e-6]
    ax = axes[2, 0]
    _lines(ax, t, [motion[k] for k in active], active); _style(ax, "motion cost per step (reward)")
    ax.legend(ncol=4, fontsize=7, frameon=False)
    ax = axes[2, 1]
    _lines(ax, t, [np.cumsum(motion[k]) for k in active], active); _style(ax, "motion cost, running total")
    for ax in axes.flat:
        _mark_phases(ax, t, phase, release_step)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]  (shaded: phase 2, dotted: release)", color=INK, fontsize=9)
    total_motion = sum(v.sum() for v in motion.values())
    fig.suptitle(f"{title}\nmotion cost total {total_motion:,.0f}  (weight {data['motion_weight']:g}, "
                 f"actual applied costs)", color=INK, fontsize=11)
    fig.savefig(out_dir / f"ep{episode}_motion.png", dpi=110, facecolor=SURFACE)
    figures.append(fig)

    # ---- Phase-2 rewards
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, constrained_layout=True, facecolor=SURFACE)
    fig.canvas.manager.set_window_title(f"Placement and retreat rewards - episode {episode}")
    place = {k: np.array(v) for k, v in data["placement"].items()}
    active = [k for k, v in place.items() if np.abs(v).max() > 1e-6]
    ax = axes[0, 0]
    _lines(ax, t, [place[k] for k in active], active); _style(ax, "placement reward per step")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    ax = axes[0, 1]
    _lines(ax, t, [np.cumsum(place[k]) for k in active], active); _style(ax, "placement reward, running total")
    ax = axes[1, 0]
    _lines(ax, t, [data["coverage"]], ["coverage"]); ax.axhline(data["success_coverage"], color=CRITICAL, linestyle="--", linewidth=1.2)
    _style(ax, "platform coverage (dashed: success)"); ax.set_ylim(-0.02, 1.02)
    ax = axes[1, 1]
    _lines(ax, t, [data["elevation"]], ["fingertip height"])
    ax.axhline(data["release_height"], color=CRITICAL, linestyle="--", linewidth=1.2)
    ax.axhline(data["max_release_height"], color=CRITICAL, linestyle=":", linewidth=1.2)
    ax.axhline(data["retreat_height"], color=COLORS[0], linestyle="--", linewidth=1.2)
    _style(ax, "fingertip height above table [m]\n(red: release credit/ceiling; blue: retreat target)")
    ax = axes[2, 0]
    _lines(ax, t, [data["gripper_state"]], ["gripper filter state"]); ax.axhline(0.0, color=MUTED, linewidth=1.0)
    _style(ax, "gripper command (> 0 opens)")
    ax = axes[2, 1]
    _lines(ax, t, [np.array(data["width"]) * 1000], ["finger width"]); ax.axhline(30, color=MUTED, linestyle="--", linewidth=1.0)
    _style(ax, "finger width [mm] (dashed: open)")
    for ax in axes.flat:
        _mark_phases(ax, t, phase, release_step)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]  (shaded: phase 2, dotted: release)", color=INK, fontsize=9)
    total_place = sum(v.sum() for v in place.values())
    fig.suptitle(f"{title}\nplacement total {total_place:,.0f}", color=INK, fontsize=11)
    fig.savefig(out_dir / f"ep{episode}_phase2_rewards.png", dpi=110, facecolor=SURFACE)
    figures.append(fig)
    return figures


def main(argv: list[str]) -> int:
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="Template-Sheet-Rl-Platform-v0")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--rl_library", default="rsl_rl", help="Ignored; RSL-RL only.")
    parser.add_argument("--real-time", dest="real_time", action="store_true")
    parser.add_argument("--episodes", type=int, default=0, help="Stop after this many episodes (0: until Ctrl-C).")
    parser.add_argument("--no_show", action="store_true", help="Only save the PNGs; do not open windows.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    add_frontend_args(parser)
    sys.argv = [sys.argv[0]] + list(argv)
    args, remaining = setup_preset_cli(parser, agent_library="rsl_rl")
    sys.argv = [sys.argv[0]] + remaining
    if not args.checkpoint:
        raise ValueError("--plot needs --checkpoint <path to model_*.pt>")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    import sheet_rl.tasks  # noqa: F401
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.grasp_progress import speed_limit
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _is_holding, _phase_reached, grasp_debug

    def run(env_cfg, agent_cfg):
        from rsl_rl.runners import OnPolicyRunner
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args),
                                                 metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs
        out_dir = Path("evaluations/play_plots") / f"{checkpoint.parent.name}__{checkpoint.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with launch_simulation(env_cfg, args):
            env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
                                     clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            inner = env.unwrapped
            robot = inner.scene["robot"]
            hand = robot.body_names.index("panda_hand")
            arm = [i for i, n in enumerate(robot.joint_names) if n.startswith("panda_joint")]
            fingers = [i for i, n in enumerate(robot.joint_names) if n.startswith("panda_finger")]
            manager = inner.reward_manager
            motion_cfg = manager.get_term_cfg("motion_cost") if "motion_cost" in manager.active_terms else None
            place_cfg = manager.get_term_cfg("placement") if "placement" in manager.active_terms else None
            gripper = inner.action_manager.get_term("gripper_action")
            success_coverage = place_cfg.params.get("coverage_threshold", 0.5) if place_cfg else 0.5

            def fresh():
                return {"dt": inner.step_dt, "phase": [], "released": [], "hand_speed": [], "speed_limit": [],
                        "hand_angular": [], "qd": [], "qdd": [], "motion": {}, "placement": {}, "coverage": [],
                        "elevation": [], "gripper_state": [], "width": [], "success_coverage": success_coverage,
                        "motion_weight": motion_cfg.weight if motion_cfg is not None else 0.0,
                        "release_height": place_cfg.params.get("release_height", 0.03) if place_cfg else 0.03,
                        "max_release_height": (inner.cfg.terminations.released_high.params.get("max_height", 0.10)
                                               if inner.cfg.terminations.released_high else float("nan")),
                        "retreat_height": (2 * place_cfg.params.get("top_offset", 0.01)
                                           + place_cfg.params.get("retreat_height", 0.20)) if place_cfg else float("nan")}

            data = fresh()
            original_compute = manager.compute

            # Record env 0 at reward time: after physics, before an ending resets it.
            def capture(dt):
                reward = original_compute(dt)
                debug = grasp_debug(inner)
                holding, phase = _is_holding(inner), _phase_reached(inner)
                velocity = robot.data.body_link_vel_w.torch[0, hand]
                data["phase"].append(bool(phase[0]))
                data["hand_speed"].append(float(velocity[:3].norm()))
                data["hand_angular"].append(float(velocity[3:].norm()))
                limit = speed_limit(debug["top_dist"][:1], (holding | phase)[:1])
                if motion_cfg is not None and phase[0] and motion_cfg.params.get("phase_two_scale") is not None:
                    limit.fill_(motion_cfg.params.get("phase_two_speed_limit", 0.15))
                data["speed_limit"].append(float(limit[0]))
                data["qd"].append(robot.data.joint_vel.torch[0, arm].tolist())
                data["qdd"].append(robot.data.joint_acc.torch[0, arm].tolist())
                data["width"].append(float(robot.data.joint_pos.torch[0, fingers].mean()))
                data["gripper_state"].append(float(gripper.raw_actions[0, 0]))
                if motion_cfg is not None:
                    term = motion_cfg.func
                    for name, value in term.applied_components.items():
                        data["motion"].setdefault(name, []).append(float(value[0]) * motion_cfg.weight)
                if place_cfg is not None:
                    term = place_cfg.func
                    for name, value in term.components.items():
                        data["placement"].setdefault(name, []).append(float(value[0]) * place_cfg.weight)
                    data["coverage"].append(float(term.coverage[0]))
                    data["elevation"].append(float(term.elevation[0]))
                    data["released"].append(bool(term.released[0]))
                else:
                    data["coverage"].append(0.0); data["elevation"].append(0.0); data["released"].append(False)
                return reward

            manager.compute = capture
            names = list(inner.termination_manager.active_terms)
            episode = 0
            obs = env.get_observations()
            print("[play --plot] close both plot windows to start the next episode; Ctrl-C to quit.", flush=True)
            try:
                while True:
                    start = time.time()
                    with torch.inference_mode():
                        obs, _, dones, _ = env.step(policy(obs))
                        policy.reset(dones)
                    if bool(dones[0]):
                        ended = [n for n in names if bool(inner.termination_manager.get_term(n)[0])]
                        title = (f"{checkpoint.parent.name}/{checkpoint.name}  episode {episode}  "
                                 f"({len(data['phase']) * inner.step_dt:.1f} s, ended: {', '.join(ended)})")
                        print(f"[play --plot] episode {episode} ended by {ended}; showing plots", flush=True)
                        plot_episode(data, title, out_dir, episode, plt)
                        if not args.no_show:
                            plt.show()  # blocks until every window is closed
                        plt.close("all")
                        episode += 1
                        data.clear(); data.update(fresh())
                        if args.episodes and episode >= args.episodes:
                            break
                    if args.real_time:
                        time.sleep(max(0.0, inner.step_dt - (time.time() - start)))
            except KeyboardInterrupt:
                pass
            print(f"[play --plot] plots saved in {out_dir.resolve()}", flush=True)
            manager.compute = original_compute
            env.close()

    hydra_task_config(args.task, args.agent, play_mode=True)(run)()
    return 0
