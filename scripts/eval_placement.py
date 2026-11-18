"""Evaluate placement checkpoints on the current reward: one deterministic episode per environment.

The simulator starts once and each checkpoint is loaded in turn, so several can be compared
cheaply. The smoothing curriculum is set to full strength, as it is late in training. Hydra
overrides pass through. After extraction, motion costs have a separate fixed scale and cap.

Per checkpoint it reports extraction, actual releases (any height), releases above the full-credit height, successes,
best platform coverage, how close the sheet's centre got (0-1, from its extraction distance),
how episodes ended, the mean episode return and each reward term and placement part per episode.
Writes ``summary.json`` and ``report.md`` into ``--out``.

Example::

    python scripts/eval_placement.py --viz none --num_envs 32 --out evaluations/placement_eval \\
        --checkpoints logs/rsl_rl/sheet_platform_grasp/<run>/model_1000.pt <other>.pt
"""

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import sheet_rl.tasks  # noqa: F401
from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _drape_success, _high_release, _phase_reached

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Template-Sheet-Rl-Platform-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--labels", nargs="*", default=None, help="Optional names, one per checkpoint.")
parser.add_argument("--out", default="evaluations/placement_eval")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining


@hydra_task_config(args.task, args.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    checkpoints = [Path(c).expanduser().resolve() for c in args.checkpoints]
    labels = args.labels or [f"{c.parent.name}/{c.stem}" for c in checkpoints]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
                                 clip_actions=agent_cfg.clip_actions)
        inner = env.unwrapped
        n, dev = inner.num_envs, inner.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        manager = inner.reward_manager
        term_names = list(manager.active_terms)
        place = manager.get_term_cfg("placement").func if "placement" in term_names else None
        motion = manager.get_term_cfg("motion_cost").func if "motion_cost" in term_names else None
        endings = list(inner.termination_manager.active_terms)
        original_compute = manager.compute
        state = {}

        def capture(dt):
            reward = original_compute(dt)
            live = ~state["done"]
            state["terms"] += manager._step_reward * dt * live.unsqueeze(-1)
            state["steps"] += live.float()
            if place is not None:
                for name, value in place.components.items():
                    state["parts"].setdefault(name, torch.zeros(n, device=dev))
                    state["parts"][name] += value * live
                state["released"] |= place.released & live
                state["high"] = torch.where(live, torch.maximum(state["high"], place.high_count), state["high"])
                state["coverage"] = torch.where(live, torch.maximum(state["coverage"], place.best_coverage), state["coverage"])
                start = place.center_start
                progress = torch.where(start > 1e-3, 1 - place.center_best / start.clamp_min(1e-3), torch.zeros_like(start))
                state["center"] = torch.where(live, torch.maximum(state["center"], progress), state["center"])
            state["extracted"] |= _phase_reached(inner) & live
            state["aborted"] |= _high_release(inner) & live
            state["success"] |= _drape_success(inner) & live
            return reward

        manager.compute = capture
        for label, checkpoint in zip(labels, checkpoints):
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=dev)
            if motion is not None:
                motion.success_rate = 1.0  # smoothing curriculum at full strength
            zeros = lambda: torch.zeros(n, device=dev)
            flags = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
            state.clear()
            state.update(done=flags(), terms=torch.zeros(n, len(term_names), device=dev), steps=zeros(), parts={},
                         released=flags(), high=zeros(), coverage=zeros(), center=zeros(), extracted=flags(),
                         aborted=flags(), success=flags(), ended=[None] * n)
            print(f"[eval] {label}", flush=True)
            with torch.inference_mode():
                # Inside inference mode: the rollout's tensors are inference tensors.
                obs, _ = env.reset()
                for _ in range(int(inner.max_episode_length) + 2):
                    obs, _, dones, _ = env.step(policy(obs))
                    dones = dones.bool()
                    new = dones & ~state["done"]
                    for e in torch.nonzero(new).flatten().tolist():
                        state["ended"][e] = [t for t in endings if bool(inner.termination_manager.get_term(t)[e])]
                    state["done"] |= dones
                    if bool(state["done"].all()):
                        break
            ended = {}
            for item in state["ended"]:
                key = ",".join(item or ["not finished"])
                ended[key] = ended.get(key, 0) + 1
            terms = state["terms"].mean(0).tolist()
            results[label] = {
                "checkpoint": str(checkpoint), "episodes": n,
                "extracted": state["extracted"].float().mean().item(),
                "released": state["released"].float().mean().item(),
                "high_releases_per_episode": state["high"].mean().item(),
                "aborted": state["aborted"].float().mean().item(),
                "success": state["success"].float().mean().item(),
                "best_coverage": state["coverage"].mean().item(),
                "center_progress": state["center"].mean().item(),
                "episode_seconds": (state["steps"].mean() * inner.step_dt).item(),
                "return": float(sum(terms)),
                "terms": dict(zip(term_names, terms)),
                "placement_parts": {k: v.mean().item() for k, v in state["parts"].items()},
                "endings": dict(sorted(ended.items(), key=lambda kv: -kv[1])),
            }
            r = results[label]
            print(f"   extracted {r['extracted']:.0%}  released {r['released']:.0%}  success {r['success']:.0%}  "
                  f"coverage {r['best_coverage']:.2f}  return {r['return']:,.0f}  endings {r['endings']}", flush=True)
        manager.compute = original_compute
        env.close()

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    lines = ["| Checkpoint | Extracted | Released | High releases / ep | Aborted | Success | Best coverage | "
             "Centre progress | Seconds | Return |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in results.items():
        lines.append(f"| {label} | {r['extracted']:.0%} | {r['released']:.0%} | {r['high_releases_per_episode']:.2f} | "
                     f"{r['aborted']:.0%} | {r['success']:.0%} | {r['best_coverage']:.2f} | {r['center_progress']:.2f} | "
                     f"{r['episode_seconds']:.1f} | {r['return']:,.0f} |")
    lines += ["", "Endings:", ""]
    for label, r in results.items():
        lines.append(f"- **{label}**: " + ", ".join(f"{k} {v}" for k, v in r["endings"].items()))
    lines += ["", "Mean reward per episode by term:", ""]
    names = sorted({k for r in results.values() for k in r["terms"]})
    lines.append("| Checkpoint | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for label, r in results.items():
        lines.append(f"| {label} | " + " | ".join(f"{r['terms'].get(k, 0):,.0f}" for k in names) + " |")
    parts = sorted({k for r in results.values() for k in r["placement_parts"]})
    if parts:
        lines += ["", "Placement parts per episode:", "", "| Checkpoint | " + " | ".join(parts) + " |",
                  "|---|" + "---:|" * len(parts)]
        for label, r in results.items():
            lines.append(f"| {label} | " + " | ".join(f"{r['placement_parts'].get(k, 0):,.0f}" for k in parts) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
