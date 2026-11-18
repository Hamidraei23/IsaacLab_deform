"""DAgger from the frozen teacher straight into a FastSAC actor (the student SAC will fine-tune).

The student is the tanh-squashed Gaussian ``Actor`` of ``sheet_rl.algorithms.fast_sac`` on the
``action_history`` group, so the checkpoint written here is a complete FastSAC checkpoint
(critic untrained) that ``scripts/train_sac.py --checkpoint`` can continue from.

Collection (user's rule: the student acts stochastically):

- Each environment is driven for whole episodes by either the teacher (sampled, clipped to the
  student's bounds) or the student (sampled). The assignment is redrawn at every reset with
  probability ``beta`` for the teacher; ``beta`` decays linearly from ``--beta_start`` to
  ``--beta_floor`` over ``--beta_iters`` iterations.
- Every visited state is labelled with the teacher's *mean* action, clipped to ``--label_margin``
  of the bounds (a label on the bound itself has no finite pre-tanh value).

Learning: the aggregate dataset is a GPU ring buffer; each iteration takes ``--updates_per_iter``
Adam steps on the negative log-likelihood of the label under the student's Gaussian, plus the
squared error of the student's mean action in units of the bounds (keeps the mean honest while the
state-dependent std is still finding the states it cannot resolve). The actor's observation
normaliser learns from the collected states for the first half of the run and is then frozen.

Usage (inside ``active_isaac``)::

    python scripts/train_dagger_sac_student.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --num_envs 1024 \\
        --teacher logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt --iterations 200 --run_name d1

Writes ``logs/fast_sac/<experiment>_dagger/<date>_<run>/model_<it>.pt`` and ``dagger.json``.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.algorithms.fast_sac import FastSAC, FrozenTeacher, IsaacLabSacEnv  # noqa: E402

parser = argparse.ArgumentParser(description="DAgger into a FastSAC actor.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--teacher", type=str, required=True, help="rsl_rl checkpoint of the teacher.")
parser.add_argument("--init_student", type=str, default=None, help="FastSAC checkpoint to start the actor from.")
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--beta_start", type=float, default=1.0)
parser.add_argument("--beta_floor", type=float, default=0.3)
parser.add_argument("--beta_iters", type=int, default=100, help="Iterations over which beta decays to the floor.")
parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Aggregate dataset size (states).")
parser.add_argument("--updates_per_iter", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--mse_weight", type=float, default=1.0)
parser.add_argument("--label_margin", type=float, default=0.95)
parser.add_argument("--save_interval", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.compute_final_obs = True
    if "gamma" in getattr(env_cfg.rewards.grasp_progress, "params", {}):
        env_cfg.rewards.grasp_progress.params["gamma"] = agent_cfg.gamma
    torch.manual_seed(args_cli.seed)
    agent_cfg.experiment_name = agent_cfg.experiment_name.replace("_sac", "") + "_dagger"

    with launch_simulation(env_cfg, args_cli):
        log_root = os.path.abspath(os.path.join("logs", "fast_sac", agent_cfg.experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
        log_dir = os.path.join(log_root, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[DAgger] logging to {log_dir}")
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        sac_env = IsaacLabSacEnv(env, agent_cfg.actor_obs_groups, agent_cfg.critic_obs_groups)
        if agent_cfg.critic_obs_groups != ["policy"]:
            raise ValueError("the teacher reads the 'policy' group; critic_obs_groups must be ['policy']")
        device = sac_env.device
        agent_cfg.device = str(device)
        alg = FastSAC(agent_cfg, sac_env.n_obs, sac_env.n_critic_obs, sac_env.n_act, sac_env.num_envs, device)
        if args_cli.init_student:
            alg.load(args_cli.init_student, load_optimizers=False)
        teacher = FrozenTeacher(args_cli.teacher, sac_env.n_critic_obs, "policy", alg.action_low, alg.action_high, device)
        teacher_std = teacher.std
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "dagger_args.json"), "w") as f:
            json.dump(vars(args_cli), f, indent=2, default=str)

        n = sac_env.num_envs
        low, high = alg.action_low * args_cli.label_margin, alg.action_high * args_cli.label_margin
        buf_obs = torch.zeros(args_cli.buffer_size, sac_env.n_obs, device=device)
        buf_lab = torch.zeros(args_cli.buffer_size, sac_env.n_act, device=device)
        ptr = count = 0
        optimizer = torch.optim.Adam(alg.actor.parameters(), lr=args_cli.lr)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        history = []

        obs, cobs = sac_env.reset()
        beta = args_cli.beta_start
        teacher_drives = torch.rand(n, device=device) < beta
        ep_return = torch.zeros(n, device=device)
        for it in range(args_cli.iterations):
            t0 = time.time()
            frac = min(it / max(args_cli.beta_iters, 1), 1.0)
            beta = args_cli.beta_start + (args_cli.beta_floor - args_cli.beta_start) * frac
            if it == args_cli.iterations // 2 and hasattr(alg.obs_normalizer, "count"):
                alg.obs_normalizer.until = int(alg.obs_normalizer.count)  # freeze the statistics
            stats = {"placed": [0, 0], "dropped": [0, 0], "episodes": [0, 0], "return": [0.0, 0.0]}  # [student, teacher]
            with torch.no_grad():
                for _ in range(args_cli.steps_per_iter):
                    label = teacher.mean(cobs).clamp(low, high)
                    idx = (ptr + torch.arange(n, device=device)) % args_cli.buffer_size
                    buf_obs[idx], buf_lab[idx] = obs, label
                    ptr = (ptr + n) % args_cli.buffer_size
                    count = min(count + n, args_cli.buffer_size)
                    alg.obs_normalizer(obs, update=True)
                    actions = torch.where(teacher_drives[:, None], teacher.act(cobs), alg.act(obs))
                    step = sac_env.step(actions)
                    ep_return += step["rewards"]
                    done = step["dones"]
                    if done.any():
                        flags = sac_env.termination_flags(("placed", "dropped"))
                        for who, mask in ((0, ~teacher_drives), (1, teacher_drives)):
                            d = done & mask
                            stats["episodes"][who] += int(d.sum())
                            stats["return"][who] += float(ep_return[d].sum())
                            for name in ("placed", "dropped"):
                                stats[name][who] += int((d & flags[name]).sum())
                        ep_return[done] = 0.0
                        redraw = torch.rand(n, device=device) < beta
                        teacher_drives = torch.where(done, redraw, teacher_drives)
                    obs, cobs = step["obs"], step["critic_obs"]
            t1 = time.time()

            nll_sum = mse_sum = std_sum = 0.0
            sq = torch.zeros(sac_env.n_act, device=device)
            for _ in range(args_cli.updates_per_iter):
                rows = torch.randint(0, count, (args_cli.batch_size,), device=device)
                obs_n = alg.obs_normalizer(buf_obs[rows], update=False)
                lab = buf_lab[rows]
                mean_action, _, log_std = alg.actor(obs_n)
                nll = -alg.actor.log_prob_of(obs_n, lab).mean()
                mse = ((mean_action - lab) / alg.actor.action_scale).pow(2).sum(-1).mean()
                loss = nll + args_cli.mse_weight * mse
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                nll_sum += nll.item(); mse_sum += mse.item(); std_sum += log_std.exp().mean().item()
                sq += ((mean_action.detach() - lab) / teacher_std).pow(2).mean(0)
            u = args_cli.updates_per_iter
            rmse = (sq / u).sqrt()
            row = {
                "iteration": it, "beta": beta, "dataset": count,
                "nll": nll_sum / u, "mse_bounds": mse_sum / u, "std_mean": std_sum / u,
                "rmse_in_teacher_std": rmse.mean().item(), "rmse_per_dim": [round(x, 3) for x in rmse.tolist()],
                "student_episodes": stats["episodes"][0], "teacher_episodes": stats["episodes"][1],
                "student_success": stats["placed"][0] / max(stats["episodes"][0], 1),
                "teacher_success": stats["placed"][1] / max(stats["episodes"][1], 1),
                "student_dropped": stats["dropped"][0] / max(stats["episodes"][0], 1),
                "student_return": stats["return"][0] / max(stats["episodes"][0], 1),
                "teacher_return": stats["return"][1] / max(stats["episodes"][1], 1),
                "collect_s": t1 - t0, "learn_s": time.time() - t1,
            }
            history.append(row)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"DAgger/{key}", value, it)
            print(f"[DAgger {it:4d}] beta {beta:.2f} nll {row['nll']:.3f} rmse/std {row['rmse_in_teacher_std']:.3f} "
                  f"std {row['std_mean']:.3f} | student eps {row['student_episodes']} success {row['student_success']:.3f} "
                  f"| teacher eps {row['teacher_episodes']} success {row['teacher_success']:.3f} "
                  f"| {row['collect_s']:.1f}+{row['learn_s']:.1f}s", flush=True)
            if (it + 1) % args_cli.save_interval == 0 or it + 1 == args_cli.iterations:
                _save(alg, log_dir, it + 1, sac_env, history)
        env.close()


def _save(alg, log_dir, it, sac_env, history):
    alg.global_step = 0  # a fresh SAC run starts from this actor
    alg.save(os.path.join(log_dir, f"model_{it}.pt"), extra={
        "actor_obs_groups": sac_env.actor_obs_groups, "critic_obs_groups": sac_env.critic_obs_groups,
        "dagger": {"iterations": it, "last": history[-1]},
    })
    with open(os.path.join(log_dir, "dagger.json"), "w") as f:
        json.dump({"args": vars(args_cli), "history": history}, f, indent=2, default=str)
    print(f"[DAgger] saved model_{it}.pt")


if __name__ == "__main__":
    main()
