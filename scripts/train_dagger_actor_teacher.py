"""DAgger from a frozen FastSAC *actor* (the teacher) into a FastSAC actor that reads a different
observation group (the student) -- written to move the twist actor from the pre-repair
``action_history_legacy`` group (slot 49 = sum of all nine joints) onto the repaired
``action_history`` group (slot 49 = fingertip distance).

The teacher is ``--teacher`` loaded onto ``--teacher_obs_group`` with its own observation
normaliser, and it acts **stochastically** (user's rule): every environment is driven for whole
episodes by either the teacher (sampled) or the student (sampled); the assignment is redrawn at
every reset with probability ``beta`` for the teacher. ``beta`` is constant within a round and
given per round by ``--betas`` (``1.0 0.5 0.0 0.0``: teacher-driven states first, then a mix,
then two rounds on the student's own state distribution -- the "DAgger depth").

Every visited state is labelled with the teacher's full action distribution at that state (its
pre-tanh mean and log-std), and the student is trained on the aggregate dataset to minimise the
KL divergence from the teacher's Gaussian to its own (tanh is a bijection, so the pre-tanh KL is
the KL between the squashed policies), plus the squared error of the squashed mean in units of the
bounds. So the student reproduces the teacher's mean *and* its state-dependent spread, which is
what the later stochastic rollouts and the SAC fine-tune use.

The student starts from the teacher's weights (``--init_from_teacher``, default): the 48 shared
channels keep the teacher's normaliser statistics and the repaired channel's statistics are
re-estimated from the first iteration's states (``--renorm_channels``); the normaliser is then frozen.
The teacher's critics (on the ``policy`` group, untouched by the repair) are carried into the
checkpoint, so ``scripts/train_sac.py --checkpoint`` can fine-tune straight from it.

Usage (inside ``active_isaac``)::

    python scripts/train_dagger_actor_teacher.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --num_envs 1024 \\
        --teacher logs/fast_sac/sheet_platform_twist_sac/<run>/model_93500.pt --run_name repair49

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

from sheet_rl.algorithms.fast_sac import FastSAC, FastSACCfg, IsaacLabSacEnv  # noqa: E402
from sheet_rl.algorithms.fast_sac.buffers import EmpiricalNormalization  # noqa: E402
from sheet_rl.algorithms.fast_sac.networks import Actor  # noqa: E402

parser = argparse.ArgumentParser(description="DAgger from a FastSAC actor on one observation group into an actor on another.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--teacher", type=str, required=True, help="FastSAC checkpoint whose actor is the teacher.")
parser.add_argument("--teacher_obs_group", type=str, default="action_history_legacy")
parser.add_argument("--init_from_teacher", type=int, default=1, help="1: student starts from the teacher's weights and critics.")
parser.add_argument("--init_student", type=str, default=None, help="FastSAC checkpoint to start the student from instead of the teacher.")
parser.add_argument("--renorm_channels", type=int, nargs="*", default=[48],
                    help="Student observation channels whose normaliser statistics are re-estimated from the first round.")
parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.5, 0.0, 0.0], help="Teacher-drives probability per round.")
parser.add_argument("--iters_per_round", type=int, default=100)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--buffer_size", type=int, default=2_000_000, help="Aggregate dataset size (states).")
parser.add_argument("--updates_per_iter", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--mse_weight", type=float, default=1.0)
parser.add_argument("--logstd_weight", type=float, default=0.0,
                    help="Weight of the squared log-std error. KL(teacher||student) lets the student stay wider than "
                         "the teacher (it absorbs mean error into the spread); this term pins the spread to the teacher's.")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args


class ActorTeacher:
    """A frozen FastSAC actor with its observation normaliser, acting on its own observation group."""

    def __init__(self, checkpoint: str, n_obs: int, n_act: int, device):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = FastSACCfg(experiment_name="teacher")
        for key, value in ckpt["cfg"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        scale = None if cfg.action_scale is None else torch.tensor(cfg.action_scale, dtype=torch.float)
        bias = None if cfg.action_bias is None else torch.tensor(cfg.action_bias, dtype=torch.float)
        self.actor = Actor(n_obs, n_act, cfg.actor_hidden_dim, cfg.log_std_min, cfg.log_std_max, cfg.use_tanh,
                           cfg.use_layer_norm, scale, bias, device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()
        self.normalizer = EmpiricalNormalization(n_obs, device)
        self.normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        self.normalizer.until = 0  # frozen
        for param in self.actor.parameters():
            param.requires_grad_(False)
        self.obs_groups = ckpt.get("actor_obs_groups")

    @torch.no_grad()
    def distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(squashed mean action, pre-tanh mean, log-std) at ``obs``."""
        return self.actor(self.normalizer(obs, update=False))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.explore(self.normalizer(obs, update=False), deterministic=deterministic)


def gaussian_kl(mu_t, log_std_t, mu_s, log_std_s) -> torch.Tensor:
    """KL(N(mu_t, std_t) || N(mu_s, std_s)) summed over the action dimensions."""
    var_t, var_s = (2 * log_std_t).exp(), (2 * log_std_s).exp()
    return (log_std_s - log_std_t + (var_t + (mu_t - mu_s).pow(2)) / (2 * var_s) - 0.5).sum(-1)


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
        sac_env = IsaacLabSacEnv(env, agent_cfg.actor_obs_groups, agent_cfg.critic_obs_groups,
                                 extra_obs_groups=[args_cli.teacher_obs_group])
        device = sac_env.device
        agent_cfg.device = str(device)
        alg = FastSAC(agent_cfg, sac_env.n_obs, sac_env.n_critic_obs, sac_env.n_act, sac_env.num_envs, device)
        n_teacher_obs = sac_env.extra_obs.shape[-1] if sac_env.extra_obs is not None else None
        teacher = None  # built after the first reset, when the teacher group's width is known
        if args_cli.init_student:
            alg.load(args_cli.init_student, load_optimizers=False)
            print(f"[DAgger] student actor, normaliser and critics initialised from {args_cli.init_student}")
        elif args_cli.init_from_teacher:
            alg.load(args_cli.teacher, load_optimizers=False)
            print("[DAgger] student actor, normaliser and critics initialised from the teacher checkpoint")
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "dagger_args.json"), "w") as f:
            json.dump(vars(args_cli), f, indent=2, default=str)

        n = sac_env.num_envs
        buf_obs = torch.zeros(args_cli.buffer_size, sac_env.n_obs, device=device)
        buf_mu = torch.zeros(args_cli.buffer_size, sac_env.n_act, device=device)
        buf_logstd = torch.zeros(args_cli.buffer_size, sac_env.n_act, device=device)
        ptr = count = 0
        optimizer = torch.optim.Adam(alg.actor.parameters(), lr=args_cli.lr)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        history = []

        obs, cobs = sac_env.reset()
        tobs = sac_env.extra_obs
        teacher = ActorTeacher(args_cli.teacher, tobs.shape[-1], sac_env.n_act, device)
        print(f"[DAgger] teacher {args_cli.teacher} on '{args_cli.teacher_obs_group}' ({tobs.shape[-1]}), "
              f"student on {agent_cfg.actor_obs_groups} ({sac_env.n_obs}); rounds betas {args_cli.betas}, "
              f"{args_cli.iters_per_round} iterations each")
        normalizer_frozen = False
        total_iters = len(args_cli.betas) * args_cli.iters_per_round
        teacher_drives = torch.rand(n, device=device) < args_cli.betas[0]
        ep_return = torch.zeros(n, device=device)
        it = 0
        for rnd, beta in enumerate(args_cli.betas):
            for _ in range(args_cli.iters_per_round):
                t0 = time.time()
                stats = {"placed": [0, 0], "dropped": [0, 0], "episodes": [0, 0], "return": [0.0, 0.0]}  # [student, teacher]
                with torch.no_grad():
                    for _ in range(args_cli.steps_per_iter):
                        _, mu_t, logstd_t = teacher.distribution(tobs)
                        idx = (ptr + torch.arange(n, device=device)) % args_cli.buffer_size
                        buf_obs[idx], buf_mu[idx], buf_logstd[idx] = obs, mu_t, logstd_t
                        ptr = (ptr + n) % args_cli.buffer_size
                        count = min(count + n, args_cli.buffer_size)
                        if not (args_cli.init_from_teacher or args_cli.init_student) and not normalizer_frozen:
                            alg.obs_normalizer(obs, update=True)
                        actions = torch.where(teacher_drives[:, None], teacher.act(tobs), alg.act(obs))
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
                        obs, cobs, tobs = step["obs"], step["critic_obs"], sac_env.extra_obs
                t1 = time.time()

                if not normalizer_frozen:
                    # after the first collection (steps_per_iter x num_envs states, teacher-driven):
                    # statistics of the repaired channel(s) from them; the shared channels keep the
                    # teacher's. Then freeze, so every update sees the same normalisation.
                    norm = alg.obs_normalizer
                    if hasattr(norm, "_mean") and args_cli.renorm_channels:
                        data = buf_obs[:count][:, args_cli.renorm_channels]
                        for j, c in enumerate(args_cli.renorm_channels):
                            norm._mean[0, c] = data[:, j].mean()
                            norm._var[0, c] = data[:, j].var()
                            norm._std[0, c] = data[:, j].std()
                        print(f"[DAgger] normaliser channels {args_cli.renorm_channels} re-estimated: "
                              f"mean {[round(v, 4) for v in data.mean(0).tolist()]} std {[round(v, 4) for v in data.std(0).tolist()]}")
                    if hasattr(norm, "count"):
                        norm.until = int(norm.count)
                    normalizer_frozen = True

                kl_sum = mse_sum = std_sum = tstd_sum = lsd_sum = 0.0
                for _ in range(args_cli.updates_per_iter):
                    rows = torch.randint(0, count, (args_cli.batch_size,), device=device)
                    obs_n = alg.obs_normalizer(buf_obs[rows], update=False)
                    mu_t, logstd_t = buf_mu[rows], buf_logstd[rows]
                    mean_action, mu_s, logstd_s = alg.actor(obs_n)
                    kl = gaussian_kl(mu_t, logstd_t, mu_s, logstd_s).mean()
                    label_action = torch.tanh(mu_t) * alg.actor.action_scale + alg.actor.action_bias
                    mse = ((mean_action - label_action) / alg.actor.action_scale).pow(2).sum(-1).mean()
                    lsd = (logstd_s - logstd_t).pow(2).sum(-1).mean()
                    loss = kl + args_cli.mse_weight * mse + args_cli.logstd_weight * lsd
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    kl_sum += kl.item(); mse_sum += mse.item(); std_sum += logstd_s.exp().mean().item()
                    tstd_sum += logstd_t.exp().mean().item(); lsd_sum += lsd.item()
                u = args_cli.updates_per_iter
                row = {
                    "iteration": it, "round": rnd, "beta": beta, "dataset": count,
                    "kl": kl_sum / u, "mse_bounds": mse_sum / u, "std_mean": std_sum / u,
                    "teacher_std_mean": tstd_sum / u, "logstd_sq_err": lsd_sum / u,
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
                print(f"[DAgger {it:4d} r{rnd}] beta {beta:.2f} kl {row['kl']:.4f} mse {row['mse_bounds']:.4f} std {row['std_mean']:.3f} "
                      f"(teacher {row['teacher_std_mean']:.3f}, lsd {row['logstd_sq_err']:.3f}) "
                      f"| student eps {row['student_episodes']} success {row['student_success']:.3f} "
                      f"| teacher eps {row['teacher_episodes']} success {row['teacher_success']:.3f} "
                      f"| data {count} | {row['collect_s']:.1f}+{row['learn_s']:.1f}s", flush=True)
                it += 1
                if it % args_cli.save_interval == 0 or it == total_iters:
                    _save(alg, log_dir, it, sac_env, history)
            # a round boundary: redraw who drives under the next beta right away
            if rnd + 1 < len(args_cli.betas):
                teacher_drives = torch.rand(n, device=device) < args_cli.betas[rnd + 1]
        # summary over the last round (student-driven episodes only)
        last = [h for h in history if h["round"] == len(args_cli.betas) - 1]
        eps = sum(h["student_episodes"] for h in last)
        placed = sum(h["student_success"] * h["student_episodes"] for h in last)
        print(f"[DAgger] done. Last round: {eps} student-driven episodes, stochastic success {placed / max(eps, 1):.3f}; "
              f"checkpoint {os.path.join(log_dir, f'model_{total_iters}.pt')}", flush=True)
        env.close()


def _save(alg, log_dir, it, sac_env, history):
    alg.global_step = 0  # a fresh SAC run starts from this actor
    alg.save(os.path.join(log_dir, f"model_{it}.pt"), extra={
        "actor_obs_groups": sac_env.actor_obs_groups, "critic_obs_groups": sac_env.critic_obs_groups,
        "dagger": {"iterations": it, "last": history[-1], "teacher": args_cli.teacher,
                   "teacher_obs_group": args_cli.teacher_obs_group},
    })
    with open(os.path.join(log_dir, "dagger.json"), "w") as f:
        json.dump({"args": vars(args_cli), "history": history}, f, indent=2, default=str)
    print(f"[DAgger] saved model_{it}.pt", flush=True)


if __name__ == "__main__":
    main()
