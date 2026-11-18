"""Roll out the teacher and the student, deterministically and stochastically, and compare them.

The teacher is an rsl_rl PPO checkpoint reading the ``policy`` group; the student is a FastSAC
checkpoint (``scripts/train_sac.py`` or ``scripts/train_dagger_sac_student.py``) reading the
groups recorded in it (``action_history``). Both run in the same simulator, in turn, for the same
number of episodes per environment. Teacher actions are clipped to the student's bounds, as they
are everywhere else in this pipeline.

Per episode: the outcome (``placed`` / ``dropped`` / ``held_too_low`` / ``time_out`` / other),
whether the sheet was extracted, the return, the length, and the coverage figures the ``coverage``
term held when the episode ended. Per mode: the release angle averaged from the environment's log.
Agreement: while the *teacher* drives, the student's mean action is compared with the teacher's
on the same states (RMSE per dimension in units of the teacher's std, and the student's
log-likelihood of the teacher's clipped action); while the *student* drives, its executed action
is compared with the teacher's mean on the student's own states.

Usage (inside ``active_isaac``)::

    python scripts/eval_teacher_student.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --num_envs 256 \\
        --episodes 2 --teacher logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt \\
        --student logs/fast_sac/sheet_platform_twist_dagger/<run>/model_200.pt --tag dagger200

Writes ``evaluations/teacher_student/<tag>_seed<N>.json`` and prints a table.
"""

import argparse
import json
import os
import sys
import time

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.algorithms.fast_sac import FastSAC, FastSACCfg, FrozenTeacher, IsaacLabSacEnv  # noqa: E402
from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import _phase_reached  # noqa: E402

OUTCOMES = ("placed", "dropped", "held_too_low", "time_out")
MODES = ("teacher_det", "teacher_stoch", "student_det", "student_stoch")

parser = argparse.ArgumentParser(description="Compare teacher and student rollouts.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point", help="FastSAC config (bounds, groups).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2, help="Episodes per environment and mode.")
parser.add_argument("--teacher", type=str, required=True, help="rsl_rl checkpoint of the teacher.")
parser.add_argument("--student", type=str, default=None, help="FastSAC checkpoint of the student.")
parser.add_argument("--modes", type=str, nargs="*", default=None, help=f"Subset of {MODES}.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag", type=str, default="eval")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args


def _load_student(path: str, device: str) -> tuple[FastSAC, list[str], list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = FastSACCfg(experiment_name="eval")
    for key, value in ckpt["cfg"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.device, cfg.compile = device, False
    dims = ckpt["dims"]
    alg = FastSAC(cfg, dims["n_obs"], dims["n_critic_obs"], dims["n_act"], 1, device)
    alg.load(path, load_optimizers=False)
    alg.actor.eval()
    return alg, list(ckpt["actor_obs_groups"]), list(ckpt["critic_obs_groups"])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.compute_final_obs = True
    if "gamma" in getattr(env_cfg.rewards.grasp_progress, "params", {}):
        env_cfg.rewards.grasp_progress.params["gamma"] = agent_cfg.gamma
    torch.manual_seed(args_cli.seed)
    modes = tuple(args_cli.modes) if args_cli.modes else MODES
    if args_cli.student is None:
        modes = tuple(m for m in modes if m.startswith("teacher"))

    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        device = str(env.unwrapped.device)
        student = None
        actor_groups, critic_groups = agent_cfg.actor_obs_groups, agent_cfg.critic_obs_groups
        if args_cli.student:
            student, actor_groups, critic_groups = _load_student(args_cli.student, device)
        sac_env = IsaacLabSacEnv(env, actor_groups, critic_groups)
        if student is None:
            # bounds from the task's SAC config, so the teacher is clipped exactly as in training
            student_like = FastSAC(agent_cfg, sac_env.n_obs, sac_env.n_critic_obs, sac_env.n_act, 1, device)
            low, high = student_like.action_low, student_like.action_high
        else:
            low, high = student.action_low, student.action_high
        if critic_groups != ["policy"]:
            raise ValueError("the teacher reads the 'policy' group; critic_obs_groups must be ['policy']")
        teacher = FrozenTeacher(args_cli.teacher, sac_env.n_critic_obs, "policy", low, high, device)
        teacher_std = teacher.std
        coverage_term = env.unwrapped.reward_manager.get_term_cfg("coverage").func
        results = {}
        for mode in modes:
            t0 = time.time()
            results[mode] = _rollout(sac_env, mode, teacher, student, teacher_std, low, high, args_cli.episodes)
            results[mode]["wall_time_s"] = time.time() - t0
            print(f"[eval] {mode}: {results[mode]['episodes']} episodes in {results[mode]['wall_time_s']:.0f} s")
        env.close()

    out_dir = os.path.join("evaluations", "teacher_student")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args_cli.tag}_seed{args_cli.seed}.json")
    with open(out, "w") as f:
        json.dump({"task": args_cli.task, "teacher": args_cli.teacher, "student": args_cli.student,
                   "num_envs": args_cli.num_envs, "episodes_per_env": args_cli.episodes, "seed": args_cli.seed,
                   "results": results}, f, indent=2)
    print(_table(results))
    print(f"[eval] written {out}")


@torch.no_grad()
def _rollout(sac_env, mode, teacher, student, teacher_std, low, high, episodes_per_env):
    env = sac_env.unwrapped
    n = sac_env.num_envs
    device = sac_env.device
    who, how = mode.split("_")
    deterministic = how == "det"
    coverage_term = env.reward_manager.get_term_cfg("coverage").func
    obs, cobs = sac_env.reset()
    done_count = torch.zeros(n, dtype=torch.long, device=device)
    ep_return = torch.zeros(n, device=device)
    ep_length = torch.zeros(n, device=device)
    rec = {k: [] for k in ("outcome", "extracted", "return", "length", "cov_best", "cov_best_released", "cov_final")}
    agree_sq = torch.zeros(sac_env.n_act, device=device)
    agree_logp = 0.0
    agree_n = 0
    angle_sum, angle_n = 0.0, 0
    max_steps = episodes_per_env * sac_env.max_episode_length + 10
    for _ in range(max_steps):
        if not (done_count < episodes_per_env).any():
            break
        # the coverage the release is judged with is the one computed on the previous step
        snap_best, snap_best_rel, snap_cov = coverage_term.best.clone(), coverage_term.best_released.clone(), coverage_term.coverage.clone()
        snap_phase = _phase_reached(env).clone()
        teacher_mu = teacher.mean(cobs)
        if who == "teacher":
            actions = teacher.act(cobs, deterministic=deterministic)
            if student is not None:
                obs_n = student.obs_normalizer(obs, update=False)
                student_mu = student.actor(obs_n)[0].float()
                agree_sq += ((student_mu - teacher_mu) / teacher_std).pow(2).sum(0)
                # scored inside 95% of the bounds, as the DAgger labels are: an action *on* the bound has
                # no finite pre-tanh value and its log-probability says nothing about the fit
                agree_logp += student.actor.log_prob_of(obs_n, actions.clamp(0.95 * low, 0.95 * high)).float().sum().item()
                agree_n += n
        else:
            actions = student.act(obs, deterministic=deterministic)
            agree_sq += ((actions - teacher_mu) / teacher_std).pow(2).sum(0)
            agree_n += n
        step = sac_env.step(actions)
        ep_return += step["rewards"]
        ep_length += 1
        done = step["dones"]
        if done.any():
            flags = sac_env.termination_flags(OUTCOMES)
            log = step["extras"].get("log", {})
            if "Placement/release_angle_min" in log and float(log.get("Placement/release_angle_measured", 1.0)) > 0:
                angle_sum += float(log["Placement/release_angle_min"])
                angle_n += 1
            for i in done.nonzero(as_tuple=False).flatten().tolist():
                if done_count[i] >= episodes_per_env:
                    continue
                outcome = next((name for name in OUTCOMES if name in flags and bool(flags[name][i])), "other")
                rec["outcome"].append(outcome)
                rec["extracted"].append(bool(snap_phase[i]))
                rec["return"].append(ep_return[i].item())
                rec["length"].append(ep_length[i].item())
                rec["cov_best"].append(snap_best[i].item())
                rec["cov_best_released"].append(snap_best_rel[i].item())
                rec["cov_final"].append(snap_cov[i].item())
                done_count[i] += 1
            ep_return[done] = 0.0
            ep_length[done] = 0.0
        obs, cobs = step["obs"], step["critic_obs"]
    m = len(rec["outcome"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    out = {
        "episodes": m,
        "success": sum(o == "placed" for o in rec["outcome"]) / max(m, 1),
        **{name: sum(o == name for o in rec["outcome"]) / max(m, 1) for name in OUTCOMES if name != "placed"},
        "other": sum(o == "other" for o in rec["outcome"]) / max(m, 1),
        "extracted": mean([float(x) for x in rec["extracted"]]),
        "return": mean(rec["return"]),
        "length": mean(rec["length"]),
        "cov_best": mean(rec["cov_best"]),
        "cov_best_released": mean(rec["cov_best_released"]),
        "cov_final": mean(rec["cov_final"]),
        "release_angle_min": angle_sum / angle_n if angle_n else float("nan"),
    }
    if agree_n:
        rmse = (agree_sq / agree_n).sqrt()
        out["agreement_rmse_per_dim_in_teacher_std"] = [round(x, 3) for x in rmse.tolist()]
        out["agreement_rmse_in_teacher_std"] = rmse.mean().item()
        if who == "teacher" and student is not None:
            out["student_logprob_of_teacher_action"] = agree_logp / agree_n
    return out


def _table(results: dict) -> str:
    cols = ("episodes", "success", "dropped", "held_too_low", "time_out", "extracted", "return", "length",
            "cov_best", "cov_best_released", "release_angle_min", "agreement_rmse_in_teacher_std")
    head = "| mode | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [head, sep]
    for mode, r in results.items():
        cells = []
        for c in cols:
            v = r.get(c, float("nan"))
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        rows.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
