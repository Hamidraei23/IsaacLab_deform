"""DAgger: distil the joint-position lift teacher into a twist (IK-Rel) student.

The student acts in ``Lift-Cube-Franka-Twist-v0`` -- Isaac Lab's IK-Rel lift task, whose arm action
is a 6D delta pose of the tool point in the robot base frame, scaled by 0.5. At every visited state
the teacher (a joint-position RSL-RL checkpoint, 36 observations -> 8 actions) is asked what it would
do, and its answer is moved into the student's action space.

The teacher's joint targets are not poses. They sit several radians from the arm (well past the joint
limits) and saturate its stiffness-80 drives, so what the teacher really commands is a torque:

    e        = q_default + teacher_scale * a_teacher[:7] - q                (teacher joint error)
    tau      = clip(K_t e - D_t qd, +-effort_t)  [- g(q) if the student arm is weightless]
    e_s      = (tau + D_s qd) / K_s                                         (student error, same torque)
    label    = [J_tool(q) e_s / ik_scale, sign(a_teacher[7])]               (raw IK-Rel action + gripper)

IK-Rel maps a delta pose back to joints through J^+, so J e_s returns (the task-space part of) e_s:
the student's drive then pushes the arm as the teacher's did. Exact forward kinematics of the target
was tried and is worse -- it turns a saturating push into a far-away pose. ``J_tool`` is the IK action's
own base-frame Jacobian at its tool offset. The teacher sees the student's observation with its own
previous action in place of the student's (7 -> 8 entries), zeroed at reset as the action manager does.

Which policy moves the arm is mixed per step: the teacher's label with probability beta, otherwise the
student's mean action. Beta falls linearly from 1 to 0 over ``--beta_iters`` iterations. Every
(observation, label) pair is kept in an aggregated buffer, and the student is regressed on it with a
per-dimension normalised MSE, so the 6 small arm entries are not drowned by the +-1 gripper.

The critic is fitted alongside on the rollout's lambda-returns, so a PPO resume starts from a value
function for the student rather than a random one. Checkpoints are ordinary RSL-RL ``model_<it>.pt``
files under ``logs/rsl_rl/lift_cube_franka_twist/<run_name>``; resume them with PPO through
``scripts/train.py --task Lift-Cube-Franka-Twist-v0 --resume --load_run <run_name> --checkpoint ...``.

Example (from the sheet_rl project)::

    python scripts/train_dagger_twist.py --run_name dagger_v1 \\
        --teacher logs/rsl_rl/lift_cube_franka/pretrained_motion_cost/model_1550.pt
"""

import argparse
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env, write_run_manifest
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import sheet_rl.tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from franka_twist_math import limit_twist  # noqa: E402

STUDENT_OBS_TERMS = ["joint_pos", "joint_vel", "object_position", "target_object_position", "actions"]
"""The policy observation layout both tasks share; only the size of ``actions`` differs (7 vs 8)."""

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Lift-Cube-Franka-Twist-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--teacher", required=True, type=Path, help="Joint-position RSL-RL checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--beta_iters", type=int, default=50, help="Iterations over which beta falls from 1 to 0.")
parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Aggregated (obs, label) pairs kept.")
parser.add_argument("--updates_per_iter", type=int, default=50, help="Student gradient steps per iteration.")
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--actor_lr", type=float, default=1e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--student_std", type=float, default=0.05, help="Exploration spread saved for the PPO resume.")
parser.add_argument(
    "--torque_match",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Convert the teacher's joint error into the error at which the student's drive applies the same torque "
    "(see the module docstring). --no-torque_match uses the raw joint error.",
)
parser.add_argument(
    "--match_gravity",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="With --torque_match on a weightless student arm, remove the share of torque the teacher spent on gravity.",
)
parser.add_argument("--label_gain", type=float, default=1.0, help="Scale on the teacher's converted delta pose.")
parser.add_argument("--max-linear-speed", type=float, default=0.0, help="Label twist limit [m/s]; 0 disables.")
parser.add_argument("--max-angular-speed", type=float, default=0.0, help="Label twist limit [rad/s]; 0 disables.")
parser.add_argument("--save_interval", type=int, default=50)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args, remaining = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining


def drive_gains(actuators: dict, joint_names: list[str]) -> torch.Tensor:
    """Per-joint (stiffness, damping, effort limit) of the named joints, shape (3, J), from actuator configs."""
    import re

    gains = []
    for name in joint_names:
        matches = [cfg for cfg in actuators.values() if any(re.fullmatch(e, name) for e in cfg.joint_names_expr)]
        values = (matches[0].stiffness, matches[0].damping, matches[0].effort_limit_sim) if len(matches) == 1 else ()
        if not values or not all(isinstance(v, float) for v in values):
            raise ValueError(f"Cannot resolve one scalar drive (stiffness, damping, effort) for joint {name}.")
        gains.append(values)
    return torch.tensor(gains).T


def load_teacher(path: Path, device: str) -> torch.nn.Module:
    """Rebuild the teacher's actor MLP from its checkpoint (no observation normalisation expected)."""
    from rsl_rl.modules import MLP

    state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    if any(key.startswith("obs_normalizer.") for key in state):
        raise ValueError(f"Teacher {path} normalises its observations; this script expects it not to.")
    weights = [state[key] for key in sorted((k for k in state if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))]
    hidden = [w.shape[0] for w in weights[:-1]]
    activation = "elu"
    agent_yaml = path.parent / "params" / "agent.yaml"
    if agent_yaml.is_file():
        import yaml

        activation = yaml.safe_load(agent_yaml.read_text())["actor"].get("activation", activation)
    mlp = MLP(weights[0].shape[1], weights[-1].shape[0], hidden, activation).to(device)
    mlp.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")})
    mlp.eval()
    for parameter in mlp.parameters():
        parameter.requires_grad_(False)
    return mlp


class TeacherInStudentSpace:
    """Ask the joint teacher what it would do and express the answer as a raw IK-Rel action."""

    def __init__(self, env, teacher: torch.nn.Module, teacher_arm_scale: float, teacher_actuators: dict):
        uenv = env.unwrapped
        self.teacher = teacher
        self.teacher_arm_scale = teacher_arm_scale
        self.robot = uenv.scene["robot"]
        self.arm = uenv.action_manager.get_term("arm_action")
        self.dt = uenv.step_dt
        self.arm_ids = self.arm._joint_ids
        self.ik_scale = self.arm._scale  # (N, 6)
        num_actions = uenv.action_manager.total_action_dim
        self.student_action_size = uenv.observation_manager.group_obs_term_dim["policy"][-1][0]
        if self.student_action_size != num_actions:
            raise ValueError("Expected the last policy observation term to be the student's last action.")
        self.teacher_inputs = teacher[0].in_features
        teacher_actions = teacher[-1].out_features
        if teacher_actions != 8 or self.arm.action_dim != 6 or num_actions != 7:
            raise ValueError(f"Expected an 8-action teacher and a 6+1-action twist student; got {teacher_actions}, {num_actions}.")
        self.previous = torch.zeros(uenv.num_envs, teacher_actions, device=uenv.device)
        self.raw_teacher = torch.zeros_like(self.previous)
        self.joint_error = torch.zeros(uenv.num_envs, device=uenv.device)
        self.teacher_gains = drive_gains(teacher_actuators, self.arm._joint_names).to(uenv.device)
        self.student_gains = drive_gains(uenv.cfg.scene.robot.actuators, self.arm._joint_names).to(uenv.device)
        self.add_gravity = bool(uenv.cfg.scene.robot.spawn.rigid_props.disable_gravity)
        print(f"[DAgger] teacher drives (K, D, effort) {self.teacher_gains.tolist()}")
        print(f"[DAgger] student drives (K, D, effort) {self.student_gains.tolist()}; weightless arm: {self.add_gravity}")

    def gravity(self) -> torch.Tensor:
        """g(q) of the arm joints; on a weightless student arm the simulator may report zero (checked once)."""
        g = self.robot.data.gravity_compensation_forces.torch[:, self.robot.num_base_dofs :][:, self.arm_ids]
        if not hasattr(self, "_gravity_reported"):
            self._gravity_reported = True
            print(f"[DAgger] |g(q)| on the student arm: {g.norm(dim=-1).mean().item():.2f} N m", flush=True)
        return g

    def teacher_obs(self, obs: torch.Tensor) -> torch.Tensor:
        teacher_obs = torch.cat((obs[:, : -self.student_action_size], self.previous), dim=-1)
        if teacher_obs.shape[-1] != self.teacher_inputs:
            raise ValueError(f"Teacher expects {self.teacher_inputs} observations, built {teacher_obs.shape[-1]}.")
        return teacher_obs

    def label(self, obs: torch.Tensor) -> torch.Tensor:
        self.raw_teacher = self.teacher(self.teacher_obs(obs))
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        q_target = self.robot.data.default_joint_pos.torch[:, self.arm_ids] + self.teacher_arm_scale * self.raw_teacher[:, :7]
        error = q_target - q
        self.joint_error = error.norm(dim=-1)
        if args.torque_match:
            q_vel = self.robot.data.joint_vel.torch[:, self.arm_ids]
            (k_t, d_t, effort_t), (k_s, d_s, _) = self.teacher_gains, self.student_gains
            torque = torch.maximum(torch.minimum(k_t * error - d_t * q_vel, effort_t), -effort_t)
            if args.match_gravity and self.add_gravity:
                torque = torque - self.gravity()  # the teacher's arm spent this holding itself up
            error = (torque + d_s * q_vel) / k_s
        jacobian = self.arm._compute_frame_jacobian()  # base frame, at the IK tool offset
        delta = args.label_gain * torch.bmm(jacobian, error.unsqueeze(-1)).squeeze(-1)
        delta = limit_twist(delta / self.dt, args.max_linear_speed, args.max_angular_speed) * self.dt
        gripper = torch.where(self.raw_teacher[:, 7:] < 0, -1.0, 1.0)
        return torch.cat((delta / self.ik_scale, gripper), dim=-1)

    def after_step(self, dones: torch.Tensor) -> None:
        self.previous = self.raw_teacher.clone()
        self.previous[dones.bool()] = 0.0


@hydra_task_config(args.task, args.agent, play_mode=False)
def main(env_cfg, agent_cfg):
    from rsl_rl.runners import OnPolicyRunner
    from torch.utils.tensorboard import SummaryWriter

    from isaaclab_tasks.contrib.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg as TeacherEnvCfg

    if min(args.iterations, args.steps_per_iter, args.beta_iters, args.updates_per_iter, args.batch_size) < 1:
        raise ValueError("Iteration, step, update and batch counts must be positive.")
    if args.student_std <= 0 or args.label_gain <= 0:
        raise ValueError("--student_std and --label_gain must be positive.")
    if not args.run_name:
        raise ValueError("--run_name is required: it names the run folder under logs/rsl_rl/<experiment>/.")
    if not args.teacher.is_file():
        raise FileNotFoundError(f"No teacher checkpoint at {args.teacher}")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = args.seed
    agent_cfg.run_name = args.run_name
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.actor.distribution_cfg.init_std = args.student_std
    log_dir = Path("logs", "rsl_rl", agent_cfg.experiment_name, args.run_name).resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Run folder already exists: {log_dir}. Choose another --run_name.")
    teacher_env_cfg = TeacherEnvCfg()
    teacher_arm_scale = float(teacher_env_cfg.actions.arm_action.scale)
    teacher_actuators = teacher_env_cfg.scene.robot.actuators

    with launch_simulation(env_cfg, args):
        write_run_manifest(str(log_dir), library="rsl_rl", task=args.task, metadata={"agent": args.agent})
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        invocation = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (log_dir / "dagger.json").write_text(json.dumps({**invocation, "teacher_arm_scale": teacher_arm_scale}, indent=2))

        env = RslRlVecEnvWrapper(
            create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False),
            clip_actions=agent_cfg.clip_actions,
        )
        uenv = env.unwrapped
        terms = uenv.observation_manager.active_terms["policy"]
        if terms != STUDENT_OBS_TERMS:
            raise ValueError(f"Unexpected policy observation terms {terms}; expected {STUDENT_OBS_TERMS}.")
        device = agent_cfg.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        actor, critic = runner.alg.actor, runner.alg.critic
        teacher = TeacherInStudentSpace(env, load_teacher(args.teacher, device), teacher_arm_scale, teacher_actuators)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        writer = SummaryWriter(str(log_dir))
        gamma, lam = agent_cfg.algorithm.gamma, agent_cfg.algorithm.lam

        obs = env.get_observations().to(device)
        obs_dim, act_dim = obs["policy"].shape[-1], env.num_actions
        buffer_obs = torch.zeros(args.buffer_size, obs_dim, device=device)
        buffer_label = torch.zeros(args.buffer_size, act_dim, device=device)
        buffer_count = buffer_head = 0
        steps, n = args.steps_per_iter, env.num_envs
        roll_obs = torch.zeros(steps, n, obs_dim, device=device)
        roll_values = torch.zeros(steps, n, device=device)
        roll_rewards = torch.zeros(steps, n, device=device)
        roll_dones = torch.zeros(steps, n, device=device)
        episode_return = torch.zeros(n, device=device)
        uenv.episode_length_buf = torch.randint_like(uenv.episode_length_buf, high=int(uenv.max_episode_length))
        print(f"[DAgger] teacher {args.teacher} -> student {args.task}; logging to {log_dir}", flush=True)

        for it in range(args.iterations):
            start = time.time()
            beta = max(0.0, 1.0 - it / args.beta_iters)
            finished_returns, successes, teacher_steps = [], [], 0.0
            label_abs = torch.zeros(act_dim, device=device)
            joint_error = 0.0
            actor.eval()
            with torch.no_grad():
                for step in range(steps):
                    policy_obs = obs["policy"]
                    label = teacher.label(policy_obs)
                    student = actor(obs)
                    use_teacher = torch.rand(n, 1, device=device) < beta
                    action = torch.where(use_teacher, label, student)
                    teacher_steps += use_teacher.float().mean().item()
                    label_abs += label.abs().mean(dim=0)
                    joint_error += teacher.joint_error.mean().item() / steps
                    rows = (buffer_head + torch.arange(n, device=device)) % args.buffer_size
                    buffer_obs[rows], buffer_label[rows] = policy_obs, label
                    buffer_head = (buffer_head + n) % args.buffer_size
                    buffer_count = min(buffer_count + n, args.buffer_size)
                    roll_obs[step], roll_values[step] = policy_obs, critic(obs).squeeze(-1)

                    obs, rewards, dones, extras = env.step(action.to(env.device))
                    obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                    teacher.after_step(dones)
                    rewards = rewards.clone()
                    if "time_outs" in extras:  # bootstrap timeouts, as RSL-RL's PPO does
                        rewards += gamma * roll_values[step] * extras["time_outs"].to(device).float()
                    roll_rewards[step], roll_dones[step] = rewards, dones.float()
                    episode_return += rewards
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids):
                        finished_returns += episode_return[done_ids].tolist()
                        episode_return[done_ids] = 0.0
                    if "Metrics/success_rate" in extras.get("log", {}):
                        successes.append(float(extras["log"]["Metrics/success_rate"]))

                # lambda-returns for the critic, bootstrapped from the last observation
                last_value = critic(obs).squeeze(-1)
                returns = torch.zeros_like(roll_rewards)
                advantage = torch.zeros(n, device=device)
                for step in reversed(range(steps)):
                    next_value = last_value if step == steps - 1 else roll_values[step + 1]
                    not_done = 1.0 - roll_dones[step]
                    td = roll_rewards[step] + gamma * next_value * not_done - roll_values[step]
                    advantage = td + gamma * lam * not_done * advantage
                    returns[step] = advantage + roll_values[step]
            collect_time = time.time() - start

            # student: aggregated-buffer regression with per-dimension normalised MSE
            actor.train()
            label_std = buffer_label[:buffer_count].std(dim=0).clamp_min(1e-3)
            actor_loss = raw_mse = 0.0
            for _ in range(args.updates_per_iter):
                rows = torch.randint(0, buffer_count, (min(args.batch_size, buffer_count),), device=device)
                prediction = actor({"policy": buffer_obs[rows]})
                error = prediction - buffer_label[rows]
                loss = (error / label_std).square().mean()
                actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
                actor_loss += loss.item() / args.updates_per_iter
                raw_mse += error.detach().square().mean().item() / args.updates_per_iter

            # critic: regress this rollout's lambda-returns
            critic.train()
            flat_obs, flat_returns = roll_obs.reshape(-1, obs_dim), returns.reshape(-1)
            critic_loss, batches = 0.0, 0
            for _ in range(args.critic_epochs):
                for rows in torch.randperm(len(flat_returns), device=device).split(max(1, len(flat_returns) // 4)):
                    loss = (critic({"policy": flat_obs[rows]}).squeeze(-1) - flat_returns[rows]).square().mean()
                    critic_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                    critic_loss, batches = critic_loss + loss.item(), batches + 1
            critic_loss /= max(1, batches)

            stats = {
                "DAgger/beta": beta,
                "DAgger/teacher_step_fraction": teacher_steps / steps,
                "DAgger/actor_loss_normalised": actor_loss,
                "DAgger/actor_mse_raw": raw_mse,
                "DAgger/critic_loss": critic_loss,
                "DAgger/buffer_size": buffer_count,
                "DAgger/label_abs_arm": (label_abs[:6] / steps).mean().item(),
                "DAgger/teacher_joint_error": joint_error,
                "Train/mean_episode_return": sum(finished_returns) / len(finished_returns) if finished_returns else math.nan,
                "Metrics/success_rate": sum(successes) / len(successes) if successes else math.nan,
            }
            for key, value in stats.items():
                if not math.isnan(value):
                    writer.add_scalar(key, value, it)
            print(
                f"[DAgger] it {it:4d}  beta {beta:.2f}  teacher-steps {stats['DAgger/teacher_step_fraction']:.2f}  "
                f"actor {actor_loss:.4f} (raw {raw_mse:.2e})  critic {critic_loss:.3f}  "
                f"return {stats['Train/mean_episode_return']:.2f}  success {stats['Metrics/success_rate']:.2f}  "
                f"|label arm| {stats['DAgger/label_abs_arm']:.3f}  |q_t-q| {joint_error:.2f}  {collect_time:.1f}s+{time.time() - start - collect_time:.1f}s",
                flush=True,
            )
            if it % args.save_interval == 0 or it == args.iterations - 1:
                saved = runner.alg.save()
                saved.update(iter=it, infos={"dagger": invocation, "teacher": str(args.teacher.resolve())})
                torch.save(saved, log_dir / f"model_{it}.pt")

        writer.close()
        print(f"[DAgger] done. Resume with PPO: --load_run {args.run_name} --checkpoint model_{args.iterations - 1}.pt")
        env.close()


if __name__ == "__main__":
    main()
