"""Isaac Lab adapter and training loop for FastSAC (holosoma's ``FastSACEnv`` and ``learn``).

The loop is holosoma's: one step of every environment per iteration, every transition into the
replay buffer, then ``num_updates`` gradient updates once ``learning_starts`` iterations have
passed. Logging is tensorboard plus a console block in the style of rsl_rl's, including the
environment's own ``extras["log"]`` episode metrics (``Placement/success``, ``Coverage/*``, ...).

Added for this project: an optional *behaviour policy* that drives the environments in place of
the actor (the frozen teacher of the critic warm-up), held-out environments whose episodes score
the critic against Monte Carlo returns (``CriticValidator``), and a stop gate on those scores.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque
from typing import Protocol

import torch
from tensordict import TensorDict

from .config import FastSACCfg
from .fast_sac import FastSAC
from .validation import CriticGate, CriticValidator

PHASES = (
    ("grasps/ep", "Events/grasp"),
    ("extracted", "Events/extraction"),
    ("edge rise m", "Milestone/best_edge_rise"),
    ("released", "Placement/released"),
    ("placed", "Episode_Termination/placed"),
    ("dropped", "Episode_Termination/dropped"),
    ("into table", "Episode_Termination/arm_into_table"),
)
"""The sheet task's stages in order, as its environment logs them, printed on one line at the top of each
console block (whichever of them the task logs); the full alphabetical table follows."""


class BehaviorPolicy(Protocol):
    """What drives the environments when it is not the actor. Actions must already be within the actor's bounds."""

    def act(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor: ...

    def next_actions(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """A sample of the policy at the *next* state, stored for the policy-evaluation target."""
        ...


class IsaacLabSacEnv:
    """A manager-based Isaac Lab environment as FastSAC needs it.

    - The configured observation groups are concatenated into one actor and one critic tensor.
    - ``dones`` (any reset) are split from ``truncations`` (time-outs that were not also true
      terminations), so time-outs bootstrap and terminations do not.
    - The *true* final observation of every environment that reset is used as its next observation,
      not the post-reset one. This needs ``env_cfg.compute_final_obs = True``, which makes Isaac Lab
      compute the observations once more just before the reset (``extras["final_obs"]``).

    Note: ``final_obs`` is computed without advancing observation histories. A history term (such as
    ``action_history``'s last two actions) therefore shows the step *before* the final one in that
    one observation. It is read only on reset steps, and only bootstrapped on time-outs.
    """

    def __init__(self, env, actor_obs_groups: list[str], critic_obs_groups: list[str],
                 extra_obs_groups: list[str] | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        if not getattr(self.unwrapped.cfg, "compute_final_obs", False):
            raise ValueError("FastSAC needs env_cfg.compute_final_obs = True (the true pre-reset observations).")
        dims = self.unwrapped.observation_manager.group_obs_dim
        # groups read by something other than the actor and critic (e.g. a teacher on another
        # group); concatenated into ``self.extra_obs`` at every reset and step
        self.extra_obs_groups = list(extra_obs_groups or [])
        self.extra_obs: torch.Tensor | None = None
        for group in [*actor_obs_groups, *critic_obs_groups, *self.extra_obs_groups]:
            if group not in dims:
                raise ValueError(f"observation group '{group}' not in the environment's groups {list(dims)}")
            if not isinstance(dims[group], tuple) or len(dims[group]) != 1:
                raise ValueError(f"observation group '{group}' must be one concatenated vector, got shape {dims[group]}")
        self.actor_obs_groups = list(actor_obs_groups)
        self.critic_obs_groups = list(critic_obs_groups)
        self.n_obs = sum(dims[g][0] for g in self.actor_obs_groups)
        self.n_critic_obs = sum(dims[g][0] for g in self.critic_obs_groups)
        self.n_act = self.unwrapped.action_manager.total_action_dim
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = int(self.unwrapped.max_episode_length)

    @staticmethod
    def _finite(x: torch.Tensor) -> torch.Tensor:
        """Non-finite entries as zero. A blown-up simulation puts NaN in the observations of the step it
        terminates on; stored as that transition's next observation, it would turn every critic target
        of a batch into NaN, even though a terminal transition never bootstraps from its value."""
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _split(self, obs: dict, record_extra: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self._finite(torch.cat([obs[g] for g in self.actor_obs_groups], dim=-1))
        critic = self._finite(torch.cat([obs[g] for g in self.critic_obs_groups], dim=-1))
        if record_extra and self.extra_obs_groups:
            self.extra_obs = self._finite(torch.cat([obs[g] for g in self.extra_obs_groups], dim=-1))
        return actor, critic

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs, _ = self.env.reset()
        return self._split(obs)

    def step(self, actions: torch.Tensor) -> dict:
        # drop the previous step's final observations so a step without resets cannot reuse them
        self.unwrapped.extras.pop("final_obs", None)
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        actor_obs, critic_obs = self._split(obs)
        dones = terminated | truncated
        final = extras.get("final_obs")
        if final is not None:
            final_actor, final_critic = self._split(final, record_extra=False)
            true_next_actor = torch.where(dones[:, None], final_actor, actor_obs)
            true_next_critic = torch.where(dones[:, None], final_critic, critic_obs)
        else:
            true_next_actor, true_next_critic = actor_obs, critic_obs
        return {
            "obs": actor_obs,
            "critic_obs": critic_obs,
            "true_next_obs": true_next_actor,
            "true_next_critic_obs": true_next_critic,
            "rewards": self._finite(rewards.float()),
            "dones": dones,
            "truncations": truncated & ~terminated,
            "extras": extras,
        }

    def termination_flags(self, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        """Which of the named termination terms fired this step, per environment."""
        manager = self.unwrapped.termination_manager
        return {name: manager.get_term(name) for name in names if name in manager.active_terms}


class FastSACRunner:
    """Collect, store, update, log and checkpoint (holosoma's ``FastSACAgent.learn``)."""

    def __init__(self, env: IsaacLabSacEnv, cfg: FastSACCfg, log_dir: str | None, device: str):
        self.env = env
        self.cfg = cfg
        self.log_dir = log_dir
        self.alg = FastSAC(cfg, env.n_obs, env.n_critic_obs, env.n_act, env.num_envs, device)
        self.writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def save(self, path: str) -> None:
        self.alg.save(path, extra={
            "actor_obs_groups": self.env.actor_obs_groups,
            "critic_obs_groups": self.env.critic_obs_groups,
        })
        print(f"[FastSAC] saved {path}")

    def load(self, path: str, load_optimizers: bool = True) -> None:
        ckpt = self.alg.load(path, load_optimizers=load_optimizers)
        for key, mine in (("actor_obs_groups", self.env.actor_obs_groups), ("critic_obs_groups", self.env.critic_obs_groups)):
            if key in ckpt and list(ckpt[key]) != mine:
                raise ValueError(f"checkpoint {key} {ckpt[key]} != configured {mine}")
        print(f"[FastSAC] loaded {path} at iteration {self.alg.global_step}")

    def hold_out(self, fraction: float, outcomes: tuple[str, ...] = ("placed", "dropped")) -> CriticValidator:
        """Hold out the last ``fraction`` of the environments and return their validator."""
        n_holdout = int(round(fraction * self.env.num_envs))
        if not 0 < n_holdout < self.env.num_envs:
            raise ValueError(f"hold-out fraction {fraction} leaves {n_holdout} of {self.env.num_envs} environments")
        n_train = self.env.num_envs - n_holdout
        self.alg.train_env_ids = torch.arange(n_train, device=self.alg.device)
        return CriticValidator(
            torch.arange(n_train, self.env.num_envs), self.env.n_critic_obs, self.env.n_act, self.cfg.gamma,
            self.env.max_episode_length, self.alg.device, outcomes=outcomes,
        )

    def learn(
        self,
        num_iterations: int,
        behavior: BehaviorPolicy | None = None,
        validator: CriticValidator | None = None,
        gate: CriticGate | None = None,
        eval_interval: int = 100,
        reference_actions=None,
        reference_std: torch.Tensor | None = None,
        probe=None,
        probe_interval: int = 5000,
        reanchor_interval: int = 0,
        curriculum=None,
    ) -> dict | None:
        """Run ``num_iterations`` more iterations (one step of every environment each).

        Args:
            behavior: drives the environments instead of the actor (e.g. the frozen teacher).
            validator: scores the critic on held-out episodes every ``eval_interval`` iterations.
            gate: stops the run early once the validator's history passes it.
            reference_actions / reference_std: the behaviour policy's mean action and exploration
                std, for the validator's "mean beats a noisy mean" comparison.
            probe: an object with ``run(alg, env) -> dict[str, float]`` executed every
                ``probe_interval`` iterations (e.g. ``probes.ActionScaleProbe``). It takes the
                environments over; its transitions are not stored and the environments are reset after it.
            reanchor_interval: if > 0, the anchor's reference is moved to the current actor every
                this many iterations -- a trust region (bounded movement per window, unbounded in
                total) instead of a fixed leash.
            curriculum: a ``curriculum.ParamCurriculum``; updated with every logging window's
                scalars and allowed to raise its environment parameter.

        Returns the validation record (also written to ``validation.json``) when a validator is given.
        """
        cfg, alg, env = self.cfg, self.alg, self.env
        device = alg.device
        obs, critic_obs = env.reset()

        ep_return = torch.zeros(env.num_envs, device=device)
        ep_length = torch.zeros(env.num_envs, device=device)
        returns: deque[float] = deque(maxlen=100)
        lengths: deque[float] = deque(maxlen=100)
        # sums stay on the device; they are read back once per logging window
        log_sums: dict[str, torch.Tensor] = {}
        log_counts: dict[str, int] = {}
        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, int] = {}
        eval_loss_sum, eval_loss_count = 0.0, 0
        episodes_ended = 0
        collect_time = learn_time = 0.0
        window_start = time.time()
        window_iters = 0
        # the replay buffer is not checkpointed, so a resumed run refills it before updating again
        start = alg.global_step
        end = start + num_iterations
        history: list[dict[str, float]] = []
        stop_reason = "max_iterations"
        passed = False
        outcome_names = validator.outcome_names if validator is not None else ()

        while alg.global_step < end:
            t0 = time.time()
            if behavior is None:
                actions = alg.act(obs)
            else:
                actions = behavior.act(obs, critic_obs)
            step = env.step(actions)
            # a sample of the behaviour policy at the true next state (the policy-evaluation target)
            if behavior is None:
                next_actions = alg.act(step["true_next_obs"])
            else:
                next_actions = behavior.next_actions(step["true_next_obs"], step["true_next_critic_obs"])
            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": actions,
                    "critic_observations": critic_obs,
                    "next": {
                        "observations": step["true_next_obs"],
                        "critic_observations": step["true_next_critic_obs"],
                        "actions": next_actions,
                        "rewards": step["rewards"],
                        "dones": step["dones"].long(),
                        "truncations": step["truncations"].long(),
                    },
                },
                batch_size=(env.num_envs,),
                device=device,
            )
            alg.store(transition)
            if validator is not None:
                validator.record(critic_obs, actions, step["rewards"], step["dones"], step["truncations"],
                                 env.termination_flags(outcome_names))
            obs, critic_obs = step["obs"], step["critic_obs"]

            # episode bookkeeping (unscaled rewards, as the environment pays them)
            ep_return += step["rewards"]
            ep_length += 1
            done_ids = step["dones"].nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:  # the one sync per step; most steps reset something at scale
                returns.extend(ep_return[done_ids].tolist())
                lengths.extend(ep_length[done_ids].tolist())
                episodes_ended += done_ids.numel()
                ep_return[done_ids] = 0.0
                ep_length[done_ids] = 0.0
                # Isaac Lab rewrites extras["log"] only on steps with a reset and leaves it stale
                # otherwise, so it is read only here
                for key, value in step["extras"].get("log", {}).items():
                    value = torch.as_tensor(value, device=device, dtype=torch.float)
                    log_sums[key] = log_sums[key] + value if key in log_sums else value.clone()
                    log_counts[key] = log_counts.get(key, 0) + 1
            t1 = time.time()
            collect_time += t1 - t0

            # ``learning_starts`` counts iterations (holosoma assumes thousands of environments); the
            # buffer must also hold at least one batch, or a few environments' first transitions
            # would be replayed tens of times each before any new data arrives
            if alg.global_step - start > cfg.learning_starts and len(alg.rb) >= cfg.batch_size:
                for key, value in alg.update().items():
                    metric_sums[key] = metric_sums[key] + value if key in metric_sums else value.clone()
                    metric_counts[key] = metric_counts.get(key, 0) + 1
            learn_time += time.time() - t1

            alg.global_step += 1
            window_iters += 1

            if alg.global_step % cfg.logging_interval == 0 or alg.global_step == end:
                elapsed = time.time() - window_start
                scalars = {f"Loss/{k}": (metric_sums[k] / metric_counts[k]).item() for k in metric_sums}
                scalars.update({k: (log_sums[k] / log_counts[k]).item() for k in log_sums})
                if "Loss/qf_loss" in scalars:
                    eval_loss_sum += scalars["Loss/qf_loss"] * metric_counts["qf_loss"]
                    eval_loss_count += metric_counts["qf_loss"]
                if returns:
                    scalars["Train/mean_reward"] = statistics.mean(returns)
                    scalars["Train/mean_episode_length"] = statistics.mean(lengths)
                scalars["Train/episodes_ended"] = episodes_ended
                scalars["Train/buffer_transitions"] = len(alg.rb)
                scalars["Perf/total_fps"] = window_iters * env.num_envs / max(elapsed, 1e-9)
                scalars["Perf/collection_time"] = collect_time / window_iters
                scalars["Perf/learning_time"] = learn_time / window_iters
                if device.type == "cuda":
                    # the torch allocator's high-water mark; the simulator's own (Warp) memory is on top
                    scalars["Perf/torch_gpu_memory_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
                if curriculum is not None:
                    promoted = curriculum.update(scalars, alg.global_step)
                    scalars[curriculum.log_name()] = curriculum.logged_value()
                    if promoted:
                        print(f"[FastSAC] curriculum: {curriculum.param} -> {curriculum.logged_value():.4g} "
                              f"at iteration {alg.global_step}", flush=True)
                self._log(scalars, end)
                log_sums.clear(); log_counts.clear(); metric_sums.clear(); metric_counts.clear()
                episodes_ended = 0
                collect_time = learn_time = 0.0
                window_start, window_iters = time.time(), 0

            if validator is not None and alg.global_step % eval_interval == 0 and validator.count > 0:
                metrics = validator.metrics(alg.q_values, alg.action_low, alg.action_high,
                                            reference_actions, reference_std)
                metrics["qf_loss"] = eval_loss_sum / max(eval_loss_count, 1)
                metrics["iteration"] = alg.global_step
                eval_loss_sum, eval_loss_count = 0.0, 0
                history.append(metrics)
                self._log({f"Validation/{k}": v for k, v in metrics.items()}, end, title="validation")
                if gate is not None:
                    passed, checks = gate.check(history)
                    print("[FastSAC] gate:", " ".join(f"{k}={'ok' if v else 'no'}" for k, v in checks.items()), flush=True)
                    if passed:
                        stop_reason = f"gate passed at iteration {alg.global_step}"
                        break

            if reanchor_interval > 0 and cfg.anchor_coef > 0 and (alg.global_step - start) % reanchor_interval == 0:
                alg.set_anchor()

            if probe is not None and probe_interval > 0 and (alg.global_step - start) % probe_interval == 0:
                scalars = probe.run(alg, env)
                self._log(scalars, end, title="probe")
                # the probe drove the environments; start the training episodes afresh
                obs, critic_obs = env.reset()
                ep_return.zero_()
                ep_length.zero_()

            if self.log_dir and cfg.save_interval > 0 and alg.global_step % cfg.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{alg.global_step}.pt"))

        if probe is not None and probe_interval > 0 and (alg.global_step - start) % probe_interval != 0:
            self._log(probe.run(alg, env), end, title="probe (final)")
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{alg.global_step}.pt"))
        if validator is None:
            return None
        record = {
            "stop_reason": stop_reason,
            "passed": passed,
            "iterations": alg.global_step,
            "transitions_seen": alg.global_step * env.num_envs,
            "holdout_envs": int(len(validator.env_ids)),
            "gate": gate.thresholds() if gate is not None else None,
            "final": history[-1] if history else None,
            "history": history,
        }
        if self.log_dir:
            with open(os.path.join(self.log_dir, "validation.json"), "w") as f:
                json.dump(record, f, indent=2)
        print(f"[FastSAC] {stop_reason}; validation {'passed' if passed else 'not passed'}")
        return record

    def _log(self, scalars: dict[str, float], end: int, title: str = "iteration") -> None:
        step = self.alg.global_step
        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)
        width = 80
        lines = [f" FastSAC {title} {step}/{end} ".center(width, "#")]
        phases = [f"{label} {scalars[key]:.3f}" for label, key in PHASES if key in scalars]
        if phases:
            lines.append("phases: " + " | ".join(phases))
        for key in sorted(scalars):
            lines.append(f"{key:>45s}: {scalars[key]:.4f}")
        print("\n".join(lines), flush=True)
