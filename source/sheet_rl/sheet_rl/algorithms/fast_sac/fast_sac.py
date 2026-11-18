"""FastSAC: soft actor-critic with a distributional twin critic, tuned for massively parallel simulation.

Ported from holosoma (https://github.com/amazon-far/holosoma, Apache-2.0),
``holosoma/agents/fast_sac/fast_sac_agent.py`` (``_update_main``, ``_update_pol``, batch preparation,
target update, save/load). The update maths is unchanged:

- **Critic:** each of the ``num_q_networks`` C51 networks is trained by cross-entropy toward its own
  target network's projection of ``r - gamma^n * alpha * log pi(a'|s') + gamma^n * Z(s', a')``,
  with ``a' ~ pi(.|s')``. Bootstrapping is cut only on true terminations; time-outs bootstrap.
- **Actor:** minimises ``alpha * log pi(a|s) - mean_k Q_k(s, a)``. Note the mean over networks,
  not the minimum.
- **Temperature:** ``alpha`` is auto-tuned toward the entropy target ``-n_act * ratio``, from the
  same next-state log-probabilities the critic target used.

Left out: multi-GPU gradient all-reduce, symmetry augmentation, the CNN encoder and ONNX export.
Added: ``train_actor=False`` for critic-only training, and a reward scale (see ``FastSACCfg``).
Changed:

- Each iteration's samples are shuffled before being split into batches. holosoma's buffer returns
  them grouped by environment, so its contiguous split gives every update only ``1/num_updates``
  of the environments.
- With ``policy_frequency == 1`` the actor trains on every update. holosoma's ``i % f == 1`` never
  fires for ``f = 1``.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn, optim
from torch.amp import GradScaler

from .buffers import EmpiricalNormalization, IdentityNormalization, SimpleReplayBuffer
from .config import FastSACCfg
from .networks import Actor, Critic


class FastSAC:
    """Networks, optimisers, replay buffer and the update step. Environment-agnostic.

    Args:
        cfg: Hyperparameters.
        n_obs: Actor input size (the concatenated actor observation groups).
        n_critic_obs: Critic input size (the concatenated critic observation groups), excluding the action.
        n_act: Action size.
        num_envs: Parallel environments feeding the buffer; 1 for single-environment training.
        device: Torch device.
    """

    def __init__(self, cfg: FastSACCfg, n_obs: int, n_critic_obs: int, n_act: int, num_envs: int,
                 device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_obs, self.n_critic_obs, self.n_act, self.num_envs = n_obs, n_critic_obs, n_act, num_envs
        self.global_step = 0

        def per_dim(values, default):
            if values is None:
                return torch.full((n_act,), default, dtype=torch.float)
            if len(values) != n_act:
                raise ValueError(f"expected {n_act} per-dimension values, got {len(values)}: {values}")
            return torch.as_tensor(values, dtype=torch.float)

        action_scale = per_dim(cfg.action_scale, 1.0) if cfg.use_tanh else torch.ones(n_act)
        action_bias = per_dim(cfg.action_bias, 0.0)

        self.actor = Actor(
            n_obs, n_act, cfg.actor_hidden_dim, cfg.log_std_min, cfg.log_std_max, cfg.use_tanh,
            cfg.use_layer_norm, action_scale, action_bias, self.device,
        )

        def make_critic() -> Critic:
            return Critic(
                n_critic_obs, n_act, cfg.num_atoms, cfg.v_min, cfg.v_max, cfg.critic_hidden_dim,
                cfg.use_layer_norm, cfg.num_q_networks, self.device,
            )

        self.qnet = make_critic()
        self.qnet_target = make_critic()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.requires_grad_(False)

        self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=self.device)
        self.target_entropy = -n_act * cfg.target_entropy_ratio

        if cfg.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(n_obs, self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(n_critic_obs, self.device)
        else:
            self.obs_normalizer = IdentityNormalization()
            self.critic_obs_normalizer = IdentityNormalization()
        if cfg.policy_evaluation and cfg.train_actor:
            raise ValueError("policy_evaluation evaluates the behaviour policy; set train_actor = False")
        self.train_env_ids: torch.Tensor | None = None
        """Environments whose transitions are sampled for updates (None: all); the rest are held out."""
        self.anchor_actor: Actor | None = None
        """Frozen copy of a reference actor that ``cfg.anchor_coef`` pulls the actor toward."""

        fused = self.device.type == "cuda"
        self.q_optimizer = optim.AdamW(self.qnet.parameters(), lr=cfg.critic_learning_rate,
                                       weight_decay=cfg.weight_decay, fused=fused, betas=(0.9, 0.95))
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=cfg.actor_learning_rate,
                                           weight_decay=cfg.weight_decay, fused=fused, betas=(0.9, 0.95))
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=cfg.alpha_learning_rate, fused=fused,
                                           betas=(0.9, 0.95))
        self.scaler = GradScaler(self.device.type, enabled=cfg.amp)

        self.rb = SimpleReplayBuffer(
            n_env=num_envs, buffer_size=cfg.buffer_size, n_obs=n_obs, n_act=n_act, n_critic_obs=n_critic_obs,
            n_steps=cfg.num_steps, gamma=cfg.gamma, device=self.device,
        )

        # torch.compile is opt-in (default mode, no CUDA graphs); the eager functions are the reference
        self._update_main_fn = self._update_main
        self._update_pol_fn = self._update_pol
        self._explore_fn = self.actor.explore
        if cfg.compile:
            self._update_main_fn = torch.compile(self._update_main)
            self._update_pol_fn = torch.compile(self._update_pol)
            self._explore_fn = torch.compile(self.actor.explore)

    # ------------------------------------------------------------------------------------------
    # acting

    @contextmanager
    def _maybe_amp(self):
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=self.cfg.amp):
            yield

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Action for raw (unnormalised) actor observations, already scaled to the environment's units."""
        with self._maybe_amp():
            action = self._explore_fn(self.obs_normalizer(obs, update=False), deterministic=deterministic)
        return action.float()

    def store(self, transition: TensorDict) -> None:
        """Append one step of every environment (keys as in ``SimpleReplayBuffer.extend``); rewards are scaled here."""
        transition["next"]["rewards"] = transition["next"]["rewards"] * self.cfg.reward_scale
        self.rb.extend(transition)

    @torch.no_grad()
    def q_values(self, critic_obs: torch.Tensor, actions: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Expected Q of every critic for *raw* critic observations, in reward units: ``(num_q, n)``."""
        net = self.qnet_target if target else self.qnet
        with self._maybe_amp():
            q = net.q_values(self.critic_obs_normalizer(critic_obs, update=False), actions)
        return q.float() / self.cfg.reward_scale

    @property
    def action_low(self) -> torch.Tensor:
        return self.actor.action_bias - self.actor.action_scale

    @property
    def action_high(self) -> torch.Tensor:
        return self.actor.action_bias + self.actor.action_scale

    # ------------------------------------------------------------------------------------------
    # updates (holosoma's _update_main / _update_pol)

    def _update_main(self, data: TensorDict) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                discount = cfg.gamma ** data["next"]["effective_n_steps"]
                if cfg.policy_evaluation:
                    # the stored next action is a sample of the behaviour policy at s'; no entropy term
                    next_state_actions = data["next"]["actions"]
                    next_state_log_probs = torch.zeros_like(rewards)
                    entropy_term = torch.zeros_like(rewards)
                else:
                    next_state_actions, next_state_log_probs = self.actor.get_actions_and_log_probs(next_observations)
                    entropy_term = discount * bootstrap * self.log_alpha.exp() * next_state_log_probs
                target_distributions = self.qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - entropy_term,
                    bootstrap,
                    discount,
                )
                target_values = self.qnet_target.get_value(target_distributions)
                if cfg.target_q_min and target_distributions.shape[0] > 1:
                    # clipped double-Q, distributional: every critic's target is the projected
                    # distribution of the target critic that expects less for this sample
                    lower = target_values.argmin(dim=0)
                    rows = torch.arange(target_values.shape[1], device=self.device)
                    chosen = target_distributions[lower, rows]
                    target_distributions = chosen.unsqueeze(0).expand_as(target_distributions)
                    target_values = target_values.min(dim=0).values.unsqueeze(0).expand_as(target_values)

            q_outputs = self.qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.q_optimizer)
        if cfg.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=cfg.max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.q_optimizer)
        self.scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if cfg.use_autotune and cfg.train_actor and not cfg.policy_evaluation:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_optimizer)
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()

        return {
            "buffer_rewards": rewards.mean().detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "qf_loss": qf_loss.detach(),
            "qf_max": target_values.max().detach(),
            "qf_min": target_values.min().detach(),
            "alpha_loss": alpha_loss.detach(),
        }

    def _update_pol(self, data: TensorDict) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        with self._maybe_amp():
            actions, log_probs = self.actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = self.actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()
            q_values = self.qnet.get_value(F.softmax(self.qnet(data["critic_observations"], actions), dim=-1))
            q_for_actor = q_values.min(dim=0).values if cfg.actor_q_min else q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - q_for_actor).mean()
            anchor_kl = torch.tensor(0.0, device=self.device)
            anchor_shift = torch.tensor(0.0, device=self.device)
            if cfg.anchor_coef > 0 and self.anchor_actor is not None:
                _, mean_pre, _ = self.actor(data["observations"])
                with torch.no_grad():
                    _, ref_mean, ref_log_std = self.anchor_actor(data["observations"])
                z = (mean_pre - ref_mean) / ref_log_std.exp()
                anchor_kl = 0.5 * z.pow(2).sum(-1).mean()
                anchor_shift = z.detach().pow(2).mean().sqrt()  # RMS shift, in reference-std units
                actor_loss = actor_loss + cfg.anchor_coef * anchor_kl

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        if cfg.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        # the critic received gradients through Q(s, pi(s)); they belong to no step and are cleared
        # before the next critic update anyway (zero_grad(set_to_none=True))
        return {
            "actor_grad_norm": actor_grad_norm.detach(),
            "actor_loss": actor_loss.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
            "anchor_kl": anchor_kl.detach(),
            "anchor_shift_std": anchor_shift.detach(),
        }

    def _prepare_batches(self) -> list[TensorDict]:
        """Sample all of this iteration's batches at once, normalise once, then split (holosoma's batching)."""
        n_train = self.num_envs if self.train_env_ids is None else len(self.train_env_ids)
        per_env = max(self.cfg.batch_size // n_train, 1)
        data = self.rb.sample(per_env * self.cfg.num_updates, env_ids=self.train_env_ids)
        update_actor_norm = not self.cfg.freeze_obs_normalizer
        data["observations"] = self.obs_normalizer(data["observations"], update=update_actor_norm)
        data["next"]["observations"] = self.obs_normalizer(data["next"]["observations"], update=update_actor_norm)
        data["critic_observations"] = self.critic_obs_normalizer(data["critic_observations"])
        data["next"]["critic_observations"] = self.critic_obs_normalizer(data["next"]["critic_observations"])
        # rb.sample returns [env0 all, env1 all, ...]; shuffle so each split mixes every environment
        data = data[torch.randperm(data.batch_size[0], device=self.device)]
        n = per_env * n_train
        return [data[i * n:(i + 1) * n] for i in range(self.cfg.num_updates)]

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        src = [p.data for p in self.qnet.parameters()]
        tgt = [p.data for p in self.qnet_target.parameters()]
        torch._foreach_mul_(tgt, 1.0 - self.cfg.tau)
        torch._foreach_add_(tgt, src, alpha=self.cfg.tau)

    def update(self) -> dict[str, torch.Tensor]:
        """One iteration's gradient updates. Call after ``store``.

        Returns the mean of each metric as a 0-d tensor on the device, so that logging can
        accumulate them without a GPU sync per iteration.
        """
        cfg = self.cfg
        freq = cfg.policy_frequency
        metrics: dict[str, list[torch.Tensor]] = {}
        for i, data in enumerate(self._prepare_batches()):
            out = self._update_main_fn(data)
            # holosoma's schedule: with several updates per iteration the actor trains on updates
            # 1, 1 + f, 1 + 2f, ... (every update when f = 1); with one, on every f-th iteration
            if cfg.num_updates > 1:
                actor_turn = i % freq == (1 if freq > 1 else 0)
            else:
                actor_turn = self.global_step % freq == 0
            if cfg.train_actor and actor_turn:
                out = {**out, **self._update_pol_fn(data)}
            out["alpha_value"] = self.log_alpha.exp().detach().mean()
            for key, value in out.items():
                metrics.setdefault(key, []).append(value)
            self._soft_update_target()
        return {key: torch.stack(values).float().mean() for key, values in metrics.items()}

    # ------------------------------------------------------------------------------------------
    # checkpoints

    def state_dict(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "qnet_state_dict": self.qnet.state_dict(),
            "qnet_target_state_dict": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_normalizer_state": self.obs_normalizer.state_dict(),
            "critic_obs_normalizer_state": self.critic_obs_normalizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "dims": {"n_obs": self.n_obs, "n_critic_obs": self.n_critic_obs, "n_act": self.n_act},
            "cfg": self.cfg.to_dict(),
            "anchor_actor_state_dict": None if self.anchor_actor is None else self.anchor_actor.state_dict(),
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({**self.state_dict(), **(extra or {})}, path)

    def set_anchor(self) -> None:
        """Freeze a copy of the current actor as the anchor's reference."""
        self.anchor_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)

    def load_actor(self, path: str) -> dict:
        """Take only the actor and its observation normaliser from a checkpoint (e.g. a DAgger student),
        leaving the critics, alpha and optimisers as they are. The actor optimiser is reset."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        dims = ckpt.get("dims", {})
        if dims and (dims["n_obs"], dims["n_act"]) != (self.n_obs, self.n_act):
            raise ValueError(f"actor checkpoint dims {dims} do not match n_obs={self.n_obs}, n_act={self.n_act}")
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        # the normaliser is part of the pretrained policy: its statistics stay as trained
        self.cfg.freeze_obs_normalizer = True
        self.set_anchor()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_learning_rate,
                                           weight_decay=self.cfg.weight_decay,
                                           fused=self.device.type == "cuda", betas=(0.9, 0.95))
        return ckpt

    def load(self, path: str, load_optimizers: bool = True) -> dict:
        """Restore a checkpoint written by ``save``; returns the raw checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        dims = ckpt.get("dims", {})
        mine = {"n_obs": self.n_obs, "n_critic_obs": self.n_critic_obs, "n_act": self.n_act}
        if dims and dims != mine:
            raise ValueError(f"checkpoint dims {dims} do not match this setup {mine}")
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qnet.load_state_dict(ckpt["qnet_state_dict"])
        self.qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
        self.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        if ckpt.get("anchor_actor_state_dict") is not None:
            self.set_anchor()
            self.anchor_actor.load_state_dict(ckpt["anchor_actor_state_dict"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
            self.q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
            # Optimizer.load_state_dict also restores the *saved* hyperparameters (lr, weight decay,
            # betas); this run's configured values win, or a fine-tune silently inherits the
            # warm-up's learning rates
            for optimizer, lr in ((self.actor_optimizer, self.cfg.actor_learning_rate),
                                  (self.q_optimizer, self.cfg.critic_learning_rate),
                                  (self.alpha_optimizer, self.cfg.alpha_learning_rate)):
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["betas"] = (0.9, 0.95)
                    if optimizer is not self.alpha_optimizer:
                        group["weight_decay"] = self.cfg.weight_decay
            self.scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
            self.global_step = ckpt["global_step"]
        return ckpt

    def learning_rates(self) -> dict[str, float]:
        """The learning rates the optimisers will actually use."""
        return {"actor": self.actor_optimizer.param_groups[0]["lr"], "critic": self.q_optimizer.param_groups[0]["lr"],
                "alpha": self.alpha_optimizer.param_groups[0]["lr"]}
