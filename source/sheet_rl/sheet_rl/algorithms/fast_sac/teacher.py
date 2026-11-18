"""A frozen rsl_rl PPO actor as a behaviour policy for FastSAC's critic warm-up and for DAgger.

The checkpoint's ``actor_state_dict`` is an rsl_rl ``MLPModel`` with a Gaussian head; the
architecture is read from the ``params/agent.yaml`` next to it. Actions are clipped to the SAC
actor's bounds before they are returned, so what is executed, stored and later imitated are the
same numbers -- a tanh student cannot represent anything beyond its bounds.
"""

from __future__ import annotations

import os

import torch
import yaml
from tensordict import TensorDict
from torch import nn


class FrozenTeacher(nn.Module):
    """``act`` samples (the user's rule: the teacher drives stochastically), ``mean`` is its label."""

    def __init__(self, checkpoint: str, obs_dim: int, obs_group: str = "policy",
                 action_low: torch.Tensor | None = None, action_high: torch.Tensor | None = None,
                 device: torch.device | str = "cpu"):
        super().__init__()
        from rsl_rl.models import MLPModel

        self.checkpoint = checkpoint
        self.obs_group = obs_group
        agent_yaml = os.path.join(os.path.dirname(checkpoint), "params", "agent.yaml")
        with open(agent_yaml) as f:
            actor_cfg = yaml.safe_load(f)["actor"]
        state = torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"]
        n_act = state["mlp." + max(k for k in state if k.startswith("mlp.") and k.endswith("weight"))[4:]].shape[0]
        cfg = {k: v for k, v in actor_cfg.items() if k != "class_name"}
        obs = TensorDict({obs_group: torch.zeros(1, obs_dim)}, batch_size=[1])
        self.model = MLPModel(obs, {"actor": [obs_group]}, "actor", n_act, **cfg)
        self.model.load_state_dict(state)
        self.model.to(device).eval().requires_grad_(False)
        self.n_act = n_act
        self.register_buffer("low", torch.full((n_act,), -torch.inf) if action_low is None else action_low.clone())
        self.register_buffer("high", torch.full((n_act,), torch.inf) if action_high is None else action_high.clone())
        self.to(device)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        out = self.model.mlp(self.model.get_latent({self.obs_group: obs}))
        self.model.distribution.update(out)
        return self.model.distribution._distribution

    @torch.no_grad()
    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic (mean) action, clipped: the DAgger label."""
        return self._dist(obs).mean.clamp(self.low, self.high)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """A clipped sample (or the clipped mean)."""
        dist = self._dist(obs)
        return (dist.mean if deterministic else dist.sample()).clamp(self.low, self.high)

    @property
    def std(self) -> torch.Tensor:
        return self.model.distribution.std_param.detach().clamp_min(1e-6)

    @torch.no_grad()
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Gaussian log-probability of (unclipped) actions; clipping is ignored, so this is approximate near a bound."""
        return self._dist(obs).log_prob(actions).sum(-1)
