"""FastSAC actor and distributional twin critic.

Ported from holosoma (https://github.com/amazon-far/holosoma, Apache-2.0),
``holosoma/agents/fast_sac/fast_sac.py``. The maths is unchanged. Two differences:

- The CNN variants are left out: this task has no image observations.
- The networks take one already-concatenated observation tensor, so no per-key index table. The
  Isaac Lab adapter concatenates the configured observation *groups* before anything reaches them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _mlp_trunk(input_dim: int, hidden_dim: int, use_layer_norm: bool, device) -> nn.Sequential:
    """The three-layer funnel both networks share: hidden, hidden/2, hidden/4, each LayerNorm + SiLU."""
    dims = (input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4)
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [
            nn.Linear(d_in, d_out, device=device),
            nn.LayerNorm(d_out, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        ]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy with a state-dependent standard deviation.

    The action is ``tanh(u) * action_scale + action_bias`` with ``u ~ N(mean(s), std(s))``, so every
    dimension is bounded to ``bias +/- scale``. ``log_std`` is squashed into
    ``[log_std_min, log_std_max]``. Both output heads start at zero, so the initial policy is the
    action bias with std ``exp((min + max) / 2)``.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        log_std_min: float,
        log_std_max: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_tanh = use_tanh
        self.net = _mlp_trunk(n_obs, hidden_dim, use_layer_norm, device)
        self.fc_mu = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)
        for layer in (self.fc_mu, self.fc_logstd):
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, 0.0)
        scale = torch.ones(n_act) if action_scale is None else torch.as_tensor(action_scale, dtype=torch.float)
        bias = torch.zeros(n_act) if action_bias is None else torch.as_tensor(action_bias, dtype=torch.float)
        self.register_buffer("action_scale", scale.to(device))
        self.register_buffer("action_bias", bias.to(device))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(deterministic action, pre-squash mean, log_std)``."""
        x = self.net(obs)
        mean = self.fc_mu(x)
        # From SpinUp / Denis Yarats: tanh into (-1, 1), then affinely into [min, max]
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        action = torch.tanh(mean) * self.action_scale + self.action_bias if self.use_tanh else mean
        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample and its log-probability (with the tanh and scale Jacobians)."""
        _, mean, log_std = self(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action)
        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob = log_prob - torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
        return action, log_prob.sum(1)

    def log_prob_of(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of given (already scaled) actions, e.g. a teacher's, under this policy.

        Not in holosoma. Actions are clipped just inside the tanh bounds first, so an action outside
        them is scored as the nearest representable one rather than as ``-inf``.
        """
        _, mean, log_std = self(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        if not self.use_tanh:
            return dist.log_prob(actions).sum(1)
        tanh_action = ((actions - self.action_bias) / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
        log_prob = dist.log_prob(torch.atanh(tanh_action))
        log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6) - torch.log(self.action_scale + 1e-6)
        return log_prob.sum(1)

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """The action to send to the environment: the squashed mean, or a squashed sample."""
        action, mean, log_std = self(obs)
        if deterministic:
            return action
        raw_action = torch.distributions.Normal(mean, log_std.exp()).rsample()
        if self.use_tanh:
            return torch.tanh(raw_action) * self.action_scale + self.action_bias
        return raw_action


@torch.compiler.disable
def _scatter_projection(
    next_dist: torch.Tensor,
    lower_indices: torch.Tensor,
    lower_mass: torch.Tensor,
    upper_indices: torch.Tensor,
    upper_mass: torch.Tensor,
) -> torch.Tensor:
    """Scatter-add the projected probability mass onto the flattened support.

    Kept eager as in holosoma: under torch.compile, Inductor can fuse the preceding softmax into the
    scatter, and that fused kernel has produced wrong projections.
    """
    proj_dist = torch.zeros_like(next_dist)
    proj_dist.view(-1).index_add_(0, lower_indices, lower_mass)
    proj_dist.view(-1).index_add_(0, upper_indices, upper_mass)
    return proj_dist


class DistributionalQNetwork(nn.Module):
    """One C51 Q-network: logits over ``num_atoms`` return values evenly spaced in ``[v_min, v_max]``."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_trunk(n_obs + n_act, hidden_dim, use_layer_norm, device),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, actions], 1))

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``rewards + bootstrap * discount * Z(s', a')`` back onto the fixed support."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        # a target landing exactly on an atom would put its mass nowhere (upper - b = b - lower = 0)
        # (both masks read the unshifted ``lower``)
        is_integer = upper == lower
        lower_mask = torch.logical_and(lower > 0, is_integer)
        upper_mask = torch.logical_and(lower == 0, is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=rewards.device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        max_index = next_dist.numel() - 1
        lower_indices = torch.clamp((lower + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper + offset).view(-1), 0, max_index)
        return _scatter_projection(
            next_dist,
            lower_indices,
            (next_dist * (upper.float() - b)).view(-1),
            upper_indices,
            (next_dist * (b - lower.float())).view(-1),
        )


class Critic(nn.Module):
    """``num_q_networks`` independent distributional Q-networks sharing one return support."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.n_obs = n_obs
        self.n_act = n_act
        self.num_q_networks = num_q_networks
        self.qnets = nn.ModuleList([
            DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, use_layer_norm, device)
            for _ in range(num_q_networks)
        ])
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Logits, shape ``(num_q_networks, batch, num_atoms)``."""
        return torch.stack([qnet(obs, actions) for qnet in self.qnets], dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Each network's own projected target distribution, shape ``(num_q_networks, batch, num_atoms)``."""
        return torch.stack(
            [qnet.projection(obs, actions, rewards, bootstrap, discount, self.q_support) for qnet in self.qnets],
            dim=0,
        )

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Expected return under a distribution over the support."""
        return torch.sum(probs * self.q_support, dim=-1)

    def q_values(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Expected Q of every network, shape ``(num_q_networks, batch)``. Not in holosoma; for evaluation."""
        return self.get_value(F.softmax(self(obs, actions), dim=-1))
