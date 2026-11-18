"""Per-environment circular replay buffer (with n-step returns) and running observation normalisation.

Ported from holosoma (https://github.com/amazon-far/holosoma, Apache-2.0),
``holosoma/agents/fast_sac/fast_sac_utils.py``. The maths is unchanged. The multi-GPU all-reduce in
the normaliser is dropped, since this project trains on one GPU. Added: the *next* action is stored
too (a sample of the behaviour policy at s', used by the policy-evaluation target), and ``sample``
can be restricted to a subset of environments so that others can be held out.

The buffer holds ``buffer_size`` transitions *per environment*. With 1024 parallel environments,
1024 slots each is a million transitions. A single environment therefore needs a correspondingly
large ``buffer_size`` -- see ``FastSACCfg.single_env_preset``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class SimpleReplayBuffer(nn.Module):
    """Circular buffer of ``(n_env, buffer_size)`` transitions with asymmetric (actor/critic) observations."""

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        def zeros(*shape, dtype=torch.float):
            return torch.zeros((n_env, buffer_size, *shape), device=device, dtype=dtype)

        self.observations = zeros(n_obs)
        self.actions = zeros(n_act)
        self.rewards = zeros()
        self.dones = zeros(dtype=torch.long)
        self.truncations = zeros(dtype=torch.long)
        self.next_observations = zeros(n_obs)
        self.next_actions = zeros(n_act)
        self.critic_observations = zeros(n_critic_obs)
        self.next_critic_observations = zeros(n_critic_obs)
        self.ptr = 0

    def __len__(self) -> int:
        """Transitions currently stored, over all environments."""
        return min(self.ptr, self.buffer_size) * self.n_env

    def extend(self, tensor_dict: TensorDict) -> None:
        """Append one step of every environment. ``next.actions`` is optional (zeros when absent)."""
        ptr = self.ptr % self.buffer_size
        nxt = tensor_dict["next"]
        self.observations[:, ptr] = tensor_dict["observations"]
        self.actions[:, ptr] = tensor_dict["actions"]
        self.rewards[:, ptr] = nxt["rewards"]
        self.dones[:, ptr] = nxt["dones"]
        self.truncations[:, ptr] = nxt["truncations"]
        self.next_observations[:, ptr] = nxt["observations"]
        self.next_actions[:, ptr] = nxt["actions"] if "actions" in nxt.keys() else 0.0
        self.critic_observations[:, ptr] = tensor_dict["critic_observations"]
        self.next_critic_observations[:, ptr] = nxt["critic_observations"]
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int, env_ids: torch.Tensor | None = None) -> TensorDict:
        """Sample ``batch_size`` transitions from *each* environment (of ``env_ids`` if given).

        Rows are ordered environment by environment: ``[env0 x batch_size, env1 x batch_size, ...]``.
        """
        n_env = self.n_env

        def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            return torch.gather(x, 1, idx)

        if self.n_steps == 1:
            indices = torch.randint(0, min(self.buffer_size, self.ptr), (n_env, batch_size), device=self.device)
            fields = {
                "observations": gather(self.observations, indices),
                "actions": gather(self.actions, indices),
                "critic_observations": gather(self.critic_observations, indices),
                "next_observations": gather(self.next_observations, indices),
                "next_critic_observations": gather(self.next_critic_observations, indices),
                "next_actions": gather(self.next_actions, indices),
                "rewards": gather(self.rewards, indices),
                "dones": gather(self.dones, indices),
                "truncations": gather(self.truncations, indices),
            }
            fields["effective_n_steps"] = torch.ones_like(fields["dones"])
        else:
            if self.ptr >= self.buffer_size:
                # A full buffer wraps, so an n-step window could run from the newest transition into
                # the oldest. Mark the newest one truncated (unless it is a real terminal) for the
                # duration of the sample, which cuts any window there, and roll it back afterwards.
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1])
                indices = torch.randint(0, self.buffer_size, (n_env, batch_size), device=self.device)
            else:
                # not yet full: every window must end inside the data written so far
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(0, max_start_idx, (n_env, batch_size), device=self.device)

            # [n_env, batch_size, n_steps] indices of each sample's window
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size

            def window(x: torch.Tensor) -> torch.Tensor:
                return torch.gather(x.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)

            all_rewards = window(self.rewards)
            all_dones = window(self.dones)
            all_truncations = window(self.truncations)

            # zero every reward after the first done (the first reward itself is never masked)
            all_dones_shifted = torch.cat([torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2)
            done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=2)
            discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
            n_step_rewards = (all_rewards * done_masks * discounts.view(1, 1, -1)).sum(dim=2)

            # the window ends at its first done or truncation, else at its last step
            first_done = torch.argmax((all_dones > 0).float(), dim=2)
            first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
            first_done = torch.where(all_dones.sum(dim=2) == 0, self.n_steps - 1, first_done)
            first_trunc = torch.where(all_truncations.sum(dim=2) == 0, self.n_steps - 1, first_trunc)
            final_indices = torch.minimum(first_done, first_trunc)
            final = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(-1)

            fields = {
                "observations": gather(self.observations, indices),
                "actions": gather(self.actions, indices),
                "critic_observations": gather(self.critic_observations, indices),
                "next_observations": gather(self.next_observations, final),
                "next_critic_observations": gather(self.next_critic_observations, final),
                "next_actions": gather(self.next_actions, final),
                "rewards": n_step_rewards,
                "dones": gather(self.dones, final),
                "truncations": gather(self.truncations, final),
                "effective_n_steps": done_masks.sum(2),
            }
            if self.ptr >= self.buffer_size:
                self.truncations[:, current_pos - 1] = curr_truncations  # type: ignore[possibly-undefined]

        if env_ids is not None:
            fields = {key: value[env_ids] for key, value in fields.items()}

        def flat(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(-1, value.shape[-1]) if value.dim() == 3 else value.reshape(-1)

        rows = fields["rewards"].numel()
        return TensorDict(
            {
                "observations": flat(fields["observations"]),
                "actions": flat(fields["actions"]),
                "critic_observations": flat(fields["critic_observations"]),
                "next": {
                    "observations": flat(fields["next_observations"]),
                    "critic_observations": flat(fields["next_critic_observations"]),
                    "actions": flat(fields["next_actions"]),
                    "rewards": flat(fields["rewards"]),
                    "dones": flat(fields["dones"]),
                    "truncations": flat(fields["truncations"]),
                    "effective_n_steps": flat(fields["effective_n_steps"]),
                },
            },
            batch_size=rows,
        )


class EmpiricalNormalization(nn.Module):
    """Running mean/variance normalisation of observations; the same update as holosoma's and rsl_rl's.

    ``until`` freezes the statistics once that many samples have been seen.
    """

    def __init__(self, shape: int | tuple[int, ...], device: torch.device | str, eps: float = 1e-2,
                 until: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m2 = self._var * self.count + batch_var * batch_size + delta2.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self._std + self.eps) + self._mean


class IdentityNormalization(nn.Module):
    """Stand-in for ``EmpiricalNormalization`` when observation normalisation is off; same call signature."""

    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        return x
