"""Held-out validation of the critic against Monte Carlo returns, and the "good enough" gate.

A fraction of the environments is held out: their transitions never reach a gradient update. The
validator keeps their complete episodes, computes each step's discounted return-to-go ``G_t`` from
the rewards the environment actually paid, and scores the critic's ``Q(s_t, a_t)`` against it.
Time-out episodes are discarded, since their ``G`` would miss the tail.

The gate (``CriticGate``) is the plan's: held-out explained variance stable above a threshold,
small calibration bias, the critic ranking ``placed`` above ``dropped`` at the release step, the
critic responding to the action, the twin critics agreeing, and the TD loss having plateaued.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class CriticValidator:
    """Episodes of the held-out environments, with per-step returns and the episode's outcome."""

    def __init__(
        self,
        env_ids: torch.Tensor,
        n_critic_obs: int,
        n_act: int,
        gamma: float,
        max_episode_length: int,
        device: torch.device | str,
        capacity: int = 300_000,
        outcomes: tuple[str, ...] = ("placed", "dropped"),
    ):
        self.env_ids = env_ids.to(device)
        self.gamma = gamma
        self.device = torch.device(device)
        self.outcome_names = tuple(outcomes)
        k, L = len(env_ids), max_episode_length + 1
        self.ep_obs = torch.zeros(k, L, n_critic_obs, device=device)
        self.ep_act = torch.zeros(k, L, n_act, device=device)
        self.ep_rew = torch.zeros(k, L, device=device)
        self.ep_len = torch.zeros(k, dtype=torch.long, device=device)
        self.capacity = capacity
        self.store_obs = torch.zeros(capacity, n_critic_obs, device=device)
        self.store_act = torch.zeros(capacity, n_act, device=device)
        self.store_ret = torch.zeros(capacity, device=device)
        self.store_last = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.store_outcome = torch.zeros(capacity, dtype=torch.long, device=device)  # 0 = other
        self.ptr = 0
        self.count = 0
        self.episodes = 0
        self.discarded_truncated = 0
        self._rows = torch.arange(k, device=device)

    @torch.no_grad()
    def record(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        outcome_flags: dict[str, torch.Tensor],
    ) -> None:
        """Append one step (all environments given; the held-out ones are picked here)."""
        ids = self.env_ids
        t = self.ep_len.clamp_max(self.ep_obs.shape[1] - 1)
        self.ep_obs[self._rows, t] = critic_obs[ids]
        self.ep_act[self._rows, t] = actions[ids]
        self.ep_rew[self._rows, t] = rewards[ids]
        self.ep_len += 1
        done, trunc = dones[ids], truncations[ids]
        if not done.any():
            return
        for i in done.nonzero(as_tuple=False).flatten().tolist():
            length = int(self.ep_len[i])
            if trunc[i]:
                self.discarded_truncated += 1
            else:
                weights = self.gamma ** torch.arange(length, device=self.device, dtype=torch.float)
                r = self.ep_rew[i, :length]
                returns = torch.flip(torch.cumsum(torch.flip(r * weights, [0]), 0), [0]) / weights
                outcome = 0
                for j, name in enumerate(self.outcome_names):
                    if name in outcome_flags and bool(outcome_flags[name][ids[i]]):
                        outcome = j + 1
                self._push(self.ep_obs[i, :length], self.ep_act[i, :length], returns, outcome)
            self.ep_len[i] = 0

    def _push(self, obs: torch.Tensor, act: torch.Tensor, returns: torch.Tensor, outcome: int) -> None:
        length = returns.shape[0]
        idx = (self.ptr + torch.arange(length, device=self.device)) % self.capacity
        self.store_obs[idx] = obs
        self.store_act[idx] = act
        self.store_ret[idx] = returns
        self.store_last[idx] = False
        self.store_last[idx[-1]] = True
        self.store_outcome[idx] = 0
        self.store_outcome[idx[-1]] = outcome
        self.ptr = (self.ptr + length) % self.capacity
        self.count = min(self.count + length, self.capacity)
        self.episodes += 1

    @torch.no_grad()
    def metrics(
        self,
        q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        reference_actions: Callable[[torch.Tensor], torch.Tensor] | None = None,
        reference_std: torch.Tensor | None = None,
        max_rows: int = 65536,
    ) -> dict[str, float]:
        """Score the critic on the stored steps.

        Args:
            q_fn: ``(critic_obs, actions) -> (num_q, n)`` expected Q in reward units.
            action_low, action_high: the actor's bounds, for the random-action comparison.
            reference_actions: optional ``obs -> mean action`` of the behaviour policy, for the
                "mean beats a noisy mean" comparison (``reference_std`` is the noise).
        """
        n = self.count
        if n == 0:
            return {}
        idx = torch.randperm(n, device=self.device)[:max_rows]
        obs, act, ret = self.store_obs[idx], self.store_act[idx], self.store_ret[idx]
        q = q_fn(obs, act)
        qm = q.mean(0)
        err = qm - ret
        ret_std = ret.std().clamp_min(1e-6)
        out = {
            "explained_variance": (1.0 - err.var() / ret.var().clamp_min(1e-12)).item(),
            "bias": (err.mean() / ret_std).item(),
            "rmse": err.pow(2).mean().sqrt().item(),
            "twin_gap": ((q[0] - q[1]).abs().mean() / ret_std).item() if q.shape[0] > 1 else 0.0,
            "return_mean": ret.mean().item(),
            "return_std": ret_std.item(),
            "q_mean": qm.mean().item(),
            "episodes": float(self.episodes),
            "steps": float(n),
            "discarded_truncated": float(self.discarded_truncated),
        }
        # does the critic respond to the action? the stored (behaviour) action vs a uniform random one
        rand = action_low + (action_high - action_low) * torch.rand_like(act)
        out["frac_stored_beats_random"] = (qm > q_fn(obs, rand).mean(0)).float().mean().item()
        if reference_actions is not None:
            mu = reference_actions(obs)
            noise = torch.randn_like(mu) * (reference_std if reference_std is not None else 1.0)
            noisy = (mu + noise).clamp(action_low, action_high)
            q_mu, q_noisy = q_fn(obs, mu).mean(0), q_fn(obs, noisy).mean(0)
            out["frac_mean_beats_noisy"] = (q_mu > q_noisy).float().mean().item()
        # outcome separation at the release (last) step
        last = self.store_last[:n]
        for j, name in enumerate(self.outcome_names):
            rows = (last & (self.store_outcome[:n] == j + 1)).nonzero(as_tuple=False).flatten()
            out[f"{name}_episodes"] = float(rows.numel())
            if rows.numel() > 0:
                out[f"q_last_{name}"] = q_fn(self.store_obs[rows], self.store_act[rows]).mean().item()
                out[f"return_last_{name}"] = self.store_ret[rows].mean().item()
        if "q_last_placed" in out and "q_last_dropped" in out:
            out["separation"] = (out["q_last_placed"] - out["q_last_dropped"]) / out["return_std"]
        return out


class CriticGate:
    """The stop criterion over a history of validation metrics (one entry per evaluation)."""

    def __init__(
        self,
        explained_variance: float = 0.90,
        bias: float = 0.10,
        action_frac: float = 0.70,
        twin_gap: float = 0.10,
        stable_evals: int = 10,
        loss_plateau: float = 0.05,
        min_episodes: int = 200,
    ):
        self.explained_variance = explained_variance
        self.bias = bias
        self.action_frac = action_frac
        self.twin_gap = twin_gap
        self.stable_evals = stable_evals
        self.loss_plateau = loss_plateau
        self.min_episodes = min_episodes

    def thresholds(self) -> dict[str, float]:
        return dict(vars(self))

    def check(self, history: list[dict[str, float]]) -> tuple[bool, dict[str, bool]]:
        """``history`` entries need the validator's metrics plus ``qf_loss`` (mean since the last evaluation)."""
        if not history or history[-1].get("episodes", 0) < self.min_episodes:
            return False, {"enough_episodes": False}
        last = history[-1]
        recent = history[-self.stable_evals:]
        checks = {
            "enough_episodes": True,
            "ev_stable": len(recent) == self.stable_evals
            and all(h.get("explained_variance", -1.0) >= self.explained_variance for h in recent),
            "bias": abs(last.get("bias", 1.0)) <= self.bias,
            "separation": last.get("separation", -1.0) > 0.0,
            "action_sensitivity": last.get("frac_stored_beats_random", 0.0) >= self.action_frac,
            "twin_agreement": last.get("twin_gap", 1.0) <= self.twin_gap,
        }
        if len(history) >= 2 * self.stable_evals:
            prev = history[-2 * self.stable_evals:-self.stable_evals]
            mean_recent = sum(h["qf_loss"] for h in recent) / len(recent)
            mean_prev = sum(h["qf_loss"] for h in prev) / len(prev)
            checks["loss_plateau"] = abs(mean_recent - mean_prev) <= self.loss_plateau * abs(mean_prev)
        else:
            checks["loss_plateau"] = False
        return all(checks.values()), checks
