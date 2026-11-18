"""The FastSAC actor as an rsl_rl model, so rsl_rl's PPO (``ReferencePPO``) can fine-tune a FastSAC actor as it is.

Same network (:class:`~.networks.Actor`), same frozen observation normaliser, same tanh-squashed and scaled actions,
and exact log-probabilities -- no distillation step between the SAC-trained actor and PPO. Use it through
:class:`RslRlFastSACActorCfg` as the runner's ``actor``; ``scripts/fastsac_actor_to_rsl_rl.py`` writes a FastSAC
checkpoint as an rsl_rl checkpoint for ``--resume``.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .buffers import EmpiricalNormalization
from .networks import Actor


class TanhGaussianDistribution(nn.Module):
    """rsl_rl's distribution interface over the actor's state-dependent, tanh-squashed Gaussian.

    The parameters are the pre-squash ``(mean, log_std)``. ``tanh`` and the per-dimension scale are bijective, so
    the KL between two squashed Gaussians is the KL between their pre-squash Gaussians. The entropy is the
    pre-squash Gaussian's (the squashed one has no closed form); PPO only logs it here (entropy_coef 0).

    The model sets the parameters on every forward pass, so :meth:`update` does nothing: ReferencePPO's
    ``reference_actor.distribution.update(reference_actor(obs))`` leaves the reference's own parameters in place.
    No learnable parameters: exploration is the actor's own std head.
    """

    def __init__(self, action_scale: torch.Tensor, action_bias: torch.Tensor):
        super().__init__()
        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_bias", action_bias.clone())
        self.mean_u: torch.Tensor | None = None
        self.log_std: torch.Tensor | None = None

    def set(self, mean_u: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean_u, self.log_std = mean_u, log_std

    def update(self, output: torch.Tensor) -> None:
        pass

    def squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u) * self.action_scale + self.action_bias

    @property
    def mean(self) -> torch.Tensor:
        """The deterministic action: the squashed pre-squash mean (what FastSAC plays with ``deterministic``)."""
        return self.squash(self.mean_u)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    @property
    def std_param(self) -> torch.Tensor:
        """Mean pre-squash std over the last batch (ReferencePPO prints it on load); NaN before any forward pass."""
        if self.log_std is None:
            return torch.full_like(self.action_scale, float("nan"))
        return self.log_std.exp().mean(0)

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean_u, self.log_std

    @property
    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum(-1)

    def sample(self) -> torch.Tensor:
        return self.squash(self.mean_u + self.log_std.exp() * torch.randn_like(self.mean_u))

    def deterministic_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """As :meth:`Actor.log_prob_of`: actions are clipped just inside the tanh bounds before ``atanh``."""
        squashed = ((actions - self.action_bias) / self.action_scale).clamp(-1 + 1e-6, 1 - 1e-6)
        log_prob = torch.distributions.Normal(self.mean_u, self.log_std.exp()).log_prob(torch.atanh(squashed))
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6) - torch.log(self.action_scale + 1e-6)
        return log_prob.sum(-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """KL(old || new), summed over the action dimensions (rsl_rl's convention)."""
        old_mean, old_log_std = old_params
        new_mean, new_log_std = new_params
        old = torch.distributions.Normal(old_mean, old_log_std.exp())
        new = torch.distributions.Normal(new_mean, new_log_std.exp())
        return torch.distributions.kl_divergence(old, new).sum(-1)


class FastSACActorModel(nn.Module):
    """rsl_rl model around :class:`Actor` (built by rsl_rl as ``class(obs, obs_groups, obs_set, output_dim, **cfg)``).

    The FastSAC observation normaliser is part of the policy and stays frozen; to rsl_rl the model reports
    ``obs_normalization = False``, which is also what ReferencePPO requires of the actor.

    ``init_std`` replaces the std head with a state-independent pre-squash std per action dimension, starting at
    ``init_std``: learned (the parameter ``learned_log_std``, clamped to the head's log-std bounds) with ``learn_std``,
    else constant. The head's weights stay in the state dict, so FastSAC checkpoints load unchanged; checkpoints
    without ``learned_log_std`` start it at ``init_std``.
    """

    is_recurrent: bool = False

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int,
                 hidden_dim: int = 512, log_std_min: float = -5.0, log_std_max: float = 0.0,
                 use_layer_norm: bool = True, action_scale: list[float] | None = None,
                 action_bias: list[float] | None = None, init_std: float | None = None, learn_std: bool = True,
                 **unused) -> None:
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
        scale = torch.as_tensor(action_scale if action_scale is not None else [1.0] * output_dim, dtype=torch.float)
        bias = torch.as_tensor(action_bias if action_bias is not None else [0.0] * output_dim, dtype=torch.float)
        if scale.numel() != output_dim or bias.numel() != output_dim:
            raise ValueError(f"action_scale/action_bias need {output_dim} values, got {scale.numel()}/{bias.numel()}")
        self.actor = Actor(self.obs_dim, output_dim, hidden_dim, log_std_min, log_std_max, True, use_layer_norm, scale, bias)
        self.input_normalizer = EmpiricalNormalization(self.obs_dim, "cpu")
        self.obs_normalization = False
        self.distribution = TanhGaussianDistribution(scale, bias)
        if init_std is not None and not math.exp(log_std_min) <= init_std <= math.exp(log_std_max):
            raise ValueError(f"init_std must lie in [{math.exp(log_std_min):.3g}, {math.exp(log_std_max):.3g}], got {init_std}")
        self.fixed_log_std = math.log(init_std) if init_std is not None and not learn_std else None
        self.learned_log_std = (nn.Parameter(torch.full((output_dim,), math.log(init_std)))
                                if init_std is not None and learn_std else None)

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        x = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        _, mean_u, log_std = self.actor(self.input_normalizer(x, update=False))
        if self.learned_log_std is not None:
            log_std = self.learned_log_std.clamp(self.actor.log_std_min, self.actor.log_std_max).expand_as(mean_u)
        elif self.fixed_log_std is not None:
            log_std = torch.full_like(mean_u, self.fixed_log_std)
        self.distribution.set(mean_u, log_std)
        return self.distribution.sample() if stochastic_output else self.distribution.mean

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.learned_log_std is not None and "learned_log_std" not in state_dict:
            state_dict = {**state_dict, "learned_log_std": self.learned_log_std.detach().clone()}
        return super().load_state_dict(state_dict, strict, assign)

    def load_fastsac(self, checkpoint: dict) -> None:
        """Take the actor and its observation normaliser from a FastSAC checkpoint dict."""
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.input_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    def as_jit(self) -> nn.Module:
        return _DeterministicFastSACActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxFastSACActor(self, verbose)


class _DeterministicFastSACActor(nn.Module):
    """The deterministic action (the squashed mean) of a :class:`FastSACActorModel`, for TorchScript export.

    ``play.py`` exports the policy before playing; this is what it writes to ``exported/policy.pt``.
    """

    def __init__(self, model: FastSACActorModel) -> None:
        super().__init__()
        norm = model.input_normalizer
        self.register_buffer("obs_mean", norm._mean.detach().clone())
        self.register_buffer("obs_scale", (norm._std + norm.eps).detach().clone())
        self.net = copy.deepcopy(model.actor.net)
        self.fc_mu = copy.deepcopy(model.actor.fc_mu)
        self.register_buffer("action_scale", model.actor.action_scale.detach().clone())
        self.register_buffer("action_bias", model.actor.action_bias.detach().clone())
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_u = self.fc_mu(self.net((x - self.obs_mean) / self.obs_scale))
        return torch.tanh(mean_u) * self.action_scale + self.action_bias

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFastSACActor(_DeterministicFastSACActor):
    """As :class:`_DeterministicFastSACActor`, with the hooks rsl_rl's ONNX export calls."""

    is_recurrent: bool = False

    def __init__(self, model: FastSACActorModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


@configclass
class RslRlFastSACActorCfg(RslRlMLPModelCfg):
    """Runner ``actor`` entry for :class:`FastSACActorModel`; the architecture must match the FastSAC checkpoint."""

    class_name: str = "sheet_rl.algorithms.fast_sac.rsl_rl_actor:FastSACActorModel"
    hidden_dims: list[int] = [512]
    """Unused (the FastSAC trunk is ``hidden_dim``, ``/2``, ``/4``); set because the base config requires it."""
    activation: str = "silu"
    hidden_dim: int = 512
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    use_layer_norm: bool = True
    action_scale: list[float] | None = None
    action_bias: list[float] | None = None
    init_std: float | None = None
    """State-independent pre-squash exploration std in place of the actor's std head (None: the head's own std)."""
    learn_std: bool = True
    """With ``init_std``: learn it (from ``init_std``) or keep it constant."""


def rsl_rl_checkpoint_from_fastsac(fastsac: dict, runner_cfg, critic_dim: int, source: str = "") -> dict:
    """An rsl_rl checkpoint whose actor is ``fastsac``'s actor (and frozen normaliser) and whose critic is fresh.

    Built for ``runner_cfg`` (its ``obs_groups``, ``actor`` and ``critic`` entries), so ``--resume`` loads it strictly.
    ReferencePPO treats it as a new phase: fresh optimiser and iteration count, and the loaded actor as reference.
    """
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.models import MLPModel

    # as train.py does before building the runner: drops the model configs' legacy fields (``stochastic``, ...)
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, version("rsl-rl-lib"))
    actor_groups, critic_groups = runner_cfg.obs_groups["actor"], runner_cfg.obs_groups["critic"]
    if list(fastsac.get("actor_obs_groups", actor_groups)) != list(actor_groups):
        raise ValueError(f"FastSAC actor reads {fastsac['actor_obs_groups']}, the runner's actor reads {actor_groups}")
    if len(critic_groups) != 1:
        raise ValueError(f"expected one critic observation group, got {critic_groups}")
    dims = fastsac["dims"]
    obs = TensorDict({actor_groups[0]: torch.zeros(1, dims["n_obs"]), critic_groups[0]: torch.zeros(1, critic_dim)},
                     batch_size=[1])
    actor_kwargs = runner_cfg.actor.to_dict()
    actor_kwargs.pop("class_name")
    actor = FastSACActorModel(obs, runner_cfg.obs_groups, "actor", dims["n_act"], **actor_kwargs)
    actor.load_fastsac(fastsac)
    critic_kwargs = runner_cfg.critic.to_dict()
    critic_kwargs.pop("class_name")
    critic = MLPModel(obs, runner_cfg.obs_groups, "critic", 1, **critic_kwargs)
    return {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "iter": 0,
            "infos": {"converted_from": source}}
