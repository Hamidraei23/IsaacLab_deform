"""FastSAC configuration: holosoma's ``FastSACConfig`` defaults, plus the fields this project adds."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class FastSACCfg:
    """FastSAC hyperparameters.

    The defaults are holosoma's (``holosoma/config_types/algo.py``, ``FastSACConfig``), tuned for
    thousands of parallel environments: a small per-environment buffer, a large global batch, and
    several gradient updates per environment step. ``single_env_preset`` switches to classic
    one-environment SAC settings; ``scripts/train_sac.py --single_env`` applies it.

    One "iteration" is one step of every environment, as in holosoma.
    """

    # -- run bookkeeping (not in holosoma's config)
    seed: int = 42
    device: str = "cuda:0"
    experiment_name: str = MISSING
    run_name: str = ""
    max_iterations: int = 25000
    """Environment steps (of every environment at once). holosoma: ``num_learning_iterations``."""
    save_interval: int = 1000
    logging_interval: int = 100

    # -- observation groups (holosoma: actor_obs_keys / critic_obs_keys)
    actor_obs_groups: list[str] = ["policy"]
    """Isaac Lab observation groups concatenated, in order, as the actor's input."""
    critic_obs_groups: list[str] = ["policy"]
    """Isaac Lab observation groups concatenated, in order, as the critics' input (asymmetric if different)."""

    # -- optimisation
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    """Global gradient-norm clip; 0 disables it."""
    batch_size: int = 8192
    """Global batch per gradient update, split evenly across environments."""
    num_updates: int = 8
    """Critic updates per iteration."""
    policy_frequency: int = 4
    """One actor update per this many critic updates."""
    learning_starts: int = 10
    """Iterations of collection before the first update."""
    train_actor: bool = True
    """False: critic-only training (actor and alpha frozen) -- a critic warm-up. Not in holosoma."""
    policy_evaluation: bool = False
    """Evaluate the *behaviour* policy whose transitions are stored, instead of the actor: the critic
    target bootstraps from the stored next action (a sample of that policy at s') and carries no
    entropy term. For the frozen-teacher critic warm-up; implies ``train_actor = False``. Not in holosoma."""

    # -- replay and targets
    buffer_size: int = 1024
    """Transitions stored *per environment*."""
    num_steps: int = 1
    """n of the n-step return."""
    gamma: float = 0.97
    tau: float = 0.125
    """Polyak rate of the target critics, per critic update."""
    reward_scale: float = 1.0
    """Multiplies every reward before it is stored. Not in holosoma, whose returns already fit the
    default support; set it so that ``reward_scale x return`` sits inside ``[v_min, v_max]``."""

    # -- entropy
    alpha_init: float = 0.001
    use_autotune: bool = True
    target_entropy_ratio: float = 0.0
    """Target entropy is ``-n_act x ratio``."""

    # -- networks
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    num_q_networks: int = 2
    use_layer_norm: bool = True
    use_tanh: bool = True
    log_std_max: float = 0.0
    log_std_min: float = -5.0
    action_scale: list[float] | None = None
    """Per-dimension half-range of the tanh-squashed action (``None``: 1 for every dimension).
    holosoma derives it from joint limits; here it is given explicitly."""
    action_bias: list[float] | None = None
    """Per-dimension centre of the action range (``None``: 0)."""

    # -- pessimism (clipped double-Q; off in holosoma, which found it harmful when training from scratch
    #    with layer norm -- the opposite regime from fine-tuning a narrow pretrained policy, where the
    #    actor's first steps leave the data and the mean of the twins is optimistic there)
    actor_q_min: bool = False
    """Actor maximises the minimum over the twin critics instead of their mean."""
    target_q_min: bool = False
    """Critic targets bootstrap, per sample, from the target critic with the lower expected value
    (the same projected distribution is then the target of every critic)."""

    # -- anchor to a frozen reference actor (fine-tuning a pretrained policy; not in holosoma)
    anchor_coef: float = 0.0
    """Weight of ``0.5 * sum_d ((m_d - m_d^ref) / sigma_d^ref)^2`` added to the actor loss: the mean
    part of KL(reference || actor) on the pre-tanh Gaussian, in units of the reference's own std.
    The reference is the actor as loaded by ``FastSAC.load_actor`` (or ``set_anchor``). The actor's
    std is left to the entropy target. 0 disables it."""

    # -- distributional critic (C51)
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0

    # -- observation normalisation
    obs_normalization: bool = True
    freeze_obs_normalizer: bool = False
    """Stop updating the *actor's* observation normaliser from replay samples. holosoma updates it on
    every batch, which is fine for an actor trained from scratch alongside it, but rewrites the input
    statistics of a pretrained actor (its normaliser is part of the policy). Set automatically by
    ``FastSAC.load_actor``; pass ``agent.freeze_obs_normalizer=true`` when resuming such a run."""

    # -- speed
    compile: bool = False
    """torch.compile the update and policy functions. holosoma defaults to True; off here because
    the first compilation is slow and the cloth simulation, not the update, dominates step time."""
    amp: bool = True
    amp_dtype: str = "bf16"

    def single_env_preset(self, keep: set[str] | frozenset[str] = frozenset()) -> None:
        """Classic single-environment SAC (Haarnoja et al. 2018 / CleanRL scale) in place of FastSAC's.

        With one environment, holosoma's 1024-slot buffer would hold 20 s of experience, and its
        8192-sample batch would draw each transition many times per step. Instead: a large buffer,
        a small batch, one critic update per step, and 5000 steps of collection before the first.

        Args:
            keep: Field names to leave as they are, e.g. the ones set explicitly on the command line.
        """
        preset = {
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "num_updates": 1,
            "policy_frequency": 2,
            "learning_starts": 5000,
            "tau": 0.005,
            "logging_interval": 1000,
            "save_interval": 25000,
        }
        for name, value in preset.items():
            if name not in keep:
                setattr(self, name, value)
