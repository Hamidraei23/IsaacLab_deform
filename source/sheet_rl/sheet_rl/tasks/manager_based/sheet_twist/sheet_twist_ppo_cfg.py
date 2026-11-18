"""PPO runner for the sheet twist student: the cube twist student's settings, its own experiment folder."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from sheet_rl.tasks.manager_based.franka_lift_twist.twist_ppo_cfg import LiftCubeTwistPPORunnerCfg


@configclass
class SheetTwistPPORunnerCfg(LiftCubeTwistPPORunnerCfg):
    experiment_name = "sheet_platform_twist"
    max_iterations = 2000
    # Exploration between the DAgger students' 0.05 (too narrow: PPO could not rediscover eroded
    # behaviours, s348_ppo_w9 decayed one-way) and a generic 0.3 (too wide: the policy's own means
    # are |a| ~ 0.02-0.08, so 0.3 drowned them -- grasp halved, half the episodes dove below the
    # table, success zero). 0.1 keeps the noise at roughly signal scale. The base cube task's 1.0
    # would command 0.5 m tool steps through the scale-0.5 IK (see LiftCubeTwistPPORunnerCfg).
    # Entropy (0.006, inherited) is the base task's. NOTE: on --resume the std is loaded from the
    # checkpoint, overriding this -- reset it with scripts/reset_policy_std.py --std 0.1.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )


@configclass
class SheetTwistActionHistoryPPORunnerCfg(SheetTwistPPORunnerCfg):
    """Actor and critic both read the ``action_history`` group (48 values) instead of ``policy``.

    Its own experiment folder, because its checkpoints have a different input size and cannot be
    loaded by, or resumed from, the 59-input runs in ``sheet_platform_twist``.
    """
    experiment_name = "sheet_platform_twist_action_history"
    obs_groups = {"actor": ["action_history"], "critic": ["action_history"]}


@configclass
class SheetFineTuneAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "sheet_rl.tasks.manager_based.sheet_twist.reference_ppo:ReferencePPO"
    critic_only: bool = False
    critic_learning_rate: float = 3e-4
    reference_coef: float = 1.0
    max_update_kl: float = 0.01
    std_learning_rate: float | None = None
    """Learning rate of a learned state-independent actor std (FastSAC actor ``init_std``); None: the actor's."""
    num_learning_epochs = 2
    num_mini_batches = 4
    learning_rate = 1e-5
    schedule = "fixed"
    gamma = 0.997
    lam = 0.97
    entropy_coef = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    value_loss_coef = 1.0
    use_clipped_value_loss = False
    clip_param = 0.1


@configclass
class SheetTwistFineTuneRunnerCfg(SheetTwistPPORunnerCfg):
    """Opt-in student fine-tuning; exploration is loaded from the selected checkpoint."""
    num_steps_per_env = 32
    init_at_random_ep_len = False
    save_interval = 25
    algorithm = SheetFineTuneAlgorithmCfg()


def _fastsac_actor_cfg():
    """The FastSAC student's actor architecture (``SheetTwistFastSACCfg``) as an rsl_rl actor entry."""
    from sheet_rl.algorithms.fast_sac.rsl_rl_actor import RslRlFastSACActorCfg

    from .sheet_twist_sac_cfg import SheetTwistFastSACCfg

    sac = SheetTwistFastSACCfg()
    return RslRlFastSACActorCfg(
        hidden_dim=sac.actor_hidden_dim, log_std_min=sac.log_std_min, log_std_max=sac.log_std_max,
        use_layer_norm=sac.use_layer_norm, action_scale=list(sac.action_scale),
        action_bias=list(sac.action_bias) if sac.action_bias is not None else None,
    )


@configclass
class UR10eFineTuneRunnerCfg(SheetTwistFineTuneRunnerCfg):
    """ReferencePPO on the UR10e with the FastSAC student's actor itself and a value critic on the UR's own state.

    The actor reads the repaired ``action_history`` (49) through :class:`FastSACActorModel` -- the same network,
    frozen normaliser and squashed actions as the FastSAC checkpoint, written as an rsl_rl checkpoint by
    ``scripts/fastsac_actor_to_rsl_rl.py``. The critic reads ``ur_critic`` (55: UR joint and knuckle state, sheet
    points, target, last action), normalised. Warm it first with ``agent.algorithm.critic_only=true``.
    """
    experiment_name = "sheet_platform_twist_ur10e_ppo"
    obs_groups = {"actor": ["action_history"], "critic": ["ur_critic"]}
    actor = _fastsac_actor_cfg()
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)


@configclass
class UR16eFineTuneRunnerCfg(UR10eFineTuneRunnerCfg):
    """:class:`UR10eFineTuneRunnerCfg` for the UR16e task, logging to its own experiment folder."""
    experiment_name = "sheet_platform_twist_ur16e_ppo"
