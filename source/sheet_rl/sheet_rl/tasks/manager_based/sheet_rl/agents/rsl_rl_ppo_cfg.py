# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # 0.99 gives a discount horizon of 100 steps, which was the same order as the episodes
        # themselves -- so the success bonus was discounted to 41% of face value simply for
        # arriving at the end, and every extra step cost real reward. That is a hurry-up pressure
        # the speed penalty has to fight rather than a property of the task. 0.995 doubles the
        # horizon to 200 steps, so a slower policy is not charged for taking longer.
        #
        # It also lengthens the reach of the coverage annuity, which at 40/step is the term most
        # able to exploit a longer horizon: a stream held to the end of a 500-step episode is now
        # discounted far less against the success bonus that would cut it off.
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )