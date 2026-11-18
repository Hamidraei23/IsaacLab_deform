"""PPO defaults shared by platform placement and grasp pretraining."""

from isaaclab.utils.configclass import configclass
from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class PlatformPPORunnerCfg(PPORunnerCfg):
    experiment_name = "sheet_platform"
    max_iterations = 3000
    num_steps_per_env = 24
    save_interval = 20
    clip_actions = 2.4

    def __post_init__(self):
        self.actor.hidden_dims = [256, 128, 64]
        self.critic.hidden_dims = [256, 128, 64]
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        self.algorithm.learning_rate = 3e-4
        # Match extraction potential shaping and the placement reward budget.
        self.algorithm.gamma = 0.99
        self.algorithm.entropy_coef = 0.003
        self.actor.distribution_cfg.init_std = 0.35


@configclass
class PlatformGraspPPORunnerCfg(PlatformPPORunnerCfg):
    experiment_name = "sheet_platform_grasp"

    def __post_init__(self):
        super().__post_init__()
        # Grasp episodes last about 2-3 s (60-100 steps), so a 100-step horizon is enough, and the
        # extraction bonus is sized against it (see ``SheetRlPlatformEnvCfg``). Training starts
        # from random weights, so explore more widely than the placement fine-tune does.
        self.algorithm.gamma = 0.99
        self.algorithm.entropy_coef = 0.003
        self.actor.distribution_cfg.init_std = 0.5
