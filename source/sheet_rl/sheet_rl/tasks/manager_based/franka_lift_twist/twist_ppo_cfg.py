"""PPO runner for the twist student: the joint teacher's settings, under its own experiment folder."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.contrib.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg


@configclass
class LiftCubeTwistPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "lift_cube_franka_twist"
    # The IK action scale is 0.5, so a raw action of 1 asks for a 0.5 m / 0.5 rad step in one 20 ms
    # policy interval. The joint teacher's exploration spread of 1.0 would throw the arm across the
    # table; the DAgger script overwrites this with --student_std before saving.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
    )
