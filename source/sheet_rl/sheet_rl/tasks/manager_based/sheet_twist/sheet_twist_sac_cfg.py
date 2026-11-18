"""FastSAC for the sheet twist student: ``action_history`` actor, asymmetric ``policy`` critic."""

from isaaclab.utils.configclass import configclass

from sheet_rl.algorithms.fast_sac import FastSACCfg


@configclass
class SheetTwistFastSACCfg(FastSACCfg):
    """holosoma's FastSAC defaults, with the observation groups, discount and scales of this task.

    The scales were measured on ``best_deformable_twist/model_2874.pt`` (256 environments, 400
    steps, gamma 0.997; about 920 episodes per mode):

    - Return-to-go runs from -19 to 5706 (stochastic) and to 4312 (deterministic). The success
      payment alone reaches 5413 on the one terminal step. ``reward_scale`` 0.01 maps that to about
      [-0.2, 57], inside the C51 support ``[v_min, v_max]`` = [-10, 70] (0.8 per atom).
    - The teacher's *mean* actions reach |a| = 2.7 / 1.8 / 3.6 (position), 6.4 / 9.7 / 7.2
      (rotation) and 14.7 (gripper). ``action_scale`` bounds the tanh-squashed actions about 25%
      beyond the position and rotation maxima, so a student can represent everything the teacher
      does. The gripper stays at 1: the gripper action clips its input to [-1, 1] anyway.
    """

    experiment_name = "sheet_platform_twist_sac"
    actor_obs_groups = ["action_history"]
    critic_obs_groups = ["policy"]
    gamma = 0.997
    """Matches the teacher's PPO and the potential-based grasp_progress term (synced by train_sac.py)."""
    reward_scale = 0.01
    v_min = -10.0
    v_max = 70.0
    action_scale = [3.5, 2.5, 4.5, 8.0, 12.0, 9.0, 1.0]
