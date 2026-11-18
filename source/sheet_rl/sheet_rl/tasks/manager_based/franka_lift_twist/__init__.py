"""Franka cube lift with a twist (relative end-effector pose) action, for a DAgger student.

The environment is Isaac Lab's contrib IK-Rel lift task -- 6D delta pose of the tool point in the robot
base frame (scale 0.5) plus the binary gripper -- with the motion costs doubled (see twist_env_cfg.py).
Registered here, with an RSL-RL runner, so the student can be trained with
``scripts/train_dagger_twist.py`` and then resumed with PPO through ``scripts/train.py``.
"""

import gymnasium as gym

gym.register(
    id="Lift-Cube-Franka-Twist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twist_env_cfg:FrankaCubeLiftTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.twist_ppo_cfg:LiftCubeTwistPPORunnerCfg",
    },
)

# The same twist action on the joint teacher's arm drives (stiffness 80, gravity on); see twist_env_cfg.py.
gym.register(
    id="Lift-Cube-Franka-Twist-TeacherGains-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twist_env_cfg:FrankaCubeLiftTwistTeacherGainsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.twist_ppo_cfg:LiftCubeTwistPPORunnerCfg",
    },
)

# The twist task with the cube on a platform 1.5-15 cm tall and charges on hand speed; see twist_env_cfg.py.
gym.register(
    id="Lift-Cube-Franka-Twist-Platform-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twist_env_cfg:FrankaCubeLiftTwistPlatformEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.twist_ppo_cfg:LiftCubeTwistPPORunnerCfg",
    },
)

# Pick from the platform, then place on a second one; see twist_env_cfg.py and pick_place_mdp.py.
gym.register(
    id="Lift-Cube-Franka-Twist-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twist_env_cfg:FrankaCubeLiftTwistPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.twist_ppo_cfg:LiftCubeTwistPPORunnerCfg",
    },
)
