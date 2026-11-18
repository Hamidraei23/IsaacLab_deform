"""The sheet platform task with the cube twist teacher's control (see ``sheet_twist_env_cfg``).

Distilled into by ``scripts/train_dagger_sheet_twist.py`` from a ``Lift-Cube-Franka-Twist-PickPlace-v0``
teacher. The sheet tasks registered in ``sheet_rl`` are unchanged.
"""

import gymnasium as gym

gym.register(
    id="Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:DivergenceTolerantEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_ur10e_env_cfg:SheetUR10eTwistStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:UR10eFineTuneRunnerCfg",
        "rsl_rl_action_history_cfg_entry_point": (
            f"{__name__}.sheet_twist_ppo_cfg:SheetTwistActionHistoryPPORunnerCfg"
        ),
        "fast_sac_cfg_entry_point": f"{__name__}.sheet_twist_sac_cfg:SheetTwistFastSACCfg",
    },
)

# The UR10e task above, unchanged, on a UR16e (same scene, base position and gripper); see sheet_ur16e_env_cfg.py.
gym.register(
    id="Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:DivergenceTolerantEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_ur16e_env_cfg:SheetUR16eTwistStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:UR16eFineTuneRunnerCfg",
        "rsl_rl_action_history_cfg_entry_point": (
            f"{__name__}.sheet_twist_ppo_cfg:SheetTwistActionHistoryPPORunnerCfg"
        ),
        "fast_sac_cfg_entry_point": f"{__name__}.sheet_twist_sac_cfg:SheetTwistFastSACCfg",
    },
)

# The UR16e task with a simulated RealSense D435i colour camera fixed around the table (sheet_ur16e_camera_env_cfg.py).
gym.register(
    id="Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:DivergenceTolerantEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_ur16e_camera_env_cfg:SheetUR16eCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:UR16eFineTuneRunnerCfg",
        "rsl_rl_action_history_cfg_entry_point": (
            f"{__name__}.sheet_twist_ppo_cfg:SheetTwistActionHistoryPPORunnerCfg"
        ),
        "fast_sac_cfg_entry_point": f"{__name__}.sheet_twist_sac_cfg:SheetTwistFastSACCfg",
    },
)

gym.register(
    id="Template-Sheet-Rl-Platform-Twist-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:PushableSheetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_twist_env_cfg:SheetRlPlatformTwistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
    },
)

# The same task with a solid plate instead of the cloth; see sheet_twist_rigid_env_cfg.py.
gym.register(
    id="Template-Sheet-Rl-Platform-Twist-Rigid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_twist_rigid_env_cfg:SheetRlPlatformTwistRigidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
    },
)

# The original platform task's reward on the twist plant: an evaluation bridge for twist students.
gym.register(
    id="Template-Sheet-Rl-Platform-Twist-OrigReward-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:PushableSheetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_twist_env_cfg:SheetRlPlatformTwistOriginalRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
    },
)

# The original platform task, whole, under the twist student's control; one self-contained file.
gym.register(
    id="Template-Sheet-Rl-Platform-TwistStudent-v0",
    entry_point="sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env:PushableSheetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_platform_twist_student_env_cfg:SheetPlatformTwistStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistPPORunnerCfg",
        "rsl_rl_finetune_cfg_entry_point": f"{__name__}.sheet_twist_ppo_cfg:SheetTwistFineTuneRunnerCfg",
        "rsl_rl_action_history_cfg_entry_point": (
            f"{__name__}.sheet_twist_ppo_cfg:SheetTwistActionHistoryPPORunnerCfg"
        ),
        "fast_sac_cfg_entry_point": f"{__name__}.sheet_twist_sac_cfg:SheetTwistFastSACCfg",
    },
)
