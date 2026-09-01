# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Sheet-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_rl_env_cfg:SheetRlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# the same task with a cosmetic mannequin hand on a random end of the arm. A separate id so the
# base task -- and any training running against it -- is untouched by hand experiments.
#
# On its own environment class rather than the stock one: the mesh arm occasionally drives a single
# environment to a state the solver cannot resolve, and without the guard that class carries, one
# environment in a thousand ends the whole run. See ``DivergenceTolerantEnv``.
gym.register(
    id="Template-Sheet-Rl-Hand-v0",
    entry_point=f"{__name__}.sheet_rl_env:DivergenceTolerantEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sheet_rl_hand_env_cfg:SheetRlHandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)