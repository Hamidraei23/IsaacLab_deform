"""Arm command smoothing and observable applied actions for the platform task."""

import math
import torch
from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import AbsBinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils.configclass import configclass


def smooth_command(previous, request, alpha, max_delta):
    """Filter requested action units and bound the change per control step."""
    return previous + max_delta * torch.tanh(alpha * (request - previous) / max_delta)


class SmoothRelativeJointPositionAction(RelativeJointPositionAction):
    """Rate-limited joint-target *velocity* commands, integrated into a persistent target.

    Each control step moves the joint position target by ``scale * filtered command``,
    interpolated across physics substeps. The target is integrated from the previous
    *target*, not re-based on the measured position as the stock relative action does. With
    gravity enabled that re-basing discards the PD error that holds the arm up, so a zero
    command sagged about 1.3 mm per step and ended most early episodes below the table. The
    target is kept within ``max_lead`` rad of the measured position so a blocked arm cannot
    wind it up.

    Bounds apply to commands, not physical acceleration under gravity/contact. Measured
    acceleration is separately penalized and recorded. Raw actions expose the applied command
    so the existing eight-action observation contains the filter state.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.time_constant <= 0 or cfg.max_action_rate <= 0 or cfg.max_lead <= 0:
            raise ValueError('Smoothing time constant, action rate and target lead must be positive')
        self.alpha = 1 - math.exp(-env.step_dt / cfg.time_constant)
        self.max_delta = cfg.max_action_rate * env.step_dt
        self.previous = torch.zeros_like(self._raw_actions)
        self.target = torch.zeros_like(self._raw_actions)
        self.needs_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.substep = 0
        self.decimation = env.cfg.decimation

    def _increment(self, command):
        delta = command * self._scale + self._offset
        if self.cfg.clip is not None:
            delta = delta.clamp(min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        return delta

    def process_actions(self, actions):
        self.previous.copy_(self._raw_actions)
        applied = smooth_command(self.previous, actions, self.alpha, self.max_delta)
        super().process_actions(applied)
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self.target[self.needs_target] = joint_pos[self.needs_target]
        self.needs_target[:] = False
        self.substep = 0

    def apply_actions(self):
        self.substep += 1
        fraction = min(self.substep / self.decimation, 1.0)
        command = self.previous + fraction * (self._raw_actions - self.previous)
        self.target += self._increment(command) / self.decimation
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self.target.copy_(torch.minimum(torch.maximum(self.target, joint_pos - self.cfg.max_lead),
                                        joint_pos + self.cfg.max_lead))
        self._asset.set_joint_position_target_index(target=self.target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0
        self._processed_actions[ids] = 0
        # Re-read the reset joint positions at the next command.
        self.needs_target[ids] = True


@configclass
class SmoothRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type: type = SmoothRelativeJointPositionAction
    time_constant: float = 0.05
    max_action_rate: float = 24.0
    max_lead: float = 0.15


class FilteredBinaryGripperAction(AbsBinaryJointPositionAction):
    """Binary gripper driven by a low-pass filtered command instead of each raw sample.

    ``state += alpha * (clip(action, -1, 1) - state)``; open when ``state`` is above the
    threshold. With the stock action one exploratory sample above zero opens the fingers:
    a policy holding at mean -0.5 and std 0.43 dropped the sheet on about 12% of steps, so a
    30-step lift almost never survived exploration and training settled on holding without
    lifting. Here opening takes several consecutive open requests, while a deliberate open or
    close still takes about two steps. ``raw_actions`` exposes the filter state, so the policy
    observation and the closure cost see the command actually applied.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0 < cfg.filter_alpha <= 1:
            raise ValueError('filter_alpha must be in (0, 1]')

    def process_actions(self, actions):
        state = self._raw_actions + self.cfg.filter_alpha * (actions.clamp(-1, 1) - self._raw_actions)
        super().process_actions(state)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # Episodes start with the fingers open; start the filter on the open side too.
        self._raw_actions[slice(None) if env_ids is None else env_ids] = self.cfg.initial_state


@configclass
class FilteredBinaryGripperActionCfg(AbsBinaryJointPositionActionCfg):
    class_type: type = FilteredBinaryGripperAction
    filter_alpha: float = 0.3
    initial_state: float = 0.5


def applied_policy_actions(env):
    """Seven filtered arm commands followed by the unchanged binary gripper request."""
    return torch.cat([env.action_manager.get_term(name).raw_actions
                      for name in ('arm_action', 'gripper_action')], dim=-1)
