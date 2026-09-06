# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the sheet task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import string as string_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv


class SquashedRelativeJointPositionAction(RelativeJointPositionAction):
    r"""Relative joint position action whose step-to-step *change* is smoothly bounded per joint.

    Applies

    .. math::

        a[t] = a[t-1] + \Delta_{\max} \tanh\!\left(\frac{u[t] - a[t-1]}{\Delta_{\max}}\right)

    where :math:`u[t]` is what the policy asked for and :math:`a[t]` is what the joints receive. The
    increment is bounded by :math:`\Delta_{\max}` for any input whatsoever, so commanded joint
    acceleration is bounded *by construction* rather than discouraged after the fact.

    **Why this bounds acceleration.** ``RelativeJointPositionAction`` re-derives its target from the
    measured position on every physics substep, so a held action is a fixed position error against
    the actuator and settles to a constant velocity -- measured at ``q_dot ~ 0.222 * a`` rad/s for
    this task's ``scale``. Velocity is therefore what the action sets, and a change in action is a
    change in commanded velocity: acceleration. Bounding the increment bounds the acceleration.

    **Why a tanh and not a clamp.** A hard clamp bounds the increment equally well and is flat
    outside the bound: once the request is past it, pushing further changes nothing that reaches the
    simulator, so the return stops depending on the output and the gradient vanishes. That plateau
    is precisely what produced the bang-bang policy this term exists to prevent -- ``clip_actions``
    is already a hard bound, the policy's distribution walked out past it and stayed, and the sign
    of each channel became its only remaining control authority. A second hard bound reproduces the
    collapse one derivative up: the policy saturates, the clamp turns that into a fixed-rate ramp,
    and a square wave becomes a triangle wave -- bounded acceleration, but no control over it.

    The tanh is strictly monotone with no flat region, so every request maps to a distinct applied
    action and the gradient never dies. It needs no companion "you hit the bound" penalty, because
    there is no lost gradient to buy back, and it cannot be ridden the way a rail can.

    The cost is that the map is nonlinear: near :math:`a[t-1]` the response is one-to-one, and it
    compresses towards :math:`\pm\Delta_{\max}` as the request grows. Fine control near the current
    command is therefore *cheaper* than large swings, which is the intended bias.

    Note:
        :attr:`raw_actions` holds the **applied** action, not the request. This is deliberate and
        load-bearing. ``ActionManager.action`` keeps the request -- it is copied before the terms
        run, and the slice handed to :meth:`process_actions` is a view of the caller's tensor rather
        than of the manager's buffer, so a term cannot write its result back. Anything reading the
        manager therefore sees what was asked for, not what was done. Observe
        :func:`~sheet_rl...mdp.observations.applied_actions`, which reads ``raw_actions``, so the
        policy sees the value its next increment is measured against. Feeding it the request instead
        hides the reference point and makes the task partially observed.

    Note:
        Reset to zero, matching ``ActionManager.reset``, so the first increment of an episode starts
        from rest and is itself bounded.
    """

    cfg: SquashedRelativeJointPositionActionCfg

    def __init__(self, cfg: SquashedRelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolved by joint *name* when given a dict, so the numbers cannot be silently transposed
        # by a change in how ``joint_names`` resolves. A scalar applies to every joint.
        if isinstance(cfg.max_delta, (float, int)):
            limits = torch.full((self.action_dim,), float(cfg.max_delta), device=self.device)
        else:
            limits = torch.zeros(self.action_dim, device=self.device)
            index_list, name_list, value_list = string_utils.resolve_matching_names_values(
                cfg.max_delta, self._joint_names
            )
            if len(index_list) != self.action_dim:
                raise ValueError(
                    "max_delta must cover every joint of the action term. Missing:"
                    f" {set(self._joint_names) - set(name_list)}"
                )
            limits[index_list] = torch.tensor(value_list, device=self.device)

        if float(limits.min()) <= 0.0:
            raise ValueError(f"max_delta must be positive for every joint, got {limits.tolist()}")
        # (1, num_joints), broadcasting over environments
        self._max_delta = limits.unsqueeze(0)

        self._applied = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_applied = torch.zeros_like(self._applied)
        # how far the request ran past what the bound allows, per joint, this step
        self._overshoot = torch.zeros_like(self._applied)

    @property
    def applied_actions(self) -> torch.Tensor:
        """The squashed action actually sent to the joints. Shape ``(num_envs, num_joints)``."""
        return self._applied

    @property
    def prev_applied_actions(self) -> torch.Tensor:
        """The applied action of the previous control step. Shape ``(num_envs, num_joints)``."""
        return self._prev_applied

    @property
    def max_delta(self) -> torch.Tensor:
        """Per-joint increment bound. Shape ``(1, num_joints)``."""
        return self._max_delta

    @property
    def overshoot(self) -> torch.Tensor:
        """How far the *request* ran past the bound this step, per joint [action units].

        ``max(|u[t] - a[t-1]| - max_delta, 0)``. Zero whenever the policy asked for an increment the
        bound could deliver, and growing linearly once it asks for more.

        The squash means an overshoot costs nothing physically -- the applied increment is capped
        either way -- so this measures *intent*, not motion: how hard the policy is leaning on a
        limit it cannot cross. That is worth charging separately, because a request far outside the
        bound is where the tanh saturates and its gradient thins out. Keeping the request near the
        admissible range is what keeps the squash in the part of its curve that still teaches.

        Shape ``(num_envs, num_joints)``.
        """
        return self._overshoot

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._applied[env_ids] = 0.0
        self._prev_applied[env_ids] = 0.0
        self._overshoot[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._prev_applied = self._applied.clone()
        delta = actions - self._applied
        # recorded before the squash, since afterwards the excess is gone by construction
        self._overshoot = (delta.abs() - self._max_delta).clamp(min=0.0)
        self._applied = self._applied + self._max_delta * torch.tanh(delta / self._max_delta)
        # hand the applied action to the affine transform and the relative-position update, and
        # publish it as ``raw_actions`` for the observation to read back
        super().process_actions(self._applied)


@configclass
class SquashedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SquashedRelativeJointPositionAction`."""

    class_type: type[SquashedRelativeJointPositionAction] = SquashedRelativeJointPositionAction

    max_delta: float | dict[str, float] = 1.0
    """Largest change in the action per control step, in action units. Per joint when a dict.

    Derived from the hardware rather than tuned. A held action settles to ``q_dot ~ 0.222 * a``
    rad/s, so a change in action is a change in commanded velocity and dividing by the control
    period gives acceleration. Inverting against a joint's acceleration limit:

    ``max_delta = q_ddot_limit * step_dt / 0.222``

    Per joint rather than one number for all seven, because the Franka's acceleration limits span
    7.5 to 20 rad/s^2 -- a single bound either wastes the wrist's capability or violates the
    shoulder's.

    Keyed by joint name, so the values cannot be silently transposed if ``joint_names`` ever
    resolves in a different order.
    """
