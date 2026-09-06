# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The task environments: one that can be poked while it runs, and one that also survives a
diverged simulation."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .sheet_edge_marker import SheetEdgeMarker
from .sheet_push import SheetPusher


class PushableSheetEnv(ManagerBasedRLEnv):
    """The stock environment plus the two overlays that make the sheet pokable by hand.

    :class:`~.sheet_edge_marker.SheetEdgeMarker` draws a red dot on the node at the middle of the
    sheet's lower edge, latched at reset, and :class:`~.sheet_push.SheetPusher` applies a real force
    to that node while ``X`` is pressed in the Newton GL viewer.

    Neither is part of the MDP -- no observation, reward or termination reads either of them -- so
    training and playback are bit-identical to the stock environment until the key is actually
    pressed. That is what makes it safe to carry them on the *training* task rather than on an
    evaluation-only variant: a probe you have to switch environments to use is a probe you will not
    reach for when a policy does something odd on screen.

    Both are optional and are built only when the config declares them, so this class is still usable
    with a config that has neither field.

    Note:
        The pusher is constructed *before* ``super().__init__``. Its force is applied through a
        callback that has to be registered before the solver captures its CUDA graph, and capture
        happens inside the base initializer; registering afterwards leaves the launch outside the
        graph, where it is silently never replayed and the key appears dead. See
        :mod:`~.sheet_push`. The marker, which only draws, is built afterwards like any other
        visualizer.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Register the callback before base initialization captures the CUDA graph.
        push_cfg = getattr(cfg, "push", None)
        self._pusher = (
            SheetPusher(cfg.scene.num_envs, cfg.sim.device, push_cfg)
            if push_cfg is not None and push_cfg.enabled
            else None
        )

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        marker_cfg = getattr(self.cfg, "edge_marker", None)
        self._edge_marker = (
            SheetEdgeMarker(self, marker_cfg) if marker_cfg is not None and marker_cfg.enabled else None
        )

        if self._pusher is not None and self._edge_marker is not None:
            self._pusher.bind(self, self._edge_marker)
        elif self._pusher is not None:
            # The push requires the marker's target node.
            print("[sheet_rl] the lower-edge marker is off, so the sheet push has no target and is inactive.")
            self._pusher = None

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)

        # Clear overlay state belonging to completed episodes.
        dones = terminated | truncated
        if self._edge_marker is not None:
            self._edge_marker.clear_latch(dones)
            # Latch from final or freshly reset particle positions.
            self._edge_marker.update()
        if self._pusher is not None:
            self._pusher.clear(dones)
            # Target the node selected by the marker above.
            self._pusher.update()

        return obs, reward, terminated, truncated, extras


class DivergenceTolerantEnv(PushableSheetEnv):
    """Replaces non-finite rewards with zero, so one blown-up environment cannot end the run.

    The sculpted arm occasionally drives an environment to a state the solver cannot resolve: on the
    step after a reset the cloth reaches velocities in the thousands of metres per second and the
    Franka's joint positions go NaN. Measured at roughly one environment in a thousand, on the first
    step, under random actions.

    That alone would be tolerable -- ``simulation_diverged`` terminates the environment and the next
    reset draws a fresh layout. What is not tolerable is the arithmetic in between. Terminations are
    computed *before* rewards, so the step that goes bad still has its rewards evaluated against NaN
    positions, and every term that carries a distance into its payout returns NaN. That single value
    reaches rsl_rl's ``check_nan``, which ends the job. One environment in a thousand, on one step,
    kills a run that was going to last three thousand iterations.

    Zeroing is the honest value here rather than a fudge. The environment is being terminated on
    this very step, its state is meaningless, and no reward computed from NaN coordinates carries
    information about what the policy did. Substituting zero says exactly that: this step teaches
    nothing. The alternative -- clamping to some large negative number -- would teach the policy to
    avoid a solver failure it has no way to see coming or influence.

    Note:
        Deliberately narrow. Only non-finite values are touched, so a legitimately large reward
        passes through untouched and the substitution cannot mask a reward-shaping bug: if a term
        starts returning NaN for reasons of its own, ``simulation_diverged`` stays quiet and the
        divergence counter in the log stays at zero while the returns go strange.

    Note:
        Hooked on :meth:`step` rather than on a reward callback, because there is no reward callback
        to hook. :class:`~isaaclab.envs.ManagerBasedRLEnv` calls ``reward_manager.compute`` inline
        and assigns ``reward_buf`` directly; ``_get_rewards`` is a :class:`DirectRLEnv` method and
        overriding it here does nothing at all. ``reward_buf`` is corrected alongside the returned
        tensor so the environment's own logging does not report a NaN mean.

    Note:
        Used by the hand task, whose mesh arm is what provokes this. The base task's capsule has
        never produced a non-finite state -- 1024 environments under random actions ran clean -- so
        it runs on :class:`PushableSheetEnv` directly rather than carrying a guard it does not need.
        The overlays are inherited from there, so both tasks are poked the same way.
    """

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)

        bad = ~torch.isfinite(reward)
        if bad.any():
            reward = torch.where(bad, torch.zeros_like(reward), reward)
            # the environment logs from this buffer, so leaving it NaN would poison the mean
            self.reward_buf = reward
            # counted rather than merely swallowed: a rising count means the solver is being driven
            # into states it cannot resolve more often, which is a physics problem to go and fix
            # rather than something this class should quietly keep absorbing
            extras.setdefault("log", {})["Events/diverged_steps"] = bad.float().mean()

        return obs, reward, terminated, truncated, extras
