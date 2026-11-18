# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The task environments: one that can be poked while it runs, and one that also survives a
diverged simulation."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp import sheet_extracted
from .sheet_edge_marker import SheetEdgeMarker
from .platform_teleop import PlatformTeleop
from .sheet_keys import ViewerKeys
from .sheet_push import SheetPusher

RESET_KEY = "R"
"""Key that restarts the episode in every environment.

Unbound in both viewers -- Newton's uses ``WASDQE`` and the arrows for the camera, ``SPACE`` to
pause, ``H`` and ``F`` for the UI -- so nothing is being taken over.
"""


class PushableSheetEnv(ManagerBasedRLEnv):
    """The stock environment plus the two overlays that make the sheet pokable by hand.

    :class:`~.sheet_edge_marker.SheetEdgeMarker` draws a red dot on the node at the middle of the
    sheet's lower edge, latched at reset, and :class:`~.sheet_push.SheetPusher` applies a real force
    to that node when ``X`` is pressed. ``R`` restarts the episode. Both keys are bound by
    :class:`~.sheet_keys.ViewerKeys`, which reaches the Newton GL window and the Kit one alike, so
    they work on ``--viz newton_gl`` and ``--viz kit`` without the run having to say which.

    Neither is part of the MDP -- no observation, reward or termination reads either of them -- so
    training and playback are bit-identical to the stock environment until the key is actually
    pressed. That is what makes it safe to carry them on the *training* task rather than on an
    evaluation-only variant: a probe you have to switch environments to use is a probe you will not
    reach for when a policy does something odd on screen.

    Both are optional and are built only when the config declares them, so this class is still usable
    with a config that has neither field.

    **The push can also be fired by the task itself.** With ``env.push.on_extraction=true`` -- what
    ``--push`` on the training command expands to -- half the episodes that pull the sheet clear of
    the slot are pushed the moment they do, once, with the same force a key press applies. That
    turns the probe into part of the task: the policy has to keep hold of a sheet that is being
    yanked, rather than only of one that is not. See :meth:`_push_on_extraction`.

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

        # -- the automatic push, and the per-episode bookkeeping it needs.
        push_cfg = getattr(self.cfg, "push", None)
        self._auto_push = self._pusher is not None and getattr(push_cfg, "on_extraction", False)
        if self._auto_push:
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Which episodes are due a push if they get that far. Drawn now for the first episode
            # and re-drawn on every reset.
            self._push_coin = self._draw_push_coins()
            # Last step's extraction latch, so the push fires on the transition rather than on
            # every step of phase two.
            self._was_extracted = zeros.clone()
            # What actually happened, reported on reset as ``Events/pushed_frac``.
            self._pushed_episode = zeros.clone()
            print(
                f"[sheet_rl] pushing {100 * push_cfg.on_extraction_probability:.0f}% of extractions"
                f" at {push_cfg.force:.1f} N for {push_cfg.duration_s:.2f} s"
            )

        # -- the keyboard. Bound to whichever viewer turns out to be running, once it has been
        #    built, which is a few steps into the run -- see ``sheet_keys``.
        self._reset_requested = False
        bindings = {RESET_KEY: self.request_reset}
        if self._pusher is not None:
            bindings[self.cfg.push.key] = self._pusher.request
        # -- keyboard teleop of the platform (play only; off unless env.platform_teleop.enabled)
        teleop_cfg = getattr(self.cfg, "platform_teleop", None)
        self._platform_teleop = (
            PlatformTeleop(self, teleop_cfg) if teleop_cfg is not None and teleop_cfg.enabled else None
        )
        self._keys = ViewerKeys(self, bindings, held=self._platform_teleop)

    """
    Operations.
    """

    def request_reset(self) -> None:
        """Restart the episode in every environment at the start of the next step.

        Runs the episode out through the normal timeout rather than calling :meth:`reset` directly.
        A direct reset would replace the state underneath a caller that is holding observations from
        before it; going through the termination manager means the step returns ``truncated`` and a
        fresh observation the way any other episode end does, so a playback loop, the policy's own
        recurrent state and the two overlays all see the reset they expect.

        Safe to call from a viewer callback on another thread: it only sets a flag, which the next
        step reads.
        """
        self._reset_requested = True

    def step(self, action: torch.Tensor):
        self._keys.update()
        if self._platform_teleop is not None:
            # before the step, so this step's physics and observation both see the moved platform
            self._platform_teleop.apply()

        if self._reset_requested:
            self._reset_requested = False
            # ``step`` increments this before the terminations are computed, so a buffer already at
            # the limit times out on this very step.
            self.episode_length_buf[:] = self.max_episode_length

        obs, reward, terminated, truncated, extras = super().step(action)

        # Clear overlay state belonging to completed episodes.
        dones = terminated | truncated
        if self._edge_marker is not None:
            self._edge_marker.clear_latch(dones)
            # Latch from final or freshly reset particle positions.
            self._edge_marker.update()
        if self._pusher is not None:
            self._pusher.clear(dones)
            # Arm before the pusher refreshes its arrays, so a push fires on the step the
            # extraction happened rather than on the one after it.
            if self._auto_push:
                self._push_on_extraction(dones, extras)
            # Target the node selected by the marker above.
            self._pusher.update()

        return obs, reward, terminated, truncated, extras

    """
    Internal helpers.
    """

    def _draw_push_coins(self) -> torch.Tensor:
        """One coin per environment: is the episode about to start due a push if it extracts?"""
        probability = self.cfg.push.on_extraction_probability
        return torch.rand(self.num_envs, device=self.device) < probability

    def _push_on_extraction(self, dones: torch.Tensor, extras: dict) -> None:
        """Push the environments that have just pulled the sheet clear of the slot.

        Fires on the *rising edge* of the extraction latch, which is set once per episode by
        ``grasp_stage_reward`` and cleared by its ``reset``. Rewards are computed inside
        ``super().step()``, so by the time this runs the latch already describes the step that just
        finished -- and environments that reset on that step have had it cleared, which is what
        keeps a push from being fired into a fresh episode.

        Args:
            dones: Environments that terminated or timed out on this step.
            extras: The step's extras dict, where the per-episode rate is reported.
        """
        if bool(dones.any()):
            # Report the episodes that just ended before their state is cleared. Averaged over the
            # environments that reset, matching how every other ``Events/`` figure is logged: the
            # fraction of *all* episodes that got the push, so it can be read against
            # ``Events/extraction`` to see whether the policy is avoiding the extraction itself.
            extras.setdefault("log", {})["Events/pushed_frac"] = self._pushed_episode[dones].float().mean()

            keep = ~dones
            self._pushed_episode &= keep
            self._was_extracted &= keep
            # A fresh coin for each new episode, drawn before it is known whether that episode will
            # extract at all.
            self._push_coin = torch.where(dones, self._draw_push_coins(), self._push_coin)

        extracted = sheet_extracted(self)
        fire = extracted & ~self._was_extracted & self._push_coin & ~self._pushed_episode
        # cloned: the latch is the live tensor the reward term keeps writing to
        self._was_extracted = extracted.clone()

        if bool(fire.any()):
            self._pusher.request_envs(fire)
            self._pushed_episode |= fire


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

    def _reset_idx(self, env_ids):
        # The reward manager keeps per-term episode sums too, and a diverged step leaves them NaN; it logs
        # their mean over the environments it resets, so one NaN would blank that window's curve for
        # every such term. Zeroed for the same reason as the step reward: that state carried nothing.
        for sums in self.reward_manager._episode_sums.values():
            torch.nan_to_num_(sums, nan=0.0, posinf=0.0, neginf=0.0)
        super()._reset_idx(env_ids)
