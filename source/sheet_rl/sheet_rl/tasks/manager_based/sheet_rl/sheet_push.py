# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A keyboard-triggered force on the red dot at the sheet's lower edge.

Press the bound key -- ``X`` by default -- in the Newton GL window and a constant force is applied to
the latched lower-edge node for :attr:`SheetPushCfg.duration_s`, pointing sideways and up: ``+y`` or
``-y`` combined with ``+z``. It is a robustness probe, not part of the MDP -- no observation, reward
or termination is aware of it -- so a policy trains and a checkpoint plays identically until the key
is pressed.

**A real force, not a velocity written into the state.** It is added to ``State.particle_f`` before
every solver substep, which is the same channel Newton's own mouse-drag picking uses, so the solver
resolves it against the cloth's stretch constraints the way it resolves gravity and contact. Newtons
therefore mean newtons here.

**Scale, so the number is not a surprise.** The sheet is 0.2 x 0.2 m at 10 kg/m^2, so it weighs
0.4 kg -- 3.9 N -- spread over 81 nodes, which puts one node's own weight at about 0.05 N. The
default 3 N on one node is therefore about three quarters of the weight of the whole sheet and
sixty times the weight of the node it is applied to. That is a real disturbance rather than a
nudge; ``env.push.force`` is the knob if it turns out to be too much or too little.

**Why the constructor runs before the environment's.** The callback has to be registered before the
solver's CUDA graph is captured, and capture happens inside ``ManagerBasedEnv.__init__``. Registering
afterwards puts the launch outside the captured graph, where it is silently never replayed -- the key
would appear to do nothing at all. :class:`~.sheet_rl_env.PushableSheetEnv` therefore builds this
object first and calls :meth:`bind` once the environment exists. Isaac Lab does the same thing for
viewer picking, through ``_prepare_newton_visualizer_for_capture``, and for the same reason.

**Why the force is read from device arrays rather than branched on in Python.** The callback body is
recorded into the graph once and replayed thereafter, so a host-side ``if pressed:`` would be frozen
at whatever it evaluated to during capture. The kernel is therefore launched unconditionally on every
substep and reads what to apply from two small arrays the host rewrites between steps: a per-
environment force, zero when idle, and a per-environment particle index, ``-1`` when idle. Updating
their *contents* is visible to the replayed graph; reallocating them would not be, which is why they
are allocated once and never replaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .sheet_edge_marker import SheetEdgeMarker


@configclass
class SheetPushCfg:
    """What the key does when it is pressed.

    Every field is reachable from the command line through Hydra, e.g.
    ``env.push.force=4.0 env.push.direction=+y``.
    """

    enabled: bool = True
    """Bind the key at all. Off leaves the environment exactly as it was."""

    key: str = "X"
    """Which key fires the push -- ``X``, ``K``, ``F1``. Bound by :class:`~.sheet_keys.ViewerKeys`,
    which reaches whichever viewer is running.

    ``X`` is unbound in Newton's viewer, so nothing is being taken over here. Avoid ``SPACE`` (pause
    rendering), ``H`` (hide UI), ``F`` (frame camera) and the ``WASDQE``/arrow camera keys.
    """

    force: float = 3.0
    """Magnitude of the force applied to the node [N].

    This is the length of the vector, not a per-axis figure: with the default :attr:`lift` the force
    is 2.12 N along y and 2.12 N along z, which together come to 3. See the module docstring for
    what 3 N means next to a 3.9 N sheet.
    """

    duration_s: float = 0.15
    """How long the force is held [s].

    Rounded to whole control steps, so the figure actually applied is a multiple of 33 ms: 0.15 s is
    4.5 steps at 30 Hz and lands on 4, a held 0.13 s. The force is applied on every solver substep
    within those steps, not once per control step, so this really is a force held for an eighth of a
    second rather than a train of impulses.
    """

    direction: str = "random"
    """Which way along y, one of ``random``, ``alternate``, ``+y``, ``-y``. The ``+z`` is always
    present, and set by :attr:`lift`.

    ``random`` draws a sign per environment per press, so a sixteen-environment playback shows both
    directions at once. ``alternate`` flips one shared sign on each press, which is the one to use
    when two runs have to be compared press for press.
    """

    lift: float = 1.0
    """How much ``+z`` goes into the direction, relative to one unit of ``y``.

    ``1.0`` is the 45-degree up-and-sideways push. ``0.0`` makes it purely horizontal; larger values
    tip it towards straight up. Only the *direction* changes -- the vector is normalised afterwards,
    so the force stays at :attr:`force` newtons whatever this is set to.
    """

    verbose: bool = True
    """Print a line per press, so a push that produced little visible motion can still be seen to
    have registered.

    Only key presses are announced. The automatic push below fires on a per-environment schedule
    over a thousand environments and would print several times a second, so it stays quiet and is
    reported through ``Events/pushed_frac`` in the training log instead.
    """

    on_extraction: bool = False
    """Push automatically, the instant the sheet is pulled clear of the slot.

    Off by default, which leaves the key as the only trigger and the environment exactly as it was.
    Armed by ``--push`` on the training command, which is what makes the push part of the *task*
    rather than a probe: the policy meets it during training and has to hold on through it.

    Fires at most once per episode, on the step the extraction latch is first set, and only in the
    fraction of episodes given by :attr:`on_extraction_probability`. Same force, direction and
    duration as a key press -- it is the same machinery, aimed by the same latched marker.
    """

    on_extraction_probability: float = 0.5
    """Fraction of *extracting* episodes that get the automatic push.

    Half, so the same policy update sees both: an extraction that is disturbed and one that is not.
    The unpushed half is the control -- without it a policy could only learn that extraction is
    always followed by a yank, and there would be nothing to compare against.

    Drawn per environment at reset, before it is known whether the episode will extract at all, so
    the coin cannot depend on how the episode went. The share of *all* episodes that end up pushed
    is therefore this figure times the extraction rate, and is logged as ``Events/pushed_frac``.
    """


@wp.kernel
def _add_particle_force(
    particle_index: wp.array(dtype=wp.int32),
    force: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
):
    """Add each environment's force to its target particle. A no-op where the index is negative.

    The branch is on device data inside the kernel rather than on the host, which is what lets the
    same launch be recorded once into the CUDA graph and then be switched on and off by rewriting
    ``particle_index`` and ``force`` from outside it.
    """
    i = wp.tid()
    p = particle_index[i]
    if p >= 0:
        wp.atomic_add(particle_f, p, force[i])


class SheetPusher:
    """Applies a held force to the sheet's latched lower-edge node when a key is pressed.

    Built before the environment -- see the module docstring -- so the constructor takes the two
    figures it needs from the config rather than from an environment that does not exist yet.
    :meth:`bind` supplies the rest, and :meth:`update` runs once per step.
    """

    def __init__(self, num_envs: int, device: str, cfg: SheetPushCfg) -> None:
        """Allocate the force arrays and register the substep callback.

        Args:
            num_envs: Number of environments, from ``cfg.scene.num_envs``.
            device: Simulation device, from ``cfg.sim.device``.
            cfg: What a press should do.

        Raises:
            ValueError: If :attr:`SheetPushCfg.direction` is not one of the four accepted values.
        """
        if cfg.direction not in ("random", "alternate", "+y", "-y"):
            raise ValueError(
                f"unknown push direction '{cfg.direction}'; expected 'random', 'alternate', '+y' or '-y'"
            )

        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device

        # Captured graphs require these addresses to remain stable.
        self._index_wp = wp.full(num_envs, value=-1, dtype=wp.int32, device=device)
        self._force_wp = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        # Torch views let the host update the same memory.
        self._index = wp.to_torch(self._index_wp)
        self._force = wp.to_torch(self._force_wp)

        from isaaclab_newton.physics.newton_manager import NewtonManager

        NewtonManager.register_state_force_callback(self._apply_forces)

        self._remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._unit = torch.zeros(num_envs, 3, device=device)

        self.env: ManagerBasedRLEnv | None = None
        self._edge: SheetEdgeMarker | None = None
        self._offsets: torch.Tensor | None = None
        self._hold_steps = 1

        self._presses = 0
        # Start alternation with +y.
        self._alternate = -1.0

    """
    Operations.
    """

    def bind(self, env: ManagerBasedRLEnv, edge: SheetEdgeMarker) -> None:
        """Attach the built environment and the marker whose node is the target.

        Args:
            env: The environment, now that it exists.
            edge: The marker holding the latched lower-edge node.

        Raises:
            RuntimeError: If the sheet does not expose the per-environment particle offsets the
                force kernel needs to turn a node index into a flat particle index.
        """
        self.env = env
        self._edge = edge
        self._hold_steps = max(1, round(self.cfg.duration_s / env.step_dt))

        sheet = env.scene[edge.cfg.asset_name]
        # Private offsets map per-environment nodes into Newton's flat particle array.
        offsets = getattr(sheet, "_particle_offsets", None)
        if offsets is None:
            offsets = getattr(sheet.data, "_particle_offsets", None)
        if offsets is None:
            raise RuntimeError(
                f"deformable asset '{edge.cfg.asset_name}' exposes no '_particle_offsets';"
                " the sheet push cannot map a node index onto Newton's flat particle array."
            )
        self._offsets = wp.to_torch(offsets).to(device=env.device, dtype=torch.long)

    def request(self, sign: float | None = None) -> None:
        """Ask for a push on the next step, as though the key had been pressed.

        Args:
            sign: ``+1`` or ``-1`` to force a direction, or ``None`` to use the configured one.
        """
        if sign is None:
            self._presses += 1
            return
        self._start(torch.full((self._num_envs,), float(sign), device=self._device))

    def request_envs(self, envs: torch.Tensor) -> None:
        """Start a push on the selected environments only, leaving every other one alone.

        This is what the automatic push uses: extraction happens at a different moment in each
        environment, so the force has to be armed per environment rather than across the batch the
        way a key press does it. Direction is drawn as configured, so ``random`` still gives each
        environment its own sign.

        Args:
            envs: Boolean mask over environments. Shape ``(num_envs,)``.
        """
        if not bool(envs.any()):
            return
        self._start(self._draw_sign(), envs=envs)

    def clear(self, dones: torch.Tensor) -> None:
        """Drop any push still owed to an environment that has just reset.

        Args:
            dones: Environments that terminated or timed out on this step.
        """
        if bool(dones.any()):
            self._remaining = torch.where(dones, torch.zeros_like(self._remaining), self._remaining)

    def update(self) -> None:
        """Start any requested push and refresh the arrays.

        Call it after the marker's own update, so the target node is the one the dot was just drawn
        on rather than the one it was on last step.
        """
        while self._presses:
            self._presses -= 1
            self._start(self._draw_sign())

        active = self._remaining > 0
        node_ids = self._edge.node_ids
        # Skip the single unlatch frame after reset.
        target = torch.where(node_ids >= 0, self._offsets + node_ids, torch.full_like(node_ids, -1))

        # Update contents without changing graph-captured addresses.
        self._index.copy_(torch.where(active & (target >= 0), target, torch.full_like(target, -1)))
        self._force.copy_(
            torch.where(active.unsqueeze(1), self._unit * self.cfg.force, torch.zeros_like(self._unit))
        )

        self._remaining = (self._remaining - 1).clamp_min(0)

    """
    Internal helpers.
    """

    def _apply_forces(self, state) -> None:
        """Add the current force to every target particle. Runs before each solver substep.

        Args:
            state: Newton's state for the substep about to run.
        """
        particle_f = getattr(state, "particle_f", None)
        if particle_f is None:
            # Nothing to push in a particle-free model.
            return
        wp.launch(
            _add_particle_force,
            dim=self._num_envs,
            inputs=[self._index_wp, self._force_wp, particle_f],
            device=self._device,
        )

    def _draw_sign(self) -> torch.Tensor:
        """One push direction per environment, as configured."""
        num_envs, device = self._num_envs, self._device
        if self.cfg.direction == "random":
            return torch.where(
                torch.rand(num_envs, device=device) < 0.5,
                -torch.ones(num_envs, device=device),
                torch.ones(num_envs, device=device),
            )
        if self.cfg.direction == "alternate":
            self._alternate = -self._alternate
            return torch.full((num_envs,), self._alternate, device=device)
        return torch.full((num_envs,), 1.0 if self.cfg.direction == "+y" else -1.0, device=device)

    def _start(self, sign: torch.Tensor, envs: torch.Tensor | None = None) -> None:
        """Arm the push, over every environment or over the given mask.

        A press during a push restarts its clock rather than stacking with it, so holding the key
        down cannot accumulate into a force larger than the one configured.

        Args:
            sign: Which way along y, per environment.
            envs: Boolean mask of environments to arm, or ``None`` for all of them. Environments
                outside the mask keep whatever push they already had.
        """
        direction = torch.zeros(self._num_envs, 3, device=self._device)
        direction[:, 1] = sign
        direction[:, 2] = self.cfg.lift
        unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        held = torch.full_like(self._remaining, self._hold_steps)

        if envs is None:
            self._unit = unit
            self._remaining = held
        else:
            self._unit = torch.where(envs.unsqueeze(1), unit, self._unit)
            self._remaining = torch.where(envs, held, self._remaining)

        # Only a key press is announced; see ``SheetPushCfg.verbose``.
        if self.cfg.verbose and envs is None:
            positive = int((sign > 0).sum())
            way = f"{positive} envs +y, {self._num_envs - positive} envs -y"
            print(
                f"[sheet_rl] push: {way} and +z, {self.cfg.force:.1f} N for"
                f" {self.cfg.duration_s:.2f} s ({self._hold_steps} steps)"
            )
