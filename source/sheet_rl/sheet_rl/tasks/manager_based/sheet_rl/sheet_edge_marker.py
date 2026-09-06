# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A red dot pinned to the middle of the sheet's lower edge for the length of an episode.

Purely a marker: it has no rigid body, no collider and no mass, so the cloth falls and drapes
exactly as it did without it. Nothing in the MDP reads it either, which is what makes it safe to
leave on while a policy is being trained or a checkpoint scored.

**Which border is "lower" is decided once, at reset, and then held.** At reset the sheet stands
upright in its slot, so the border with the lowest nodes is unambiguously the bottom edge --
and that is the moment to ask. Asking again later would give a different answer as soon as the
gripper turns the sheet over or lays it flat, and the dot would jump from edge to edge while the
cloth was being handled. Latching keeps the dot on one identified physical node, so it means the
same thing on step 400 as it did on step 1, and anything aimed at it -- see
:class:`~.sheet_push.SheetPusher` -- is aimed at the same material point all episode.

The latch is dropped when an environment resets and re-taken from the fresh layout on the next step.

Attached by :class:`~.sheet_rl_env.PushableSheetEnv`, which both the base task and the hand task
run on, so the dot is present in either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class SheetEdgeMarkerCfg:
    """Configuration for :class:`SheetEdgeMarker`.

    Reachable from the command line through Hydra, e.g. ``env.edge_marker.enabled=false``.
    """

    enabled: bool = True
    """Draw the dot at all.

    Switching it off also removes the target :class:`~.sheet_push.SheetPusher` aims at, so the push
    key goes quiet with it. The dot is the target, not a decoration on top of one.
    """

    resolution: tuple[int, int] = (8, 8)
    """The sheet mesh's ``(x, y)`` element counts, as given to its spawner.

    The node grid is one larger in each direction. The same parameter the observation and reward
    terms take, and it has to agree with them or the dot lands on the wrong node -- which is why
    ``SheetRlEnvCfg.__post_init__`` overwrites it with ``SHEET_RESOLUTION`` rather than trusting
    the two to be edited together. The literal here is only what a marker built outside the task
    would get; importing the constant instead would make this module and the environment config
    import each other.
    """

    asset_name: str = "deformable"
    """Scene entity holding the sheet."""

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Sheet/lower_edge",
        markers={
            "dot": sim_utils.SphereCfg(
                # The sheet is 0.2 m across an 8-element grid, so its nodes sit 25 mm apart. A 6 mm
                # sphere is comfortably smaller than that spacing -- it reads as a dot marking one
                # node rather than as a bead threaded onto the edge.
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 0.05)),
            )
        },
    )
    """The dot itself: one instanced red sphere per environment.

    Deliberately carries no ``rigid_props``, ``mass_props`` or ``collision_props``. A marker with any
    of those would be a body the cloth could hit, and the point of this is to change what is drawn
    and nothing else.
    """


class SheetEdgeMarker:
    """Latches the centre node of the sheet's lower edge at reset and draws a red dot on it.

    Driven by the environment: :meth:`clear_latch` with the step's dones and then :meth:`update`,
    once per step.

    Attributes:
        node_ids: The latched node, per environment, as an index into the sheet's node grid.
            ``-1`` means the environment has reset and is waiting to be re-latched on the next
            :meth:`update`. Shape ``(num_envs,)``.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: SheetEdgeMarkerCfg) -> None:
        """Build the node indices and spawn the marker.

        Args:
            env: The environment.
            cfg: What to draw and where to look for the sheet.
        """
        self.env = env
        self.cfg = cfg

        # -- the node grid. Row-major, one node more than there are elements in each direction, which
        # is the convention ``sheet_key_points`` and every reward term that reads a border share.
        cols, rows = cfg.resolution[0] + 1, cfg.resolution[1] + 1
        grid = torch.arange(rows * cols, device=env.device).view(rows, cols)
        # kept as four separate rows rather than one stacked tensor so a non-square sheet works: the
        # borders then have different lengths and cannot be stacked, while their centres still can
        self._borders = [grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]
        # integer division, matching ``sheet_key_points``: on an even-sided grid the "midpoint" is
        # the node just past centre, which is the closest thing to a midpoint that is a real node
        self._centers = torch.stack([ids[len(ids) // 2] for ids in self._borders])

        # -1 everywhere: nothing is latched until the first update sees a settled reset layout
        self.node_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

        self._markers = VisualizationMarkers(cfg.visualizer_cfg)
        self._markers.set_visibility(True)

    """
    Operations.
    """

    def clear_latch(self, dones: torch.Tensor) -> None:
        """Forget the latched edge for environments that have just reset.

        Args:
            dones: Environments that terminated or timed out on this step.
        """
        if bool(dones.any()):
            self.node_ids = torch.where(dones, torch.full_like(self.node_ids, -1), self.node_ids)

    def update(self) -> None:
        """Latch any environment that needs it, then move the dot onto its node."""
        sheet: DeformableObject = self.env.scene[self.cfg.asset_name]
        nodes = sheet.data.nodal_pos_w.torch

        stale = self.node_ids < 0
        if bool(stale.any()):
            # mean height of each border, then the centre node of the lowest one
            heights = torch.stack([nodes[:, ids, 2].mean(dim=-1) for ids in self._borders], dim=1)
            latched = self._centers[heights.argmin(dim=1)]
            self.node_ids = torch.where(stale, latched, self.node_ids)

        self._markers.visualize(translations=self.positions(nodes))

    def positions(self, nodes: torch.Tensor | None = None) -> torch.Tensor:
        """World position of the latched node in each environment. Shape ``(num_envs, 3)``.

        Args:
            nodes: The sheet's nodal positions, if the caller already has them. Read from the scene
                when omitted.
        """
        if nodes is None:
            sheet: DeformableObject = self.env.scene[self.cfg.asset_name]
            nodes = sheet.data.nodal_pos_w.torch
        # clamped rather than masked: an unlatched environment reads node 0 for one frame instead of
        # indexing out of bounds, and the next update replaces it with the real latch
        ids = self.node_ids.clamp_min(0)
        return nodes[torch.arange(len(nodes), device=nodes.device), ids]
