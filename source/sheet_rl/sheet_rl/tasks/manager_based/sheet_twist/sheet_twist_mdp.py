"""Terms of the sheet platform twist task: a platform of random height, and sheet coverage of it.

The platform is the sheet task's ``mannequin_arm`` cuboid (its scene name is kept so the Newton
rigid-to-cloth coupling still includes it), made ``PLATFORM_DEPTH`` tall and sunk into the table
until its top is at a height drawn at every reset, as in the cube twist tasks. Its centre is the
root, so every term that measures from "root + top_offset" sees the true top when
``top_offset = PLATFORM_DEPTH / 2``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from ..sheet_rl.mdp.gripper_geometry import finger_half_width

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from ..sheet_rl.mdp.platform import platform_placement
from ..sheet_rl.mdp.platform_geometry import sheet_triangles, surface_coverage, surface_samples
from ..sheet_rl.mdp.rewards import _is_holding, _phase_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PLATFORM_DEPTH = 0.20
"""Height of the platform box [m]; must exceed the tallest platform drawn."""


def set_platform_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    height_range: tuple[float, float],
    platform_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
):
    """Reset event, after the layout event: sink the platform until its top is at a random height.

    The layout event has already chosen its position and yaw; only the height changes here.
    """
    low, high = height_range
    if not 0.0 < low <= high < PLATFORM_DEPTH:
        raise ValueError(f"Platform heights {height_range} must lie in (0, {PLATFORM_DEPTH}).")
    platform = env.scene[platform_cfg.name]
    pose = torch.cat(
        (platform.data.root_pos_w.torch[env_ids], platform.data.root_quat_w.torch[env_ids]), dim=-1
    )
    height = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + height - PLATFORM_DEPTH / 2
    platform.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def platform_top_height(env: ManagerBasedRLEnv, platform_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm")):
    """Height of the platform top above the env origin [m]."""
    return env.scene[platform_cfg.name].data.root_pos_w.torch[:, 2] + PLATFORM_DEPTH / 2 - env.scene.env_origins[:, 2]


def fingers_open(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[finger_cfg.name]
    return finger_half_width(env, finger_cfg) > 0.030


class platform_coverage(ManagerTermBase):
    """Supported platform coverage in [0, 1], per step; logs per-episode coverage at reset.

    Coverage is the sheet task's measure (``surface_coverage``): the share of a 16 x 16 grid on the
    platform top covered by cloth triangles lying within 6 mm of the expected cloth height. Logged:

    - ``Coverage/best``: best coverage at any point of the episode (held or not);
    - ``Coverage/best_released``: best coverage with the sheet released (unheld, fingers open);
    - ``Coverage/final``: coverage on the episode's last step;
    - ``Coverage/released``: share of episodes in which the sheet was released after extraction.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.triangles = sheet_triangles(cfg.params["resolution"], env.device)
        self.samples = surface_samples(cfg.params["platform_size"], (16, 16), env.device)
        n = env.num_envs
        self.coverage = torch.zeros(n, device=env.device)
        self.best = torch.zeros(n, device=env.device)
        self.best_released = torch.zeros(n, device=env.device)
        self.released = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        log["Coverage/best"] = self.best[ids].mean()
        log["Coverage/best_released"] = self.best_released[ids].mean()
        log["Coverage/final"] = self.coverage[ids].mean()
        log["Coverage/released"] = self.released[ids].float().mean()
        for buffer in (self.coverage, self.best, self.best_released, self.released):
            buffer[ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        resolution,
        platform_size,
        top_offset: float,
        particle_clearance: float = 0.002,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
        finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
    ) -> torch.Tensor:
        sheet, target = env.scene[asset_cfg.name], env.scene[target_cfg.name]
        nodes = sheet.data.nodal_pos_w.torch
        quat = target.data.root_quat_w.torch[:, None, :].expand(-1, nodes.shape[1], -1)
        local = quat_apply_inverse(
            quat.reshape(-1, 4), (nodes - target.data.root_pos_w.torch[:, None]).reshape(-1, 3)
        ).reshape_as(nodes)
        self.coverage = surface_coverage(local, self.triangles, self.samples, top_offset + particle_clearance)
        free = _phase_reached(env) & ~_is_holding(env) & fingers_open(env, finger_cfg)
        self.released |= free
        self.best = torch.maximum(self.best, self.coverage)
        self.best_released = torch.where(free, torch.maximum(self.best_released, self.coverage), self.best_released)
        return self.coverage


def _coverage_state(env: ManagerBasedRLEnv) -> platform_coverage:
    """The live :class:`platform_coverage` reward-term instance (the twist tasks all carry one)."""
    return env.reward_manager.get_term_cfg("coverage").func


class _commanded_release(ManagerTermBase):
    """Base of the two placement outcomes: the episode's single commanded gripper opening.

    The gripper is a binary command channel, and an episode is one close-then-open cycle (user
    request). The first closing command is the grasp; the first opening command after it is the
    release, and that release ends the episode either way -- as ``placed`` if the sheet covers the
    platform at that moment, as ``dropped`` if it does not. Opening commands before any closure are
    the approach with an open hand and are ignored, and since the release ends the episode there is
    no second closure to be had: closed once, opened once.

    The edge is read from the applied command (``raw_actions`` against the action term's threshold)
    rather than from the fingers, so the outcome is settled on the step the policy asks to let go,
    not the tens of milliseconds later when the fingers have physically parted.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.was_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.closed_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.was_open[ids] = True
        self.closed_once[ids] = False

    def _released(self, env: ManagerBasedRLEnv, gripper_action_name: str) -> torch.Tensor:
        """This step's opening edge: an open command that follows a closing one."""
        gripper = env.action_manager.get_term(gripper_action_name)
        command = gripper.raw_actions[:, 0]
        cfg = gripper.cfg
        opened = command > cfg.threshold if cfg.positive_threshold else command < cfg.threshold
        released = opened & ~self.was_open & self.closed_once
        self.closed_once |= ~opened
        self.was_open.copy_(opened)
        return released


def _well_placed(env: ManagerBasedRLEnv, coverage_threshold: float, max_angle: float) -> torch.Tensor:
    """Is the sheet both covered enough and square enough for the release to count as a success?

    Two independent tests, both of which must hold (user request): at least ``coverage_threshold``
    of the platform covered, and the sheet's marked edge within ``max_angle`` degrees of a platform
    edge. A sheet dumped across the platform corner-on is not a placement however much of the top
    it happens to cover, and past ``max_angle`` the geometry caps coverage anyway -- a 30 degree
    yaw cannot exceed 0.844 even perfectly centred and perfectly flat.

    Coverage is the ``coverage`` reward term's value, and the reward manager runs after the
    termination manager within a step, so it is one control step (20 ms) old here; the angle is
    measured live. The sheet is still in the fingers on that step -- this asks whether it was
    already lying square on the platform when the policy let go, not where it ends up afterwards.
    """
    covered = _coverage_state(env).coverage >= coverage_threshold
    return covered & (sheet_yaw_error(env) <= max_angle)


class sheet_placed(_commanded_release):
    """Success: the episode's one release, taken covered enough and square enough -- see
    :func:`_well_placed`."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        coverage_threshold: float = 0.5,
        max_angle: float = 20.0,
        gripper_action_name: str = "gripper_action",
    ) -> torch.Tensor:
        return self._released(env, gripper_action_name) & _well_placed(env, coverage_threshold, max_angle)


class sheet_dropped(_commanded_release):
    """Failure: the episode's one release, taken either under-covered or turned past ``max_angle``.

    The complement of :class:`sheet_placed` on the same step, so every release resolves to exactly
    one outcome and the episode always ends there.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        coverage_threshold: float = 0.5,
        max_angle: float = 20.0,
        gripper_action_name: str = "gripper_action",
    ) -> torch.Tensor:
        return self._released(env, gripper_action_name) & ~_well_placed(env, coverage_threshold, max_angle)


class lowered_without_release(ManagerTermBase):
    """Failure: the carried sheet was brought down to the platform with the gripper still shut.

    The release has a floor as well as a verdict (user request): once the tool point descends to
    ``min_height`` above the platform top while the sheet is still held and no opening command has
    been applied, the episode ends as a failure. Without it the policy can lower the sheet onto the
    platform, collect the holding-side coverage budget and simply never let go -- an ending that
    costs nothing, since there is no alive cost and a time-out is free.

    Armed only once the tool point has been *above* the line while carrying, so the step after
    extraction does not fail an episode that starts below a tall platform's top (the slot sits at
    table level, the platform top up to 14 cm above it). The line is read from the same tool point
    and the same platform-local frame as ``released_high``: ``local z + top_offset``, so the
    student's ``top_offset`` of ``-PLATFORM_DEPTH / 2`` makes it a height above the moving top.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.armed[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float = 0.01,
        top_offset: float = -PLATFORM_DEPTH / 2,
        gripper_action_name: str = "gripper_action",
        hand_body_name: str = "panda_hand",
        grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.107),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    ) -> torch.Tensor:
        gripper = env.action_manager.get_term(gripper_action_name)
        command = gripper.raw_actions[:, 0]
        cfg = gripper.cfg
        opened = command > cfg.threshold if cfg.positive_threshold else command < cfg.threshold
        robot, target = env.scene[robot_cfg.name], env.scene[target_cfg.name]
        hand = robot.data.body_link_pose_w.torch[:, robot.body_names.index(hand_body_name)]
        offset = torch.tensor(grasp_offset, device=hand.device).expand(len(hand), 3)
        tip = hand[:, :3] + quat_apply(hand[:, 3:7], offset)
        local = quat_apply_inverse(target.data.root_quat_w.torch, tip - target.data.root_pos_w.torch)
        height = local[:, 2] + top_offset
        carrying = _phase_reached(env) & _is_holding(env)
        self.armed |= carrying & (height > min_height)
        # An opening command on this very step is the release: sheet_placed/sheet_dropped judge it.
        return self.armed & carrying & ~opened & (height <= min_height)


# -- the win-or-bleed accounting: together, these three terms make every episode that does not end
# in the ``placed`` success a net loss, however it ends, while a placed episode is a net win,
# however late it finishes. Terminations are computed before rewards within a step, so the flags
# read here are this step's, and a terminal step's reward still reaches the learner.


def alive_cost(env: ManagerBasedRLEnv, cost: float) -> torch.Tensor:
    """A constant charge per control step, start to end of the episode.

    In literal reward units (the manager's dt is undone): ``cost`` per step, so an episode that
    runs its full length pays ``cost x max_episode_length``.
    """
    return torch.full((env.num_envs,), -cost / env.step_dt, device=env.device)


def unfinished_cost(env: ManagerBasedRLEnv, cost: float, term_name: str = "placed") -> torch.Tensor:
    """On any episode ending other than ``placed``, charge the alive cost of the remaining steps.

    This closes the early-exit escape: without it, a cheap failure (dropping the sheet, a high
    release, diving below the table) stops the per-step bleed, and a policy that doubts it can
    finish is paid to end the episode fast. With it, every non-placed ending costs exactly the
    full-episode alive total, so nothing beats finishing and nothing is gained by dying sooner.
    A time-out charges ~nothing here because the alive cost has already been paid in full.
    Literal units, the manager's dt undone; pass the same ``cost`` as :func:`alive_cost`.
    """
    done = env.termination_manager.dones
    placed = env.termination_manager.get_term(term_name)
    remaining = (env.max_episode_length - env.episode_length_buf).clamp_min(0).float()
    return -cost * remaining * (done & ~placed) / env.step_dt


def placed_success_bonus(env: ManagerBasedRLEnv, bonus: float, term_name: str = "placed") -> torch.Tensor:
    """A flat one-time pay on the terminal step of the ``placed`` success termination.

    Flat rather than coverage-graded so the win-or-bleed inequality holds at the 0.5-coverage
    margin: ``placed`` already demands half the platform covered for a full second, and the
    coverage shaping elsewhere still grades quality above that bar. Literal units, dt undone.
    """
    placed = env.termination_manager.get_term(term_name)
    return bonus * placed / env.step_dt


class release_edge_angle(ManagerTermBase):
    """How square the sheet lands: the angle its marked bottom edge makes with the platform's edges.

    A diagnostic, not a reward -- it is carried as a zero-weight reward term because that is where
    this task keeps its per-step measurements (see ``coverage``). On the step the release ends the
    episode it measures the horizontal angle between the sheet's bottom edge and each of the four
    edges of the platform top, and keeps the four of them plus their minimum. The minimum is the
    one number that matters: the platform top is a square, so its four edges point at 0, 90, 180
    and 270 degrees and the smallest angle to any of them is the yaw error of the placement, in
    [0, 45] degrees. Zero is a sheet laid square with the platform.

    "Bottom edge" is the one the red dot sits on, found by :class:`SheetEdgeMarker`'s rule and at
    the same moment: the border whose nodes are lowest on the episode's first step, while the sheet
    still stands in its slot, latched for the rest of the episode. The edge's direction is taken
    from its two end nodes.

    Logs ``Placement/release_angle_min`` (degrees, averaged over the episodes that released) and,
    when the run is small enough to watch -- at most ``verbose_max_envs`` environments, so playback
    prints and training does not -- a line per release:

        [release] env 0  placed    coverage 0.62  edge angles 12.4 / 77.6 / 167.6 / 102.4 deg  min 12.4
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Same node grid as SheetEdgeMarker: row-major, one node more than elements per direction.
        cols, rows = cfg.params["resolution"][0] + 1, cfg.params["resolution"][1] + 1
        grid = torch.arange(rows * cols, device=env.device).view(rows, cols)
        borders = [grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]
        self._border_nodes = borders
        self._ends = torch.stack([torch.stack((ids[0], ids[-1])) for ids in borders])
        self._asset_name = cfg.params.get("asset_cfg", SceneEntityCfg("deformable")).name
        self._target_name = cfg.params.get("target_cfg", SceneEntityCfg("mannequin_arm")).name
        self.border = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.angles = torch.zeros(env.num_envs, 4, device=env.device)
        self.min_angle = torch.zeros(env.num_envs, device=env.device)
        self.measured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        measured = self.measured[ids]
        log["Placement/release_angle_min"] = (
            (self.min_angle[ids] * measured).sum() / measured.sum().clamp_min(1)
        )
        log["Placement/release_angle_measured"] = measured.float().mean()
        self.border[ids] = -1
        self.angles[ids] = 0.0
        self.min_angle[ids] = 0.0
        self.measured[ids] = False

    def current_angles(self, env: ManagerBasedRLEnv, asset_name=None, target_name=None) -> torch.Tensor:
        """This step's angles to the four platform edges, in degrees. Shape ``(num_envs, 4)``.

        Latches the dot's border on first use in an episode, then measures from the live scene, so
        a caller gets the current value whatever order the reward terms run in.
        """
        nodes = env.scene[asset_name or self._asset_name].data.nodal_pos_w.torch
        stale = self.border < 0
        if bool(stale.any()):  # the marker's rule: the lowest border of the upright sheet
            heights = torch.stack([nodes[:, ids, 2].mean(dim=-1) for ids in self._border_nodes], dim=1)
            self.border = torch.where(stale, heights.argmin(dim=1), self.border)

        ends = self._ends[self.border.clamp_min(0)]
        rows = torch.arange(len(nodes), device=nodes.device)
        edge = nodes[rows, ends[:, 1], :2] - nodes[rows, ends[:, 0], :2]
        # The platform top is a square: its four edges run along +x, +y, -x, -y of its own frame.
        quat = env.scene[target_name or self._target_name].data.root_quat_w.torch
        axis = torch.zeros(len(quat), 3, device=quat.device)
        axis[:, 0] = 1.0
        along = quat_apply(quat, axis)[:, :2]
        edges = torch.stack((along, torch.stack((-along[:, 1], along[:, 0]), dim=-1)), dim=1)
        edges = torch.cat((edges, -edges), dim=1)  # +x, +y, -x, -y
        cosine = torch.einsum("ni,nei->ne", torch.nn.functional.normalize(edge, dim=-1),
                              torch.nn.functional.normalize(edges, dim=-1))
        return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        resolution,
        verbose_max_envs: int = 16,
        success_name: str = "placed",
        failure_name: str = "dropped",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        target_cfg: SceneEntityCfg = SceneEntityCfg("mannequin_arm"),
    ) -> torch.Tensor:
        angles = self.current_angles(env, asset_cfg.name, target_cfg.name)
        done = env.termination_manager
        placed = done.get_term(success_name)
        released = placed | done.get_term(failure_name)
        self.angles = torch.where(released[:, None], angles, self.angles)
        self.min_angle = torch.where(released, angles.amin(dim=1), self.min_angle)
        self.measured |= released
        if env.num_envs <= verbose_max_envs and bool(released.any()):
            coverage = _coverage_state(env).coverage
            for i in released.nonzero(as_tuple=False).flatten().tolist():
                print(f"[release] env {i}  {'placed ' if placed[i] else 'dropped'}  "
                      f"coverage {coverage[i]:.2f}  edge angles "
                      + " / ".join(f"{a:.1f}" for a in angles[i].tolist())
                      + f" deg  min {angles[i].min():.1f}", flush=True)
        return torch.zeros(env.num_envs, device=env.device)


def sheet_yaw_error(env: ManagerBasedRLEnv, term_name: str = "release_angle") -> torch.Tensor:
    """Angle between the sheet's marked edge and the nearest edge of the platform top [deg].

    In ``[0, 45]`` -- the platform top is square, so no placement is ever further than 45 degrees
    from one of its edges. Read live from :class:`release_edge_angle`, so terminations (which run
    before the reward manager) and reward terms all see this step's value.
    """
    return env.reward_manager.get_term_cfg(term_name).func.current_angles(env).amin(dim=1)


def sheet_squareness(env: ManagerBasedRLEnv, term_name: str = "release_angle") -> torch.Tensor:
    """``cos^8`` of the angle between the sheet's marked edge and the nearest platform edge.

    1.0 for a sheet laid square with the platform, collapsing to ``cos^8 45 = 0.06`` for one laid
    corner-on (user request). Every coverage and success payment is multiplied by it, so coverage
    reached with the sheet turned is worth very little -- squaring up is no longer a refinement on
    the way to coverage, it is most of the task:

    ======== ========
    yaw      factor
    ======== ========
    0 deg    1.000
    5 deg    0.970
    10 deg   0.885
    15 deg   0.758
    20 deg   0.608
    30 deg   0.316
    45 deg   0.063
    ======== ========

    The eighth power rather than the plain cosine, which only reaches 0.707 at the worst yaw a
    square top allows and is far too gentle to steer against the coverage budget. Note how sharp
    this is: half the pay is gone by 22 degrees. Measured live from :class:`release_edge_angle`,
    which is why the order the reward terms run in does not matter.
    """
    return torch.cos(torch.deg2rad(sheet_yaw_error(env, term_name))) ** 8


def placed_coverage_bonus(
    env: ManagerBasedRLEnv,
    bonus: float = 300000.0,
    gain: float = 8.0,
    term_name: str = "placed",
) -> torch.Tensor:
    """Pay the success exponentially in the coverage the release was taken with (user request).

    ``bonus x exp(gain x (coverage - 1))`` on the terminal step of ``term_name``, so ``bonus`` is
    what a perfect placement is worth and every point of coverage below that divides the pay by
    ``exp(gain / 100)``. At the defaults each extra 0.1 of coverage multiplies the pay by ``e^0.8``
    -- about 2.2x -- which is what turns "clear the bar" into "cover as much as you can":

    ======== ============== ==============
    coverage literal        in the return
    ======== ============== ==============
    0.50     5495           157
    0.70     27215          778
    0.85     90358          2582
    1.00     300000         8571
    ======== ============== ==============

    A flat bar pays the same for 0.51 as for 0.99; this does not. Read from the ``coverage`` reward
    term, which is the value the termination judged the release on as long as this term is ordered
    before it. Literal units, the manager's dt undone.
    """
    placed = env.termination_manager.get_term(term_name)
    coverage = _coverage_state(env).coverage
    pay = bonus * torch.exp(gain * (coverage - 1.0)) * sheet_squareness(env)
    return pay * placed / env.step_dt


class placement_success_on_placed(platform_placement):
    """``platform_placement`` whose success component is paid by the task's real outcome.

    The parent's success test -- settled coverage plus a retreat-and-hold of the fingertips that
    the twist policies never perform -- never fires here, so its bonus never paid
    (``PlacementReward/success`` read 0.0 in every run). This override pays ``success_bonus`` flat
    on the terminal step of the ``placed`` termination instead, so an episode that ends in success
    logs the full 5000 there.

    It also multiplies that success and every coverage payment by :func:`sheet_squareness`, the
    eighth power of the cosine of the sheet's angle to the nearest platform edge (user request), so
    coverage collected with the sheet turned pays between 6% and 100% of its face value. Every other component and
    all bookkeeping are the parent's.
    """

    COVERAGE_COMPONENTS = ("coverage_progress", "released_coverage_progress", "released_coverage")
    """The parent's coverage payments, all scaled by :func:`sheet_squareness` (user request)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.placed_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.completed_count = torch.zeros((), device=env.device)
        self.placed_count = torch.zeros((), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        done = self._env.termination_manager.dones[ids]
        success = self.placed_seen[ids].clone()
        self.completed_count += done.sum()
        self.placed_count += (success & done).sum()
        super().reset(env_ids)
        log = self._env.extras.setdefault("log", {})
        log["Placement/success"] = success.float().mean()
        log["Outcomes/completed_count"] = self.completed_count.clone()
        log["Outcomes/placed_count"] = self.placed_count.clone()
        log["Outcomes/success_rate"] = self.placed_count / self.completed_count.clamp_min(1)

    # wraps: the manager validates a term's params against inspect.signature of its __call__, and a
    # bare (*args, **kwargs) override reads as "mandatory parameters ['args', 'kwargs']" and is
    # refused; carrying the parent's signature keeps the parent's validation.
    @functools.wraps(platform_placement.__call__)
    def __call__(self, env, *args, **kwargs):
        reward = super().__call__(env, *args, **kwargs)
        placed = env.termination_manager.get_term("placed")
        self.placed_seen |= placed
        square = sheet_squareness(env)
        pay = kwargs.get("success_bonus", 5000.0) * placed.float() * square
        delta = pay - self.components["success"]
        self.components["success"] = pay
        self.component_totals["success"].add_(delta)
        # Coverage reached with the sheet turned is worth less than the same coverage laid square.
        # Rewritten after the fact, as the success is, so the logged components carry the factor.
        for name in self.COVERAGE_COMPONENTS:
            scaled = self.components[name] * square
            change = scaled - self.components[name]
            self.components[name] = scaled
            self.component_totals[name].add_(change)
            delta = delta + change
        return reward + delta / env.step_dt


# ---------------------------------------------------------------------------------------------
# action-scale randomisation (user request): the IK step per unit action differs between episodes


def randomize_action_scale(env, env_ids, scale_range=None, action_name: str = "arm_action") -> None:
    """Reset event: draw the IK action term's scale uniformly from ``scale_range`` for ``env_ids``.

    The term's ``_scale`` is a per-environment tensor, so every environment keeps its own draw for
    the whole episode. ``None`` leaves the configured scale alone, so the term can sit in the
    config and be switched on from the command line
    (``env.events.randomize_arm_scale.params.scale_range=[0.04,0.1]``).
    """
    if scale_range is None:
        return
    term = env.action_manager.get_term(action_name)
    low, high = float(scale_range[0]), float(scale_range[1])
    draws = torch.rand(len(env_ids), 1, device=env.device) * (high - low) + low
    term._scale[env_ids] = draws


def set_action_scale(env, scale: float, action_name: str = "arm_action", env_ids=None) -> torch.Tensor:
    """Force the IK action term's scale (all environments unless ``env_ids``); returns the previous values."""
    term = env.action_manager.get_term(action_name)
    ids = slice(None) if env_ids is None else env_ids
    previous = term._scale[ids].clone()
    term._scale[ids] = float(scale)
    return previous
