"""First-phase (grasp and extraction) staged reward and strict motion-smoothness costs.

Design goals, in priority order:

1. Converge from random weights quickly. Reaching and grasping are paid by a small per-step
   *state* reward (at most 0.5) built from observable quantities, so the critic can fit it; any
   hold pays more than every pre-grasp state. Lifting is paid by potential-based shaping, so
   raising the sheet gives immediate, strong credit without a hold-still plateau.
2. Never prefer stalling to finishing. Holding still near the top is slightly negative under
   the lift shaping, and the extraction bonus (``grasp_stage``) exceeds ``0.5 / (1 - gamma)``,
   the value of holding at the grasp forever.
3. Make smoothness the strict part. Motion costs are envelopes: slow, gradual motion costs almost
   nothing (so exploration is not frozen), while fast approaches, abrupt acceleration and closing
   the gripper on the move cost more than rushing can gain. The costs ramp in with the observed
   extraction rate so the grasp is found first and then refined.

All values returned are literal per-control-step amounts (divided by ``step_dt`` here, because the
reward manager multiplies by ``weight * dt``).
"""

import math

import torch
from isaaclab.managers import ManagerTermBase
from .rewards import _is_holding, _phase_reached, grasp_debug


def huber_cost(value):
    """Quadratic near zero, linear for outliers; avoids unbounded quadratic spikes."""
    value = value.abs()
    return torch.where(value < 1, 0.5 * value.square(), value - 0.5)


def stage_value(dist_to_edge, top_dist, width, facing_down, square_to_slot, holding,
                reach_weight=0.25, align_weight=0.05, close_weight=0.10, hold_weight=0.10,
                open_width=0.04, closed_width=0.012, min_hold_width=0.001):
    """Per-step state reward: at most 0.4 before the grasp, 0.5 while holding.

    Pre-grasp: reach the middle of the top edge with a broad and a fine scale, point down and
    square to the slot once close, and close the fingers only when the top edge is between them.
    Fingers shut on nothing (width at or below ``min_hold_width``) earn no closure credit.
    Holding pays the full pre-grasp credit plus a small premium. The premium is deliberately
    small: a 0.9-per-step hold plateau made holding without lifting a stable optimum when this
    task was trained from scratch. Lifting is paid separately by :func:`lift_potential`.
    """
    reach = 0.5 * (1 - torch.tanh(dist_to_edge / 0.25)) + 0.5 * (1 - torch.tanh(dist_to_edge / 0.04))
    align = torch.exp(-dist_to_edge / 0.10) * 0.5 * (facing_down + square_to_slot)
    closure = ((open_width - width) / (open_width - closed_width)).clamp(0, 1)
    close = torch.exp(-top_dist / 0.02) * closure * (width > min_hold_width)
    pre = reach_weight * reach + align_weight * align + close_weight * close
    held = torch.full_like(pre, reach_weight + align_weight + close_weight + hold_weight)
    return torch.where(holding, held, pre)


def lift_potential(lowest_z, edge_rise, holding, clear_height=0.17, edge_travel=0.20,
                   edge_share=0.5, scale=100.0):
    """Lift potential while held: part top-edge rise, part lowest-node clearance.

    The pinched top edge rises with the fingers immediately, but the sheet's lowest node only
    leaves the table after 1-4 cm of slack is taken up (median 2.8 cm in recorded extractions).
    A potential on the lowest node alone gave no signal across that gap, and no from-scratch run
    lifted. The edge share supplies the immediate signal; the rest still requires extraction.
    """
    edge = (edge_rise / edge_travel).clamp(0, 1)
    bottom = (lowest_z / clear_height).clamp(0, 1)
    return scale * (edge_share * edge + (1 - edge_share) * bottom) * holding


def speed_limit(top_dist, holding, far_speed=0.40, near_speed=0.05, near_scale=0.08, lift_speed=0.25):
    """Allowed hand speed: slows to ``near_speed`` at the edge, ``lift_speed`` while holding."""
    approach = near_speed + (far_speed - near_speed) * (1 - torch.exp(-top_dist / near_scale))
    return torch.where(holding, torch.full_like(approach, lift_speed), approach)


class grasp_progress(ManagerTermBase):
    """Staged grasp state reward plus potential-based lift shaping.

    Lift is paid as ``gamma * phi(s') - phi(s)`` with ``phi`` from :func:`lift_potential`
    (0 to 100) while holding: about 2.5 per centimetre the pinched edge rises plus about 3 per
    centimetre the whole sheet clears, returned if the sheet is dropped, and slightly
    negative for holding still (``(gamma - 1) * phi``), so it cannot be farmed and does not
    create a plateau. The term pays through the step extraction is reached, then stops.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.potential = torch.zeros(env.num_envs, device=env.device)
        self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.edge_rest = torch.zeros(env.num_envs, device=env.device)
        self.sums = torch.zeros(env.num_envs, 4, device=env.device)
        self.steps = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault('log', {})
        lived = self.steps[ids].clamp_min(1)
        log['Stage/state_per_step'] = (self.sums[ids, 0] / lived).mean()
        log['Stage/lift_shaping'] = self.sums[ids, 1].mean()
        log['Stage/holding_fraction'] = (self.sums[ids, 2] / lived).mean()
        log['Stage/max_lift'] = self.sums[ids, 3].mean()
        self.sums[ids] = 0
        self.steps[ids] = 0
        self.potential[ids] = 0
        self.finished[ids] = False

    def __call__(self, env, clear_height=0.17, lift_scale=100.0, gamma=0.99, min_hold_width=0.001,
                 edge_travel=0.20, edge_share=0.5, terminal_potential_zero=False):
        debug = grasp_debug(env)
        holding = _is_holding(env)
        state = stage_value(debug['dist_to_edge'], debug['top_dist'], debug['width'],
                            debug['facing_down'], debug['square_to_slot'], holding,
                            min_hold_width=min_hold_width)
        edge = debug['edge_z'] - env.scene.env_origins[:, 2]
        first = self.steps == 0  # the sheet is at rest on the first step after a reset
        self.edge_rest[first] = edge[first]
        potential = lift_potential(debug['lowest_z'], edge - self.edge_rest, holding, clear_height,
                                   edge_travel, edge_share, lift_scale)
        if terminal_potential_zero:
            # Extraction ends this shaping phase. True terminals also have zero
            # potential; time-limit truncations retain it for value bootstrapping.
            terminal = _phase_reached(env) | env.termination_manager.terminated
            potential = torch.where(terminal, torch.zeros_like(potential), potential)
        shaping = gamma * potential - self.potential
        self.potential.copy_(potential)
        active = ~self.finished
        self.finished |= _phase_reached(env)
        state, shaping = state * active, shaping * active
        self.sums[:, 0] += state
        self.sums[:, 1] += shaping
        self.sums[:, 2] += holding.float()
        self.sums[:, 3] = torch.maximum(self.sums[:, 3], potential / lift_scale)
        self.steps += 1
        return (state + shaping) / env.step_dt


def lift_curve_scale(first_pay=500.0, first_height=0.01, total_pay=5000.0, travel=0.20):
    """Solve ``P(h) = C * (1 - exp(-h / L))`` with ``P(first_height) = first_pay`` and
    ``P(travel) = total_pay``. Returns ``(C, L)``.

    500 of 5000 in the first centimetre of a 20 cm travel is twice the linear share, so the
    per-millimetre pay starts high and decays exponentially; a growing exponential cannot meet
    both anchors.
    """
    share = first_pay / total_pay
    if not first_height / travel < share < 1:
        raise ValueError('The first-step share must exceed the linear share for a decaying curve')
    ratio = lambda length: (1 - math.exp(-first_height / length)) / (1 - math.exp(-travel / length))
    low, high = 1e-4, 1e3  # ratio falls from ~1 toward first_height / travel as length grows
    for _ in range(200):
        mid = (low * high) ** 0.5
        low, high = (mid, high) if ratio(mid) > share else (low, mid)
    length = (low * high) ** 0.5
    return total_pay / (1 - math.exp(-travel / length)), length


class grasp_lift_milestones(ManagerTermBase):
    """Large grasp and lift milestones, in addition to the staged reward and lift shaping.

    * ``grasp_bonus`` once per episode, on the first real top-edge grasp.
    * Lift pay ``P(best edge rise)`` on new episode-best height of the pinched top edge while
      holding: ``first_pay`` for the first centimetre, ``total_pay`` by extraction (any remainder
      is paid on the extraction step). Best-so-far, so bobbing and regrasping cannot repeat it.
    * ``still_penalty`` per step while holding with the edge rising less than ``still_rise``.
    * ``release_penalty`` when a hold ends before extraction having raised the edge less than
      ``release_lift``. A release counts after ``release_steps`` consecutive steps without a
      hold, so a one- or two-step flicker of the hold detector is not charged.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.curve_scale, self.curve_length = lift_curve_scale(
            p.get('first_pay', 500.0), p.get('first_height', 0.01), p.get('total_pay', 5000.0),
            p.get('edge_travel', 0.20))
        n, dev = env.num_envs, env.device
        self.steps = torch.zeros(n, device=dev)
        self.edge_rest = torch.zeros(n, device=dev)
        self.last_edge = torch.zeros(n, device=dev)
        self.best = torch.zeros(n, device=dev)
        self.paid = torch.zeros(n, device=dev)
        self.hold_start = torch.zeros(n, device=dev)
        self.hold_best = torch.zeros(n, device=dev)
        self.in_hold = torch.zeros(n, dtype=torch.bool, device=dev)
        self.unheld = torch.zeros(n, dtype=torch.long, device=dev)
        self.bonus_paid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
        self.totals = torch.zeros(n, 4, device=dev)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault('log', {})
        for i, name in enumerate(('grasp_bonus', 'lift_pay', 'still_penalty', 'release_penalty')):
            log[f'Milestone/{name}'] = self.totals[ids, i].mean()
        log['Milestone/best_edge_rise'] = self.best[ids].mean()
        for buffer in (self.steps, self.edge_rest, self.last_edge, self.best, self.paid,
                       self.hold_start, self.hold_best, self.totals):
            buffer[ids] = 0
        for flag in (self.in_hold, self.bonus_paid, self.finished):
            flag[ids] = False
        self.unheld[ids] = 0

    def curve(self, height):
        return self.curve_scale * (1 - torch.exp(-height / self.curve_length))

    def __call__(self, env, grasp_bonus=400.0, first_pay=500.0, first_height=0.01, total_pay=5000.0,
                 edge_travel=0.20, still_penalty=0.5, still_rise=0.0005, release_penalty=400.0,
                 release_lift=0.01, release_steps=3):
        debug = grasp_debug(env)
        holding = _is_holding(env)
        phase = _phase_reached(env)
        edge = debug['edge_z'] - env.scene.env_origins[:, 2]
        first = self.steps == 0
        self.edge_rest[first] = edge[first]
        self.last_edge[first] = edge[first]
        rise = (edge - self.edge_rest).clamp(0, edge_travel)
        active = ~self.finished

        bonus = grasp_bonus * (debug['hit'] & ~self.bonus_paid & active)
        self.bonus_paid |= debug['hit']

        # Lift pay on new episode-best edge height while holding; top up to the total at extraction.
        self.best = torch.where(holding & active, torch.maximum(self.best, rise), self.best)
        target = torch.where(phase & active, torch.full_like(self.paid, total_pay), self.curve(self.best))
        lift_pay = (target - self.paid).clamp_min(0) * active
        self.paid += lift_pay

        step_rise = edge - self.last_edge
        still = still_penalty * (holding & active & ~phase & (step_rise < still_rise))

        # Per-hold bookkeeping for the release penalty.
        started = holding & ~self.in_hold
        self.hold_start = torch.where(started, edge, self.hold_start)
        self.hold_best = torch.where(started, torch.zeros_like(self.hold_best), self.hold_best)
        self.in_hold |= holding
        self.hold_best = torch.where(holding, torch.maximum(self.hold_best, edge - self.hold_start),
                                     self.hold_best)
        self.unheld = torch.where(holding, torch.zeros_like(self.unheld), self.unheld + self.in_hold.long())
        released = self.in_hold & (self.unheld >= release_steps)
        release = release_penalty * (released & active & ~phase & (self.hold_best < release_lift))
        self.in_hold &= ~released

        self.finished |= phase
        self.last_edge.copy_(edge)
        self.steps += 1
        self.totals += torch.stack((bonus, lift_pay, -still, -release), -1)
        return (bonus + lift_pay - still - release) / env.step_dt


class gripper_closure_penalty(ManagerTermBase):
    """Charge ``penalty`` for every commanded gripper closure after the first, before extraction.

    A closure is the policy's raw gripper action switching from open to closed, read before the
    gripper filter, using the gripper term's own threshold. Episodes start open, so an initial
    closed action is the first (free) closure. Exploration noise that flips the raw sign also
    counts, which pushes the policy toward a decisive gripper command.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        name = cfg.params.get('action_name', 'gripper_action')
        manager = env.action_manager
        position = manager.active_terms.index(name)
        self.column = sum(manager.action_term_dim[:position])
        self.gripper = manager.get_term(name)
        self.was_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault('log', {})
        log['Events/commanded_closures'] = self.count[ids].float().mean()
        self.was_open[ids] = True
        self.count[ids] = 0

    def __call__(self, env, penalty=50.0, free_closures=1, action_name='gripper_action'):
        raw = env.action_manager.action[:, self.column]
        cfg = self.gripper.cfg
        is_open = raw > cfg.threshold if cfg.positive_threshold else raw < cfg.threshold
        closed = self.was_open & ~is_open & ~_phase_reached(env)
        self.was_open.copy_(is_open)
        self.count += closed
        charged = closed & (self.count > free_closures)
        return -penalty * charged.float() / env.step_dt


MOTION_NAMES = ('speed', 'angular', 'hand_acc', 'joint_acc', 'jerk', 'command', 'closure', 'joint_limit')


class grasp_motion_cost(ManagerTermBase):
    """Strict smoothness costs, per control step, scaled in by the extraction success rate.

    * ``speed``: hand speed above an envelope that slows to 5 cm/s at the top edge and 25 cm/s
      while lifting. Produces a decelerating approach instead of hitting the edge at speed.
    * ``angular``: hand angular speed above 1 rad/s.
    * ``hand_acc``: Cartesian hand acceleration (control-rate difference of velocity).
    * ``joint_acc`` / ``jerk``: measured arm joint acceleration and its rate of change.
    * ``command``: change of the *applied* (filtered) arm command.
    * ``closure``: hand translation/rotation while the fingers are closing near the sheet.
    * ``joint_limit``: arm joints inside the last ``joint_limit_margin`` (10%) of their range
      at either end, ``sum(p^2)`` with ``p`` = 0 at the zone edge, 1 at the joint limit and above
      1 beyond it. Uses the simulator's joint position limits.

    Reset discontinuities are excluded from the differenced terms. Finger joints are excluded.
    With ``phase_one_only=True``, every cost stops after extraction, including joint-limit costs.
    Otherwise an optional ``phase_two_scale`` replaces the extraction curriculum after
    extraction (including phase-three retreat), with a cap on the summed per-step cost.
    ``components`` are raw diagnostics; ``applied_components`` include the scale and cap.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.ids = [i for i, n in enumerate(self.robot.joint_names) if n.startswith('panda_joint')]
        self.hand = self.robot.body_names.index('panda_hand')
        n = env.num_envs
        self.last_acc = torch.zeros(n, len(self.ids), device=env.device)
        self.last_vel = torch.zeros(n, 3, device=env.device)
        self.last_cmd = torch.zeros(n, len(self.ids), device=env.device)
        self.age = torch.zeros(n, dtype=torch.long, device=env.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.total = torch.zeros(n, len(MOTION_NAMES), device=env.device)
        self.components = {}
        self.applied_components = {}
        self.phase_two_total = torch.zeros_like(self.total)
        self.success_rate = 0.0
        self.scale = torch.tensor(cfg.params.get('min_scale', 0.05), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault('log', {})
        for i, name in enumerate(MOTION_NAMES):
            log[f'MotionCost/{name}'] = self.total[ids, i].mean()
            log[f'MotionCostPhase2/{name}'] = self.phase_two_total[ids, i].mean()
        ended = self.age[ids] > 0
        if bool(ended.any()):
            # Batch extraction rate folded into a slow running average (about 1000 episodes).
            finished = int(ended.sum())
            rate = float(self.success[ids][ended].float().mean())
            alpha = 1 - 0.999 ** finished
            self.success_rate += alpha * (rate - self.success_rate)
        log['Curriculum/success_rate'] = torch.tensor(self.success_rate)
        log['Curriculum/motion_scale'] = self.scale.clone()
        self.total[ids] = 0
        self.phase_two_total[ids] = 0
        self.last_acc[ids] = 0
        self.last_vel[ids] = 0
        self.last_cmd[ids] = 0
        self.age[ids] = 0
        self.success[ids] = False

    def __call__(self, env, min_scale=0.05, ramp_start=0.2, ramp_end=0.8,
                 speed_weight=2.0, angular_weight=0.2, hand_acc_weight=2.5, joint_acc_weight=0.5,
                 jerk_weight=0.2, command_weight=0.1, closure_weight=1.0,
                 joint_limit_weight=5.0, joint_limit_margin=0.10,
                 speed_tolerance=0.5, hand_acc_reference=1.5, joint_acc_reference=5.0,
                 jerk_reference=100.0, command_reference=0.2, angular_limit=1.0, phase_one_only=False,
                 phase_two_scale=None, phase_two_max_cost=5.0, phase_two_speed_limit=0.15):
        debug = grasp_debug(env)
        holding = _is_holding(env)
        phase = _phase_reached(env)
        self.success |= phase
        progress = min(max((self.success_rate - ramp_start) / (ramp_end - ramp_start), 0.0), 1.0)
        self.scale.fill_(min_scale + (1 - min_scale) * progress)

        velocity = self.robot.data.body_link_vel_w.torch[:, self.hand]
        linear, angular = velocity[:, :3], velocity[:, 3:].norm(dim=-1)
        speed = linear.norm(dim=-1)
        qdd = self.robot.data.joint_acc.torch[:, self.ids]
        cmd = env.action_manager.get_term('arm_action').raw_actions[:, :len(self.ids)]
        self.age += 1
        first, second = self.age > 1, self.age > 2

        # After extraction (placement task) the lifting limit applies whether or not it is held.
        limit = speed_limit(debug['top_dist'], holding | phase)
        if phase_two_scale is not None:
            limit = torch.where(phase, phase_two_speed_limit, limit)
        # Relative to the local limit, so 12 cm/s over costs far more at the edge (limit 5 cm/s)
        # than in free space (40 cm/s). An absolute scale let time pressure win near the edge.
        speed_cost = huber_cost((speed - limit).clamp_min(0) / (speed_tolerance * limit))
        angular_cost = huber_cost((angular - angular_limit).clamp_min(0) / 0.5)
        hand_acc = (linear - self.last_vel).norm(dim=-1) / env.step_dt
        hand_acc_cost = huber_cost(hand_acc / hand_acc_reference) * first
        joint_acc_cost = huber_cost(qdd / joint_acc_reference).mean(-1) * first
        jerk = (qdd - self.last_acc) / env.step_dt
        jerk_cost = huber_cost(jerk / jerk_reference).mean(-1) * second
        command_cost = huber_cost((cmd - self.last_cmd) / command_reference).mean(-1)
        # Penetration into the warning zone near either joint limit: 0 at its edge, 1 at the limit.
        q = self.robot.data.joint_pos.torch[:, self.ids]
        bounds = self.robot.data.joint_pos_limits.torch[:, self.ids]
        low, high = bounds[..., 0], bounds[..., 1]
        zone = joint_limit_margin * (high - low)
        clearance = torch.minimum(q - low, high - q)
        joint_limit_cost = ((zone - clearance) / zone).clamp_min(0).square().sum(-1)

        gripper = env.action_manager.get_term('gripper_action')
        raw = gripper.raw_actions[:, 0]
        opening = raw > gripper.cfg.threshold if gripper.cfg.positive_threshold else raw < gripper.cfg.threshold
        closing = ~opening & (debug['width'] > 0.012) & ~phase
        near = torch.exp(-debug['top_dist'] / 0.08)
        closure_cost = closing * near * (huber_cost(speed / 0.03) + huber_cost(angular / 0.3))

        self.last_vel.copy_(linear)
        self.last_acc.copy_(qdd)
        self.last_cmd.copy_(cmd)
        costs = torch.stack((speed_cost, angular_cost, hand_acc_cost, joint_acc_cost, jerk_cost,
                             command_cost, closure_cost, joint_limit_cost), -1)
        weights = torch.tensor((speed_weight, angular_weight, hand_acc_weight, joint_acc_weight,
                                jerk_weight, command_weight, closure_weight, joint_limit_weight),
                               device=env.device)
        weighted = costs * weights
        if phase_one_only:
            weighted = torch.where(phase[:, None], 0.0, weighted)
        self.components = dict(zip(MOTION_NAMES, weighted.unbind(-1)))
        self.total += weighted
        applied = weighted * self.scale
        if phase_two_scale is not None:
            # Fixed, bounded phase-two cost; independent of the extraction curriculum.
            phase_two = weighted * phase_two_scale
            cap = (phase_two_max_cost / phase_two.sum(-1).clamp_min(1e-6)).clamp(max=1.0)
            phase_two = phase_two * cap[:, None]
            applied = torch.where(phase[:, None], phase_two, applied)
        self.applied_components = dict(zip(MOTION_NAMES, applied.unbind(-1)))
        self.phase_two_total += applied * phase[:, None]
        return applied.sum(-1) / env.step_dt
