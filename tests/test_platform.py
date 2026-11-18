"""Run with the Isaac Lab interpreter: python -m unittest discover -s tests -v."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils.hydra import resolve_presets
from sheet_rl.tasks.manager_based.sheet_rl.mdp.platform import platform_opened_high, platform_placement, PlatformPoseCommand
from sheet_rl.tasks.manager_based.sheet_rl.mdp.platform_geometry import (
    sheet_triangles, surface_coverage, surface_samples,
)
from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import (
    _drape_success, _high_release, _is_holding, _phase_reached, ee_motion_while_closing,
    exponential_progress_potential,
)
from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env_cfg import SheetRlEnvCfg
from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_platform_env_cfg import (
    SheetRlPlatformEnvCfg, SheetRlPlatformGraspEnvCfg,
)
from sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_mdp import (
    lowered_without_release, placed_coverage_bonus, placement_success_on_placed,
    release_edge_angle, sheet_dropped, sheet_placed,
)


def flat_nodes(size=(0.2, 0.2), resolution=(8, 8)):
    x = torch.linspace(-size[0] / 2, size[0] / 2, resolution[0] + 1)
    y = torch.linspace(-size[1] / 2, size[1] / 2, resolution[1] + 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), 0.012)), -1)[None]


class CoverageTests(unittest.TestCase):
    def coverage(self, nodes, size=(0.2, 0.2), resolution=(8, 8)):
        return surface_coverage(nodes, sheet_triangles(resolution, "cpu"),
                                surface_samples(size, (16, 16), "cpu"), 0.012)

    def test_full_and_half_coverage(self):
        nodes = flat_nodes().repeat(2, 1, 1)
        nodes[1, :, 0] += 0.1
        torch.testing.assert_close(self.coverage(nodes), torch.tensor([1.0, 0.5]))

    def test_hover_table_and_outside_do_not_cover(self):
        nodes = flat_nodes().repeat(3, 1, 1)
        nodes[0, :, 2] += 0.04
        nodes[1, :, 2] -= 0.02
        nodes[2, :, 0] += 0.3
        torch.testing.assert_close(self.coverage(nodes), torch.zeros(3))

    def test_overlapping_fold_counts_once(self):
        nodes = flat_nodes()
        nodes[..., 0] = nodes[..., 0].abs()
        self.assertAlmostEqual(self.coverage(nodes).item(), 0.5)

    def test_vertical_sheet_has_no_area(self):
        nodes = flat_nodes()
        nodes[..., 2] = nodes[..., 0] + 0.15
        nodes[..., 0] = 0
        self.assertEqual(self.coverage(nodes).item(), 0)

    def test_nonsquare_sheet(self):
        size, resolution = (0.3, 0.16), (6, 4)
        self.assertEqual(self.coverage(flat_nodes(size, resolution), size, resolution).item(), 1)


class GraspRewardTests(unittest.TestCase):
    def test_exponential_progress_is_normalized_and_accelerating(self):
        progress = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        potential = exponential_progress_potential(progress, 2.0)
        self.assertEqual(potential[0], 0.0)
        self.assertEqual(potential[-1], 1.0)
        increments = potential.diff()
        self.assertTrue(torch.all(increments[1:] > increments[:-1]))

    def test_zero_sharpness_is_linear(self):
        progress = torch.linspace(0, 1, 5)
        torch.testing.assert_close(exponential_progress_potential(progress, 0.0), progress)

    def test_ee_motion_penalty_only_while_commanded_close_and_not_closed(self):
        action = NS(
            raw_actions=torch.tensor([[-1.0], [1.0], [-1.0]]),
            cfg=NS(threshold=0.0, positive_threshold=True),
        )
        velocity = torch.zeros(3, 1, 6)
        velocity[:, 0, 0] = 0.2
        velocity[:, 0, 5] = 1.0
        robot = NS(
            body_names=["panda_hand"],
            data=NS(
                joint_pos=NS(torch=torch.tensor([[0.02, 0.02], [0.02, 0.02], [0.005, 0.005]])),
                body_link_vel_w=NS(torch=velocity),
            ),
        )
        env = NS(
            num_envs=3,
            device="cpu",
            step_dt=0.1,
            scene={"robot": robot},
            action_manager=NS(get_term=lambda _: action),
        )
        params = {
            "robot_cfg": SceneEntityCfg("robot"),
            "finger_cfg": SceneEntityCfg("robot", joint_ids=[0, 1]),
        }
        term = ee_motion_while_closing(
            RewardTermCfg(func=ee_motion_while_closing, params=params), env
        )
        value = term(env, **params)
        # 0.2 m/s + 0.1 m/rad * 1 rad/s, divided by the 0.1 s control step.
        torch.testing.assert_close(value, torch.tensor([3.0, 0.0, 0.0]))


class CommandedReleaseTests(unittest.TestCase):
    """One gripper cycle per episode: the opening after the closure ends it, coverage decides how."""

    def setUp(self):
        n = 4
        self.gripper = NS(raw_actions=torch.full((n, 1), -1.0), cfg=NS(threshold=0.0, positive_threshold=True))
        self.coverage = NS(coverage=torch.tensor([0.0, 0.49, 0.5, 0.9]))
        self.angles = torch.zeros(n, 4)  # square with the platform unless a test turns the sheet
        terms = {"coverage": self.coverage,
                 "release_angle": NS(current_angles=lambda env: self.angles)}
        self.env = NS(num_envs=n, device="cpu", step_dt=0.02,
                      action_manager=NS(get_term=lambda _: self.gripper),
                      reward_manager=NS(get_term_cfg=lambda name: NS(func=terms[name])))
        self.placed = sheet_placed(TerminationTermCfg(func=sheet_placed), self.env)
        self.dropped = sheet_dropped(TerminationTermCfg(func=sheet_dropped), self.env)

    def step(self, command):
        """One control step at ``command`` (+1 opens, -1 closes); the manager calls both terms."""
        self.gripper.raw_actions[:] = torch.as_tensor(command, dtype=torch.float).reshape(-1, 1)
        return self.placed(self.env), self.dropped(self.env)

    def test_open_hand_before_any_closure_is_the_approach(self):
        for _ in range(3):
            placed, dropped = self.step(+1)
            self.assertFalse(placed.any())
            self.assertFalse(dropped.any())

    def test_release_after_the_grasp_ends_the_episode_and_coverage_decides(self):
        self.step(-1)  # the grasp
        placed, dropped = self.step(+1)  # the release
        self.assertEqual(placed.tolist(), [False, False, True, True])  # >= 0.5 covered
        self.assertEqual(dropped.tolist(), [True, True, False, False])
        self.assertTrue((placed | dropped).all())  # every release resolves ...
        self.assertFalse((placed & dropped).any())  # ... to exactly one outcome

    def test_a_sheet_turned_past_the_angle_limit_is_not_a_success_however_covered(self):
        self.angles = torch.full((4, 4), 20.1)  # just past the 20 degree limit
        self.step(-1)
        placed, dropped = self.step(+1)
        self.assertFalse(placed.any())  # env 3 covers 0.9 and still fails
        self.assertTrue(dropped.all())  # every release still resolves, and ends the episode

    def test_the_angle_limit_is_inclusive_and_coverage_still_has_to_hold(self):
        self.angles = torch.full((4, 4), 20.0)
        self.step(-1)
        placed, dropped = self.step(+1)
        self.assertEqual(placed.tolist(), [False, False, True, True])  # exactly as with no yaw
        self.assertEqual(dropped.tolist(), [True, True, False, False])

    def test_outcome_fires_on_the_edge_only_not_while_the_hand_stays_open(self):
        self.step(-1)
        placed, dropped = self.step(+1)
        self.assertTrue((placed | dropped).all())  # the release step itself
        for _ in range(3):
            placed, dropped = self.step(+1)
            self.assertFalse(placed.any())
            self.assertFalse(dropped.any())

    def test_closing_again_without_opening_does_not_fire(self):
        for command in (-1, -1, -1):
            placed, dropped = self.step(command)
            self.assertFalse((placed | dropped).any())

    def test_reset_starts_a_fresh_cycle(self):
        self.step(-1)
        self.step(+1)  # episode one's release
        for term in (self.placed, self.dropped):
            term.reset([0, 1])
        placed, dropped = self.step(+1)  # a reset env opens with no closure behind it
        self.assertFalse((placed | dropped)[:2].any())
        self.assertFalse(self.placed.closed_once[:2].any())
        self.assertTrue(self.placed.closed_once[2:].all())  # untouched envs keep their state

    def test_negative_open_threshold(self):
        self.gripper.cfg.positive_threshold = False  # open is now the command below the threshold
        placed, dropped = self.step(+1)  # closed
        self.assertFalse((placed | dropped).any())
        placed, dropped = self.step(-1)  # opened
        self.assertEqual(placed.tolist(), [False, False, True, True])
        self.assertEqual(dropped.tolist(), [True, True, False, False])


class ReleaseFloorTests(unittest.TestCase):
    """Carrying the sheet down to the platform without letting go ends the episode as a failure."""

    TOP_OFFSET = -0.10  # -PLATFORM_DEPTH / 2, the student's setting

    def setUp(self):
        n = 3
        quat = quat_from_euler_xyz(torch.zeros(n), torch.zeros(n), torch.zeros(n))
        pose = torch.cat((torch.zeros(n, 3), quat), dim=-1)[:, None]
        self.robot = NS(body_names=["panda_hand"], data=NS(body_link_pose_w=NS(torch=pose)))
        self.target = NS(data=NS(root_pos_w=NS(torch=torch.zeros(n, 3)), root_quat_w=NS(torch=quat)))
        self.gripper = NS(raw_actions=torch.full((n, 1), -1.0), cfg=NS(threshold=0.0, positive_threshold=True))
        self.env = NS(num_envs=n, device="cpu", step_dt=0.02,
                      scene={"robot": self.robot, "mannequin_arm": self.target},
                      action_manager=NS(get_term=lambda _: self.gripper))
        self.term = lowered_without_release(TerminationTermCfg(func=lowered_without_release), self.env)
        _phase_reached(self.env)[:] = True  # extracted ...
        _is_holding(self.env)[:] = True  # ... and still carrying

    def step(self, height, command=-1.0):
        """One step with the tool point ``height`` above the platform top (0.107 m past the hand)."""
        self.robot.data.body_link_pose_w.torch[:, 0, 2] = (
            torch.as_tensor(height) - 0.107 - self.TOP_OFFSET)
        self.gripper.raw_actions[:] = command
        return self.term(self.env, min_height=0.01, top_offset=self.TOP_OFFSET)

    def test_carry_starting_below_the_line_is_not_failed_until_it_has_been_above_it(self):
        for _ in range(3):  # the slot sits below a tall platform's top
            self.assertFalse(self.step(-0.05).any())
        self.step(0.05)  # arms
        self.assertTrue(self.step(-0.05).all())

    def test_coming_down_onto_the_platform_still_shut_fails(self):
        self.assertFalse(self.step(0.08).any())
        self.assertFalse(self.step(0.02).any())
        self.assertTrue(self.step(0.005).all())

    def test_the_line_is_inclusive(self):
        self.step(0.05)
        self.assertEqual(self.step([0.0101, 0.01, 0.0099]).tolist(), [False, True, True])

    def test_an_opening_command_on_that_step_is_a_release_not_a_floor_failure(self):
        self.step(0.05)
        self.assertFalse(self.step(0.005, command=+1.0).any())  # placed/dropped judge this one
        self.assertTrue(self.step(0.005).all())  # shut again, and still down there

    def test_nothing_fires_once_the_sheet_is_no_longer_carried(self):
        self.step(0.05)
        _is_holding(self.env)[:] = False
        self.assertFalse(self.step(0.005).any())
        _is_holding(self.env)[:] = True
        _phase_reached(self.env)[:] = False
        self.assertFalse(self.step(0.005).any())

    def test_reset_disarms(self):
        self.step(0.05)
        self.term.reset([0, 1])
        fired = self.step(0.005)
        self.assertEqual(fired.tolist(), [False, False, True])


class ExponentialSuccessPayTests(unittest.TestCase):
    """The success pay grows exponentially with the coverage the release was taken with."""

    def setUp(self):
        self.coverage = NS(coverage=torch.tensor([0.5, 0.7, 0.85, 1.0]))
        self.placed = torch.ones(4, dtype=torch.bool)
        self.angles = torch.zeros(4, 4)  # laid square with the platform
        terms = {"coverage": self.coverage,
                 "release_angle": NS(current_angles=lambda env: self.angles)}
        self.env = NS(num_envs=4, device="cpu", step_dt=0.02,
                      termination_manager=NS(get_term=lambda _: self.placed),
                      reward_manager=NS(get_term_cfg=lambda name: NS(func=terms[name])))

    def literal(self, **kwargs):
        return placed_coverage_bonus(self.env, **kwargs) * self.env.step_dt

    def test_pay_is_exponential_in_coverage_and_full_at_full_coverage(self):
        pay = self.literal()
        torch.testing.assert_close(pay, 300000.0 * torch.exp(8.0 * (self.coverage.coverage - 1.0)))
        self.assertAlmostEqual(pay[3].item(), 300000.0, places=3)  # bonus is the perfect placement
        self.assertAlmostEqual(pay[2].item(), 90358.0, delta=5.0)  # 0.85 -> 16x what 0.5 pays
        self.assertAlmostEqual((pay[2] / pay[0]).item(), math.exp(8 * 0.35), places=3)

    def test_every_extra_tenth_of_coverage_multiplies_the_pay(self):
        self.coverage.coverage = torch.tensor([0.6, 0.7, 0.8, 0.9])
        pay = self.literal()
        for near, far in zip(pay[:-1], pay[1:]):
            self.assertAlmostEqual((far / near).item(), math.exp(0.8), places=4)

    def test_nothing_is_paid_on_a_step_that_is_not_the_success(self):
        self.placed = torch.zeros(4, dtype=torch.bool)
        torch.testing.assert_close(self.literal(), torch.zeros(4))

    def test_the_pay_is_scaled_by_the_eighth_power_of_the_cosine_of_the_sheet_angle(self):
        square = self.literal()
        self.angles = torch.full((4, 4), 45.0)  # laid corner-on: the worst a square top allows
        torch.testing.assert_close(self.literal(), square * 0.0625)  # a sixteenth of face value
        self.angles = torch.full((4, 4), 20.0)
        torch.testing.assert_close(self.literal(), square * math.cos(math.radians(20)) ** 8)

    def test_scale_and_steepness_are_tunable(self):
        flat = self.literal(bonus=5000.0, gain=0.0)  # the old flat bar, for comparison
        torch.testing.assert_close(flat, torch.full((4,), 5000.0))


class ReleaseEdgeAngleTests(unittest.TestCase):
    """The angle the sheet's marked bottom edge makes with the four edges of the platform top."""

    def setUp(self):
        self.placed = torch.ones(1, dtype=torch.bool)
        self.dropped = torch.zeros(1, dtype=torch.bool)
        self.nodes = flat_nodes().clone()  # 9 x 9 grid, z = 0.012 everywhere
        self.platform = NS(data=NS(root_quat_w=NS(torch=self.yaw_quat(0.0))))
        self.env = NS(num_envs=1, device="cpu", step_dt=0.02, extras={},
                      scene={"deformable": NS(data=NS(nodal_pos_w=NS(torch=self.nodes))),
                             "mannequin_arm": self.platform},
                      termination_manager=NS(get_term=lambda n: self.placed if n == "placed" else self.dropped),
                      reward_manager=NS(get_term_cfg=lambda _: NS(func=NS(coverage=torch.tensor([0.9])))))
        self.term = release_edge_angle(
            RewardTermCfg(func=release_edge_angle, params={"resolution": (8, 8)}), self.env)

    @staticmethod
    def yaw_quat(degrees):
        z = torch.tensor([math.radians(degrees)])
        return quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), z)

    def lay_sheet(self, yaw, low_border=0):
        """A flat sheet rotated by ``yaw`` degrees, with one border dropped so the dot latches there."""
        flat = flat_nodes()[0]
        angle = math.radians(yaw)
        x, y = flat[:, 0].clone(), flat[:, 1].clone()
        self.nodes[0, :, 0] = x * math.cos(angle) - y * math.sin(angle)
        self.nodes[0, :, 1] = x * math.sin(angle) + y * math.cos(angle)
        self.nodes[0, :, 2] = 0.012
        grid = torch.arange(81).view(9, 9)
        self.nodes[0, [grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]][low_border], 2] -= 0.1

    def measure(self, **kwargs):
        self.term(self.env, resolution=(8, 8), verbose_max_envs=0, **kwargs)
        return self.term.angles[0].tolist(), self.term.min_angle[0].item()

    def test_a_square_placement_reads_zero(self):
        self.lay_sheet(0.0)
        angles, minimum = self.measure()
        self.assertEqual([round(a) for a in angles], [0, 90, 180, 90])
        self.assertAlmostEqual(minimum, 0.0, places=3)

    def test_a_skewed_placement_reads_its_yaw_error(self):
        self.lay_sheet(30.0)
        angles, minimum = self.measure()
        self.assertEqual([round(a) for a in angles], [30, 60, 150, 120])
        self.assertAlmostEqual(minimum, 30.0, places=3)

    def test_the_minimum_is_the_error_to_the_nearest_edge_so_never_above_45(self):
        self.lay_sheet(100.0)
        angles, minimum = self.measure()
        self.assertEqual([round(a) for a in angles], [100, 10, 80, 170])
        self.assertAlmostEqual(minimum, 10.0, places=3)

    def test_the_platform_yaw_is_taken_into_account(self):
        self.platform.data.root_quat_w.torch = self.yaw_quat(30.0)
        self.lay_sheet(30.0)  # square with a platform that is itself turned 30 degrees
        self.assertAlmostEqual(self.measure()[1], 0.0, places=3)

    def test_the_edge_is_the_lowest_border_latched_on_the_first_step(self):
        self.lay_sheet(0.0, low_border=2)  # grid[:, 0], the border running along y
        self.measure()
        self.assertEqual(int(self.term.border[0]), 2)
        self.assertAlmostEqual(self.term.min_angle[0].item(), 0.0, places=3)  # along y, still square
        self.lay_sheet(40.0, low_border=0)  # a different border is lowest now; the latch holds
        self.measure()
        self.assertEqual(int(self.term.border[0]), 2)

    def test_nothing_is_recorded_until_the_release(self):
        self.placed = torch.zeros(1, dtype=torch.bool)
        self.lay_sheet(30.0)
        self.assertEqual(self.measure()[1], 0.0)
        self.assertFalse(self.term.measured.any())
        self.placed = torch.ones(1, dtype=torch.bool)
        self.assertAlmostEqual(self.measure()[1], 30.0, places=3)
        self.assertTrue(self.term.measured.all())

    def test_reset_logs_the_episode_angle_and_clears_the_latch(self):
        self.lay_sheet(30.0)
        self.measure()
        self.term.reset()
        self.assertAlmostEqual(self.env.extras["log"]["Placement/release_angle_min"].item(), 30.0, places=3)
        self.assertEqual(self.env.extras["log"]["Placement/release_angle_measured"].item(), 1.0)
        self.assertEqual(int(self.term.border[0]), -1)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        nodes = flat_nodes().repeat(2, 1, 1)
        self.env = NS(num_envs=2, device="cpu", step_dt=1 / 30, extras={},
                      termination_manager=NS(dones=torch.zeros(2, dtype=torch.bool)))
        self.env.scene = {
            "deformable": NS(data=NS(nodal_pos_w=NS(torch=nodes), nodal_vel_w=NS(torch=torch.zeros_like(nodes)))),
            "mannequin_arm": NS(data=NS(root_pos_w=NS(torch=torch.zeros(2, 3)),
                                        root_quat_w=NS(torch=quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.zeros(2))))),
            "robot": NS(body_names=["panda_hand"],
                        data=NS(joint_pos=NS(torch=torch.full((2, 2), 0.04)),
                                body_link_vel_w=NS(torch=torch.zeros(2, 1, 6)),
                                body_link_pose_w=NS(torch=self.hand_pose(0.04)))),
        }
        self.gripper = NS(raw_actions=torch.full((2, 1), -1.0), cfg=NS(threshold=0.0, positive_threshold=True))
        self.env.action_manager = NS(active_terms=["arm_action", "gripper_action"], action_term_dim=[7, 1],
                                     action=torch.full((2, 8), -2.4), get_term=lambda name: self.gripper)
        self.angles = torch.zeros(2, 4)  # laid square unless a test turns the sheet
        self.env.reward_manager = NS(
            get_term_cfg=lambda _: NS(func=NS(current_angles=lambda env: self.angles)))
        self.params = {"resolution": (8, 8), "platform_size": (0.2, 0.2),
                       "finger_cfg": SceneEntityCfg("robot", joint_ids=[0, 1])}
        self.term = platform_placement(RewardTermCfg(func=platform_placement, params=self.params), self.env)
        self.abort = platform_opened_high(TerminationTermCfg(func=platform_opened_high), self.env)

    def step(self):
        self.env.termination_manager.dones.copy_(self.abort(self.env))
        return self.term(self.env, **self.params) * self.env.step_dt

    @staticmethod
    def hand_pose(elevation):
        """Hand pose whose fingertips (identity rotation, +0.1034 m) are ``elevation`` above the table."""
        pose = torch.zeros(2, 1, 7)
        pose[..., 6] = 1.0  # identity quaternion, (x, y, z, w) layout
        pose[..., 2] = elevation - 0.01 - 0.1034  # table is 0.01 below the platform centre
        return pose

    def set_elevation(self, elevation):
        self.env.scene["robot"].data.body_link_pose_w.torch[:] = self.hand_pose(elevation)

    def part(self, name):
        return self.term.components[name]

    def hold(self):
        _is_holding(self.env)[:] = True
        self.env.scene["robot"].data.joint_pos.torch[:] = 0.002
        self.gripper.raw_actions[:] = -1.0
        return self.step()

    def let_go(self):
        _is_holding(self.env)[:] = False
        self.env.scene["robot"].data.joint_pos.torch[:] = 0.04
        self.gripper.raw_actions[:] = 1.0
        return self.step()

    def test_no_placement_reward_before_extraction(self):
        torch.testing.assert_close(self.step(), torch.zeros(2))
        self.assertFalse(_drape_success(self.env).any())

    def test_coverage_and_success_pay_scale_with_how_square_the_sheet_is(self):
        _phase_reached(self.env)[:] = True
        self.env.termination_manager.get_term = lambda _: torch.ones(2, dtype=torch.bool)

        def pay(angle):
            self.angles = torch.full((2, 4), angle)
            term = placement_success_on_placed(
                RewardTermCfg(func=placement_success_on_placed, params=self.params), self.env)
            term(self.env, **self.params)
            return term.components["coverage_progress"].clone(), term.components["success"].clone()

        coverage, success = pay(0.0)
        self.assertGreater(coverage[0].item(), 0.0)  # there is something to scale
        self.assertAlmostEqual(success[0].item(), 5000.0, places=3)
        turned_coverage, turned_success = pay(45.0)
        torch.testing.assert_close(turned_coverage, coverage * 0.0625)  # cos^8 45 = 1/16
        torch.testing.assert_close(turned_success, success * 0.0625)

    def test_student_success_metric_and_counts_follow_placed_not_old_retreat_flag(self):
        placed = torch.tensor([True, False])
        self.env.termination_manager.get_term = lambda name: placed
        term = placement_success_on_placed(RewardTermCfg(func=placement_success_on_placed,
                                                        params=self.params), self.env)
        term.reset()  # Initial scene reset is not a completed episode.
        self.assertEqual(term.completed_count.item(), 0)
        term(self.env, **self.params)
        self.assertFalse(_drape_success(self.env).any())
        torch.testing.assert_close(term.components['success'], torch.tensor([5000., 0.]))
        self.env.termination_manager.dones[:] = True
        term.reset([0])
        self.assertEqual(self.env.extras['log']['Placement/success'].item(), 1.)
        term.reset([1])
        self.assertEqual(term.completed_count.item(), 2)
        self.assertEqual(term.placed_count.item(), 1)
        self.assertEqual(self.env.extras['log']['Outcomes/success_rate'].item(), .5)

    def test_approach_pays_progress_only(self):
        nodes = self.env.scene["deformable"].data.nodal_pos_w.torch
        nodes[..., 2] += 0.4
        _phase_reached(self.env)[:] = True
        self.hold()
        torch.testing.assert_close(self.part("approach"), torch.zeros(2))
        nodes[..., 2] -= 0.2
        self.step()
        expected = 500 * (math.exp(-0.2 / 0.2) - math.exp(-0.4 / 0.2))
        torch.testing.assert_close(self.part("approach"), torch.full((2,), expected), atol=1e-3, rtol=0)
        self.step()
        torch.testing.assert_close(self.part("approach"), torch.zeros(2))
        nodes[..., 2] += 0.1
        self.step()
        nodes[..., 2] -= 0.1
        self.step()
        torch.testing.assert_close(self.part("approach"), torch.zeros(2), atol=1e-4, rtol=0)

    def test_holding_has_no_standing_reward(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        for _ in range(80):
            torch.testing.assert_close(self.step(), torch.zeros(2))
        self.assertFalse(_drape_success(self.env).any())

    def test_lowering_only_pays_new_bests_near_target_while_held(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.2)
        self.hold()
        self.set_elevation(0.1)
        self.step()
        self.assertTrue((self.part("lower") > 0).all())
        self.set_elevation(0.2)
        self.step()
        self.set_elevation(0.1)
        self.step()
        torch.testing.assert_close(self.part("lower"), torch.zeros(2))
        self.let_go()
        self.set_elevation(0.03)
        self.step()
        torch.testing.assert_close(self.part("lower"), torch.zeros(2))

    def test_low_release_pays_once_and_more_coverage_pays_more(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.03)
        # Equal horizontal alignment; one sheet hovers beyond contact tolerance.
        self.env.scene["deformable"].data.nodal_pos_w.torch[1, :, 2] += 0.04
        self.hold()
        self.let_go()
        torch.testing.assert_close(self.part("release"), torch.tensor([1500., 375.]), atol=1e-2, rtol=0)
        self.assertTrue(self.term.released.all())
        self.hold()
        self.let_go()
        torch.testing.assert_close(self.part("release"), torch.zeros(2))

    def test_lower_release_below_ceiling_is_better(self):
        _phase_reached(self.env)[:] = True
        poses = self.env.scene["robot"].data.body_link_pose_w.torch
        poses[0] = self.hand_pose(0.04)[0]
        poses[1] = self.hand_pose(0.09)[1]
        self.hold()
        reward = self.let_go()
        self.assertGreater(self.part("release")[0], self.part("release")[1])
        self.assertGreater(self.part("release")[1], 0)
        self.assertTrue((reward >= 0).all())
        self.assertFalse(_high_release(self.env).any())

    def test_release_near_platform_is_better_than_release_beside_it(self):
        _phase_reached(self.env)[:] = True
        self.env.scene["deformable"].data.nodal_pos_w.torch[1, :, 0] += 0.4
        self.hold()
        self.let_go()
        self.assertGreater(self.part("release")[0], 100 * self.part("release")[1])

    def test_delayed_open_uses_height_at_hold_loss(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.2)
        self.hold()
        _is_holding(self.env)[:] = False
        self.step()  # hold lost, fingers still closed
        self.set_elevation(0.03)
        self.let_go()
        expected = 1500 * math.exp(-(0.2 - 0.03) / 0.08)
        torch.testing.assert_close(self.part("release"), torch.full((2,), expected), atol=1e-3, rtol=0)

    def test_opening_progress_has_a_finite_budget_and_uses_filtered_action(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.03)
        self.hold()
        self.env.action_manager.action[:, 7] = 2.4  # raw request alone earns nothing
        self.step()
        torch.testing.assert_close(self.part("opening_progress"), torch.zeros(2))
        total = torch.zeros(2)
        for state in (-0.8, -0.4, 0., 0.5, -1., 1.):
            self.gripper.raw_actions[:] = state
            self.step()
            total += self.part("opening_progress")
        torch.testing.assert_close(total, torch.full((2,), 100.))
        self.step()
        torch.testing.assert_close(self.part("opening_progress"), torch.zeros(2))

    def test_settling_after_release_gets_coverage_credit(self):
        _phase_reached(self.env)[:] = True
        nodes = self.env.scene["deformable"].data.nodal_pos_w.torch
        nodes[..., 2] += 0.04
        self.hold()
        self.let_go()
        torch.testing.assert_close(self.part("released_coverage"), torch.zeros(2))
        nodes[..., 2] -= 0.04
        self.step()
        torch.testing.assert_close(self.part("coverage_progress"), torch.full((2,), 1500.))
        torch.testing.assert_close(self.part("released_coverage_progress"), torch.full((2,), 1000.))
        torch.testing.assert_close(self.part("released_coverage"), torch.full((2,), 5.))
        self.hold()
        torch.testing.assert_close(self.part("released_coverage"), torch.zeros(2))

    def test_success_needs_release_and_one_second_settled(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        for _ in range(80):
            self.step()
        self.assertFalse(_drape_success(self.env).any())
        self.let_go()
        self.set_elevation(0.22)  # 20 cm above the 2 cm platform top
        for _ in range(29):
            self.step()
        self.assertFalse(_drape_success(self.env).any())
        self.step()
        torch.testing.assert_close(self.part("success"), torch.full((2,), 5000.))
        self.assertTrue(_drape_success(self.env).all())
        self.step()
        torch.testing.assert_close(self.part("success"), torch.zeros(2))

    def test_moving_cloth_cannot_succeed_and_resets_dwell(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.let_go()
        self.set_elevation(0.22)
        for _ in range(20):
            self.step()
        self.env.scene["deformable"].data.nodal_vel_w.torch[:, 0, 2] = 0.1
        self.step()
        torch.testing.assert_close(self.term.dwell, torch.zeros(2, dtype=torch.long))
        self.assertFalse(_drape_success(self.env).any())

    def test_unheld_without_a_recorded_release_cannot_succeed(self):
        _phase_reached(self.env)[:] = True
        for _ in range(80):
            self.step()
        self.assertFalse(_drape_success(self.env).any())
        torch.testing.assert_close(self.part("released_coverage"), torch.zeros(2))

    def test_partial_reset_preserves_other_environment(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.let_go()
        self.term.reset([0])
        self.assertFalse(self.term.released[0])
        self.assertTrue(self.term.released[1])
        self.assertEqual(self.term.best_coverage.tolist(), [0., 1.])
        self.assertEqual(self.term.dwell.tolist(), [0, 0])  # still at release pose, not retreat goal

    def test_high_opening_terminates_before_fingers_are_fully_open(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.101)
        self.hold()
        self.gripper.raw_actions[:] = 0.1
        # Fingers still pinching; the applied open command is already invalid.
        torch.testing.assert_close(self.step(), torch.zeros(2))
        self.assertTrue(self.env.termination_manager.dones.all())
        self.assertTrue(_high_release(self.env).all())
        self.assertFalse(_drape_success(self.env).any())
        self.abort.reset([0])
        self.assertEqual(_high_release(self.env).tolist(), [False, True])

    def test_high_opening_gate_is_phase_two_only_and_uses_filtered_command(self):
        self.set_elevation(0.2)
        self.hold()
        self.let_go()  # before extraction: no high-release ending
        self.assertFalse(_high_release(self.env).any())
        _phase_reached(self.env)[:] = True
        self.hold()
        self.env.action_manager.action[:, 7] = 2.4
        self.step()  # raw policy request changed; applied filter still closed
        self.assertFalse(_high_release(self.env).any())

    def test_delayed_physical_opening_above_ceiling_also_terminates(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.gripper.raw_actions[:] = 1.
        self.step()  # command opens low, but fingers have not moved yet
        self.assertFalse(_high_release(self.env).any())
        self.set_elevation(0.15)
        self.let_go()
        self.assertTrue(self.env.termination_manager.dones.all())
        torch.testing.assert_close(self.part("release"), torch.zeros(2))

    def test_opening_at_ceiling_and_then_retreating_is_allowed(self):
        _phase_reached(self.env)[:] = True
        self.set_elevation(0.10)
        self.hold()
        self.let_go()
        self.assertFalse(_high_release(self.env).any())
        self.set_elevation(0.22)
        self.step()
        self.assertFalse(_high_release(self.env).any())

    def test_off_target_penalty_is_horizontal_gated_and_capped(self):
        nodes = self.env.scene["deformable"].data.nodal_pos_w.torch
        nodes[0, :, 0] += 0.30
        nodes[1, :, 0] += 0.35
        self.step()
        torch.testing.assert_close(self.part("off_target"), torch.zeros(2))
        _phase_reached(self.env)[:] = True
        nodes[..., 2] += 1.0  # height does not change horizontal distance
        self.step()
        torch.testing.assert_close(self.part("off_target"), torch.tensor([0., -2.5]), atol=1e-4, rtol=0)
        nodes[..., 0] += 1.
        self.step()
        torch.testing.assert_close(self.part("off_target"), torch.full((2,), -5.))

    def test_settled_sheet_cannot_succeed_until_end_effector_retreats(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.let_go()
        for _ in range(40):
            self.step()
        self.assertFalse(_drape_success(self.env).any())
        self.assertEqual(self.term.dwell.tolist(), [0, 0])
        self.set_elevation(0.22)
        self.step()
        self.assertTrue((self.part("retreat_progress") > 0).all())
        self.assertTrue(self.term.retreat_reached.all())
        self.step()
        torch.testing.assert_close(self.part("retreat_progress"), torch.zeros(2))
        self.set_elevation(0.12)
        self.step()
        self.set_elevation(0.22)
        self.step()
        torch.testing.assert_close(self.part("retreat_progress"), torch.zeros(2))

    def test_retreat_goal_is_platform_centre_not_release_point(self):
        _phase_reached(self.env)[:] = True
        pose = self.env.scene["robot"].data.body_link_pose_w.torch
        pose[..., 0] = 0.15  # low release to the side of the centre
        self.hold()
        self.let_go()
        pose[..., 2] = self.hand_pose(0.22)[..., 2]  # directly above release is still off goal
        self.step()
        self.assertFalse(self.term.retreat_reached.any())
        pose[..., 0] = 0.
        self.step()
        self.assertTrue(self.term.retreat_reached.all())

    def test_phase_three_wandering_costs_even_when_sheet_is_well_placed(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.let_go()
        self.set_elevation(0.22)
        self.env.scene["robot"].data.body_link_pose_w.torch[..., 0] = 0.4
        self.step()
        torch.testing.assert_close(self.part("retreat_distance"), torch.full((2,), -5.))
        torch.testing.assert_close(self.part("off_target"), torch.zeros(2))
        self.assertFalse(_drape_success(self.env).any())

    def test_motion_at_retreat_target_costs_and_restarts_success_dwell(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        self.let_go()
        self.set_elevation(0.22)
        for _ in range(20):
            self.step()
        self.assertEqual(self.term.dwell.tolist(), [20, 20])
        velocity = self.env.scene["robot"].data.body_link_vel_w.torch
        velocity[0, 0, 0] = 0.20
        velocity[1, 0, 5] = 1.0  # rotation alone also prevents success
        self.step()
        self.assertTrue((self.part("retreat_motion") < 0).all())
        self.assertEqual(self.term.dwell.tolist(), [0, 0])
        velocity[:] = 0
        for _ in range(29):
            self.step()
        self.assertFalse(_drape_success(self.env).any())
        self.step()
        self.assertTrue(_drape_success(self.env).all())

    def test_component_logs_are_episode_totals_and_reset_per_environment(self):
        _phase_reached(self.env)[:] = True
        total = self.hold() + self.let_go() + self.step()
        expected = {name: value.clone() for name, value in self.term.component_totals.items()}
        torch.testing.assert_close(sum(expected.values()), total)
        self.term.reset([0])
        for name, value in expected.items():
            torch.testing.assert_close(self.env.extras["log"][f"PlacementReward/{name}"], value[0])
            self.assertEqual(self.term.component_totals[name][0], 0.)
            torch.testing.assert_close(self.term.component_totals[name][1], value[1])

    def test_terminal_failure_has_no_negative_placement_reward(self):
        _phase_reached(self.env)[:] = True
        self.env.termination_manager.dones[:] = True
        self.assertTrue((self.step() >= 0).all())
        self.assertNotIn("failure", self.term.components)

    def test_legacy_speed_and_table_costs_stop_after_extraction(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards import ee_speed_penalty, ee_table_clearance
        _phase_reached(self.env)[:] = torch.tensor([False, True])
        velocity = torch.zeros(2, 1, 6)
        velocity[:, 0, 0] = 1.
        self.env.scene["robot"].data.body_link_vel_w = NS(torch=velocity)
        value = ee_speed_penalty(self.env, phase_one_only=True)
        self.assertGreater(value[0], 0.)
        self.assertEqual(value[1], 0.)
        with patch("sheet_rl.tasks.manager_based.sheet_rl.mdp.rewards._grasp_site",
                   return_value=(torch.zeros(2, 3), None)):
            value = ee_table_clearance(self.env, phase_one_only=True)
            torch.testing.assert_close(value, torch.tensor([1., 0.]))
            torch.testing.assert_close(ee_table_clearance(self.env), torch.ones(2))

    def test_translation_does_not_change_reward(self):
        _phase_reached(self.env)[:] = True
        self.hold()
        shift = torch.tensor([1.2, -0.7, 2.0])
        self.env.scene["deformable"].data.nodal_pos_w.torch[1] += shift
        self.env.scene["mannequin_arm"].data.root_pos_w.torch[1] += shift
        self.env.scene["robot"].data.body_link_pose_w.torch[1, :, :3] += shift
        reward = self.let_go()
        torch.testing.assert_close(reward[0], reward[1], atol=1e-2, rtol=0)

    def test_command_uses_platform_pose_and_robot_frame(self):
        target = self.env.scene["mannequin_arm"].data
        target.root_pos_w.torch[:] = torch.tensor([0.5, -0.2, 0.01])
        target.root_quat_w.torch[:] = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.full((2,), torch.pi / 2))
        command = PlatformPoseCommand.__new__(PlatformPoseCommand)
        command._env = self.env
        command.cfg = NS(platform_name="mannequin_arm", top_offset=0.01)
        command.robot = NS(data=NS(root_pos_w=NS(torch=torch.tensor([[0.1, 0., 0.]]).repeat(2, 1)),
                                  root_quat_w=NS(torch=quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.zeros(2)))))
        command.pose_command_b = torch.zeros(2, 7)
        command._resample_command(torch.arange(2))
        torch.testing.assert_close(command.pose_command_b[:, :3], torch.tensor([[0.4, -0.2, 0.02]]).repeat(2, 1))
        torch.testing.assert_close(command.pose_command_b[:, 3:], target.root_quat_w.torch)


class ConfigTests(unittest.TestCase):
    def test_scene_and_landmarks_preserved_with_new_grasp_rewards(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.grasp_control import SmoothRelativeJointPositionActionCfg, applied_policy_actions
        base = resolve_presets(SheetRlEnvCfg())
        cfg = resolve_presets(SheetRlPlatformEnvCfg())
        self.assertEqual(cfg.scene.mannequin_arm.spawn.size, (0.2, 0.2, 0.02))
        self.assertEqual(cfg.scene.deformable.to_dict(), base.scene.deformable.to_dict())
        self.assertIs(cfg.observations.policy.actions.func, applied_policy_actions)
        cfg.observations.policy.actions = base.observations.policy.actions
        self.assertEqual(cfg.observations.to_dict(), base.observations.to_dict())
        self.assertIsInstance(cfg.actions.arm_action, SmoothRelativeJointPositionActionCfg)
        self.assertEqual(cfg.actions.arm_action.scale, 0.02)
        self.assertEqual(cfg.edge_marker.to_dict(), base.edge_marker.to_dict())
        self.assertFalse(cfg.push.enabled)
        self.assertIsNone(cfg.events.randomize_arm_radius)
        for name in ("approach", "alignment", "square_progress", "lift_progress", "gripper_recommit", "drape_coverage"):
            self.assertIsNone(getattr(cfg.rewards, name))
        self.assertEqual(cfg.rewards.motion_cost.weight, -1.0)
        for name in ("ee_speed", "table_clearance", "grasp_stage"):
            self.assertTrue(getattr(cfg.rewards, name).params["phase_one_only"])
        self.assertFalse(cfg.rewards.motion_cost.params["phase_one_only"])
        self.assertEqual(cfg.rewards.motion_cost.params["phase_two_max_cost"], 5.0)
        self.assertEqual(cfg.terminations.released_high.params["max_height"], 0.10)
        self.assertEqual(cfg.rewards.placement.params["settle_time"], 1.0)
        self.assertEqual(cfg.rewards.grasp_progress.weight, 1.0)
        self.assertIsNone(cfg.terminations.released_early)

    def test_student_outcomes_are_one_gripper_cycle_judged_at_half_coverage(self):
        from sheet_rl.tasks.manager_based.sheet_twist.sheet_platform_twist_student_env_cfg import (
            RELEASE_COVERAGE, SheetPlatformTwistStudentEnvCfg,
        )
        cfg = SheetPlatformTwistStudentEnvCfg()
        self.assertEqual(RELEASE_COVERAGE, 0.5)
        self.assertIs(cfg.terminations.placed.func, sheet_placed)
        self.assertIs(cfg.terminations.dropped.func, sheet_dropped)
        for term in (cfg.terminations.placed, cfg.terminations.dropped):
            self.assertEqual(term.params, {"coverage_threshold": RELEASE_COVERAGE, "max_angle": 20.0})
            self.assertFalse(term.time_out)  # a real ending, not a truncation
        self.assertIs(cfg.rewards.release_angle.func, release_edge_angle)
        # Non-zero or the manager skips the term; it returns zeros, so it pays nothing regardless.
        self.assertNotEqual(cfg.rewards.release_angle.weight, 0.0)
        self.assertEqual(release_edge_angle(
            RewardTermCfg(func=release_edge_angle, params={"resolution": (8, 8)}),
            NS(num_envs=2, device="cpu")).angles.sum().item(), 0.0)
        self.assertIs(cfg.rewards.placed_coverage.func, placed_coverage_bonus)
        self.assertEqual(cfg.rewards.placed_coverage.params, {"bonus": 300000.0, "gain": 8.0})
        self.assertAlmostEqual(cfg.rewards.placed_coverage.weight, 1.0 / 35)  # the global divisor
        names = list(vars(cfg.rewards))
        self.assertLess(names.index("placed_coverage"), names.index("coverage"))  # reads the judged value
        self.assertIs(cfg.terminations.held_too_low.func, lowered_without_release)
        self.assertEqual(cfg.terminations.held_too_low.params["min_height"], 0.01)
        self.assertFalse(cfg.terminations.held_too_low.time_out)

    def test_grasp_only_has_same_first_phase_and_no_placement(self):
        cfg = SheetRlPlatformGraspEnvCfg()
        full = SheetRlPlatformEnvCfg()
        self.assertIsNone(cfg.rewards.placement)
        self.assertIsNone(cfg.terminations.draped)
        self.assertIsNotNone(cfg.terminations.extracted)
        for name in ('grasp_stage', 'grasp_progress', 'motion_cost'):
            self.assertEqual(getattr(cfg.rewards, name).to_dict(), getattr(full.rewards, name).to_dict())
        self.assertEqual(cfg.rewards.grasp_stage.params["first_grasp_bonus"], 10.0)
        # A shut, empty gripper (width 0) must not count as holding; a real pinch (>= 1.7 mm) must.
        self.assertEqual(cfg.rewards.grasp_stage.params["min_hold_width"], 0.001)
        self.assertEqual(cfg.rewards.grasp_progress.params["min_hold_width"], 0.001)
        # Must exceed the 0.5-per-step hold reward held forever at the grasp gamma of 0.99.
        self.assertGreater(cfg.rewards.grasp_stage.params["extraction_bonus"], 0.5 / (1 - 0.99))
        from sheet_rl.tasks.manager_based.sheet_rl.agents.platform_ppo_cfg import PlatformGraspPPORunnerCfg
        self.assertEqual(cfg.rewards.grasp_progress.params["gamma"], PlatformGraspPPORunnerCfg().algorithm.gamma)

    def test_platform_episodes_are_1000_steps_and_base_stays_500(self):
        import math
        for cfg in (SheetRlPlatformEnvCfg(), SheetRlPlatformGraspEnvCfg()):
            self.assertEqual(math.ceil(cfg.episode_length_s / (cfg.decimation * cfg.sim.dt)), 1000)
            self.assertEqual(cfg.commands.deformable_pose.resampling_time_range[0], cfg.episode_length_s)
        base = SheetRlEnvCfg()
        self.assertEqual(math.ceil(base.episode_length_s / (base.decimation * base.sim.dt)), 500)

    def test_wandb_flag_expands_to_logger_arguments(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train.py"
        spec = importlib.util.spec_from_file_location("sheet_train", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module._expand_flags(["--task", "T", "--wandb", "agent.run_name=x"])
        self.assertNotIn("--wandb", args)
        self.assertEqual(args[-2:], ["--logger", "wandb"])
        self.assertIn("agent.run_name=x", args)
        args = module._expand_flags(["--wandb=sheet_platform", "--slow"])
        self.assertEqual(args[:4], ["--logger", "wandb", "--log_project_name", "sheet_platform"])
        self.assertEqual(args[-1], module.SLOW_OVERRIDE)
        with self.assertRaises(ValueError):
            module._expand_flags(["--wandb", "--logger", "tensorboard"])
        self.assertEqual(module._expand_flags(["--task", "T"]), ["--task", "T"])

    def test_play_record_flag_takes_its_path_off_the_arguments(self):
        path = Path(__file__).resolve().parents[1] / "scripts/play.py"
        spec = importlib.util.spec_from_file_location("sheet_play", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module._take_record_path(["--task", "T", "--record", "a.csv", "x=1"]),
                         ("a.csv", ["--task", "T", "x=1"]))
        self.assertEqual(module._take_record_path(["--record=b.csv", "--task", "T"]), ("b.csv", ["--task", "T"]))
        self.assertEqual(module._take_record_path(["--task", "T"]), (None, ["--task", "T"]))
        for argv in (["--record"], ["--record", "--task", "T"]):
            with self.assertRaises(ValueError):
                module._take_record_path(argv)

    def test_run_folder_is_the_run_name(self):
        import os, tempfile
        from types import SimpleNamespace
        path = Path(__file__).resolve().parents[1] / "scripts/train.py"
        spec = importlib.util.spec_from_file_location("sheet_train_names", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args, name = module._run_folder_name(["--rl_library", "rsl_rl", "--task", "T", "agent.run_name=my_run"])
        self.assertEqual(name, "my_run")
        self.assertNotIn("agent.run_name=my_run", args)
        self.assertIsNone(module._run_folder_name(["--rl_library", "rsl_rl", "--task", "T"])[1])
        args, name = module._run_folder_name(["--rl_library", "rsl_rl", "agent.run_name=x", "--timestamp"])
        self.assertIsNone(name)
        self.assertEqual(args, ["--rl_library", "rsl_rl", "agent.run_name=x"])
        self.assertIsNone(module._run_folder_name(["--rl_library", "skrl", "agent.run_name=x"])[1])
        written = []
        backend = SimpleNamespace(datetime=None, write_run_manifest=lambda log_dir, **kw: written.append(log_dir))
        module._use_plain_run_folder(backend, "my_run")
        self.assertEqual(backend.datetime.now().strftime("%Y"), "my_run")
        with tempfile.TemporaryDirectory() as root:
            backend.write_run_manifest(os.path.join(root, "my_run"), library="rsl_rl")
            self.assertEqual(len(written), 1)
            os.makedirs(os.path.join(root, "taken"))
            open(os.path.join(root, "taken", "model_0.pt"), "w").close()
            with self.assertRaises(FileExistsError):
                backend.write_run_manifest(os.path.join(root, "taken"), library="rsl_rl")

    def test_reset_policy_std_sets_arm_and_gripper_spread(self):
        path = Path(__file__).resolve().parents[1] / "scripts/reset_policy_std.py"
        spec = importlib.util.spec_from_file_location("reset_policy_std", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = {"actor_state_dict": {module.STD_KEY: torch.tensor([2.4, 6.4, 1.2, 1.56]), "w": torch.ones(2)}}
        old, new = module.reset_std(state, 0.5, 3.0)
        self.assertEqual(old[1], torch.tensor(6.4).item())
        torch.testing.assert_close(state["actor_state_dict"][module.STD_KEY], torch.tensor([0.5, 0.5, 0.5, 3.0]))
        torch.testing.assert_close(state["actor_state_dict"]["w"], torch.ones(2))  # weights untouched
        module.reset_std(state, None, 2.0)  # keep the arm, change only the gripper
        torch.testing.assert_close(state["actor_state_dict"][module.STD_KEY], torch.tensor([0.5, 0.5, 0.5, 2.0]))

    def test_reset_policy_std_sets_the_fastsac_actors_learned_log_std(self):
        path = Path(__file__).resolve().parents[1] / "scripts/reset_policy_std.py"
        spec = importlib.util.spec_from_file_location("reset_policy_std", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = {"actor_state_dict": {module.LOG_STD_KEY: torch.tensor([0.08, 0.1, 0.36]).log()}}
        old, new = module.reset_std(state, 0.45, None)
        self.assertAlmostEqual(old[2], 0.36, places=5)  # reported as spreads, not logs
        torch.testing.assert_close(torch.tensor(new), torch.full((3,), 0.45))
        torch.testing.assert_close(state["actor_state_dict"][module.LOG_STD_KEY], torch.full((3,), 0.45).log())

    def test_training_flag(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_platform.py"
        spec = importlib.util.spec_from_file_location("train_platform", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.training_args(["--grasp", "--num_envs", "8", "agent.max_iterations=2"])
        self.assertEqual(args[1], module.GRASP_TASK)
        self.assertNotIn("--grasp", args)
        self.assertIn("agent.max_iterations=2", args)
        self.assertIn("rsl_rl", args)
        self.assertEqual(module.training_args([])[1], module.PLACEMENT_TASK)


if __name__ == "__main__":
    unittest.main()


class TestSheetPoseJitter(unittest.TestCase):
    """reset_platform_and_sheet with per-env sheet jitter, on a fake scene that records what is written."""

    def _fake(self, n=64, nodes=9):
        from unittest.mock import MagicMock
        device = "cpu"
        grid = torch.stack(torch.meshgrid(torch.linspace(-0.1, 0.1, 3), torch.linspace(-0.1, 0.1, 3), indexing="ij"), -1)
        nodal = torch.zeros(n, nodes, 6)
        nodal[:, :, :2] = grid.reshape(1, nodes, 2)
        sheet = NS(device=device, data=NS(default_nodal_state_w=NS(torch=nodal)), write_nodal_state_to_sim_index=MagicMock())
        wall = lambda y: NS(data=NS(default_root_pose=NS(torch=torch.tensor([[0.0, y, 0.05, 0, 0, 0, 1.0]]).repeat(n, 1))),  # noqa: E731
                            write_root_pose_to_sim_index=MagicMock(), write_root_velocity_to_sim_index=MagicMock())
        target = NS(data=NS(default_root_pose=NS(torch=torch.tensor([[0.4, -0.3, 0.1, 0, 0, 0, 1.0]]).repeat(n, 1))),
                    write_root_pose_to_sim_index=MagicMock(), write_root_velocity_to_sim_index=MagicMock())
        scene = {"deformable": sheet, "mannequin_arm": target, "slot_neg_y": wall(-0.12), "slot_pos_y": wall(0.12)}
        env = NS(scene=NS(__getitem__=lambda self, k: scene[k], env_origins=torch.zeros(n, 3)))
        env.scene = type("Scene", (), {"__getitem__": lambda self, k: scene[k], "env_origins": torch.zeros(n, 3)})()
        return env, sheet, scene

    def _run(self, **kw):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.platform import reset_platform_and_sheet
        env, sheet, scene = self._fake()
        reset_platform_and_sheet(env, torch.arange(64), {"x": (0.0, 0.0), "y": (0.0, 0.0)}, (0.0, 0.0),
                                 platform_size=(0.2, 0.2), slot_radius=0.15, **kw)
        nodal = sheet.write_nodal_state_to_sim_index.call_args.args[0]
        centroid = nodal[..., :3].mean(1)
        wall_pose = scene["slot_pos_y"].write_root_pose_to_sim_index.call_args.kwargs["root_pose"]
        q = wall_pose[:, 3:]
        yaw = torch.atan2(2 * q[:, 2] * q[:, 3], 1 - 2 * q[:, 2] ** 2)  # xyzw, rotation about z only
        return centroid, yaw, wall_pose

    def test_no_jitter_is_the_fixed_pose(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.platform import FIXED_SHEET_POSITION, FIXED_SHEET_YAW
        centroid, yaw, _ = self._run()
        torch.testing.assert_close(centroid, torch.tensor(FIXED_SHEET_POSITION).expand(64, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(yaw, torch.full((64,), FIXED_SHEET_YAW), atol=1e-5, rtol=0)

    def test_jitter_is_per_env_within_range_and_walls_follow(self):
        from sheet_rl.tasks.manager_based.sheet_rl.mdp.platform import FIXED_SHEET_POSITION, FIXED_SHEET_YAW
        torch.manual_seed(0)
        centroid, yaw, wall_pose = self._run(sheet_xy_jitter=0.05, sheet_yaw_jitter=math.pi / 2)
        dxy = centroid[:, :2] - torch.tensor(FIXED_SHEET_POSITION[:2])
        self.assertTrue((dxy.abs() <= 0.05 + 1e-6).all())
        self.assertGreater(dxy.std(0).min().item(), 0.01)  # genuinely different per env
        dyaw = yaw - FIXED_SHEET_YAW
        self.assertTrue((dyaw.abs() <= math.pi / 2 + 1e-5).all())
        self.assertGreater(dyaw.std().item(), 0.5)
        torch.testing.assert_close(centroid[:, 2], torch.full((64,), FIXED_SHEET_POSITION[2]))
        # the +y wall sits 0.12 m from the sheet centre along the sheet's rotated +y axis
        offset = wall_pose[:, :2] - centroid[:, :2]
        torch.testing.assert_close(offset.norm(dim=-1), torch.full((64,), 0.12), atol=1e-5, rtol=0)
        expected = torch.stack([-0.12 * torch.sin(yaw), 0.12 * torch.cos(yaw)], -1)
        torch.testing.assert_close(offset, expected, atol=1e-5, rtol=0)
