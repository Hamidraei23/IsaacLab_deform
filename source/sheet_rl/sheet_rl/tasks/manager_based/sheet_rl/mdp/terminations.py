# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations specific to lifting the sheet out of its slot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .rewards import _drape_success, _early_release, _high_release, _table_drop

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def released_before_extraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode if the gripper opens while holding a sheet still in the slot.

    The attempt is over at that point: the policy gets one scored closure per episode, so a sheet
    dropped back into the slot cannot be picked up again for credit. Ending immediately stops the
    remaining steps being spent on a run that can no longer achieve anything.

    The flag is set by :class:`~.rewards.grasp_stage_reward`, which also charges the penalty. The
    reward manager runs before the termination manager inside a step, so it is never stale.
    """
    return _early_release(env)


def released_too_high(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the gripper was opened above the release ceiling in phase two.

    Warning:
        Not registered in :class:`~..sheet_rl_env_cfg.SheetTerminationsCfg`. A high release is
        charged by :class:`~.rewards.release_stage_reward` and the episode carries on. Kept because
        the flag it reads is still maintained and wiring it back is a one-line change, but nothing
        calls this today.

    The argument for ending here was that a sheet dropped from height lands wherever it falls, so
    the placement reward that follows is decided by luck rather than by anything the policy did.
    The argument against, which is the one in force, is that this cannot be known at the moment of
    release: a sheet let go high over the band may still land on the arm, settle and be scored a
    genuine success, and terminating on the release forecloses that before the physics has run. The
    charge prices the risk taken; the outcome decides the rest.

    The flag is set by :class:`~.rewards.release_stage_reward` on the step the gripper opens. Were
    this wired, it would fire one step later -- terminations are computed *before* rewards inside a
    step, so a flag written during reward computation is read on the following one.
    """
    return _high_release(env)


def sheet_dropped_on_table(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the sheet is lying on the table, out of the gripper and off the arm.

    Warning:
        Not registered in :class:`~..sheet_rl_env_cfg.SheetTerminationsCfg`. The drop is charged
        once by :class:`~.rewards.table_drop_penalty` and the episode carries on to whatever ending
        does arrive, which is normally the time-out and its ``drape_failure`` charge. Kept because
        the latch it reads is still maintained and wiring it back is a one-line change, but nothing
        calls this today.

    The condition, the settling window and the penalty all belong to
    :class:`~.rewards.table_drop_penalty`; this reads the latch it sets, one step later.
    """
    return _table_drop(env)


def drape_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End the episode once the sheet has sat on the red band, covered, for a full second.

    The task's only success condition, and the one termination here that is not a failure. The
    cloth is off the gripper and on the arm and has stayed there; there is nothing further the
    policy can do that the task asks for, so the remaining steps are better spent on a fresh
    episode.

    The second of settling is the substance of the test, not a formality. Coverage crossing the
    success threshold is direction-blind -- a sheet sliding off the arm passes the bar going down
    just as one settling onto it passes going up -- so ending the episode on the crossing would
    score the two alike and hand out the bonus before the physics had said which had happened.
    Nothing requires the robot to hold still during the wait, but a retraction that drags the cloth
    shows up as coverage falling, which restarts the count.

    Not flagged as a time-out, so the value function treats it as a true terminal state and does
    not bootstrap past it. That is the correct reading: the episode is over because the goal was
    met, not because the clock ran out.

    The latch is set by :class:`~.rewards.drape_milestones`, which also pays the bonus. Because
    terminations are computed before rewards inside a step, this fires the step after the drape is
    judged complete -- the latch is held rather than momentary precisely so that lag is harmless.
    """
    return _drape_success(env)


def finger_in_slot(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float, float],
    margin: float = 0.005,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Sequence rather than tuple: the config passes a list, because Hydra's slice-to-string pass
    # does not recurse into tuples and would leave a SceneEntityCfg's default slice in place.
    slot_cfgs: Sequence[SceneEntityCfg] = (
        SceneEntityCfg("slot_neg_y"),
        SceneEntityCfg("slot_pos_y"),
    ),
    finger_body_names: tuple[str, ...] = ("panda_leftfinger", "panda_rightfinger"),
) -> torch.Tensor:
    """End the episode if either fingertip enters the box enclosing the slot.

    The sheet has to be taken by the strip projecting above the walls, so a finger down inside the
    slot means the gripper is digging at the walls rather than picking the sheet off the top.

    Tested geometrically against the slot's own frame rather than through a contact sensor. The
    walls are kinematic and their pose is known exactly, so the box test is both cheaper and
    steadier than reading contacts through the rigid-soft coupler -- and it fires slightly *before*
    a real touch, which is the useful side to err on.

    Args:
        env: The environment.
        half_extents: Half the size of the box enclosing both walls, in the slot frame [m].
        margin: Extra clearance added to every side of that box [m].
        robot_cfg: The robot carrying the fingers.
        slot_cfgs: The two walls; their midpoint is taken as the slot's centre.
        finger_body_names: Bodies treated as fingertips.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    walls: list[RigidObject] = [env.scene[cfg.name] for cfg in slot_cfgs]

    centre = torch.stack([wall.data.root_pos_w.torch for wall in walls]).mean(dim=0)
    # both walls share an orientation, so either one gives the slot's frame
    quat = walls[0].data.root_quat_w.torch

    limits = torch.tensor(half_extents, device=centre.device) + margin
    inside = torch.zeros(env.num_envs, dtype=torch.bool, device=centre.device)
    for name in finger_body_names:
        body_id = robot.body_names.index(name)
        offset = robot.data.body_link_pose_w.torch[:, body_id, :3] - centre
        local = quat_apply_inverse(quat, offset)
        inside |= (local.abs() < limits).all(dim=-1)
    return inside
