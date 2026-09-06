# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified training executable for 'sheet_rl' reinforcement learning tasks."""

import sys

import sheet_rl.tasks  # noqa: F401

from isaaclab_rl.entrypoints import run_train_cli

SLOW_FLAG = "--slow"
"""Charge the arm for how fast its hand travels, via the ``ee_speed`` reward term."""

SLOW_OVERRIDE = "env.rewards.ee_speed.weight=-4.0"
"""Hydra override the flag expands to. The term sits at weight zero without it.

A penalty rather than a speed cap, because the two ask for different things: a cap forbids the
policy from ever moving quickly, while a charge lets it decide where speed is worth paying for --
crossing the table quickly and slowing down over the sheet, which is the behaviour actually wanted.
"""


PHASE_ONE_FLAG = "--phase_one"
"""Train the pick alone: pay for extraction, end the episode there, and drop the drape."""

PHASE_TWO_REWARD_TERMS = (
    "drape_closeness",
    "drape_coverage",
    "goal_proximity",
    "drape_stage",
    "hold_over_band",
    "blocked_by_gripper",
    "finger_touches_arm",
    "release_stage",
    "release_retreat",
    "dropped_on_table",
    "timeout_gripping",
    "drape_failure",
)
"""Every reward the placement phase owns. All of them are zeroed by :data:`PHASE_ONE_FLAG`.

Zeroing is not merely tidy. ``drape_failure`` charges for ending the episode away from the band, and
with the flag on *every* episode ends away from the band -- the moment the sheet is out of the slot.
Left armed it would fine the policy for succeeding at the only thing it was being asked to do.
"""

PHASE_ONE_OVERRIDES = [
    # end the episode the moment the sheet is clear of the slot
    "env.terminations.extracted.params.enabled=true",
    # and pay for arriving there, since the drape that used to be the outcome is gone
    "env.rewards.extraction_complete.weight=1.0",
] + [f"env.rewards.{name}.weight=0.0" for name in PHASE_TWO_REWARD_TERMS]
"""Hydra overrides the flag expands to.

Three changes that only make sense together: without the termination the bonus is collected and the
episode wanders on through a phase nothing is paying for; without the bonus the pick has no outcome
and only its shaping terms to go on; without the zeroing the phase-two terms keep grading a drape
that can no longer happen.

Phase one's own terms -- ``approach``, ``alignment``, ``grasp_stage``, ``lift_progress`` -- are left
exactly as they are, as are ``table_clearance``, ``ee_speed`` and the two smoothness charges, which
mean the same thing in either mode.
"""


def _expand_phase_one_flag(argv: list[str]) -> list[str]:
    """Turn ``--phase_one`` into the overrides that make extraction the whole task.

    Handled the same way as :func:`_expand_slow_flag`, and for the same reason: the backend parsers
    live in Isaac Lab and reject unknown flags before Hydra ever sees them.
    """
    if PHASE_ONE_FLAG not in argv:
        return argv
    return [arg for arg in argv if arg != PHASE_ONE_FLAG] + PHASE_ONE_OVERRIDES


def _expand_slow_flag(argv: list[str]) -> list[str]:
    """Turn ``--slow`` into the Hydra override that arms the hand-speed penalty.

    Handled here rather than as a real argument because the backend parsers live in Isaac Lab and
    reject unknown flags before Hydra ever sees them. Stripping it first means it can be written
    anywhere in the command, including after the Hydra overrides where it reads most naturally.
    """
    if SLOW_FLAG not in argv:
        return argv
    return [arg for arg in argv if arg != SLOW_FLAG] + [SLOW_OVERRIDE]


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training library."""
    return run_train_cli(_expand_phase_one_flag(_expand_slow_flag(list(sys.argv[1:] if argv is None else argv))))


if __name__ == "__main__":
    raise SystemExit(main())
