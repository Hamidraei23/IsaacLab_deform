# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified playback executable for 'sheet_rl' reinforcement learning tasks."""

import sys

import sheet_rl.tasks  # noqa: F401

from isaaclab_rl.entrypoints import run_play_cli

LITE_FLAG = "--lite"
"""Watch one environment at wall-clock speed, cheaply enough to sit alongside a training run."""

LITE_ARGS = ("--num_envs", "1", "--real-time")
"""What the flag expands to.

Sixteen environments is a sampling tool -- it shows how the policy handles sixteen different slot
yaws at once. One environment running at wall-clock speed is a watching tool, and it wants roughly
a sixteenth of the simulation work, which is what makes it usable while a training run has the GPU.

``--real-time`` sleeps out the remainder of each control period, so the motion plays at the speed
the robot would actually move rather than as fast as the solver can manage.
"""


def _expand_lite_flag(argv: list[str]) -> list[str]:
    """Turn ``--lite`` into the playback arguments it stands for.

    Appended rather than prepended so it wins: ``argparse`` keeps the last occurrence of an option,
    which means ``--lite`` still gives one environment when the command it is added to already
    asked for sixteen.
    """
    if LITE_FLAG not in argv:
        return argv
    return [arg for arg in argv if arg != LITE_FLAG] + list(LITE_ARGS)


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning play library."""
    return run_play_cli(_expand_lite_flag(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
