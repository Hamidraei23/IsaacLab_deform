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
    return run_train_cli(_expand_slow_flag(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
