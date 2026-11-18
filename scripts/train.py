# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified training executable for 'sheet_rl' reinforcement learning tasks."""

import os
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


PUSH_FLAG = "--push"
"""Disturb half the extractions with the same force the ``X`` key applies."""

PUSH_OVERRIDE = "env.push.on_extraction=true"
"""Hydra override the flag expands to. The push is a key-press probe without it.

Only the *trigger* is turned on here. How hard, how long and which way are the push's own settings,
so they stay tunable alongside the flag -- ``--push env.push.force=8.0`` is a harder yank, and
``--push env.push.on_extraction_probability=0.25`` disturbs a quarter of the extractions instead of
half.

Half rather than all of them on purpose: the policy has to see both a disturbed extraction and an
undisturbed one to learn that holding on is what pays, rather than that extraction is simply
dangerous. Watch ``Events/pushed_frac`` against ``Events/extraction`` in the training table -- the
first is the share of all episodes that were pushed, and if the policy starts refusing to extract
in order to dodge the push, both fall together.
"""

FLAG_OVERRIDES = {SLOW_FLAG: SLOW_OVERRIDE, PUSH_FLAG: PUSH_OVERRIDE}
"""Every bare flag this script understands, and the Hydra override it stands for."""

WANDB_FLAG = "--wandb"
"""Log to Weights & Biases instead of TensorBoard: ``--wandb`` or ``--wandb=<project>``.

Expands to Isaac Lab's ``--logger wandb`` (plus ``--log_project_name <project>`` when a project is
given; otherwise the project is the experiment name). Checkpoints still go to ``logs/rsl_rl``.
Needs a ``wandb login`` once; ``WANDB_USERNAME`` optionally selects the entity.
"""


def _expand_wandb(argv: list[str]) -> list[str]:
    """Replace ``--wandb`` / ``--wandb=<project>`` with the logger arguments it stands for."""
    flags = [arg for arg in argv if arg == WANDB_FLAG or arg.startswith(WANDB_FLAG + "=")]
    if not flags:
        return argv
    if any(arg == "--logger" or arg.startswith("--logger=") for arg in argv):
        raise ValueError("Use either --wandb or --logger, not both.")
    project = flags[-1].partition("=")[2]
    if flags[-1].startswith(WANDB_FLAG + "=") and not project:
        raise ValueError("--wandb= needs a project name, e.g. --wandb=sheet_platform")
    expanded = ["--logger", "wandb"] + (["--log_project_name", project] if project else [])
    return [arg for arg in argv if arg not in flags] + expanded


def _expand_flags(argv: list[str]) -> list[str]:
    """Turn the bare flags above into the Hydra overrides they stand for.

    Handled here rather than as real arguments because the backend parsers live in Isaac Lab and
    reject unknown flags before Hydra ever sees them. Stripping them first means they can be written
    anywhere in the command, including after the Hydra overrides where they read most naturally.
    """
    argv = _expand_wandb(argv)
    overrides = [override for flag, override in FLAG_OVERRIDES.items() if flag in argv]
    if not overrides:
        return argv
    return [arg for arg in argv if arg not in FLAG_OVERRIDES] + overrides


TIMESTAMP_FLAG = "--timestamp"
"""Keep Isaac Lab's ``<date>_<time>_<run_name>`` run folder instead of just ``<run_name>``."""


def _run_folder_name(argv: list[str]) -> tuple[list[str], str | None]:
    """Split off ``agent.run_name=<name>`` so the run folder can be exactly ``<name>``.

    Returns the remaining arguments and the name, or ``None`` when the run should keep Isaac Lab's
    timestamped folder: no run name given, ``--timestamp`` passed, or a library other than RSL-RL.
    """
    if TIMESTAMP_FLAG in argv:
        return [arg for arg in argv if arg != TIMESTAMP_FLAG], None
    library = next((argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--rl_library"), None)
    library = library or next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--rl_library=")), None)
    names = [arg.split("=", 1)[1] for arg in argv if arg.startswith("agent.run_name=")]
    if library != "rsl_rl" or not names or not names[-1]:
        return argv, None
    if os.sep in names[-1]:
        raise ValueError(f"Run name must not contain '{os.sep}': {names[-1]}")
    return [arg for arg in argv if not arg.startswith("agent.run_name=")], names[-1]


def _use_plain_run_folder(backend, name: str) -> None:
    """Make the RSL-RL training backend name the run folder ``<name>`` (no timestamp).

    Isaac Lab builds the folder as ``datetime.now().strftime(...)`` plus ``_<run_name>``, with no
    option to drop the time. With ``agent.run_name`` removed from the arguments and the backend's
    clock answering with the name, the folder is exactly ``<name>``. An existing, non-empty folder
    of that name is refused rather than mixed into.
    """

    class _Now:
        @staticmethod
        def strftime(_format):
            return name

    class _Clock:
        @staticmethod
        def now():
            return _Now()

    write_manifest = backend.write_run_manifest

    def refuse_existing(log_dir, *args, **kwargs):
        if os.path.isdir(log_dir) and os.listdir(log_dir):
            raise FileExistsError(
                f"Run folder already exists: {log_dir}. Choose another agent.run_name, or pass --timestamp.")
        return write_manifest(log_dir, *args, **kwargs)

    backend.datetime = _Clock
    backend.write_run_manifest = refuse_existing


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training library."""
    argv, name = _run_folder_name(_expand_flags(list(sys.argv[1:] if argv is None else argv)))
    library = next((argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--rl_library"), None)
    library = library or next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--rl_library=")), None)
    if library == "rsl_rl":
        from training_table import install_training_table

        install_training_table()
    if name is not None:
        from isaaclab_rl.entrypoints.backends import train_rsl_rl

        _use_plain_run_folder(train_rsl_rl, name)
    return run_train_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
