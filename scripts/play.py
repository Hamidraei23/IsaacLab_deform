# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified playback executable for 'sheet_rl' reinforcement learning tasks."""

import sys
import time

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


PUSH_FLAG = "--push"
"""Disturb half the extractions automatically, exactly as ``--push`` does in training."""

PUSH_ALL_FLAG = "--push_all"
"""Disturb every extraction, so a short watch is certain to show one."""

PUSH_ARGS = ("env.push.on_extraction=true",)
PUSH_ALL_ARGS = ("env.push.on_extraction=true", "env.push.on_extraction_probability=1.0")
"""Hydra overrides the two flags expand to.

Without either, the push is still bound to the ``X`` key and fires only when it is pressed. These
arm the same force on the extraction itself, which is what the policy met during training -- so
watching with ``--push`` is watching the task the policy was actually trained on, and watching
without it is watching the undisturbed half.

For a *scored* comparison over many episodes, use ``scripts/eval_push.py``, which pairs each
episode's outcome with whether that episode was pushed.
"""

PLATFORM_TELEOP_FLAG = "--platform_teleop"
"""Drive the platform from the viewer while the policy runs: hold W/S (x), A/D (y), Q/E (yaw).

Platform tasks only; see ``sheet_rl/platform_teleop.py``. In Newton GL those six keys stop flying
the camera while this is on -- use the arrow keys and the mouse for the camera instead.
"""

PLATFORM_TELEOP_ARGS = ("env.platform_teleop.enabled=true",)

FLAG_ARGS = {
    LITE_FLAG: LITE_ARGS,
    PUSH_FLAG: PUSH_ARGS,
    PUSH_ALL_FLAG: PUSH_ALL_ARGS,
    PLATFORM_TELEOP_FLAG: PLATFORM_TELEOP_ARGS,
}
"""Every bare flag this script understands, and the arguments it stands for."""


def _expand_flags(argv: list[str]) -> list[str]:
    """Turn the bare flags above into the arguments they stand for.

    Appended rather than prepended so they win: ``argparse`` keeps the last occurrence of an option,
    which means ``--lite`` still gives one environment when the command it is added to already
    asked for sixteen.
    """
    extra = [arg for flag, args in FLAG_ARGS.items() if flag in argv for arg in args]
    if not extra:
        return argv
    return [arg for arg in argv if arg not in FLAG_ARGS] + extra


PLOT_FLAG = "--plot"
"""Plot each finished episode (motion, and phase-2 rewards) before the next starts; see play_plots.py."""

RECORD_FLAG = "--record"
"""Write environment 0's per-step trace (the action sent, and the robot's response) to the CSV path
that follows, ``--record <path.csv>`` or ``--record=<path.csv>``; see play_record.py."""


def _take_record_path(argv: list[str]) -> tuple[str | None, list[str]]:
    """Split ``--record <path>`` off the arguments: (path or None, the arguments without it)."""
    for i, arg in enumerate(argv):
        if arg.startswith(RECORD_FLAG + "="):
            return arg.split("=", 1)[1], argv[:i] + argv[i + 1:]
        if arg == RECORD_FLAG:
            if i + 1 == len(argv) or argv[i + 1].startswith("-"):
                raise ValueError(f"{RECORD_FLAG} needs the CSV path to write, e.g. {RECORD_FLAG} logs/play/run.csv")
            return argv[i + 1], argv[:i] + argv[i + 2:]
    return None, argv



def _install_rsl_rl_episode_timing() -> None:
    # Isaac Lab owns the RSL-RL stepping loop; wrap its vector env step to report episode durations.
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step
    if getattr(original_step, "_sheet_rl_episode_timing", False):
        return

    def step_with_episode_timing(self, actions):
        if not hasattr(self, "_sheet_rl_episode_lengths"):
            self._sheet_rl_episode_lengths = [0] * self.num_envs
            now = time.monotonic()
            self._sheet_rl_episode_started = [now] * self.num_envs
            self._sheet_rl_episode_count = 0

        observations, rewards, dones, extras = original_step(self, actions)
        for env_id in range(self.num_envs):
            self._sheet_rl_episode_lengths[env_id] += 1

        for env_id in dones.nonzero().flatten().tolist():
            steps = self._sheet_rl_episode_lengths[env_id]
            elapsed = time.monotonic() - self._sheet_rl_episode_started[env_id]
            self._sheet_rl_episode_count += 1
            simulated = steps * float(self.unwrapped.step_dt)
            print(
                f"[play] episode {self._sheet_rl_episode_count} (env {env_id}): "
                f"{steps} steps ({simulated:.2f} s simulated, {elapsed:.1f} s wall-clock)",
                flush=True,
            )
            self._sheet_rl_episode_lengths[env_id] = 0
            self._sheet_rl_episode_started[env_id] = time.monotonic()

        return observations, rewards, dones, extras

    step_with_episode_timing._sheet_rl_episode_timing = True
    RslRlVecEnvWrapper.step = step_with_episode_timing


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning play library."""
    argv = _expand_flags(list(sys.argv[1:] if argv is None else argv))
    record_path, argv = _take_record_path(argv)
    if record_path is not None:
        from play_record import install

        install(record_path, argv)
    if PLOT_FLAG in argv:
        from play_plots import main as play_with_plots

        return play_with_plots([arg for arg in argv if arg != PLOT_FLAG])
    if "--rl_library" in argv:
        library_index = argv.index("--rl_library")
        if library_index + 1 < len(argv) and argv[library_index + 1] == "rsl_rl":
            _install_rsl_rl_episode_timing()
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
