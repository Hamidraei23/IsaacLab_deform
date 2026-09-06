# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the recorded channels of a trajectory against time.

Reads a CSV written by ``scripts/record_episodes.py`` -- ``step``, ``time_s``, the ``act_*`` action
channels and the ``pos_*`` measured joint positions -- and draws each as its own panel. Time is the
x axis throughout.

The two groups are annotated differently because different things bound them: the actions get the
``clip_actions`` bound they were clipped to, and the positions get the real Franka's joint limits,
which the simulated arm is not held to.

Which trajectory is plotted can be given as a path, or chosen from a numbered list: pointed at a
directory, or given nothing at all, the script finds the recorded episodes, prints them with what
ended each one, and asks which to draw.

Nothing here is specific to the eight dimensions this task happens to have. The columns are taken
from the CSV header, so a trace with a different action layout plots correctly and labels itself.

Usage::

    python3 scripts/plot_trajectory.py                       # pick from everything under logs/
    python3 scripts/plot_trajectory.py <run>/trajectories     # pick from one run
    python3 scripts/plot_trajectory.py "<run>/trajectories/1J(0.000),...csv"
    python3 scripts/plot_trajectory.py <csv> --out actions.png   # write instead of show

Needs only ``matplotlib`` and the standard library -- deliberately no simulator, so it runs in a
plain Python environment while training has the GPU.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

SEARCH_GLOB = os.path.join("logs", "rsl_rl", "*", "*", "trajectories", "*.csv")
"""Where recorded episodes land by default, matching ``record_episodes.py``'s own default."""

FRANKA_JOINT_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}
"""Real Franka Emika Panda position limits [rad], drawn behind the measured position traces.

The simulated arm is not held to these, so a trace approaching one is worth seeing: it is the
difference between a motion that could be replayed on hardware and one that only exists in
simulation. Joints 4 and 6 are one-sided, which is the pair most likely to be violated.
"""


def _load(path: str) -> tuple[list[str], list[list[float]], dict]:
    """Read one trace, returning its column names, its rows, and any sidecar metadata.

    The sidecar is optional: a CSV moved or renamed on its own still plots, it simply loses the
    ending and the clip bound from the title.
    """
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]

    meta = {}
    sidecar = os.path.splitext(path)[0] + ".json"
    if os.path.exists(sidecar):
        with open(sidecar) as handle:
            meta = json.load(handle)
    return columns, rows, meta


def _describe(path: str) -> str:
    """One line for the chooser: the filename, plus what the sidecar knows about the episode."""
    name = os.path.basename(path)
    sidecar = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(sidecar):
        return name
    try:
        with open(sidecar) as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return name
    ending = "+".join(meta.get("ended_by", [])) or "unknown"
    return f"{name}\n      {meta.get('steps', '?')} steps, {meta.get('duration_s', 0):.1f} s, ended on {ending}"


def _choose(candidates: list[str]) -> str | None:
    """Print a numbered list and ask which trace to draw.

    Returns ``None`` if the choice is abandoned, so an empty answer or a closed stdin exits quietly
    rather than raising over what is a perfectly reasonable way to change your mind.
    """
    print(f"\n{len(candidates)} recorded episode(s):\n")
    for index, path in enumerate(candidates, start=1):
        print(f"  [{index:>2}] {_describe(path)}")
    try:
        answer = input("\nwhich one? [1] ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(candidates):
        print(f"[ERROR]: '{answer}' is not one of 1..{len(candidates)}")
        return None
    return candidates[int(answer) - 1]


def _resolve(target: str | None) -> str | None:
    """Turn whatever was asked for into the single CSV to plot.

    A file is taken as given. A directory is searched one level deep, and nothing at all falls back
    to every trajectory folder under ``logs/``. Either way, more than one candidate goes to the
    chooser and exactly one is used without asking.
    """
    if target and os.path.isfile(target):
        return target
    if target:
        pattern = os.path.join(target, "*.csv")
    else:
        pattern = SEARCH_GLOB
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        print(f"[ERROR]: no trajectory CSVs found matching {pattern}")
        return None
    if len(candidates) == 1:
        print(f"[INFO]: only one trace found: {candidates[0]}")
        return candidates[0]
    return _choose(candidates)


def _groups(columns: list[str]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split the data columns into the action channels and the measured joint positions.

    Keyed on the ``act_``/``pos_`` prefixes ``record_episodes.py`` writes. Traces recorded before
    those prefixes existed carry bare joint names and are all actions, which is what the fallback
    says -- an older file still plots, it simply has no position panels.
    """
    actions = [(index, name[4:]) for index, name in enumerate(columns) if name.startswith("act_")]
    positions = [(index, name[4:]) for index, name in enumerate(columns) if name.startswith("pos_")]
    if not actions and not positions:
        actions = [(index, name) for index, name in enumerate(columns[2:], start=2)]
    return actions, positions


def _plot(path: str, columns: list[str], rows: list[list[float]], meta: dict, use_steps: bool, out: str | None):
    """Draw every recorded channel as its own panel on a shared time axis.

    One panel per channel rather than lines stacked on one axes. The channels do not share a
    meaning or a unit: the arm actions are relative commands in action units, the gripper channel is
    read for its *sign* alone, and the positions are absolute angles in radians. Overlaid, the
    gripper's swings set the y range and flatten everything else into a band.

    The two groups get different reference lines, because different things bound them -- the actions
    are bounded by ``clip_actions``, and the positions by what the robot can physically reach.
    """
    import matplotlib

    if out:
        # chosen before pyplot is imported, which is the only point at which it takes effect
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_index = 0 if use_steps else 1
    x = [row[x_index] for row in rows]
    x_label = "step" if use_steps else "time [s]"

    actions, positions = _groups(columns)
    panels = actions + positions
    clip = meta.get("clip_actions")

    figure, axes = plt.subplots(
        len(panels), 1, figsize=(12, 1.45 * len(panels) + 1.4), sharex=True, constrained_layout=True
    )
    axes = [axes] if len(panels) == 1 else list(axes)

    for panel, ((column, name), axis) in enumerate(zip(panels, axes)):
        is_action = panel < len(actions)
        values = [row[column] for row in rows]
        axis.plot(x, values, linewidth=1.1, color="#4f81bd" if is_action else "#c0504d")
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.set_ylabel(("act " if is_action else "pos ") + name, fontsize=8, rotation=0, ha="right", va="center")
        axis.tick_params(labelsize=7)
        axis.margins(x=0.01)

        if is_action and clip:
            # the bound rsl-rl applied. A trace riding it is a policy asking for more than the
            # action space allows, which is worth seeing at a glance -- and these policies do
            for bound in (-clip, clip):
                axis.axhline(bound, color="0.55", linestyle="--", linewidth=0.8)
            axis.set_ylim(-clip * 1.12, clip * 1.12)
        elif not is_action and name in FRANKA_JOINT_LIMITS:
            # the hardware's reach, not the simulation's. Drawn only when it would be visible, so a
            # joint working in the middle of its range is not squashed into a band by a limit
            # metres away from anything the trace does
            low, high = FRANKA_JOINT_LIMITS[name]
            span = max(values) - min(values) or 1.0
            for bound in (low, high):
                if min(values) - span < bound < max(values) + span:
                    axis.axhline(bound, color="#a04040", linestyle="--", linewidth=0.8)

        # The gripper is the one channel read for its sign rather than its magnitude: anything
        # above the threshold commands fully open, anything below it fully closed. Shading where it
        # is open turns a square wave into the grasp/release sequence the rewards key on.
        if is_action and panel == len(actions) - 1:
            axis.fill_between(
                x, *axis.get_ylim(), where=[value > 0.0 for value in values], color="#4f81bd", alpha=0.13, lw=0
            )
            axis.annotate(
                "shaded = commanded open",
                xy=(0.995, 0.06),
                xycoords="axes fraction",
                ha="right",
                fontsize=7,
                color="#3a5f8a",
            )

    if positions:
        axes[len(actions)].annotate(
            "measured joint position [rad]; dashed = Franka hardware limit",
            xy=(0.005, 1.06),
            xycoords="axes fraction",
            fontsize=7,
            color="#a04040",
        )

    axes[-1].set_xlabel(x_label)

    ending = "+".join(meta.get("ended_by", [])) or "unknown"
    subtitle = f"{len(rows)} steps"
    if not use_steps and rows:
        subtitle += f" / {rows[-1][1]:.1f} s"
    subtitle += f"  -  ended on {ending}"
    if "return" in meta:
        subtitle += f"  -  return {meta['return']:,.0f}"
    figure.suptitle(f"{os.path.basename(path)}\n{subtitle}", fontsize=9)

    if out:
        figure.savefig(out, dpi=130)
        print(f"[INFO]: wrote {out}")
    else:
        plt.show()
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Resolve which trace to draw, load it, and plot it."""
    parser = argparse.ArgumentParser(description="Plot the action dimensions of a recorded trajectory.")
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=None,
        help="A trace CSV, or a directory to choose one from. Defaults to searching under logs/.",
    )
    parser.add_argument("--out", type=str, default=None, help="Write a PNG here instead of opening a window.")
    parser.add_argument(
        "--steps", action="store_true", default=False, help="Use the step index on the x axis rather than seconds."
    )
    args = parser.parse_args(argv)

    path = _resolve(args.trajectory)
    if path is None:
        return 1

    columns, rows, meta = _load(path)
    if not rows:
        print(f"[ERROR]: {path} has a header but no rows")
        return 1
    if len(columns) < 3:
        print(f"[ERROR]: {path} has no action columns -- expected 'step', 'time_s', then one per dimension")
        return 1

    actions, positions = _groups(columns)
    print(
        f"[INFO]: {path}: {len(rows)} steps, {len(actions)} action dim(s)"
        f" and {len(positions)} joint position(s)"
    )
    _plot(path, columns, rows, meta, args.steps, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
