# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded ``teleop_ur10e.py`` session and measure how far the rerun drifts from it.

Rebuilds the environment from the recording's sidecar (same task, seed, flags and action scale),
feeds the recorded seven-channel commands back in step for step, resetting where the original
did, and writes the same columns to ``<trace>.replay.csv``. A per-episode drift summary is printed
at the end: the largest joint-angle and tool-position gap between the two runs, and the steps on
which the grasp state (``is_holding``) differed. Small drift is the GPU solver's own
non-determinism; a divergence that starts at one step and grows is the thing to look at.

Headless by default, so it can run in the background while the original is being flown again.

Usage::

    python scripts/replay_teleop_ur10e.py
    python scripts/replay_teleop_ur10e.py --trace logs/teleop/ur10e_last.csv --viz newton_gl
"""

import warp as wp

wp.config.enable_backward = False

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded UR10e teleop trace.")
parser.add_argument("--trace", default="logs/teleop/ur10e_last.csv", help="Recording to replay.")
parser.add_argument("--out", default=None, help="Where to write the rerun's trace (default: <trace>.replay.csv).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below here runs after the simulator is up."""

import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import resolve_presets  # noqa: E402

from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env import PushableSheetEnv  # noqa: E402
from sheet_rl.tasks.manager_based.sheet_twist.sheet_ur10e_env_cfg import (  # noqa: E402
    SheetUR10eTwistStudentEnvCfg,
)
from sheet_rl.tasks.manager_based.sheet_twist.teleop_trace import (  # noqa: E402
    ACTION_COLUMNS, TWIST_COLUMNS, TraceWriter, capture_row, load_trace,
)

TELEOP_EPISODE_SECONDS = 3600.0
FREE_RUN_TERMINATIONS = ("ee_below_table", "deformable_out_of_bounds", "joint_vel_out_of_limit",
                         "released_high", "held_too_low", "draped")


def build_env(meta):
    """The environment ``teleop_ur10e.py`` built, from what it wrote down about itself."""
    env_cfg = resolve_presets(SheetUR10eTwistStudentEnvCfg())
    env_cfg.scene.num_envs = int(meta["num_envs"])
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = int(meta["seed"])
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = TELEOP_EPISODE_SECONDS
    env_cfg.commands.deformable_pose.resampling_time_range = (TELEOP_EPISODE_SECONDS, TELEOP_EPISODE_SECONDS)
    if meta["free_run"]:
        for name in FREE_RUN_TERMINATIONS:
            setattr(env_cfg.terminations, name, None)
    # A hand-written test trace may switch off more. Not ``placed`` or ``dropped``: reward terms
    # read both by name (sheet_twist_mdp), so the one-cycle rule -- the first opening after a
    # closure ends the episode -- holds in every replay, and a test can close once per episode.
    for name in meta.get("disable_terminations", ()):
        if name in ("placed", "dropped"):
            raise RuntimeError(f"{name!r} cannot be disabled; a reward term reads it by name")
        setattr(env_cfg.terminations, name, None)
    scale = np.broadcast_to(np.asarray(env_cfg.actions.arm_action.scale, dtype=np.float64), (6,))
    recorded = np.asarray(meta["action_scale"], dtype=np.float64)
    if not np.allclose(scale, recorded):
        raise RuntimeError(f"Action scale is {scale.tolist()} now but was {recorded.tolist()} when recorded;"
                           " the replayed commands would not mean the same thing.")
    return PushableSheetEnv(cfg=env_cfg), scale


def summarize(original, rerun):
    """Per-episode drift between two traces of the same length."""
    # A synthetic trace (e.g. a gripper cycle written by hand) carries commands but no state
    # columns; drift is then simply not measurable, and the rerun's own trace is the result.
    joints = [key for key in original[0] if key.startswith("q_") and key in rerun[0]]
    if not joints or "tool_x" not in original[0]:
        print("\n(original trace has no state columns; nothing to diff against)")
        return
    print("\nreplay drift, per episode")
    print(f"  {'ep':>3} {'steps':>6} {'max|dq| rad':>12} {'max|dtool| m':>13} {'hold mismatches':>16}  ended_by (orig / replay)")
    episodes = sorted({int(row["episode"]) for row in original})
    for episode in episodes:
        pairs = [(a, b) for a, b in zip(original, rerun) if int(a["episode"]) == episode]
        if not pairs:
            continue
        dq = max(abs(a[k] - b[k]) for a, b in pairs for k in joints)
        dtool = max(math.dist((a["tool_x"], a["tool_y"], a["tool_z"]), (b["tool_x"], b["tool_y"], b["tool_z"]))
                    for a, b in pairs)
        mismatches = [int(a["step"]) for a, b in pairs if a["is_holding"] != b["is_holding"]]
        first = f"{len(mismatches)} (first at step {mismatches[0]})" if mismatches else "0"
        ended = f"{pairs[-1][0]['ended_by'] or '-'} / {pairs[-1][1]['ended_by'] or '-'}"
        print(f"  {episode:>3} {len(pairs):>6} {dq:>12.4f} {dtool:>13.4f} {first:>16}  {ended}")


def main():
    meta, original = load_trace(args_cli.trace)
    if not original:
        raise SystemExit(f"{args_cli.trace} holds no steps")
    out = args_cli.out or os.path.splitext(args_cli.trace)[0] + ".replay.csv"
    env, scale = build_env(meta)
    rerun = TraceWriter(out, {**meta, "replay_of": args_cli.trace})
    print(f"[replay] {len(original)} steps from {args_cli.trace} (seed {meta['seed']}, free_run={meta['free_run']})")

    env.reset()
    episode = 0
    try:
        with torch.inference_mode():
            for row in original:
                if row["reset_before"] and int(row["step"]) > 0:
                    env.reset()
                    episode += 1
                action = torch.tensor([row[c] for c in ACTION_COLUMNS], dtype=torch.float32, device=env.device)
                twist = [row[c] for c in TWIST_COLUMNS]
                _, reward, terminated, truncated, _ = env.step(action.expand(env.num_envs, -1))
                done = bool(terminated[0]) or bool(truncated[0])
                ended = [name for name in env.termination_manager.active_terms
                         if bool(env.termination_manager.get_term(name)[0])] if done else []
                rerun.append(capture_row(
                    env, step=int(row["step"]), episode=episode, action=action.tolist(), twist=twist,
                    reset_before=bool(row["reset_before"]), reward=float(reward[0]), ended=ended,
                ))
                if done and not row["ended_by"]:
                    print(f"[replay] step {int(row['step'])}: rerun ended ({'+'.join(ended)}) where the"
                          " original did not -- the two have diverged; stopping here.")
                    break
                if done:
                    # The original reset after this step as well; the next row carries the flag.
                    pass
                if int(row["step"]) % 500 == 0:
                    print(f"[replay] step {int(row['step'])}", flush=True)
    finally:
        rerun.flush()
        print(f"[replay] {len(rerun.rows)} steps written to {out}")
        summarize(original[:len(rerun.rows)], rerun.rows)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
