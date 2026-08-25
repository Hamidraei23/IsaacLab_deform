# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the sheet task's end-effector by hand, to see what the policy is being asked to do.

Runs the *same* scene as training -- same sheet, slot, mannequin arm, same randomised reset, and
the same terminations -- with one substitution: the seven joint-delta action channels are replaced
by an end-effector pose target, because a keyboard produces a direction and not seven joint deltas.
The gripper channel is left exactly as the policy will see it, binary open/close, so what you feel
through the keys is the gripper the policy has to learn.

The reward is printed live, broken down by term, so the shaping can be watched rather than
inferred: hold a key, see which number moves. When the episode ends -- on the clock or on any of
the task's terminations -- a table of what the whole episode earned is printed, which is the same
accounting the training logs report and so directly comparable with a policy's numbers.

Keys are read from the **terminal**, not from the 3D window. Kit's input stack is not usable for
this: its viewport binds W/A/S/D to the fly-camera, and its device layer needs the Kit window to
hold OS focus, which fights the terminal the reward trace is being read from. Reading the tty
directly avoids both, and keeps working under viewers that have no Kit window at all.

Key bindings::

    W / S     end-effector +x / -x        Z / X   roll  +/-
    A / D     end-effector +y / -y        T / G   pitch +/-
    Q / E     end-effector +z / -z        C / V   yaw   +/-
    K         toggle gripper open/close
    L         stop all motion
    R         reset the environment
    Ctrl-C    quit

Usage::

    ${ISAACLAB}/isaaclab.sh -p scripts/teleop_keyboard.py
    ${ISAACLAB}/isaaclab.sh -p scripts/teleop_keyboard.py --sensitivity 2.0

    # explore without the episode ever ending on its own
    ${ISAACLAB}/isaaclab.sh -p scripts/teleop_keyboard.py --free_run
"""

# Warp captures ``enable_backward`` when a module is created, which happens at import time, so it
# has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for the sheet task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Multiplier on how far one key press moves the end-effector."
)
parser.add_argument(
    "--keep_flying",
    action="store_true",
    help=(
        "Stay in the scene after an episode ends instead of quitting. Off by default: the recorded"
        " trace holds one episode, so continuing risks replacing the attempt just flown."
    ),
)
parser.add_argument(
    "--free_run",
    action="store_true",
    help=(
        "Also disable the terminations a human trips by accident -- dropping below the table,"
        " shoving the sheet out of bounds -- leaving only R and the task's own success and failure"
        " conditions. The time limit is off either way."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A keyboard needs a window to receive events from, so this script is never headless. Clearing the
# headless flag is not enough on its own: the launcher forces headless back on unless a Kit
# visualizer is explicitly requested, so ask for one here as if '--viz kit' had been passed. A
# visualizer chosen on the command line is left alone.
args_cli.headless = False
if getattr(args_cli, "visualizer", None) is None:
    args_cli.visualizer = ["kit"]
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below here runs after Kit is up."""

import csv  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import sys  # noqa: E402
import termios  # noqa: E402
import tty  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

from sheet_rl.tasks.manager_based.sheet_rl.mdp import grasp_debug  # noqa: E402
from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env_cfg import SheetRlEnvCfg  # noqa: E402

# metres and radians applied per control step for one held key, before --sensitivity. Chosen for
# roughly 0.2 m/s and 0.9 rad/s at the task's 30 Hz control rate, which is quick enough to cross
# the table in a couple of seconds and slow enough to place an edge between the fingers.
STEP_TRANSLATION = 0.007
STEP_ROTATION = 0.03

RECORDING_PATH = "logs/teleop/last_episode.csv"
"""Where the step-by-step trace of the most recent episode is written.

Holds one episode: the point is to capture a hand-flown attempt and pick it apart, not to
accumulate a pile of files. The script exits when an episode ends, so what is on disk is always the
attempt just flown.
"""

MIN_RECORDED_STEPS = 300
"""Episodes shorter than this are not written.

A deliberate attempt at the task takes hundreds of steps to fly by hand, so anything shorter is a
fumble or a stray reset -- and since the file holds only one episode, letting a fumble through
destroys the attempt worth keeping. A 31-step fragment overwrote a complete 500-step run once.
"""

TELEOP_EPISODE_SECONDS = 3600.0
"""Episode length while teleoperating [s]. An hour, i.e. effectively never."""

HOLD_STEPS = 8
"""How many control steps one key press stays active for.

A terminal reports key presses but never key *releases*, so "held" has to be inferred. Holding a
key makes the terminal auto-repeat it at roughly 30 Hz, which re-arms the key faster than this
expires and reads as continuous motion; letting go lets it lapse after about a quarter second.
"""

# axis index and sign each key drives, in the twist's own (x, y, z, rx, ry, rz) order
KEY_BINDINGS = {
    "w": (0, +1.0), "s": (0, -1.0),
    "a": (1, +1.0), "d": (1, -1.0),
    "q": (2, +1.0), "e": (2, -1.0),
    "z": (3, +1.0), "x": (3, -1.0),
    "t": (4, +1.0), "g": (4, -1.0),
    "c": (5, +1.0), "v": (5, -1.0),
}

BANNER = """
    ============================================================
      CLICK THIS TERMINAL WINDOW, THEN PRESS KEYS.
      Keys are read here, NOT in the 3D window. The 3D window is
      only for looking -- it will ignore everything you type.
      Every accepted key prints a '[key] ...' line below, and the
      'rx=' counter in the status line goes up. If neither moves,
      this terminal does not have focus.
    ============================================================
      W / S   end-effector +x / -x      Z / X   roll  +/-
      A / D   end-effector +y / -y      T / G   pitch +/-
      Q / E   end-effector +z / -z      C / V   yaw   +/-
      K       toggle gripper open/close
      L       stop all motion
      R       reset the scene
      Ctrl-C  quit
    ------------------------------------------------------------
"""


class TerminalTwistReader:
    """Reads teleop keys from the terminal rather than from the Kit window.

    Kit's own input stack is not usable for this: the viewport binds W/A/S/D to its fly-camera, and
    the device layer needs the Kit window to hold OS focus, which competes with the terminal the
    reward trace is being read from. Reading the tty directly avoids both problems and keeps
    working under the lighter ``--viz newton_gl`` viewer, which has no Kit window at all.
    """

    LABELS = {
        "w": "+x fwd", "s": "-x back", "a": "+y left", "d": "-y right",
        "q": "+z up", "e": "-z down", "z": "+roll", "x": "-roll",
        "t": "+pitch", "g": "-pitch", "c": "+yaw", "v": "-yaw",
    }

    def __init__(self, pos_step: float, rot_step: float):
        self._pos_step = pos_step
        self._rot_step = rot_step
        self._hold = {}  # key -> control steps still counting down
        self._close_gripper = False
        # cbreak mode turns off terminal echo, so a key press produces no visible sign whatsoever.
        # Without an explicit acknowledgement there is no way to tell "the key did not reach us"
        # from "the key reached us and the arm is moving too slowly to notice".
        self.key_count = 0
        self.last_key = "-"
        self._interactive = sys.stdin.isatty()
        self._settings = None
        if self._interactive:
            self._settings = termios.tcgetattr(sys.stdin)
            # cbreak rather than raw: keys arrive unbuffered without waiting for Enter, but Ctrl-C
            # still reaches Python as KeyboardInterrupt instead of arriving as a byte
            tty.setcbreak(sys.stdin.fileno())

    def restore(self) -> None:
        """Put the terminal back the way it was found."""
        if self._settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)
            self._settings = None

    def _drain(self) -> list[str]:
        """Every key pressed since the last call, without ever blocking the simulation."""
        keys = []
        while self._interactive and select.select([sys.stdin], [], [], 0.0)[0]:
            char = sys.stdin.read(1)
            if not char:
                break
            keys.append(char.lower())
        return keys

    def advance(self, on_reset) -> np.ndarray:
        """Return the 7-element action: 6 twist components then the gripper command."""
        for key in self._drain():
            self.key_count += 1
            self.last_key = key
            if key == "k":
                self._close_gripper = not self._close_gripper
                print(f"  [key] k -> gripper {'CLOSE' if self._close_gripper else 'OPEN'}\r")
            elif key == "l":
                self._hold.clear()
                print("  [key] l -> stop\r")
            elif key == "r":
                self._hold.clear()
                print("  [key] r -> reset\r")
                on_reset()
            elif key in KEY_BINDINGS:
                self._hold[key] = HOLD_STEPS
                print(f"  [key] {key} -> {self.LABELS[key]}\r")
            else:
                print(f"  [key] {key!r} is not bound\r")

        twist = np.zeros(6)
        for key, remaining in list(self._hold.items()):
            axis, sign = KEY_BINDINGS[key]
            twist[axis] += sign * (self._pos_step if axis < 3 else self._rot_step)
            if remaining <= 1:
                del self._hold[key]
            else:
                self._hold[key] = remaining - 1

        gripper = -1.0 if self._close_gripper else 1.0
        return twist, gripper

    def status(self) -> str:
        """One-line summary of what the reader currently believes the operator is doing."""
        held = "".join(sorted(self._hold)).upper() or "-"
        return (
            f"keys={held:<4} grip={'CLOSE' if self._close_gripper else 'OPEN ':<5}"
            f" rx={self.key_count:<4}"
        )


class EndEffectorTarget:
    """The latched pose the IK controller is asked to hold, in the robot's root frame.

    The operator's keys move *this*, not the robot. Keeping a target that persists between steps is
    what stops the arm sinking: the controller always has a fixed pose to pull back to, instead of
    re-adopting whatever gravity has already dragged it down to.
    """

    GRASP_OFFSET = (0.0, 0.0, 0.1034)
    """Hand-frame to grasp-frame offset, matching the action term's ``body_offset``."""

    def __init__(self, env: ManagerBasedRLEnv, hand_body_name: str = "panda_hand"):
        self._robot = env.scene["robot"]
        self._body_id = self._robot.body_names.index(hand_body_name)
        self._offset = torch.tensor(self.GRASP_OFFSET, device=env.device)
        self.pos = torch.zeros(env.num_envs, 3, device=env.device)
        self.quat = torch.zeros(env.num_envs, 4, device=env.device)

    def measure_world(self) -> torch.Tensor:
        """Where the grasp frame actually is right now, in world coordinates."""
        pose = self._robot.data.body_link_pose_w.torch[:, self._body_id]
        return pose[:, :3] + quat_apply(pose[:, 3:7], self._offset.expand(len(pose), 3))

    def latch(self) -> None:
        """Snap the target onto where the end-effector actually is, in the root frame."""
        pose = self._robot.data.body_link_pose_w.torch[:, self._body_id]
        ee_pos_w = pose[:, :3] + quat_apply(pose[:, 3:7], self._offset.expand(len(pose), 3))
        self.pos[:], self.quat[:] = subtract_frame_transforms(
            self._robot.data.root_pos_w.torch, self._robot.data.root_quat_w.torch, ee_pos_w, pose[:, 3:7]
        )

    def integrate(self, twist: np.ndarray) -> torch.Tensor:
        """Advance the target by one operator twist and return it as a 7-dim pose command."""
        delta = torch.as_tensor(twist, dtype=torch.float32, device=self.pos.device)
        # translation in the root frame, so the keys stay bound to fixed table axes rather than
        # swinging around with the wrist
        self.pos += delta[:3]

        angle = delta[3:].norm()
        if angle > 1e-6:
            axis = (delta[3:] / angle).unsqueeze(0)
            delta_quat = quat_from_angle_axis(angle.view(1), axis)
            # pre-multiply: the rotation is about root-frame axes, not the tool's own
            self.quat[:] = quat_mul(delta_quat.expand_as(self.quat), self.quat)

        return torch.cat([self.pos, self.quat], dim=-1)


class EpisodeRecorder:
    """Buffers one episode of per-step data and writes it out when the episode ends.

    Everything the grasp test looks at is recorded from the reward term's own published values
    rather than recomputed here, so what lands in the file is exactly what the reward saw -- a
    reconstruction could differ in precisely the details worth investigating.
    """

    def __init__(self, path: str):
        self._path = path
        self._rows: list[dict[str, float]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def record(self, env, step: int, reward: float) -> None:
        row = {"step": step, "reward_total": reward}
        for name, value in grasp_debug(env).items():
            row[name] = float(value[0])
        for name, value in env.reward_manager.get_active_iterable_terms(0):
            # the manager reports per-second values; scale to what the term actually paid this step
            row[f"r_{name}"] = value[0] * env.step_dt
        self._rows.append(row)

    def discard(self) -> None:
        self._rows.clear()

    def write(self, terminated_by: str) -> str | None:
        """Write the buffer to disk and clear it. Returns the path, or None if nothing was kept."""
        if len(self._rows) < MIN_RECORDED_STEPS:
            self.discard()
            return None
        for row in self._rows:
            row["terminated_by"] = ""
        self._rows[-1]["terminated_by"] = terminated_by

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        # union of keys, since a term can appear part-way through if the manager order shifts
        columns = list(dict.fromkeys(key for row in self._rows for key in row))
        with open(self._path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self._rows)
        self.discard()
        return self._path


def print_episode_summary(env, totals: dict[str, float], length: int) -> None:
    """Print what the episode that just ended earned, and what ended it.

    Every line uses a carriage return: the terminal is in cbreak mode for the key reader, which
    does not translate newlines on output.
    """
    fired = [name for name in env.termination_manager.active_terms if env.termination_manager.get_term(name)[0]]

    print("\r")
    print("=" * 66 + "\r")
    print(f"  EPISODE ENDED after {length} steps -- {', '.join(fired) or 'unknown'}\r")
    print("=" * 66 + "\r")

    print("  reward term            total     per step\r")
    print("  " + "-" * 62 + "\r")
    # largest contributors first: with one-shot bonuses mixed in among per-step shaping, ordering
    # by magnitude is the quickest way to see what actually drove the return
    for name, value in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<20} {value:>+10.3f}   {value / max(length, 1):>+10.4f}\r")
    print("  " + "-" * 62 + "\r")
    print(f"  {'TOTAL':<20} {sum(totals.values()):>+10.3f}\r")

    # the reward manager and the grasp term both write their per-episode figures into the log dict
    # during the reset that has just happened, so they are current here
    gauges = {k: v for k, v in env.extras.get("log", {}).items() if k.startswith("Events/")}
    if gauges:
        print("\r")
        print("  grasp diagnostics\r")
        print("  " + "-" * 62 + "\r")
        for key in sorted(gauges):
            print(f"  {key[len('Events/'):]:<20} {float(gauges[key]):>10.4f}\r")
    print("=" * 66 + "\r")
    print("\r")


def main() -> None:
    # ``ik_rel`` swaps the joint-delta arm action for a differential-IK twist; every other part of
    # the config -- scene, events, rewards, commands -- is untouched, which is the point.
    env_cfg = resolve_presets(SheetRlEnvCfg(), selected={"ik_rel"})
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # No time limit, ever. A person works far slower than a policy: 500 steps is under twenty
    # seconds, not enough to line up a grasp by hand, and an episode cut off by the clock says
    # nothing about whether the attempt would have worked. The task's own terminations still fire,
    # so the episode ends when something actually happens -- the sheet comes out, or it is dropped.
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = TELEOP_EPISODE_SECONDS
    # the goal resamples on a timer tied to the episode length, and a resample re-rolls where the
    # red band sits on the arm. Left at the training value the band would silently jump every 16
    # seconds mid-session, so it has to be stretched to match.
    env_cfg.commands.deformable_pose.resampling_time_range = (
        TELEOP_EPISODE_SECONDS,
        TELEOP_EPISODE_SECONDS,
    )

    if args_cli.free_run:
        # a human operator trips these constantly -- dropping the end-effector below the table
        # while feeling for the sheet's edge, or shoving the sheet out of bounds -- so there is an
        # opt-out for when the point is to explore the scene rather than to fly a real episode.
        for name in ("ee_below_table", "deformable_out_of_bounds", "joint_vel_out_of_limit"):
            setattr(env_cfg.terminations, name, None)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    keyboard = TerminalTwistReader(
        pos_step=STEP_TRANSLATION * args_cli.sensitivity,
        rot_step=STEP_ROTATION * args_cli.sensitivity,
    )
    if not sys.stdin.isatty():
        print("[teleop] stdin is not a terminal, so no keys can be read. Run this in a shell.")
    print(BANNER)

    env.reset()
    target = EndEffectorTarget(env)
    # a reset writes new joint positions but the body poses derived from them are only refreshed by
    # the next step, so a target latched immediately would be the pose from *before* the reset and
    # the controller would lunge for it. Re-latching on the next couple of steps picks up the real
    # post-reset pose; the operator loses two steps of input, which is under a tenth of a second.
    pending_latch = 2

    def on_reset() -> None:
        nonlocal pending_latch
        env.reset()
        pending_latch = 2

    step = 0
    episode_step = 0
    totals = {name: 0.0 for name in env.reward_manager.active_terms}
    recorder = EpisodeRecorder(RECORDING_PATH)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                twist, gripper = keyboard.advance(on_reset)
                if pending_latch > 0:
                    target.latch()
                    pending_latch -= 1
                    twist = np.zeros(6)
                # 8 values: 3 position + 4 quaternion + 1 gripper, as the action manager expects
                pose = target.integrate(twist)
                gripper_col = torch.full((env.num_envs, 1), gripper, device=env.device)
                actions = torch.cat([pose, gripper_col], dim=-1)
                _, reward, terminated, truncated, _ = env.step(actions)

                # accumulate this environment's return, term by term. The manager reports
                # per-second values, so scaling by the step gives what each term actually paid.
                for name, value in env.reward_manager.get_active_iterable_terms(0):
                    totals[name] += value[0] * env.step_dt
                episode_step += 1
                recorder.record(env, episode_step, float(reward[0]))

                if bool(terminated[0]) or bool(truncated[0]):
                    fired = [
                        name
                        for name in env.termination_manager.active_terms
                        if env.termination_manager.get_term(name)[0]
                    ]
                    written = recorder.write(", ".join(fired) or "unknown")
                    print_episode_summary(env, totals, episode_step)
                    if written:
                        print(f"  trace written to {written}\r\n\r")
                    # Stop at the first episode that ends on its own, so the trace on disk is
                    # always the attempt just flown and there is nothing to press to preserve it.
                    if not args_cli.keep_flying:
                        break
                    totals = {name: 0.0 for name in totals}
                    episode_step = 0
                    # the environment has already reset itself, so the latched target now refers to
                    # a pose the arm no longer holds and the controller would lunge for it
                    pending_latch = 2

                # env_0 only: the other environments are along for the ride, and printing all of
                # them would scroll the useful one off the screen. The manager reports per-second
                # values, so scale back to the per-step numbers the reward was designed around.
                if step % 8 == 0:
                    parts = " ".join(
                        f"{name}={value[0] * env.step_dt:+.2f}"
                        for name, value in env.reward_manager.get_active_iterable_terms(0)
                    )
                    # where the grasp frame actually ended up, in the env's own frame -- the number
                    # to compare against the carry waypoint at (0.3, 0.0, 0.35)
                    ee = target.measure_world()[0] - env.scene.env_origins[0]
                    # carriage returns, because cbreak mode does not translate newlines
                    print(
                        f"[{step:5d}] {keyboard.status()}"
                        f" ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f})"
                        f" total={reward[0]:+8.3f} | {parts}\r"
                    )
                step += 1
    except KeyboardInterrupt:
        # keep whatever was flown before the interrupt: quitting straight after a good attempt is
        # the reliable way to stop the next episode overwriting it
        written = recorder.write("interrupted")
        if written:
            print(f"\r\n  trace written to {written}\r")
    finally:
        # always hand the terminal back, or the shell is left without echo after a crash
        keyboard.restore()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
