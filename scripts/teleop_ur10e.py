# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fly the UR10e twist task from the numeric keypad, in place of the policy.

The same environment the checkpoint runs in -- ``SheetUR10eTwistStudentEnvCfg``, same scene,
same reset, same rewards and terminations -- with the policy's seven outputs supplied by hand.
Nothing is substituted: the task's action *is* a relative tool twist plus a binary gripper, so
a keypress maps straight onto it with no pose latching or IK of its own.

Keys are read from the **terminal**, not from the 3D window, for the reasons set out in
``teleop_keyboard.py``: Kit binds its own keys in the viewport, and ``--viz newton_gl`` has no
Kit window to receive events at all. Click the terminal, not the render window.

Numeric keypad, NumLock ON::

      [7] roll -    [8] +Z up      [9] roll +
      [4] +Y left   [5] GRIPPER    [6] -Y right
      [1] pitch -   [2] -Z down    [3] pitch +
      [0] +X fwd    [.] -X back
      [-] yaw -     [+] yaw +

      5         toggle gripper: close, and open again if closed
      L         stop all motion
      R         reset the scene
      Ctrl-C    quit

With NumLock off the keypad sends cursor escape sequences instead of digits and nothing will
move; the top-row digits work as well, except that they cannot be held comfortably.

Usage::

    python scripts/teleop_ur10e.py --viz newton_gl
    python scripts/teleop_ur10e.py --viz newton_gl --sensitivity 2.0 --free_run
"""

# Warp captures ``enable_backward`` when a module is created, which happens at import time, so it
# has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keypad teleoperation for the UR10e twist task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Multiplier on how far one key press moves the tool."
)
parser.add_argument(
    "--free_run",
    action="store_true",
    help=(
        "Disable the terminations a person trips by accident -- the tool dipping below the table,"
        " the sheet leaving its bounds, the arm exceeding joint speed limits -- and, importantly,"
        " and the released_high/held_too_low pair. (The one-gripper-cycle rule is already off in"
        " the UR10e task itself, so opening after a closure does not end the episode.)"
    ),
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed, so a recording can be replayed.")
parser.add_argument(
    "--record",
    default="logs/teleop/ur10e_last.csv",
    help=(
        "Where to write the per-step trace of this session (plus a .meta.json sidecar)."
        " Replay it with scripts/replay_teleop_ur10e.py. Empty string disables recording."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# A viewer is the whole point here, so never headless. Clearing the flag is not enough on its own:
# the launcher forces headless back on unless a visualizer is explicitly requested.
args_cli.headless = False
if getattr(args_cli, "visualizer", None) is None:
    args_cli.visualizer = ["newton_gl"]
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below here runs after the simulator is up."""

import select  # noqa: E402
import sys  # noqa: E402
import termios  # noqa: E402
import tty  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import resolve_presets  # noqa: E402

from sheet_rl.tasks.manager_based.sheet_rl.sheet_rl_env import PushableSheetEnv  # noqa: E402
from sheet_rl.tasks.manager_based.sheet_twist.sheet_ur10e_env_cfg import (  # noqa: E402
    SheetUR10eTwistStudentEnvCfg,
)
from sheet_rl.tasks.manager_based.sheet_twist.teleop_trace import TraceWriter, capture_row  # noqa: E402
from sheet_rl.tasks.manager_based.sheet_twist.ur10e_mdp import never  # noqa: E402

# Metres and radians of tool motion per held key per control step, before --sensitivity. The task
# runs at 50 Hz, so these are about 0.35 m/s and 1.5 rad/s: quick enough to cross the table in a
# couple of seconds, slow enough to place a fingertip on the sheet's edge.
STEP_TRANSLATION = 0.007
STEP_ROTATION = 0.03

TELEOP_EPISODE_SECONDS = 3600.0
"""Episode length while teleoperating [s]. An hour, i.e. effectively never."""

HOLD_STEPS = 8
"""How many control steps one key press stays active for.

A terminal reports presses but never releases, so "held" has to be inferred. Holding a key makes
the terminal auto-repeat at roughly 30 Hz, which re-arms it faster than this expires and reads as
continuous motion; letting go lets it lapse after about a quarter second.
"""

# key -> (axis index in the twist's own (x, y, z, rx, ry, rz) order, sign)
KEY_BINDINGS = {
    "0": (0, +1.0), ".": (0, -1.0), ",": (0, -1.0),  # some keypads send ',' for Del
    "4": (1, +1.0), "6": (1, -1.0),
    "8": (2, +1.0), "2": (2, -1.0),
    "9": (3, +1.0), "7": (3, -1.0),
    "3": (4, +1.0), "1": (4, -1.0),
    "+": (5, +1.0), "-": (5, -1.0),
}

LABELS = {
    "0": "+x fwd", ".": "-x back", ",": "-x back",
    "4": "+y left", "6": "-y right",
    "8": "+z up", "2": "-z down",
    "9": "+roll", "7": "-roll",
    "3": "+pitch", "1": "-pitch",
    "+": "+yaw", "-": "-yaw",
}

BANNER = """
    ============================================================
      CLICK THIS TERMINAL WINDOW, THEN PRESS KEYS.  NUMLOCK ON.
      Keys are read here, NOT in the 3D window. The 3D window is
      only for looking -- it will ignore everything you type.
      Every accepted key prints a '[key] ...' line below, and the
      'rx=' counter in the status line goes up. If neither moves,
      this terminal does not have focus, or NumLock is off.
    ============================================================
      [7] roll -    [8] +Z up      [9] roll +
      [4] +Y left   [5] GRIPPER    [6] -Y right
      [1] pitch -   [2] -Z down    [3] pitch +
      [0] +X fwd    [.] -X back
      [-] yaw -     [+] yaw +
    ------------------------------------------------------------
      5       toggle gripper (close; open again if closed)
      L       stop all motion
      R       reset the scene
      Ctrl-C  quit
    ------------------------------------------------------------
"""


class KeypadTwistReader:
    """Turns keypad presses into the task's seven action channels."""

    def __init__(self, pos_step: float, rot_step: float):
        self._pos_step = pos_step
        self._rot_step = rot_step
        self._hold = {}  # key -> control steps still counting down
        self._close_gripper = False
        # cbreak mode turns off terminal echo, so a key press produces no visible sign whatsoever.
        # Without an explicit acknowledgement there is no telling "the key did not reach us" from
        # "the key reached us and the arm is moving too slowly to notice".
        self.key_count = 0
        self._interactive = sys.stdin.isatty()
        self._settings = None
        if self._interactive:
            self._settings = termios.tcgetattr(sys.stdin)
            # cbreak rather than raw: keys arrive unbuffered without waiting for Enter, but Ctrl-C
            # still reaches Python as KeyboardInterrupt instead of arriving as a byte.
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

    def advance(self, on_reset):
        """Return this step's (6-element twist, gripper command)."""
        for key in self._drain():
            self.key_count += 1
            if key == "5":
                self._close_gripper = not self._close_gripper
                print(f"  [key] 5 -> gripper {'CLOSE' if self._close_gripper else 'OPEN'}\r")
            elif key == "l":
                self._hold.clear()
                print("  [key] l -> stop\r")
            elif key == "r":
                self._hold.clear()
                self._close_gripper = False
                print("  [key] r -> reset\r")
                on_reset()
            elif key in KEY_BINDINGS:
                self._hold[key] = HOLD_STEPS
                print(f"  [key] {key} -> {LABELS[key]}\r")
            elif key == "\x1b":
                # NumLock off: the keypad sends cursor escapes, which arrive as ESC [ A and would
                # otherwise be reported one meaningless byte at a time.
                self._drain()
                print("  [key] escape sequence ignored -- is NumLock off?\r")
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

        # positive_threshold=True on the task's gripper term: above the threshold is OPEN.
        return twist, -1.0 if self._close_gripper else 1.0

    def release_gripper(self) -> None:
        """Forget a latched closure, so a new episode starts with the fingers open."""
        self._close_gripper = False

    def status(self) -> str:
        held = "".join(sorted(self._hold)) or "-"
        return (
            f"keys={held:<6} grip={'CLOSE' if self._close_gripper else 'OPEN ':<5}"
            f" rx={self.key_count:<4}"
        )


def main() -> None:
    env_cfg = resolve_presets(SheetUR10eTwistStudentEnvCfg())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Seeded, so the replay's resets draw the same layouts and platform heights in the same order.
    env_cfg.seed = args_cli.seed

    # No time limit, ever. A person works far slower than a policy, and an episode cut off by the
    # clock says nothing about whether the attempt would have worked.
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = TELEOP_EPISODE_SECONDS
    # The goal resamples on a timer tied to the episode length, and a resample re-rolls where the
    # band sits on the arm. Left at the training value it would jump mid-session.
    env_cfg.commands.deformable_pose.resampling_time_range = (
        TELEOP_EPISODE_SECONDS,
        TELEOP_EPISODE_SECONDS,
    )

    if args_cli.free_run:
        # ``placed``/``dropped`` end the episode on the first opening after a closure (the task's
        # one-gripper-cycle rule); a hand-flown session wants to open and close freely. They are
        # replaced by a never-firing stand-in rather than removed, because reward terms read them
        # by name (see ``ur10e_mdp.never``).
        # ``draped`` too: it is the original task's success test (sheet settled on the band for
        # two seconds) and it ended a hand-flown carry over the platform, whereupon the reset
        # that followed crashed the coupled solver (see replay_teleop_ur10e / repro notes).
        for name in ("ee_below_table", "deformable_out_of_bounds", "joint_vel_out_of_limit",
                     "released_high", "held_too_low", "draped"):
            setattr(env_cfg.terminations, name, None)
        for name in ("placed", "dropped"):
            setattr(env_cfg.terminations, name, getattr(env_cfg.terminations, name).replace(func=never))

    # The action term multiplies by its own scale, so divide it out and let the step sizes above
    # stay in metres and radians whatever the config is tuned to.
    scale = np.broadcast_to(np.asarray(env_cfg.actions.arm_action.scale, dtype=np.float64), (6,))

    env = PushableSheetEnv(cfg=env_cfg)
    keyboard = KeypadTwistReader(
        pos_step=STEP_TRANSLATION * args_cli.sensitivity,
        rot_step=STEP_ROTATION * args_cli.sensitivity,
    )
    if not sys.stdin.isatty():
        print("[teleop] stdin is not a terminal, so no keys can be read. Run this in a shell.")
    print(BANNER)
    print(f"    action scale {env_cfg.actions.arm_action.scale} -- one held key is "
          f"{STEP_TRANSLATION * args_cli.sensitivity:.4f} m / "
          f"{STEP_ROTATION * args_cli.sensitivity:.3f} rad per step\r")

    trace = None
    if args_cli.record:
        trace = TraceWriter(args_cli.record, {
            "task": "Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0",
            "seed": args_cli.seed, "free_run": args_cli.free_run, "num_envs": args_cli.num_envs,
            "sensitivity": args_cli.sensitivity, "action_scale": scale.tolist(),
            "step_dt": env_cfg.sim.dt * env_cfg.decimation,
        })
        print(f"    recording to {args_cli.record}\r")

    with torch.inference_mode():
        env.reset()
    episode = 0
    reset_pending = True  # the row after a reset says so, and the replay resets at the same point

    def on_reset() -> None:
        nonlocal episode, reset_pending
        # Under inference mode like the step: the step's tensors are inference tensors, and the
        # reset events update joint state in place, which PyTorch refuses outside the mode.
        # (Pressing the gripper twice ended the episode and crashed here.)
        with torch.inference_mode():
            env.reset()
        episode += 1
        reset_pending = True

    step = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                twist, gripper = keyboard.advance(on_reset)
                command = torch.tensor(twist / scale, dtype=torch.float32, device=env.device)
                actions = torch.cat(
                    [command.expand(env.num_envs, 6),
                     torch.full((env.num_envs, 1), gripper, device=env.device)], dim=-1,
                )
                _, reward, terminated, truncated, _ = env.step(actions)
                done = bool(terminated[0]) or bool(truncated[0])
                ended = [name for name in env.termination_manager.active_terms
                         if bool(env.termination_manager.get_term(name)[0])] if done else []
                if trace is not None:
                    trace.append(capture_row(
                        env, step=step, episode=episode, action=actions[0].tolist(), twist=twist,
                        reset_before=reset_pending, reward=float(reward[0]), ended=ended,
                    ))
            reset_pending = False

            step += 1
            if step % 5 == 0:
                print(f"  step {step:<6} {keyboard.status()} reward={float(reward[0]):+8.3f}\r",
                      end="", flush=True)
            if trace is not None and step % 250 == 0:
                trace.flush()
            if done:
                print(f"\r\n  [episode ended] {', '.join(ended) or 'unknown'} -- resetting\r")
                if trace is not None:
                    trace.flush()
                on_reset()
                keyboard.release_gripper()
    except KeyboardInterrupt:
        print("\r\n[teleop] stopping.\r")
    finally:
        keyboard.restore()
        if trace is not None:
            trace.flush()
            print(f"[teleop] {len(trace.rows)} steps written to {trace.path}\r")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
