"""Tune the UR10e's Robotiq 2F-85 by hand: press Close / Open, watch it, read whether it bounced, save.

The UR10e task's own scene and gripper (``MimicGripperAction``: the two knuckles are driven, the four
followers copy them) run in the Newton GL viewer, with a control window beside it:

- **Sliders** for the TUNING block of ``ur10e_gripper.py`` -- knuckle stiffness, damping, effort
  (the pinch), velocity limit, armature, the followers' stiffness, damping and effort, and the open
  and close angles. A change goes into the running simulation at once.
- **Close / Open** command the gripper exactly as the policy does (the task's filtered binary
  command), so what you see is what an episode does.
- **Measurement of the last command**: time to 90 % of the travel, overshoot, how many times the
  knuckle reversed direction after arriving (a bounce), settling time, final pad gap, peak knuckle
  speed against the velocity limit (which is also the task's joint-speed termination), and the
  left/right knuckle mismatch -- with a plot of the pad gap and knuckle speed.
- **Jog** buttons (hold to move) to bring the fingers to the sheet's top edge and try a real grasp.
- The file's own restriction checks (``ur10e_gripper.validate``) run on every change; **Save**
  writes the values into ``ur10e_gripper.py`` (previous file kept as ``ur10e_gripper.py.bak``) and
  refuses values that fail them, since the task would not build with those.

Saved values take effect the next time the task is created (training, play, etc.).

Usage (inside ``active_isaac``)::

    python scripts/ur10e_gripper_tuning_gui.py
"""

import argparse
import contextlib
import math
import os
import re
import shutil
import statistics
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")  # the Newton GL viewer needs GLX on this machine

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from sheet_rl.tasks.manager_based.sheet_twist import ur10e_gripper as gripper  # noqa: E402

parser = argparse.ArgumentParser(description="Tune the UR10e gripper's closing and opening with a GUI.")
parser.add_argument("--task", type=str, default="Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0")
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--selftest", action="store_true", default=False,
                    help="Press CLOSE, then OPEN, by itself, print the two measurements and exit.")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args

PARAMS = (  # TUNING constant, slider range and step, unit
    ("KNUCKLE_STIFFNESS", 0.0, 50.0, 0.5, "Nm/rad"),
    ("KNUCKLE_DAMPING", 0.0, 2.0, 0.005, "Nm s/rad"),
    ("KNUCKLE_EFFORT", 0.05, 3.0, 0.05, "Nm"),
    ("VELOCITY_LIMIT", 0.5, 15.0, 0.1, "rad/s"),
    ("ARMATURE", 0.0005, 0.02, 0.0005, "kg m^2"),
    ("FOLLOWER_STIFFNESS", 0.0, 50.0, 0.5, "Nm/rad"),
    ("FOLLOWER_DAMPING", 0.0, 0.5, 0.005, "Nm s/rad"),
    ("FOLLOWER_EFFORT", 0.05, 5.0, 0.05, "Nm"),
    ("OPEN_ANGLE", 0.0, 0.4, 0.005, "rad"),
    ("CLOSE_ANGLE", 0.3, 0.82, 0.005, "rad"),
)
WINDOW_S = 2.0
"""Seconds of each command's response that are measured and plotted."""
GAP_BAND_MM = 0.5
"""Settled = the pad gap stays within this of its final value [mm]."""
REVERSAL_DEADBAND = 0.05
"""Knuckle speeds below this [rad/s] do not count as moving when counting direction reversals."""


@contextlib.contextmanager
def candidate(values):
    """Temporarily put ``values`` into the tuning module, so its checks and description see them."""
    saved = {name: getattr(gripper, name) for name in values}
    try:
        for name, value in values.items():
            setattr(gripper, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(gripper, name, value)


def problems(values):
    """The restriction violations of ``ur10e_gripper.validate`` for these values (empty when they pass)."""
    with candidate(values):
        try:
            gripper.validate()
            return []
        except ValueError as error:
            return [line.strip() for line in str(error).splitlines()[1:]]


def describe(values):
    with candidate(values):
        return gripper.describe()


def save_to_file(values):
    """Write the values into the TUNING block of ur10e_gripper.py, keeping each line's comment."""
    path = gripper.__file__
    text = open(path).read()
    for name, value in values.items():
        number = repr(float(f"{value:.6g}"))
        text, count = re.subn(rf"^({name}\s*=\s*)[-+0-9.eE]+", lambda m: m.group(1) + number, text, count=1, flags=re.M)
        if count != 1:
            raise RuntimeError(f"{name} not found in {path}")
    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        f.write(text)
    return path


def measure(trace, closing):
    """Response of one command from its trace [(t, gap_m, qL, qR, vL, vR)], as a dict, or None if too short."""
    if len(trace) < 5:
        return None
    t = [r[0] - trace[0][0] for r in trace]
    gap = [r[1] * 1000 for r in trace]
    tail = [g for ti, g in zip(t, gap) if ti >= t[-1] - 0.2] or gap[-1:]
    final = statistics.median(tail)
    travel = abs(final - gap[0])
    t90 = next((ti for ti, g in zip(t, gap) if abs(g - gap[0]) >= 0.9 * travel), None) if travel > 1 else None
    overshoot = max(0.0, (final - min(gap)) if closing else (max(gap) - final))
    after = [r for ti, r in zip(t, trace) if t90 is not None and ti >= t90]
    reversals = 0
    for side in (4, 5):
        signs = [1 if r[side] > REVERSAL_DEADBAND else -1 for r in after if abs(r[side]) > REVERSAL_DEADBAND]
        reversals = max(reversals, sum(a != b for a, b in zip(signs, signs[1:])))
    settled = max((ti for ti, g in zip(t, gap) if abs(g - final) > GAP_BAND_MM), default=0.0)
    return {"elapsed": t[-1], "travel": travel, "t90": t90, "overshoot": overshoot, "reversals": reversals,
            "settled": settled, "final": final, "peak": max(max(abs(r[4]), abs(r[5])) for r in trace),
            "mismatch": abs(trace[-1][2] - trace[-1][3])}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    import tkinter as tk

    from isaaclab.managers import SceneEntityCfg
    from sheet_rl.tasks.manager_based.sheet_rl.mdp.gripper_geometry import finger_half_width

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.scene.table.spawn.visible = True  # Newton GL draws only visible prims; the table collider is hidden
    if not getattr(args_cli, "visualizer", None):  # Newton GL, close on the gripper (--viz kit for Isaac Sim)
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(0.95, -0.55, 0.55), lookat=(0.45, 0.0, 0.33))]
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        env.reset()
        robot = inner.scene["robot"]
        actions = inner.action_manager
        grip_term = actions.get_term("gripper_action")
        arm_scale = float(actions.get_term("arm_action")._scale[0, 0])
        knuckle_ids = robot.find_joints(gripper.KNUCKLES, preserve_order=True)[0]
        follower_ids = robot.find_joints(list(gripper.LEADERS), preserve_order=True)[0]
        left, right = knuckle_ids
        signs = torch.tensor([gripper.SIGNS[n] for n in grip_term._joint_names], device=inner.device)
        fingers = SceneEntityCfg("robot")
        dt = inner.step_dt if inner._physics_handles_decimation else inner.physics_dt
        rate = f"{1 / dt:.0f} Hz"

        def apply(values):
            for ids, k, d, e in ((knuckle_ids, "KNUCKLE_STIFFNESS", "KNUCKLE_DAMPING", "KNUCKLE_EFFORT"),
                                 (follower_ids, "FOLLOWER_STIFFNESS", "FOLLOWER_DAMPING", "FOLLOWER_EFFORT")):
                robot.write_joint_stiffness_to_sim_index(stiffness=values[k], joint_ids=ids)
                robot.write_joint_damping_to_sim_index(damping=values[d], joint_ids=ids)
                robot.write_joint_effort_limit_to_sim_index(limits=values[e], joint_ids=ids)
                robot.write_joint_velocity_limit_to_sim_index(limits=values["VELOCITY_LIMIT"], joint_ids=ids)
                robot.write_joint_armature_to_sim_index(armature=values["ARMATURE"], joint_ids=ids)
            grip_term._open_command[:] = values["OPEN_ANGLE"] * signs
            grip_term._close_command[:] = values["CLOSE_ANGLE"] * signs

        # ---------------------------------------------------------------- window
        root = tk.Tk()
        root.title("UR10e gripper tuning")
        left_col, right_col = tk.Frame(root), tk.Frame(root)
        left_col.pack(side="left", fill="y", padx=8, pady=6)
        right_col.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        tk.Label(left_col, text="ur10e_gripper.py TUNING (applied live)", font=("Helvetica", 11, "bold")).pack(anchor="w")
        variables = {}
        for name, lo, hi, step, unit in PARAMS:
            row = tk.Frame(left_col)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, font=("Courier", 10), width=19, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=getattr(gripper, name))
            tk.Scale(row, from_=lo, to=hi, resolution=step, orient="horizontal", length=260, showvalue=False,
                     variable=var).pack(side="left")
            tk.Entry(row, width=8, textvariable=var).pack(side="left", padx=3)
            tk.Label(row, text=unit, width=8, anchor="w").pack(side="left")
            variables[name] = var
        checks = tk.StringVar()
        check_label = tk.Label(left_col, textvariable=checks, justify="left", anchor="w", wraplength=480)
        check_label.pack(fill="x", pady=(6, 2))
        description = tk.StringVar()
        tk.Label(left_col, textvariable=description, justify="left", anchor="w", font=("Courier", 9),
                 wraplength=480).pack(fill="x")
        status = tk.StringVar(value=f"tuning file: {gripper.__file__}")
        tk.Label(left_col, textvariable=status, justify="left", anchor="w", fg="#204080", wraplength=480).pack(fill="x", pady=4)
        files = tk.Frame(left_col)
        files.pack(anchor="w", pady=4)

        state = {"cmd": 1.0, "closing": False, "trace": [], "t": 0.0, "t_cmd": None, "jog": None, "jog_until": 0.0,
                 "reset": False, "closed": False, "applied": None, "reported": True}

        def values():
            out = {}
            for name, var in variables.items():
                try:
                    out[name] = float(var.get())
                except (tk.TclError, ValueError):
                    out[name] = getattr(gripper, name)
            return out

        def on_save():
            v = values()
            bad = problems(v)
            if bad:
                status.set("NOT saved -- the task refuses these values:\n  " + "\n  ".join(bad))
                return
            path = save_to_file(v)
            for name, value in v.items():  # the saved values are now this session's baseline too
                setattr(gripper, name, value)
            status.set(f"saved to {path} (previous file kept as .bak). Takes effect the next time the task is created.")

        def on_reload():
            for name, var in variables.items():
                var.set(getattr(gripper, name))
            status.set("reloaded the saved values")

        tk.Button(files, text="Save to ur10e_gripper.py", command=on_save).pack(side="left", padx=2)
        tk.Button(files, text="Reload saved values", command=on_reload).pack(side="left", padx=2)

        def command(close):
            state.update(cmd=-1.0 if close else 1.0, closing=close, trace=[], t_cmd=state["t"], reported=False)

        buttons = tk.Frame(right_col)
        buttons.pack(anchor="w")
        tk.Button(buttons, text="CLOSE", width=12, height=2, bg="#c86464", command=lambda: command(True)).pack(side="left", padx=4)
        tk.Button(buttons, text="OPEN", width=12, height=2, bg="#64a064", command=lambda: command(False)).pack(side="left", padx=4)
        tk.Button(buttons, text="Reset scene", command=lambda: state.update(reset=True)).pack(side="left", padx=12)
        speed = tk.DoubleVar(value=1.0)
        tk.Label(buttons, text="speed").pack(side="left")
        tk.Scale(buttons, from_=0.05, to=1.0, resolution=0.05, orient="horizontal", length=120, variable=speed).pack(side="left")
        tk.Label(buttons, text="x real time").pack(side="left")

        jog = tk.Frame(right_col)
        jog.pack(anchor="w", pady=6)
        tk.Label(jog, text="jog the tool (hold):").pack(side="left")
        for text, vec in (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("+y", (0, 1, 0)), ("-y", (0, -1, 0)),
                          ("up", (0, 0, 1)), ("down", (0, 0, -1))):
            tk.Button(jog, text=text, width=5, repeatdelay=1, repeatinterval=40,
                      command=lambda v=vec: state.update(jog=v, jog_until=time.time() + 0.08)).pack(side="left", padx=1)
        jog_speed = tk.DoubleVar(value=0.2)
        tk.Scale(jog, from_=0.05, to=1.0, resolution=0.05, orient="horizontal", length=100, variable=jog_speed,
                 label="jog speed").pack(side="left", padx=6)

        result = tk.StringVar(value="press CLOSE or OPEN")
        tk.Label(right_col, textvariable=result, justify="left", anchor="w", font=("Courier", 10)).pack(fill="x", pady=4)
        W, H = 640, 260
        canvas = tk.Canvas(right_col, width=W, height=H, bg="white")
        canvas.pack()
        live = tk.StringVar()
        tk.Label(right_col, textvariable=live, justify="left", anchor="w", font=("Courier", 10)).pack(fill="x", pady=4)
        root.protocol("WM_DELETE_WINDOW", lambda: state.update(closed=True))

        def draw(trace, limit):
            canvas.delete("all")
            pad = 34
            canvas.create_rectangle(pad, 10, W - 10, H - 24, outline="#888")
            for frac in (0.25, 0.5, 0.75):
                y = 10 + (H - 34) * frac
                canvas.create_line(pad, y, W - 10, y, fill="#eee")
            canvas.create_text(pad, H - 12, text=f"0 s   (one {'CLOSE' if state['closing'] else 'OPEN'} command, {WINDOW_S:.0f} s window, "
                               f"sampled at {rate})", anchor="w", font=("Helvetica", 8))
            canvas.create_text(W - 12, H - 12, text=f"{WINDOW_S:.0f} s", anchor="e", font=("Helvetica", 8))
            vmax = max(10.0, 1.5 * limit)
            y_vlim = 10 + (H - 34) * (1 - limit / vmax)
            canvas.create_line(pad, y_vlim, W - 10, y_vlim, fill="#e8a060", dash=(4, 3))
            canvas.create_text(W - 14, y_vlim - 7, text=f"velocity limit {limit:.1f} rad/s", anchor="e", fill="#c07030", font=("Helvetica", 8))
            canvas.create_text(4, 14, text="90mm", anchor="w", fill="#2060c0", font=("Helvetica", 8))
            canvas.create_text(4, H - 30, text="0", anchor="w", fill="#2060c0", font=("Helvetica", 8))
            if len(trace) < 2:
                return
            t0 = trace[0][0]
            xs = [pad + (W - 10 - pad) * min((r[0] - t0) / WINDOW_S, 1.0) for r in trace]
            gap = [10 + (H - 34) * (1 - min(max(r[1] * 1000 / 90, 0), 1)) for r in trace]
            vel = [10 + (H - 34) * (1 - min(max(abs(r[4]), abs(r[5])) / vmax, 1)) for r in trace]
            canvas.create_line(*[c for xy in zip(xs, vel) for c in xy], fill="#e08030", width=1)
            canvas.create_line(*[c for xy in zip(xs, gap) for c in xy], fill="#2060c0", width=2)
            canvas.create_text(W - 14, 18, text="pad gap [mm]", anchor="e", fill="#2060c0", font=("Helvetica", 9, "bold"))
            canvas.create_text(W - 14, 32, text="knuckle speed [rad/s]", anchor="e", fill="#e08030", font=("Helvetica", 9, "bold"))

        viewers = [type(viz).__name__ for viz in getattr(inner.sim, "_visualizers", [])]
        print(f"[gripper-gui] ready: sampling the gripper at {rate}; viewer {', '.join(viewers) or 'none'}", flush=True)
        last_draw = 0.0
        while not state["closed"]:
            tick = time.time()
            if state["reset"]:
                env.reset()
                state.update(reset=False, trace=[], t_cmd=None, cmd=1.0)
                state["applied"] = None  # re-apply, in case the reset restored the configured gains
            v = values()
            if v != state["applied"]:
                apply(v)
                state["applied"] = v
                bad = problems(v)
                checks.set("checks: OK" if not bad else "checks FAIL (cannot save):\n  " + "\n  ".join(bad))
                check_label.config(fg="#207020" if not bad else "#c02020")
                description.set(describe(v))
            arm = torch.zeros(6, device=inner.device)
            if state["jog"] is not None and time.time() < state["jog_until"]:
                arm[:3] = torch.tensor(state["jog"], device=inner.device, dtype=torch.float) * jog_speed.get()
            action = torch.cat((arm, torch.tensor([state["cmd"]], device=inner.device)))[None]
            # the environment's own step, physics only (no rewards, terminations or resets)
            actions.process_action(action)
            substeps = 1 if inner._physics_handles_decimation else inner.cfg.decimation
            for _ in range(substeps):
                actions.apply_action()
                inner.scene.write_data_to_sim()
                inner.sim.step(render=False)
                inner.scene.update(dt=dt)
                state["t"] += dt
                if state["t_cmd"] is not None and state["t"] - state["t_cmd"] <= WINDOW_S:
                    q, qd = robot.data.joint_pos.torch[0], robot.data.joint_vel.torch[0]
                    gap = 2 * finger_half_width(inner, fingers)[0].item()
                    state["trace"].append((state["t"], gap, q[left].item(), q[right].item(), qd[left].item(), qd[right].item()))
            inner.sim.render()

            if state["t_cmd"] is not None and not state["reported"] and state["t"] - state["t_cmd"] > WINDOW_S:
                state["reported"] = True  # the command's window is complete: put its measurement in the terminal too
                m = measure(state["trace"], state["closing"])
                if m:
                    print(f"[gripper-gui] {'CLOSE' if state['closing'] else 'OPEN '}: "
                          f"90% in {m['t90'] if m['t90'] is None else round(m['t90'], 3)} s, overshoot {m['overshoot']:.2f} mm, "
                          f"{m['reversals']} reversal(s), settled {m['settled']:.3f} s, final gap {m['final']:.1f} mm, "
                          f"peak {m['peak']:.2f} rad/s (limit {v['VELOCITY_LIMIT']:.1f}), left-right {m['mismatch']:.3f} rad",
                          flush=True)
            if args_cli.selftest:  # CLOSE at 0.5 s, OPEN at 3 s, done at 5.5 s (simulated time)
                if state["t_cmd"] is None and state["t"] >= 0.5:
                    command(True)
                elif state["closing"] and state["t"] >= 3.0:
                    command(False)
                elif not state["closing"] and state["t_cmd"] is not None and state["t"] >= 5.5:
                    state["closed"] = True
            q = robot.data.joint_pos.torch[0]
            live.set(f"now: pad gap {2 * finger_half_width(inner, fingers)[0].item() * 1000:5.1f} mm | knuckles "
                     f"{q[left].item():.3f} / {q[right].item():.3f} rad | command {'CLOSE' if state['cmd'] < 0 else 'OPEN'}")
            if time.time() - last_draw > 0.1:
                last_draw = time.time()
                m = measure(state["trace"], state["closing"]) if state["t_cmd"] is not None else None
                if m:
                    word = "CLOSE" if state["closing"] else "OPEN"
                    bounce = "NO BOUNCE" if m["reversals"] == 0 and m["overshoot"] < 0.3 else f"BOUNCE: {m['reversals']} reversal(s)"
                    over = " OVER THE LIMIT" if m["peak"] > v["VELOCITY_LIMIT"] else ""
                    timing = (f"90% of travel in {m['t90']:.3f} s" if m["t90"] is not None
                              else f"travel {m['travel']:.1f} mm (too small to time)")
                    result.set(f"{word}  ({m['elapsed']:.2f} s measured)   {bounce}\n"
                               f"  {timing} | overshoot {m['overshoot']:.2f} mm | settled after {m['settled']:.3f} s "
                               f"(+-{GAP_BAND_MM} mm)\n"
                               f"  final gap {m['final']:.1f} mm | peak knuckle speed {m['peak']:.2f} rad/s "
                               f"(limit {v['VELOCITY_LIMIT']:.1f}){over} | left-right {m['mismatch']:.3f} rad")
                draw(state["trace"], v["VELOCITY_LIMIT"])
            root.update()
            wait = dt * substeps / max(speed.get(), 0.05) - (time.time() - tick)
            if wait > 0:
                time.sleep(wait)
        root.destroy()
        env.close()


if __name__ == "__main__":
    main()
