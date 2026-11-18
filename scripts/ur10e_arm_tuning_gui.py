"""Tune the UR10e arm's control so its tool answers twist commands like the Franka's.

The same twist pulse -- one axis held at an amplitude for 0.5 s (adjustable), through the task's own
IK action, exactly as the policy commands it -- goes to the Franka task and the UR10e task at the
same time. Both start from the same tool pose (the Franka's unrandomized start), and their tool
responses in all six axes (x, y, z in mm, rx, ry, rz in deg, robot-frame axes, measured from the pose
at the pulse onset, sampled every physics step) are plotted on top of each other, with a table: how
far each tool moved by the end of the pulse and in the end, time to 90 %, overshoot, the UR10e's
difference from the Franka, the cross-coupling into the other axes, and the peak joint speed against
the UR10e's limits.

The UR10e's arm control is edited live, per joint group (shoulder, elbow, wrist): drive stiffness,
damping, armature, effort limit and velocity limit, plus the IK's damping. **Save** writes it to
``ur10e_arm_control.json`` beside ``ur10e_mdp.py``; the UR10e task builds its arm from that file from
then on (delete it to go back to the defaults). The Franka is the reference and is not changed.

The Franka and the UR10e tasks cannot share one process, so each runs in its own worker simulation
(this same file with ``--worker``); the window talks to both. Worker output goes to
``logs/arm_tuning/<arm>.log``.

Usage (inside ``active_isaac``)::

    python scripts/ur10e_arm_tuning_gui.py             # control window + the two simulations
    python scripts/ur10e_arm_tuning_gui.py --viewer    # plus a Newton GL window for each arm
    python scripts/ur10e_arm_tuning_gui.py --selftest  # pulse all six axes once, print the table, exit
"""

import argparse
import json
import math
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

AXES = ("x", "y", "z", "rx", "ry", "rz")
UNITS = ("mm", "mm", "mm", "deg", "deg", "deg")
TASKS = {"franka": "Template-Sheet-Rl-Platform-TwistStudent-v0", "ur10e": "Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0"}
GROUPS = ("shoulder", "elbow", "wrist")
FIELDS = (  # arm control key, label, unit, spinbox step
    ("stiffness", "stiffness", "Nm/rad", 10.0),
    ("damping", "damping", "Nm s/rad", 2.0),
    ("armature", "armature", "kg m^2", 0.001),
    ("effort_limit", "effort limit", "Nm", 1.0),
    ("velocity_limit", "velocity limit (0 = asset's)", "rad/s", 0.1),
)
FRANKA_LIKE = {  # the Franka task's arm drive, for "Copy Franka values"; its arm has no velocity limit
    "stiffness": {"shoulder": 400.0, "elbow": 400.0, "wrist": 400.0},
    "damping": {"shoulder": 80.0, "elbow": 80.0, "wrist": 80.0},
    "armature": {"shoulder": 0.001, "elbow": 0.001, "wrist": 0.001},
    "effort_limit": {"shoulder": 87.0, "elbow": 87.0, "wrist": 12.0},
    "velocity_limit": {"shoulder": 100.0, "elbow": 100.0, "wrist": 100.0},
    "ik_lambda": 0.01,
}
PRE_S, POST_S = 0.2, 1.3
"""Seconds of zero command before and after the pulse."""

parser = argparse.ArgumentParser(description="Tune the UR10e arm's control against the Franka's with twist pulses.")
parser.add_argument("--worker", choices=TASKS, default=None, help=argparse.SUPPRESS)
parser.add_argument("--viewer", action="store_true", default=False, help="A Newton GL window for each arm.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ur_start", choices=("franka", "saved"), default="franka",
                    help="UR10e start: the Franka's tool pose (default, a like-for-like comparison) or its saved start pose.")
parser.add_argument("--selftest", action="store_true", default=False,
                    help="Pulse all six axes once, print the table, save the plots to logs/arm_tuning, exit.")
parser.add_argument("--amplitude", type=float, default=1.0, help="Starting pulse amplitude (action units).")
args_cli, rest_args = parser.parse_known_args()


def emit(kind, payload):
    print(f"@@{kind} " + json.dumps(payload), flush=True)


# =============================================================================================== worker
def run_worker():
    """One arm's simulation, driven by JSON commands on stdin; answers with ``@@KIND {json}`` lines."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

    import sheet_rl.tasks  # noqa: F401
    import warp as wp

    wp.config.enable_backward = False
    import torch

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config

    sys.argv = [sys.argv[0]] + rest_args
    wparser = argparse.ArgumentParser()
    add_launcher_args(wparser)
    add_frontend_args(wparser)
    launcher_args, remaining = setup_preset_cli(wparser, agent_library="fast_sac")
    sys.argv = [sys.argv[0]] + remaining
    arm, task = args_cli.worker, TASKS[args_cli.worker]

    def main(env_cfg, agent_cfg):
        from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

        env_cfg.scene.num_envs = 1
        env_cfg.seed = args_cli.seed
        env_cfg.scene.table.spawn.visible = True
        events = env_cfg.events
        # the same, unrandomized start every pulse: the Franka's tool pose, jaws open, a fixed layout
        events.reset_robot_arm_joints.params["position_range"] = (1.0, 1.0)
        if arm == "franka":
            events.reset_robot_gripper_joints.params["position_range"] = (0.0, 0.0)
        else:
            events.reset_robot_arm_joints.params["gripper_offset_range"] = (0.0, 0.0)
            if args_cli.ur_start == "franka":
                events.reset_robot_arm_joints.params["start_pose"] = "panda"
        events.reset_layout.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        events.reset_layout.params["yaw_range"] = (0.0, 0.0)
        events.set_platform_height.params["height_range"] = (0.08, 0.08)
        if args_cli.viewer:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
            env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(1.6, -1.4, 1.0), lookat=(0.3, 0.0, 0.25))]
        with launch_simulation(env_cfg, launcher_args):
            env = create_isaaclab_env(task, env_cfg, launcher_args, convert_marl_to_single_agent=False)
            inner = env.unwrapped
            env.reset()
            robot = inner.scene["robot"]
            actions = inner.action_manager
            arm_term = actions.get_term("arm_action")
            term = env_cfg.observations.action_history.deformable_sampled_points
            hand_id = robot.body_names.index(term.params["hand_body_name"])
            tool_offset = torch.tensor(term.params["tool_offset"], device=inner.device, dtype=torch.float)
            dt = inner.step_dt if inner._physics_handles_decimation else inner.physics_dt
            substeps = 1 if inner._physics_handles_decimation else inner.cfg.decimation
            arm_ids = robot.find_joints(arm_term.cfg.joint_names, preserve_order=True)[0]
            limits = torch.full((robot.num_joints,), float("inf"), device=inner.device)
            for actuator in robot.actuators.values():
                limits[actuator.joint_indices] = torch.as_tensor(actuator.velocity_limit_sim, device=inner.device)[0]
            asset_limits = limits.clone()
            control = None
            if arm == "ur10e":
                from sheet_rl.tasks.manager_based.sheet_twist.ur10e_mdp import ARM_GROUPS, load_arm_control, save_arm_control
                control = load_arm_control()
                group_ids = {g: robot.find_joints(expr)[0] for g, expr in ARM_GROUPS.items()}

                def apply(c):
                    for g, ids in group_ids.items():
                        robot.write_joint_stiffness_to_sim_index(stiffness=float(c["stiffness"][g]), joint_ids=ids)
                        robot.write_joint_damping_to_sim_index(damping=float(c["damping"][g]), joint_ids=ids)
                        robot.write_joint_armature_to_sim_index(armature=float(c["armature"][g]), joint_ids=ids)
                        robot.write_joint_effort_limit_to_sim_index(limits=float(c["effort_limit"][g]), joint_ids=ids)
                        vel = c["velocity_limit"][g]
                        limits[ids] = asset_limits[ids] if vel is None else float(vel)
                        robot.write_joint_velocity_limit_to_sim_index(limits=limits[ids][None].clone(), joint_ids=ids)
                    arm_term._ik_controller.cfg.ik_params["lambda_val"] = float(c["ik_lambda"])

                apply(control)
                reference = control
            else:
                drive = env_cfg.scene.robot.actuators["panda_arm"]
                reference = {"stiffness": drive.stiffness, "damping": drive.damping, "armature": drive.armature,
                             "effort_limit": drive.effort_limit_sim,
                             "velocity_limit": "none" if float(limits[arm_ids].max()) > 1e3 else limits[arm_ids].tolist(),
                             "ik_lambda": arm_term._ik_controller.cfg.ik_params["lambda_val"]}

            def tool_pose():
                pose = robot.data.body_link_pose_w.torch[0, hand_id]
                tool_w = pose[:3] + quat_apply(pose[3:7], tool_offset)
                p, q = subtract_frame_transforms(robot.data.root_pos_w.torch[:1], robot.data.root_quat_w.torch[:1],
                                                 tool_w[None], pose[None, 3:7])
                return p[0], q[0]

            def pulse(axis, amplitude, duration, scale, realtime):
                env.reset()
                if control is not None:
                    apply(control)
                arm_term._scale[:] = scale
                n_pre, n_pulse = round(PRE_S / inner.step_dt), round(duration / inner.step_dt)
                n_post = round(POST_S / inner.step_dt)
                ts, ps, qs, speeds = [], [], [], []
                t, onset = 0.0, None
                for k in range(n_pre + n_pulse + n_post):
                    tick = time.time()
                    a = torch.zeros(1, actions.total_action_dim, device=inner.device)
                    a[0, -1] = 1.0  # gripper held open
                    if n_pre <= k < n_pre + n_pulse:
                        a[0, axis] = amplitude
                    if k == n_pre:
                        onset = len(ts) - 1  # the last sample before the pulse acts
                    actions.process_action(a)
                    for _ in range(substeps):
                        actions.apply_action()
                        inner.scene.write_data_to_sim()
                        inner.sim.step(render=False)
                        inner.scene.update(dt=dt)
                        t += dt
                        p, q = tool_pose()
                        ts.append(t)
                        ps.append(p)
                        qs.append(q)
                        speeds.append(robot.data.joint_vel.torch[0, arm_ids].abs())
                    if args_cli.viewer:
                        inner.sim.render()
                        if realtime:
                            time.sleep(max(0.0, inner.step_dt - (time.time() - tick)))
                p = torch.stack(ps)
                q = torch.stack(qs)
                pos = (p - p[onset]) * 1000.0
                rot = torch.rad2deg(axis_angle_from_quat(quat_mul(q, quat_inv(q[onset].expand_as(q)))))
                speed = torch.stack(speeds)
                over = (speed > limits[arm_ids]).any(0)
                t0 = ts[onset]
                emit("RESULT", {
                    "arm": arm, "axis": axis, "amplitude": amplitude, "duration": duration, "scale": scale,
                    "t": [round(x - t0, 5) for x in ts], "pos": pos.tolist(), "rot": rot.tolist(),
                    "joint_speed_peak": speed.max(0).values.tolist(),
                    "joint_limits": limits[arm_ids].tolist(), "joints": [robot.joint_names[i] for i in arm_ids],
                    "over_limit": [robot.joint_names[arm_ids[i]] for i in over.nonzero().flatten().tolist()],
                })

            # Commands are read by a thread and handed over whole. Polling stdin with select() and then
            # readline() lost commands: the window sends the UR10e "params" and "pulse" back to back,
            # readline() pulled both into Python's buffer, select() then saw an empty pipe, and the
            # pulse waited unread until the next button press -- every other press did nothing.
            commands = queue.Queue()

            def read_commands():
                for text in sys.stdin:
                    commands.put(text)
                commands.put(None)  # the window closed its end

            threading.Thread(target=read_commands, daemon=True).start()
            emit("READY", {"arm": arm, "sample_hz": round(1 / dt), "control": reference,
                           "start_joints": robot.data.joint_pos.torch[0, arm_ids].tolist()})
            while True:
                try:
                    line = commands.get(timeout=0.03 if args_cli.viewer else None)
                except queue.Empty:
                    inner.sim.render()  # keep the viewer alive between commands
                    continue
                if line is None:
                    break
                msg = json.loads(line)
                if msg["cmd"] == "pulse":
                    pulse(msg["axis"], msg["amplitude"], msg["duration"], msg["scale"], msg.get("realtime", False))
                elif msg["cmd"] == "params" and control is not None:
                    control = msg["control"]
                    apply(control)
                    emit("PARAMS", {"arm": arm})
                elif msg["cmd"] == "save" and control is not None:
                    emit("SAVED", {"path": str(save_arm_control(msg["control"]))})
                elif msg["cmd"] == "quit":
                    break
            env.close()

    hydra_task_config(task, "fast_sac_cfg_entry_point")(main)()


# =============================================================================================== window
class Worker:
    """A worker simulation as seen from the window: JSON commands in, ``@@`` messages out."""

    def __init__(self, arm, log_dir):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", arm, "--seed", str(args_cli.seed),
               "--ur_start", args_cli.ur_start] + (["--viewer"] if args_cli.viewer else []) + rest_args
        env = dict(os.environ)
        env.setdefault("PYOPENGL_PLATFORM", "glx")
        self.arm = arm
        self.log_path = log_dir / f"{arm}.log"
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env)
        self.messages = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("@@"):
                kind, _, payload = line[2:].partition(" ")
                self.messages.put((kind, json.loads(payload)))
            else:
                self.log.write(line)
                self.log.flush()
        self.messages.put(("EXIT", {"code": self.proc.wait()}))

    def send(self, **msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def stop(self):
        self.send(cmd="quit")
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def response_metrics(result, axis):
    """Along ``axis``: (value at the end of the pulse, final value, time to 90 % of it, overshoot)."""
    t = result["t"]
    values = [row[axis % 3] for row in (result["pos"] if axis < 3 else result["rot"])]
    late = [v for ti, v in zip(t, values) if ti >= t[-1] - 0.2]
    final = sum(late) / len(late)
    at_end = min(zip(t, values), key=lambda tv: abs(tv[0] - result["duration"]))[1]
    t90 = next((ti for ti, v in zip(t, values) if ti >= 0 and abs(v) >= 0.9 * abs(final)), None) if abs(final) > 1e-6 else None
    sign = 1.0 if final >= 0 else -1.0
    overshoot = max(0.0, max(sign * v for ti, v in zip(t, values) if ti >= 0) - sign * final)
    return at_end, final, t90, overshoot


def curve(result, axis):
    return result["t"], [row[axis % 3] for row in (result["pos"] if axis < 3 else result["rot"])]


def difference(franka, ur, axis):
    """RMS difference of the UR10e's response from the Franka's along ``axis``, in % of the Franka's peak."""
    tf, yf = curve(franka, axis)
    tu, yu = curve(ur, axis)
    j, diffs = 0, []
    for ti, yi in zip(tf, yf):
        while j + 1 < len(tu) and tu[j + 1] <= ti:
            j += 1
        diffs.append((yi - yu[j]) ** 2)
    peak = max(abs(v) for v in yf) or 1.0
    return 100.0 * math.sqrt(sum(diffs) / len(diffs)) / peak


def coupling(result, axis):
    """Largest movement along the other axes: (translation mm, rotation deg)."""
    trans = max(abs(row[i]) for row in result["pos"] for i in range(3) if i != axis)
    rot = max(abs(row[i]) for row in result["rot"] for i in range(3) if i + 3 != axis)
    return trans, rot


def run_gui():
    import tkinter as tk
    from tkinter import ttk

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    log_dir = pathlib.Path(__file__).resolve().parent.parent / "logs" / "arm_tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = {arm: Worker(arm, log_dir) for arm in TASKS}

    root = tk.Tk()
    root.title("UR10e arm control tuning -- against the Franka")
    left, right = tk.Frame(root), tk.Frame(root)
    left.pack(side="left", fill="y", padx=8, pady=6)
    right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    status = tk.StringVar(value="starting the Franka and UR10e simulations (about a minute)...")
    tk.Label(left, textvariable=status, justify="left", anchor="w", wraplength=430, fg="#204080").pack(fill="x")

    # ---- input
    box = tk.LabelFrame(left, text="Twist pulse (same input to both arms)")
    box.pack(fill="x", pady=6)
    axis_var = tk.IntVar(value=0)
    row = tk.Frame(box)
    row.pack(anchor="w")
    for i, name in enumerate(AXES):
        tk.Radiobutton(row, text=name, variable=axis_var, value=i).pack(side="left")
    amp_var, dur_var, scale_var = tk.DoubleVar(value=args_cli.amplitude), tk.DoubleVar(value=0.5), tk.DoubleVar(value=0.05)
    for label, var, lo, hi, step in (("amplitude (action units)", amp_var, -3.0, 3.0, 0.1),
                                     ("duration [s]", dur_var, 0.05, 1.5, 0.05),
                                     ("IK scale", scale_var, 0.01, 0.15, 0.005)):
        r = tk.Frame(box)
        r.pack(fill="x")
        tk.Label(r, text=label, width=22, anchor="w").pack(side="left")
        tk.Scale(r, from_=lo, to=hi, resolution=step, orient="horizontal", length=170, variable=var).pack(side="left")
    command_text = tk.StringVar()
    tk.Label(box, textvariable=command_text, anchor="w", font=("Courier", 9)).pack(fill="x")
    realtime_var = tk.BooleanVar(value=args_cli.viewer)
    if args_cli.viewer:
        tk.Checkbutton(box, text="play in real time in the viewers", variable=realtime_var).pack(anchor="w")
    buttons = tk.Frame(box)
    buttons.pack(anchor="w", pady=4)
    pulse_button = tk.Button(buttons, text="Pulse this axis", state="disabled")
    pulse_button.pack(side="left", padx=2)
    all_button = tk.Button(buttons, text="Pulse all 6 axes", state="disabled")
    all_button.pack(side="left", padx=2)

    # ---- UR10e control
    box = tk.LabelFrame(left, text="UR10e arm control (applied before every pulse)")
    box.pack(fill="x", pady=6)
    grid = tk.Frame(box)
    grid.pack(anchor="w")
    for c, g in enumerate(GROUPS):
        tk.Label(grid, text=g, font=("Helvetica", 9, "bold")).grid(row=0, column=c + 1)
    cells = {}
    for r, (key, label, unit, step) in enumerate(FIELDS, start=1):
        tk.Label(grid, text=f"{label} [{unit}]", anchor="w").grid(row=r, column=0, sticky="w")
        for c, g in enumerate(GROUPS):
            var = tk.StringVar()
            ttk.Spinbox(grid, from_=0.0, to=10000.0, increment=step, width=8, textvariable=var).grid(row=r, column=c + 1, padx=1)
            cells[(key, g)] = var
    lam_var = tk.StringVar()
    r = tk.Frame(box)
    r.pack(anchor="w", pady=2)
    tk.Label(r, text="IK damping lambda").pack(side="left")
    ttk.Spinbox(r, from_=0.0, to=1.0, increment=0.005, width=8, textvariable=lam_var).pack(side="left", padx=4)
    franka_text = tk.StringVar(value="Franka (reference): waiting for its simulation")
    tk.Label(box, textvariable=franka_text, justify="left", anchor="w", wraplength=430, font=("Courier", 8)).pack(fill="x", pady=2)
    files = tk.Frame(box)
    files.pack(anchor="w", pady=4)

    def get_control():
        control = {key: {} for key, *_ in FIELDS}
        for (key, g), var in cells.items():
            try:
                value = float(var.get())
            except ValueError:
                raise ValueError(f"{key} / {g}: '{var.get()}' is not a number")
            control[key][g] = None if key == "velocity_limit" and value <= 0 else value
        control["ik_lambda"] = float(lam_var.get())
        return control

    def set_control(control):
        for (key, g), var in cells.items():
            value = control[key][g]
            var.set("0" if value is None else f"{value:g}")
        lam_var.set(f"{control['ik_lambda']:g}")

    state = {"saved": None, "queue": [], "results": {}, "mode": "single", "busy": False, "ready": set(), "plotted": None}

    def on_save():
        try:
            workers["ur10e"].send(cmd="save", control=get_control())
        except ValueError as error:
            status.set(f"not saved: {error}")

    tk.Button(files, text="Save UR10e control", command=on_save).pack(side="left", padx=2)
    tk.Button(files, text="Copy Franka values", command=lambda: set_control(FRANKA_LIKE)).pack(side="left", padx=2)
    tk.Button(files, text="Reload saved", command=lambda: state["saved"] and set_control(state["saved"])).pack(side="left", padx=2)

    # ---- plots and table
    figure = Figure(figsize=(10.5, 5.4), dpi=90)
    axes = [figure.add_subplot(2, 3, i + 1) for i in range(6)]
    figure.tight_layout(pad=2.0)
    canvas = FigureCanvasTkAgg(figure, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    table = tk.Text(right, height=13, font=("Courier", 9))
    table.pack(fill="x")

    def update_command_text(*_):
        scale, amp = scale_var.get(), amp_var.get()
        unit, per_step = ("mm", 1000 * amp * scale) if axis_var.get() < 3 else ("deg", math.degrees(amp * scale))
        command_text.set(f"commands {per_step:+.1f} {unit} per 20 ms control step for {dur_var.get():.2f} s")

    for var in (axis_var, amp_var, dur_var, scale_var):
        var.trace_add("write", update_command_text)
    update_command_text()

    def start(axes_to_run, mode):
        try:
            control = get_control()
        except ValueError as error:
            status.set(str(error))
            return
        workers["ur10e"].send(cmd="params", control=control)
        state.update(queue=list(axes_to_run), results={}, mode=mode, busy=True)
        pulse_button.config(state="disabled")
        all_button.config(state="disabled")
        next_pulse()

    def next_pulse():
        axis = state["queue"][0]
        state["pending"] = set(TASKS)
        for w in workers.values():
            w.send(cmd="pulse", axis=axis, amplitude=amp_var.get(), duration=dur_var.get(), scale=scale_var.get(),
                   realtime=bool(realtime_var.get()))
        status.set(f"pulsing {AXES[axis]} on both arms...")

    pulse_button.config(command=lambda: start([axis_var.get()], "single"))
    all_button.config(command=lambda: start(range(6), "all"))

    def draw():
        res = state["results"]
        for i, ax in enumerate(axes):
            ax.clear()
            axis = i if state["mode"] == "all" else state["plotted"]
            pair = res.get(axis)
            title = f"{AXES[i]} [{UNITS[i]}]" + ("" if state["mode"] == "all" else ("  (pulsed)" if i == axis else ""))
            if state["mode"] == "all":
                title = f"{AXES[i]} [{UNITS[i]}], pulse on {AXES[i]}"
            ax.set_title(title, fontsize=9)
            ax.axvspan(0, dur_var.get() if not pair else pair["franka"]["duration"], color="#eeeeee")
            ax.axhline(0, color="#bbbbbb", lw=0.6)
            if pair:
                for arm, color in (("franka", "#2060c0"), ("ur10e", "#e07020")):
                    t, _ = curve(pair[arm], i)
                    values = [row[i % 3] for row in (pair[arm]["pos"] if i < 3 else pair[arm]["rot"])]
                    ax.plot(t, values, color=color, lw=1.6, label="Franka" if arm == "franka" else "UR10e")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc="best")
            if i >= 3:
                ax.set_xlabel("time from pulse onset [s]", fontsize=7)
        canvas.draw_idle()
        lines = [f"{'axis':4s} | {'arm':6s} | {'at pulse end':>12s} | {'final':>8s} | {'t to 90%':>8s} | {'overshoot':>9s} | "
                 f"{'UR vs Franka':>12s} | {'off-axis mm / deg':>17s} | peak joint speed"]
        for axis in sorted(res):
            pair = res[axis]
            for arm in ("franka", "ur10e"):
                r = pair[arm]
                at_end, final, t90, over = response_metrics(r, axis)
                trans, rot = coupling(r, axis)
                diff = f"{difference(pair['franka'], pair['ur10e'], axis):10.0f} %" if arm == "ur10e" else ""
                peak = max(r["joint_speed_peak"])
                flag = f"  OVER limit: {', '.join(r['over_limit'])}" if r["over_limit"] else ""
                lines.append(f"{AXES[axis]:4s} | {arm:6s} | {at_end:9.1f} {UNITS[axis]:3s}| {final:8.1f} | "
                             f"{'-' if t90 is None else f'{t90:.3f} s':>8s} | {over:9.2f} | {diff:>12s} | "
                             f"{trans:7.1f} / {rot:6.2f}  | {peak:5.2f} rad/s{flag}")
        table.delete("1.0", "end")
        table.insert("end", "\n".join(lines))
        return lines

    def poll():
        for arm, w in workers.items():
            while not w.messages.empty():
                kind, payload = w.messages.get()
                if kind == "READY":
                    state["ready"].add(arm)
                    if arm == "ur10e":
                        state["saved"] = payload["control"]
                        set_control(payload["control"])
                    else:
                        c = payload["control"]
                        franka_text.set(f"Franka (reference): stiffness {c['stiffness']}, damping {c['damping']}, "
                                        f"armature {c['armature']}, effort {c['effort_limit']}, velocity limit "
                                        f"{c['velocity_limit']}, IK lambda {c['ik_lambda']}")
                    if state["ready"] == set(TASKS):
                        status.set(f"ready -- both arms start from the same tool pose; sampled at {payload['sample_hz']} Hz")
                        pulse_button.config(state="normal")
                        all_button.config(state="normal")
                        if args_cli.selftest:
                            start(range(6), "all")
                elif kind == "RESULT":
                    axis = payload["axis"]
                    state["results"].setdefault(axis, {})[arm] = payload
                    state["pending"].discard(arm)
                    if not state["pending"]:
                        state["queue"].pop(0)
                        state["plotted"] = axis
                        lines = draw()
                        if state["queue"]:
                            next_pulse()
                        else:
                            state["busy"] = False
                            pulse_button.config(state="normal")
                            all_button.config(state="normal")
                            status.set("done -- orange is the UR10e, blue the Franka; edit the UR10e control and pulse again")
                            if args_cli.selftest:
                                print("\n".join(lines), flush=True)
                                stem = log_dir / f"selftest_amp{amp_var.get():g}"
                                figure.savefig(f"{stem}.png", dpi=110)
                                pathlib.Path(f"{stem}.txt").write_text("\n".join(lines) + "\n")
                                print(f"saved {stem}.png and .txt", flush=True)
                                close()
                                return
                elif kind == "SAVED":
                    state["saved"] = get_control()
                    status.set(f"saved to {payload['path']} -- the UR10e task uses it from its next start")
                elif kind == "EXIT":
                    status.set(f"the {arm} simulation stopped (exit {payload['code']}); see {w.log_path}")
                    if args_cli.selftest:
                        close()
                        return
        root.after(50, poll)

    def close():
        for w in workers.values():
            w.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(50, poll)
    root.mainloop()


if __name__ == "__main__":
    if args_cli.worker:
        run_worker()
    else:
        run_gui()
