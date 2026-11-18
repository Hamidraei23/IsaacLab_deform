"""Pick one episode out of a ``play.py --record`` CSV, look at it and save it as its own CSV.

The window lists every episode in the recording (length in steps and seconds, peak tool speed, how it ended).
Selecting one plots it:

- tool-point position x, y, z [m] (base_link axes, ``tool_*`` columns);
- tool-point speed [m/s];
- arm joint speeds as a fraction of the arm's full velocity limits, with the recording's joint-speed limit
  (``limit_scale`` from its command, if it was set) as a dashed line;
- the policy's actions a0..a5 (arm) and a6 (gripper).

**Save episode** writes it with ``trace_episode.save_episode``: every column, ``step`` and ``t`` from the
episode's start, plus a ``.meta.json``. **Open...** loads another recording.

Usage (plain Python, no simulator)::

    python scripts/trace_episode_gui.py logs/play/ur16e_velterm45.csv
"""

import argparse
import os
import re

from trace_episode import ending, load_recording, save_episode, tool_speeds

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_VELOCITY_LIMITS = (2.0944, 2.0944, 3.1416, 3.1416, 3.1416, 3.1416)
"""rad/s: 120/120/180/180/180/180 deg/s, the same on the UR10e and UR16e assets."""

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", nargs="?", default="logs/play/ur16e_velterm45.csv", help="A recording written by play.py --record.")
args = parser.parse_args()


def limit_scale(meta):
    """The joint-speed termination's ``limit_scale`` from the recording's command, or None if it was not set."""
    for arg in meta.get("argv", []):
        match = re.fullmatch(r"env\.terminations\.joint_vel_out_of_limit\.params\.limit_scale=([\d.]+)", arg)
        if match:
            return float(match.group(1))
    return None


def main():
    import tkinter as tk
    from tkinter import filedialog, ttk

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Recorded episodes")
    state = {"path": None, "columns": None, "episodes": {}, "meta": {}}

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)
    file_text = tk.StringVar()
    tk.Label(top, textvariable=file_text, anchor="w", justify="left", font=("Courier", 9)).pack(side="left", fill="x", expand=True)
    tk.Button(top, text="Open...", command=lambda: choose_file()).pack(side="right", padx=2)
    save_button = tk.Button(top, text="Save episode", state="disabled", command=lambda: save())
    save_button.pack(side="right", padx=2)

    table = ttk.Treeview(root, columns=("episode", "steps", "seconds", "peak", "ended"), show="headings", height=7,
                         selectmode="browse")
    for key, label, width in (("episode", "episode", 70), ("steps", "steps", 70), ("seconds", "seconds", 80),
                              ("peak", "peak tool m/s", 110), ("ended", "ended by", 260)):
        table.heading(key, text=label)
        table.column(key, width=width, anchor="w" if key == "ended" else "e")
    table.pack(fill="x", padx=8)
    status = tk.StringVar(value="select an episode")
    tk.Label(root, textvariable=status, anchor="w", fg="#204080").pack(fill="x", padx=8, pady=2)

    figure = Figure(figsize=(11, 6.5), dpi=90)
    axes = [figure.add_subplot(2, 2, i + 1) for i in range(4)]
    canvas = FigureCanvasTkAgg(figure, master=root)
    NavigationToolbar2Tk(canvas, root).update()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def load(path):
        columns, episodes, meta = load_recording(path)
        state.update(path=path, columns=columns, episodes=episodes, meta=meta)
        dt = float(meta.get("step_dt", 0.02))
        checkpoint = next((a for p, a in zip(meta.get("argv", []), meta.get("argv", [])[1:]) if p == "--checkpoint"), "?")
        scale = meta.get("action_scale", ["?"])[0]
        file_text.set(f"{path}\n{len(episodes)} episodes | checkpoint {os.path.basename(checkpoint)} | "
                      f"scale {scale if isinstance(scale, str) else f'{scale:.3g}'} | speed limit "
                      f"{'off / full' if limit_scale(meta) is None else f'{limit_scale(meta):g} of full'}")
        table.delete(*table.get_children())
        for number, rows in episodes.items():
            peak = max(tool_speeds(rows, dt), default=0.0)
            table.insert("", "end", iid=str(number),
                         values=(number, len(rows), f"{len(rows) * dt:.2f}", f"{peak:.2f}", ending(rows)))
        for ax in axes:
            ax.clear()
        canvas.draw_idle()
        save_button.config(state="disabled")
        status.set("select an episode")
        if episodes:
            table.selection_set(str(next(iter(episodes))))

    def selected():
        chosen = table.selection()
        return int(chosen[0]) if chosen else None

    def draw(_event=None):
        number = selected()
        if number is None:
            return
        rows, meta = state["episodes"][number], state["meta"]
        dt = float(meta.get("step_dt", 0.02))
        t = [i * dt for i in range(len(rows))]
        col = lambda key: [float(r[key]) for r in rows]
        for ax in axes:
            ax.clear()
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        ax = axes[0]
        for key, color in (("tool_x", "#d04040"), ("tool_y", "#40a040"), ("tool_z", "#4060d0")):
            ax.plot(t, col(key), color=color, label=key[-1])
        ax.set_title("tool position [m], base_link", fontsize=9)
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(t[1:], tool_speeds(rows, dt), color="#404040")
        ax.set_title("tool speed [m/s]", fontsize=9)
        ax = axes[2]
        for joint, limit in zip(ARM_JOINTS, ARM_VELOCITY_LIMITS):
            ax.plot(t, [abs(v) / limit for v in col(f"qd_{joint}")], label=joint.replace("_joint", ""))
        scale = limit_scale(meta)
        ax.axhline(1.0 if scale is None else scale, color="#c00000", ls="--", lw=1.2,
                   label="full limit" if scale is None else f"limit x{scale:g}")
        ax.set_title("|joint speed| / full limit", fontsize=9)
        ax.set_xlabel("time from episode start [s]", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax = axes[3]
        for i in range(6):
            ax.plot(t, col(f"a{i}"), lw=1.0, label=f"a{i}")
        ax.plot(t, col("a6"), color="black", ls="--", lw=1.2, label="a6 gripper")
        ax.set_title("policy actions (arm a0-a5: x y z rx ry rz; a6 gripper)", fontsize=9)
        ax.set_xlabel("time from episode start [s]", fontsize=8)
        ax.legend(fontsize=7, ncol=4)
        figure.suptitle(f"episode {number}: {len(rows)} steps, {len(rows) * dt:.2f} s, {ending(rows)}", fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        canvas.draw_idle()
        save_button.config(state="normal", text=f"Save episode {number}")
        status.set(f"episode {number} selected")

    def save():
        number = selected()
        if number is None:
            return
        path = state["path"]
        default = f"{os.path.splitext(path)[0]}_ep{number}.csv"
        out = filedialog.asksaveasfilename(initialdir=os.path.dirname(default) or ".", initialfile=os.path.basename(default),
                                           defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if out:
            out = save_episode(path, state["columns"], state["episodes"], state["meta"], number, out)
            status.set(f"saved episode {number} -> {out} (and its .meta.json)")

    def choose_file():
        path = filedialog.askopenfilename(initialdir=os.path.dirname(state["path"] or "logs/play") or ".",
                                          filetypes=[("Recordings", "*.csv")])
        if path:
            load(path)

    table.bind("<<TreeviewSelect>>", draw)
    load(args.csv)
    root.mainloop()


if __name__ == "__main__":
    main()
