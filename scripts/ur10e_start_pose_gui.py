"""Pose the UR10e by hand in its task scene and save the pose as the task's start pose.

A slider window drives the six arm joints of the UR10e task's robot (the scene -- table, slot,
sheet, platform -- is the task's own), shown in the viewer chosen with ``--viz``. The window also
reports where that puts the tool, how far it is from the Franka's start, how low the arm links
reach and the self-collision clearance.

Buttons:

- **Save as start pose** writes ``ur10e_start_pose.json`` beside ``ur10e_mdp.py``. From then on the
  UR10e task starts from that posture (each reset still adds the Franka's random variation of its
  start; set ``env.events.reset_robot_arm_joints.params.position_range=[1.0,1.0]`` for no
  variation). It takes effect the next time the task is created. Delete the file to go back to
  starting where the Franka does.
- **Snap tool to the Franka start** solves the joints that put the tool at the Franka's start
  pose, starting from the current sliders, so you can pick the arm's posture (elbow up or down,
  wrist flipped or not) while keeping the observations identical to the Franka's.
- **Task start** / **Saved pose** load those postures into the sliders; **Reset scene** resets the
  sheet and layout.

Usage (inside ``active_isaac``); the viewer defaults to Newton GL, light and fast (``--viz kit`` for Isaac Sim)::

    python scripts/ur10e_start_pose_gui.py
"""

import argparse
import math
import os
import sys

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

parser = argparse.ArgumentParser(description="Pose the UR10e by hand and save its start pose.")
parser.add_argument("--task", type=str, default="Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0")
parser.add_argument("--agent", type=str, default="fast_sac_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="fast_sac")
sys.argv = [sys.argv[0]] + remaining_args

LINKS = ("upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
"""Arm bodies whose height is reported (their origins: the shoulder, elbow and wrist joints)."""


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    import tkinter as tk

    from isaaclab.utils.math import compute_pose_error
    from sheet_rl.tasks.manager_based.sheet_twist import ur10e_gripper
    from sheet_rl.tasks.manager_based.sheet_twist.self_collision import self_collision_clearance
    from sheet_rl.tasks.manager_based.sheet_twist.ur10e_mdp import (
        ARM_JOINTS, GRIPPER_SIGNS, START_POSE_FILE, adapter, load_start_pose, save_start_pose,
    )

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    # the table collider is hidden (Isaac Sim draws a separate success-coloured copy); Newton GL draws
    # only visible prims, so show it here to see where the base and arm are relative to the table
    env_cfg.scene.table.spawn.visible = True
    if not getattr(args_cli, "visualizer", None):  # Newton GL, framing the arm and table (--viz kit for Isaac Sim)
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(1.5, -1.3, 0.9), lookat=(0.3, 0.0, 0.15))]
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        env.reset()
        robot = inner.scene["robot"]
        state = adapter(inner)
        arm_ids, finger_ids = state.arm_ids, state.finger_ids
        ids = torch.tensor([0], device=inner.device)
        limits = robot.data.soft_joint_pos_limits.torch[0, arm_ids].cpu()
        link_ids = [robot.body_names.index(n) for n in LINKS]
        open_fingers = ur10e_gripper.OPEN_ANGLE * torch.tensor(list(GRIPPER_SIGNS.values()), device=inner.device)
        base = robot.data.root_pos_w.torch[:1] - inner.scene.env_origins[:1]
        franka_pos, franka_rot, _ = state.panda.forward(state.home[:1])
        franka_pos = franka_pos - base  # the Franka's (unrandomized) start tool pose, in this arm's root frame
        task_start = robot.data.joint_pos.torch[0, arm_ids].tolist()  # where the task's reset put the arm

        root = tk.Tk()
        root.title("UR10e start pose")
        sliders = []
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = (math.degrees(v) for v in limits[i].tolist())
            var = tk.DoubleVar(value=math.degrees(task_start[i]))
            row = tk.Frame(root)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=f"{name:20s}", font=("Courier", 10), width=20, anchor="w").pack(side="left")
            tk.Scale(row, from_=max(lo, -360.0), to=min(hi, 360.0), resolution=0.5, orient="horizontal",
                     length=520, variable=var).pack(side="left")
            entry = tk.Entry(row, width=8, textvariable=var)
            entry.pack(side="left", padx=4)
            tk.Label(row, text="deg").pack(side="left")
            sliders.append(var)
        info = tk.StringVar()
        tk.Label(root, textvariable=info, font=("Courier", 10), justify="left", anchor="w").pack(fill="x", padx=8, pady=6)
        status = tk.StringVar(value=f"start pose file: {START_POSE_FILE}"
                              + (" (exists)" if START_POSE_FILE.exists() else " (none saved: the task starts where the Franka does)"))
        tk.Label(root, textvariable=status, wraplength=780, justify="left", fg="#204080").pack(fill="x", padx=8)

        def current():
            return torch.tensor([math.radians(v.get()) for v in sliders], device=inner.device)

        def set_sliders(q):
            for var, value in zip(sliders, q):
                var.set(round(math.degrees(float(value)) * 2) / 2)

        def on_save():
            q = current()
            pos, rot, _ = state.ur.forward(q[None])
            path = save_start_pose(q.tolist(), tool_pos_in_robot_frame=[round(v, 4) for v in pos[0].tolist()],
                                   tool_quat_xyzw=[round(v, 5) for v in rot[0].tolist()])
            status.set(f"saved to {path}. The UR10e task starts from this posture the next time it is created "
                       "(plus the usual random variation per reset). Delete the file to go back.")

        def on_snap():
            q = state.ur.solve(franka_pos, franka_rot, current()[None], iterations=200)
            p, r, _ = state.ur.forward(q)
            dp, dr = compute_pose_error(p, r, franka_pos, franka_rot, rot_error_type="axis_angle")
            if dp.norm() > 2e-3 or dr.norm() > 2e-2:
                status.set(f"no solution near this posture (left {dp.norm().item() * 1000:.1f} mm away); move the sliders closer and retry")
                return
            set_sliders(q[0].tolist())
            status.set("tool snapped to the Franka start pose in this posture branch")

        def on_task_start():
            set_sliders(task_start)
            status.set("loaded where the task's reset put the arm")

        def on_saved():
            saved = load_start_pose()
            if saved is None:
                status.set("no saved start pose yet")
            else:
                set_sliders(saved)
                status.set(f"loaded the saved pose from {START_POSE_FILE}")

        reset_requested = [False]
        buttons = tk.Frame(root)
        buttons.pack(pady=8)
        for text, command in (("Save as start pose", on_save), ("Snap tool to the Franka start", on_snap),
                              ("Task start", on_task_start), ("Saved pose", on_saved),
                              ("Reset scene", lambda: reset_requested.__setitem__(0, True))):
            tk.Button(buttons, text=text, command=command).pack(side="left", padx=4)
        closed = [False]
        root.protocol("WM_DELETE_WINDOW", lambda: closed.__setitem__(0, True))

        viewers = [type(viz).__name__ for viz in getattr(inner.sim, "_visualizers", [])]
        print(f"[pose-gui] ready: move the sliders; Save writes the start pose; viewer {', '.join(viewers) or 'none'}", flush=True)
        while not closed[0]:
            if reset_requested[0]:
                env.reset()
                reset_requested[0] = False
            q = current()
            positions = robot.data.default_joint_pos.torch[:1].clone()
            positions[0, arm_ids] = q
            positions[0, finger_ids] = open_fingers
            robot.write_joint_state_to_sim_index(position=positions, velocity=torch.zeros_like(positions), env_ids=ids)
            robot.set_joint_position_target_index(target=positions, env_ids=ids)
            inner.scene.write_data_to_sim()
            inner.sim.step(render=False)
            inner.sim.render()
            inner.scene.update(dt=inner.physics_dt)

            pos, rot, _ = state.ur.forward(q[None])
            dp, dr = compute_pose_error(pos, rot, franka_pos, franka_rot, rot_error_type="axis_angle")
            heights = robot.data.body_link_pose_w.torch[0, link_ids, 2] - robot.data.root_pos_w.torch[0, 2]
            clearance = self_collision_clearance(inner)[0].item()
            info.set(
                f"tool (robot frame)      x {pos[0, 0]:+.3f}  y {pos[0, 1]:+.3f}  z {pos[0, 2]:+.3f} m\n"
                f"from the Franka start   {dp.norm().item() * 1000:6.1f} mm, {math.degrees(dr.norm().item()):5.1f} deg"
                f"{'   (same start as the Franka)' if dp.norm() < 2e-3 and dr.norm() < 2e-2 else ''}\n"
                f"link heights above table: " + "  ".join(f"{n.split('_link')[0]} {h:+.2f}" for n, h in zip(LINKS, heights.tolist())) + " m\n"
                f"self-collision clearance {clearance * 100:+.1f} cm{'   COLLIDING' if clearance < 0.01 else ''}"
                f"{'   ARM BELOW TABLE' if heights.min() < 0 else ''}"
            )
            root.update()
        root.destroy()
        env.close()


if __name__ == "__main__":
    main()
