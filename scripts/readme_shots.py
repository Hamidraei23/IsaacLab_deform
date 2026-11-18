"""Screenshots and observation/action tables of a task, for the README.

Runs one environment with a headless Newton GL camera, drives it with an exported TorchScript policy (or zero
actions), saves PNG frames at the requested steps and writes the environment's own observation- and action-manager
tables::

    python scripts/readme_shots.py --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0 \\
        --policy logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/exported_real/policy.pt \\
        --obs_group action_history --shots 0 30 60 90 --camera 1.3 -1.0 0.9 0.35 0.0 0.15 --name ur16e \\
        env.actions.arm_action.scale=0.03

Output: ``<out>/<name>_step<NNN>.png`` and ``<out>/<name>_spaces.txt``.
"""

import argparse
import os
import sys

import sheet_rl.tasks  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--policy", type=str, default=None, help="Exported TorchScript policy (none: zero actions).")
parser.add_argument("--obs_group", type=str, default="policy", help="Observation group the policy reads.")
parser.add_argument("--shots", type=int, nargs="+", default=[0], help="Env steps at which to save a frame.")
parser.add_argument("--camera", type=float, nargs=6, default=(1.4, -1.1, 0.95, 0.4, 0.0, 0.15),
                    metavar=("EX", "EY", "EZ", "LX", "LY", "LZ"), help="Eye and look-at point in world coordinates.")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=1000)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--out", type=str, default="media/readme")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stochastic", action="store_true", help="Unused for TorchScript exports (deterministic).")
parser.add_argument("--every", type=int, default=0, help="Also a frame every N steps (with --until).")
parser.add_argument("--until", type=int, default=0, help="Last step for --every.")
parser.add_argument("--sensor", type=str, default=None, help="A camera sensor in the scene whose rgb is saved too.")
parser.add_argument("--light_sky", action="store_true", help="A pale sky instead of the default blue.")
parser.add_argument("--warmup_frames", type=int, default=3, help="Extra renders before each shot (GL settle).")
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, agent_library="rsl_rl")
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
    from PIL import Image

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    torch.manual_seed(args_cli.seed)
    env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
        eye=tuple(args_cli.camera[:3]), lookat=tuple(args_cli.camera[3:]), headless=True,
        window_width=args_cli.width, window_height=args_cli.height,
        **(dict(sky_upper_color=(0.80, 0.86, 0.93), sky_lower_color=(0.96, 0.96, 0.97)) if args_cli.light_sky else {}))]
    os.makedirs(args_cli.out, exist_ok=True)
    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        inner = env.unwrapped
        device = inner.device
        om, am = inner.observation_manager, inner.action_manager
        with open(os.path.join(args_cli.out, f"{args_cli.name}_spaces.txt"), "w") as f:
            f.write(f"task: {args_cli.task}\n")
            f.write(f"observation groups: { {k: tuple(v) for k, v in om.group_obs_dim.items()} }\n")
            f.write(f"action dim: {am.total_action_dim}\n\n{om}\n\n{am}\n")
            if hasattr(inner, "command_manager"):
                f.write(f"\n{inner.command_manager}\n")
            if hasattr(inner, "termination_manager"):
                f.write(f"\n{inner.termination_manager}\n")
            if hasattr(inner, "reward_manager"):
                f.write(f"\n{inner.reward_manager}\n")

        policy = torch.jit.load(args_cli.policy, map_location=device).eval() if args_cli.policy else None
        viz = inner.sim.visualizers[0]
        obs, _ = inner.reset()
        dump_positions(inner, os.path.join(args_cli.out, f"{args_cli.name}_spaces.txt"))
        shots = set(args_cli.shots)
        if args_cli.every:
            shots |= set(range(0, args_cli.until + 1, args_cli.every))
        last = max(shots)
        for step in range(last + 1):
            if step in shots:
                for _ in range(args_cli.warmup_frames):
                    frame = viz.render_rgb_array()
                frame = viz.render_rgb_array()
                if frame is not None:
                    image = np.asarray(frame)[..., :3].astype(np.uint8)
                    Image.fromarray(image).save(os.path.join(args_cli.out, f"{args_cli.name}_step{step:03d}.png"))
                    print(f"[readme_shots] saved {args_cli.name}_step{step:03d}.png", flush=True)
                else:
                    print("[readme_shots] the visualizer returned no frame", flush=True)
                if args_cli.sensor:
                    rgb = inner.scene[args_cli.sensor].data.output["rgb"]
                    rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy().astype(np.uint8)
                    Image.fromarray(rgb).save(os.path.join(args_cli.out, f"{args_cli.name}_{args_cli.sensor}_step{step:03d}.png"))
            if step == last:
                break
            with torch.inference_mode():
                if policy is None:
                    actions = torch.zeros(inner.num_envs, am.total_action_dim, device=device)
                else:
                    actions = policy(obs[args_cli.obs_group])
            obs, _, terminated, truncated, _ = inner.step(actions)
            if bool((terminated | truncated).any()):
                print(f"[readme_shots] episode ended at step {step + 1}", flush=True)
        env.close()


def dump_positions(inner, path):
    with open(path, "a") as f:
        # where things are, to aim the camera
        f.write(f"\nenv origin: {inner.scene.env_origins[0].tolist()}\n")
        for name in inner.scene.keys():
            try:
                obj = inner.scene[name]
            except Exception:   # noqa: BLE001
                continue
            data = getattr(obj, "data", None)
            for attr in ("root_pos_w", "nodal_pos_w", "body_pos_w"):
                value = getattr(data, attr, None) if data is not None else None
                if value is None:
                    continue
                value = value.torch if hasattr(value, "torch") else value
                try:
                    value = torch.as_tensor(value)[0]
                    if value.ndim > 1:
                        f.write(f"{name}.{attr}: mean {value.mean(0).tolist()} min {value.min(0).values.tolist()} "
                                f"max {value.max(0).values.tolist()}\n")
                    else:
                        f.write(f"{name}.{attr}: {value.tolist()}\n")
                except Exception as exc:   # noqa: BLE001
                    f.write(f"{name}.{attr}: ({exc})\n")
                break


if __name__ == "__main__":
    main()
