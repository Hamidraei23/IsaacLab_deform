"""Export a twist-student checkpoint for the real UR16e: the deterministic policy as ONNX and TorchScript, plus a
JSON spec of exactly what it reads and writes.

The policy is ``FastSACActorModel``'s deterministic action (``_DeterministicFastSACActor``): the frozen observation
normaliser, the network, ``tanh`` and the action scale, all inside the exported graph, so the runtime feeds the raw
49 observation values and gets the 7 actions back. Input ``obs`` (batch, 49) float32, output ``actions`` (batch, 7).

The spec (``policy_spec.json``) records the observation layout and frames as measured against the simulator
(``hrii_sheet_mocap.policy_obs`` rebuilds the observation from real measurements and is tested to reproduce the
simulator's to float precision), the control rate, the action meaning and scale, the gripper filter, and the
finger-angle-to-pad-gap curve of the simulated 2F-85.

Usage (Isaac venv, no simulator needed)::

    python scripts/export_real_policy.py \\
        --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt \\
        --arm_action_scale 0.03 [--check scratch/obs_capture.npz]

writes ``exported_real/`` next to the checkpoint. ``--check`` compares the exported ONNX and TorchScript policies with
the actions the simulator's actor took on a recorded episode (``obs_capture.npz``: ``obs`` and ``action`` arrays).
"""

import argparse
import datetime
import hashlib
import json
import os

import numpy as np
import torch
from tensordict import TensorDict

OBS_DIM, ACTION_DIM = 49, 7
TOOL_OFFSET = (0.0, 0.0, 0.145)
LANDMARK_NODES = [0, 4, 8, 36, 40, 44, 72, 76, 80]


def load_actor(checkpoint):
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from sheet_rl.algorithms.fast_sac.rsl_rl_actor import FastSACActorModel, _DeterministicFastSACActor
    from sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_ppo_cfg import UR10eFineTuneRunnerCfg

    runner_cfg = handle_deprecated_rsl_rl_cfg(UR10eFineTuneRunnerCfg(), version("rsl-rl-lib"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    kwargs = runner_cfg.actor.to_dict()
    kwargs.pop("class_name")
    if "learned_log_std" in state:
        kwargs["init_std"] = float(state["learned_log_std"].exp().mean())
    obs = TensorDict({group: torch.zeros(1, OBS_DIM) for group in runner_cfg.obs_groups["actor"]}, batch_size=[1])
    model = FastSACActorModel(obs, runner_cfg.obs_groups, "actor", ACTION_DIM, **kwargs)
    model.load_state_dict(state)
    policy = _DeterministicFastSACActor(model.eval()).eval()
    if policy.input_size != OBS_DIM:
        raise ValueError(f"actor reads {policy.input_size} values, expected {OBS_DIM}")
    return policy


def gripper_curve(capture):
    """(angle [rad], gap [m]) of the simulated 2F-85 from a recorded close/open sweep, sorted, gap monotonic."""
    data = np.load(capture)
    q, gap = data["sweep_q"], data["sweep_gap"]
    order = np.argsort(q)
    q, gap = q[order], gap[order]
    gap = np.minimum.accumulate(gap)                       # closing never widens the gap
    keep = np.r_[True, np.diff(q) > 1e-4]
    return q[keep].round(5).tolist(), gap[keep].round(6).tolist()


def spec(args, curve):
    angles, gaps = curve if curve else ([], [])
    return {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha1": hashlib.sha1(open(args.checkpoint, "rb").read()).hexdigest(),
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0",
        "control_rate_hz": 50.0,
        "quaternion_order": "x, y, z, w",
        "frames": {
            "base": "ROS base_link of the UR (the simulated robot root; the UR controller's 'base' turned 180 deg about z)",
            "tool_point": "tool0 (= wrist_3_link) origin + tool0 rotation @ tool_offset",
            "tool_offset": list(TOOL_OFFSET),
        },
        "observation": {
            "size": OBS_DIM,
            "layout": [
                {"slice": [0, 7], "name": "applied action at t-1", "content": "6 raw arm outputs, filtered gripper state"},
                {"slice": [7, 14], "name": "applied action at t", "content": "the action applied in the step that led here"},
                {"slice": [14, 41], "name": "sheet landmarks 0..8 minus the tool point",
                 "content": "x, y, z per landmark, base_link axes [m]; landmark k is sim node "
                            + str(LANDMARK_NODES) + "[k]"},
                {"slice": [41, 44], "name": "platform top-centre minus the tool point", "content": "base_link axes [m]"},
                {"slice": [44, 48], "name": "platform orientation", "content": "quaternion x, y, z, w in base_link"},
                {"slice": [48, 49], "name": "finger pad gap", "content": "[m], 0.080 open, ~0 shut"},
            ],
            "after_reset": "both action slots hold [0, 0, 0, 0, 0, 0, 0.5]",
        },
        "action": {
            "size": ACTION_DIM,
            "arm": "actions[0:6]: relative pose step of the tool point, (dx, dy, dz, rx, ry, rz) times arm_action_scale, "
                   "base_link axes, rotation as axis-angle; one step per control period (Isaac Lab "
                   "DifferentialInverseKinematicsAction, relative pose mode, body_offset = tool_offset)",
            "arm_action_scale": args.arm_action_scale,
            "gripper": "actions[6]: request; state <- state + filter_alpha * (clip(actions[6], -1, 1) - state), "
                       "state starts at initial_state; the state is what the observation records",
            "gripper_filter_alpha": 0.3,
            "gripper_initial_state": 0.5,
        },
        "gripper_gap_curve": {"finger_joint_rad": angles, "pad_gap_m": gaps,
                              "source": "simulated 2F-85 swept shut and open"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm_action_scale", type=float, default=0.03,
                        help="The arm action scale the policy is meant to run with (the play setting).")
    parser.add_argument("--out", default=None, help="Output folder (default: exported_real/ next to the checkpoint).")
    parser.add_argument("--check", default=None, help="obs_capture.npz to verify against (also gives the gripper curve).")
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "exported_real")
    os.makedirs(out, exist_ok=True)
    policy = load_actor(args.checkpoint)
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(policy, (dummy,), os.path.join(out, "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                      opset_version=17, dynamo=False)
    torch.jit.save(torch.jit.script(policy), os.path.join(out, "policy.pt"))
    curve = gripper_curve(args.check) if args.check else None
    with open(os.path.join(out, "policy_spec.json"), "w") as f:
        json.dump(spec(args, curve), f, indent=2)
    print(f"[export] wrote policy.onnx, policy.pt and policy_spec.json to {out}")
    if args.check:
        data = np.load(args.check)
        obs, actions, done = data["obs"], data["action"], data["done"]
        rows = [t for t in range(len(obs) - 1) if not done[t]]            # action t+1 was computed from obs t
        x = torch.as_tensor(obs[rows], dtype=torch.float32)
        with torch.no_grad():
            torch_out = policy(x).numpy()
        script_out = torch.jit.load(os.path.join(out, "policy.pt"))(x).detach().numpy()
        expected = actions[[t + 1 for t in rows]]
        print(f"[export] {len(rows)} recorded steps: max |policy - simulator's action| "
              f"torch {np.abs(torch_out - expected).max():.2e}, TorchScript {np.abs(script_out - expected).max():.2e}")
        np.savez(os.path.join(out, "check_inputs.npz"), obs=obs[rows].astype(np.float32), actions=expected.astype(np.float32))
        print(f"[export] check_inputs.npz saved for verifying the ONNX file in the runtime's Python")


if __name__ == "__main__":
    main()
