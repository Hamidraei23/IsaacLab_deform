"""Copy a checkpoint with the policy's exploration spread reset, ready for --resume.

Resuming loads the actor's per-action standard deviation along with its weights. When that spread
has blown up (the placement runs reached 5-6 on the wrist joints against a 2.4 action clip, so
~90% of arm actions were pinned at the clip), this writes a copy with new values and leaves the
original untouched. The weights, critic, optimizer state and iteration count are kept.

Example (arm 0.5, gripper 3.0, the last action)::

    python scripts/reset_policy_std.py \\
        --checkpoint logs/rsl_rl/sheet_platform_grasp/placement_3319_joint_limits/model_3600.pt \\
        --out_run placement_3600_std_reset --std 0.5 --gripper_std 3.0

Then resume with ``--load_run placement_3600_std_reset --checkpoint model_3600.pt``. Omit ``--std`` to
keep the arm's spread and change only the gripper's.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

STD_KEY = "distribution.std_param"
LOG_STD_KEY = "learned_log_std"
"""The FastSAC actor's (``FastSACActorModel``) learned pre-squash spread, stored as its log."""


def reset_std(state: dict, std: float | None, gripper_std: float | None, gripper_index: int = -1) -> tuple[list, list]:
    """Set every entry of the actor's std parameter to ``std`` (kept as is when ``None``) and the
    gripper entry to ``gripper_std`` if given. Returns the old and new values (as spreads, not logs)."""
    actor = state["actor_state_dict"]
    key = STD_KEY if STD_KEY in actor else LOG_STD_KEY if LOG_STD_KEY in actor else None
    if key is None:
        raise KeyError(f"No '{STD_KEY}' or '{LOG_STD_KEY}' in the actor state; keys: {sorted(actor)}")
    values = actor[key] if key == STD_KEY else actor[key].exp()
    old = values.tolist()
    new = values.clone() if std is None else torch.full_like(values, std)
    if gripper_std is not None:
        new[..., gripper_index] = gripper_std
    actor[key] = new if key == STD_KEY else new.log()
    return old, new.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out_run", required=True, help="New run folder, created beside the checkpoint's run.")
    parser.add_argument("--std", type=float, default=None, help="Spread for every action (default: keep).")
    parser.add_argument("--gripper_std", type=float, default=None, help="Spread for the gripper (last) action.")
    args = parser.parse_args()
    if (args.std is not None and args.std <= 0) or (args.gripper_std is not None and args.gripper_std <= 0):
        raise ValueError("Standard deviations must be positive.")
    source = args.checkpoint.resolve()
    out_dir = source.parent.parent / args.out_run
    if out_dir.exists():
        raise FileExistsError(f"Run folder already exists: {out_dir}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    old, new = reset_std(state, args.std, args.gripper_std)
    out_dir.mkdir(parents=True)
    torch.save(state, out_dir / source.name)
    if (source.parent / "params").is_dir():
        shutil.copytree(source.parent / "params", out_dir / "params")
    (out_dir / "std_reset.json").write_text(json.dumps(
        {"source": str(source), "old_std": old, "new_std": new}, indent=2))
    print(f"old std: {[round(v, 3) for v in old]}")
    print(f"new std: {[round(v, 3) for v in new]}")
    print(f"written: {out_dir / source.name}")
    print(f"resume with: --load_run {args.out_run} --checkpoint {source.name}")


if __name__ == "__main__":
    main()
