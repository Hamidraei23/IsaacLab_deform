"""Stable console grouping for RSL-RL, without changing logged scalars or aggregation."""

import builtins
import re


GROUPS = (
    ("Episode_Reward", "Reward totals (manager-normalized)"),
    ("Placement", "Placement outcomes"),
    ("Retreat", "Phase 3: end-effector retreat and hold"),
    ("PlacementReward", "Placement components (reward per episode, before term weight)"),
    ("MotionCost", "Motion costs (before curriculum scaling and term weight)"),
    ("MotionCostPhase2", "Phase 2/3 motion costs (applied per episode, before term weight)"),
    ("Curriculum", "Curriculum"),
    ("Events", "Grasp events"),
    ("Stage", "Grasp shaping"),
    ("Milestone", "Grasp milestones"),
    ("AtClosure", "Geometry at gripper closure"),
    ("Metrics", "Command diagnostics"),
    ("Episode_Termination", "Episode endings"),
)
ROW_ORDER = {
    "Episode_Reward": ("grasp_stage", "grasp_progress", "lift_milestones", "gripper_closures",
                       "placement", "motion_cost", "ee_speed", "table_clearance"),
    "Placement": ("success", "released", "best_coverage", "best_released_coverage",
                  "center_progress", "high_releases"),
    "Retreat": ("entered", "reached_target", "final_distance"),
    "PlacementReward": ("approach", "lower", "coverage_progress", "opening_progress", "release",
                        "released_coverage_progress", "released_coverage", "retreat_progress",
                        "off_target", "retreat_distance", "retreat_motion", "success"),
    "MotionCost": ("speed", "angular", "hand_acc", "joint_acc", "jerk", "command", "closure", "joint_limit"),
    "MotionCostPhase2": ("speed", "angular", "hand_acc", "joint_acc", "jerk", "command", "closure", "joint_limit"),
    "Events": ("extraction", "grasp", "closures", "commanded_closures", "bad_closure", "early_release"),
    "Episode_Termination": ("draped", "extracted", "time_out"),
}
METRIC_ROW = re.compile(r"^\s*([A-Za-z][\w]*(?:/[^:\n]+)+):\s+(.+)$")


def format_training_table(message: str) -> str:
    """Group metric rows; preserve the runner's summary, values, and timing footer."""
    if "Learning iteration " not in message:
        return message
    lines = message.splitlines(keepends=True)
    rows = []
    indices = set()
    for i, line in enumerate(lines):
        match = METRIC_ROW.match(line.rstrip("\n"))
        if match:
            key = match.group(1)
            prefix, _, name = key.partition("/")
            rows.append((prefix, name, line))
            indices.add(i)
    if not rows:
        return message

    groups = dict(GROUPS)
    order = {name: i for i, (name, _) in enumerate(GROUPS)}

    def row_key(row):
        prefix, name, _ = row
        names = ROW_ORDER.get(prefix, ())
        return (order.get(prefix, len(order)), prefix,
                names.index(name) if name in names else len(names), name)

    grouped = []
    previous = None
    for prefix, name, line in sorted(rows, key=row_key):
        if prefix != previous:
            grouped.append(f"\n  -- {groups.get(prefix, prefix)} --\n")
            previous = prefix
        grouped.append(line if line.endswith("\n") else line + "\n")
    grouped.append("\n")
    first = min(indices)
    return "".join(
        "".join(grouped) if i == first else "" if i in indices else line
        for i, line in enumerate(lines)
    )


def install_training_table() -> None:
    """Replace only the RSL-RL logger module's console print function.

    Its upstream logger iterates a set of extra keys. Ordering env.extras cannot fix
    that. Format its completed console message instead, leaving W&B/TensorBoard,
    scalar averaging, distributed logging, and minimal output under its control.
    No installed dependency files or process-wide builtins are modified.
    """
    from rsl_rl.utils import logger

    if getattr(logger, "_sheet_training_table_installed", False):
        return
    original_print = getattr(logger, "print", builtins.print)

    def print_table(*args, **kwargs):
        args = tuple(format_training_table(arg) if isinstance(arg, str) else arg for arg in args)
        return original_print(*args, **kwargs)

    logger.print = print_table
    logger._sheet_training_table_installed = True
