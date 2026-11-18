"""Train platform placement, or extraction only with --grasp.

All other arguments are passed to the existing Isaac Lab training entry point.
"""

import sys

PLACEMENT_TASK = "Template-Sheet-Rl-Platform-v0"
GRASP_TASK = "Template-Sheet-Rl-Platform-Grasp-v0"


def training_args(argv):
    if any(arg == "--task" or arg.startswith("--task=") for arg in argv):
        raise ValueError("train_platform.py selects its task automatically; use --grasp for extraction only.")
    grasp = "--grasp" in argv
    args = [arg for arg in argv if arg != "--grasp"]
    if not any(arg == "--rl_library" or arg.startswith("--rl_library=") for arg in args):
        args = ["--rl_library", "rsl_rl", *args]
    return ["--task", GRASP_TASK if grasp else PLACEMENT_TASK, *args]


def main(argv=None):
    from train import main as train_main
    return train_main(training_args(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
