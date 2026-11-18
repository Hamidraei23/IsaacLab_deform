"""Write a FastSAC checkpoint's actor as an rsl_rl checkpoint, so rsl_rl PPO (ReferencePPO) can fine-tune it as it is.

The actor becomes ``FastSACActorModel`` (same network, frozen normaliser, squashed actions); the critic is fresh,
to be warmed with ``agent.algorithm.critic_only=true``. The file lands where ``train.py --resume`` looks:
``logs/rsl_rl/<experiment_name>/<run_name>/model_0.pt``.

Usage (inside ``active_isaac``)::

    python scripts/fastsac_actor_to_rsl_rl.py \
        --checkpoint logs/fast_sac/sheet_platform_twist_sac/2026-10-27_11-50-01_repaired_student_D2/model_4000.pt \
        --run_name from_fastsac_model_4000
"""

import argparse
import json
import os

import torch

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--checkpoint", required=True, help="FastSAC checkpoint (model_*.pt).")
parser.add_argument("--run_name", required=True, help="Run folder name under logs/rsl_rl/<experiment_name>/.")
parser.add_argument("--runner_cfg", default="sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_ppo_cfg:UR10eFineTuneRunnerCfg")
parser.add_argument("--critic_dim", type=int, default=55, help="Size of the runner's critic observation group (ur_critic: 55).")
args = parser.parse_args()

import importlib  # noqa: E402

from sheet_rl.algorithms.fast_sac.buffers import EmpiricalNormalization  # noqa: E402
from sheet_rl.algorithms.fast_sac.networks import Actor  # noqa: E402
from sheet_rl.algorithms.fast_sac.rsl_rl_actor import FastSACActorModel, rsl_rl_checkpoint_from_fastsac  # noqa: E402

module, name = args.runner_cfg.split(":")
runner_cfg = getattr(importlib.import_module(module), name)()
fastsac = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
converted = rsl_rl_checkpoint_from_fastsac(fastsac, runner_cfg, args.critic_dim, source=os.path.abspath(args.checkpoint))

# check: the wrapped actor reproduces the FastSAC actor's deterministic actions and log-probabilities
from tensordict import TensorDict  # noqa: E402

group = runner_cfg.obs_groups["actor"][0]
obs = torch.randn(256, fastsac["dims"]["n_obs"])
kwargs = runner_cfg.actor.to_dict()
kwargs.pop("class_name")
wrapped = FastSACActorModel(TensorDict({group: obs}, batch_size=[256]), runner_cfg.obs_groups, "actor",
                            fastsac["dims"]["n_act"], **kwargs)
wrapped.load_state_dict(converted["actor_state_dict"])
cfg = fastsac["cfg"]
reference = Actor(fastsac["dims"]["n_obs"], fastsac["dims"]["n_act"], cfg["actor_hidden_dim"], cfg["log_std_min"],
                  cfg["log_std_max"], cfg["use_tanh"], cfg["use_layer_norm"])
reference.load_state_dict(fastsac["actor_state_dict"])
normalizer = EmpiricalNormalization(fastsac["dims"]["n_obs"], "cpu")
normalizer.load_state_dict(fastsac["obs_normalizer_state"])
with torch.no_grad():
    expected, _, _ = reference(normalizer(obs, update=False))
    got = wrapped(TensorDict({group: obs}, batch_size=[256]))
    actions = wrapped.distribution.sample()
    lp_expected = reference.log_prob_of(normalizer(obs, update=False), actions)
    lp_got = wrapped.get_output_log_prob(actions)
action_err, lp_err = (expected - got).abs().max().item(), (lp_expected - lp_got).abs().max().item()
if action_err > 1e-5 or lp_err > 1e-3:
    raise SystemExit(f"conversion check failed: action error {action_err:.2e}, log-prob error {lp_err:.2e}")

folder = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name, args.run_name)
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, "model_0.pt")
torch.save(converted, path)
with open(os.path.join(folder, "converted_from.json"), "w") as f:
    json.dump({"fastsac_checkpoint": os.path.abspath(args.checkpoint), "runner_cfg": args.runner_cfg,
               "actor_obs_groups": runner_cfg.obs_groups["actor"], "critic_obs_groups": runner_cfg.obs_groups["critic"],
               "critic": "freshly initialised", "check": {"max_action_error": action_err, "max_log_prob_error": lp_err}},
              f, indent=2)
print(f"wrote {path} (actions match to {action_err:.1e}, log-probs to {lp_err:.1e})")
