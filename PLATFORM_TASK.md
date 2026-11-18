# Sheet placement on a raised platform

Activate the existing environment with `active_isaac`, then run from this directory:

```bash
# Grasp, extract, and place over the rectangular platform.
python scripts/train_platform.py --num_envs 1024 --viz none

# Train grasp and extraction only.
python scripts/train_platform.py --grasp --num_envs 1024 --viz none
```

Platform episodes are 1,000 control steps (33.3 s at 30 Hz); the capsule and hand tasks keep 500.
The wrapper defaults to RSL-RL PPO, 3,000 iterations, 24 steps per rollout, and
256/128/64 actor and critic layers with observation normalization. Other training
options and Hydra overrides pass through, for example `agent.max_iterations=5000`.
Logs are separated into `logs/rsl_rl/sheet_platform` and
`logs/rsl_rl/sheet_platform_grasp`. Add `--wandb` to log to Weights & Biases instead of
TensorBoard (project = experiment name), or `--wandb=<project>` to choose the project.
With `agent.run_name=<name>` the run folder is exactly `<name>` (no date prefix); an existing
folder of that name is refused. Without a run name, or with `--timestamp`, Isaac Lab's
`<date>_<time>_<name>` folder is used.

RSL-RL console tables group metrics in a fixed order: reward totals, placement/retreat outcomes,
placement reward components, motion costs, curriculum, grasp diagnostics, and episode endings.
Rows within each group have a stable order. `PlacementReward/*` reports each placement
component's mean accumulated reward per completed episode, before the placement term weight.
`Episode_Reward/*` uses the reward manager's normalization by the configured episode duration;
`MotionCost/*` reports component costs before curriculum scaling and the overall motion weight.
`MotionCostPhase2/*` reports applied costs after extraction (phases 2 and 3), including the
fixed scale and cap, before the overall motion weight.
These different units are identified in the group headings. TensorBoard/W&B scalar names
and aggregation remain unchanged, with the new placement component metrics added.
The grouped table applies to new processes launched through `scripts/train.py` or
`scripts/train_platform.py`; an already running process must be restarted to use it.

To continue a grasp checkpoint on placement, keep the same network configuration:

```bash
python scripts/train_platform.py --num_envs 1024 --viz none \
  --resume --load_run <grasp-run-directory-name> --checkpoint model_1000.pt \
  agent.experiment_name=sheet_platform_grasp agent.run_name=placement_from_grasp
```

The checkpoint loads the optimizer as well as the policy; the iteration budget is
additional training. The two tasks have identical observation and action layouts.
This command uses the grasp experiment directory to find the checkpoint and writes
the new placement run beside it. `--load_run` takes a run-name regular expression, not an absolute path.

Watch a placement checkpoint using the existing playback script:

```bash
python scripts/play.py --rl_library rsl_rl --task Template-Sheet-Rl-Platform-v0 \
  --checkpoint /absolute/path/to/model_1000.pt --viz newton_gl --lite
```

For a grasp-only checkpoint, select `Template-Sheet-Rl-Platform-Grasp-v0` instead.
This Isaac Lab checkout uses `--viz none` for headless execution.

## Task behavior

The platform footprint is taken directly from `SHEET_SIZE` (currently 20 × 20 cm).
Its top is 2 cm above the table. Only its position and yaw are randomized at reset,
on the opposite side of the workspace from the sheet; overlapping layouts move the
platform outward until it clears the slot and sheet. The sheet and slot are fixed at
the best of 100 tested world-right (`-Y`) poses: centroid
`(0.40946364, -0.07716751, 0.102)` and yaw `-0.53996944 rad`. With `model_920.pt`,
75 poses extracted successfully and this one had the highest return (`11,377.20`).
The internal `mannequin_arm` scene name remains so the existing Newton rigid-to-cloth
coupling includes the cuboid collider.

The original sheet material, resolution, grasp detector, nine observed landmarks,
and lower-edge marker remain. Both keyboard and automatic disturbances are disabled.
The existing seven-value goal observation now contains the platform's top-centre
position and orientation in the robot frame. No mid-episode goal movement occurs.

Placement rewards activate after extraction. A 16 × 16 equal-area sampling grid
estimates how much of the top is covered by the union of actual cloth triangles.
Overlapping folds count once. Triangles must lie within 6 mm of the expected cloth
height (2 mm above the platform); table-level and hovering sheets do not count.

The task has three phases: extract; lower and release onto the platform; retreat to a fixed
point above its centre and hold. Observation/action layouts remain checkpoint-compatible.
The old table-clearance and grasp/closure shaping stop after extraction. Motion costs remain
active with a separate smaller scale and cap for placement and retreat.

A new filtered open command above **10 cm fingertip height above the table** immediately
ends the episode as `released_high`. The check also catches fingers physically opening above
the ceiling after an earlier low command. It runs before rewards; that terminal step earns
no positive placement/retreat credit. A low release followed by retreat while the fingers
remain open is allowed. There is no additional one-shot high-release penalty.

Phase 3 starts when a valid hold has ended and the fingers physically open. The target is
the **fingertip grasp frame at platform top-centre plus 20 cm**, i.e. 22 cm above the table
with this 2 cm platform. It is not 20 cm above wherever release happened. Success requires
all of these conditions continuously for **1 second** (30 control steps):

- At least 50% platform coverage, sheet unheld, and fingers open.
- Every cloth node moving below 3 cm/s.
- Fingertips within 3 cm of the retreat target, moving at most 3 cm/s.
- End-effector angular speed at most 0.2 rad/s.

Movement or leaving the target restarts the timer. Retreat is reward-driven, not a scripted
controller, so an existing throwing policy needs further training for these conditions.

Let `C` be supported platform coverage in [0, 1], `d` the sheet-centre 3-D distance to the
platform-top centre, `r` its horizontal distance, and `h` fingertip height above the table.
Define `A = exp(-r / 0.10)`, `L = exp(-max(h - 0.03, 0) / 0.08)` and readiness `Q = A*L`.
A lower release gets more credit down to 3 cm above the table (1 cm above the platform).
Above that height credit decays smoothly, up to the hard 10 cm opening ceiling.

For a bounded score `x`, progress means `max(x - previous_episode_best, 0)`.
Each score's best is updated only when eligible, and is cleared at episode reset.
Approach/lowering start from their extraction values, so the phase transition itself earns
no placement bonus. These are finite progress budgets, not policy-invariant potential shaping.

| Part | Maximum / formula | Eligibility |
|---|---|---|
| Approach | 500 × progress of `exp(-d / 0.20)` | Holding after extraction |
| Lowering | 500 × progress of `Q` | Holding after extraction |
| Supported coverage | 1500 × progress of `C` | After extraction, held or released |
| Opening | 100 × progress of `Q * opening_fraction` | Still holding, before first release |
| Actual release | `1500 * A_release * L_release * (0.25 + 0.75*C)` once | Hold has ended, then fingers physically open |
| Released coverage progress | 1000 × progress of `C` | After release, unheld, fingers open |
| Released coverage | `5*C` per step | After release, unheld, fingers open |
| Retreat progress | 1000 × progress of `exp(-tip_target_distance / 0.10)` | Phase 3, unheld, fingers open; baseline captured at first release |
| Sheet off target | 0 inside 30 cm, linear to -5/step at 40 cm, capped | Sheet centre horizontal distance from platform centre; after extraction |
| End effector off target | 0 inside 30 cm, linear to -5/step at 40 cm, capped | Phase 3; fingertip 3-D distance from platform top-centre |
| Motion near retreat target | Up to -2/step for excess linear/angular speed | Phase 3, within 6 cm of the retreat target |
| Success | `5000*C` once, episode ends | All sheet and end-effector conditions above hold for 1 s |

The general motion term uses a 15 cm/s free translation allowance after extraction, with
speed, angular speed, acceleration, command change and joint-limit costs scaled by 0.2 and
capped at **5 reward units per control step total**, with the default term weight -1.
This scale does not ramp with extraction rate. Phase-one costs retain their previous curriculum.
Near the retreat goal, the additional motion cost starts above 3 cm/s or 0.2 rad/s and reaches
its cap according to `2 * min(1, max(v-0.03,0)/0.10 + max(omega-0.20,0))`.

`opening_fraction` is the applied gripper filter's progress from fully closed to the opening
threshold, clipped to [0, 1]. Raw policy action magnitude alone earns nothing. Both that
score and coverage progress have finite budgets: holding, opening/closing repeatedly, or
backing away and returning cannot repeatedly collect them. Held cloth has no standing reward.
The release stores height/alignment when the hold ends, so lowering an already empty gripper
cannot improve the release payout. Coverage earned as the cloth subsequently falls still counts.

At gamma 0.99, even 100% released coverage forever is worth at most `5/(1-0.99) = 500`;
the success payout at the 50% threshold is 2500. Thus settling and finishing beats stalling
below the success threshold. Progress plus release budgets total at most 6100 before success
and ongoing released coverage. Units are literal rewards; the reward manager's dt is canceled.
Phase-one reward magnitudes are retained to preserve the pretrained extraction incentive.

`Placement/best_released_coverage` separates genuine released coverage from coverage while
held. `Placement/released` counts valid completed releases; `Placement/high_releases` is a
diagnostic for those releases above the 3 cm full-credit height. `Episode_Termination/released_high`
counts rejected openings above 10 cm. `Retreat/entered`, `Retreat/reached_target`, and
`Retreat/final_distance` describe phase 3 (final distance averages over released episodes).
`PlacementReward/retreat_*` shows its reward/cost components. Success is read by the termination
manager on the next step. Playback plots show the release ceiling and retreat target height.

Resume the learned placement run into a new run folder, using its existing exploration spread:

```bash
active_isaac
python scripts/train_platform.py --num_envs 1024 --viz none \
  --resume --load_run '^placement_880_drape_v2$' --checkpoint 'model_.*.pt' \
  agent.experiment_name=sheet_platform_grasp agent.run_name=placement_drape_v3 \
  agent.max_iterations=2000 agent.algorithm.gamma=0.99 agent.algorithm.entropy_coef=0.003 \
  --wandb=sheet_placement
```

`--load_run` is a regular expression: the anchors select exactly the learned run rather
than also matching a similarly named smoke-test directory.

Do not carry over the old `env.rewards.motion_cost.weight=-15.0` override. The default
weight is -1, with the separate scale and cap described above for phases 2/3. A -15 override
would multiply that cost by 15. Nonzero entropy encourages continued exploration; it cannot
guarantee convergence. Resuming restores the actor, critic, optimizer and learned action std;
changing the configured initial std does not override the loaded std.

Evaluate the new run:

```bash
PYOPENGL_PLATFORM=glx python scripts/play.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-v0 \
  --checkpoint "$(ls -t "$PWD"/logs/rsl_rl/sheet_platform_grasp/placement_drape_v3/model_*.pt | head -1)" \
  --viz newton_gl --num_envs 1 --plot
```

Historical validation of the earlier two-phase reward: the initial deterministic
16-environment check of the unchanged `model_880.pt` extracted in
16/16 episodes, released in 8/16, and succeeded in 0/16. Mean best coverage was 3.76%.
These are baseline compatibility measurements, not evidence of convergence with the new reward.
Results: `evaluations/placement_880_reward_rewrite/summary.json`.

For that earlier version, 60 regression tests passed. A separate 64-environment, 8-iteration PPO
resume from the original checkpoint completed successfully in
`logs/rsl_rl/sheet_platform_grasp/placement_880_drape_v2_smoke`. This is a loading/update
smoke test, not a trained placement policy. For the new phase 3, resume the learned placement
run using the command above.

Phase-3 validation: 73 regression tests passed. An 8-environment evaluation of the learned
`placement_880_drape_v2/model_2800.pt` extracted in all episodes, then all eight ended with
`released_high`, correctly rejecting its high-opening behavior under the new rule.
Results are in `evaluations/placement_phase3_2800`. A four-iteration, 32-environment training
resume from that exact checkpoint also completed in `placement_drape_v3_smoke_validated`.
These validate execution and the new constraints, not convergence of phase 3.

`--grasp` removes placement rewards and ends on extraction. Both platform tasks now
share the redesigned first phase described below. The capsule/hand task is unchanged.

## First-phase reward (staged, trained from scratch)

The goal is fast convergence from random weights with smoothness as the strict part. The
900/920 checkpoints are used only as reference paths: `scripts/record_grasp.py` recordings
of them are replayed offline to check that the reward rises smoothly along a successful grasp.

The existing top-edge grasp and extraction detector is kept (node indices, 4 cm capture
radius, finger thresholds), with one addition for the platform tasks: a hold also requires a
finger width above 1 mm (`min_hold_width`). A real pinch stops at about 2 mm (recorded minimum
1.7 mm, the cloth's particle radius); fingers shut on nothing read 0. Without this, the first
from-scratch runs approached with the gripper already shut, and the detector scored fingertips
touching the edge as a grasp in 99% of episodes, although nothing could be lifted.

`grasp_progress` combines a small per-step *state* reward for reaching and grasping with
potential-based shaping for lifting. Both are computed from observable quantities, with no
hidden episode-best memory:

| Part | Value | Behavior |
|---|---:|---|
| Reach | 0–0.25 per step | Top-edge centre distance, `tanh` scales of 25 cm and 4 cm |
| Alignment | 0–0.05 per step | Downward and slot-square, weighted by `exp(-d / 10 cm)` |
| Closure | 0–0.10 per step | Finger closure within about 2 cm of the top edge, only with cloth between the pads |
| Holding | 0.5 per step | Full pre-grasp credit plus 0.1, so it only just beats every pre-grasp state |
| Lift | `0.99·Φ' − Φ` | While holding, `Φ = 50 × top-edge rise / 20 cm + 50 × lowest node / 17 cm` |

Lift shaping pays about 2.5 per centimetre the pinched top edge rises, plus about 3 per
centimetre the whole sheet clears. The edge part matters: in recorded extractions the hand
rose 1–4 cm (median 2.8 cm) before the sheet's lowest node moved, as slack was taken up. A
potential on the lowest node alone gave no signal across that gap, and no from-scratch run
lifted. The edge rest height is captured on each episode's first step. Dropping the sheet gives
the credit back, and holding still at height costs `0.01·Φ` per step, so it cannot be farmed.
A 0.9-per-step hold reward with a weak linear lift term learned to hold without lifting.

`grasp_stage` pays 10 for the first top-edge grasp and 200 for extraction, more than the 50
value of holding forever at gamma 0.99. A missed closure costs 0.5 and early release 5. The
term stops paying after extraction; placement uses the separate reward described above.
The grasp task uses gamma 0.99 (the shaping gamma must match), initial action std 0.5 and
entropy 0.003.

### Lift milestones (`lift_milestones`, added to all of the above)

| Event | Reward |
|---|---:|
| First real top-edge grasp in the episode | +400 once |
| Top-edge rise while holding, on new episode-best height | `P(h) = 6106·(1 − e^(−h / 11.7 cm))`: 500 for the first cm, 5,000 by extraction |
| Holding while the edge rises less than 0.5 mm that step | −0.5 per step |
| Hold ends before extraction having raised the edge less than 1 cm | −400 per release |

The two lift anchors fix the curve: 500 of 5,000 in the first centimetre of 20 cm travel is twice
the linear share, so the pay per millimetre starts at about 52 and decays (22 at 10 cm, 10 at 19 cm).
In recorded extractions the edge has risen about 12–14 cm at extraction (about 4,000 paid); the
remainder up to 5,000 is paid on the extraction step. A release is counted only after 3 steps
without a hold, so a brief flicker of the hold detector is not charged. These magnitudes are much
larger than the motion costs below, so the smoothing curriculum weighs little against them.

### Gripper closure cost (`gripper_closures`)

Every commanded gripper closure after the first costs 50, before extraction. A closure is the
policy's raw gripper action switching from open (> 0) to closed (≤ 0), read before the gripper
filter. Episodes start open, so an initial closed action is the free first closure. Exploration
noise that flips the raw sign also counts. `Events/commanded_closures` logs the per-episode count.

## Motion smoothness

The command filter from the previous redesign is kept: arm scale 0.02, 50 ms low-pass,
24 action units/s rate limit, interpolation across physics substeps. The joint target is now
*integrated* (`target += 0.02 × filtered command` per control step) instead of re-based on the
measured position each substep. With gravity enabled, the stock relative action discarded the
PD error that holds the arm up: a zero command sagged about 1.3 mm per step, and most early
from-scratch episodes ended below the table. With the integrated target, a zero command holds
the hand with no measurable drift, after a one-time settle of about 26 mm just after reset.
The target stays within 0.15 rad of the measured joint position, so a blocked arm cannot wind
it up.

The binary gripper is driven by a filtered command, `state += 0.3 × (clip(action, ±1) − state)`,
and opens when the state is above 0 (it starts open). With the stock action, a single exploratory
sample above zero opened the fingers: a policy holding at mean −0.5, std 0.43 dropped the sheet on
about 12% of steps, so a 30-step lift almost never survived and training settled on holding without
lifting. Now opening takes several consecutive open requests, and a deliberate open or close takes
about two steps. The observation's gripper value is this filter state. `motion_cost`
charges per control step (Huber costs, finger joints excluded, resets excluded):

| Component | Definition | Weight |
|---|---|---:|
| speed | Hand speed above an envelope, relative to it: 40 cm/s in free space, 5 cm/s at the top edge (8 cm scale), 25 cm/s while holding | 2.0 |
| angular | Hand angular speed above 1 rad/s (0.5 rad/s scale) | 0.2 |
| hand_acc | Cartesian hand acceleration, 1.5 m/s² scale | 2.5 |
| joint_acc | Arm joint acceleration, 5 rad/s² scale | 0.5 |
| jerk | Arm joint jerk, 100 rad/s³ scale | 0.2 |
| command | Change of applied (filtered) arm command, 0.2 units | 0.1 |
| closure | Hand speed (3 cm/s scale) and rotation (0.3 rad/s) while closing near the edge | 1.0 |
| joint_limit | Arm joints in the last 10% of their range at either end: sum of p², p = 0 at the zone edge, 1 at the limit | 5.0 |

Motion inside the envelope costs almost nothing, so the intended path (decelerating approach,
stationary closure, lift at about 25 cm/s, roughly 2.5 s) is cheap. Replayed offline, the
900 path costs 177 per episode and the smoother warm-start pilot 115. Both exceed what rushing
saves in discounted extraction reward. The whole cost is scaled by a curriculum, from 5% to 100%
while the running extraction rate (about 1,000 episodes) rises from 20% to 80%. If success
falls, the scale falls with it. Logs: `Curriculum/success_rate`, `Curriculum/motion_scale`,
`MotionCost/*` (per-episode, unscaled), and `Stage/*`.

Train from scratch with the standard command:

```bash
python scripts/train_platform.py --grasp --num_envs 1024 --viz none agent.run_name=<name>
```

`scripts/train_grasp_progress.py --warm_start` still exists but is no longer the recommended
path. Use full-state `--resume` only for checkpoints trained with this reward.

## Saved evaluations and checks

`evaluations/grasp_redesign/baseline_900` and `baseline_920` contain original-controller
trajectories. `filtered_900` contains the same actor with the new controller.
Each has `traces.npz` and `summary.json`. The recorder captures terminal physics states
before automatic reset, including joint position/velocity/acceleration, requested and
applied actions, hand pose/velocity, grasp geometry, reward terms and episode outcomes.
Later recordings also save environment and agent YAML and motion component values.
`smoothed_900` was an intermediate reward-only check: its action preset was still the
original controller. It is not used to claim smoothing improvements.
`progress_v2_final` records the 100-iteration, 256-environment warm-start pilot
(`logs/rsl_rl/sheet_platform_grasp/progress_v2/model_final.pt`); `comparison_v2` compares
it with the baselines. It extracted 64/64. Motion cost was 8.6 per episode against a task
return of about 98, and 91% of it came from the closure term. The acceleration, jerk and
command terms together were below 1, so they did not constrain the policy.

```bash
python scripts/record_grasp.py --viz none --num_envs 32 --episodes 2 \
  --checkpoint /absolute/path/to/model.pt --out evaluations/my_grasp_check
python -m unittest discover -s tests -v
```
