# Franka cube lift: from joint control to a twist policy

This is a side project next to the sheet task. Isaac Lab's Franka cube-lift task is trained with
joint-position actions. Its policy is then distilled with DAgger into a student that commands the
hand directly, as a twist (a delta pose of the tool point), and the student is fine-tuned with PPO.
This document covers the reward and the robot model, the conversion from the joint teacher to the
twist student, every command in the pipeline, and the results measured so far, including a test with
the cube on a 5 cm platform.

```
joint teacher (PPO)  ──DAgger──▶  twist student  ──PPO resume──▶  twist policy  ──▶  evaluation
IsaacContrib-Lift-Cube-Franka     Lift-Cube-Franka-Twist-v0       (same task)        play_franka_twist.py
36 obs → 8 joint actions          35 obs → 6D delta pose + gripper                   (table / platform)
```

Activate the existing environment with `active_isaac`, then run every command from this directory
(`sheet_rl/sheet_rl`): the scripts find `logs/` relative to it. The interpreter must be the one in
`IsaacLab/env_isaaclab`. The one in `IsaacLab/.venv` does not have `sheet_rl` installed, and
`scripts/train.py` stops with `No module named 'sheet_rl'`.

## 1. The task and its reward

The environment is `IsaacContrib-Lift-Cube-Franka`. Its base config is
`IsaacLab/.../contrib/lift/lift_env_cfg.py`, and the robot, actions and cube are set in
`config/franka/joint_pos_env_cfg.py`. A Franka Panda picks a 4.8 cm cube off a table and carries it
to a goal position, sampled at x 0.4–0.6 m, y ±0.25 m, z 0.25–0.5 m in the robot base frame. Each
episode lasts 5 s at 50 Hz (sim dt 0.01 s, decimation 2). The cube's start is randomised by ±0.1 m
in x and ±0.25 m in y.

Per step, with `d_ee` the hand–cube distance, `d_goal` the cube–goal distance and `L` = 1 when the
cube is above z = 0.04 m:

| Term | Function | Weight | Pays for |
|---|---|---|---|
| `reaching_object` | `1 - tanh(d_ee / 0.1)` | 1.0 | moving the hand to the cube |
| `lifting_object` | `L` | 15.0 | holding the cube above 4 cm |
| `object_goal_tracking` | `L · (1 - tanh(d_goal / 0.3))` | 16.0 | carrying it toward the goal (coarse) |
| `object_goal_tracking_fine_grained` | `L · (1 - tanh(d_goal / 0.05))` | 5.0 | placing it on the goal (fine) |
| `action_rate` | `‖a_t - a_{t-1}‖²` | see below | jerky commands |
| `joint_vel` | `‖q̇‖²` | see below | fast joint motion |

The goal terms pay only while the cube is lifted, so the reward is staged: reach, then lift, then
carry. The reward manager multiplies every term by the 0.02 s step, so logged per-step values are
50× smaller than the weights. `success_threshold=0.10` changes no reward. It only feeds
`Metrics/success_rate`: the share of episodes in which the cube was ever within 10 cm of the goal
while lifted. Episodes end on timeout, or when the cube falls below z = −0.05 m.

**Motion costs.** Upstream, `action_rate` and `joint_vel` start at −1e-4, and a curriculum raises them
to −0.1 after 10,000 env steps. That count is `env.step` calls, so it is reached after about 420
iterations of 24 steps, and it restarts from zero on every resume. Two changes here:

- `joint_pos_env_cfg.py` (in the Isaac Lab checkout) starts both terms at −0.1, so the joint teacher
  feels the full cost from step 0. The curriculum then has no effect.
- The twist task (`franka_lift_twist/twist_env_cfg.py`) sets both terms, and the curriculum, to
  **−0.2**. The curriculum must change too: once past 10,000 steps it assigns its own weight on every
  step, and at −0.1 it would undo the increase partway through a run.

**Gravity** is on: the scene uses −9.81 m/s², and `FRANKA_PANDA_CFG` has `disable_gravity=False`.
There is no gravity compensation. The joint teacher's PD drives (stiffness 80, damping 4, effort
limits 87 N·m on joints 1–4 and 12 N·m on joints 5–7) hold the arm up with the policy's targets.
The IK-Rel arm used by the twist task (`FRANKA_PANDA_HIGH_PD_CFG`) is different: it has stiffness 400,
damping 80 and gravity disabled. That difference is the central problem in section 3.

## 2. The joint teacher

The cached pretrained weights in `.pretrained_checkpoints/rsl_rl/Isaac-Lift-Cube-Franka-v0/checkpoint.pt`
were saved at iteration 299 for the core `Isaac-Lift-Cube-Franka-v0`. They load unchanged into the
contrib task: both are 36 → 8 MLPs (256-128-64, ELU). The trainer resumes only from a run folder, so
the checkpoint is copied unchanged to `logs/rsl_rl/lift_cube_franka/pretrained_isaac_v0/model_299.pt`.

| 10 iterations, 1024 envs, seed 42 | From scratch | From pretrained |
|---|---|---|
| Mean episode return | 0.70 | 85.98 |
| Success rate | 0.00 | 0.57 |

The pretrained curve climbing from 0.7 to 86 over those 10 iterations is not learning. Episodes start
at random lengths and only finished episodes are counted, so the first iterations average short ones.

`pretrained_motion_cost` resumes that checkpoint with the −0.1 motion costs:

```bash
python scripts/train.py --rl_library rsl_rl --task IsaacContrib-Lift-Cube-Franka --num_envs 1024 --seed 42 \
  --resume --load_run pretrained_isaac_v0 --checkpoint model_299.pt \
  agent.run_name=pretrained_motion_cost agent.max_iterations=1500
```

It was stopped at iteration 1591. Over its last 50 iterations success was 0.99, the return 134 and
`Episode_Reward/action_rate` −0.68 (down from −3.1 at the start). **`model_1550.pt` is the teacher
used below.** Checkpoints are saved every 50 iterations (`save_interval`), plus one at the last
iteration.

## 3. The twist task and the teacher-to-student conversion

`Lift-Cube-Franka-Twist-v0` is registered in `source/sheet_rl/sheet_rl/tasks/manager_based/franka_lift_twist/`.
It is Isaac Lab's IK-Rel lift task (`ik_rel_env_cfg.py`) with the −0.2 motion costs, and Isaac Lab
registers that task without an RSL-RL runner. Its action has 7 entries, one per 20 ms:

| Index | Meaning | After the 0.5 scale |
|---|---|---|
| 0–2 | Δx, Δy, Δz of the tool point | m per step |
| 3–5 | rotation as an axis-angle vector | rad per step |
| 6 | gripper | < 0 closes (fingers to 0.0 m), ≥ 0 opens (0.04 m) |

Everything is in the robot base frame, at the tool point 0.107 m along `panda_hand`'s z axis. Each
action is added to the current tool pose, and Isaac Lab's damped least-squares IK turns the result
into joint targets on every physics step. `clip_actions` is unset. The 35 observations are the
teacher's 36 with a 7-entry last action in place of the 8-entry one.

`Lift-Cube-Franka-Twist-TeacherGains-v0` is the same task on the teacher's arm drives (stiffness 80,
gravity on). It is kept for comparison.

### Why the conversion is a torque match

The teacher's joint targets are not poses. With the teacher acting, they sit 2–8 rad (norm) from
the joints, far past the joint limits. Any error beyond `effort / stiffness` (1.09 rad on joints 1–4,
0.15 rad on joints 5–7) just means full torque. The teacher has learned to command torque through
saturated position drives, so converting its targets geometrically fails:

| Teacher driving every step, label conversion | IK-Rel arm | Teacher's arm |
|---|---|---|
| `J(q)·(q_target − q)`, `model_1550` | ~1% | 7% |
| `J(q)·(q_target − q)`, original pretrained (299) | 48% | 68% |
| exact pose of `FK(q_target)`, either teacher | 0% | 0% |
| torque match without gravity, `model_1550` | 68% | — |
| **full torque match, `model_1550`** | **75%** | **89%** |

The full torque match gives the student's joint error at which its own drive applies the teacher
arm's net torque:

```
e     = q_default + 0.5 · a_teacher[:7] − q                 teacher joint error
τ     = clip(K_t·e − D_t·q̇, ±effort_t)  − g(q)              − g(q) only if the student arm is weightless
e_s   = (τ + D_s·q̇) / K_s                                   student error, same torque
label = [ J_tool(q)·e_s / 0.5 ,  sign(a_teacher[7]) ]       raw IK-Rel action, binary gripper
```

IK-Rel maps a delta pose back through J⁺, so `J·e_s` comes back as the task-space part of `e_s`.
The 7-joint arm's one null-space direction is lost. `g(q)` is read from the simulator
(`gravity_compensation_forces`, about 22 N·m), and `J_tool` is the IK action's own Jacobian at its
tool offset. The teacher sees the student's observation with its own previous action in place of the
student's, zeroed at reset.

**Before trusting a new teacher, arm or label, run a teacher-only check:**
`--beta_iters 100000 --iterations 40` makes the teacher drive every step. A low success rate there
means DAgger has nothing good to imitate, and nothing will raise an error to warn you.

## 4. DAgger

`scripts/train_dagger_twist.py`:

- **Mixing.** At each step the arm runs the teacher's label with probability β, otherwise the
  student's mean action. β falls linearly from 1 to 0 over `--beta_iters` (50) iterations.
- **Aggregated buffer.** Every (observation, label) pair is kept in a 1M-sample buffer, and each
  iteration does 50 Adam steps (batch 8192, lr 1e-3) on it. The loss is a per-dimension normalised
  MSE, so the six small arm entries are not drowned by the ±1 gripper.
- **Critic.** The critic is fitted on the rollouts' λ-returns (γ 0.98, λ 0.95), so a PPO resume does
  not start from a random value function.
- **Exploration spread.** The saved exploration spread is `--student_std` 0.05. A raw action of 1
  asks for a 0.5 m step, so the teacher's spread of 1.0 would throw the arm around.
- **Output.** Checkpoints are standard RSL-RL `model_<it>.pt` files in
  `logs/rsl_rl/lift_cube_franka_twist/<run_name>/`, written every 50 iterations and at the last
  iteration. `dagger.json` records the teacher and every setting.

```bash
python scripts/train_dagger_twist.py --run_name dagger_v1 \
  --teacher logs/rsl_rl/lift_cube_franka/pretrained_motion_cost/model_1550.pt
```

`dagger_v1` took 300 iterations, about 3 minutes on the RTX 4090 laptop GPU. The student's network
is new and randomly initialised; only the teacher's labels come from `model_1550`. With β = 0 (from
iteration 50 on), its stochastic training success settled at about 50–60%. Played deterministically
it scored **8/10** within 10 cm of the goal (7/10 within 5 cm).

## 5. PPO resume

```bash
python scripts/train.py --rl_library rsl_rl --task Lift-Cube-Franka-Twist-v0 --num_envs 1024 --seed 42 \
  --resume --load_run dagger_v1 --checkpoint model_299.pt \
  agent.run_name=twist_ppo agent.max_iterations=1500
```

`dagger_v1/model_299.pt` is the student after DAgger's last iteration (0–299). It is not the
pretrained teacher's `pretrained_isaac_v0/model_299.pt`: `--load_run` selects the folder, and the
task selects the experiment (`lift_cube_franka_twist`). The resume loads the actor, the fitted
critic and the 0.05 spread, and counts on from iteration 299, so 1500 iterations end at
`model_1798.pt`. An existing, non-empty run folder of the same name is refused.

`twist_ppo` with the −0.2 motion costs: success rose from 0.32 over its first 20 iterations to 0.98
over its last 50, and the return from 20 to 125. `Episode_Reward/action_rate` went from −0.03 to −0.10
and `joint_vel` from −1.2 to −1.0.

**The twist is a push command, not a velocity the arm reaches.** Like its teacher, the student
asks for large deltas (about 0.34 raw, or 17 cm per step, during DAgger), and the drive limits set
the real motion: tool speed RMS 0.5–0.9 m/s, about 1–2 cm per step. The `--max-linear-speed` and
`--max-angular-speed` limits in `play_franka_twist.py` apply only at evaluation, not in training.

## 6. Evaluation

To watch 16 environments:

```bash
PYOPENGL_PLATFORM=glx python scripts/play.py --rl_library rsl_rl --task Lift-Cube-Franka-Twist-v0 \
  --num_envs 16 --viz newton_gl --real-time \
  --checkpoint $(pwd)/logs/rsl_rl/lift_cube_franka_twist/twist_ppo/model_1798.pt
```

Use this project's `scripts/play.py`, not Isaac Lab's: the twist tasks are registered by the
`sheet_rl` package, which Isaac Lab's script does not import.

To measure, use `scripts/play_franka_twist.py`. It plays one environment for a fixed number of
seeded episodes and writes a JSON file with per-episode success (10 cm and 5 cm), distances, tool
speed and acceleration, and a trace of every 5th step, to `evaluations/franka_twist/`
(or `--output`). It has three modes:

| `--control` | Policy | Task |
|---|---|---|
| `joint` | joint checkpoint, unchanged | `IsaacContrib-Lift-Cube-Franka` |
| `twist` | joint checkpoint, its targets converted to a twist and back through IK at run time | `IsaacContrib-Lift-Cube-Franka` |
| `student` | a twist checkpoint (DAgger or PPO); without `--checkpoint`, the newest one | `Lift-Cube-Franka-Twist-v0` |

```bash
PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control student \
  --checkpoint logs/rsl_rl/lift_cube_franka_twist/twist_ppo/model_1798.pt --episodes 20 --real-time
```

`--viz none` runs it headless.

### Cube on a 5 cm platform

`--platform <height>` (with `--platform_size`, default 0.10 m) stands the cube on a square box in
any mode. It uses the same platform code as the platform task in section 7: a kinematic box moved
under the cube by a reset event that runs after the cube's own, with the cube raised by the same
height. In `Lift-Cube-Franka-Twist-Platform-v0` the option fixes the task's random height; in the
other tasks it adds a platform, and their rewards are unchanged. "Lifted" in the script's metrics is
always measured from what the cube stands on.

```bash
PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control student \
  --checkpoint logs/rsl_rl/lift_cube_franka_twist/twist_ppo/model_1798.pt \
  --platform 0.05 --episodes 20 --real-time
```

`twist_ppo/model_1798.pt`, 20 episodes, seed 42:

| | On the table | On a 5 cm platform |
|---|---|---|
| Success within 10 cm | **20/20** | **12/20** |
| Success within 5 cm | 17/20 | 10/20 |
| Mean closest approach to the goal | 2.9 cm | 15.0 cm |

The platform sat exactly under the cube in every episode (0.0 mm offset, top at 5.0 cm). The policy
did not adapt to the higher cube:

- It reaches for table height. In most episodes the fingers clip the cube and knock it off the
  platform (it slides 13–31 cm).
- 6 of the 12 successes are pick-ups from the table after the cube fell. Only 2–3 grasps closed on
  the cube while it was still on the platform.
- 6 of the 8 failures: the cube fell and was not recovered before the 5 s timeout.
- 2 of the 8 failures: the hand stopped 7.5 cm away, below the platform top, beside it.

The cube's height is an observation, but it was always about 2.4 cm in training, so the policy
learned to ignore it. Section 7 trains it on platforms of random height.

## 7. Platform training with a slower hand

`Lift-Cube-Franka-Twist-Platform-v0` (`FrankaCubeLiftTwistPlatformEnvCfg` in `twist_env_cfg.py`,
terms in `platform_mdp.py`) is the twist task with three changes:

- **A platform of random height.** At every reset the cube stands on a 10 cm-square platform whose
  top is drawn uniformly from **1.5–15 cm**. A box cannot change size at run time, so the platform is
  one 20 cm-tall kinematic box. The reset event sinks it into the table until its top is at the drawn
  height, moves it under wherever the cube's own reset put the cube, and raises the cube by that
  height. A kinematic box does not collide with the static table, so the buried part has no effect.
- **Lift measured from the support.** The original lift test is z > 0.04 m in absolute terms. On
  anything taller than 1.6 cm that is true of a cube just standing on the platform, which would pay
  the lift (15) and goal terms for nothing. `lifting_object` and both goal-tracking terms (and so
  `Metrics/success_rate`) now require the cube 4 cm above the platform top, or above the table if it
  has been knocked off.
- **A slower hand.** Two new charges on the hand link's linear speed `v`. The hand link is used, not
  the tool point, so a wrist turn is not billed as travel.

| Term | Value | Weight | Example charges |
|---|---|---|---|
| `ee_speed` | `max(0, v − 0.2 m/s)²`, everywhere | −10 | 0.5 m/s → 0.9 per second, 1 m/s → 6.4 per second |
| `approach_speed` | `v² · (1 − tanh(d_ee / 0.1))`, until the cube is lifted | −40 | 0.5 m/s 1 cm from the cube → 9 per second, 0.1 m/s → 0.4 per second |

`approach_speed` uses the reach kernel, so it grows as the fingers close in on the cube and covers
the grasp. It switches off once the cube is lifted, so carrying the cube is charged only by
`ee_speed`. For scale, lifting pays 15 per second and reaching up to 1. The `action_rate` and
`joint_vel` costs stay at −0.2. All weights can be changed with Hydra overrides such as
`env.rewards.approach_speed.weight=-80`.

The observations (35) and actions (7) are unchanged, so the twist checkpoint resumes directly:

```bash
python scripts/train.py --rl_library rsl_rl --task Lift-Cube-Franka-Twist-Platform-v0 --num_envs 1024 --seed 42 \
  --resume --load_run twist_ppo --checkpoint model_1798.pt \
  agent.run_name=twist_platform_slow agent.max_iterations=1500
```

A 40-iteration test of this resume started at a success rate of about 0.3–0.5, compared with 0.98 on
the table. Both speed charges came in at about −2 per second each, and `lifting_object` at 4–5 rather
than the 15 a free lift would pay. Before any platform training, `model_1798.pt` succeeded in 4 of 10
random-height episodes in `play_franka_twist.py`, mostly on the lower platforms.

To evaluate on random heights, or on one fixed height:

```bash
PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control student --task Lift-Cube-Franka-Twist-Platform-v0 \
  --checkpoint logs/rsl_rl/lift_cube_franka_twist/twist_platform_slow/<model>.pt --episodes 20 --real-time
# add --platform 0.05 (or any height) to fix the height
```

Each episode's JSON records `support_height_m`, so success can be compared against platform height.
Check `ee_speed_rms_m_s` against the 0.47 m/s of `model_1798.pt` on the table to see whether the
hand has slowed.

`twist_platform_slow` was stopped at `model_3200.pt`. That checkpoint picks from the platform in
nearly every episode, at a hand speed RMS of 0.24–0.33 m/s.

## 8. Pick and place: a second platform

`Lift-Cube-Franka-Twist-PickPlace-v0` (`FrankaCubeLiftTwistPickPlaceEnvCfg`; terms in
`pick_place_mdp.py`, goal command in `pick_place_command.py`) adds a second phase to the platform task.

- **Phase 1** is the platform task unchanged: lift the cube off its platform (1.5–15 cm) and carry it
  to the goal in the air.
- **The switch.** When the lifted cube comes within 10 cm of that goal (the task's success radius),
  the goal command switches, for the rest of the episode, to the cube resting on the centre of a
  second platform: its top plus 2.4 cm, half the cube's side.
- **Phase 2** is to set the cube down there and let go.
- **The second platform** (green) is 10 cm square, with its own height drawn from 1.5–15 cm. Its
  centre is at x 0.4–0.6 m and |y| 0.15–0.30 m, on the opposite side of the robot's x axis from the
  cube, so the two platforms cannot overlap.
- **Observations and actions are unchanged.** The goal is the same observation, so the policy sees
  the switch as its target dropping onto the second platform. Platform-task checkpoints resume
  directly.
- **Episodes are 10 s** instead of 5, to fit both phases. The command never resamples mid-episode,
  since a resample would undo the switch.

**Reward.** The reward is staged, so finishing phase 1 never costs anything. Once the goal has
switched, `lifting_object` and both phase-1 goal-tracking terms pay their maximum (15 + 16 + 5 per
second). Phase 2 adds:

| Term | Value | Weight |
|---|---|---|
| `place_tracking` | `1 − tanh(d / 0.15)`, cube to placement pose | 16 |
| `place_tracking_fine_grained` | `1 − tanh(d / 0.05)` | 5 |
| `place_height` | `1 − tanh(\|dz\| / 0.05)`, height error alone | 8 |
| `release_shaping` | `(1 − tanh(d / 0.03)) · (clip(a_gripper, −6, 0) + 6) / 6` | 10 |
| `placed` | cube within 2.5 cm of the placement pose and slower than 0.05 m/s | 10 |
| `released` | placed, with the gripper commanded open (raw action ≥ 0) | 10 |
| `place_bonus` | once per episode, the first step placed and released | +300 (literal) |
| `yaw_rate` | the hand's yaw rate squared, about the vertical | −2 |

`approach_speed` is the phase-1 approach charge only. `Metrics/success_rate` is now the pick (phase 1
reached). `Metrics/object_pose/pick_success` repeats it, and `Metrics/object_pose/place_success` is
the share of episodes in which the cube rested at the placement pose with the gripper open.

**Why the phase-2 terms look like this.** The first run (`twist_pick_place`, then
`twist_pick_place_v2`) reached place success of 2–4% early on, then fell to 0. In phase 2 the
policy held the cube at z ≥ 0.19 m, 8–13 cm above the placement pose. Three things kept it there:

- **The gripper never sampled "open".** Its raw action sat at −2.7 with a spread of 0.5, so `released`
  and the bonus were never found.
- **No pull downward.** Phase-1 goals are always 25–50 cm high, so the policy had never carried the
  cube low. The coarse term at std 0.3 paid little for the last 10 cm, and the damped IK-Rel arm
  follows only about 5% of small commands, so exploration barely moved the hand.
- **A charge that rewarded hovering.** The first version of `approach_speed` charged phase-2 hand
  speed in proportion to nearness to the placement pose. Moving near the target cost more than
  moving far from it, and the policy hovered 12–17 cm out, where the charge fades.

The changes: `place_height`, the coarse std cut to 0.15, `approach_speed` limited to phase 1, and
`release_shaping`, a slope toward opening near the target. In 450-iteration tests from
`twist_platform_slow/model_3200.pt`, the cube came down to the platform: `place_height` rose from
about 1 to 2.5–5 per second, and `placed` from 0 to as high as 0.9. **Release was not learned in any
test:**

- **Gripper spread 3.0** (via `reset_policy_std.py`): chance openings drop the cube, and pick success
  fell to 0.57.
- **Gripper spread 1.5:** chance openings in transit taught the policy to push the action below −3.
- **The original spread (0.5)** with the ramp down to −6 (used now): the action never reached 0.

The ramp is linear because `(1 + tanh(a)) / 2` was flat at −2.7 (slope 0.009).

```bash
python scripts/train.py --rl_library rsl_rl --task Lift-Cube-Franka-Twist-PickPlace-v0 --num_envs 1024 --seed 42 \
  --resume --load_run '^twist_platform_slow$' --checkpoint model_3200.pt \
  agent.run_name=twist_pick_place_v3 agent.max_iterations=3000
```

`--load_run` is a regular expression, matched against the run folders, and the last match is used.
Anchor it with `^…$` whenever another run's name starts with the same text. Otherwise a new empty
folder such as `<name>_v2` can be the match, and the resume fails with "No checkpoints".

**Starting point.** In a 60-iteration test of this resume, `pick_success` was 0.96–1.0 from the start,
so the switch fires. `place_tracking` reached 8–9 of 16 per second, and the first released
placements appeared after about 35 iterations, in 2–8% of episodes. Evaluated after those 60
iterations, the policy picked in all 5 episodes within 0.7–1.1 s. It then carried the cube toward the
second platform but held it in the air 12–17 cm short, without placing it.

**What to watch.** The phase-1 maximum keeps paying while the cube is held in the air, so a policy
that never sets the cube down still earns most of the reward. Watch `place_success` and
`Episode_Reward/release_shaping` rather than the return. If release does not appear, the next
step is to take release off the policy: open the gripper automatically once the cube rests at the
placement pose.

`play_franka_twist.py` reports `pick_success`, `place_success`, `placed_at_end`, the step of the pick
and the closest approach to the placement pose. `--platform <h>` fixes both platforms to height h.

```bash
PYOPENGL_PLATFORM=glx python scripts/play_franka_twist.py --control student --task Lift-Cube-Franka-Twist-PickPlace-v0 \
  --checkpoint logs/rsl_rl/lift_cube_franka_twist/twist_pick_place/<model>.pt --episodes 20 --real-time
```

## Files

| Path | What it is |
|---|---|
| `scripts/train_dagger_twist.py` | DAgger trainer: teacher-to-student conversion, aggregated buffer, critic fit |
| `scripts/play_franka_twist.py` | evaluation: `joint` / `twist` / `student` modes, `--platform`, JSON output |
| `scripts/franka_twist_math.py` | Jacobian shift, twist limits and resolved-rate IK used by the `twist` mode |
| `source/.../franka_lift_twist/__init__.py` | registers `Lift-Cube-Franka-Twist-v0`, `-TeacherGains-v0`, `-Platform-v0` and `-PickPlace-v0` |
| `source/.../franka_lift_twist/twist_env_cfg.py` | the twist env configs: −0.2 motion costs, the teacher-gains arm, the platform and pick-and-place tasks |
| `source/.../franka_lift_twist/platform_mdp.py` | the platform and its reset event, support-relative lift and goal terms, hand-speed charges |
| `source/.../franka_lift_twist/pick_place_mdp.py` | second platform, staged pick-and-place rewards, the goal command's configuration |
| `source/.../franka_lift_twist/pick_place_command.py` | the goal command that switches to the second platform (loaded by name, after the simulator starts) |
| `source/.../franka_lift_twist/twist_ppo_cfg.py` | PPO runner: the teacher's settings, experiment `lift_cube_franka_twist`, spread 0.05 |
| `IsaacLab/.../contrib/lift/config/franka/joint_pos_env_cfg.py` | (Isaac Lab checkout) teacher motion costs at −0.1 from the start |

| Run folder (`logs/rsl_rl/`) | Contents |
|---|---|
| `lift_cube_franka/pretrained_isaac_v0` | unchanged copy of the cached pretrained checkpoint, as `model_299.pt` |
| `lift_cube_franka/pretrained_motion_cost` | the joint teacher with −0.1 motion costs; `model_1550.pt` is the DAgger teacher |
| `lift_cube_franka/cmp_scratch_10it`, `cmp_pretrained_10it` | the 10-iteration scratch vs pretrained comparison |
| `lift_cube_franka_twist/dagger_v1` | the DAgger student (`model_299.pt` = final) |
| `lift_cube_franka_twist/twist_ppo` | the student after PPO with −0.2 motion costs (`model_1798.pt` = final) |
| `lift_cube_franka_twist/twist_platform_slow` | platform task with the slower hand, from `twist_ppo` (stopped at `model_3200.pt`) |
