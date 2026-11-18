# Handoff: sheet_rl workspace, state on 2026-10-27 (about 13:00); real-world side added 2026-11-17 (§9)

**Real-world implementation continuation (2026-11-17 evening):** see
[`HANDOFF_REAL_ENV_2026-11-17.md`](HANDOFF_REAL_ENV_2026-11-17.md). The real environment and optional guarded
sender are now installed in `hrii_sheet_mocap`; 91 tests pass, including isolated ROS2 synthetic inputs.
Start observations with `/home/hami/workspaces/deformable_isaac_ws/real_env.sh` (default observe, no inference).
Live command execution and physical tracking have not been commissioned.
The user selected `best_twist_slow_3950`, 50 Hz, scale 0.03, manual positioning/history reset, and explicitly
prohibited moving the robot. No robot or gripper commands were sent. The continuation records important
discoveries about the actual Cartesian and gripper interfaces.

Workspace root: `/home/hami/workspaces/deformable_isaac_ws/sheet_rl/sheet_rl`. Every path below is relative
to it unless it starts with `/` or `~`. This file replaces `logs/fast_sac/SESSION_HANDOFF_2026-10-26.md` as the
status document. The older file is still the best account of the scale-randomisation experiments.

## 0. Status in brief

- **Real world (2026-11-17), see §9.**
  - **Sheet detection on OptiTrack:** done, including the missing-marker fill and re-finding after dropouts.
  - **Platform on rigid body 2:** set up.
  - **UR16e:** its ROS 1 topics are relayed one way into ROS 2, and it is calibrated to OptiTrack (frame editor and bar editor).
  - **Policy:** exported to ONNX; its 49 observations are built from real measurements.
  - **Not built:** executing actions on the robot.

- **Nothing is running.** The last training run (the UR10e PPO fine-tune) was stopped by the user at iteration
  961 of 1000 at about 23:55 on 2026-10-27, and the GPU is idle.
- **Task.** A Franka (or UR10e) pulls a cloth sheet out of a slot and places it on a platform. Success
  (`placed`) means releasing it with at least 50 % platform coverage and at most 20° yaw. Physics is Newton:
  cloth by VBD, the robot by MJWarp. Control runs at 50 Hz, an episode is about 95–110 steps, and the action is
  a 6-D relative-pose IK command plus the gripper.
- **Franka, current actor (honest observations):**
  `logs/fast_sac/sheet_platform_twist_dagger/2026-10-26_23-17-21_repair49_rand_std/model_200.pt`. Under full
  randomisation it places 0.677 stochastic and 0.680 deterministic over 512 episodes.
- **Franka, strongest older policy:**
  `logs/fast_sac/sheet_platform_twist_sac/2026-10-26_16-47-05_par_I_yaw45_fixed/model_93500.pt`, at 0.699 and
  0.717. It must be run on the *legacy* observation group (§3).
- **UR10e, kept policy:** `logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_ur10e_ppo/model_925.pt`
  (PROVENANCE.md inside). The Franka student `repaired_student_D2/model_4000` fine-tuned on the UR with PPO
  (no KL bounds, learned std from 0.45, 10× learning rates). **About 80 % placed** over training windows of
  about 9000 episodes; deterministic `play.py` check 30/32 releases placed. Untransferred, the student placed
  0/50 on the UR. See §5, "PPO fine-tune".
- **Kept (2026-10-30): best slow twist:**
  `logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt` (PROVENANCE.md inside).
  model_1924 trained through joint limits of 80 % and then 60 %, plus a tool-speed penalty above 0.4 m/s. 81.9 %
  placed, tool penalty −639 per episode. At scale 0.03 the tool stays under 0.4 m/s and an episode takes about
  4.3 s (model_1924: 1.9 s).
- **Also kept (2026-10-29): best twist, joint-speed limited:**
  `logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_velterm_1924/model_1924.pt` (PROVENANCE.md inside).
  model_925 trained on with the arm joint-speed limit on and scale 0.015–0.1; last window 79.3 % placed. Runs
  unchanged on the UR10e and the UR16e (`Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0`, added 2026-10-28).
  **On the Franka it needs the tool point moved 13.2 mm toward the flange (0.107 → 0.0938 m)**; play it there with
  `scripts/play_actor.py`.
- **Git:** the last commit is `d3158e09` (2026-09-06). Nothing since then is committed: `git status` lists 56
  modified or untracked paths, including all the SAC, DAgger and UR10e work. Commit only when the user asks.

## 1. Working rules from the user (keep them)

- **Run every script inside the `active_isaac` venv.** It is
  `cd ~/workspaces/deformable_isaac_ws/IsaacLab && source env_isaaclab/bin/activate && cd ../sheet_rl/sheet_rl`.
  It provides Python 3.12, torch 2.11 and rsl-rl-lib 5.4.1, with `sheet_rl` installed in editable mode.
  - Run tests with `python -m unittest discover -s tests`; there is no pytest.
  - Result on 2026-10-27: **186 tests OK in 26 s.**
- **Use the Newton GL viewer, not Kit or Isaac Sim:**
  `PYOPENGL_PLATFORM=glx python scripts/<script>.py ... --viz newton_gl`. Add
  `env.scene.table.spawn.visible=True`, otherwise the table collider isn't drawn.
- **Approvals and commands:**
  - The user approves each training phase personally.
  - Do not start or queue runs that weren't asked for.
  - When the user asks for "the command", give the command and don't run it.
- **Evaluation:**
  - Evaluate stochastically by default, and report deterministic results next to stochastic ones.
  - Quick checks use **at most 10 episodes** unless the user asks for more. The user pushed back on 100.
  - The user wants tables for results.
- **Honesty:** be upfront about mistakes and their cost. The channel-49 bug (§3) cost a DAgger round, and the
  user was angry about it.
- **Killing processes:** kill by PID, or use `pgrep -f "[t]rain_sac.py"`. A bare `pkill -f <pattern>` matches
  your own shell and kills it.
- **Persistent notes** for Claude sessions are in `~/.claude/projects/-home-hami-workspaces-deformable-isaac-ws-sheet-rl/memory/`.
  Start with `MEMORY.md`. The most relevant notes are `sac-student-pipeline.md` (the full experiment
  history) and `ur10e-transfer-findings.md`.

## 2. Tasks and how to configure them

| Gym id | What it is |
|---|---|
| `Template-Sheet-Rl-Platform-TwistStudent-v0` | **The main task.** Franka with the IK twist action. The FastSAC config is `sheet_twist/sheet_twist_sac_cfg.py` (`fast_sac_cfg_entry_point`). |
| `Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0` | The same task on a UR10e with a Robotiq 2F-85. The config is `sheet_twist/sheet_ur10e_env_cfg.py`, a subclass of the Franka student config (§5). |
| `Template-Sheet-Rl-Platform-Twist-v0`, `-Twist-Rigid-v0`, `-Twist-OrigReward-v0`, `Template-Sheet-Rl-v0`, `Template-Sheet-Rl-Hand-v0`, `Lift-Cube-Franka-Twist-*` | Earlier phases. See `README.md`, `PLATFORM_TASK.md` and `FRANKA_LIFT_TWIST.md`. |

The task code is in `source/sheet_rl/sheet_rl/tasks/manager_based/sheet_twist/`, and the shared terms are in
`.../sheet_rl/mdp/`.

**Observation groups on the main task:**

- **`policy` (59):** the PPO teacher's observations. The SAC critics read it plus the action.
- **`action_history` (49):** the actor's observations.
  - The last 2 actions (14).
  - 9 sheet landmarks measured from the tool point (27). The axes are the robot frame and quaternions are xyzw.
  - The goal, measured from the tool (7).
  - **Slot 49 (`finger_width`): the fingertip pad distance in metres**, from the `gripper_opening` term in
    `mdp/observations.py`.
- **`action_history_legacy` (49):** identical, except that slot 49 holds the **old buggy value**, the sum of all
  nine Franka joint angles (0.2–0.6).
  - Background: the original term had no `params`, so its default `SceneEntityCfg` was never resolved.
  - This was found and repaired on 2026-10-26.
  - Every actor trained before the repair depends on the buggy value. With the true width in slot 49 it
    placed 0/3. Run those actors with `--actor_obs_groups action_history_legacy`.
  - The UR10e task has no legacy group, so pre-repair actors cannot run on the UR.

**Randomisation is set with Hydra overrides.** All of it is off by default. The default IK scale is 0.1.

| Override | Meaning |
|---|---|
| `env.events.randomize_arm_scale.params.scale_range=[0.04,0.1]` | IK action scale drawn per episode |
| `env.actions.arm_action.scale=0.05` | a fixed IK scale instead |
| `env.events.reset_layout.params.sheet_xy_jitter=0.05` | sheet and slot position, ±5 cm |
| `env.events.reset_layout.params.sheet_yaw_jitter=0.7854` | sheet and slot yaw, ±45° (in radians) |

"Full randomisation" is the scale range together with ±5 cm and ±45°. Every run since `par_I` used it. The actor
cannot observe the IK scale.

## 3. Weights

All FastSAC files are complete checkpoints. Each contains:
- the actor and its frozen observation normaliser;
- the twin C51 critics, their targets and the critic normaliser;
- alpha and the optimisers;
- the anchor actor, when one was used.

Checkpoints are saved every 1000 iterations, so stopping a run loses up to 999 iterations. The repaired-student
fine-tune lost iterations 6001–7000 this way.

### 3a. The ones that matter

| Purpose | Checkpoint | Observation group | Evidence |
|---|---|---|---|
| PPO teacher (fixed IK scale 0.1, nominal layout) | `logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt` (PROVENANCE.md inside) | `policy` | 512 episodes, seed 42: det 0.871, stoch 0.738; extraction 0.99 |
| First FastSAC student, by DAgger, tool-relative | `logs/fast_sac/sheet_platform_twist_dagger/2026-10-23_22-12-52_dagger_v3_tool/model_200.pt` | legacy | stoch 0.850, det 0.859 |
| Best fixed-scale policy | `logs/fast_sac/sheet_platform_twist_sac/best_of_sac/model_66500.pt` (PROVENANCE.md inside) | legacy | stoch 0.893, det 0.896 at scale 0.1. Under random scale it places 0.52 (0.05 → 5 %, 0.1 → 89 %). |
| Scale-robust policy, nominal layout | `logs/fast_sac/sheet_platform_twist_sac/2026-10-26_10-08-12_par_D2_noanchor_9k/model_75500.pt` | legacy | Probes: 0.05 → 96.5 %, 0.1 → 81.5 %. Random scale, 10 episodes: det 10/10, stoch 9/10. Sheet jitter of ±5 cm and ±90° breaks it (3–4/10). |
| ±15° yaw policy (the best checkpoint before the curriculum widened to ±30°) | `.../2026-10-26_15-24-10_par_G_yawcurriculum/model_81000.pt` | legacy | Probe just before it, at 80500: 0.05 → 0.857, 0.1 → 0.517 |
| **Full-randomisation policy (±45°), the repair teacher** | `.../2026-10-26_16-47-05_par_I_yaw45_fixed/model_93500.pt` | **legacy** | Probes: 0.05 → 0.715, 0.1 → 0.448. 512 episodes, seed 7: stoch 0.699, det 0.717. |
| **Repaired student (honest slot 49): current actor** | **`logs/fast_sac/sheet_platform_twist_dagger/2026-10-26_23-17-21_repair49_rand_std/model_200.pt`** | `action_history` | 512 episodes, seed 7, full randomisation. Stoch: 0.677 placed, 0.232 dropped, 0.072 held too low, return 2570. Det: 0.680 placed, 0.203 dropped, 0.096 held too low. One recorded Franka episode at scale 0.05 and yaw +30° placed. |
| Repaired student, fine-tuned with the D2 recipe (stopped) | `logs/fast_sac/sheet_platform_twist_sac/2026-10-27_11-50-01_repaired_student_D2/model_{1000..6000}.pt` | `action_history` | Results below. |
| **UR10e policy: PPO fine-tune of D2's model_4000** | **`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_ur10e_ppo/model_925.pt`** (PROVENANCE.md inside; rsl_rl checkpoint, load with `agent.actor.init_std=0.45`) | `action_history` (critic `ur_critic`) | Training windows 875–950: 79.8–80.4 % placed, 86 % extracted, return about 4100. `play.py` deterministic: 30/32 releases placed, coverage 0.88. Fails below scale 0.03; does not work on the Franka. |
| **Twist policy, joint-speed limited: model_925 trained on** | **`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_velterm_1924/model_1924.pt`** (PROVENANCE.md inside; load with `agent.actor.init_std=0.45`) | `action_history` (critic `ur_critic`) | Last window 1901–1924: 79.3 % placed, 4.1 % ended by the speed limit. 10 deterministic episodes, seed 42, scale 0.03: 8 placed on the UR10e and identically on the UR16e. **Franka: needs the tool point moved 13.2 mm toward the flange (0.0938 m), via `scripts/play_actor.py`**; at the default 0.107 m it mostly drops. |
| **Slow twist policy: model_1924 trained on with the 60 % joint limit and the tool-speed penalty** | **`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt`** (PROVENANCE.md inside; load with any `agent.actor.init_std`) | `action_history` (critic `ur_critic`) | 25 iterations around 3950: 81.9 % placed, 7.7 % ended by the 60 % joint limit, tool-speed penalty −639 per episode. Same-recipe spot check (model_3375, UR16e, nominal, scale 0.03): placed in 4.30 s, tool peak 0.34 m/s (model_1924: 1.92 s, 0.87 m/s). |

The fine-tune of the repaired student gave no clear gain. Probe success rates by checkpoint:

| | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| IK scale 0.05 | 0.668 | 0.638 | 0.644 | **0.720** | 0.649 | 0.658 |
| IK scale 0.1 | 0.375 | **0.444** | 0.409 | 0.372 | 0.348 | 0.412 |

- **Training-time placed rate:** peaked at 0.70–0.72 around iterations 3000–3500. It fell to 0.58 by 6000 and
  was 0.62 at 7000.
- **Actor exploiting the critic:** from 3500 to 6000, `qf_max` rose from 63 to 66.4 (the support maximum is 70)
  and entropy fell from −4.3 to −9.3. Both partly recovered by 7000 (64.3 and −2.6).
- **Checkpoint choice:** `model_4000.pt` is the best at 0.05. No 512-episode evaluation of it has been run.

**How to read the numbers:**
- **Probes:** one episode per environment at a fixed IK scale, with sheet-pose randomisation. With 1024
  environments that is about 1024 episodes. An episode still running at the 400-step cap counts as a failure.
- **512-episode evaluations:** use the random scale range. They are not directly comparable with probes.
- **Noise:** Newton cloth is non-deterministic. The same seed and the same actions flip 17–19 % of outcomes, so
  expect ±3–4 points of noise even at 1024 episodes. 10-episode checks are very coarse.
- **Lost logs:** the 512-episode evaluation logs and all videos from 2026-10-26/27 were in `/tmp` and were lost
  on reboot. The numbers above come from the session transcript. The saved evaluation JSONs are in
  `evaluations/teacher_student/`.

### 3b. Lineage of the current actors

- **Teacher and first student:** the PPO teacher `model_2874` led, by DAgger, to `dagger_v3_tool/model_200`.
  Critic warm-up on the stochastic teacher came first: `*_teacher_cwarm_v1_resume/model_5000`, held-out EV 0.38,
  where the measured ceiling is about 0.40.
- **Fixed-scale fine-tune:** critic recalibration on the student (`*_student_recal_v2/model_6500`), then
  single-environment SAC with an anchor (`single_env_sac_v4_anchor`). The result was copied to
  `best_of_sac/model_66500`.
- **Scale randomisation:** `par_D2_noanchor_9k/model_75500`, trained with 1024 environments, no anchor and the
  scale randomised.
- **Yaw randomisation:**
  - `par_G_yawcurriculum/model_84500` (yaw widened from ±15° to ±30°);
  - then `par_H_yawcurriculum_cont/model_85000`;
  - then `par_I_yaw45_fixed/model_93500` (fixed ±45°, full randomisation).
- **Channel-49 repair**, by DAgger with `model_93500` as a **stochastic** teacher on the legacy group:
  - `repair49_from93500/model_200`: nominal layout only, an omission of mine;
  - then `repair49_rand/model_400`: full randomisation, β 1 / 0.5 / 0 / 0, 100 iterations per round;
  - then `repair49_rand_std/model_200`: β 0 / 0 with a log-std term. This is **the repaired student**.
- **Last run:** a fine-tune of the repaired student, `repaired_student_D2`.

"D2 recipe" means `par_D2_noanchor_9k`'s settings:
- 1024 environments and no anchor (`agent.anchor_coef=0.0`);
- pessimistic Q (`actor_q_min`, `target_q_min`);
- learning rates: actor 2e-5, critic 4e-5, alpha 1e-4;
- `policy_frequency=4`, `target_entropy_ratio=0.73`;
- a frozen actor normaliser.

### 3c. Superseded or failed runs (don't use)

All are under `logs/fast_sac/sheet_platform_twist_sac/` unless the row says otherwise.

| Run(s) | Outcome |
|---|---|
| `single_env_sac_v1`, `single_env_sac_v3_pessimistic` | Collapsed within about 1000 updates: no anchor. v1 also had an lr-restore bug, since fixed. |
| `scale_rand_v1`, `scale_rand_v2`, `scale_rand_test`, `scale_rand_noanchor_test`, `scale_rand_v3_noanchor_200k` | Single environment: flat with an anchor, collapsed without one. |
| `par_A_anchor1`, `par_B_anchor0p1`, `par_C_trust500` | 1024 environments with an anchor: flat at 0.52. |
| `par_D_noanchor` | Recovered and climbed after an early crash, but only ran 3000 iterations. Superseded by D2. |
| `par_E_posejitter_9k` | ±90° yaw with no curriculum: collapsed into a degenerate attractor (the episode ends in 3 steps). |
| `par_F_posejitter15deg` | ±15° fixed: a short run, replaced by G. |
| `par_H_yawcurriculum_cont` | One checkpoint (85000), only the bridge to I. |
| `logs/fast_sac/sheet_platform_twist_dagger/dagger_v1`, `dagger_v2` | Base-frame observations: plateaued at 0.52. |

### 3d. Older phases (PPO, August to October)

These are in `logs/rsl_rl/`. Each has its own notes in the run folders, in `README.md`, `PLATFORM_TASK.md` and
`FRANKA_LIFT_TWIST.md`.

- **Sheet twist student:** `sheet_platform_twist/hr0032_window1/model_348.pt`, from DAgger plus one PPO window.
  It is the root of the teacher's lineage.
- **Grasp reference policies:** `sheet_grasp/main_train_grasp/model_900_std06.pt` and `model_920`. They grasp
  reliably but are fast and not smooth.
- **Cube-lift joint teacher:** `lift_cube_franka/pretrained_motion_cost/model_1550.pt`.
- **Cube-lift twist runs:** `lift_cube_franka_twist/*`.
- **Hand drape traces:** `sheet_grasp/saved_trajectories/` (its README points to `controller_ref/`).
- **Older sheet grasp and placement policies:** `sheet_grasp/final_weights`, `final_on_arm` and
  `sheet_platform_grasp/placement_*`.

Disk use: `logs/rsl_rl` takes 4.5 GB and `logs/fast_sac` 2.2 GB.

## 4. Commands (copy-paste, inside `active_isaac`)

**Play a FastSAC actor in Newton GL.** It acts stochastically by default; add `--mean_action` for
deterministic actions.

```bash
PYOPENGL_PLATFORM=glx python scripts/play_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --checkpoint logs/fast_sac/sheet_platform_twist_dagger/2026-10-26_23-17-21_repair49_rand_std/model_200.pt \
  --viz newton_gl --lite --seed 42 env.scene.table.spawn.visible=True \
  env.events.randomize_arm_scale.params.scale_range=[0.04,0.1] \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

- For a pre-repair actor (`model_93500`, D2, `best_of_sac` and so on), also pass
  `--actor_obs_groups action_history_legacy`.
- For the UR10e, use `--task Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0` and add
  `env.terminations.joint_vel_out_of_limit=null` (see §5).
- Other `play_sac.py` options:
  - `--episodes N` and `--first_episode_only`;
  - `--obs_override` and `--true_finger_width` (diagnostics).

**Evaluate on 512 episodes.** This is how the numbers in §3a were produced: headless, run once stochastic and
once with `--mean_action`.

```bash
python scripts/play_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --checkpoint <ckpt> \
  --num_envs 512 --episodes 512 --seed 7 [--mean_action] [--actor_obs_groups action_history_legacy] \
  env.events.randomize_arm_scale.params.scale_range=[0.04,0.1] \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

To get the rates, tally the lines `[play_sac] env N: <outcome> return R length L`, one per finished episode.

- `scripts/eval_teacher_student.py` compares teacher and student in four modes and writes JSON to
  `evaluations/teacher_student/`.
- **Caution:** it uses the observation groups recorded in the checkpoint and has no override. It would feed a
  pre-repair actor the honest slot 49, so it evaluates those actors **wrongly** now.

**Fine-tune with the D2 recipe.** This is the exact command of the last run:

```bash
python scripts/train_sac.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --checkpoint logs/fast_sac/sheet_platform_twist_dagger/2026-10-26_23-17-21_repair49_rand_std/model_200.pt \
  --num_envs 1024 --max_iterations 9000 --seed 42 --run_name <name> \
  --probe_scales 0.05 0.1 --probe_interval 1000 \
  agent.actor_learning_rate=2e-5 agent.critic_learning_rate=4e-5 agent.alpha_learning_rate=1e-4 \
  agent.learning_starts=100 agent.target_entropy_ratio=0.73 \
  agent.actor_q_min=true agent.target_q_min=true agent.freeze_obs_normalizer=true agent.anchor_coef=0.0 \
  env.events.randomize_arm_scale.params.scale_range=[0.04,0.1] \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

- Other `train_sac.py` options: `--single_env`, `--actor_from`, `--teacher` (critic warm-up),
  `--reanchor_interval`, `--yaw_curriculum START STEP MAX` (with `--curriculum_threshold`,
  `--curriculum_windows` and `--curriculum_cooldown`), and `--probe_max_steps`.
- Logs go to `logs/fast_sac/sheet_platform_twist_sac/<date>_<name>/`, with TensorBoard events and `params/`.

**DAgger from a FastSAC actor teacher.** The channel-49 repair, first round. The arguments are those recorded in
the run's `dagger.json`; the three randomisation overrides were added on the command line.

```bash
python scripts/train_dagger_actor_teacher.py --task Template-Sheet-Rl-Platform-TwistStudent-v0 --num_envs 1024 \
  --teacher logs/fast_sac/sheet_platform_twist_sac/2026-10-26_16-47-05_par_I_yaw45_fixed/model_93500.pt \
  --teacher_obs_group action_history_legacy --init_from_teacher 1 --renorm_channels 48 \
  --betas 1.0 0.5 0.0 0.0 --iters_per_round 100 --run_name <name> \
  env.events.randomize_arm_scale.params.scale_range=[0.04,0.1] \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

- For the follow-up round, add `--init_student <previous model> --betas 0.0 0.0 --logstd_weight 1.0`.
- Loss: KL on the teacher's mean and log-std, plus MSE and an optional log-std term.
- Defaults: a 2M dataset, batch 8192, lr 1e-4.

**UR10e GUIs.** All use Newton GL.

```bash
python scripts/ur10e_start_pose_gui.py      # joint sliders; "Save as start pose" writes ur10e_start_pose.json
python scripts/ur10e_gripper_tuning_gui.py  # 2F-85 Close/Open, bounce metrics, Save edits ur10e_gripper.py (keeps .bak); --selftest
python scripts/ur10e_arm_tuning_gui.py      # same 0.5 s twist pulse to Franka and UR, 6-axis plots, UR drive sliders,
                                            # Save writes ur10e_arm_control.json; --viewer adds GL windows; --selftest
```

**Franka-versus-UR diagnostics:**
- `scripts/dump_reset_obs.py` prints the labelled reset observations.
- `scripts/compare_first_steps.py` shows the first steps from the same seed. Options:
  - `--replay other.json` drives the arm with recorded actions (open loop);
  - `--video --camera EX EY EZ LX LY LZ --clip_dir DIR` records a headless Newton GL video;
  - `--show_collision`, `--stochastic` and `--actor_obs_groups`.
- `scripts/ik_axis_response.py [--amplitude A]` measures the response to a step on each axis.

The arm GUI self-tests are saved in `logs/arm_tuning/selftest_amp{1,3}.{txt,png}`.

## 5. UR10e transfer: where it stands

### The task configuration

The files are `sheet_ur10e_env_cfg.py`, `ur10e_mdp.py`, `ur10e_gripper.py` and `ur10e_asset.py` in
`sheet_twist/`. Each change below is pinned by tests in `tests/test_ur10e_transfer.py`.

- **Inherited from the Franka task:** the same IK action (relative pose, inherited scale), rewards,
  terminations (`placed` and `dropped` included) and actor observations.
  - The tool point is `TOOL_POS` (0, 0, 0.145) on `wrist_3`, the pad midpoint to within 0.1 mm.
  - Slot 49 is the 2F-85 pad gap.
- **IK offset:** the IK body offset carries **position only**. Including the tool rotation had swapped the
  rotation axes.
- **Drives:** 400/80 with the **Franka's effort limits, 87/87/12 Nm**. This was the smallest-error setting for
  open-loop replay, and the user chose to keep it. It is not the real UR10e's torque limits.
  - The IK damping λ is 0.01.
  - All values are loaded from `ur10e_arm_control.json` through `load_arm_control()` and `arm_actuator()`.
- **Base and start posture:**
  - The base is at (−0.12, 0.02, 0).
  - The IK seeds from an elbow-up home posture, (−2.723, −1.415, −2.076, 2.516, −1.326, −2.829).
  - The table is extended back to x = −0.35 (`TABLE_BACK`) so the base sits on it. The success markers were
    resized to match.
- **Reset:** `reset_ur10e` draws exactly the Franka's reset random numbers, so a seed gives both arms the same
  layout.
  - With `start_pose="panda"`, the tool starts at the Franka's start pose.
  - The 49 observations then match the Franka's to 5e-7. The one exception is slot 49: 0.0800 against 0.0799 m.
    An `OPEN_ANGLE` of about 0.0791 rad would give 8.00 cm; it is a hand-tuned value and was not changed.
- **Critic observations:** the critic's `policy` group comes from a virtual Panda (`FrankaObservationAdapter`).
  It has a NaN guard that holds the last posture and warns once.

### Settings files the user changed with the GUIs on 2026-10-27

| File | Content |
|---|---|
| `sheet_twist/ur10e_start_pose.json` (10:47) | The user's own pose: joints (−3.089, −1.361, −2.365, −0.812, −4.581, 0.436); tool at (0.519, −0.113, 0.230) in the robot frame, about 3 cm above the sheet's top edge. |
| `sheet_twist/ur10e_gripper.py` (11:07, Save from the gripper GUI) | `KNUCKLE_DAMPING` changed from 0.125 to **0.375**. `ur10e_gripper.py.bak` holds the previous version. |
| `sheet_twist/ur10e_arm_control.json` (11:43) | Exactly the defaults listed above, so behaviour is unchanged. |

- **Start pose:**
  - Because the JSON exists, the task now uses `start_pose="home"`: each reset starts from the user's pose plus
    the Franka's random variation.
  - Reset observations therefore **no longer match the Franka's**.
  - Delete the JSON to return to the Franka-matched start.
- **Gripper:** the effect of the damping change has not been measured. Re-run
  `ur10e_gripper_tuning_gui.py --selftest`.
  - Baseline at 0.125, closing: 90 % of travel in 0.167 s, 3 knuckle reversals, peak speed 6.61 rad/s (above the
    6.0 limit).
  - Baseline at 0.125, opening: 7.6 mm overshoot, settling in 0.35 s.
- **Elbow limit (changed later on 2026-10-27):** the forearm folded through the upper arm, because
  self-collision is off and the stock elbow allows ±180°.
  - Measured on the meshes, the links interpenetrate from |elbow| = 169°.
  - `ur10e_asset.limit_elbow` now authors ±160°, and `arm_self_collision` also ends an episode at |elbow| > 169°.
  - Verified in sim: the new limit loads, the overshoot is at most about 5°, and no fold-through in 8 student
    episodes.
  - The capsule check still does not test forearm against tool, or the wrist_1/wrist_2 links.
- **Arm into the table (added later on 2026-10-27):** the `arm_into_table` termination (UR10e only,
  a failure) fires when any point of the upper arm, forearm or wrist links comes within 1 cm of the
  table-top plane.
  - It tests the hulls of the USD collision meshes, not capsules.
  - The lowest arm point is 10.4 cm above the table at the start pose.
  - It ended 1 of 8 student episodes.
- **Joint speeds:** the UR arm keeps the asset's real limits (2.09 and 3.14 rad/s), and the Franka has no
  limit in sim. Newton does **not** clamp joint speed, even when the limit is written to the solver; the
  limits exist only as the termination.
  - Under the policy, the task's `joint_vel_out_of_limit` termination would end UR episodes at step 1, so every
    UR play or eval so far passed `env.terminations.joint_vel_out_of_limit=null`.
  - Whether to keep the real limits is undecided.

### Results

These were measured with the Franka-matched start and the 0.125 gripper damping, before the 2026-10-27 GUI
changes.

- **Open-loop replay** of a Franka placing episode (scale 0.03) on the UR fails every way tried.
  - The best case, the current base and posture, holds the sheet up to 83 steps, lifts it and carries it toward
    the platform.
  - It still ends dropped at the release with coverage 0.
- **Single-axis 0.5 s pulses** at the defaults:
  - On-axis responses match the Franka within 0–2 % RMS.
  - The UR has more off-axis coupling: at amplitude 3, rotation pulses move the tool 85–91 mm against the
    Franka's 63–68 mm.
- **Closed loop with the repaired student:** from identical observations and the same first action, the tool
  paths diverge by 2–8 cm within 10 steps.
  - 50 stochastic episodes, full randomisation: **0/50 placed** (82 % dropped, 12 % arm below the table, 6 %
    sheet out of bounds, including 2 NaN cloth blow-ups).
  - The same layouts at a fixed scale of 0.03: 1/50.
  - The Franka with the same student places about 0.68 overall and 6/10 at 0.03.
- **Conclusion:** better observation and controller matching won't close this gap. The UR needs closed-loop
  training.

### PPO fine-tune (2026-10-27 evening): the gap is closed

FastSAC on the UR collapsed twice (entropy blow-up, free fault exits), so the student was moved to rsl_rl PPO
unchanged: `FastSACActorModel` (`algorithms/fast_sac/rsl_rl_actor.py`) wraps the FastSAC actor with its frozen
normaliser and exact tanh log-probabilities, and `scripts/fastsac_actor_to_rsl_rl.py` converts a checkpoint.
The 2F-85 fingertips touch the cloth as boxes (Franka-like pads; 290 vs 425 ms per step at 256 environments).

| Run (all from model_4000, 1000 environments) | Exploration std | Actor lr | KL bounds | Placed per 25-iteration window |
|---|---|---|---|---|
| `ur10e_ppo_finetune_box` (stopped at 175) | head's own, ~0.13 | 1e-5 | reference 1.0, update stop 0.01 | 1.4–1.9 % |
| `ur10e_ppo_nokl` (stopped at ~160) | head's own, ~0.13 | 1e-5 | none | 1.1–1.9 % |
| **`ur10e_ppo_nokl_std045_lr10x`** (stopped at 961) | **learned, from 0.45** | **1e-4** (critic 3e-3, std 1e-3) | none | 0.1 % → 12 % (150) → 52 % (400) → 72 % (600) → **80 % (900)** |

- Removing the KL bounds alone changed nothing; the policy drifted from model_4000 with flat results. The gain
  came with the wider exploration and higher learning rates (the user's settings). The many short failed
  episodes early on gave PPO far more successes per iteration than the narrow policy found.
- By the end: extraction 86 %, 91 % of extractions placed, dropped 10 %, arm into the table 2 %, finger touched
  the slot 4.5 %, std 0.18, critic EV 0.89. The distance from model_4000 grew to a reference KL of about 50.
- The kept checkpoint is `model_925` (best window 900–925, tied with 875–950 within noise); details and the
  reproduce command are in `best_ur10e_ppo/PROVENANCE.md`.
- Single deterministic episodes: placed at a random scale and at 0.04; dropped at 0.03 (model_425); failed at
  0.015 and 0.01, below the training range. One Franka episode at 0.05 did not extract: the fine-tune is
  UR-specific.
- Not done yet: a multi-episode evaluation of model_875/900/925 (deterministic and stochastic), and a scale
  sweep.
- Code added for this: `init_std`/`learn_std` on `FastSACActorModel` (state-independent std, learned or fixed),
  `std_learning_rate` on `ReferencePPO` (own Adam group), and TorchScript/ONNX export of the deterministic
  actor so `play.py` works. Tests in `tests/test_fast_sac.py` and `tests/test_reference_ppo.py`.

## 6. Pitfalls that cost time before

1. **Slot 49 and the legacy group** (§2). Check which group an actor was trained on before evaluating it. The
   checkpoint stores `actor_obs_groups`, but the pre-repair checkpoints say `action_history`.
2. **Frozen normaliser.** Whenever a pretrained actor is loaded into FastSAC, set
   `agent.freeze_obs_normalizer=true`. Replay updates to the normaliser drove the student to 0 % success within
   200 iterations.
3. **When to anchor.** Single-environment SAC without an anchor collapses within about 1000 updates. With 1024
   environments and no anchor (the D recipe) it recovers and improves. An anchor prevents both the collapse and
   any large behaviour change.
4. **Critic exploitation.** Watch `Loss/qf_max` and `Loss/policy_entropy`. qf_max climbing toward the support
   maximum of 70 while entropy falls is the actor exploiting the critic. A falling count of episodes finished
   per window is the early warning for the "hover" attractor.
5. **Newton crashes.** Newton crashes at startup in about 1 of 6 launches (segfault or malloc); relaunch. A
   cloth NaN blow-up can make the arm state NaN.
6. **Isaac Lab CLI quirks:**
   - There is no `--headless` flag; headless is the default without `--viz`.
   - Argparse prefix matching breaks custom flags that share a prefix with launcher flags. `--video_dir` was
     renamed `--clip_dir` for this reason.
   - Hydra overrides `env.*` and `agent.*` go last.
7. **Video clips.** The last frame of each recorded clip is already after the reset; trim it.
8. **Resources.** The GPU has 16 GB. 1024 environments use about 3.9 GB and run at about 0.45 s per iteration,
   including probes.

## 7. Open items (each needs the user's approval before running)

1. **UR10e (done: `best_ur10e_ppo/model_925`, about 80 % placed in training).** Next, if wanted:
   - a multi-episode evaluation of model_875/900/925, deterministic and stochastic, to pick on real numbers;
   - a scale sweep (0.01–0.1); widen the training range to 0.01–0.1 if low scales are needed;
   - sensor noise or friction randomisation for sim-to-real (none is on now);
   - the joint-velocity limits are still off (`joint_vel_out_of_limit=null`).
2. **Gripper:** re-measure after the damping change. Optionally set `OPEN_ANGLE` to about 0.0791 rad for an
   8.00 cm gap.
3. **Franka actor choice:** keep `repair49_rand_std/model_200.pt`, or run the 512-episode evaluation (§4) on
   `repaired_student_D2/model_4000.pt` and compare.
4. **Long-standing ideas:**
   - zero-initialised extra inputs that reveal the IK scale (for example the tool's last-step displacement); the
     user does not want DAgger for the scale problem;
   - redefining the critic gate relative to the measured EV ceiling;
   - a PPO fine-tune of the Franka student through the rsl_rl adapter (built and used for the UR10e, §5).

## 8. Where else to look

| What | Where |
|---|---|
| Full experiment history with all numbers | `~/.claude/projects/-home-hami-workspaces-deformable-isaac-ws-sheet-rl/memory/sac-student-pipeline.md` |
| UR10e findings (sections 1–6) | `.../memory/ur10e-transfer-findings.md` |
| Scale-randomisation narrative | `logs/fast_sac/SESSION_HANDOFF_2026-10-26.md`, `logs/fast_sac/sheet_platform_twist_sac/SCALE_RANDOMIZATION_SUMMARY.md` |
| Provenance of the two kept policies | `logs/rsl_rl/sheet_platform_twist/best_deformable_twist/PROVENANCE.md`, `logs/fast_sac/sheet_platform_twist_sac/best_of_sac/PROVENANCE.md` |
| Saved evaluations | `evaluations/teacher_student/*.json` (2026-10-23 to 10-26) |
| Console log of the last run | `logs/fast_sac/repaired_student_D2.console.log` |
| Plan file of the SAC pipeline | `~/.claude/plans/would-it-be-possible-fluttering-pnueli.md` |
| Real-world package guide (sheet detection, platform, calibration, robot link, policy obs) | `/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/README.md` |
| The Franka lab's calibration (source received from vel_fr_ws) | `/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/docs/` |
| Saved real-world setup files | `~/.ros/hrii_sheet_mocap/` (snapshot: `.../hrii_sheet_mocap/config/saved_2026-11-17/`) |
| Claude session transcript (2026-10-23 to 10-27, 51 MB, searchable JSONL) | `~/.claude/projects/-home-hami-workspaces-deformable-isaac-ws-sheet-rl/5313a276-ee9e-4ed1-8d56-a6c54737ec24.jsonl` |

## 9. Real world: camera, OptiTrack, UR16e (added 2026-11-17)

Everything from here on is the sim-to-real side, built 2026-10-31 to 2026-11-17. It spans three places:

| Place | What | Python |
|---|---|---|
| `/home/hami/workspaces/deformable_isaac_ws/sheet_rl/sheet_rl` (this workspace) | the simulated D435i camera, DA3 depth, AprilTags, point cloud, policy export | `active_isaac` venv |
| `/home/hami/workspaces/ros_two/optitrack_ws` | ROS 2 Humble: OptiTrack driver, sheet detection, platform, UR16e calibration | system `/usr/bin/python3` |
| `/home/hami/workspaces/camera_ws/da3_d435i_ros2` | the Depth-Anything-3 docker (`run_da3_sim_server.sh`) | docker |

### 9a. Machines and networks

| Thing | Address / name |
|---|---|
| This laptop | `192.168.0.223` (OptiTrack client NIC and `ROS_IP` for ROS 1) |
| Motive PC (OptiTrack server) | `192.168.0.63`, NatNet 2.9, multicast `239.255.42.99:1511`, Y-up (the driver converts to Z-up) |
| Robot PC (RB-Kairos with UR16e `ur16e_a` and Robotiq 2F-85) | ROS 1 Noetic master `http://192.168.0.52:11311` |
| ROS 1 docker that reaches the robot | `docker_deformable_isaac_ws` (Noetic, host network). Use `bash -ic` inside: its `.bashrc` sets the master |
| ROS 2 domain | **the user's terminals use `ROS_DOMAIN_ID=25`**. Tools must use the same domain to see the session |

Robot topics (ROS 1):
- **`/ur16e_a/joint_states`:** 500 Hz, six joints named `ur16e_a_ur16e_<joint>_joint`, in alphabetical order.
- **`/ur16e_a/gripper/width`:** in **decimetres** (0.839 open, 0.358 on the 35 mm bar).
- **`/ur16e_a/robot_description`:** the HRII URDF.
- **Robot commands:** `/ur16e_a/joint_cartesian_position_controller/command`, `/ur16e_a/twist_to_pose/cmd_twist` and others. Nothing on our side publishes to them.

### 9b. The OptiTrack workspace

```
/home/hami/workspaces/ros_two/optitrack_ws/
  src/hrii_optitrack_ros2/   OptiTrack driver (markers.launch.py: unidentified markers, rigid bodies as TF rigid_body_<id>)
  src/hrii_sheet_mocap/      our package; README.md has the full guide; its own local git with tags
  ros1_docker/               the original ROS 1 hrii_optitrack in a diagnostic container (README.md there)
```

`hrii_sheet_mocap` contents:
- **Sheet detection:** fit, tracking and missing-marker fill.
- **Platform setup.**
- **Robot link:** from the ROS 1 docker.
- **Calibration GUIs:** frame editor and bar editor.
- **Policy observation builder and the OptiTrack-to-base least-squares calibration.**

Git tags, oldest to newest: `before-rigid-fill`, `rigid-fill`, `relocate`, `real-obs`, `platform`, `platform-height`, `helper-fix`, `ur16e-calibration`. The latest commit is `91f5a3b`, a snapshot of the saved setup.

**Build and test.** The installed Python files are **copies**, so rebuild after every edit:

```bash
source /opt/ros/humble/setup.bash
cd /home/hami/workspaces/ros_two/optitrack_ws
colcon build --packages-select hrii_sheet_mocap --symlink-install
source install/setup.bash
cd src/hrii_sheet_mocap && /usr/bin/python3 -m unittest discover -s test     # 69 tests
```

### 9c. Commands

Every terminal starts with:

```bash
source /opt/ros/humble/setup.bash
source /home/hami/workspaces/ros_two/optitrack_ws/install/setup.bash
```

| Purpose | Command |
|---|---|
| **Sheet detection** (driver + fit + platform + RViz) | `ros2 launch hrii_sheet_mocap sheet.launch.py` (add `optitrack:=false` if the driver already runs) |
| Platform setup GUI (sheet on the platform, then on the surface) | `ros2 run hrii_sheet_mocap platform_setup` |
| **UR16e ↔ OptiTrack calibration** (robot link + robot model + frame editor + bar editor + RViz) | `ros2 launch hrii_sheet_mocap ur16e_calibration.launch.py optitrack:=true` |
| … headless, from the saved presets | `… ur16e_calibration.launch.py frame_gui:=false bar_gui:=false` |
| Robot link alone, ROS 1 half (the launch starts it itself) | `bash /home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/scripts/start_ros1_relay.sh` |
| Robot link alone, ROS 2 half | `ros2 run hrii_sheet_mocap robot_relay` |
| Least-squares OptiTrack → base calibration (one marker on the gripper, many poses) | `ros2 run hrii_sheet_mocap calibrate_mocap --ros-args -p base_frame:=ur16e_a_ur16e_base_link -p tool_frame:=ur16e_a_ur16e_flange` |
| Record a session | `ros2 bag record -o sheet_session /optitrack/unidentified_markers` |
| Tracking report on a recording | `ros2 run hrii_sheet_mocap tracking_report sheet_session` |
| Fill accuracy on a recording | `ros2 run hrii_sheet_mocap evaluate_fill sheet_session` |
| Check the bar while held | `ros2 run tf2_ros tf2_echo ur16e_a_ur16e_EE object` |

**How the robot link works.** It is one way: nothing is ever sent to the robot.
1. Inside the docker, `ros1/ros1_robot_relay.py` is piped in by `start_ros1_relay.sh`. It subscribes to the joint states and the width and reads the description.
2. It streams them as JSON lines over `127.0.0.1:47811`.
3. On the host, `robot_relay` republishes them in ROS 2 under the same names. It stamps them with the host clock (the robot's clock is within 3 ms) and adds the 2F-85 finger joint, computed from the width.
4. `robot_state_publisher`, in namespace `ur16e_a`, publishes the TF from the robot's own URDF. The URDF is cached in `~/.ros/hrii_sheet_mocap/ur16e_a_description.urdf` and copied in `urdf/ur16e_a.urdf`.
5. Ctrl-C on the launch also stops the reader in the container.

`dynamic_bridge` was deliberately not used: it would push ROS 2 `/tf` into the robot's ROS 1.

**TF tree during calibration.**

```
ur16e_a_ur16e_ctrl_link → ur16e_a_ur16e_base_link ─┬→ … → ur16e_a_ur16e_flange → 2F-85 base → ur16e_a_ur16e_EE (+0.15 m z)
                                                     └→ optitrack_world (frame_editor) ─┬→ rigid_body_1 (bar) → tracked_cuboid_top → object
                                                                                         └→ rigid_body_2 (platform)
```

- `ur16e_a_ur16e_base_link` is ROS `base_link`, the frame the policy uses.
- `ur16e_a_ur16e_flange` has `tool0`'s orientation.
- The policy's tool point is `tool0` + 0.145 m along z; the robot's EE is +0.15 m.
- The fingers sit at ±y of the gripper base.

### 9d. Calibration done on 2026-11-17 (the Franka lab's approach)

The approach was received from the Claude session on IITAVR003DW001 (`vel_fr_ws`). Its full source is in `src/hrii_sheet_mocap/docs/`.

1. **Frame editor.** Six hand-tuned numbers for `ur16e_a_ur16e_base_link → optitrack_world`, tuned until the OptiTrack scene sits on the robot in RViz. Saved: x −0.40, y −1.29, z 0.07 m, roll 0, pitch 0, yaw 90°.
2. **Bar editor.** The bar (35 × 45 × 150 mm, rigid body 1) is held upright from the side, and **Align to the grasp** is pressed once. This solves the markers-to-bar mounting rotation and puts the bar's centre on `ur16e_a_ur16e_EE`.
   - Closing axis: `0,1,0`.
   - **Object-length axis: `1,0,0` for the side grasp.** The first default, `0,0,1`, only fits a top-down grasp and made the button refuse ("87° from upright").
   - A dry run on the live TF gave the bar upright to 0°, the centre on the EE, and offsets of 10.5 mm.

Saved files:

| File (read by the nodes) | Snapshot (git) |
|---|---|
| `~/.ros/hrii_sheet_mocap/optitrack_frame.yaml` | `src/hrii_sheet_mocap/config/saved_2026-11-17/` |
| `~/.ros/hrii_sheet_mocap/bar.yaml` | same folder |
| `~/.ros/hrii_sheet_mocap/platform.yaml` (platform on rigid body 2, height 0.148 m) | same folder |

Caveats:
- The robot is on a mobile base, so the frame holds only while the base stays put.
- The frame editor and `sheet_fit`'s `calibration_file` must not both publish `optitrack_world`'s parent. That file, `~/.ros/hrii_sheet_mocap/optitrack_to_base_link.yaml`, does not exist yet.

### 9e. Sheet detection in brief (details: `src/hrii_sheet_mocap/README.md`)

- **Markers:** 10 unlabelled markers. Nine are the sheet's landmarks, nodes [0, 4, 8, 36, 40, 44, 72, 76, 80] of the 9 × 9 mesh, 10 cm apart. The tenth is a helper between landmarks 6 and 7 that tells top from bottom.
- **Pipeline:**
  - identification from the first frame with all ten markers;
  - Hungarian tracking with a 5 cm jump threshold;
  - re-finding the sheet after dropouts;
  - missing markers filled by a rigid fit (0.3 mm), with the line rule as fallback;
  - a biquadratic 9 × 9 mesh in RViz, greyed out as "LOST" when the sheet is not seen.
- **Fixed 2026-11-17:** the helper got lost. While hidden it now moves with the sheet, and it is re-acquired as a marker 3–8 cm from both landmarks 6 and 7.

### 9f. Policy export for the real robot

- **Command:** `python scripts/export_real_policy.py --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt` (active_isaac).
- **Output:** `policy.onnx`, `policy.pt` and `policy_spec.json` in `.../best_twist_slow_3950/exported_real/`.
- **Check:** verified at 4.8e-6 against the simulator.
- **Observation:** the 49 values are built from real measurements by `hrii_sheet_mocap/policy_obs.py`, verified under 10 µm against a sim episode:
  - the last two actions (14 values);
  - the landmarks minus the tool point (27);
  - the platform top-centre minus the tool point (3) and its quaternion (4);
  - the pad gap (1).
- **Action:** a relative pose step of the tool point, `a[0:6] × 0.03`, in base axes, at 50 Hz.
- **Installed 2026-11-17:** observation-only real environment and optional guarded ROS1 Cartesian pose/gripper
  sender, with manual history reset. See the continuation handoff and mocap README for usage and validation.
  Hardware commands and policy inference were not run. Actual gripper feedback is width in dm, converted
  directly to metres; no finger-angle curve is needed for this driver's real observations.
- **Still uncommissioned:** physical controller tracking, live latency and exclusive ownership
  of the command topic (the existing twist node advertises it). Same spaces/target math does not imply same dynamics. Per the user’s follow-up, the sender uses the simulation tool floor z>=0 and no added XYZ box by default; the simulator’s XYZ box belongs to the cloth. Extra tool bounds are optional.

### 9g. Simulated camera (this workspace)

- **Camera task:** `Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0`. A D435i at (1.2, 0, 0.75) m, tilted 35° down, 640 × 480, rendered with Newton Warp (no Kit).
- **AprilTags (36h11):**
  - tag 0: 10 cm, by the base;
  - tag 1: on the right;
  - tag 2: an 80 cm floor tag at (−3.3, −1.6), placed with `python scripts/floor_tag_gui.py` and saved to `config/floor_tag.json`.
- **Viewer:** the camera window opens with
  `PYOPENGL_PLATFORM=glx python scripts/play_actor.py --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0 --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt --viz newton_gl --num_envs 1 --real-time env.actions.arm_action.scale=0.03 env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6 --camera d435i`.
- **Options:**
  - **`--depth`:** Depth-Anything-3 metric depth. Start `run_da3_sim_server.sh` in `camera_ws/da3_d435i_ros2` first.
  - **`--depth_scale affine|scale|inverse|tag|none`:** the tag-based scale correction. Error near the tags: one tag about 3.5 %, two tags at different depths 1.9 %, raw DA3 37–41 %.
  - **`--marker`:** measures the tag distance (1.182 m measured, 1.181 m true).
  - **`--point_cloud`:** opens the calibrated point cloud in a separate process (an in-process Open3D blanked the Newton viewer); range set with `--point_cloud_range`.
  - **`--depth_eval`:** an error report against the true depth.

### 9h. The simulated UR16e environment the exported policy was trained for (exact files)

The exported policy (`best_twist_slow_3950/model_3950`) runs on this task. Paths are relative to
`source/sheet_rl/sheet_rl/tasks/manager_based/`.

| What | Exact location |
|---|---|
| **Task id** | `Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0`, registered in `sheet_twist/__init__.py:26` |
| **Env config (the UR16e env)** | `sheet_twist/sheet_ur16e_env_cfg.py`, class `SheetUR16eTwistStudentEnvCfg` (line 39). Its docstring lists everything the UR16e changes |
| Camera variant (same env plus the D435i and AprilTags) | `Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0` → `sheet_twist/sheet_ur16e_camera_env_cfg.py`, `SheetUR16eCameraEnvCfg` (line 70) |
| Inherits from (the UR task: actions, gripper, tool point) | `sheet_twist/sheet_ur10e_env_cfg.py`, `SheetUR10eTwistStudentEnvCfg` (line 54) |
| … which inherits from (the student task: observations) | `sheet_twist/sheet_platform_twist_student_env_cfg.py`, `SheetPlatformTwistStudentEnvCfg` (line 231) |
| … which inherits from (scene: sheet, slot, platform; rewards, terminations) | `sheet_rl/sheet_rl_platform_env_cfg.py`, `SheetRlPlatformEnvCfg` (line 61) |
| **Policy observation group (49 values)** | `ActionHistoryObsCfg`, `sheet_twist/sheet_platform_twist_student_env_cfg.py:168` |
| Observation functions | `sheet_rl/mdp/observations.py`: `sheet_key_points_from_tool` (98), `target_from_tool` (119), `finger_width` (133), `gripper_opening` (155) |
| The "last two applied actions" input | `sheet_rl/mdp/grasp_control.py:120`, `applied_policy_actions` |
| **Arm action** (relative-pose differential IK on `wrist_3_link` + tool offset) | `sheet_twist/sheet_ur10e_env_cfg.py:149`. Scale is overridden at run time: `env.actions.arm_action.scale=0.03` |
| Gripper action (2F-85 mimic gripper) | `sheet_twist/sheet_ur10e_env_cfg.py:157` |
| **Tool point** | `sheet_twist/ur10e_mdp.py:31`: `TOOL_POS = (0, 0, 0.145)` from `wrist_3_link`, i.e. `tool0`; `TOOL_ROT` on line 33 |
| UR16e asset (stock Isaac Sim UR16e with the 2F-85) | `sheet_twist/ur16e_asset.py` |
| Start pose | `sheet_twist/ur16e_start_pose.json`: joint angles and the start tool pose `(0.519, −0.113, 0.23)` in the robot frame |
| Self-collision | `sheet_twist/ur16e_self_collision.py` (elbow limit 153°) |
| Arm drives | `ur10e_arm_control.json`, 400/80 (`ur16e_arm_control.json` instead when one is saved) |
| Camera and tags | `sheet_twist/d435i_camera.py`, `sheet_twist/fiducial.py` |
| PPO runner used for UR fine-tunes | `sheet_twist/sheet_twist_ppo_cfg.py:102`, `UR16eFineTuneRunnerCfg` |

Run settings the policy was evaluated with: `env.actions.arm_action.scale=0.03` and
`env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6`. The robot base sits at the UR10e's (−0.12, 0.02).
The real-world mapping of every observation is in `/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/README.md`, section "Using the policy on the real robot"; `exported_real/policy_spec.json` holds the same in machine-readable form.

### 9i. Pitfalls specific to the real-world side

1. **ROS domain.** The user works on domain 25; a tool on the default domain 0 sees nothing. Tests go on a separate domain (`export ROS_DOMAIN_ID=77`).
2. **Never `pkill` or `killall` by name** on this machine: it killed the user's own `sheet_fit` and RViz once. Start your own processes with `setsid`, write their PID, and kill that process group only.
3. **Rebuild after editing** `hrii_sheet_mocap`: the installed modules do not follow the sources.
4. **Don't name a method `handle`** in an rclpy `Node` subclass: it shadows `Node.handle`, and the node dies with `AttributeError: __enter__`.
5. **In a launch, scope included launch files** (`GroupAction(scoped=True)`). Otherwise an include's `rviz:=false` turns off the parent's RViz.
6. **No top-level `pxr` / `omni` imports** in task configs: they crash `--viz kit`.
7. **Only one OptiTrack driver** may publish `/optitrack/unidentified_markers`. Stop the live driver before replaying a bag on the same topic.
