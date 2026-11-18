# Operations reference (the previous README, kept verbatim)

> Moved here on 2026-11-18 when [README.md](../README.md) was rewritten as the technical overview. Commands, flags and per-policy notes below are unchanged.

# Sheet RL — pulling a deformable sheet out of a slot

For the new 2 cm raised rectangular placement task and its `--grasp` training mode,
see [Platform task](../PLATFORM_TASK.md).

For the Franka cube lift distilled from joint control into a twist (delta-pose) policy with DAgger
and PPO, including the 5 cm platform evaluation, see [Franka lift twist](../FRANKA_LIFT_TWIST.md).

<p align="center">
  <img src="../media/hand_placement.gif" alt="Franka Panda draping a cloth sheet over the mannequin arm, palm down" width="49%">
  <img src="../media/Second_arm.gif" alt="The same task with the mannequin arm rolled palm up" width="49%">
</p>
<p align="center">
  <sub>Left: the sculpted arm as authored, palm down. Right: the same task with the arm rolled palm up &mdash; a mounting rotation the goal command is told about, so the policy's observation is unchanged.</sub>
</p>

A Franka Panda learns to pinch the free edge of a cloth sheet standing upright in a narrow slot and
draw it straight up and clear. Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) as a
standalone extension — the scene, robot and manager framework are inherited; the task is not.

**Task id:** `Template-Sheet-Rl-v0`  ·  **RL library:** rsl-rl (PPO)  ·  **Physics:** Newton / MJWarp with a VBD cloth solver

## The task

Two thin walls 1 cm apart hold a 20 × 20 cm sheet upright, with about 5 cm projecting above them.
That projecting strip is the point of the slot: a free edge, held vertical and presented side-on, is
far easier to pinch than a sheet lying flat with nothing to get a finger under. Slot and sheet are
placed together at reset with a shared offset and yaw drawn over ±90°, and a mannequin arm is
mirrored to the opposite side of the table.

Success is the sheet's *lowest* node clearing the wall tops by 2 cm while the gripper is holding it —
measured on the lowest node so the whole sheet is out, not merely tilted out.

## Reward

Two dense terms shape the approach, two one-shot terms pay for the outcome, and the rest are charges.

| term | what it pays for |
|---|---|
| `approach` | fingertips to the middle of the sheet's top edge, on two `tanh` scales so the pull is felt from across the table and again at the grasp |
| `alignment` | hand pointing down and the closing axis square across the slot |
| `square_progress` | the *change* in squareness — potential-based, so it pays for turning the wrist rather than for having turned it, and cannot be farmed |
| `grasp_stage` | one-shot bonuses for the first grasp on the top edge and for extraction; charges for closing on nothing and for letting go early |
| `lift_progress` | potential-based climb from the grasp up to the extraction bonus, gated on actually holding |
| `table_clearance` | a slope pushing back against drifting into the table |
| `gripper_recommit` | every gripper closure after the first, counted on the commanded bit rather than the measured width |
| `ee_speed` | hand speed above 0.2 m/s, exponential to 9.52 at 1 m/s then flat (off unless `--slow`) |

A grasp is judged by whether cloth is actually between the pads — fingers shut with a sheet node
inside `capture_radius` — not by the pose the gripper struck. Pose gates were tried and rejected a
hand-flown grasp that plainly worked; `alignment` still encourages the posture, it just cannot veto
a grasp that succeeded.

## Setup

```bash
python -m pip install -e source/sheet_rl
python scripts/list_envs.py          # confirms the task registered
```

Requires a Python interpreter with Isaac Lab installed. See the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

## Running

**Train**

```bash
python scripts/train.py --rl_library rsl_rl --task Template-Sheet-Rl-v0 --num_envs 1024 \
  agent.run_name=grasp_v1 agent.experiment_name=sheet_grasp \
  agent.max_iterations=3000 agent.num_steps_per_env=24 \
  agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
  agent.actor.obs_normalization=true agent.critic.obs_normalization=true \
  agent.clip_actions=2.4 agent.algorithm.learning_rate=3.0e-4 agent.save_interval=20
```

Add `--slow` to arm the hand-speed charge, which trades a slower arm for a lower return.

Add `--push` to disturb the extraction: half the episodes that pull the sheet clear of the slot get
a 5 N yank on the middle of its lower edge the instant they do, once, for a quarter of a second —
the same force the `X` key applies while watching. `Events/pushed_frac` in the training table is the
share of *all* episodes that were pushed; read it against `Events/extraction` to see whether the
policy is dodging the push by refusing to extract. Tunable alongside the flag, e.g.
`--push env.push.force=8.0 env.push.on_extraction_probability=0.25`.

To resume, add `--resume --load_run <run-dir> --checkpoint model_<n>.pt` and repeat every
`agent.*` override — the checkpoint stores weights and optimizer state, not configuration.
`max_iterations` is *additive* from the loaded iteration, not a target.

### Sheet twist student: stochastic critic warm-up and PPO

For `Template-Sheet-Rl-Platform-TwistStudent-v0`, select
`--agent rsl_rl_finetune_cfg_entry_point`. This uses PPO with a frozen copy of the
starting actor as a KL reference. Both phases sample the checkpoint's Gaussian
distribution, with its per-action standard deviations frozen. Defaults are
32 rollout steps, gamma 0.997, lambda 0.97, actor LR 1e-5, critic LR 3e-4,
two PPO epochs, clip 0.1, zero entropy bonus, and reference coefficient 1.0.
The grasp potential's gamma is synchronized with the learner.

The existing prepared critic can be used with 32 rollout steps without another warm-up.
To reproduce its original warm-up at 128 steps, leaving every actor parameter unchanged:

```bash
python scripts/train.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --agent rsl_rl_finetune_cfg_entry_point \
  --num_envs 512 --seed 42 --max_iterations 26 \
  --resume --load_run '^student348_critic_warmup$' --checkpoint model_149.pt \
  agent.run_name=s348_stochastic_cwarm_v1 agent.num_steps_per_env=128 \
  agent.algorithm.critic_only=true agent.algorithm.num_learning_epochs=5
```

Warm-up logs value error and explained variance on held-out environment
trajectories against lambda-return targets. Inspect these across complete episode
cycles. The prepared checkpoint below was selected after 26 updates (1,703,936
transitions), with mean held-out explained variance 0.932 over the last ten updates.
Its actor and standard deviations are bit-for-bit identical to the source checkpoint.
See `logs/rsl_rl/sheet_platform_twist/s348_stochastic_cwarm_v1/validation.json`.
Then start PPO from the prepared checkpoint:

```bash
python scripts/train.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --agent rsl_rl_finetune_cfg_entry_point \
  --num_envs 512 --seed 42 --max_iterations 100 \
  --resume --load_run '^s348_stochastic_cwarm_v1$' --checkpoint model_25.pt \
  agent.run_name=s348_ppo_ref_v1 agent.save_interval=10
```

Importing an old checkpoint or moving from warm-up to PPO starts a fresh optimizer
and iteration counter. Continuing an existing reference-PPO run preserves optimizer
moments and the original reference actor; configured learning rates always apply.
Changing gamma or lambda after warm-up requires preparing the critic again.

Evaluate saved actors with:

```bash
python scripts/train_dagger_sheet_twist.py --eval_student PATH \
  --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --agent rsl_rl_finetune_cfg_entry_point --edge right \
  --num_envs 128 --eval_episodes 2 --seed 123 --stochastic
```

Remove `--stochastic` to measure deterministic execution.
The evaluator counts the same number of episodes per environment and records actual
`placed` success. JSON filenames include the seed. `Placement/success` now uses the
same outcome; `Outcomes/success_rate` is cumulative over completed training episodes.
The task still defines `placed` as at least 5% coverage for one second, so compare
final coverage as well as success. KL regularization does not guarantee improvement;
select checkpoints using evaluation.

### The kept policy: best deformable twist

`logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt` is the policy to
use for the deformable twist placement task. It is a copy of the final iteration of
`s348_angle20_v2`, kept under a stable name so runs can be deleted without losing it;
`PROVENANCE.md` beside it records the lineage, the metrics and the command that produced
it, and `params/` is the exact configuration it was trained with.

At its final iteration, over 1024 environments: **68% `placed`**, 18.5% `dropped`,
13.2% `held_too_low`, mean yaw error **11.3 deg**, best coverage **0.713**, final
coverage 0.680, extraction 0.974, mean return 1880.

Watch it:

```bash
PYOPENGL_PLATFORM=glx python scripts/play.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-TwistStudent-v0 \
  --agent rsl_rl_cfg_entry_point \
  --load_run '^best_deformable_twist$' \
  --train_env_cfg --viz newton_gl --num_envs 1 --seed 42 \
  env.rewards.grasp_progress.params.gamma=0.997
```

Pass `--num_envs 1`, not `--lite`: the Newton GL viewer segfaults under `--real-time`,
which is half of what `--lite` expands to. One environment on its own is fine.

Playback prints one `[release]` line per episode with the coverage and the angle to each
of the four platform edges. Over 37 consecutive releases this policy placed 34 (0.92),
mean coverage 0.734, mean yaw error 6.9 degrees (best 0.1, worst 21.5) -- better than its
training averages, which carry the exploration noise. Score it against the scripted expert with the evaluator in
the section above; the expert reaches 0.95 success and 0.85 coverage over 40 episodes,
so there is still headroom.

Note the asymmetry: `play.py` takes the weight from `--load_run` alone, because its
`--checkpoint` is a literal file path rather than a name inside the run (pass
`logs/rsl_rl/sheet_platform_twist/best_deformable_twist/model_2874.pt` to name it).
`train.py` is the other way round -- continue training from it with
`--resume --load_run '^best_deformable_twist$' --checkpoint model_2874.pt` and a fresh
`agent.run_name`. The folder is a run folder
like any other, so `train.py` refuses to write into it, which is what keeps it intact.

### The kept UR10e policy: best UR10e PPO

`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_ur10e_ppo/model_925.pt` is the policy to use on the
UR10e with the Robotiq 2F-85 (`Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0`). It is the Franka-trained
FastSAC student (`repaired_student_D2/model_4000`, reading the repaired 49-value `action_history`) fine-tuned
on the UR with PPO: no KL bounds, a learned exploration std started at 0.45, and actor/critic/std learning
rates of 1e-4/3e-3/1e-3. `PROVENANCE.md` beside it records the lineage, the full table and the command.

Over training windows of about 9000 episodes (1000 environments, exploration on): **80% `placed`**, 86%
extraction, 10% `dropped`, 2% arm into the table, 4.5% finger touched the slot, mean return about 4100. A
deterministic `play.py` check placed 30 of 32 releases (mean coverage 0.88, edge-angle error 7.3 deg). It was
trained with the arm action scale in 0.03–0.1 and fails below that range; it does not transfer back to the
Franka.

Watch it:

```bash
PYOPENGL_PLATFORM=glx python scripts/play.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0 \
  --agent rsl_rl_finetune_cfg_entry_point \
  --load_run '^best_ur10e_ppo$' \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_ur10e_ppo/model_925.pt \
  --train_env_cfg --viz newton_gl --num_envs 1 --seed 42 \
  agent.actor.init_std=0.45 env.terminations.joint_vel_out_of_limit=null \
  "env.events.randomize_arm_scale.params.scale_range=[0.03,0.1]" \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

`agent.actor.init_std=0.45` is needed to load it (the checkpoint holds the learned std); playback itself is
deterministic.

### Also kept: best twist, joint-speed limited (model_1924)

`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_velterm_1924/model_1924.pt` is `best_ur10e_ppo/model_925`
trained a further 1000 iterations on the UR10e with the arm joint-speed limit on (the real UR10e limits) and the
action scale randomised over 0.015–0.1. Its last training window placed 79.3 %, with 4.1 % of episodes ended by the
speed limit. `PROVENANCE.md` beside it has the checks and the commands.

- **UR10e and UR16e:** the same weights run on both. On 10 deterministic episodes (seed 42, scale 0.03) both arms
  gave the same outcome in every environment: 8 placed, 1 dropped, 1 unfinished.
- **Franka:** it needs the tool point moved **13.2 mm toward the flange**, from panda_hand + 0.107 m to 0.0938 m, in
  the two observation terms and the IK action. With the default 0.107 m it mostly drops the sheet or never grasps
  it. Play it on the Franka with `scripts/play_actor.py` and the three `tool_offset` / `body_offset` overrides
  shown in its PROVENANCE; the Franka task's own agent config cannot load this checkpoint.

Watch it on the UR16e (or the UR10e, with `--task Template-Sheet-Rl-Platform-TwistStudent-UR10e-v0`):

```bash
PYOPENGL_PLATFORM=glx python scripts/play.py --rl_library rsl_rl \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-v0 \
  --agent rsl_rl_finetune_cfg_entry_point \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_velterm_1924/model_1924.pt \
  --train_env_cfg --viz newton_gl --num_envs 1 --seed 42 \
  agent.actor.init_std=0.45 env.actions.arm_action.scale=0.03 \
  env.events.reset_layout.params.sheet_xy_jitter=0.05 env.events.reset_layout.params.sheet_yaw_jitter=0.7854
```

### Best slow twist: joint-speed 60 % + tool-speed penalty (model_3950)

`logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt` is model_1924 trained on through a
speed curriculum: the joint-speed limit went 80 % → 60 %, then a per-step penalty was added on tool-point speed
above 0.4 m/s (30 per step there, rising to 80 at 0.75 m/s; `ur10e_mdp.tool_speed_over`, off by default in the
task). Around this checkpoint the training placed 81.9 %, with the lowest tool-speed penalty of the curriculum
(−639 per episode). `PROVENANCE.md` beside it has every step, the numbers and the commands.

On the same recipe, at scale 0.03 on the UR16e, one nominal episode placed in 4.30 s against 1.92 s for
model_1924. The tool peaked at 0.34 m/s against 0.87 m/s, so it stays under the real arm's 0.45 m/s cap. At
scale 0.09 it peaks at about 0.58 m/s.

Load it with any `agent.actor.init_std` (the checkpoint's learned std, about 0.08, replaces it). The command is
the model_1924 one above with this checkpoint, `agent.actor.init_std=0.25` and
`env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6`.

### UR16e with a RealSense D435i camera

`Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0` is the UR16e task plus a simulated D435i colour camera
(`scene.d435i`, `sheet_twist/d435i_camera.py`) in its 640 x 480 colour mode: 55 x 42.6 degrees of view, the 4:3
crop the real camera makes of its 69 x 42-degree imager (factory intrinsics fx = fy = 615 px), fixed just past the table's front
edge at (1.2, 0, 0.75) looking back at the robot and tilted 35 degrees down (raised from 0.45 m and 20 degrees on
2026-11-04; the tag, DA3 and point-cloud figures below were measured at the earlier pose), so the slot, the platform and the
arm's working volume fill the frame. It renders through Newton's Warp ray tracer, so it needs neither Kit nor
`--enable_cameras`, and the cloth renders too. The policy does not read it; the UR16e task itself is unchanged.

The table is pink in the camera as in the viewer, but for a different reason: the task's real table is invisible
(the viewer draws the pink success-marker box over it, green once the sheet is placed), and the camera renders
only physics shapes, so this task makes the table itself visible with the marker's colour as its
`displayColor` (a PreviewSurface colour would not reach the Newton renderer). The green success tint stays
viewer-only. For a room-like backdrop there is a flat mid-grey floor and a light blue-grey sky
(`D435I_BACKGROUND`), with the far clip at 10 m. The floor is a visual-only 30 m slab (`scene.floor`,
`GROUND_COLOR`) just under the table's bottom face, a table height (1.05 m) below the top, so the table stands
on it as a table: the base task's ground-plane mesh (black, lit from below in this renderer, at the table-top
level) is hidden, while its collision plane stays at z = 0 and does the physics exactly as in the UR16e task.

Watch it in its own window while a policy plays (closing the window, or q in it, stops the run). `--viz kit` works
in place of `--viz newton_gl`; the camera renders the same way under either viewer:

```bash
PYOPENGL_PLATFORM=glx python scripts/play_actor.py \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0 \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt \
  --viz newton_gl --num_envs 1 --episodes 10 --real-time --camera d435i \
  env.actions.arm_action.scale=0.03 env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6
```

**Depth-Anything-3 window.** `--depth` adds a third window of the same size showing the camera's frames as DA3
metric depth (DA3Metric-Large, coloured as `da3_d435i_view.py` colours it: TURBO, near warm, 2-98 % percentiles).
The model runs in your `da3-d435i` container as a small TCP server; start it first, in another terminal:

```bash
~/workspaces/camera_ws/da3_d435i_ros2/run_da3_sim_server.sh      # 127.0.0.1:5555; --process-res 392 is faster
```

then add `--depth` to the play command (`--da3_address host:port` if it is elsewhere). The client
(`scripts/da3_depth_window.py`) sends frames from a background thread and drops the ones it cannot keep up with,
so the simulation never waits; at about 60 ms a frame the depth window runs a few frames behind the colour one.
The metric scale uses the simulated camera's focal length (615 px). `da3_server.py` in that folder is the server
and documents the byte protocol; the Dockerfile copies it into the image, and the run script mounts the host
file over it so edits need no rebuild.

**DA3's scale, corrected from the tags.** The camera task has a second tag of the same family (`scene.tag2`, id 1)
flat on the table on the camera's right at (0.05, 0.40): 1.23 m deep against the first tag's 1.14 m, 38 cm to the
base's +y side, outside the platform's range (x 0.38..0.62, y 0.03..0.27) and mostly out of the arm's way (it is
still hidden in 45 % of the frames of an episode; its mirror image of the first tag, (0.15, 0.30), was hidden in
80 %). `--marker` finds every tag in the frame (the "Tag distance" window outlines each with its distance; the
printout adds the other tags' distance and depth). With `--depth` as well, each DA3 map is corrected from the
depths of the tags visible in that frame (the PnP translation's z, the same quantity as a depth map's values)
against DA3's median inside each tag's outline; nothing is carried over from before a tag was hidden: one visible
tag gives a scale, none leaves the map raw (`--depth_scale_hold` keeps the last fit instead). The depth window
shows the corrected map. `--depth_scale` picks the fit: `affine` (default) a scale and an offset through the
anchors when they span at least 25 cm of depth, else a scale; `scale` one factor from all tags; `inverse` the
affine fit in inverse depth; `tag` one factor from the lowest-id tag only (the first version;
`--depth_scale_smoothing` blends it over frames, which measured worse because DA3's scale jumps between frames);
`none` raw. `--depth_eval` renders the camera's true depth too (`distance_to_image_plane`) and prints, once a
second and summarised at the end, how far the raw map and every variant are from it on the same frames: the median
of DA3 over truth (the scale error), the median absolute relative error (AbsRel), the RMSE and the share of pixels
within 5 %, over all pixels that see geometry and over the near band (truth up to 2 m: the table and workspace).
Measured over 3 episodes each, near band:

| correction | tag 2 at (0.62, -0.20), 0.70 m deep, 679 frames | tag 2 at (0.05, 0.40), 1.23 m deep, 673 frames |
|---|---|---|
| raw | x1.42 (1.02..2.20), AbsRel 41.5 %, 3 % within 5 % | x1.40 (1.03..2.19), AbsRel 40.2 %, 4 % within 5 % |
| `tag` (one tag) | x1.02 (0.98..1.07), 3.5 %, 68 % | x1.02 (0.98..1.08), 3.4 %, 68 % |
| `scale` (both tags) | x1.01 (0.98..1.05), 3.0 %, 73 % | x1.02 (0.97..1.08), 3.4 %, 68 % |
| `affine` (both tags) | x1.00 (0.976..1.020), 1.9 %, 79 % | offset fitted from 9 cm of spread: 6.3 %, 48 % |
| `inverse` (both tags) | x1.00 (0.976..1.017), 1.9 %, 80 % | same: 5.4 %, 51 % |

So the second tag improves on one tag only when it sits at a clearly different depth, which is what fixes the
offset; at the first tag's distance it adds nothing (one scale from both: the same 3.4 %), and an offset fitted
across 9 cm of depth is noise (hence the 25 cm rule, under which `affine` is a scale in the right-side layout).
An earlier single-tag run gave 4.4 % (x1.03, 0.995..1.08; with smoothing 0.3: 5.4 %, 0.88..1.25). The anchors
themselves: tag 0 at 1.142 m against a true 1.139 in every run, tag 1 at 0.698 against 0.693 and 1.228 against
1.229. The RMSE over all pixels is dominated by the far floor and does not improve; the relative numbers are the
ones to read. Code: `scripts/depth_scale.py` (`scale_from_tag`, `anchors_from_tags`, `fit_anchors`, `TagFit`,
`DepthErrors`), `tests/test_depth_scale.py`.

**A third tag on the floor, placed with a GUI.** The camera task also has a large tag (`scene.tag3`, id 2, 0.8 m
black square) flat on the floor behind the robot, on the camera's left at (-3.3, -1.6): the table is a solid block
1.05 m thick, so the only floor the camera sees is the strip beyond the far edge (from x = -2.5), 5 m away and
foreshortened, hence the size (77 px wide in the frame). Measured over 3 episodes per spot, the arm's elbow hid a
tag at (-3.3, 1.0) in 55 % of frames and at (-3.3, 1.6) in 42 %; at (-3.3, -1.6) it was seen in every one of 702
frames. Nearer spots beside the table, e.g. (-1.5, +-1.5), leave the frame. To place it yourself:

```bash
python scripts/floor_tag_gui.py        # while play_actor.py --camera d435i ... runs (it serves frames on :5556)
```

draws over the live camera frame where a tag of the chosen size at the chosen floor position would appear
(projected with the camera's pose and intrinsics), with the table top's outline, and says whether the whole printed
quad is inside the frame and clear of the table, its size in pixels, how steeply it is seen, and, while episodes
play, in what share of frames that patch of the image stayed undisturbed by the arm (reset when you move it).
Sliders, entries or arrow keys move it (5 cm; Shift 25 cm; +/- change the size); Save writes
`config/floor_tag.json`, which the camera task reads when it is next built (so the tag moves on the next start; the
GUI also prints the Hydra override for one run). `--marker` solves every tag with its own size (the detector takes
`sizes` per id from the scene).

The floor tag's use in the depth correction: DA3's error is far from affine over range (at the floor tag, 4.7 m,
DA3 read 1.7 times less relative to truth than at the table tags), so a far anchor inside the fit made near
pixels wrong, even negative (black in the depth window): the fit now uses only tags nearer than 2.5 m (`affine`
with two, a scale with one), and a far tag alone gives a scale through the near-over-far factor ratio last seen
with both in view. The last fit is held while no tag is visible (`--no_depth_scale_hold` turns that off), and
corrected depths beyond 10 m are invalid. With the floor tag on the left (3 episodes, 694 frames; tag 0 in every
frame, so the far-only path never ran): raw AbsRel 13.8 % in the near band, one tag 2.4 %, both table tags 2.7 %
(they are only 9 cm apart in depth, so no offset); the floor tag's PnP depth was 4.718 m against a true 4.692.

**Point cloud from the corrected depth.** `--point_cloud` (with `--depth`; `--marker` for the correction) opens a
live Open3D window with the DA3 depth back-projected through the camera's intrinsics and coloured from the frame
(`scripts/point_cloud.py`): only points within `--point_cloud_range` (2 m) of the camera, so the table and
workspace without the far floor, and without depth edges (`--point_cloud_edge`, 3 cm: a monocular model smears
the arm's outline into streaks down to the table; 0 keeps them). While tag 0 is visible the cloud is expressed in
the tag's frame, x and y along the tag's edges on the table and z the height above it, so it is a map of the
cell in table coordinates (the axes gizmo marks the tag); otherwise it stays in the camera frame. Mouse rotates,
scroll zooms, s saves a PLY to `--point_cloud_dir` (`logs/point_clouds`), and the last cloud is saved there when
the run ends (`cloud_last_<frame>_frame.ply`, binary PLY with colours; Open3D, MeshLab and CloudCompare open it).
The window is its own process, started by the play script and fed over a local socket (newest cloud only): an
Open3D window inside the playing process took the GL context from the Newton GL viewer, which then showed only
its blue sky. From one frame of an episode, 235 k points: 65 % lie within 3 cm of the table plane (mean height
-3 mm, spread 10 mm), the rest are the arm, platform and sheet. About 0.2 s more wall-clock per control step.

```bash
PYOPENGL_PLATFORM=glx python scripts/play_actor.py \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0 \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt \
  --viz newton_gl --num_envs 1 --real-time --camera d435i --marker --depth --point_cloud \
  env.actions.arm_action.scale=0.03 env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6
```

```bash
PYOPENGL_PLATFORM=glx python scripts/play_actor.py \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0 \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt \
  --viz newton_gl --num_envs 1 --episodes 3 --camera d435i --marker --depth --depth_eval \
  env.actions.arm_action.scale=0.03 env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6
```

**Fiducial tag and distance.** The camera task always has a printed 10 x 10 cm AprilTag (`scene.tag`,
`sheet_twist/fiducial.py`: family 36h11, id 0, the black square 10 cm with a one-cell white margin, a textured quad
the ray tracer samples) lying flat on the table at (0.15, -0.30), 27 cm in front of the robot base. It replaced a QR
code, whose 29 modules fell below a pixel each from the camera; the tag's 8 cells get 3-4 times the pixels, and tag
detectors are made for pose. Families: `apriltag_36h11` (default, the robust general choice), `apriltag_16h5` (fewer
bits, longer range), `aruco_4x4` (`env.scene.tag.spawn.family=...` and `--tag_family` together).

`--marker` on the play command measures the camera-to-tag distance from the colour frames alone, in the same process,
under either viewer: a fourth window, "Tag distance", shows the frame with the tag outlined, the distance and the
camera's pose in the tag's frame written on it; the terminal prints, once a second, the distance, the tag centre in
the camera frame (x right, y down, z forward) and the camera's position and orientation in the tag's frame (x, y
along the tag's edges, z its normal, so for the flat tag z is the camera's height above the table; roll/pitch/yaw
of the camera axes in that frame, the camera's elevation seen from the tag, and how far below the table plane it
looks), plus a summary at the end. Measured from the tag's place: camera 1.054 m along, 0.299 m across, 0.444 m
up, elevation 22.1 degrees, looking 19.6 degrees down (true: 1.05, 0.30, 0.4495, 22.4, 20).
OpenCV's ArUco detector finds the tag (also on the frame enlarged 2-3x; the tag is 39 px across from its place), with
AprilTag-style corner refinement, and `solvePnPGeneric` (`IPPE_SQUARE`, lower-reprojection-error solution, then LM
refinement) solves the pose from the 10 cm square and the camera's own intrinsics (615 px). Measured against the
scene's geometry: **1.182 m against a true 1.181 m** at its place by the base (every frame of an episode, both
viewers), 0.754 m against 0.7555 m at (0.62, -0.18) near the camera. Corner refinement is what makes the accuracy:
OpenCV's SUBPIX gave 1.200 m, CONTOUR 1.207, none 1.211 on the same frame. It runs every 2nd control step
(`--marker_every`; tens of ms per frame). The same code is in `scripts/marker_distance.py`, which also runs
standalone on `--image` files or on the camera window's stream (`--camera_port`, 127.0.0.1:5556), with
`--family/--size/--fx/--fy/--cx/--cy` for a real camera.

```bash
PYOPENGL_PLATFORM=glx python scripts/play_actor.py \
  --task Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0 \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt \
  --viz newton_gl --num_envs 1 --episodes 10 --real-time --camera d435i --marker \
  env.actions.arm_action.scale=0.03 env.terminations.joint_vel_out_of_limit.params.limit_scale=0.6
```

Move the tag with `env.scene.tag.init_state.pos=[x,y,0.0005]`; stand it up with
`"env.scene.tag.init_state.rot=[0,0.7071068,0,0.7071068]"` (x y z w, 90 degrees about y) and a z of half the quad.

To move the sheet and its slot for a run, override the layout reset's fixed pose, e.g. +0.2 m in y with no jitter:
`env.events.reset_layout.params.fixed_sheet_position=[0.40946364,0.12283249,0.102]
env.events.reset_layout.params.sheet_xy_jitter=0.0 env.events.reset_layout.params.sheet_yaw_jitter=0.0`
(the default is (0.409, -0.077, 0.102), yaw -0.54 rad; `fixed_sheet_yaw` is overridable the same way). The task's
workspace box (`deformable_out_of_bounds`, any sheet node outside ends the episode) grows to cover that position
(`deformable_outside_layout_bounds`; it is the usual box at the default layout): with the box fixed, a +0.2 m
shift put the carry's sideways sweep through its y = 0.5 wall and ended good episodes. The platform is placed
on its own (the +y side, randomised) and only moves if the sheet is put within its clearance distance, as a +0.2 m
shift does; -0.2 m leaves it where it is.

Move the camera with `env.scene.d435i.offset.pos=[x,y,z]` and `env.scene.d435i.offset.rot=[x,y,z,w]`
(`d435i_camera.camera_rotation(yaw_deg, pitch_deg)` gives the quaternion). In code the frames are
`env.scene["d435i"].data.output["rgb"]`, `(num_envs, 480, 640, 3)` uint8, rendered when read (about 4 ms a frame).

### Watch the original sheet task

```bash
python scripts/play.py --rl_library rsl_rl --task Template-Sheet-Rl-v0 --viz newton_gl \
  --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_<n>.pt \
  agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
  agent.actor.obs_normalization=true agent.critic.obs_normalization=true agent.clip_actions=2.4 \
  --lite
```

`--lite` is one environment at wall-clock speed — a watching tool rather than a sampling one, and
cheap enough to sit alongside a training run.

**Measure how fast the arm actually moves**

```bash
python scripts/eval_joint_speed.py --task Template-Sheet-Rl-v0 --num_envs 16 --headless \
  --checkpoint logs/rsl_rl/sheet_grasp/<run>/model_<n>.pt --steps 600 \
  agent.actor.hidden_dims=[256,128,64] agent.critic.hidden_dims=[256,128,64] \
  agent.actor.obs_normalization=true agent.critic.obs_normalization=true agent.clip_actions=2.4
```

Prints per-joint and end-effector speed percentiles and writes `joint_speed.png` beside the
checkpoint. Capping `arm_action.scale` sets how fast the arm *may* move; this says how fast it
chooses to.

**Fly it by hand**

```bash
python scripts/teleop_keyboard.py --free_run
```

Same scene the policy trains on, with the reward printed live per term.

### Export for the real UR16e

`scripts/export_real_policy.py` writes a checkpoint's deterministic policy as `policy.onnx` and `policy.pt`
(observation normaliser and action squashing inside: raw 49 observation values in, 7 actions out) plus
`policy_spec.json` (observation layout and frames, action meaning and scale, gripper filter, the simulated 2F-85's
finger-angle-to-pad-gap curve), into `exported_real/` next to the checkpoint. `--check obs_capture.npz` compares it
with the simulator's actions on a recorded episode; for model_3950 they agree to 5e-6.

```bash
python scripts/export_real_policy.py \
  --checkpoint logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/model_3950.pt --arm_action_scale 0.03
```

Measured against the simulator: the robot root is the UR's ROS `base_link` (the controller's `base` turned 180
degrees about z), `wrist_3_link` is ROS `tool0`, the tool point is `tool0` + 0.145 m along its z, quaternions are
x y z w. The actor reads the `action_history` group only (49 values, 50 Hz): the last two applied actions (14),
the nine sheet landmarks minus the tool point (27), the platform top-centre minus the tool point and its
quaternion (7), the finger pad gap (1). Every position is taken from the tool point, so the base's position
cancels; its orientation remains (the offsets' axes, the platform quaternion, the action's frame). What has to be
measured on the real setup, and how, is written out in
`/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/README.md`, section "What is needed to run the
exported policy".

## Reading the logs

`Episode_Reward/*` is the episodic sum divided by `max_episode_length_s` (**16.65 s**), so multiply
by 16.65 to get the real per-episode value. `Events/*` are per-episode means of the grasp
diagnostics, published on reset.

After resuming, ignore `Mean reward` for the first ~25 iterations: `init_at_random_ep_len`
randomises episode phase, so the statistics buffer fills with truncated stubs and reads far below
the policy's actual performance. Wait for `Mean episode length` to recover before judging anything.

## Layout

```
scripts/                     train, play, teleop, joint-speed evaluation
source/sheet_rl/.../sheet_rl/
├── sheet_rl_env_cfg.py      scene, actions, rewards, terminations, events, commands
├── agents/                  PPO hyperparameters
└── mdp/                     rewards, terminations, observations, events, commands
```

## Pinned against

This task subclasses Isaac Lab internals (`FrankaClothEnvCfg`, `isaaclab_tasks.core.lift.mdp`,
Newton's articulation data), so upstream changes can break it. Developed against:

| | version |
|---|---|
| Isaac Lab | `68d7f932d` (`perf-2026-07-06-389-g68d7f932d`) |
| rsl-rl-lib | `5.4.1` |
