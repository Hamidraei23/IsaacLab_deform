# Real-world UR16e environment handover — 2026-11-17

**Current compact reference (2026-11-18):** [UR16e simulation to real transfer](/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap/SIM_TO_REAL_ARCHITECTURE.md).
It includes subsequent live observation logs and controller connection findings; the sections below retain the implementation-session history.

## Status and user instructions

**SOFTWARE SETUP COMPLETE, HARDWARE EXECUTION UNCOMMISSIONED.** After the handover the user requested “please resume the work”. The environment and optional guarded sender are implemented and installed. All 91 tests pass, including recorded simulation observations, fake sender endpoints and isolated ROS2 synthetic inputs. The default launch is observation-only; no robot or gripper motion or exported-policy inference was performed.

Original request: read `SESSION_HANDOFF_2026-10-27.md` (especially §9), use the calibrated sheet/platform/robot, and prepare a real-world environment with the **exact simulation action and observation spaces**.

The user explicitly answered:

- Use `best_twist_slow_3950`, **50 Hz**, **action scale 0.03**.
- **“do not run the policy, meaning robot moving”**. Do not move the robot or gripper. Offline tests and read-only checks are allowed. Default runtime should be observation-only; shadow inference is an optional explicit mode, not something to start without explaining it.
- **Manual positioning, then reset policy history.** Do not implement or run automatic homing/reset motion.
- Latest request: **“please resume the work”**; implementation resumed and completed within the no-motion scope.

No robot/gripper commands, controller changes, remote file edits, exported-policy inference, or training were performed. ROS2 tests/launch checks used isolated domain 77 and synthetic inputs only. The user’s robot/container were already running and remain running. Remote operations were read-only introspection/source reads and copying the robot’s `hrii_robot_msgs` package into the local Docker workspace. The only live service called was read-only `controller_manager/list_controllers`. No live sensor launch or execution sender was started.

Do not commit unless the user asks. Existing simulation changes are extensive and unrelated; preserve them. No agents were used; current instructions prohibit delegation unless the user or applicable skill/instructions explicitly requests it.

## Workspaces

| Path | Purpose |
|---|---|
| `/home/hami/workspaces/deformable_isaac_ws/sheet_rl/sheet_rl` | Simulation repository; original handoff, policy, this handover |
| `/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap` | Existing calibrated real-world ROS2 package; **new implementation is here** |
| `/home/hami/workspaces/ros_two/optitrack_ws` | ROS2 Humble build/install workspace |
| `/home/hami/workspaces/deformable_isaac_ws` | ROS1 Docker workspace; image `deformable_isaac_ws:devel`, Ubuntu 20.04 / Noetic / HRII |

No applicable AGENTS.md was found for the edited locations. `IsaacLab/AGENTS.md` is outside their ancestry (read it if editing IsaacLab).

Execution conventions inherited from the handoff:

- Simulation/pure Python work: activate `/home/hami/workspaces/deformable_isaac_ws/IsaacLab/env_isaaclab/bin/activate` (`active_isaac`, Python 3.12).
- ROS2 nodes/tests: `/usr/bin/python3`, source `/opt/ros/humble/setup.bash` and the OptiTrack workspace's `install/setup.bash` (Python 3.10).
- ROS1: `docker exec docker_deformable_isaac_ws bash -ic '...'`; the container shell sets the actual remote master.
- Live ROS2 domain **25**; isolated synthetic tests **77**.
- Use `unittest`, not pytest. The expanded suite now has **91 passing tests**, including the opt-in isolated ROS2 integration.
- Rebuild `hrii_sheet_mocap` after edits; installed modules may be copies even when symlink install was used.
- Never kill processes by broad names. Track only the PID/process group of processes you start.

## Existing verified policy contract

Export directory:

```
/home/hami/workspaces/deformable_isaac_ws/sheet_rl/sheet_rl/logs/rsl_rl/sheet_platform_twist_ur10e_ppo/best_twist_slow_3950/exported_real/
```

Contains `policy.onnx`, `policy.pt`, `policy_spec.json`. The frozen normalizer and tanh squashing are inside the export. CPU ONNX Runtime 1.17.1 is installed in system Python. The actor is deterministic; do not normalize/squash a second time.

49 float32 observations, in this order:

| Slice | Meaning |
|---|---|
| `0:7` | Older applied action: 6 raw arm outputs and filtered gripper state |
| `7:14` | Most recent applied action |
| `14:41` | Nine ordered landmark positions minus tool position, base-link axes, metres |
| `41:44` | Platform top-centre minus tool position, base-link axes, metres |
| `44:48` | Platform quaternion **xyzw**, relative to base-link |
| `48` | Measured finger pad gap, metres |

Both history entries reset to `[0,0,0,0,0,0,0.5]`. Only successfully sent actions should enter live history. Shadow proposals must **not** be recorded as if applied.

7 actions:

- Arm `a[:6] * 0.03`: `(dx,dy,dz,rx,ry,rz)`; translation and axis-angle rotation in **base-link axes**.
- Rebase each step on the **measured** current tool pose, not the previous target.
- Quaternion target is `delta_quaternion * measured_quaternion` (PRE-multiplication).
- Tool point = ROS flange/tool0 origin + its rotation @ `[0,0,0.145]`.
- Convert a rotated tool target back to flange/EE; rotations must pivot about this tool point.
- Gripper: `s += 0.3 * (clip(a[6], -1, 1) - s)`, initial `s=0.5`, **open iff `s > 0`**.
- Sim `limit_scale=0.6` is a termination condition, not an action scale or actual speed limiter.
- Matching these interfaces does **not** match hardware dynamics. Simulation uses DLS IK and deliberately altered drives/effort limits. Do not claim controller/physical-response equivalence.

Sources read:

- `sheet_rl/.../sheet_twist/sheet_ur10e_env_cfg.py`: arm config around line 149, gripper around 157.
- `sheet_platform_twist_student_env_cfg.py`: `ActionHistoryObsCfg`, control dt 1/150 × decimation 3. Current `EPISODE_STEPS=1500`.
- `sheet_rl/mdp/grasp_control.py`: gripper filter and action history.
- `IsaacLab/source/isaaclab/isaaclab/utils/math.py:969`: `apply_delta_pose`, angle cutoff 1e-6, premultiplication.
- `IsaacLab/.../envs/mdp/actions/task_space_actions.py:190`: raw action scaling and measured-state-relative target.
- Existing `hrii_sheet_mocap/policy_obs.py` and `test/test_policy_obs.py`. Existing fixture `test/fixtures/ur16e_sim_episode.npz` reproduces real observation builder vs simulation within 1e-5. Reuse this fixture for new tests.

## Hardware discoveries (confirmed read-only against the live robot)

Master: `http://192.168.0.52:11311`, laptop ROS_IP `192.168.0.223`. Container `docker_deformable_isaac_ws` running on host network. SSH works with `ssh -o BatchMode=yes -o ConnectTimeout=5 192.168.0.52 ...`; configured account is `robot`.

Robot source workspace: `/home/robot/workspaces/tornado_kairos_ws`.

### Arm

- `/ur16e_a/joint_cartesian_position_controller/command` is **trajectory_msgs/JointTrajectory**, not a Cartesian command.
- `/ur16e_a/joint_cartesian_position_controller/desired_EE_pose` is **geometry_msgs/PoseStamped**. HRII Cartesian controller subscribes. Its callback **ignores header.frame_id**, so the sender must guarantee the correct base frame.
- `/ur16e_a/robot_state` is custom `hrii_robot_msgs/RobotState`.
- `/ur16e_a/controller_manager/list_controllers`: `pos_joint_traj_controller` running; `vel_joint_traj_controller` initialized. Cartesian HRII node is separate and has `controller_started: true`.
- `/ur16e_a/twist_to_pose` already advertises `desired_EE_pose`. It publishes nothing while idle, but a new sender must prevent competing command sources rather than assume it stays idle.

Read exact remote source:

```
src/hrii_ee_twist_control/scripts/twist_to_pose.py
src/hrii_robot_controllers/hrii_fixed_base_robot_controllers/src/joint_cartesian_position_controller.cpp
```

The existing twist node integrates a **persistent target**, adds feed-forward, and clamps velocities/lead. Thus simply converting a policy delta to a velocity and publishing to `cmd_twist` **does not reproduce the simulation action semantics**. Planned backend is a dedicated measured-state-relative absolute pose sender, with explicit hardware activation, watchdog and publisher conflict checks. Do not stop/change the existing twist node without a later user-authorized motion setup.

Live twist parameters read:

```
base_frame: ur16e_a_ur16e_base_link
ee_frame: ur16e_a_ur16e_EE
tcp_frame: ur16e_a_ur16e_flange
tcp_offset: [0, 0, 0.145]
rate: 100
cmd_timeout: 0.2
max_tf_age: 0.2
max_linear_velocity: 0.45
max_angular_velocity: 0.75
max_linear_lead: 0.1
max_angular_lead: 0.2
feedforward_time: 0.12
feedforward_filter: 0.05
controller_motion_linear_gain: 0.02
controller_motion_angular_gain: 0.02
stop_at_current_pose: true
```

Frames: policy base `ur16e_a_ur16e_base_link`; flange `ur16e_a_ur16e_flange`; HRII EE `ur16e_a_ur16e_EE`. **Do not assume EE equals flange or policy tool**. EE has +0.15 m in the gripper frame; account for its full fixed transform (including orientation), using TF/URDF.

The UR driver publishes `robot_program_running` and `safety_mode` as **latched, change-only** states. This was confirmed in `src/hrii_universal_robots/universal_robots_ros_driver/ur_robot_driver/src/hardware_interface.cpp`. The sender retains them while their subscriber connections remain present, and uses fresh joints/TF as the heartbeat. Requiring periodic state messages would incorrectly block a healthy robot.

### Gripper — important trap

`/ur16e_a/gripper/command` is advertised as Float64, but the live driver **does not implement `sendCommand`**. Publishing there does not operate this Robotiq.

Verified remote sources:

```
src/hrii_robot_interface/hrii_gri_interface/include/hrii_gri_interface/GripperInterfaceNode.h
src/hrii_robot_interface/hrii_gri_interface/src/GripperInterfaceBase.cpp
src/hrii_robotiq_2f85/hrii_robotiq_2f85_interface/src/Robotiq2F85.cpp
```

Use services instead (NOT called in this session):

- `/ur16e_a/gripper/set_width`: service type `hrii_robot_msgs/SetGripperData`. Despite its field name, `request.force` supplies the **width**: callback calls `setWidth(req.velocity, req.force)`. Width is decimetres in this hardware implementation: **0.8 means 0.080 m**.
- `/ur16e_a/gripper/grasp_from_outside`: same type; velocity and force are percentages, and the driver closes with force.
- `/ur16e_a/gripper/open` / `close` also exist. Full open is 0.085 m, not sim's approximately 0.080 m.
- Existing defaults: speed 100, force 30. Do not silently change robot parameters.
- Driver methods block until completion; use an edge-triggered worker, not synchronous calls in the 50 Hz control loop. Do not flood the service every tick. Handle errors and outstanding calls explicitly.
- Width feedback `/ur16e_a/gripper/width`: **decimetres**, confirmed in source (`max_width_=0.85`) and measurements (0.839 open, 0.358 on 35 mm bar). Observation = data × **0.1** metres. No need for an angle curve when physical width is available.
- `robot_relay`'s synthesized finger angle is linear and **only for RViz**. Do not pass it through the simulator's GapCurve for observations.

Service definition (remote `src/hrii_robot_msgs/srv/SetGripperData.srv`):

```
float64 velocity
float64 force
---
bool success
string message
float64 closure_width
```

**Resolved on resume:** `hrii_robot_msgs` was initially missing. The actual package was copied from the robot to `/home/hami/workspaces/deformable_isaac_ws/src/hrii_robot_msgs` and built successfully inside the existing ROS1 container. Importing the generated `SetGripperDataRequest` confirms fields `velocity` and `force`. No service types were fabricated, and no gripper service was called.

## Calibration and sensor integration

Existing saved files, already calibrated; preserve:

- `~/.ros/hrii_sheet_mocap/optitrack_frame.yaml`: base → OptiTrack, x=-0.40, y=-1.29, z=0.07 m, yaw=90°.
- `~/.ros/hrii_sheet_mocap/platform.yaml`: rigid body 2 → platform top-centre, platform height 0.148175 m.
- `~/.ros/hrii_sheet_mocap/bar.yaml`: calibrated bar on rigid body 1.
- Git snapshots in mocap package `config/saved_2026-11-17/`.

Least-squares `optitrack_to_base_link.yaml` does **not** exist. Existing `sheet_fit/key_points_base` and `platform_pose/pose_base` therefore need not exist. The installed node uses raw `/sheet_fit/key_points` and `/platform_pose/pose`, transformed through the **saved frame-editor TF**. Do not require recalibration or add a second publisher of OptiTrack's parent.

`platform.for_policy` chooses a square-equivalent yaw in ±45° and removes tilt. The installed node enables this existing mapping by default; make this explicit in docs. Exact input ordering/units do not imply the physical platform was level. Test this mapping separately from raw observation parity.

Existing launches:

- `ur16e_calibration.launch.py`: robot read-only relay, URDF TF, frame editor, optional bar/driver/RViz.
- `sheet.launch.py`: sheet fit + platform + optional driver/RViz.
- `real_env.launch.py` combines these with **scoped GroupAction**, one OptiTrack driver, headless saved frame editor, no bar GUI, and observe mode. Use `sensors:=false` for an existing complete stack, `optitrack:=false` if only the driver already runs.
- The complete live sensor stack was not launched during setup. Only `real_env.launch.py sensors:=false` was tested on isolated domain 77. Recheck live nodes before starting sensors; do not assume the graph remains empty.

## Implemented and installed files

Mocap package `/home/hami/workspaces/ros_two/optitrack_ws/src/hrii_sheet_mocap`:

- `hrii_sheet_mocap/real_env.py`: pure 49/7 contract, measured-state-relative targets, history/filter/reset, release/1500-step stop, optional CPU ONNX actor.
- `hrii_sheet_mocap/command_protocol.py`: loopback JSON protocol on 47812, session/sequence/freshness/workspace/tool-offset/lead checks, acknowledgment before history commit.
- `hrii_sheet_mocap/real_policy_node.py`: observe/shadow/execute modes; source age, receive age and skew checks; ROS2 diagnostics and reset/start/stop Trigger services. Observe loads no actor and makes no command connection; shadow records no hypothetical history. Faults require explicit reset. Ctrl-C cleanup handles launch's duplicate forwarded SIGINT.
- `ros1/ros1_policy_sender.py`: optional ROS1 sender, preview by default. Full flange-to-EE transform; gripper service calls on edges in a worker; measured pose hold on stop when available; no automatic gripper release. Explicit execute required; an extra operator XYZ box is optional per the user’s follow-up. The simulation tool floor z >= 0 remains enforced. Guards cover feedback, controller/program/safety state, competing writers and command watchdog.
- `scripts/start_ros1_policy_sender.sh`: stages only the sender and pure Python modules inside the existing container; unique PID/environment marker and owned-process cleanup. No broad process kills.
- `launch/real_env.launch.py`: calibrated feedback/sensors plus observe-mode node; sender deliberately separate.
- `test/test_real_env.py`, `test/test_policy_sender.py`, `test/test_real_ros_runtime.py`: contract, recorded observations, target math, history, fake transport/driver, watchdog/replay/conflict and isolated ROS2 tests. The ROS2 mode tests use a constant fake actor, never the exported policy.
- `setup.py`, `package.xml`: node/script/module installation and ROS dependencies.
- `README.md`: current commands, full contract and hardware commissioning limitations; obsolete finger-angle instructions corrected.

Docker workspace `/home/hami/workspaces/deformable_isaac_ws`:

- `real_env.sh`: convenient ROS2 launch wrapper; defaults to live domain 25 and **observe** mode.
- `src/hrii_robot_msgs`: copied from actual robot source and successfully built with `catkin build hrii_robot_msgs` inside the existing container. No remote edits.
- Existing image remains `deformable_isaac_ws:devel`, ID `sha256:5637121d7e64930bfd1c28ead783ef22936e59ae2fc9ce7ec25e8cdf8198fed1`, 4,865,158,381 bytes. Ubuntu 20.04 / Noetic / HRII, built earlier. Helpers are `docker_hrii_build` / `docker_hrii_run` (`dhb` / `dhr`). The message package is in the bind-mounted catkin workspace.

## Commands for the user

```bash
cd /home/hami/workspaces/deformable_isaac_ws
./real_env.sh                       # calibrated sensors + observations; no actor/commands
# Or, if the entire calibrated sensor + robot relay stack already runs:
./real_env.sh sensors:=false
```

A separate ROS2 terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/hami/workspaces/ros_two/optitrack_ws/install/setup.bash
export ROS_DOMAIN_ID=25
ros2 topic echo /real_policy/status
ros2 service call /real_policy/reset std_srvs/srv/Trigger '{}'
```

Observation topic: `/real_policy/observation`. Optional `mode:=shadow` runs inference on ROS2 diagnostics only; not started during this task. Execute additionally requires the separate explicitly enabled sender and `/real_policy/start`; nothing automatically moves at launch or reset.

## Verification actually performed

- **91 tests pass** with:
  `source /opt/ros/humble/setup.bash && HRII_ISOLATED_ROS_TEST=1 ROS_DOMAIN_ID=77 /usr/bin/python3 -m unittest discover -s test`
  in the mocap package. Includes observation replay within 1e-5, independent scipy rotation/pivot math, fake loopback sender and services, latched-state vs heartbeat checks, history failure paths, reset and stale inputs.
- `colcon build --packages-select hrii_sheet_mocap --symlink-install`: successful. Only existing setuptools/pkg_resources deprecation warning.
- ROS1 `catkin build hrii_robot_msgs --no-status`: successful.
- Installed `OnnxActor` loads the actual selected export with CPU provider and shapes `[batch,49]` / `[batch,7]`; no inference was executed.
- Both source and installed `start_ros1_policy_sender.sh` previews: successful, no ROS node/publisher/service/socket created.
- Installed `real_env.sh --show-args`: successful. Installed observation-only launch with `sensors:=false` on domain 77: starts correctly, reports waiting for sheet, exits cleanly on Ctrl-C after fixing duplicate SIGINT cleanup.
- Shell syntax and `git diff --check`: passed.
- No exported policy inference, hardware sender execution, live sensor stack, robot/gripper command or controller change. Synthetic tests own and clean up their nodes/processes. User's original Docker container remains running.

## Remaining live commissioning (not authorized in this task)

- User follow-up: “same as simulation, if doesnt exist in there leave it , I press emergency button”. Traced UR16e → UR10e → student → platform → sheet → FrankaSoft terminations: the XYZ box limits deformable nodes, not the tool. `ee_below_table` has minimum_height=0.0; the UR base z=0 and yaw-only orientation make that base tool z>=0. The sender now defaults to this floor and no extra XYZ box; optional min/max must be supplied together. It rejects measured and target positions below the floor. This is a command rejection check, whereas simulation terminates after crossing. No physical table/collision model is implied. Other checks remain intact. Do not require invented XYZ bounds again.
- Existing `/ur16e_a/twist_to_pose` advertises the target pose topic. The sender intentionally refuses competing writers. A future authorized operator session must arrange exclusive ownership; do not kill/reconfigure it automatically.
- Validate sensor readiness/calibration/latency and physical controller response before an authorized policy rollout. Matching spaces/target math does not match simulated DLS/drive/gripper dynamics.
- Confirm actual gripper service behavior under authorized commissioning. It uses worker threads because services block until complete; stop cannot cancel an already in-flight call and never adds an open command.
- Default watchdog is 100 ms, target lead limits 0.06 m/0.06 rad, joint speed stop at 60% model limits. These are software rejection checks, not a safety-rated stop or collision checker. Acknowledgment means sender accepted publication/service dispatch, not physical convergence.
- Existing tracker can fill missing markers by rigid/line rules (or retain an individual unfillable landmark); age/finite checks cannot certify each inferred landmark's physical accuracy.
- If modifying code, rebuild copied ROS2 installation and rerun relevant tests. Preserve unrelated simulation changes; do not commit unless requested.
