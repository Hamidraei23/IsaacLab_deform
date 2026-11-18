"""The 2F-85's force and pose control, in one place, for tuning by hand.

Edit the constants in the TUNING block. ``validate()`` runs when the task config is built and
refuses combinations that this session showed do not work; ``describe()`` prints what the
numbers mean in metres, newtons and seconds. Print it with::

    python -m sheet_rl.tasks.manager_based.sheet_twist.ur10e_gripper

How the drive works (ImplicitActuatorCfg, per hinge)::

    torque = clip(stiffness * (target - q) - damping * qd,  -effort_limit, +effort_limit)

and MJWarp caps |qd| at ``velocity_limit``. So with a 0.74 rad close travel and a stiff gain
the torque sits at the *effort limit* for almost the whole stroke: closing is bang-bang, the
effort limit is the pinch force, the velocity cap is the impact speed, and the damping only
costs top speed (under a torque cap, top speed = effort / damping).

Restrictions, each learned the hard way (logs/teleop/weird_grasp*.csv, logs/teleop/fix/):

1. PINCH CEILING. The cloth contact is 8e3 N/m on 2 mm particles. Above roughly 10--15 N at
   the pads the pads go *through* the sheet to zero gap, the hold detector (needs > 1 mm)
   drops out, and the zero-gap pinch kicks the arm or blows the solver up. 2 Nm at the
   knuckle (~35 N) failed; 0.6 Nm (~10 N) holds at 1.7--2.5 mm. Keep KNUCKLE_EFFORT <= 0.8.
2. STIFFNESS 200 WAS THE ORIGINAL BUG: it put ~7 Nm on the knuckle at a 2 mm gap. The gain
   must be stiff enough to saturate the effort limit over the stroke (bang-bang) but the
   *pinch* must be set by the effort limit, never by the gain: KNUCKLE_STIFFNESS * 0.035
   (the 2 mm gap in radians) must stay below the pinch ceiling as well.
3. FOLLOWER DAMPING IS A TAX ON THE KNUCKLE. The four followers ride at the knuckle's speed,
   so their damping torque is paid out of the knuckle's effort budget (two followers per
   side). Follower damping 0.06 halved the closing speed and made a grasp miss the sheet;
   0.03 is the working value. Keep 2 * FOLLOWER_DAMPING * VELOCITY_LIMIT under 80% of
   KNUCKLE_EFFORT.
4. THE VELOCITY CAP IS TWO THINGS. It bounds the drive, and it is what the task's
   ``joint_vel_out_of_limit`` termination compares against outside ``--free_run``. Closing
   overshoots it by ~10% (6.6 rad/s at a cap of 6), which that termination tolerates because
   it reads the same cap; a cap of 8 let a low grasp rebound fully open on impact.
5. ARMATURE SETS REBOUND AND ACCELERATION TOGETHER. 0.002 is of the order of the real finger
   links; 0.01 made the drive crawl, 0.004 slowed closing enough that a recorded grasp
   missed the sheet. Change it only with the cap.
6. OPEN ANGLE <-> GAP is the linkage formula in ``open_half_width``. The hardware maximum is
   8.7 cm at 0 rad; 0.08 rad is 8.0 cm; the pads touch at 0.82. The open half width must stay
   above the task's 3 cm ``is_open`` threshold, and it is what the reset samples from.
7. FOLLOWERS ARE NEVER DRIVEN TO A FIXED TARGET. Newton imports the 2F-85 as an open tree
   (no parallelogram); ``MimicGripperAction`` sets each follower's target to its knuckle's
   measured angle every physics step. The vendor's PhysX mimic joints and drive values are
   ignored by Newton and play no part here.
"""

import math

from isaaclab.actuators import ImplicitActuatorCfg

# ----------------------------------------------------------------------------------------------
# TUNING
# ----------------------------------------------------------------------------------------------
KNUCKLE_STIFFNESS = 8.0      # Nm/rad: stiff enough to saturate the effort limit over the stroke
KNUCKLE_DAMPING = 0.375      # Nm s/rad: light, so top speed (= effort / damping) stays high
KNUCKLE_EFFORT = 0.6         # Nm: THE PINCH. ~10 N at the pads. Ceiling ~0.8 (restriction 1)
VELOCITY_LIMIT = 6.0         # rad/s: impact speed, and the joint-speed termination's bar
ARMATURE = 0.002             # kg m^2: of the order of the real finger links

FOLLOWER_STIFFNESS = 8.0     # Nm/rad: holds the pad parallel to the knuckle
FOLLOWER_DAMPING = 0.03      # Nm s/rad: a tax on the knuckle, keep near zero (restriction 3)
FOLLOWER_EFFORT = 1.0        # Nm: none of it goes into the pinch, so it can exceed the knuckle's

OPEN_ANGLE = 0.08            # rad: 8.0 cm between the pads (restriction 6)
CLOSE_ANGLE = 0.82           # rad: pads touching
# ----------------------------------------------------------------------------------------------

PAD_LEVER = 0.057
"""Metres of pad travel per radian of knuckle angle near the shut end [m/rad]."""
OPEN_HALF_WIDTH_MAX = 0.0435484
"""Half the pad gap at zero knuckle angle, the linkage's 87 mm hardware maximum [m]."""
PINCH_CEILING_NM = 0.8
"""Knuckle torque above which the pads go through the sheet (restriction 1) [Nm]."""
IS_OPEN_HALF_WIDTH = 0.030
"""The task's ``is_open`` threshold on half width [m]; the open command must clear it."""
CLOTH_GAP_RAD = 0.035
"""The 2 mm of cloth that stops the knuckle, in radians of knuckle angle."""

KNUCKLES = ["finger_joint", "right_outer_knuckle_joint"]
"""The two driven hinges, one per side; everything else follows them."""
LEADERS = {
    "left_inner_finger_joint": "finger_joint", "left_inner_finger_knuckle_joint": "finger_joint",
    "right_inner_finger_joint": "right_outer_knuckle_joint",
    "right_inner_finger_knuckle_joint": "right_outer_knuckle_joint",
}
"""follower hinge -> the knuckle on its side whose measured angle it tracks, 1:1."""
SIGNS = {
    "finger_joint": 1., "right_outer_knuckle_joint": 1.,
    "left_inner_finger_joint": -1., "right_inner_finger_joint": 1.,
    "left_inner_finger_knuckle_joint": -1., "right_inner_finger_knuckle_joint": -1.,
}
"""Sign of each hinge's angle when the gripper closes, in this asset's joint conventions."""


def open_half_width(angle):
    """Half the pad gap at a knuckle angle, from the linkage's planar geometry [m]."""
    return OPEN_HALF_WIDTH_MAX - .03716 * (1 - math.cos(angle)) - .04343 * math.sin(angle)


def pad_force(knuckle_torque):
    """Pinch force at the pads for a knuckle torque, by the lever [N]."""
    return knuckle_torque / PAD_LEVER


def top_speed():
    """Closing speed the drive can sustain under its torque cap, before the velocity cap [rad/s]."""
    # the knuckle pays its own damping and that of the two followers on its side
    return KNUCKLE_EFFORT / (KNUCKLE_DAMPING + 2 * FOLLOWER_DAMPING)


def validate():
    """Refuse the combinations this session showed do not work."""
    problems = []
    if KNUCKLE_EFFORT > PINCH_CEILING_NM:
        problems.append(f"KNUCKLE_EFFORT {KNUCKLE_EFFORT} Nm > {PINCH_CEILING_NM}: pads go through the sheet (restriction 1)")
    if KNUCKLE_STIFFNESS * CLOTH_GAP_RAD > PINCH_CEILING_NM:
        problems.append(f"KNUCKLE_STIFFNESS {KNUCKLE_STIFFNESS} puts {KNUCKLE_STIFFNESS * CLOTH_GAP_RAD:.2f} Nm on a 2 mm gap (restriction 2)")
    if KNUCKLE_STIFFNESS * (CLOSE_ANGLE - OPEN_ANGLE) < 2 * KNUCKLE_EFFORT:
        problems.append("KNUCKLE_STIFFNESS too low to saturate the effort limit over the stroke: closing will be gain-limited and slow (restriction 2)")
    # 0.03 at a cap of 6 costs 0.36 of the 0.6 Nm and closes in 0.14 s; 0.06 (0.72 Nm, more than
    # the whole budget) halved the speed and missed a grasp. The line is drawn between them.
    if 2 * FOLLOWER_DAMPING * VELOCITY_LIMIT > .8 * KNUCKLE_EFFORT:
        problems.append(f"FOLLOWER_DAMPING {FOLLOWER_DAMPING} taxes the knuckle {2 * FOLLOWER_DAMPING * VELOCITY_LIMIT:.2f} Nm at the cap, over 80% of its effort (restriction 3)")
    if VELOCITY_LIMIT > 8.:
        problems.append(f"VELOCITY_LIMIT {VELOCITY_LIMIT} rad/s: above 8 the pads rebound open on impact (restriction 4)")
    if not 0 < OPEN_ANGLE < CLOSE_ANGLE <= .82:
        problems.append("need 0 < OPEN_ANGLE < CLOSE_ANGLE <= 0.82 rad (the hinge's limit)")
    if open_half_width(OPEN_ANGLE) <= IS_OPEN_HALF_WIDTH:
        problems.append(f"OPEN_ANGLE {OPEN_ANGLE} gives {2 * open_half_width(OPEN_ANGLE) * 100:.1f} cm, which the task does not read as open (restriction 6)")
    if ARMATURE <= 0 or ARMATURE > .01:
        problems.append("ARMATURE outside (0, 0.01]: 0.01 made the drive crawl (restriction 5)")
    if problems:
        raise ValueError("ur10e_gripper tuning:\n  " + "\n  ".join(problems))


def actuators():
    """The two actuator groups the UR10e task installs, from the tuning above."""
    validate()
    return {
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=KNUCKLES, stiffness=KNUCKLE_STIFFNESS, damping=KNUCKLE_DAMPING,
            effort_limit_sim=KNUCKLE_EFFORT, velocity_limit_sim=VELOCITY_LIMIT, armature=ARMATURE,
        ),
        "gripper_followers": ImplicitActuatorCfg(
            joint_names_expr=list(LEADERS), stiffness=FOLLOWER_STIFFNESS, damping=FOLLOWER_DAMPING,
            effort_limit_sim=FOLLOWER_EFFORT, velocity_limit_sim=VELOCITY_LIMIT, armature=ARMATURE,
        ),
    }


def describe():
    """What the tuning means, in the units a person reasons in."""
    stroke = CLOSE_ANGLE - OPEN_ANGLE
    accel = KNUCKLE_EFFORT / ARMATURE
    v = min(top_speed(), VELOCITY_LIMIT)
    t_close = v / accel + max(stroke - .5 * v * v / accel, 0.) / v
    return "\n".join([
        f"open command {OPEN_ANGLE} rad -> {2 * open_half_width(OPEN_ANGLE) * 100:.2f} cm between the pads"
        f" (hardware max {2 * OPEN_HALF_WIDTH_MAX * 100:.1f} cm)",
        f"close command {CLOSE_ANGLE} rad -> {2 * max(open_half_width(CLOSE_ANGLE), 0) * 1000:.1f} mm gap in free air",
        f"pinch: effort {KNUCKLE_EFFORT} Nm -> about {pad_force(KNUCKLE_EFFORT):.0f} N at the pads"
        f" (ceiling {PINCH_CEILING_NM} Nm / {pad_force(PINCH_CEILING_NM):.0f} N; gain alone gives"
        f" {pad_force(KNUCKLE_STIFFNESS * CLOTH_GAP_RAD):.1f} N on a 2 mm gap)",
        f"closing: sustained speed {top_speed():.1f} rad/s before the cap, cap {VELOCITY_LIMIT} rad/s,"
        f" acceleration {accel:.0f} rad/s^2 -> about {t_close:.2f} s / {t_close / .02:.0f} steps at 50 Hz",
        f"followers tax the knuckle {2 * FOLLOWER_DAMPING * VELOCITY_LIMIT:.2f} Nm at the cap"
        f" ({100 * 2 * FOLLOWER_DAMPING * VELOCITY_LIMIT / KNUCKLE_EFFORT:.0f}% of its effort)",
    ])


if __name__ == "__main__":
    validate()
    print(describe())
