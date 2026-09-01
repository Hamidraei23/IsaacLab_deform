# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation variant of the sheet task: the mannequin arm is a sculpted arm-and-hand mesh.

**For evaluation, not training.** The arm collides as a full triangle mesh, so the cloth settles on
the real sculpt -- knuckles, tendons and the gaps between fingers -- instead of on a smooth capsule.
That is the point of the environment and also its cost: Newton resolves cloth-vs-mesh contact with a
BVH closest-point query per particle per solver iteration, against 34,288 triangles. Run it to watch
and to score a policy; train on ``Template-Sheet-Rl-v0``.

Nothing about the *observation* changes, which is what makes a checkpoint from the base task load
and behave here. The policy sees joint state, nine sheet landmarks, the 7-dim goal command and its
last action -- none of which reference the arm's shape. The arm reaches the policy only through the
goal command, which is derived from the arm's *task frame*.

That frame is the load-bearing detail, and it is deliberately not the body's root pose. Four of the
command's seven dimensions are the frame's orientation and the other three are a point lifted off
the arm along its +z, so anything that turns the body -- including a purely cosmetic re-mounting of
the asset -- rewrites the observation wholesale and a base-task checkpoint stops making sense.
``ARM_ROLL_QUAT`` below is declared to the command as ``arm_mount_quat`` for exactly that reason:
the body carries the roll, the task frame does not.

What is set below is the *geometry the reward terms assume*. They model the arm analytically rather
than reading the collider, so the sculpt has to be described to them: a capsule running elbow to
wrist, centred on the body's root. The mesh's task frame is built to make that true -- its origin is
the midpoint of the elbow-to-wrist segment -- so the capsule the rewards imagine and the surface the
cloth lands on are concentric.

The capsule and hand it replaces are both gone. The mesh is spawned at the same ``MannequinArm``
prim path under the same ``mannequin_arm`` scene entity, so the command term, every reward that
reads ``arm_cfg``, and the reset event all resolve without a change outside this file.

Registered as ``Template-Sheet-Rl-Hand-v0``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import NewtonCfg

from .mdp import simulation_diverged

from .sheet_rl_env_cfg import (
    TARGET_BAND_WIDTH,
    SheetRlEnvCfg,
    SheetSceneCfg,
    SheetScenePresetCfg,
)

##
# Arm-and-hand asset
##

ARM_AND_HAND_USD_PATH = str(
    Path(__file__).resolve().parents[6] / "hand_model" / "arm_and_hand_arm_frame.usda"
)
"""The converted sculpt, resolved relative to the repository root so the path survives a re-clone.

Points at the *task-frame wrapper* rather than at ``arm_and_hand.usd`` itself. The canonical asset
carries the geometry, the material and the physics schemas, and puts its origin at the wrist, which
is the right landmark for an asset. The wrapper is a few-hundred-byte layer that references it and
slides the mesh along its own axis so the origin lands on the midpoint of the elbow-to-wrist
segment -- the centre of the capsule the rewards below assume.

The offset is authored on the mesh inside the reference rather than on the wrapper's own root, and
that detail is load-bearing: the spawner writes ``xformOp:translate`` on the prim it references the
asset onto, and USD *replaces* a referenced transform rather than composing with it, so an offset
left on the root would be silently erased by the spawn position.

Both files are written by ``hand_model/convert_arm_and_hand.py``. The mesh arrives Z-up and in
metres with the scale baked into the geometry, so the spawner asks for no scale -- the lesson
``hand.usd`` encoded, where a spawner-side scale was dropped by the kit renderer and produced a
219-metre hand.
"""

ARM_ROLL_QUAT = (0.0, 1.0, 0.0, 0.0)
"""180 deg roll about the limb's own long axis, as ``(w, x, y, z)``.

+X is the axis running elbow -> wrist -> fingertips: the task frame is built with the forearm along
it, ``elbowX_m = -0.211`` to ``fingertipX_m = +0.396``, and no other axis is even approximately
aligned with the limb. Rotating about it is therefore a pure roll -- it turns the hand over without
moving the arm off the spot -- and half a turn is ``(cos 90 deg, sin 90 deg, 0, 0) = (0, 1, 0, 0)``.
The sculpt is authored palm-down, so this presents the palm and the inside of the forearm upwards.

Declared to the goal command as ``arm_mount_quat`` below, which is what keeps this presentational.
The command publishes the arm's orientation as four of its seven dimensions and lifts the goal off
the arm along that frame's +z, so left undeclared this roll would hand the policy a quaternion it
never met in training and drive the goal point through the limb to its underside -- and a
checkpoint from the base task, which is the whole reason this environment exists, would come apart.
"""

ARM_REST_HEIGHT = 0.0560
"""Height of the arm's root above the table so the limb rests on it rather than sinking in [m].

The mesh's lowest point sits this far below its own origin -- the origin is on the forearm axis, not
on the skin.

This is the *rolled* figure, and it is not the ``restHeight_m`` of 0.0655 recorded in the wrapper's
``customLayerData``: that one measures the composed bounding box as authored, palm-down. A limb is
thicker through the palm than across the back of the hand -- the box runs -0.0655 to +0.0560 in z --
so turning it over swaps which face is down and brings the root nearly a centimetre closer to the
table. Left at 0.0655 the rolled arm would hover; taken from the box's *max* z it seats as before.

This is the one thing the roll does change for the policy, and it is a 9.5 mm drop of the arm and
the goal that rides on it -- the same kind of displacement the reset's own xy jitter already applies
every episode, not the frame change that breaks a checkpoint. Pin it back to 0.0655 for a
bit-identical observation, at the cost of an arm floating just clear of the table.

The reset event masks the z component of its random offset to zero and composes its yaw onto the
configured spawn orientation, so this height and the roll above both survive every reset.
"""

##
# The capsule the observation and rewards assume
##

ARM_CAPSULE_RADIUS = 0.0427
"""Radius of the elbow-to-wrist capsule the reward terms model the arm as [m].

The sculpt is a real forearm and tapers -- 0.026 m at the wrist to 0.059 m at the elbow -- so no
single radius describes it. This is the mean over the forearm's slices, and it replaces the base
task's ``MANNEQUIN_ARM_RADIUS`` of 0.04 everywhere the rewards use it.

Only the *reward geometry* is a capsule. The collider is the triangle mesh, so contact happens
against the true surface, and the two disagree by more than the averaging suggests: measured over
the band's reach the gap is 1.5 cm on average, and runs from -3.9 cm where the elbow swells past the
capsule to +4.3 cm where the capsule stands proud of the narrowing wrist.

That is inherent to describing a tapered limb with one cylinder, and it is the main reason this
environment scores rather than trains. Concretely: ``drape_coverage`` samples its band on the
*imagined* capsule, so towards the elbow it can sample points that sit inside the real arm, and
cloth resting correctly on the skin there reads as further away than it is.
"""

ARM_CAPSULE_LENGTH = 0.3368
"""Cylindrical section of that capsule [m]; total span is length + 2 * radius.

Sized so the span recovers the sculpt's elbow-to-wrist length of 0.4222 m exactly. Combined with the
task frame's origin at the segment's midpoint, the capsule the rewards imagine runs from the elbow
to the wrist and stops there -- it does not reach into the hand, which is what the base task's
``MANNEQUIN_ARM_LENGTH`` would have done at this scale.

Both figures come from ``capsuleRadius_m`` / ``capsuleLength_m`` in the wrapper's
``customLayerData``, so they follow the asset if it is ever regenerated.
"""

ARM_BAND_TRAVEL = 0.5 * (ARM_CAPSULE_LENGTH - TARGET_BAND_WIDTH)
"""Half-range the red band may slide along the arm [m].

The base task's own formula, ``(arm_length - band_width) / 2``, applied to the capsule above --
which is what keeps the red region behaving as it does in the main environment rather than being
special-cased here. It works out at 0.118 m, so the band stays inside the forearm and never runs
onto the wrist or the elbow cap.

One honest consequence: the band is a constant-radius sleeve and the forearm is not. Across the full
travel the skin runs roughly 0.038 to 0.059 m, so a 0.044 m sleeve sits proud near the wrist and
sinks into the arm towards the elbow. Narrowing the travel would hide that, at the cost of sampling
the goal from a smaller set of places than the base task does.
"""

BAND_MARKER_RADIUS = 0.0219
"""Radius of the red sleeve [m]. Half of what the base task's rule would give.

The base rule is ``arm_radius + 0.001``, which here would be 0.0437 and sized to read as paint on
the surface. Halved, the sleeve sits *inside* a forearm whose radius runs 0.038 to 0.059 m, so it is
sunk into the limb rather than wrapped around it -- asked for deliberately, to stop it reading as a
hoop standing off the arm.

Being buried is the point, and also the consequence: with an opaque arm the sleeve is not visible
from outside at all. Set it back to ``ARM_CAPSULE_RADIUS + 0.001`` to get the painted band back.
"""

ARM_CENTERLINE_OFFSETS = (
    (-0.200, +0.0184, +0.0039),
    (-0.180, +0.0151, -0.0009),
    (-0.160, +0.0155, -0.0054),
    (-0.140, +0.0170, -0.0031),
    (-0.120, +0.0121, -0.0005),
    (-0.100, +0.0080, +0.0032),
    (-0.080, +0.0007, +0.0047),
    (-0.060, -0.0014, +0.0054),
    (-0.040, -0.0047, +0.0016),
    (-0.020, -0.0056, -0.0009),
    (+0.000, -0.0101, -0.0014),
    (+0.020, -0.0177, +0.0004),
    (+0.040, -0.0196, +0.0007),
    (+0.060, -0.0226, +0.0023),
    (+0.080, -0.0273, +0.0028),
    (+0.100, -0.0281, +0.0005),
    (+0.120, -0.0255, -0.0032),
    (+0.140, -0.0240, -0.0034),
    (+0.160, -0.0238, -0.0063),
    (+0.180, -0.0216, -0.0083),
    (+0.200, -0.0166, -0.0106),
)
"""The limb's real centre line in the arm's frame: ``(x, y, z)`` every 2 cm [m].

Measured as the centroid of the mesh's cross-section at each station. The sculpt is a relaxed human
arm, so it curves, and its centre line strays from the root frame's x-axis by 17 mm on average and
28 mm at worst -- on a limb of radius 0.043 m, enough that a sleeve drawn on the axis visibly hangs
off one side instead of encircling the arm.

Regenerate with ``hand_model/convert_arm_and_hand.py``, which records the same samples under
``centerlineOffsets`` in the wrapper's ``customLayerData``.
"""

PARTICLE_CONTACT_BUFFER = 8192
"""Rigid-body-vs-particle contact buffer for the soft solver.

Raised from the base task's 1024. That figure was sized for a smooth capsule; full-surface contact
against a 34k-triangle sculpt generates far more contacts per particle, and an undersized buffer
drops them, which shows up as cloth sinking through the arm rather than as an error. This is
headroom for evaluation rather than a measured figure -- if Newton reports overflow, raise it again.
"""

TRIANGLE_PAIR_BUFFER = 8_000_000
"""Triangle-pair buffer for the rigid-soft collision pipeline.

The second buffer the capsule never needed, and the one :data:`PARTICLE_CONTACT_BUFFER` does not
cover: that sizes the *solver's* contact list, this sizes the *narrow phase's* candidate list, and
Newton defaults it to 1,000,000 for the whole scene rather than per environment. A capsule is an
analytic shape with no triangles to pair against, so the default has never been approached before;
with the sculpt and ``enable_rigid_soft_full_surface_contact`` it is exceeded roughly threefold at
1024 environments -- a measured 2,960,190 -- and Newton reports the overflow as a warning, drops the
excess pairs and carries on.

That is the dangerous part. Dropping narrow-phase candidates does not raise; it lets cloth
interpenetrate the arm and the VBD solver then resolves an impossible configuration, which diverges
into NaN nodal positions on some environments. Because the observations expose only the nine
landmark nodes while the rewards read all eighty-one, the first thing that notices is the reward,
and training dies on rsl_rl's NaN check with nothing wrong in the reward code.

Sized as ~7,800 per environment at 1024, against the ~2,900 measured, so the headroom survives a
denser configuration than the one that happened to be sampled. It scales with environment count, so
a larger batch may need it raised again; the symptom to watch for is the warning itself, which
appears long before the NaN does.
"""


##
# Scene
##


@configclass
class SheetArmMeshSceneCfg(SheetSceneCfg):
    """The base scene with the capsule swapped for the sculpted arm.

    Overriding ``mannequin_arm`` rather than adding a second entity is what keeps the rest of the
    task working: the prim path, the scene-entity name and the root frame are all unchanged, so
    nothing downstream can tell the difference except by looking.
    """

    mannequin_arm: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MannequinArm",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.15, ARM_REST_HEIGHT), rot=ARM_ROLL_QUAT
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=ARM_AND_HAND_USD_PATH,
            # static: kinematic and gravity-free, exactly as the capsule was. The arm must not fall
            # and the sheet must not be able to shove it around.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            # a body without mass is imported as static geometry and never registers with Newton,
            # leaving the RigidObject wrapper with nothing to find. Irrelevant for a kinematic body;
            # its presence is what matters. The asset authors 2.0 kg itself, this pins it.
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            # Switches on the triangle-mesh collider the asset carries. It can only *modify* what is
            # already authored -- Isaac Lab's writers return False when the schema is absent -- so
            # the ``physics:approximation = "none"`` that makes this the real triangles rather than
            # a hull is baked into the USD by the converter, not requested here.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # grippier than the table (0.1) so a draped sheet settles instead of sliding straight
            # off, carried over from the capsule
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
            # deliberately no ``visual_material``: the capsule had a flat skin tone here, and setting
            # one would override the sculpt's base-colour, roughness and normal maps with a solid
            # colour, throwing away the reason for using the mesh at all.
        ),
    )


@configclass
class SheetArmMeshScenePresetCfg(SheetScenePresetCfg):
    # Far fewer environments than the base task's 2048. Mesh contact is the expensive part of this
    # environment, and evaluation does not need the batch -- override with ``--num_envs`` to push it.
    newton_mjwarp_vbd_proxy: SheetArmMeshSceneCfg = SheetArmMeshSceneCfg(
        num_envs=64, env_spacing=2.0, replicate_physics=True
    )

    default = newton_mjwarp_vbd_proxy


##
# Environment
##


@configclass
class SheetRlHandEnvCfg(SheetRlEnvCfg):
    """The sheet task with a sculpted, mesh-collided arm and hand. Evaluation only."""

    scene: SheetArmMeshScenePresetCfg = SheetArmMeshScenePresetCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- reset an environment the solver has lost.
        #
        # The mesh arm occasionally produces a state with no solution -- cloth at thousands of
        # metres per second and NaN joint positions, on the step after a reset, in about one
        # environment in a thousand. The inherited ``deformable_out_of_bounds`` cannot catch it:
        # its test is a pair of comparisons, and every comparison against NaN is False, so a sheet
        # that has gone NaN reads as inside the workspace. Without this the environment runs on in
        # a state it can never recover from.
        self.terminations.diverged = DoneTerm(func=simulation_diverged)

        # -- there is no capsule here to resize.
        #
        # ``randomize_arm_radius`` writes a radius into Newton's ``shape_scale`` for the one shape
        # of type ``CAPSULE`` under each ``MannequinArm`` body. This task replaces that capsule with
        # the sculpted triangle mesh, at the same prim path, so the term finds nothing and raises
        # during ``startup`` -- before the policy is ever loaded. Dropping it is the whole fix, and
        # nothing is lost: a mesh has no radius to draw, and the thickness this task is *for* is the
        # sculpt's own.
        #
        # Setting ``jitter`` to 0.0 does not do it. The term raises on the empty lookup before it
        # reads the jitter at all, so the escape hatch is removing it, not zeroing it.
        self.events.randomize_arm_radius = None

        # -- tell the reward terms what shape the arm is.
        #
        # They model it analytically from the body's root pose rather than reading the collider, so
        # each one that carries the capsule's dimensions has to be re-pointed at the sculpt's.
        # Missing one would leave that term scoring against a 0.04 m capsule while the cloth lands
        # on a 0.043 m arm.
        self.rewards.drape_coverage.params["band_radius"] = ARM_CAPSULE_RADIUS
        for term in (self.rewards.finger_touches_arm, self.rewards.dropped_on_table):
            term.params["arm_radius"] = ARM_CAPSULE_RADIUS
            term.params["arm_length"] = ARM_CAPSULE_LENGTH

        # -- the red band. Travel is left on the base task's own formula; only how the sleeve is
        # sized and where it is centred differ, for the reasons on the two constants above.
        command = self.commands.deformable_pose
        # Tell the command the arm is mounted rolled, so it keeps working in the frame the arm would
        # have had unrolled. This is what holds the 7-dim goal observation fixed under the roll --
        # without it the published quaternion turns over with the body and the goal's lift points
        # down into the table. Must match ``init_state.rot`` on the spawn above.
        command.arm_mount_quat = ARM_ROLL_QUAT
        command.ranges.pos_x = (-ARM_BAND_TRAVEL, ARM_BAND_TRAVEL)
        command.band_visualizer_cfg.markers["band"].radius = BAND_MARKER_RADIUS
        command.band_visualizer_cfg.markers["band"].height = TARGET_BAND_WIDTH
        command.band_centerline_offsets = ARM_CENTERLINE_OFFSETS

        # -- put the arm back in the solver.
        #
        # The base task lists it twice: once as a body owned by the rigid entry, and once in the
        # proxy mapping that pushes rigid bodies into the soft solver so the sheet can touch them.
        # Both are inherited unchanged; the loop below only widens the contact buffer, which was
        # sized for a capsule.
        for value in vars(self.sim.physics).values():
            if not isinstance(value, NewtonCfg):
                continue
            # ``default`` may or may not be a distinct copy of the preset it aliases, so every
            # NewtonCfg the preset exposes is visited rather than just the named one
            for entry in getattr(value.solver_cfg, "entries", []) or []:
                solver = entry.solver_cfg
                if hasattr(solver, "rigid_body_particle_contact_buffer_size"):
                    solver.rigid_body_particle_contact_buffer_size = PARTICLE_CONTACT_BUFFER
            # The narrow phase's own buffer, which lives on the proxy mapping rather than on a
            # solver entry and so is missed entirely by the loop above. Left at Newton's scene-wide
            # default it overflows at large batch sizes, silently drops contacts, and surfaces as a
            # NaN reward rather than as an error -- see ``TRIANGLE_PAIR_BUFFER``.
            for proxy in getattr(value.solver_cfg, "proxies", []) or []:
                pipeline = getattr(proxy, "collision_pipeline", None)
                if pipeline is not None and hasattr(pipeline, "max_triangle_pairs"):
                    pipeline.max_triangle_pairs = TRIANGLE_PAIR_BUFFER
