"""A simulated Intel RealSense D435i colour camera, fixed around the table and tilted down at the workspace.

Only the colour stream is modelled, in its 640 x 480 mode (``D435I_WIDTH`` x ``D435I_HEIGHT``): 55 x 42.6 degrees of
view, the 4:3 crop the camera makes of its 69 x 42-degree imager, from the mode's factory intrinsics
(``D435I_FOCAL_PX``). Any 4:3 size keeps that view. It renders through
Newton's Warp ray tracer, so it works with the Newton physics backend and the ``newton_gl`` viewer, without Kit.

The camera sits at ``pos`` in the environment frame (the table top is z = 0, the robot base at x = -0.12, the
sheet slot and the platform around x = 0.5) and looks along ``yaw_deg`` tilted ``pitch_deg`` down: the defaults
put it just past the table's front edge, facing the robot, 35 degrees down, so the slot, the platform and the
arm's working volume fill the frame.

Override the pose from the command line with ``env.scene.d435i.offset.pos=[x,y,z]`` and
``env.scene.d435i.offset.rot=[x,y,z,w]`` (``camera_rotation`` gives the quaternion for a yaw and a pitch).
"""

import math

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

D435I_WIDTH, D435I_HEIGHT = 640, 480
"""The colour stream's 640 x 480 mode (user's choice): a 4:3 crop of the 16:9 imager."""
D435I_FOCAL_PX = 615.
"""Focal length of that mode in pixels (fx = fy, the D435i's factory intrinsics at 640 x 480, about 615 px):
55 x 42.6 degrees of view, the full 42-degree height of the 16:9 imager and a 4:3 crop of its 69-degree width."""
D435I_HFOV_DEG = math.degrees(2. * math.atan(D435I_WIDTH / (2. * D435I_FOCAL_PX)))
"""Horizontal field of view [deg] at 640 x 480: about 55."""
D435I_APERTURE = 20.955
"""Isaac's default horizontal aperture [mm]; the focal length below is expressed against it."""
D435I_FOCAL_LENGTH = D435I_APERTURE / (2. * math.tan(math.radians(D435I_HFOV_DEG) / 2.))
"""Pinhole focal length [mm, against D435I_APERTURE] that gives D435I_HFOV_DEG; the vertical view follows from the
4:3 image (Isaac derives the vertical aperture from the aspect ratio)."""
D435I_CLIPPING = (.1, 10.)
"""Near and far clip [m]: the colour stream is good from 0.1 m; the far clip bounds the ray tracer (the floor is
drawn out to it, then the background colour takes over)."""
D435I_BACKGROUND = (.72, .80, .88)
"""What rays that hit nothing show, a light blue-grey "sky": a room-like backdrop rather than the renderer's
default empty light grey (digital-twin basics, user request)."""

D435I_POS = (1.2, 0., .75)
"""Default camera position in the environment frame [m]: past the table's front edge (x = 1.15), on the
table's centre line, 75 cm above the top (45 cm until 2026-11-04)."""
D435I_YAW_DEG = 180.
"""Default heading [deg]: looking along -x, back at the robot."""
D435I_PITCH_DEG = 35.
"""Default downward tilt [deg]. The first placement was 0.45 m up and 20 degrees down (the tag and DA3 figures in the
README were measured there); on 2026-11-04 the user raised the camera 30 cm and tilted it 15 degrees further,
so the optical axis meets the table 1.07 m ahead of the camera (x = 0.13, between the slot and the base) and the
frame spans x from 0.70 (its bottom edge) to far behind the robot."""


def camera_rotation(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w), world convention (forward +x, up +z), for a heading of ``yaw_deg`` about +z and a
    tilt of ``pitch_deg`` down from the horizontal.

    Yaw first about the world's z, then the tilt about the turned y axis: ``q = q_yaw * q_pitch``.
    """
    half_yaw, half_pitch = math.radians(yaw_deg) / 2., math.radians(pitch_deg) / 2.
    ay, by, cy, dy = 0., 0., math.sin(half_yaw), math.cos(half_yaw)  # q_yaw as (x, y, z, w)
    ap, bp, cp, dp = 0., math.sin(half_pitch), 0., math.cos(half_pitch)  # q_pitch
    # Hamilton product q_yaw * q_pitch, both as (x, y, z, w)
    x = dy * ap + ay * dp + by * cp - cy * bp
    y = dy * bp - ay * cp + by * dp + cy * ap
    z = dy * cp + ay * bp - by * ap + cy * dp
    w = dy * dp - ay * ap - by * bp - cy * cp
    return (x, y, z, w)


def d435i_camera_cfg(pos=D435I_POS, yaw_deg=D435I_YAW_DEG, pitch_deg=D435I_PITCH_DEG, width=D435I_WIDTH,
                     height=D435I_HEIGHT, prim_path="{ENV_REGEX_NS}/D435i") -> CameraCfg:
    """The fixed colour camera as a scene entity (``scene.d435i``); see the module docstring."""
    if width * 3 != height * 4:
        raise ValueError(f"The D435i's 640 x 480 colour mode is 4:3; {width} x {height} is not")
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.,
        width=width,
        height=height,
        data_types=["rgb"],
        background_color=D435I_BACKGROUND,
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=PinholeCameraCfg(focal_length=D435I_FOCAL_LENGTH, horizontal_aperture=D435I_APERTURE,
                               clipping_range=D435I_CLIPPING),
        offset=CameraCfg.OffsetCfg(pos=tuple(pos), rot=camera_rotation(yaw_deg, pitch_deg), convention="world"),
    )
