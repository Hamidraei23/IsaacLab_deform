"""The UR16e task (``SheetUR16eTwistStudentEnvCfg``) with a simulated RealSense D435i colour camera fixed around the
table, 0.75 m up and tilted 35 degrees down at the workspace (``d435i_camera``). Nothing else changes: the policy's observations,
rewards and terminations are the UR16e task's, so any twist checkpoint plays here unchanged, and the camera is read
only by whoever asks for it (``env.scene["d435i"].data.output["rgb"]``; ``scripts/play_actor.py --camera d435i``
shows it in its own window).

The task's real table is invisible (the viewer draws a pink success-marker box over it, green once the sheet is
placed, and the camera cannot see markers), so here the table itself is made visible in the marker's pink
(``TABLE_COLOR``): the camera then sees a table where the viewer shows one. The green success tint stays viewer-only. The floor is drawn mid grey a table height below the top, where a
real floor is (the task's physics ground plane stays at z = 0, invisible), and the camera's background is a light
blue-grey sky (digital-twin basics), so the frame shows a table standing in a room instead of black and empty areas.
The colour is authored as ``primvars:displayColor`` on the table's mesh (``spawn_table``): that is what Isaac Lab
copies into the Newton model's shape colours, which the ray tracer shades with; a PreviewSurface material is
ignored there and leaves the importer's grey.

Kept as a separate task (``Template-Sheet-Rl-Platform-TwistStudent-UR16e-Camera-v0``) so the UR16e task itself
stays as it was: a camera in the scene costs a renderer at start-up and a frame per step when read.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from .d435i_camera import d435i_camera_cfg
from .fiducial import TAG2_ID, TAG2_POS, TAG3_ID, TAG_POS, TagCfg, floor_tag_placement
from .sheet_ur16e_env_cfg import SheetUR16eTwistStudentEnvCfg


TABLE_COLOR = (.8, .5, .5)
"""The success visualiser's "failure" marker colour, which is what the viewer shows as the table."""
GROUND_COLOR = (.5, .5, .5)
"""The floor, a flat mid grey. The base task's ground-plane mesh is black and its normals face down, so the ray
tracer lit it from below; it is hidden here and a visual-only slab (``spawn_floor``) stands in for it."""


def _spawn_cuboid_with_display_color(prim_path, cfg, color, translation=None, orientation=None):
    """``spawn_cuboid``, then ``primvars:displayColor`` = ``color`` on every mesh under it: that is what Isaac Lab copies
    into the Newton model's shape colours, which the ray tracer shades with (a PreviewSurface colour is ignored)."""
    from pxr import Gf, UsdGeom  # at call time: importing pxr before Kit's SimulationApp starts crashes --viz kit

    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation)
    root = str(prim.GetPath())
    for child in [prim, *(p for p in prim.GetStage().Traverse() if str(p.GetPath()).startswith(root + "/"))]:
        if child.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(child).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return prim


def spawn_table(prim_path, cfg, translation=None, orientation=None):
    return _spawn_cuboid_with_display_color(prim_path, cfg, TABLE_COLOR, translation, orientation)


def spawn_floor(prim_path, cfg, translation=None, orientation=None):
    return _spawn_cuboid_with_display_color(prim_path, cfg, GROUND_COLOR, translation, orientation)


FLOOR_SIZE = (30., 30., .02)
"""The visual floor slab [m]: wide enough to reach the camera's 10 m far clip in every direction."""
FLOOR_GAP = .001
"""The slab's top sits this far under the table's bottom face, so the two never fight for pixels."""


def floor_top(table) -> float:
    """z of the floor's top [m]: just under the table's bottom face (the table stands on the floor; its top is z = 0,
    so the floor is a table height down, -1.05 m), not at the physics ground plane, which stays at z = 0."""
    return table.init_state.pos[2] - table.spawn.size[2] / 2. - FLOOR_GAP


@configclass
class SheetUR16eCameraEnvCfg(SheetUR16eTwistStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.d435i = d435i_camera_cfg()
        self.scene.table.spawn = self.scene.table.spawn.replace(func=spawn_table, visible=True, visual_material=None)
        self.scene.ground.spawn = self.scene.ground.spawn.replace(visible=False)  # physics only; the slab is the look
        # printed 10 x 10 cm AprilTags flat on the table, for the camera (fiducial.py): id 0 near the base
        self.scene.tag = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tag", init_state=AssetBaseCfg.InitialStateCfg(pos=TAG_POS), spawn=TagCfg())
        # a second tag of the same family on the camera's side: two depths anchor a depth map's scale and offset
        self.scene.tag2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tag2", init_state=AssetBaseCfg.InitialStateCfg(pos=TAG2_POS),
            spawn=TagCfg(tag_id=TAG2_ID))
        # a large third tag flat on the floor behind the robot, where the arm never crosses the camera's view of it:
        # placed with scripts/floor_tag_gui.py (config/floor_tag.json), else fiducial.TAG3_XY / TAG3_SIZE
        tag3_x, tag3_y, tag3_size = floor_tag_placement()
        self.scene.tag3 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tag3",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(tag3_x, tag3_y, floor_top(self.scene.table) + .0005)),
            spawn=TagCfg(tag_id=TAG3_ID, size=tag3_size))
        self.scene.floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., floor_top(self.scene.table) - FLOOR_SIZE[2] / 2.)),
            spawn=sim_utils.CuboidCfg(size=FLOOR_SIZE, func=spawn_floor, collision_props=None),
        )
