"""Physical pad geometry for grippers whose joints measure angles, not jaw travel."""

import torch

from isaaclab.utils.math import quat_apply


def pad_positions(env, robot):
    geometry = getattr(getattr(env, "cfg", None), "finger_pad_geometry", None)
    if geometry is None:
        return None
    points = []
    for name, offset in geometry.items():
        pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index(name)]
        offset = torch.tensor(offset, device=pose.device).expand(len(pose), 3)
        points.append(pose[:, :3] + quat_apply(pose[:, 3:7], offset))
    return torch.stack(points, 1)


def finger_half_width(env, finger_cfg):
    """Half the physical pad opening [m]; retains Franka's original joint reading."""
    robot = env.scene[finger_cfg.name]
    points = pad_positions(env, robot)
    if points is None:
        return robot.data.joint_pos.torch[:, finger_cfg.joint_ids].mean(-1)
    return 0.5 * (points[:, 1] - points[:, 0]).norm(dim=-1)
