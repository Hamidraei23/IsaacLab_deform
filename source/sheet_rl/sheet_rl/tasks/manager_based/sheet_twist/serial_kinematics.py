"""Batched rigid-chain FK/IK for policy transfer; no extra simulated robot.

Joint frames come from the same USD assets as the simulator. USD is imported only
when constructing a chain, after the application has selected its USD runtime.
"""

import torch

from isaaclab.utils.math import (
    compute_pose_error, matrix_from_quat, quat_from_matrix,
)


class SerialKinematics:
    """Inputs and outputs are float32; the chain itself is evaluated in float64.

    Float32 is not enough once TF32 matmuls are on (Isaac Lab's rsl_rl training launcher turns them on): the
    chained 4x4 products lose orthonormality, ``quat_from_matrix`` returns NaN and the IK diverges -- the UR10e
    reset put NaN joints in every environment. TF32 never applies to float64.
    """

    def __init__(self, frames, joint_names, limits, tip, device="cpu"):
        self.frames = [(torch.tensor(a, device=device, dtype=torch.float64),
                        torch.tensor(b, device=device, dtype=torch.float64), index,
                        torch.tensor(axis, device=device, dtype=torch.float64))
                       for a, b, index, axis in frames]
        self.joint_names = joint_names
        self.limits = torch.tensor(limits, device=device, dtype=torch.float32).T
        self.tip = torch.tensor(tip, device=device, dtype=torch.float64)

    @classmethod
    def from_usd(cls, path, joint_names, end_body, tip_pos, tip_rot=(0., 0., 0., 1.),
                 variants=None, device="cpu"):
        import math
        from pxr import Usd, UsdPhysics
        from isaaclab.utils.assets import retrieve_file_path

        stage = Usd.Stage.Open(retrieve_file_path(path))
        for name, value in (variants or {}).items():
            stage.GetDefaultPrim().GetVariantSet(name).SetVariantSelection(value)
        joints = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                children = joint.GetBody1Rel().GetTargets()
                if children:
                    joints[str(children[0])] = joint
        candidates = [p for p in joints if p.endswith('/' + end_body) or p == end_body]
        if len(candidates) != 1:
            raise ValueError(f"Expected one end body {end_body!r}, found {candidates}")

        def transform(pos, rot):
            q = torch.tensor([*rot.GetImaginary(), rot.GetReal()])
            t = torch.eye(4)
            t[:3, :3] = matrix_from_quat(q)
            t[:3, 3] = torch.tensor(list(pos))
            return t

        chain, limits = [], [None] * len(joint_names)
        body = candidates[0]
        while body in joints:
            joint = joints[body]
            parents = joint.GetBody0Rel().GetTargets()
            if not parents:  # fixed world joint; FK is expressed in the robot root frame
                break
            name = joint.GetPrim().GetName()
            index = joint_names.index(name) if name in joint_names else None
            axis = [0., 0., 1.]
            if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                if index is None:
                    raise ValueError(f"Unspecified revolute joint in chain: {name}")
                rev = UsdPhysics.RevoluteJoint(joint)
                axis = [float(c == rev.GetAxisAttr().Get()) for c in "XYZ"]
                limits[index] = [math.radians(rev.GetLowerLimitAttr().Get()),
                                 math.radians(rev.GetUpperLimitAttr().Get())]
            a = transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            b = transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()).inverse()
            chain.append((a.tolist(), b.tolist(), index, axis))
            body = str(parents[0])
        if any(limit is None for limit in limits):
            raise ValueError(f"Missing arm joints in USD chain to {end_body}")
        tip = torch.eye(4)
        tip[:3, :3] = matrix_from_quat(torch.tensor(tip_rot))
        tip[:3, 3] = torch.tensor(tip_pos)
        return cls(chain[::-1], joint_names, limits, tip.tolist(), device)

    def forward(self, q):
        dtype = q.dtype
        position, rotation, jacobian = self._forward(q.double())
        return position.to(dtype), rotation.to(dtype), jacobian.to(dtype)

    def _forward(self, q):
        n = len(q)
        pose = torch.eye(4, device=q.device, dtype=q.dtype).expand(n, 4, 4).clone()
        origins, axes = [None] * q.shape[1], [None] * q.shape[1]
        for a, b, index, axis in self.frames:
            pose = pose @ a
            if index is not None:
                origins[index] = pose[:, :3, 3].clone()
                axes[index] = (pose[:, :3, :3] @ axis).clone()
                half = q[:, index] * 0.5
                quat = torch.cat((half.sin()[:, None] * axis, half.cos()[:, None]), -1)
                rotation = torch.eye(4, device=q.device, dtype=q.dtype).expand(n, 4, 4).clone()
                rotation[:, :3, :3] = matrix_from_quat(quat)
                pose = pose @ rotation
            pose = pose @ b
        pose = pose @ self.tip
        angular = torch.stack(axes, -1)
        lever = pose[:, :3, 3, None] - torch.stack(origins, -1)
        linear = torch.linalg.cross(angular, lever, dim=1)
        return pose[:, :3, 3], quat_from_matrix(pose[:, :3, :3]), torch.cat((linear, angular), 1)

    @staticmethod
    def inverse_velocity(jacobian, velocity, damping=0.02):
        dtype = velocity.dtype
        jacobian, velocity = jacobian.double(), velocity.double()
        jt = jacobian.transpose(1, 2)
        eye = torch.eye(6, device=jacobian.device, dtype=torch.float64)
        return (jt @ torch.linalg.solve(jacobian @ jt + damping**2 * eye, velocity[..., None])).squeeze(-1).to(dtype)

    def solve(self, position, rotation, seed, iterations=12, damping=0.02):
        position, rotation = position.double(), rotation.double()
        lower, upper = self.limits[0].double(), self.limits[1].double()
        q = seed.double()
        for _ in range(iterations):
            p, r, jac = self._forward(q)
            dp, dr = compute_pose_error(p, r, position, rotation, rot_error_type="axis_angle")
            dq = self.inverse_velocity(jac, torch.cat((dp, dr), -1), damping)
            q = (q + dq.clamp(-0.2, 0.2)).clamp(lower, upper)
        return q.to(seed.dtype)
