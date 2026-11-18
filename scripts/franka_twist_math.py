"""Tensor operations for comparing a joint policy with Cartesian twist control.

The inferred twist is a command, not measured velocity: the joint target error
is divided by one policy interval. The inverse mapping drops joint null-space
motion and uses damping. It is not an exact conversion of the original policy.
"""

import torch


def shift_jacobian(jacobian: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Shift a geometric Jacobian to a tool point; offset [m] shares its frame."""
    shifted = jacobian.clone()
    angular_columns = jacobian[:, 3:, :].transpose(1, 2)
    shifted[:, :3, :] += torch.linalg.cross(
        angular_columns, offset[:, None, :].expand_as(angular_columns), dim=-1
    ).transpose(1, 2)
    return shifted


def limit_twist(twist: torch.Tensor, linear_limit: float, angular_limit: float) -> torch.Tensor:
    """Limit vector norms [m/s, rad/s]; a zero limit disables that bound."""
    limited = twist.clone()
    for section, limit in ((slice(0, 3), linear_limit), (slice(3, 6), angular_limit)):
        if limit > 0:
            magnitude = torch.linalg.vector_norm(limited[:, section], dim=-1, keepdim=True)
            limited[:, section] *= (limit / magnitude.clamp_min(1e-12)).clamp(max=1.0)
    return limited


def joint_target_to_twist(
    jacobian: torch.Tensor, joint_target: torch.Tensor, joint_position: torch.Tensor, dt: float
) -> torch.Tensor:
    """Infer tool-point linear/angular velocity [m/s, rad/s] over dt [s]."""
    return torch.bmm(jacobian, ((joint_target - joint_position) / dt).unsqueeze(-1)).squeeze(-1)


def twist_to_joint_target(
    jacobian: torch.Tensor, twist: torch.Tensor, joint_position: torch.Tensor, dt: float, damping: float
) -> torch.Tensor:
    """Damped resolved-rate IK, integrated over dt [s] into joint targets [rad]."""
    transpose = jacobian.transpose(1, 2)
    eye = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device)
    dual = torch.linalg.solve(jacobian @ transpose + damping**2 * eye, twist.unsqueeze(-1))
    return joint_position + dt * (transpose @ dual).squeeze(-1)
