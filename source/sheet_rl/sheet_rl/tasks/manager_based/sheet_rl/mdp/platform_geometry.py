"""Torch-only surface coverage geometry, independent of the simulator."""

import torch


def sheet_triangles(resolution, device):
    """Match MeshRectangleCfg's alternating diagonals and row-major node order."""
    nx, ny = resolution
    faces = []
    for y in range(ny):
        for x in range(nx):
            a = y * (nx + 1) + x
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            faces.extend(((a, b, c), (b, d, c)) if (x % 2) != (y % 2)
                         else ((a, b, d), (a, d, c)))
    return torch.tensor(faces, dtype=torch.long, device=device)


def surface_samples(size, resolution, device):
    """Equal-area cell centres on the target, avoiding ambiguous boundary samples."""
    nx, ny = resolution
    x = ((torch.arange(nx, device=device) + 0.5) / nx - 0.5) * size[0]
    y = ((torch.arange(ny, device=device) + 0.5) / ny - 0.5) * size[1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def surface_coverage(nodes, triangles, samples, surface_z, height_tolerance=0.006):
    """Fraction of target samples covered by the union of near-surface cloth triangles.

    Nodes are in the platform frame. Barycentric interpolation tests the actual cloth
    surface between nodes, so mesh spacing leaves no artificial holes. Overlapping folds
    count once; cloth on the table, below the platform, or hovering above it does not count.
    Chunking samples bounds intermediate memory for large environment batches.
    """
    a, b, c = nodes[:, triangles[:, 0]], nodes[:, triangles[:, 1]], nodes[:, triangles[:, 2]]
    ab, ac = b - a, c - a
    det = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
    valid = det.abs() > 1e-10
    denominator = torch.where(valid, det, torch.ones_like(det)).unsqueeze(-1)
    count = torch.zeros(nodes.shape[0], device=nodes.device)
    for points in samples.split(24):
        offset = points[None, None, :, :] - a[:, :, None, :2]
        u = (offset[..., 0] * ac[..., 1, None] - offset[..., 1] * ac[..., 0, None]) / denominator
        v = (ab[..., 0, None] * offset[..., 1] - ab[..., 1, None] * offset[..., 0]) / denominator
        z = a[..., 2, None] + u * ab[..., 2, None] + v * ac[..., 2, None]
        inside = valid[..., None] & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.000001)
        supported = (z - surface_z).abs() <= height_tolerance
        count += (inside & supported).any(dim=1).float().sum(dim=-1)
    return count / len(samples)
