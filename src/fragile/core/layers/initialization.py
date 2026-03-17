import math

import torch


# ---------------------------------------------------------------------------
# Quasi-uniform initialization helpers
# ---------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> torch.Tensor:
    """Generate *n* quasi-uniformly spaced points on S² (Fibonacci lattice).

    Returns a [n, 3] tensor on the unit sphere. For D != 3, falls back to
    ``_spread_directions``.
    """
    golden = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n, dtype=torch.float32)
    theta = 2 * math.pi * indices / golden  # azimuth
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)  # polar
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


def spread_directions(n: int, dim: int) -> torch.Tensor:
    """Generate *n* spread-out unit directions in R^dim.

    Uses the Fibonacci sphere for dim == 3. Otherwise generates random
    directions and iteratively repels them (simple Lloyd-like relaxation).
    """
    if dim == 3:
        return _fibonacci_sphere(n)

    # Random init + greedy repulsion (5 iterations suffice for init quality)
    pts = torch.randn(n, dim)
    pts = torch.nn.functional.normalize(pts, dim=-1)
    for _ in range(20):
        # Compute pairwise cosine similarity
        sim = pts @ pts.t()  # [n, n]
        sim.fill_diagonal_(-1e9)  # ignore self
        # Push each point away from its nearest neighbor
        nearest = sim.argmax(dim=1)  # [n]
        neighbors = pts[nearest]  # [n, dim]
        pts = pts - 0.3 * neighbors  # repel
        pts = torch.nn.functional.normalize(pts, dim=-1)
    return pts


def spread_codebook(
    num_charts: int, codes_per_chart: int, dim: int, radius: float = 0.3
) -> torch.Tensor:
    """Initialize codebook entries spread around the local origin.

    Each chart gets ``codes_per_chart`` codes arranged as quasi-uniform
    directions scaled to ``radius`` in the Poincaré ball.  This avoids the
    usual failure mode where all codes start near zero and instantly collapse
    to a single nearest-neighbor.

    Returns [num_charts, codes_per_chart, dim].
    """
    cb = torch.zeros(num_charts, codes_per_chart, dim)
    for c in range(num_charts):
        dirs = _spread_directions(codes_per_chart, dim)
        # Uniform radii in [radius/2, radius] so codes aren't on a thin shell
        r = torch.rand(codes_per_chart, 1) * (radius / 2) + (radius / 2)
        cb[c] = dirs * r
    return cb


def resolve_bundle_params(
    hidden_dim: int,
    latent_dim: int,
    bundle_size: int | None,
) -> tuple[int, int]:
    if bundle_size is None:
        if latent_dim > 0 and hidden_dim % latent_dim == 0:
            bundle_size = latent_dim
        else:
            bundle_size = 1
    if bundle_size <= 0:
        msg = "bundle_size must be positive."
        raise ValueError(msg)
    if hidden_dim % bundle_size != 0:
        msg = "hidden_dim must be divisible by bundle_size."
        raise ValueError(msg)
    return bundle_size, hidden_dim // bundle_size


def init_soft_equiv_layers(layers: nn.ModuleList) -> None:
    """Initialize soft-equivariant layers to be purely equivariant (no mixing)."""
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer.mixing_weights, torch.Tensor):
                layer.mixing_weights.zero_()
            else:
                for row in layer.mixing_weights:
                    for weight in row:
                        weight.zero_()
