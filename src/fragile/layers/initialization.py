import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Quasi-uniform initialization helpers
# ---------------------------------------------------------------------------


def fibonacci_sphere(n: int) -> torch.Tensor:
    """Generate *n* quasi-uniformly spaced points on S² via a Fibonacci lattice.

    Args:
        n: Number of points to place on the unit sphere.

    Returns:
        torch.Tensor: A tensor of shape ``[n, 3]`` where each row is a unit
            vector on S².
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

    Uses the Fibonacci sphere for ``dim == 3``. Otherwise generates random
    directions and iteratively repels them (simple Lloyd-like relaxation).

    Args:
        n: Number of unit directions to generate.
        dim: Dimensionality of the ambient space.

    Returns:
        torch.Tensor: A tensor of shape ``[n, dim]`` whose rows are unit
            vectors approximately maximally spread on the unit sphere.
    """
    if dim == 3:
        return fibonacci_sphere(n)

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
    directions scaled to ``radius`` in the Poincare ball.  This avoids the
    usual failure mode where all codes start near zero and instantly collapse
    to a single nearest-neighbor.

    Args:
        num_charts: Number of charts (first dimension of the codebook).
        codes_per_chart: Number of code vectors per chart.
        dim: Dimensionality of each code vector.
        radius: Maximum norm for the initialized code vectors. Actual norms
            are drawn uniformly from ``[radius / 2, radius]`` so that codes
            are not confined to a thin shell.

    Returns:
        torch.Tensor: A tensor of shape ``[num_charts, codes_per_chart, dim]``
            containing the initialized codebook entries.
    """
    cb = torch.zeros(num_charts, codes_per_chart, dim)
    for c in range(num_charts):
        dirs = spread_directions(codes_per_chart, dim)
        # Uniform radii in [radius/2, radius] so codes aren't on a thin shell
        r = torch.rand(codes_per_chart, 1) * (radius / 2) + (radius / 2)
        cb[c] = dirs * r
    return cb


def resolve_bundle_params(
    hidden_dim: int,
    latent_dim: int,
    bundle_size: int | None,
) -> tuple[int, int]:
    """Resolve the bundle size and compute the number of bundles.

    If ``bundle_size`` is not provided, it is inferred from ``hidden_dim`` and
    ``latent_dim``: when ``latent_dim`` evenly divides ``hidden_dim`` the
    bundle size equals ``latent_dim``; otherwise it defaults to 1.

    Args:
        hidden_dim: Total hidden dimensionality. Must be divisible by the
            resolved ``bundle_size``.
        latent_dim: Latent dimensionality used as a candidate for the bundle
            size when ``bundle_size`` is ``None``.
        bundle_size: Explicit bundle size. When ``None``, the value is
            inferred automatically.

    Returns:
        tuple[int, int]: A tuple ``(bundle_size, num_bundles)`` where
            ``num_bundles = hidden_dim // bundle_size``.

    Raises:
        ValueError: If the resolved ``bundle_size`` is not positive or if
            ``hidden_dim`` is not divisible by ``bundle_size``.
    """
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
    """Initialize soft-equivariant layers to be purely equivariant (no mixing).

    Sets all ``mixing_weights`` in the given layers to zero so that each layer
    starts as a strictly equivariant operation.

    Args:
        layers: A ``ModuleList`` of layers, each expected to have a
            ``mixing_weights`` attribute that is either a single
            ``torch.Tensor`` or a nested iterable of tensors.

    Returns:
        None
    """
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer.mixing_weights, torch.Tensor):
                layer.mixing_weights.zero_()
            else:
                for row in layer.mixing_weights:
                    for weight in row:
                        weight.zero_()
