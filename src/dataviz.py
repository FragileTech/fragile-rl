from __future__ import annotations

import gc
import math
import os
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import torch


if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from fragile.core.layers import FactorizedJumpOperator, TopoEncoder


def visualize_latent(
    model: TopoEncoder,
    X: torch.Tensor,
    colors: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    epoch: int | None = None,
    indices_stack: torch.Tensor | None = None,
    jump_op: FactorizedJumpOperator | None = None,
) -> None:
    """Visualize latent space with 6-panel layout.

    Layout (2x3):
    Row 1: Input 3D | Latent Space | Reconstruction 3D
    Row 2: Chart Assignments (with jumps) | Code Usage per Chart | Hyperbolic Tree

    The Hyperbolic Tree visualizes the macro-state hierarchy:
    - Z=0: Root node (entire observation space)
    - Z=1: Chart nodes (one per chart)
    - Z=2: Code nodes (codes within each chart)
    - Z=3: Data points at their 2D latent coordinates

    Args:
        model: TopoEncoder model
        X: Input data [N, D] (typically 3D nightmare dataset)
        colors: Continuous colors for rainbow [N]
        labels: Ground truth manifold labels [N]
        save_path: Path to save visualization
        epoch: Current epoch (for title), None for final
        indices_stack: [N, num_charts] code indices per chart (for code usage plot)
        jump_op: FactorizedJumpOperator for visualizing chart transitions
    """
    # Clean up any lingering figures to prevent memory leaks
    plt.close("all")
    gc.collect()

    model.eval()
    device = X.device
    input_dim = X.shape[1]

    with torch.no_grad():
        # Get encoder outputs
        K_chart, K_code, _, _z_tex, enc_w, z_geo, _, indices_out, z_n_all_charts, _c_bar = (
            model.encoder(X)
        )

        # Use provided indices_stack or the one from encoder
        if indices_stack is None:
            indices_stack = indices_out

        z = z_geo.cpu().numpy()
        X_np = X.cpu().numpy()
        enc_w.cpu().numpy()
        hard_assign = K_chart.cpu().numpy()
        code_assign = K_code.cpu().numpy()  # Code assignment for each point
        indices_np = indices_stack.cpu().numpy()

        # Get reconstruction
        recon, _, _, _, _, _, _, _ = model(X, use_hard_routing=False)
        recon_np = recon.cpu().numpy()

    fig = plt.figure(figsize=(20, 13))
    title_suffix = f" (Epoch {epoch})" if epoch is not None else " (Final)"

    # --- Row 1 ---

    # Panel 1: 3D Input space colored by structure (rainbow)
    if input_dim >= 3:
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.scatter(
            X_np[:, 0],
            X_np[:, 1],
            X_np[:, 2],
            c=colors,
            cmap="rainbow",
            s=2,
            alpha=0.7,
        )
        ax1.set_title(f"Input: The Nightmare{title_suffix}\n(Roll, Sphere, Moons)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
    else:
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(X_np[:, 0], X_np[:, 1], c=colors, cmap="rainbow", s=2, alpha=0.7)
        ax1.set_title(f"Input Space{title_suffix}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

    # Panel 2: Latent by structure (rainbow colormap)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(z[:, 0], z[:, 1], c=colors, cmap="rainbow", s=3, alpha=0.7)
    ax2.set_title("Latent Space\n(Colored by Structure)")
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")

    # Panel 3: Reconstruction (3D)
    mse = np.mean((X_np - recon_np) ** 2)
    if input_dim >= 3:
        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.scatter(
            recon_np[:, 0],
            recon_np[:, 1],
            recon_np[:, 2],
            c=colors,
            cmap="rainbow",
            s=2,
            alpha=0.7,
        )
        ax3.set_title(f"Reconstruction\nMSE: {mse:.5f}")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
    else:
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.scatter(recon_np[:, 0], recon_np[:, 1], c=colors, cmap="rainbow", s=2, alpha=0.7)
        ax3.set_title(f"Reconstruction\nMSE: {mse:.5f}")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

    # --- Row 2 ---

    # Panel 4: Latent by chart assignment with jump transitions
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(z[:, 0], z[:, 1], c=hard_assign, cmap="tab10", s=3, alpha=0.7)

    # Visualize jump operator transitions if available
    num_jumps = 0
    if jump_op is not None:
        num_jumps = _plot_jump_transitions(
            ax4, z_geo, z_n_all_charts, enc_w, hard_assign, jump_op, device
        )

    title_jump = f"\n({num_jumps} jump arrows)" if jump_op is not None else ""
    ax4.set_title(f"Chart Assignment{title_jump}\n(Topological Surgery)")
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    plt.colorbar(scatter4, ax=ax4, ticks=range(model.num_charts), label="Chart")

    # Panel 5: Latent by code/symbol assignment (different palette per chart)
    ax5 = fig.add_subplot(2, 3, 5)
    num_charts = model.num_charts
    codes_per_chart = model.encoder.codes_per_chart
    chart_cmap = plt.get_cmap("tab10")

    # Create colors using different palettes per chart
    # Each chart gets a sequential colormap (Blues, Oranges, Greens, etc.)
    chart_palettes = [
        "Blues",
        "Oranges",
        "Greens",
        "Purples",
        "Reds",
        "YlOrBr",
        "BuGn",
        "PuRd",
    ]
    symbol_colors = _compute_chart_code_colors(
        hard_assign, code_assign, num_charts, codes_per_chart, chart_palettes
    )

    num_unique_codes = len(np.unique(code_assign))
    ax5.scatter(z[:, 0], z[:, 1], c=symbol_colors, s=3, alpha=0.7)
    ax5.set_title(f"Code Assignment\n({num_unique_codes} codes used)")
    ax5.set_xlabel("z₁")
    ax5.set_ylabel("z₂")

    # Panel 6: Hyperbolic Tree (3D with 2D latent as base)
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    _plot_hyperbolic_tree(
        ax6,
        z,
        hard_assign,
        code_assign,
        indices_np,
        num_charts,
        codes_per_chart,
        chart_cmap,
        chart_palettes,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.3, hspace=0.25)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")  # Extra safety for nested GridSpec objects
    gc.collect()


def _plot_jump_transitions(
    ax: plt.Axes,
    z_geo: torch.Tensor,
    z_n_all_charts: torch.Tensor,
    enc_w: torch.Tensor,
    hard_assign: np.ndarray,
    jump_op: FactorizedJumpOperator,
    device: torch.device,
    max_arrows: int = 100,
    overlap_threshold: float = 0.01,
) -> int:
    """Plot jump operator transitions between charts.

    For boundary points (where router gives weight to multiple charts),
    draw arrows showing how the jump operator maps z_n from one chart to another.

    Args:
        ax: Matplotlib axes to plot on
        z_geo: [N, D] geometric latent coordinates
        z_n_all_charts: [N, num_charts, D] z_n per chart
        enc_w: [N, num_charts] router weights
        hard_assign: [N] hard chart assignments
        jump_op: FactorizedJumpOperator for computing transitions
        device: Torch device
        max_arrows: Maximum number of arrows to draw
        overlap_threshold: Minimum weight product to consider as overlap

    Returns:
        Number of arrows drawn
    """
    N = z_geo.shape[0]
    num_charts = enc_w.shape[1]
    z_np = z_geo.cpu().numpy()
    enc_w_np = enc_w.cpu().numpy()

    # Find boundary points: points with significant weight in multiple charts
    # Compute overlap score: sum of w_i * w_j for all i < j
    boundary_scores = np.zeros(N)
    for i in range(num_charts):
        for j in range(i + 1, num_charts):
            boundary_scores += enc_w_np[:, i] * enc_w_np[:, j]

    # Select top boundary points
    boundary_mask = boundary_scores > overlap_threshold
    boundary_indices = np.where(boundary_mask)[0]

    if len(boundary_indices) == 0:
        return 0

    # Sample if too many
    if len(boundary_indices) > max_arrows:
        np.random.seed(42)  # Reproducible sampling
        boundary_indices = np.random.choice(boundary_indices, max_arrows, replace=False)

    # For each boundary point, draw arrows showing jump transitions
    chart_cmap = plt.get_cmap("tab10")
    arrows_drawn = 0

    with torch.no_grad():
        for idx in boundary_indices:
            src_chart = hard_assign[idx]
            src_pos = z_np[idx]

            # Get z_n for source chart
            z_n_src = z_n_all_charts[idx, src_chart].unsqueeze(0)  # [1, D]

            # Find target charts with significant weight
            for tgt_chart in range(num_charts):
                if tgt_chart == src_chart:
                    continue
                if enc_w_np[idx, tgt_chart] < 0.01:  # Skip negligible targets
                    continue

                # Apply jump operator: L_{src->tgt}(z_n_src)
                src_idx = torch.tensor([src_chart], device=device)
                tgt_idx = torch.tensor([tgt_chart], device=device)
                z_n_jumped = jump_op(z_n_src, src_idx, tgt_idx)  # [1, D]

                # The jumped z_n represents where this point would be in target chart
                # We visualize by showing arrow from src_pos toward the jumped residual direction
                jump_delta = z_n_jumped.cpu().numpy()[0] - z_n_src.cpu().numpy()[0]

                # Scale arrow for visibility - use fixed min length + scaled component
                delta_norm = np.linalg.norm(jump_delta)
                if delta_norm > 1e-6:
                    # Normalize and scale to reasonable visual length
                    arrow_length = 0.15 + 0.2 * min(delta_norm, 1.0)
                    direction = jump_delta / delta_norm
                    dx, dy = direction[0] * arrow_length, direction[1] * arrow_length
                else:
                    dx, dy = 0.1, 0.0  # Default arrow if delta is tiny

                # Draw arrow with color based on target chart
                color = chart_cmap(tgt_chart / max(num_charts - 1, 1))
                ax.annotate(
                    "",
                    xy=(src_pos[0] + dx, src_pos[1] + dy),
                    xytext=(src_pos[0], src_pos[1]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "alpha": 0.7,
                        "linewidth": 1.5,
                        "mutation_scale": 8,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
                arrows_drawn += 1

    return arrows_drawn


def _compute_chart_code_colors(
    K_chart: np.ndarray,
    K_code: np.ndarray,
    num_charts: int,
    codes_per_chart: int,
    chart_palettes: list[str],
) -> np.ndarray:
    """Compute RGB colors for each point based on chart and code assignment.

    Each chart uses a different sequential colormap, so adjacent charts
    with the same code index will have visually distinct colors.

    Args:
        K_chart: [N] chart assignment per point
        K_code: [N] code assignment per point
        num_charts: Number of charts
        codes_per_chart: Codes per chart
        chart_palettes: List of colormap names per chart

    Returns:
        colors: [N, 3] RGB colors in [0, 1]
    """
    N = len(K_chart)
    colors = np.zeros((N, 3))

    for c in range(num_charts):
        mask = K_chart == c
        if mask.sum() == 0:
            continue

        # Get colormap for this chart
        cmap_name = chart_palettes[c % len(chart_palettes)]
        cmap = plt.get_cmap(cmap_name)

        # Map code indices to [0.3, 0.9] range (avoid too light/dark)
        codes_in_chart = K_code[mask]
        unique_codes = np.unique(codes_in_chart)
        num_used = len(unique_codes)

        # Create mapping from code index to color intensity
        if num_used > 1:
            code_to_intensity = {
                code: 0.3 + 0.6 * i / (num_used - 1) for i, code in enumerate(sorted(unique_codes))
            }
        else:
            code_to_intensity = {unique_codes[0]: 0.6}

        # Assign colors
        for i, idx in enumerate(np.where(mask)[0]):
            intensity = code_to_intensity[K_code[idx]]
            colors[idx] = cmap(intensity)[:3]

    return colors


def _plot_hyperbolic_tree(
    ax,
    z_geo: np.ndarray,
    K_chart: np.ndarray,
    K_code: np.ndarray,
    indices_stack: np.ndarray,
    num_charts: int,
    codes_per_chart: int,
    chart_cmap,
    chart_palettes: list[str],
) -> None:
    """Plot the hierarchical tree with 2D latent as base.

    Z-levels (inverted so data is at bottom):
    - Z=3: Root node (center, top)
    - Z=2: Chart nodes (spread around center)
    - Z=1: Code nodes (spread under each chart)
    - Z=0: Data points (at their z_geo coordinates, bottom)

    Data points are colored by chart-specific palettes to show
    the hierarchical clustering structure. Each chart uses a different
    color palette so adjacent charts can be distinguished.

    This visualizes Section 7.11's concept: the discrete macro-register
    forms the "skeleton" of a hyperbolic space, with data points at the boundary.
    """
    N = len(z_geo)

    # Compute colors using chart-specific palettes
    symbol_colors = _compute_chart_code_colors(
        K_chart, K_code, num_charts, codes_per_chart, chart_palettes
    )

    # Level 0: Data points on the X-Y plane at Z=0, colored by chart+code
    ax.scatter(z_geo[:, 0], z_geo[:, 1], np.zeros(N), c=symbol_colors, s=2, alpha=0.5)

    # Compute chart centers (mean of points assigned to each chart)
    chart_centers = []
    for c in range(num_charts):
        mask = K_chart == c
        if mask.sum() > 0:
            center = z_geo[mask].mean(axis=0)
        else:
            center = np.array([0.0, 0.0])
        chart_centers.append(center)
    chart_centers = np.array(chart_centers)

    # Level 2: Chart nodes at Z=2
    ax.scatter(
        chart_centers[:, 0],
        chart_centers[:, 1],
        np.full(num_charts, 2.0),
        c=[chart_cmap(i / max(num_charts - 1, 1)) for i in range(num_charts)],
        s=100,
        marker="s",
        edgecolors="black",
        linewidths=0.5,
    )

    # Level 3: Root node at Z=3 (center of all charts)
    root = chart_centers.mean(axis=0)
    ax.scatter([root[0]], [root[1]], [3.0], c="black", s=200, marker="^")

    # Draw edges: Root → Charts
    for c in range(num_charts):
        ax.plot(
            [root[0], chart_centers[c, 0]],
            [root[1], chart_centers[c, 1]],
            [3.0, 2.0],
            color=chart_cmap(c / max(num_charts - 1, 1)),
            alpha=0.7,
            linewidth=1.5,
        )

    # Level 1: Code nodes at Z=1 (cluster centers per chart)
    for c in range(num_charts):
        mask = K_chart == c
        if mask.sum() == 0:
            continue

        # Get chart-specific colormap
        cmap_name = chart_palettes[c % len(chart_palettes)]
        code_cmap = plt.get_cmap(cmap_name)

        chart_points = z_geo[mask]
        chart_codes = indices_stack[mask, c]

        # Get unique codes used in this chart
        unique_codes = np.unique(chart_codes)
        num_used = len(unique_codes)

        for i, code in enumerate(sorted(unique_codes)):
            code_mask = chart_codes == code
            if code_mask.sum() > 0:
                code_center = chart_points[code_mask].mean(axis=0)

                # Map code to intensity within chart's palette
                intensity = 0.3 + 0.6 * i / max(num_used - 1, 1) if num_used > 1 else 0.6
                code_color = code_cmap(intensity)

                # Draw small marker at Z=1 with chart-specific color
                ax.scatter(
                    [code_center[0]],
                    [code_center[1]],
                    [1.0],
                    c=[code_color],
                    s=25,
                    marker="o",
                    alpha=0.8,
                )

                # Edge from chart (Z=2) to code (Z=1)
                ax.plot(
                    [chart_centers[c, 0], code_center[0]],
                    [chart_centers[c, 1], code_center[1]],
                    [2.0, 1.0],
                    color=code_color,
                    alpha=0.4,
                    linewidth=0.5,
                )

    # Labels and title
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_zlabel("Hierarchy Level")
    ax.set_title("Hyperbolic Tree\n(Root → Charts → Codes → Data)")

    # Set Z-axis ticks
    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels(["Data", "Codes", "Charts", "Root"])

    # Rotate view: elevation 25°, azimuth -60° (counterclockwise rotation)
    # This makes the 2D latent plane appear more horizontal
    ax.view_init(elev=25, azim=-60)


def visualize_results(results: dict, save_path: str = "benchmark_result.png") -> None:
    """Create final visualization comparing ground truth, charts, and reconstructions.

    Layout (2 rows x 4 cols):
    Row 1: Input (3D rainbow) | Chart Assignments | Loss Curves | AMI Comparison
    Row 2: VanillaAE Recon | Standard VQ Recon | TopoEncoder Recon | Error Histogram
    """
    # Clean up any lingering figures to prevent memory leaks
    plt.close("all")
    gc.collect()

    X = results["X"].cpu().numpy()
    colors = results["colors"]
    chart_assignments = results["chart_assignments"]
    recon_ae = results["recon_ae"].cpu().numpy() if results["recon_ae"] is not None else None
    recon_std = results["recon_std"].cpu().numpy() if results["recon_std"] is not None else None
    recon_atlas = results["recon_atlas"].cpu().numpy()

    fig = plt.figure(figsize=(24, 10))

    # ========== Row 1 ==========

    # Panel 1: 3D Input with rainbow coloring
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap="rainbow", s=2, alpha=0.7)
    ax1.set_title("Input: The Nightmare\n(Roll, Sphere, Moons)", fontsize=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Panel 2: Atlas Chart Assignments (3D)
    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    scatter2 = ax2.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=chart_assignments, cmap="tab10", s=2, alpha=0.7
    )
    ax2.set_title("Atlas Chart Assignments\n(Learned Topology)", fontsize=12)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label="Chart")

    # Panel 3: Loss Curves (3-way)
    ax3 = fig.add_subplot(2, 4, 3)
    epochs = range(len(results["atlas_losses"]))
    if recon_ae is not None:
        ax3.plot(
            epochs,
            results["ae_losses"],
            label="VanillaAE",
            alpha=0.8,
            linewidth=1.5,
            color="C2",
        )
    if recon_std is not None:
        ax3.plot(
            epochs,
            results["std_losses"],
            label="Standard VQ",
            alpha=0.8,
            linewidth=1.5,
            color="C0",
        )
    ax3.plot(
        epochs,
        results["atlas_losses"],
        label="TopoEncoder",
        alpha=0.8,
        linewidth=1.5,
        color="C1",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Convergence", fontsize=12)
    ax3.legend()
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Panel 4: AMI Comparison (Bar Chart)
    ax4 = fig.add_subplot(2, 4, 4)
    models = []
    ami_scores = []
    bar_colors = []
    if recon_ae is not None:
        models.append("VanillaAE")
        ami_scores.append(results["ami_ae"])
        bar_colors.append("C2")
    if recon_std is not None:
        models.append("Standard VQ")
        ami_scores.append(results["ami_std"])
        bar_colors.append("C0")
    models.append("TopoEncoder")
    ami_scores.append(results["ami_atlas"])
    bar_colors.append("C1")
    bars = ax4.bar(models, ami_scores, color=bar_colors, alpha=0.8)
    ax4.set_ylabel("AMI Score")
    ax4.set_title("Topology Discovery\n(Adjusted Mutual Information)", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Excellent threshold")
    ax4.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Good threshold")
    # Add value labels on bars
    for bar, score in zip(bars, ami_scores):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # ========== Row 2: Reconstructions ==========

    # Panel 5: VanillaAE Reconstruction (3D)
    ax5 = fig.add_subplot(2, 4, 5, projection="3d")
    if recon_ae is not None:
        ax5.scatter(
            recon_ae[:, 0],
            recon_ae[:, 1],
            recon_ae[:, 2],
            c=colors,
            cmap="rainbow",
            s=2,
            alpha=0.7,
        )
        mse_ae = results["mse_ae"]
        ax5.set_title(f"VanillaAE Reconstruction\nMSE: {mse_ae:.5f}", fontsize=12)
    else:
        ax5.set_title("VanillaAE\n(DISABLED)", fontsize=12)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")

    # Panel 6: Standard VQ Reconstruction (3D)
    ax6 = fig.add_subplot(2, 4, 6, projection="3d")
    if recon_std is not None:
        ax6.scatter(
            recon_std[:, 0],
            recon_std[:, 1],
            recon_std[:, 2],
            c=colors,
            cmap="rainbow",
            s=2,
            alpha=0.7,
        )
        mse_std = results["mse_std"]
        ax6.set_title(f"Standard VQ Reconstruction\nMSE: {mse_std:.5f}", fontsize=12)
    else:
        ax6.set_title("Standard VQ\n(DISABLED)", fontsize=12)
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_zlabel("Z")

    # Panel 7: TopoEncoder Reconstruction (3D)
    ax7 = fig.add_subplot(2, 4, 7, projection="3d")
    ax7.scatter(
        recon_atlas[:, 0],
        recon_atlas[:, 1],
        recon_atlas[:, 2],
        c=colors,
        cmap="rainbow",
        s=2,
        alpha=0.7,
    )
    mse_atlas = results["mse_atlas"]
    ax7.set_title(f"TopoEncoder Reconstruction\nMSE: {mse_atlas:.5f}", fontsize=12)
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_zlabel("Z")

    # Panel 8: Reconstruction Error Histogram
    ax8 = fig.add_subplot(2, 4, 8)
    error_atlas = np.linalg.norm(X - recon_atlas, axis=1)
    if recon_ae is not None:
        error_ae = np.linalg.norm(X - recon_ae, axis=1)
        ax8.hist(
            error_ae,
            bins=50,
            alpha=0.5,
            label=f"AE (μ={error_ae.mean():.3f})",
            color="C2",
        )
    if recon_std is not None:
        error_std = np.linalg.norm(X - recon_std, axis=1)
        ax8.hist(
            error_std,
            bins=50,
            alpha=0.5,
            label=f"VQ (μ={error_std.mean():.3f})",
            color="C0",
        )
    ax8.hist(
        error_atlas,
        bins=50,
        alpha=0.5,
        label=f"Topo (μ={error_atlas.mean():.3f})",
        color="C1",
    )
    ax8.set_xlabel("Reconstruction Error (L2)")
    ax8.set_ylabel("Count")
    ax8.set_title("Error Distribution", fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Add summary text with 3-way comparison
    ami_ae = results["ami_ae"]
    ami_std = results["ami_std"]
    ami_atlas = results["ami_atlas"]
    perp_std = results["std_perplexity"]
    perp_atlas = results["atlas_perplexity"]
    fig.suptitle(
        f"3-Way Benchmark | AMI: AE={ami_ae:.3f}, VQ={ami_std:.3f}, Topo={ami_atlas:.3f} | "
        f"Perplexity: VQ={perp_std:.1f}, Topo={perp_atlas:.1f}",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")  # Extra safety for nested GridSpec objects
    gc.collect()
    print(f"\nFinal visualization saved to: {save_path}")


# ==========================================
# IMAGE VISUALIZATION (CIFAR-10, MNIST, etc.)
# ==========================================


def _select_class_representatives(
    labels: np.ndarray,
    num_classes: int = 10,
) -> list[int]:
    """Select one representative index per class.

    Args:
        labels: [N] class labels
        num_classes: Number of classes

    Returns:
        List of indices (one per class, in class order)
    """
    indices = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            # Take first sample of this class
            idx = np.where(mask)[0][0]
            indices.append(idx)
        else:
            # If class not present, use first sample
            indices.append(0)
    return indices


def _tensor_to_image(x: torch.Tensor, image_shape: tuple = (32, 32, 3)) -> np.ndarray:
    """Convert flattened tensor to displayable image.

    Args:
        x: Flattened tensor [D] where D = H*W*C (e.g., 3072 for CIFAR-10)
        image_shape: Target shape (H, W, C)

    Returns:
        Image array [H, W, C] in [0, 1] range
    """
    H, W, C = image_shape
    img = x.cpu().numpy().reshape(H, W, C)
    # Clip to valid range
    return np.clip(img, 0, 1)


def _create_image_grid(
    ax: plt.Axes,
    images: list[np.ndarray],
    class_names: list[str],
    title: str,
    nrows: int = 2,
    ncols: int = 5,
) -> None:
    """Create a grid of images within a single axes.

    Args:
        ax: Matplotlib axes
        images: List of images [H, W, C] in [0, 1] range
        class_names: Class names for labels
        title: Title for the grid
        nrows: Number of rows
        ncols: Number of columns
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Create nested grid
    gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax.get_subplotspec())
    fig = ax.figure

    for i, (img, name) in enumerate(zip(images, class_names)):
        row = i // ncols
        col = i % ncols
        sub_ax = fig.add_subplot(gs[row, col])
        sub_ax.imshow(img)
        sub_ax.set_title(name, fontsize=7)
        sub_ax.axis("off")


def visualize_latent_images(
    model: TopoEncoder,
    X: torch.Tensor,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str,
    epoch: int | None = None,
    jump_op: FactorizedJumpOperator | None = None,
    image_shape: tuple = (32, 32, 3),
) -> None:
    """Visualize latent space for image data with 6-panel layout.

    Layout (2x3):
    Row 1: Sample Images (2x5 grid) | Latent Space (by class) | Reconstruction Grid
    Row 2: Chart Assignments | Code Usage per Chart | Hyperbolic Tree

    Args:
        model: TopoEncoder model
        X: Flattened image data [N, D] (e.g., [N, 3072] for CIFAR-10)
        labels: Class labels [N] (0-9 for CIFAR-10)
        class_names: List of class names
        save_path: Path to save visualization
        epoch: Current epoch (for title), None for final
        jump_op: FactorizedJumpOperator for chart transitions
        image_shape: Shape to reshape images (H, W, C)
    """
    # Clean up any lingering figures to prevent memory leaks
    plt.close("all")
    gc.collect()

    model.eval()
    num_classes = len(class_names)

    with torch.no_grad():
        # Get encoder outputs
        K_chart, K_code, _, _z_tex, _enc_w, z_geo, _, indices_out, _z_n_all_charts, _c_bar = (
            model.encoder(X)
        )

        # Get reconstruction
        recon, _, _, _, _, _, _, _ = model(X, use_hard_routing=False)

        # Convert to numpy
        z = z_geo.cpu().numpy()
        hard_assign = K_chart.cpu().numpy()
        code_assign = K_code.cpu().numpy()
        indices_np = indices_out.cpu().numpy()

    # Get model config
    num_charts = model.encoder.num_charts
    codes_per_chart = model.encoder.codes_per_chart

    # Chart visualization setup
    chart_cmap = plt.get_cmap("tab10")
    chart_palettes = [
        "Blues",
        "Oranges",
        "Greens",
        "Purples",
        "Reds",
        "Grays",
        "YlOrBr",
        "PuRd",
        "BuGn",
        "RdPu",
    ]

    # Select one sample per class
    sample_indices = _select_class_representatives(labels, num_classes)

    # Convert selected samples to images
    sample_images = [_tensor_to_image(X[idx], image_shape) for idx in sample_indices]
    recon_images = [_tensor_to_image(recon[idx], image_shape) for idx in sample_indices]

    # Create figure with 6 panels
    fig = plt.figure(figsize=(20, 13))

    title_suffix = f" (Epoch {epoch})" if epoch is not None else " (Final)"

    # --- Panel 1: Sample Images (2x5 grid) ---
    ax1 = fig.add_subplot(2, 3, 1)
    _create_image_grid(ax1, sample_images, class_names, f"Input Samples{title_suffix}")

    # --- Panel 2: Latent Space (colored by CLASS) ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=max(9, num_classes - 1),
        s=3,
        alpha=0.7,
    )

    # Add class legend
    cmap = plt.get_cmap("tab10")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i / max(9, num_classes - 1)),
            markersize=8,
            label=class_names[i],
        )
        for i in range(num_classes)
    ]
    ax2.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)

    ax2.set_title(f"Latent Space (by Class){title_suffix}", fontsize=12)
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Reconstruction Grid ---
    ax3 = fig.add_subplot(2, 3, 3)
    mse = ((X - recon) ** 2).mean().item()
    _create_image_grid(ax3, recon_images, class_names, f"Reconstructions (MSE: {mse:.5f})")

    # --- Panel 4: Chart Assignments ---
    ax4 = fig.add_subplot(2, 3, 4)
    chart_colors = [chart_cmap(k / max(1, num_charts - 1)) for k in hard_assign]
    ax4.scatter(z[:, 0], z[:, 1], c=chart_colors, s=3, alpha=0.7)

    # Add chart legend
    handles4 = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=chart_cmap(k / max(1, num_charts - 1)),
            markersize=8,
            label=f"Chart {k}",
        )
        for k in range(num_charts)
    ]
    ax4.legend(handles=handles4, loc="upper right", fontsize=6, ncol=2)

    ax4.set_title("Chart Assignments", fontsize=12)
    ax4.set_xlabel("z₁")
    ax4.set_ylabel("z₂")
    ax4.grid(True, alpha=0.3)

    # Optionally add jump arrows
    if jump_op is not None:
        from fragile.datasets import find_boundary_pairs

        boundary_pairs = find_boundary_pairs(
            z, hard_assign, X.cpu().numpy(), k=5, max_latent_dist=2.0
        )
        for i, j in boundary_pairs[:50]:
            ax4.annotate(
                "",
                xy=(z[j, 0], z[j, 1]),
                xytext=(z[i, 0], z[i, 1]),
                arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.3, "lw": 0.5},
            )

    # --- Panel 5: Code Usage per Chart ---
    ax5 = fig.add_subplot(2, 3, 5)
    symbol_colors = _compute_chart_code_colors(
        hard_assign, code_assign, num_charts, codes_per_chart, chart_palettes
    )
    ax5.scatter(z[:, 0], z[:, 1], c=symbol_colors, s=3, alpha=0.7)
    ax5.set_title("Code Usage per Chart", fontsize=12)
    ax5.set_xlabel("z₁")
    ax5.set_ylabel("z₂")
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Hyperbolic Tree ---
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    _plot_hyperbolic_tree(
        ax6,
        z,
        hard_assign,
        code_assign,
        indices_np,
        num_charts,
        codes_per_chart,
        chart_cmap,
        chart_palettes,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")  # Extra safety for nested GridSpec objects
    gc.collect()
    print(f"Saved: {save_path}")


def visualize_results_images(
    results: dict,
    class_names: list[str],
    save_path: str = "cifar10_result.png",
    image_shape: tuple = (32, 32, 3),
) -> None:
    """Visualize benchmark results for image data with 8-panel layout.

    Layout (2x4):
    Row 1: Sample Images | Chart Assignments | Loss Curves | AMI Comparison
    Row 2: AE Recon | VQ Recon | TopoEncoder Recon | Error Histogram

    Args:
        results: Dictionary with benchmark results
        class_names: List of class names
        save_path: Path to save visualization
        image_shape: Shape to reshape images (H, W, C)
    """
    # Clean up any lingering figures to prevent memory leaks
    plt.close("all")
    gc.collect()

    X = results["X"]
    labels = results["labels"]
    results["chart_assignments"]
    recon_ae = results.get("recon_ae")
    recon_std = results.get("recon_std")
    recon_atlas = results["recon_atlas"]
    num_classes = len(class_names)

    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X

    z_geo = results.get("z_geo")
    if z_geo is not None:
        if isinstance(z_geo, torch.Tensor):
            z = z_geo.cpu().numpy()
        else:
            z = z_geo
    else:
        # Get model for latent computation
        model_atlas = results["model_atlas"]

        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X_tensor = X
            else:
                X_tensor = torch.from_numpy(X).float()

            _, _, _, _, _, z_geo, _, _, _, _c_bar = model_atlas.encoder(
                X_tensor.to(next(model_atlas.parameters()).device)
            )
            z = z_geo.cpu().numpy()

    # Select representatives
    sample_indices = _select_class_representatives(labels, num_classes)

    fig = plt.figure(figsize=(20, 10))

    # --- Panel 1: Sample Images ---
    ax1 = fig.add_subplot(2, 4, 1)
    sample_images = [
        _tensor_to_image(torch.from_numpy(X_np[idx]), image_shape) for idx in sample_indices
    ]
    _create_image_grid(ax1, sample_images, class_names, "Input Samples")

    # --- Panel 2: Latent Space by Class ---
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=max(9, num_classes - 1),
        s=3,
        alpha=0.7,
    )
    ax2.set_title("Latent Space (by Class)", fontsize=12)
    ax2.set_xlabel("z₁")
    ax2.set_ylabel("z₂")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Loss Curves ---
    ax3 = fig.add_subplot(2, 4, 3)
    ae_losses = results.get("ae_losses", [])
    std_losses = results.get("std_losses", [])
    atlas_losses = results["atlas_losses"]

    if ae_losses:
        ax3.plot(ae_losses, label="AE", alpha=0.7, color="C2")
    if std_losses:
        ax3.plot(std_losses, label="VQ", alpha=0.7, color="C0")
    ax3.plot(atlas_losses, label="Topo", alpha=0.7, color="C1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Loss", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    # --- Panel 4: AMI vs Readout Accuracy ---
    ax4 = fig.add_subplot(2, 4, 4)
    ami_ae = results.get("ami_ae", 0)
    ami_std = results.get("ami_std", 0)
    ami_atlas = results["ami_atlas"]
    sup_acc = results.get("sup_acc", 0)
    ae_cls_acc = results.get("ae_cls_acc")
    std_cls_acc = results.get("std_cls_acc")
    cls_acc = results.get("cls_acc")

    ami_vals = [ami_ae, ami_std, ami_atlas]
    acc_vals = [ae_cls_acc, std_cls_acc, cls_acc]
    acc_vals = [float(val) if val is not None else float("nan") for val in acc_vals]
    has_acc = any(math.isfinite(val) for val in acc_vals)
    x = np.arange(3)
    width = 0.36
    bars_ami = ax4.bar(x - width / 2, ami_vals, width, label="AMI", color="C1", alpha=0.8)
    if has_acc:
        bars_acc = ax4.bar(x + width / 2, acc_vals, width, label="Test Acc", color="C0", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["AE", "VQ", "Topo"])
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Score")
    ax4.set_title("AMI vs Readout Accuracy", fontsize=12)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend()

    for bar, val in zip(bars_ami, ami_vals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
        )
    if has_acc:
        for bar, val in zip(bars_acc, acc_vals):
            if not math.isfinite(val):
                label = "N/A"
                height = 0.0
            else:
                label = f"{val:.3f}"
                height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                label,
                ha="center",
                fontsize=9,
            )

    # --- Panel 5: AE Reconstruction ---
    ax5 = fig.add_subplot(2, 4, 5)
    if recon_ae is not None:
        if isinstance(recon_ae, torch.Tensor):
            recon_ae_np = recon_ae.cpu().numpy()
        else:
            recon_ae_np = recon_ae
        ae_images = [
            _tensor_to_image(torch.from_numpy(recon_ae_np[idx]), image_shape)
            for idx in sample_indices
        ]
        mse_ae = results.get("mse_ae", 0)
        _create_image_grid(ax5, ae_images, class_names, f"AE Recon (MSE: {mse_ae:.5f})")
    else:
        ax5.text(0.5, 0.5, "AE Disabled", ha="center", va="center", fontsize=12)
        ax5.axis("off")

    # --- Panel 6: VQ Reconstruction ---
    ax6 = fig.add_subplot(2, 4, 6)
    if recon_std is not None:
        if isinstance(recon_std, torch.Tensor):
            recon_std_np = recon_std.cpu().numpy()
        else:
            recon_std_np = recon_std
        vq_images = [
            _tensor_to_image(torch.from_numpy(recon_std_np[idx]), image_shape)
            for idx in sample_indices
        ]
        mse_std = results.get("mse_std", 0)
        _create_image_grid(ax6, vq_images, class_names, f"VQ Recon (MSE: {mse_std:.5f})")
    else:
        ax6.text(0.5, 0.5, "VQ Disabled", ha="center", va="center", fontsize=12)
        ax6.axis("off")

    # --- Panel 7: TopoEncoder Reconstruction ---
    ax7 = fig.add_subplot(2, 4, 7)
    if isinstance(recon_atlas, torch.Tensor):
        recon_atlas_np = recon_atlas.cpu().numpy()
    else:
        recon_atlas_np = recon_atlas
    topo_images = [
        _tensor_to_image(torch.from_numpy(recon_atlas_np[idx]), image_shape)
        for idx in sample_indices
    ]
    mse_atlas = results.get("mse_atlas", 0)
    _create_image_grid(ax7, topo_images, class_names, f"Topo Recon (MSE: {mse_atlas:.5f})")

    # --- Panel 8: Error Histogram ---
    ax8 = fig.add_subplot(2, 4, 8)
    error_atlas = np.linalg.norm(X_np - recon_atlas_np, axis=1)
    if recon_ae is not None:
        error_ae = np.linalg.norm(X_np - recon_ae_np, axis=1)
        ax8.hist(error_ae, bins=50, alpha=0.5, label=f"AE (μ={error_ae.mean():.3f})", color="C2")
    if recon_std is not None:
        error_std = np.linalg.norm(X_np - recon_std_np, axis=1)
        ax8.hist(error_std, bins=50, alpha=0.5, label=f"VQ (μ={error_std.mean():.3f})", color="C0")
    ax8.hist(
        error_atlas, bins=50, alpha=0.5, label=f"Topo (μ={error_atlas.mean():.3f})", color="C1"
    )
    ax8.set_xlabel("Reconstruction Error (L2)")
    ax8.set_ylabel("Count")
    ax8.set_title("Error Distribution", fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Suptitle
    ami_atlas = results["ami_atlas"]
    perp_atlas = results["atlas_perplexity"]
    fig.suptitle(
        f"CIFAR-10 Benchmark | AMI: {ami_atlas:.3f} | "
        f"Sup Acc: {sup_acc:.3f} | Perplexity: {perp_atlas:.1f}",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")  # Extra safety for nested GridSpec objects
    gc.collect()
    print(f"\nFinal visualization saved to: {save_path}")
