"""Poincaré disk plots and diagnostic visualizations for VLA experiments."""

from __future__ import annotations

import holoviews as hv
import numpy as np
import torch


def conformal_factor_np(z_geo: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Poincare conformal factor lambda(z) = 2 / (1 - |z|^2)."""
    r_sq = np.sum(z_geo**2, axis=1)
    r_sq = np.clip(r_sq, 0.0, 1.0 - eps)
    return 2.0 / (1.0 - r_sq)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _pca_2d(z: np.ndarray) -> np.ndarray:
    """Project to 2D via PCA if latent_dim > 2."""
    if z.shape[1] <= 2:
        return z[:, :2]
    z_centered = z - z.mean(axis=0)
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    return z_centered @ Vt[:2].T


def plot_poincare_disk(
    z_geo: torch.Tensor | np.ndarray,
    K_chart: torch.Tensor | np.ndarray,
    task_labels: torch.Tensor | np.ndarray | None = None,
    ax=None,
    title: str = "Poincaré Disk Embedding",
):
    """Scatter plot of latent embeddings on the unit disk.

    Colors by chart assignment; optionally marks tasks with markers.
    """
    import matplotlib.pyplot as plt

    z = _to_numpy(z_geo)
    K = _to_numpy(K_chart)
    z_2d = _pca_2d(z)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.3)

    scatter = ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=K,
        cmap="tab10",
        s=8,
        alpha=0.6,
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Chart")

    if task_labels is not None:
        tl = _to_numpy(task_labels)
        for label in np.unique(tl):
            mask = tl == label
            ax.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                marker=f"${int(label)}$",
                s=30,
                alpha=0.5,
                color="black",
            )

    return fig


def plot_radius_histogram(
    z_geo: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Radial Distribution",
):
    """Histogram of ||z|| norms."""
    import matplotlib.pyplot as plt

    z = _to_numpy(z_geo)
    r = np.linalg.norm(z, axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    else:
        fig = ax.figure

    ax.hist(r, bins=50, density=True, alpha=0.7, color="steelblue")
    ax.axvline(r.mean(), color="red", linestyle="--", label=f"mean={r.mean():.3f}")
    ax.set_xlabel("||z||")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    return fig


def plot_chart_task_alignment(
    K_chart: torch.Tensor | np.ndarray,
    task_labels: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Chart-Task Alignment",
):
    """Confusion heatmap and AMI score for chart vs. task alignment."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_mutual_info_score

    K = _to_numpy(K_chart).astype(int)
    T = _to_numpy(task_labels).astype(int)

    n_charts = K.max() + 1
    n_tasks = T.max() + 1
    confusion = np.zeros((n_charts, n_tasks), dtype=int)
    for k, t in zip(K, T):
        confusion[k, t] += 1

    ami = adjusted_mutual_info_score(K, T)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(confusion, aspect="auto", cmap="Blues")
    ax.set_xlabel("Task")
    ax.set_ylabel("Chart")
    ax.set_title(f"{title}  (AMI={ami:.3f})")
    fig.colorbar(im, ax=ax)

    return fig


def plot_chart_transitions(
    K_chart: torch.Tensor | np.ndarray,
    episode_ids: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Chart Transition Matrix",
):
    """Transition matrix of chart assignments across consecutive timesteps."""
    import matplotlib.pyplot as plt

    K = _to_numpy(K_chart).astype(int)
    ep = _to_numpy(episode_ids).astype(int)

    n_charts = K.max() + 1
    trans = np.zeros((n_charts, n_charts), dtype=int)

    for i in range(len(K) - 1):
        if ep[i] == ep[i + 1]:
            trans[K[i], K[i + 1]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True).clip(min=1)
    trans_norm = trans / row_sums

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.figure

    im = ax.imshow(trans_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("To Chart")
    ax.set_ylabel("From Chart")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    return fig


def plot_dynamics_trajectory(
    z_pred: torch.Tensor | np.ndarray,
    z_target: torch.Tensor | np.ndarray,
    ax=None,
    title: str = "Dynamics: Predicted vs Target",
):
    """Overlay predicted and target trajectories on the Poincaré disk."""
    import matplotlib.pyplot as plt

    zp = _pca_2d(_to_numpy(z_pred))
    zt = _pca_2d(_to_numpy(z_target))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.3)

    ax.plot(zt[:, 0], zt[:, 1], "b-o", markersize=4, alpha=0.7, label="Target")
    ax.plot(zp[:, 0], zp[:, 1], "r--s", markersize=4, alpha=0.7, label="Predicted")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()

    return fig


def hv_chart_transitions(
    labels: torch.Tensor | np.ndarray,
    episode_ids: torch.Tensor | np.ndarray,
    title: str = "Chart Transition Matrix",
    label_name: str = "Chart",
) -> hv.HeatMap:
    """HoloViews heatmap of label-to-label transition probabilities."""
    L = np.asarray(_to_numpy(labels))
    ep = _to_numpy(episode_ids).astype(int)

    unique_labels = sorted(set(L.tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    n = len(unique_labels)

    trans = np.zeros((n, n), dtype=int)
    for i in range(len(L) - 1):
        if ep[i] == ep[i + 1]:
            trans[label_to_idx[L[i]], label_to_idx[L[i + 1]]] += 1

    row_sums = trans.sum(axis=1, keepdims=True).clip(min=1)
    trans_norm = trans / row_sums

    data = [
        (str(unique_labels[to_i]), str(unique_labels[from_i]), float(trans_norm[from_i, to_i]))
        for from_i in range(n)
        for to_i in range(n)
    ]
    to_dim = f"To {label_name}"
    from_dim = f"From {label_name}"
    return hv.HeatMap(data, kdims=[to_dim, from_dim], vdims=["Probability"]).opts(
        cmap="YlOrRd",
        colorbar=True,
        width=450,
        height=400,
        tools=["hover"],
        xlabel=to_dim,
        ylabel=from_dim,
        title=title,
    )


def hv_chart_alignment(
    labels: torch.Tensor | np.ndarray,
    group_labels: torch.Tensor | np.ndarray,
    title: str = "Chart Alignment",
    label_name: str = "Chart",
    group_name: str = "Group",
) -> hv.HeatMap:
    """HoloViews heatmap of labels vs group alignment with AMI score."""
    from sklearn.metrics import adjusted_mutual_info_score

    L = np.asarray(_to_numpy(labels))
    G = _to_numpy(group_labels).astype(int)

    unique_labels = sorted(set(L.tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    n_labels = len(unique_labels)

    n_groups = G.max() + 1
    confusion = np.zeros((n_labels, n_groups), dtype=int)
    for lab, g in zip(L, G):
        confusion[label_to_idx[lab], g] += 1

    ami = adjusted_mutual_info_score(L, G)

    data = [
        (str(unique_labels[li]), str(gi), int(confusion[li, gi]))
        for li in range(n_labels)
        for gi in range(n_groups)
    ]
    return hv.HeatMap(data, kdims=[group_name, label_name], vdims=["Count"]).opts(
        cmap="Blues",
        colorbar=True,
        width=500,
        height=400,
        tools=["hover"],
        xlabel=group_name,
        ylabel=label_name,
        title=f"{title}  (AMI={ami:.3f})",
    )


def hv_dynamics_trajectory(
    z_pred: torch.Tensor | np.ndarray,
    z_target: torch.Tensor | np.ndarray,
    title: str = "Dynamics: Predicted vs Target",
) -> hv.Overlay:
    """HoloViews overlay of predicted vs target trajectories on the Poincaré disk."""
    zp = _pca_2d(_to_numpy(z_pred))
    zt = _pca_2d(_to_numpy(z_target))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = hv.Curve(
        (np.cos(theta), np.sin(theta)),
        kdims=["x"],
        vdims=["y"],
    ).opts(color="black", line_width=0.8, alpha=0.3)

    # Target trajectory
    target_line = hv.Curve(
        (zt[:, 0], zt[:, 1]),
        kdims=["x"],
        vdims=["y"],
    ).opts(color="blue", line_width=1.5, alpha=0.7)
    target_pts = hv.Points(
        {"x": zt[:, 0], "y": zt[:, 1], "step": np.arange(len(zt)), "z0": zt[:, 0], "z1": zt[:, 1]},
        kdims=["x", "y"],
        vdims=["step", "z0", "z1"],
    ).opts(color="blue", size=5, alpha=0.7, tools=["hover"])

    # Predicted trajectory
    pred_line = hv.Curve(
        (zp[:, 0], zp[:, 1]),
        kdims=["x"],
        vdims=["y"],
    ).opts(color="red", line_width=1.5, line_dash="dashed", alpha=0.7)
    pred_pts = hv.Points(
        {"x": zp[:, 0], "y": zp[:, 1], "step": np.arange(len(zp)), "z0": zp[:, 0], "z1": zp[:, 1]},
        kdims=["x", "y"],
        vdims=["step", "z0", "z1"],
    ).opts(color="red", size=5, marker="square", alpha=0.7, tools=["hover"])

    return (circle * target_line * target_pts * pred_line * pred_pts).opts(
        width=800,
        height=600,
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        aspect="equal",
        title=title,
    )


def full_diagnostic(
    encoder: torch.nn.Module,
    features: torch.Tensor,
    config,
    task_labels: torch.Tensor | np.ndarray | None = None,
    save_dir: str | None = None,
) -> dict[str, float]:
    """Run all diagnostics on a batch of features.

    Args:
        encoder: Trained TopoEncoderPrimitives.
        features: [N, D] feature tensor.
        config: VLAConfig.
        task_labels: Optional [N] task labels for alignment analysis.
        save_dir: If provided, save plots to this directory.

    Returns:
        Dict of diagnostic metrics.
    """
    import matplotlib.pyplot as plt

    device = next(encoder.parameters()).device
    encoder.eval()

    with torch.no_grad():
        (
            x_recon,
            _vq_loss,
            _enc_rw,
            _dec_rw,
            K_chart,
            z_geo,
            _z_n,
            _c_bar,
            _aux,
        ) = encoder(features.to(device))

    z_np = z_geo.cpu().numpy()
    K_np = K_chart.cpu().numpy()

    metrics: dict[str, float] = {}

    # Active charts
    unique_charts = len(np.unique(K_np))
    metrics["active_charts"] = unique_charts
    print(f"Active charts: {unique_charts}/{config.num_charts}")

    # Radial stats
    r = np.linalg.norm(z_np, axis=1)
    metrics["mean_radius"] = float(r.mean())
    metrics["std_radius"] = float(r.std())
    print(f"Radius: mean={r.mean():.3f}, std={r.std():.3f}")

    # Reconstruction R²
    x_np = features.cpu().numpy()
    xr_np = x_recon.cpu().numpy()
    ss_res = ((x_np - xr_np) ** 2).sum()
    ss_tot = ((x_np - x_np.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics["recon_r2"] = float(r2)
    print(f"Reconstruction R²: {r2:.4f}")

    # Conformal factors
    cf = conformal_factor_np(z_np)
    metrics["mean_conformal"] = float(cf.mean())
    print(f"Conformal factor: mean={cf.mean():.2f}")

    # Closure ratio (fraction of variance explained by chart membership)
    from sklearn.metrics import adjusted_mutual_info_score

    if task_labels is not None:
        tl = _to_numpy(task_labels).astype(int)
        ami = adjusted_mutual_info_score(K_np, tl)
        metrics["ami"] = float(ami)
        print(f"AMI (chart vs task): {ami:.4f}")

    # Generate plots
    figs = []

    fig1 = plot_poincare_disk(z_np, K_np, task_labels, title="Poincaré Disk Embedding")
    figs.append(("poincare_disk", fig1))

    fig2 = plot_radius_histogram(z_np)
    figs.append(("radius_hist", fig2))

    if task_labels is not None:
        fig3 = plot_chart_task_alignment(K_np, _to_numpy(task_labels))
        figs.append(("chart_task", fig3))

    if save_dir is not None:
        import os

        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs:
            fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"Plots saved to {save_dir}")
    else:
        plt.show()

    # Success criteria
    print("\n--- Success Criteria ---")
    print(f"  Active charts >= 3: {'PASS' if unique_charts >= 3 else 'FAIL'}")
    print(f"  Recon R² > 0.8:    {'PASS' if r2 > 0.8 else 'FAIL'}")
    if task_labels is not None and "ami" in metrics:
        print(f"  AMI > 0.3:         {'PASS' if metrics['ami'] > 0.3 else 'FAIL'}")

    return metrics
