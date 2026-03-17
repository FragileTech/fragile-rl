"""Pure plotting functions for the TopoEncoder learning dashboard.

Each function takes numpy arrays and returns a plot object (HoloViews or Plotly).
No state, no widgets.
"""

from __future__ import annotations

from bokeh.models import HoverTool, TapTool
import holoviews as hv
import numpy as np
import plotly.graph_objects as go


# Consistent color palette
COLORS = {
    "atlas": "#4c78a8",
    "std": "#f58518",
    "ae": "#54a24b",
}

SPLIT_OUTLINE_COLORS = {
    "train": "#1f1f1f",
    "test": "#d62728",
    "unknown": "#7f7f7f",
}

SPLIT_MARKERS = {
    "train": "circle",
    "test": "diamond",
    "unknown": "square",
}


def _to_numpy(t) -> np.ndarray:
    """Convert tensor/array-like to numpy."""
    if hasattr(t, "cpu"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) integers."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def chart_to_label_map(K_chart: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build majority-vote mapping from chart index to most frequent label.

    Returns an array of length ``num_charts`` where ``map[c]`` is the label
    most frequently assigned to chart ``c``.
    """
    num_charts = int(K_chart.max()) + 1
    mapping = np.zeros(num_charts, dtype=int)
    for c in range(num_charts):
        mask = K_chart == c
        if mask.any():
            vals, counts = np.unique(labels[mask], return_counts=True)
            mapping[c] = vals[counts.argmax()]
    return mapping


def plot_chart_usage(usage_array: np.ndarray) -> hv.Bars:
    """Bar chart of routing mass per chart."""
    usage = _to_numpy(usage_array)
    hover = HoverTool(tooltips=[("Chart", "@Chart"), ("Usage", "@Usage{0.4f}")])
    bars = hv.Bars(
        [(f"C{i}", float(v)) for i, v in enumerate(usage)],
        kdims="Chart",
        vdims="Usage",
    )
    return bars.opts(
        title="Chart Usage", width=700, height=300, color=COLORS["atlas"], tools=[hover]
    )


def build_latent_scatter(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray,
    correct: np.ndarray,
    color_by: str,
    point_size: int,
    dim_i: int,
    dim_j: int,
    indices: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
    K_code: np.ndarray | None = None,
    show_code_centers: bool = False,
    show_points: bool = True,
    marker_groups: np.ndarray | None = None,
    marker_markers: dict[str, str] | None = None,
    marker_outline_colors: dict[str, str] | None = None,
    trajectory_episode_ids: np.ndarray | None = None,
    trajectory_timesteps: np.ndarray | None = None,
    trajectory_episode: int | None = None,
    trajectory_color: str = "red",
) -> hv.Points | hv.Overlay:
    """Build a single 2D scatter for one dimension pair.

    Parameters
    ----------
    indices
        Original dataset index for each displayed point (enables click-to-inspect).
    confidence
        Per-point confidence (max softmax probability) in [0, 1].
    alpha_by_confidence
        When True, map per-point alpha to the confidence value.
    """
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int)
    corr = _to_numpy(correct).astype(int)
    correct_str = np.where(corr == 1, "yes", "no")

    # Convert integer labels/charts to strings so holoviews treats them as
    # categorical (discrete colors) rather than continuous.
    charts_str = np.array([str(v) for v in charts])

    # Auto-detect: many unique labels -> continuous colormap, few -> categorical
    n_unique_labels = len(np.unique(labs))
    if n_unique_labels <= 20:
        labs_str = np.array([str(v) for v in labs])
        label_col, label_cmap, label_colorbar = "label_str", "Category10", False
    else:
        # Use a continuous numeric color scale for high-cardinality labels
        label_col, label_cmap, label_colorbar = "label", "Viridis", True

    if color_by == "confidence":
        color_col, cmap, show_colorbar = "confidence", "Viridis", True
    elif color_by == "chart":
        color_col, cmap, show_colorbar = "chart_str", "Category10", False
    elif color_by == "correct":
        color_col, cmap, show_colorbar = "correct_str", {"yes": "#54a24b", "no": "#e45756"}, False
    else:
        color_col, cmap, show_colorbar = label_col, label_cmap, label_colorbar

    data = {
        "x": z[:, dim_i].copy(),
        "y": z[:, dim_j].copy(),
        "label": labs,
        "chart": charts,
        "chart_str": charts_str,
        "correct": corr,
        "correct_str": correct_str,
    }
    vdims = ["label", "chart", "chart_str", "correct", "correct_str"]
    if n_unique_labels <= 20:
        data["label_str"] = labs_str
        vdims.append("label_str")
    else:
        data["label"] = labs.astype(float)

    if confidence is not None:
        data["confidence"] = confidence.astype(float)
        vdims.append("confidence")

    if indices is not None:
        data["idx"] = np.asarray(indices)
        vdims.append("idx")

    tooltips = [
        (f"z{dim_i}", "@x{0.3f}"),
        (f"z{dim_j}", "@y{0.3f}"),
        ("Label", "@label"),
        ("Chart", "@chart"),
        ("Correct", "@correct_str"),
    ]
    if confidence is not None:
        tooltips.append(("Confidence", "@confidence{0.3f}"))
    hover = HoverTool(tooltips=tooltips)

    opts_kw: dict = {
        "color": color_col,
        "cmap": cmap,
        "size": point_size,
        "width": 350,
        "height": 350,
        "xlabel": f"z{dim_i}",
        "ylabel": f"z{dim_j}",
        "title": f"z{dim_i} vs z{dim_j}",
        "colorbar": show_colorbar,
        "tools": [hover, TapTool()],
    }
    if color_by == "confidence" and confidence is not None:
        opts_kw["clim"] = (0, 1)
    if alpha_by_confidence and confidence is not None:
        opts_kw["alpha"] = hv.dim("confidence")

    if show_points:
        scatter = hv.Points(data, kdims=["x", "y"], vdims=vdims).opts(**opts_kw)
    else:
        # Invisible placeholder that preserves axes range for overlays
        scatter = hv.Points(data, kdims=["x", "y"], vdims=vdims).opts(
            **{**opts_kw, "alpha": 0, "size": 0},
        )

    result = scatter

    if show_code_centers and K_code is not None:
        codes_arr = _to_numpy(K_code).astype(int)
        center_x, center_y, center_chart, center_code = [], [], [], []
        unique_charts = np.unique(charts)
        for c in unique_charts:
            c_mask = charts == c
            for k in np.unique(codes_arr[c_mask]):
                mask = c_mask & (codes_arr == k)
                if mask.any():
                    center_x.append(z[:, dim_i][mask].mean())
                    center_y.append(z[:, dim_j][mask].mean())
                    center_chart.append(int(c))
                    center_code.append(int(k))
        if center_x:
            center_colors = [_CATEGORY10[c % len(_CATEGORY10)] for c in center_chart]
            center_data = {
                "x": np.array(center_x),
                "y": np.array(center_y),
                "chart": center_chart,
                "code": center_code,
                "color": center_colors,
            }
            center_hover = HoverTool(
                tooltips=[
                    ("Chart", "@chart"),
                    ("Code", "@code"),
                    (f"z{dim_i}", "@x{0.3f}"),
                    (f"z{dim_j}", "@y{0.3f}"),
                ]
            )
            result = result * hv.Points(
                center_data,
                kdims=["x", "y"],
                vdims=["chart", "code", "color"],
                label="code centers",
            ).opts(
                color="color",
                marker="diamond",
                size=point_size * 3,
                alpha=0.7,
                tools=[center_hover],
            )

    if show_points and marker_groups is not None:
        groups = np.asarray(marker_groups).astype(str)
        marker_markers = marker_markers or SPLIT_MARKERS
        marker_outline_colors = marker_outline_colors or SPLIT_OUTLINE_COLORS
        split_hover = HoverTool(tooltips=[("Split", "@split")])
        overlays = [result]
        for group in np.unique(groups):
            mask = groups == group
            if not np.any(mask):
                continue
            overlays.append(
                hv.Points(
                    {
                        "x": z[mask, dim_i].copy(),
                        "y": z[mask, dim_j].copy(),
                        "split": groups[mask],
                    },
                    kdims=["x", "y"],
                    vdims=["split"],
                    label=f"{group} split",
                ).opts(
                    marker=marker_markers.get(group, "circle"),
                    size=max(point_size + 3, 5),
                    color=marker_outline_colors.get(group, "#333333"),
                    fill_alpha=0.0,
                    line_alpha=0.95,
                    line_width=1.4,
                    alpha=1.0,
                    tools=[split_hover, TapTool()],
                ),
            )
        result = hv.Overlay(overlays)

    # Episode trajectory overlay
    if (
        trajectory_episode is not None
        and trajectory_episode_ids is not None
        and trajectory_timesteps is not None
    ):
        traj_overlay = _build_trajectory_overlay_2d(
            z,
            dim_i,
            dim_j,
            trajectory_episode_ids,
            trajectory_timesteps,
            trajectory_episode,
            trajectory_color,
        )
        if traj_overlay is not None:
            result = result * traj_overlay

    return result


def plot_latent_2d_slices(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 3,
    indices: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
    K_code: np.ndarray | None = None,
    show_code_centers: bool = False,
    show_points: bool = True,
    marker_groups: np.ndarray | None = None,
    marker_markers: dict[str, str] | None = None,
    marker_outline_colors: dict[str, str] | None = None,
) -> hv.Layout:
    """2D scatter panels for every pair among the first 3 latent dims.

    ``color_by`` selects the coloring: ``"label"``, ``"chart"``, ``"correct"``,
    or ``"confidence"``.
    """
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int) if K_chart is not None else np.zeros_like(labs)
    corr = _to_numpy(correct).astype(int) if correct is not None else np.ones_like(labs)
    dim = z.shape[1]

    pairs = []
    if dim >= 2:
        pairs.append((0, 1))
    if dim >= 3:
        pairs.append((0, 2))
        pairs.append((1, 2))

    panels = []
    for i, j in pairs:
        scatter = build_latent_scatter(
            z,
            labs,
            charts,
            corr,
            color_by,
            point_size,
            i,
            j,
            indices=indices,
            confidence=confidence,
            alpha_by_confidence=alpha_by_confidence,
            K_code=K_code,
            show_code_centers=show_code_centers,
            show_points=show_points,
            marker_groups=marker_groups,
            marker_markers=marker_markers,
            marker_outline_colors=marker_outline_colors,
        )
        panels.append(scatter)

    if not panels:
        return hv.Layout([hv.Points([], kdims=["x", "y"]).opts(title="No latent dims")])
    return hv.Layout(panels).opts(shared_axes=False).cols(min(3, len(panels)))


_CATEGORY10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _seg(
    lx: list,
    ly: list,
    lz: list,
    a: np.ndarray,
    b: np.ndarray,
) -> None:
    """Append a line segment with None separator for Plotly."""
    lx.extend([a[0], b[0], None])
    ly.extend([a[1], b[1], None])
    lz.extend([a[2], b[2], None])


def _add_hierarchy_traces(
    traces: list,
    x: np.ndarray,
    y: np.ndarray,
    z_ax: np.ndarray,
    charts: np.ndarray,
    codes: np.ndarray,
    point_size: int,
    line_color_mode: str,
    line_width: float,
    hierarchy_z: bool = False,
    show_chart_centers: bool = True,
    show_code_centers: bool = True,
    show_lines: bool = True,
    show_leaf_lines: bool = True,
) -> None:
    """Append hierarchy center markers and tree-edge lines to *traces* (in-place).

    When *hierarchy_z* is True (2D embeddings), z-coordinates are overridden to
    show hierarchy levels: root=1.0, chart centers=0.66, symbol centers=0.33,
    data points=0.0.
    """
    unique_charts = np.unique(charts)

    # Compute chart centers
    chart_centers: dict[int, np.ndarray] = {}
    for c in unique_charts:
        mask = charts == c
        cz = 0.66 if hierarchy_z else z_ax[mask].mean()
        chart_centers[c] = np.array([x[mask].mean(), y[mask].mean(), cz])

    # Compute symbol (chart, code) centers
    symbol_centers: dict[tuple[int, int], np.ndarray] = {}
    for c in unique_charts:
        c_mask = charts == c
        for k in np.unique(codes[c_mask]):
            mask = c_mask & (codes == k)
            sz = 0.33 if hierarchy_z else z_ax[mask].mean()
            symbol_centers[c, k] = np.array([x[mask].mean(), y[mask].mean(), sz])

    # Root: centered above all chart centers
    if hierarchy_z:
        all_chart_xy = np.array([cc[:2] for cc in chart_centers.values()])
        root = np.array([all_chart_xy[:, 0].mean(), all_chart_xy[:, 1].mean(), 1.0])
    else:
        root = np.array([0.0, 0.0, 0.0])

    # --- Center markers ---
    if show_chart_centers:
        # Root
        traces.append(
            go.Scatter3d(
                x=[root[0]],
                y=[root[1]],
                z=[root[2]],
                mode="markers",
                name="root",
                marker={"size": point_size * 4, "color": "black", "symbol": "diamond"},
                hoverinfo="name",
                showlegend=False,
            )
        )
        # Chart centers
        for c, ctr in chart_centers.items():
            col = _CATEGORY10[int(c) % len(_CATEGORY10)]
            traces.append(
                go.Scatter3d(
                    x=[ctr[0]],
                    y=[ctr[1]],
                    z=[ctr[2]],
                    mode="markers",
                    name=f"chart {c} center",
                    marker={"size": point_size * 3, "color": col, "symbol": "diamond"},
                    hoverinfo="name",
                    showlegend=False,
                )
            )
    if show_code_centers:
        # Symbol centers
        for (c, k), ctr in symbol_centers.items():
            col = _CATEGORY10[int(c) % len(_CATEGORY10)]
            traces.append(
                go.Scatter3d(
                    x=[ctr[0]],
                    y=[ctr[1]],
                    z=[ctr[2]],
                    mode="markers",
                    name=f"chart {c} code {k}",
                    marker={
                        "size": point_size * 2,
                        "color": col,
                        "opacity": 0.6,
                        "symbol": "diamond",
                    },
                    hoverinfo="name",
                    showlegend=False,
                )
            )

    # --- Line traces ---
    if not show_lines:
        return
    if line_color_mode == "black":
        # Single trace with all edges
        lx, ly, lz = [], [], []
        for c in unique_charts:
            _seg(lx, ly, lz, root, chart_centers[c])
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[c, k]
                _seg(lx, ly, lz, chart_centers[c], sym)
                if show_leaf_lines:
                    for idx in np.where(c_mask & (codes == k))[0]:
                        _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
        traces.append(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                name="hierarchy",
                line={"color": "black", "width": line_width},
                hoverinfo="none",
                showlegend=False,
            )
        )
    elif line_color_mode == "chart":
        # One trace per chart
        for c in unique_charts:
            col = _CATEGORY10[int(c) % len(_CATEGORY10)]
            lx, ly, lz = [], [], []
            _seg(lx, ly, lz, root, chart_centers[c])
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[c, k]
                _seg(lx, ly, lz, chart_centers[c], sym)
                if show_leaf_lines:
                    for idx in np.where(c_mask & (codes == k))[0]:
                        _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    name=f"tree chart {c}",
                    line={"color": col, "width": line_width},
                    hoverinfo="none",
                    showlegend=False,
                )
            )
    else:  # "symbol"
        for c in unique_charts:
            col_chart = _CATEGORY10[int(c) % len(_CATEGORY10)]
            # root → chart edge (chart color)
            lx, ly, lz = [], [], []
            _seg(lx, ly, lz, root, chart_centers[c])
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line={"color": col_chart, "width": line_width},
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            c_mask = charts == c
            for k in np.unique(codes[c_mask]):
                sym = symbol_centers[c, k]
                # chart → symbol (chart color)
                lx, ly, lz = [], [], []
                _seg(lx, ly, lz, chart_centers[c], sym)
                traces.append(
                    go.Scatter3d(
                        x=lx,
                        y=ly,
                        z=lz,
                        mode="lines",
                        line={"color": col_chart, "width": line_width},
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                # symbol → data (per-symbol color)
                if show_leaf_lines:
                    sym_col = _CATEGORY10[int(k) % len(_CATEGORY10)]
                    lx, ly, lz = [], [], []
                    for idx in np.where(c_mask & (codes == k))[0]:
                        _seg(lx, ly, lz, sym, np.array([x[idx], y[idx], z_ax[idx]]))
                    if lx:
                        traces.append(
                            go.Scatter3d(
                                x=lx,
                                y=ly,
                                z=lz,
                                mode="lines",
                                line={"color": sym_col, "width": line_width},
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )


def plot_latent_3d(
    z_geo: np.ndarray,
    labels: np.ndarray,
    K_chart: np.ndarray | None = None,
    correct: np.ndarray | None = None,
    color_by: str = "label",
    point_size: int = 2,
    K_code: np.ndarray | None = None,
    show_hierarchy: bool = False,
    tree_line_color: str = "black",
    tree_line_width: float = 0.5,
    confidence: np.ndarray | None = None,
    alpha_by_confidence: bool = False,
    show_points: bool = True,
    show_chart_centers: bool = False,
    show_code_centers: bool = False,
    show_tree_lines: bool = False,
    show_leaf_lines: bool = True,
    marker_groups: np.ndarray | None = None,
    marker_markers: dict[str, str] | None = None,
    marker_outline_colors: dict[str, str] | None = None,
    trajectory_episode_ids: np.ndarray | None = None,
    trajectory_timesteps: np.ndarray | None = None,
    trajectory_episode: int | None = None,
    trajectory_color: str = "red",
) -> go.Figure:
    """3D scatter of z_geo[:,0:3] colored by label, chart, correct, or confidence."""
    z = _to_numpy(z_geo)
    labs = _to_numpy(labels).astype(int)
    charts = _to_numpy(K_chart).astype(int) if K_chart is not None else np.zeros_like(labs)
    corr = _to_numpy(correct).astype(int) if correct is not None else np.ones_like(labs)

    ndim = z.shape[1]
    x = z[:, 0] if ndim > 0 else np.zeros(len(z))
    y = z[:, 1] if ndim > 1 else np.zeros(len(z))
    z_ax = z[:, 2] if ndim > 2 else np.zeros(len(z))
    hierarchy_z = ndim < 3  # use z-axis for hierarchy levels when embedding is 2D

    base_opacity = 0.7
    traces = []

    # Backward compat: if show_hierarchy is True and no granular bools were set, enable all
    if show_hierarchy and not (show_chart_centers or show_code_centers or show_tree_lines):
        show_chart_centers = show_code_centers = show_tree_lines = True

    color_label = color_by.capitalize()

    # Detect whether labels should use continuous or categorical coloring
    n_unique_labels = len(np.unique(labs))
    use_continuous_labels = n_unique_labels > 20

    if not show_points:
        pass  # skip scatter traces
    elif color_by == "confidence" and confidence is not None:
        # Continuous colorscale — single trace
        conf = confidence.astype(float)
        hover_text = [
            f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
            f"<br>Label={labs[k]}<br>Chart={charts[k]}"
            f"<br>Correct={'yes' if corr[k] else 'no'}"
            f"<br>Confidence={conf[k]:.3f}"
            for k in range(len(x))
        ]
        marker: dict = {
            "size": point_size,
            "color": conf,
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "Confidence"},
            "opacity": base_opacity,
        }
        # Plotly scatter3d.marker.opacity is scalar-only; use global opacity
        # and encode per-point alpha via the colorscale when requested.
        if alpha_by_confidence:
            marker["opacity"] = 1.0
            # Build per-point RGBA from Viridis sampled at conf value
            from matplotlib.cm import viridis as _viridis_cm

            rgba_arr = _viridis_cm(conf)  # (N, 4)
            rgba_arr[:, 3] = conf  # set alpha = confidence
            marker.pop("colorscale", None)
            marker.pop("cmin", None)
            marker.pop("cmax", None)
            marker.pop("colorbar", None)
            marker["color"] = [
                f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})"
                for r, g, b, a in rgba_arr
            ]
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z_ax,
                mode="markers",
                name="confidence",
                marker=marker,
                text=hover_text,
                hoverinfo="text",
            )
        )
        color_label = "Confidence"
    elif color_by == "label" and use_continuous_labels:
        # Continuous colorscale on raw labels
        color_vals_f = labs.astype(float)
        hover_text = [
            f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
            f"<br>Label={labs[k]}<br>Chart={charts[k]}"
            f"<br>Correct={'yes' if corr[k] else 'no'}"
            for k in range(len(x))
        ]
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z_ax,
                mode="markers",
                name="label",
                marker={
                    "size": point_size,
                    "color": color_vals_f,
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Label"},
                    "opacity": base_opacity,
                },
                text=hover_text,
                hoverinfo="text",
            )
        )
        color_label = "Label"
    else:
        # Categorical traces
        if color_by == "chart":
            color_vals = charts
            color_label = "Chart"
            palette = _CATEGORY10
        elif color_by == "correct":
            color_vals = corr
            color_label = "Correct"
            palette = ["#e45756", "#54a24b"]
        else:
            color_vals = labs
            color_label = "Label"
            palette = _CATEGORY10

        categories = sorted(np.unique(color_vals))
        cat_names = {0: "no", 1: "yes"} if color_by == "correct" else None

        for cat in categories:
            mask = color_vals == cat
            cat_color = palette[int(cat) % len(palette)]
            name = cat_names[cat] if cat_names else str(cat)
            hover_text = [
                f"z0={x[k]:.3f}<br>z1={y[k]:.3f}<br>z2={z_ax[k]:.3f}"
                f"<br>Label={labs[k]}<br>Chart={charts[k]}"
                f"<br>Correct={'yes' if corr[k] else 'no'}"
                for k in np.where(mask)[0]
            ]
            mk: dict = {"size": point_size}
            if alpha_by_confidence and confidence is not None:
                # Encode per-point alpha via RGBA color strings
                r, g, b = _hex_to_rgb(cat_color)
                alphas = confidence[mask].astype(float)
                mk["color"] = [f"rgba({r},{g},{b},{a:.3f})" for a in alphas]
                mk["opacity"] = 1.0
            else:
                mk["color"] = cat_color
                mk["opacity"] = base_opacity
            traces.append(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z_ax[mask],
                    mode="markers",
                    name=name,
                    marker=mk,
                    text=hover_text,
                    hoverinfo="text",
                )
            )

    if show_points and marker_groups is not None:
        _add_split_marker_traces_3d(
            traces,
            x,
            y,
            z_ax,
            marker_groups,
            point_size,
            marker_markers=marker_markers,
            marker_outline_colors=marker_outline_colors,
        )

    # Hierarchy tree overlay
    if (show_chart_centers or show_code_centers or show_tree_lines) and K_code is not None:
        codes = _to_numpy(K_code).astype(int)
        _add_hierarchy_traces(
            traces,
            x,
            y,
            z_ax,
            charts,
            codes,
            point_size,
            tree_line_color,
            tree_line_width,
            hierarchy_z=hierarchy_z,
            show_chart_centers=show_chart_centers,
            show_code_centers=show_code_centers,
            show_lines=show_tree_lines,
            show_leaf_lines=show_leaf_lines,
        )

    # Episode trajectory overlay
    if (
        trajectory_episode is not None
        and trajectory_episode_ids is not None
        and trajectory_timesteps is not None
    ):
        _add_trajectory_traces_3d(
            traces,
            x,
            y,
            z_ax,
            trajectory_episode_ids,
            trajectory_timesteps,
            trajectory_episode,
            trajectory_color,
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Latent Space (color={color_label})",
        scene={
            "xaxis_title": "z0",
            "yaxis_title": "z1",
            "zaxis_title": "hierarchy" if hierarchy_z else "z2",
            "xaxis": {"range": [-1, 1]},
            "yaxis": {"range": [-1, 1]},
            "zaxis": {"range": [-0.1, 1.1]} if hierarchy_z else {"range": [-1, 1]},
        },
        width=700,
        height=600,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig
