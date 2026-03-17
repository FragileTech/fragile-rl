"""Panel-based interactive dashboard for VLA World Model checkpoint inspection.

Reuses latent-space visualisation patterns from the TopoEncoder dashboard
(``plots.py``) but replaces class labels with task labels, removes all
accuracy/classification widgets, and shows original LeRobot camera images.

Usage:
    uv run fragile vla-dashboard --port 5009
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import io
import json
import logging
import os
import pathlib
import re
import traceback

import holoviews as hv
import numpy as np
import panel as pn
import torch
import torch.nn.functional as F

from fragile.layers import FactorizedJumpOperator, TopoEncoder
from fragile.vla.extract_features import load_feature_cache_metadata
from fragile.vla.plots import (
    _to_numpy,
    build_latent_scatter,
    plot_chart_usage,
    plot_latent_3d,
)


logger = logging.getLogger(__name__)

os.environ.setdefault("PLOTLY_RENDERER", "json")
hv.extension("bokeh")

__all__ = ["create_app"]

# ---------------------------------------------------------------------------
# Checkpoint filename patterns
# ---------------------------------------------------------------------------

# p{phase}_epoch_{epoch}.pt  OR  epoch_{epoch}.pt  OR  checkpoint_final.pt
_CKPT_RE = re.compile(r"(?:p(\d+)_)?(?:epoch_(\d+)|checkpoint_final)\.pt$")

# Keys forwarded to TopoEncoder.__init__ from checkpoint args dict
_ENCODER_INIT_KEYS = {
    "input_dim",
    "hidden_dim",
    "latent_dim",
    "num_charts",
    "codes_per_chart",
    "covariant_attn",
    "covariant_attn_tensorization",
    "covariant_attn_rank",
    "covariant_attn_tau_min",
    "covariant_attn_denom_min",
    "covariant_attn_use_transport",
    "covariant_attn_transport_eps",
    "conv_backbone",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VLACheckpointInfo:
    """Metadata about a discovered VLA checkpoint."""

    path: str
    phase: int
    epoch: int
    label: str  # human-friendly label for selector


@dataclass
class VLALoaded:
    """Loaded VLA checkpoint data ready for inference."""

    encoder: TopoEncoder
    jump_op: FactorizedJumpOperator
    world_model: object | None  # GeometricWorldModel or None
    probe: object | None  # EnclosureProbe or None
    args: dict
    epoch: int
    phase: int


# ---------------------------------------------------------------------------
# Scanning & loading
# ---------------------------------------------------------------------------


def scan_vla_runs(outputs_dir: str) -> list[VLACheckpointInfo]:
    """Scan *outputs_dir* for VLA checkpoint files."""
    results: list[VLACheckpointInfo] = []
    if not os.path.isdir(outputs_dir):
        return results
    for root, _dirs, files in os.walk(outputs_dir):
        for fname in sorted(files):
            m = _CKPT_RE.search(fname)
            if not m:
                continue
            phase = int(m.group(1)) if m.group(1) else 0
            epoch = int(m.group(2)) if m.group(2) else -1
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, outputs_dir)
            label = f"{rel}  (P{phase} E{epoch})" if epoch >= 0 else f"{rel}  (final)"
            results.append(VLACheckpointInfo(path=path, phase=phase, epoch=epoch, label=label))
    return results


def load_vla_checkpoint(ckpt_path: str) -> VLALoaded:
    """Load a VLA checkpoint and reconstruct models."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    args = ckpt.get("args") or ckpt.get("config") or {}

    # Encoder state: key may be "model" (unsup / joint) or "encoder" (3-phase train.py)
    enc_state = ckpt.get("model") or ckpt.get("encoder")
    if enc_state is None:
        msg = "Checkpoint has no 'model' or 'encoder' state_dict"
        raise RuntimeError(msg)

    # Build encoder kwargs from args
    enc_kwargs: dict = {}
    for k in _ENCODER_INIT_KEYS:
        # Handle argparse-style keys (e.g. hidden_dim vs hidden-dim)
        for candidate in (k, k.replace("_", "-")):
            if candidate in args:
                enc_kwargs[k] = args[candidate]
                break
    # Defaults
    enc_kwargs.setdefault("input_dim", args.get("input_dim", args.get("feature_dim", 720)))
    enc_kwargs.setdefault("hidden_dim", 256)
    enc_kwargs.setdefault("latent_dim", args.get("latent_dim", 16))
    enc_kwargs.setdefault("num_charts", args.get("num_charts", 8))
    enc_kwargs.setdefault("codes_per_chart", args.get("codes_per_chart", 32))
    enc_kwargs.setdefault("covariant_attn", True)
    enc_kwargs.setdefault("covariant_attn_tensorization", "full")
    enc_kwargs.setdefault("soft_equiv_metric", True)
    enc_kwargs.setdefault("conv_backbone", False)

    encoder = TopoEncoder(film_conditioning=True, **enc_kwargs)
    result = encoder.load_state_dict(enc_state, strict=False)
    if result.missing_keys:
        logger.warning("Encoder: %d missing keys", len(result.missing_keys))
    encoder.eval()

    # Jump operator
    jump_state = ckpt.get("jump_op")
    jump_op = FactorizedJumpOperator(
        num_charts=enc_kwargs["num_charts"],
        latent_dim=enc_kwargs["latent_dim"],
    )
    if jump_state is not None:
        jump_op.load_state_dict(jump_state, strict=False)
    jump_op.eval()

    # World model (optional)
    world_model = None
    wm_state = ckpt.get("world_model")
    if wm_state is not None:
        try:
            from fragile.vla.covariant_world_model import GeometricWorldModel

            wm_kwargs: dict = {
                "latent_dim": enc_kwargs["latent_dim"],
                "action_dim": args.get("action_dim", 6),
                "num_charts": enc_kwargs["num_charts"],
                "d_model": args.get("wm_d_model"),
                "hidden_dim": args.get("wm_hidden_dim"),
                "dt": args.get("wm_dt"),
                "gamma_friction": args.get("wm_gamma_friction"),
                "T_c": args.get("wm_T_c"),
                "alpha_potential": args.get("wm_alpha_potential"),
                "beta_curl": args.get("wm_beta_curl"),
                "gamma_risk": args.get("wm_gamma_risk"),
                "use_boris": args.get("wm_use_boris"),
                "use_jump": args.get("wm_use_jump"),
                "n_refine_steps": args.get("wm_refine_steps"),
                "jump_beta": args.get("wm_jump_beta"),
                "min_length": args.get("wm_min_length"),
                "risk_metric_alpha": args.get("wm_risk_metric_alpha"),
            }
            # Drop None values so GeometricWorldModel defaults are used
            wm_kwargs = {k: v for k, v in wm_kwargs.items() if v is not None}
            world_model = GeometricWorldModel(**wm_kwargs)
            world_model.load_state_dict(wm_state, strict=False)
            world_model.eval()
        except Exception:
            logger.warning("Could not load world model: %s", traceback.format_exc())
            world_model = None

    # Enclosure probe (optional)
    probe = None
    probe_state = ckpt.get("probe")
    if probe_state is not None:
        try:
            from fragile.losses.macro import EnclosureProbe

            probe = EnclosureProbe(
                chart_dim=enc_kwargs["latent_dim"],
                ztex_dim=enc_kwargs["latent_dim"],
                action_dim=args.get("action_dim", 6),
                num_charts=enc_kwargs["num_charts"],
                codes_per_chart=enc_kwargs.get("codes_per_chart", 32),
                hidden_dim=args.get("enclosure_probe_hidden_dim", 128),
            )
            probe.load_state_dict(probe_state, strict=False)
            probe.eval()
        except Exception:
            logger.warning("Could not load enclosure probe: %s", traceback.format_exc())
            probe = None

    return VLALoaded(
        encoder=encoder,
        jump_op=jump_op,
        world_model=world_model,
        probe=probe,
        args=args,
        epoch=ckpt.get("epoch", -1),
        phase=ckpt.get("phase", 0),
    )


# ---------------------------------------------------------------------------
# Feature + task label loading
# ---------------------------------------------------------------------------


def load_feature_cache(feature_dir: str) -> dict:
    """Load all features + task labels from the feature cache directory.

    Returns dict with keys:
        features: [N, D] tensor
        task_labels: [N] int array
        episode_ids: [N] int array
        timesteps: [N] int array
        split_labels: [N] string array
    """
    from pathlib import Path

    cache = Path(feature_dir)
    meta = load_feature_cache_metadata(cache)
    episode_ids_list: list[int] = meta.get("episode_ids", [])
    train_episode_ids = {int(ep_id) for ep_id in meta.get("train_episode_ids", [])}
    test_episode_ids = {int(ep_id) for ep_id in meta.get("test_episode_ids", [])}

    all_features: list[torch.Tensor] = []
    all_task_labels: list[torch.Tensor] = []
    all_ep_ids: list[int] = []
    all_timesteps: list[int] = []
    all_split_labels: list[str] = []

    for ep_id in episode_ids_list:
        ep_dir = cache / f"episode_{ep_id}"
        feat_path = ep_dir / "features.pt"
        if not feat_path.exists():
            continue
        feat = torch.load(feat_path, weights_only=True)  # [T, D]
        T = feat.shape[0]
        all_features.append(feat)

        task_path = ep_dir / "task_indices.pt"
        if task_path.exists():
            tl = torch.load(task_path, weights_only=True)
            if tl.shape[0] < T:
                tl = F.pad(tl, (0, T - tl.shape[0]), value=0)
            all_task_labels.append(tl[:T])
        else:
            all_task_labels.append(torch.zeros(T, dtype=torch.long))

        if ep_id in test_episode_ids:
            split_label = "test"
        elif ep_id in train_episode_ids:
            split_label = "train"
        else:
            split_label = "unknown"

        all_ep_ids.extend([ep_id] * T)
        all_timesteps.extend(range(T))
        all_split_labels.extend([split_label] * T)

    if not all_features:
        return {
            "features": torch.zeros(0, 720),
            "task_labels": np.zeros(0, dtype=int),
            "episode_ids": np.zeros(0, dtype=int),
            "timesteps": np.zeros(0, dtype=int),
            "split_labels": np.zeros(0, dtype=str),
            "meta": meta,
        }

    return {
        "features": torch.cat(all_features, dim=0),
        "task_labels": _to_numpy(torch.cat(all_task_labels, dim=0)).astype(int),
        "episode_ids": np.array(all_ep_ids, dtype=int),
        "timesteps": np.array(all_timesteps, dtype=int),
        "split_labels": np.array(all_split_labels, dtype=str),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Lazy image provider
# ---------------------------------------------------------------------------


class VLAImageProvider:
    """Lazy loader for LeRobot camera images via direct video decoding."""

    def __init__(self, dataset_name: str, feature_cache_dir: str):
        self._dataset_name = dataset_name
        self._cache_dir = feature_cache_dir
        self._video_root: str | None = None
        self._ep_offsets: dict[int, int] | None = None  # ep_id -> global frame offset
        self._cameras: list[str] | None = None
        self._ready = False

    def _ensure_loaded(self) -> bool:
        if self._ready:
            return True
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(self._dataset_name)
            self._video_root = str(ds.root / "videos")
            self._cameras = [
                k.replace("observation.images.", "") for k in (ds.meta.camera_keys or [])
            ]

            # Build episode-start-offset table from the parquet metadata
            hf = ds.hf_dataset
            ep_col = np.asarray(hf["episode_index"])
            offset = 0
            self._ep_offsets = {}
            for ep_id in sorted(np.unique(ep_col)):
                self._ep_offsets[int(ep_id)] = offset
                offset += int((ep_col == ep_id).sum())

            self._ready = True
            logger.info(
                "Image provider ready: %d episodes, cameras=%s",
                len(self._ep_offsets),
                self._cameras,
            )
            return True
        except Exception:
            logger.warning("Could not init image provider: %s", traceback.format_exc())
            return False

    def get_image(self, ep_id: int, timestep: int, camera: str = "top") -> np.ndarray | None:
        """Return [H, W, 3] uint8 image or None."""
        if not self._ensure_loaded():
            return None
        try:
            import imageio.v3 as iio

            offset = self._ep_offsets.get(ep_id)
            if offset is None:
                return None
            global_idx = offset + timestep

            # Resolve camera to video path
            cam_key = f"observation.images.{camera}"
            video_dir = os.path.join(self._video_root, cam_key, "chunk-000")
            if not os.path.isdir(video_dir):
                # Fallback: try first available camera
                for c in self._cameras or []:
                    alt_key = f"observation.images.{c}"
                    alt_dir = os.path.join(self._video_root, alt_key, "chunk-000")
                    if os.path.isdir(alt_dir):
                        video_dir = alt_dir
                        break
                else:
                    return None

            # Find the video file (typically file-000.mp4)
            video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
            if not video_files:
                return None
            video_path = os.path.join(video_dir, min(video_files))

            return iio.imread(video_path, index=global_idx, plugin="pyav")
        except Exception:
            logger.debug(
                "Image read failed ep=%d t=%d: %s", ep_id, timestep, traceback.format_exc()
            )
            return None


# ---------------------------------------------------------------------------
# Helpers for image/latent display
# ---------------------------------------------------------------------------


def _tensor_to_png_pane(img_tensor_or_array, width: int = 150):
    """Convert image tensor/array to Panel PNG pane."""
    from PIL import Image

    if isinstance(img_tensor_or_array, torch.Tensor):
        arr = (img_tensor_or_array.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = img_tensor_or_array
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return pn.pane.PNG(buf.getvalue(), width=width)


def _numpy_to_png_pane(arr: np.ndarray, width: int = 150):
    """Convert [H, W, 3] uint8 array to Panel PNG pane."""
    from PIL import Image

    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return pn.pane.PNG(buf.getvalue(), width=width)


def _symbol_seed(symbol_label: str) -> int:
    """Deterministic integer seed derived from a symbol label."""
    return sum(ord(ch) for ch in symbol_label)


@torch.no_grad()
def infer_symbol_assignments(
    loaded: VLALoaded,
    features: torch.Tensor,
    *,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Infer chart/code assignments for every cached feature frame."""
    if features.shape[0] == 0:
        return {
            "charts": np.zeros(0, dtype=int),
            "codes": np.zeros(0, dtype=int),
            "labels": np.zeros(0, dtype=str),
        }

    chart_batches: list[np.ndarray] = []
    code_batches: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        enc_out = loaded.encoder.encoder(batch)
        chart_batches.append(_to_numpy(enc_out[0]).astype(int))
        code_batches.append(_to_numpy(enc_out[1]).astype(int))

    charts = np.concatenate(chart_batches, axis=0)
    codes = np.concatenate(code_batches, axis=0)
    labels = np.array([f"c{chart}:s{code}" for chart, code in zip(charts, codes)], dtype=str)
    return {"charts": charts, "codes": codes, "labels": labels}


def build_symbol_options(symbol_labels: np.ndarray) -> dict[str, str]:
    """Build MultiChoice options ordered by descending symbol frequency."""
    if len(symbol_labels) == 0:
        return {}

    counts = Counter(np.asarray(symbol_labels).astype(str).tolist())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {f"{label} (n={count})": label for label, count in ordered}


def _camera_example_pane(
    provider: VLAImageProvider | None,
    ep_id: int,
    timestep: int,
    camera: str,
    *,
    width: int = 130,
) -> pn.Column:
    """Build one camera pane for an example card."""
    pane = pn.pane.Markdown("*Unavailable*", width=width)
    if provider is not None:
        img = provider.get_image(ep_id, timestep, camera=camera)
        if img is not None:
            pane = _numpy_to_png_pane(img, width=width)
    return pn.Column(
        pn.pane.Markdown(f"**{camera}**"),
        pane,
        width=width + 10,
    )


def _symbol_example_card(
    provider: VLAImageProvider | None,
    *,
    ep_id: int,
    timestep: int,
    task: int,
    split: str,
    symbol_label: str,
    width: int = 300,
) -> pn.Column:
    """Build a card with both camera views for one symbol example."""
    return pn.Column(
        pn.Row(
            _camera_example_pane(provider, ep_id, timestep, "top"),
            _camera_example_pane(provider, ep_id, timestep, "wrist"),
            sizing_mode="fixed",
        ),
        pn.pane.Markdown(
            f"`{symbol_label}` | split={split} | ep={ep_id} | t={timestep} | task={task}",
            width=width,
        ),
        width=width,
        margin=(0, 8, 12, 0),
    )


def _latent_bar_chart(z_vec: np.ndarray, width: int = 300, height: int = 100):
    """Horizontal bar chart of latent vector components with signed coloring."""
    z = _to_numpy(z_vec).ravel()
    sign = np.where(z >= 0, "pos", "neg")
    data = {
        "dim": [f"z{i}" for i in range(len(z))],
        "value": z.astype(float),
        "sign": sign,
    }
    return hv.Bars(data, "dim", ["value", "sign"]).opts(
        color="sign",
        cmap={"pos": "#4c78a8", "neg": "#e45756"},
        width=width,
        height=height,
        xrotation=90,
        xaxis="bare",
        title="Latent vector",
    )


def _feature_recon_bar(
    orig: np.ndarray, recon: np.ndarray, top_k: int = 20, width: int = 300, height: int = 100
):
    """Bar chart of top-K largest reconstruction error dimensions."""
    diff = np.abs(orig - recon)
    top_idx = np.argsort(diff)[-top_k:][::-1]
    data = [(f"d{i}", float(diff[i])) for i in top_idx]
    return hv.Bars(data, "dim", "|error|").opts(
        color="#f58518",
        width=width,
        height=height,
        xrotation=90,
        xaxis="bare",
        title=f"Top-{top_k} recon error dims",
    )


# ---------------------------------------------------------------------------
# Dashboard app
# ---------------------------------------------------------------------------


def create_app(outputs_dir: str = "outputs/vla") -> pn.template.FastListTemplate:
    """Create the interactive VLA World Model dashboard."""
    pn.extension("plotly", "tabulator")

    # ---- Shared state ----
    app_state: dict = {
        "checkpoints": [],
        "loaded": None,
        "cache": None,  # feature cache dict
        "image_provider": None,
        "symbol_index": None,
    }

    # ---- Sidebar widgets ----
    scan_btn = pn.widgets.Button(name="Scan runs", button_type="primary", width=300)
    ckpt_selector = pn.widgets.Select(name="Checkpoint", options=[], width=300)
    load_btn = pn.widgets.Button(name="Load checkpoint", button_type="success", width=300)
    n_samples = pn.widgets.IntSlider(
        name="Recon samples",
        start=4,
        end=24,
        value=8,
        step=4,
        width=300,
    )
    latent_samples = pn.widgets.IntSlider(
        name="Latent samples",
        start=100,
        end=50000,
        value=2000,
        step=100,
        width=300,
    )
    seed_input = pn.widgets.IntInput(name="Random seed", value=42, step=1, width=300)
    color_by = pn.widgets.RadioButtonGroup(
        name="Color by",
        options=["timestep", "chart", "episode", "radius"],
        value="timestep",
        button_type="default",
    )
    split_overlay_mode = pn.widgets.RadioButtonGroup(
        name="Split overlay",
        options={"Off": "off", "Shape": "shape", "Shape+Color": "shape_color"},
        value="shape_color",
        button_type="default",
    )
    point_size = pn.widgets.IntSlider(name="Point size", start=1, end=10, value=3, width=300)
    show_latents = pn.widgets.Checkbox(name="Show latent points", value=True, width=300)
    show_chart_centers = pn.widgets.Checkbox(name="Show chart centers", value=False, width=300)
    show_code_centers = pn.widgets.Checkbox(name="Show code centers", value=False, width=300)
    show_tree_lines = pn.widgets.Checkbox(name="Show tree lines", value=False, width=300)
    tree_line_color = pn.widgets.Select(
        name="Line color",
        options=["black", "chart", "symbol"],
        value="black",
        width=300,
    )
    tree_line_width = pn.widgets.EditableFloatSlider(
        name="Line width",
        start=0.1,
        end=5.0,
        value=0.5,
        step=0.1,
        width=300,
    )
    camera_selector = pn.widgets.Select(
        name="Camera",
        options=["top", "wrist"],
        value="top",
        width=300,
    )
    alignment_mode = pn.widgets.RadioButtonGroup(
        name="Alignment mode",
        options=["episode", "timestep"],
        value="episode",
        button_type="default",
    )
    dynamics_granularity = pn.widgets.RadioButtonGroup(
        name="Granularity",
        options=["chart", "symbol"],
        value="chart",
        button_type="default",
    )
    trajectory_episode = pn.widgets.Select(
        name="Trajectory episode",
        options={"None": -1},
        value=-1,
        width=300,
    )
    trajectory_color = pn.widgets.RadioButtonGroup(
        name="Trajectory color",
        options=["red", "viridis"],
        value="red",
        button_type="default",
    )
    symbol_selector = pn.widgets.MultiChoice(
        name="Symbols",
        options={},
        value=[],
        width=500,
    )
    symbol_examples_limit = pn.widgets.IntSlider(
        name="Examples / symbol",
        start=1,
        end=10,
        value=6,
        width=250,
    )
    symbol_split_filter = pn.widgets.RadioButtonGroup(
        name="Split",
        options=["all", "train", "test"],
        value="all",
        button_type="default",
    )
    status = pn.pane.Markdown("Click **Scan runs** to begin.", width=300)

    sidebar = pn.Column(
        pn.pane.Markdown("## VLA Dashboard"),
        scan_btn,
        ckpt_selector,
        load_btn,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        n_samples,
        latent_samples,
        seed_input,
        color_by,
        split_overlay_mode,
        point_size,
        show_latents,
        pn.layout.Divider(),
        pn.pane.Markdown("### Trajectory"),
        trajectory_episode,
        trajectory_color,
        pn.layout.Divider(),
        pn.pane.Markdown("### Hierarchy"),
        show_chart_centers,
        show_code_centers,
        show_tree_lines,
        tree_line_color,
        tree_line_width,
        pn.layout.Divider(),
        pn.pane.Markdown("### Images"),
        camera_selector,
        pn.layout.Divider(),
        pn.pane.Markdown("### Dynamics"),
        alignment_mode,
        dynamics_granularity,
        pn.layout.Divider(),
        status,
        width=350,
    )

    # ---- Main panes (placeholders) ----
    # Tab 1: Latent Space
    latent_3d_pane = pn.pane.Plotly(None, sizing_mode="stretch_width", height=600)
    latent_2d_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    latent_split_summary = pn.pane.Markdown("")
    usage_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    code_time_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width")
    inspect_label = pn.pane.Markdown("*Click a point in z0 vs z1 to inspect*")
    inspect_image = pn.Column(
        pn.pane.Markdown("*(click a point)*"),
        width=200,
        height=200,
    )
    inspect_latent_bar = pn.pane.HoloViews(hv.Div(""), width=350, height=150)
    inspect_meta = pn.pane.Markdown("")
    inspect_row = pn.Row(
        pn.Column("**Camera image**", inspect_image),
        pn.Column("**Latent vector**", inspect_latent_bar),
        pn.Column("**Metadata**", inspect_meta),
    )
    tap_stream = hv.streams.Tap(x=0, y=0)

    # Tab 2: Reconstructions
    recon_pane = pn.Column(pn.pane.Markdown("*Load a checkpoint to see reconstructions.*"))
    recon_summary = pn.pane.Markdown("")

    # Tab 3: Dynamics - use dedicated HoloViews panes
    dyn_transition_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width", height=450)
    dyn_alignment_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width", height=450)
    dyn_trajectory_pane = pn.pane.HoloViews(hv.Div(""), sizing_mode="stretch_width", height=600)
    dyn_status = pn.pane.Markdown("*Load a checkpoint with a world model to see dynamics.*")

    # Tab 4: Symbol examples
    symbol_examples_summary = pn.pane.Markdown("*Load a checkpoint to browse symbol examples.*")
    symbol_examples_pane = pn.Column(
        pn.pane.Markdown("*No symbol examples yet.*"),
        sizing_mode="stretch_width",
    )

    # ---- Tabs ----
    latent_tab = pn.Column(
        latent_split_summary,
        latent_3d_pane,
        latent_2d_pane,
        inspect_label,
        inspect_row,
        usage_pane,
        code_time_pane,
        sizing_mode="stretch_width",
    )
    recon_tab = pn.Column(recon_pane, recon_summary, sizing_mode="stretch_width")
    dynamics_tab = pn.Column(
        dyn_status,
        dyn_transition_pane,
        dyn_alignment_pane,
        dyn_trajectory_pane,
        sizing_mode="stretch_width",
    )
    symbol_examples_tab = pn.Column(
        pn.Row(symbol_selector, symbol_examples_limit, symbol_split_filter),
        symbol_examples_summary,
        symbol_examples_pane,
        sizing_mode="stretch_width",
    )

    tabs = pn.Tabs(
        ("Latent Space", latent_tab),
        ("Reconstructions", recon_tab),
        ("Dynamics", dynamics_tab),
        ("Symbol Examples", symbol_examples_tab),
        sizing_mode="stretch_both",
    )

    # ---- Callbacks ----
    def _on_scan(_event=None):
        ckpts = scan_vla_runs(outputs_dir)
        app_state["checkpoints"] = ckpts
        if not ckpts:
            ckpt_selector.options = []
            status.object = "No checkpoints found."
            return
        ckpt_selector.options = {c.label: c.label for c in ckpts}
        ckpt_selector.value = ckpts[-1].label
        status.object = f"Found **{len(ckpts)}** checkpoint(s)."

    def _on_load(_event=None):
        label = ckpt_selector.value
        info = next((c for c in app_state["checkpoints"] if c.label == label), None)
        if info is None:
            status.object = "**Error:** Select a checkpoint first."
            return
        status.object = f"Loading {info.label}..."
        try:
            loaded = load_vla_checkpoint(info.path)
        except Exception as exc:
            status.object = f"**Error loading:** {exc}"
            traceback.print_exc()
            return
        app_state["loaded"] = loaded
        app_state["symbol_index"] = None

        # Find feature cache dir
        feature_dir = _find_feature_dir(info.path, outputs_dir, loaded.args)
        if feature_dir is not None:
            try:
                cache = load_feature_cache(feature_dir)
                app_state["cache"] = cache
            except Exception as exc:
                status.object = f"Loaded model but feature cache error: {exc}"
                app_state["cache"] = None
        else:
            app_state["cache"] = None
            status.object = "Loaded model but no feature cache found."

        # Image provider — try checkpoint args first, then feature cache meta.json
        dataset_name = loaded.args.get("dataset_name", loaded.args.get("dataset", ""))
        if not dataset_name and feature_dir:
            meta_path = os.path.join(feature_dir, "meta.json")
            if pathlib.Path(meta_path).is_file():
                try:
                    meta = json.loads(open(meta_path, encoding="utf-8").read())
                    dataset_name = meta.get("dataset", "")
                except Exception:
                    pass
        if dataset_name and feature_dir:
            app_state["image_provider"] = VLAImageProvider(dataset_name, feature_dir)
        else:
            app_state["image_provider"] = None

        _refresh_all()
        wm_str = "with world model" if loaded.world_model is not None else "encoder only"
        n_feat = app_state["cache"]["features"].shape[0] if app_state["cache"] else 0
        split_labels = (
            app_state["cache"]["split_labels"] if app_state["cache"] else np.zeros(0, dtype=str)
        )
        n_train = int(np.sum(split_labels == "train"))
        n_test = int(np.sum(split_labels == "test"))
        status.object = (
            f"Loaded P{loaded.phase} E{loaded.epoch} ({wm_str}). "
            f"{n_feat} feature frames ({n_train} train / {n_test} test)."
        )

    def _find_feature_dir(ckpt_path: str, outputs_dir: str, args: dict) -> str | None:
        """Try to locate the feature cache directory."""
        # 1. From args
        for key in ("feature_cache_dir", "feature-cache-dir"):
            if key in args:
                candidate = args[key]
                if os.path.isdir(candidate):
                    return candidate

        # 2. Sibling features/ directory
        ckpt_dir = os.path.dirname(ckpt_path)
        for d in (ckpt_dir, os.path.dirname(ckpt_dir), outputs_dir):
            candidate = os.path.join(d, "features")
            if os.path.isdir(candidate):
                return candidate
        return None

    def _refresh_all():
        _refresh_latent()
        _refresh_recon()
        _refresh_dynamics()
        _refresh_symbol_examples()

    def _ensure_symbol_index() -> dict[str, np.ndarray] | None:
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or cache is None:
            return None

        symbol_index = app_state.get("symbol_index")
        if symbol_index is None:
            symbol_index = infer_symbol_assignments(loaded, cache["features"])
            app_state["symbol_index"] = symbol_index

            options = build_symbol_options(symbol_index["labels"])
            symbol_selector.options = options
            valid_values = set(options.values())
            retained = [value for value in symbol_selector.value if value in valid_values]
            if retained:
                symbol_selector.value = retained
            elif options:
                symbol_selector.value = [next(iter(options.values()))]
            else:
                symbol_selector.value = []

        return symbol_index

    def _refresh_latent():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or cache is None:
            latent_split_summary.object = ""
            return

        features = cache["features"]
        task_labels = cache["task_labels"]
        episode_ids = cache["episode_ids"]
        split_labels = cache["split_labels"]

        N = features.shape[0]
        n_lat = min(latent_samples.value, N)
        rng = np.random.RandomState(seed_input.value)
        idx = rng.choice(N, size=n_lat, replace=False) if n_lat < N else np.arange(N)

        x_sub = features[idx]
        task_sub = task_labels[idx]
        ep_sub = episode_ids[idx]
        split_sub = split_labels[idx]

        # Forward pass
        with torch.no_grad():
            enc_out = loaded.encoder.encoder(x_sub)
            K_code = enc_out[1]  # index 1 is K_code from PrimitiveAttentiveAtlasEncoder
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
            ) = loaded.encoder(x_sub)

        K_code_np = _to_numpy(K_code).astype(int)

        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        radii = np.linalg.norm(z_np, axis=1)

        # Map color_by -> labels for the scatter functions
        timesteps = cache["timesteps"]
        ts_sub = timesteps[idx]
        cb = color_by.value
        if cb == "timestep":
            labels_for_color = ts_sub
        elif cb == "chart":
            labels_for_color = K_np
        elif cb == "episode":
            labels_for_color = ep_sub
        elif cb == "radius":
            # Bin radii into 10 bins for discrete coloring
            labels_for_color = np.digitize(radii, np.linspace(0, radii.max() + 1e-8, 11)) - 1
        else:
            labels_for_color = ts_sub

        dummy_correct = np.ones(len(labels_for_color), dtype=int)

        # Use "label" color mode since we've already set labels_for_color
        scatter_color = "label"

        # Populate trajectory episode selector from available episodes
        unique_eps = sorted(np.unique(ep_sub))
        ep_options = {"None": -1}
        ep_options.update({str(e): int(e) for e in unique_eps})
        trajectory_episode.options = ep_options
        if trajectory_episode.value not in ep_options.values():
            trajectory_episode.value = -1

        # Trajectory params
        traj_ep = trajectory_episode.value if trajectory_episode.value != -1 else None
        traj_color = trajectory_color.value
        split_markers = {"train": "circle", "test": "diamond", "unknown": "square"}
        split_overlay = split_overlay_mode.value
        if split_overlay == "shape":
            split_colors = {"train": "#1f1f1f", "test": "#1f1f1f", "unknown": "#1f1f1f"}
            split_marker_groups = split_sub
        elif split_overlay == "shape_color":
            split_colors = {"train": "#1f1f1f", "test": "#d62728", "unknown": "#7f7f7f"}
            split_marker_groups = split_sub
        else:
            split_colors = {"train": "#1f1f1f", "test": "#d62728", "unknown": "#7f7f7f"}
            split_marker_groups = None

        meta = cache["meta"]
        total_train = int(np.sum(split_labels == "train"))
        total_test = int(np.sum(split_labels == "test"))
        latent_split_summary.object = (
            f"**Split overlay:** {split_overlay}. "
            f"Train = circle, test = diamond when enabled.  "
            f"**Frames:** {total_train} train / {total_test} test.  "
            f"**Episodes:** {meta.get('num_train_episodes', 0)} train / "
            f"{meta.get('num_test_episodes', 0)} test."
        )

        # 3D scatter
        try:
            fig3d = plot_latent_3d(
                z_np,
                labels_for_color,
                K_chart=K_np,
                correct=dummy_correct,
                color_by=scatter_color,
                point_size=point_size.value,
                show_points=show_latents.value,
                show_chart_centers=show_chart_centers.value,
                show_code_centers=show_code_centers.value,
                show_tree_lines=show_tree_lines.value,
                tree_line_color=tree_line_color.value,
                tree_line_width=tree_line_width.value,
                K_code=K_code_np,
                show_leaf_lines=False,
                marker_groups=split_marker_groups,
                marker_markers=split_markers,
                marker_outline_colors=split_colors,
                trajectory_episode_ids=ep_sub,
                trajectory_timesteps=ts_sub,
                trajectory_episode=traj_ep,
                trajectory_color=traj_color,
            )
            latent_3d_pane.object = fig3d
        except Exception:
            logger.warning("3D scatter error: %s", traceback.format_exc())

        # 2D slices — build manually so we can wire the Tap stream
        try:
            dim = z_np.shape[1]
            pairs = []
            if dim >= 2:
                pairs.append((0, 1))
            if dim >= 3:
                pairs.extend([(0, 2), (1, 2)])

            scatter_panels = []
            for di, dj in pairs:
                scatter = build_latent_scatter(
                    z_np,
                    labels_for_color,
                    K_np,
                    dummy_correct,
                    scatter_color,
                    point_size.value,
                    di,
                    dj,
                    indices=idx,
                    K_code=K_code_np,
                    show_code_centers=show_code_centers.value,
                    show_points=show_latents.value,
                    marker_groups=split_marker_groups,
                    marker_markers=split_markers,
                    marker_outline_colors=split_colors,
                    trajectory_episode_ids=ep_sub,
                    trajectory_timesteps=ts_sub,
                    trajectory_episode=traj_ep,
                    trajectory_color=traj_color,
                )
                scatter_panels.append(scatter)

            if scatter_panels:
                # Wire tap stream to the first scatter (z0 vs z1)
                tap_stream.source = scatter_panels[0]
                layout_2d = (
                    hv
                    .Layout(scatter_panels)
                    .opts(
                        shared_axes=False,
                    )
                    .cols(min(3, len(scatter_panels)))
                )
                latent_2d_pane.object = layout_2d
        except Exception:
            logger.warning("2D scatter error: %s", traceback.format_exc())

        # Chart usage
        try:
            n_charts = int(K_np.max()) + 1
            usage = np.zeros(n_charts)
            for c in K_np:
                usage[c] += 1
            usage /= usage.sum() + 1e-8
            usage_pane.object = plot_chart_usage(usage)
        except Exception:
            logger.warning("Chart usage error: %s", traceback.format_exc())

        # Code-timestep boxplot
        try:
            timesteps = cache["timesteps"]
            ts_sub_all = timesteps[idx]
            # Build (code_label, timestep) pairs for each point
            code_labels = [f"c{K_np[i]}:{K_code_np[i]}" for i in range(len(K_np))]
            box_data = {
                "code": code_labels,
                "timestep": ts_sub_all.astype(float),
            }
            boxwhisker = hv.BoxWhisker(
                box_data,
                kdims=["code"],
                vdims=["timestep"],
            ).opts(
                width=max(400, 30 * len(set(code_labels))),
                height=300,
                xrotation=90,
                title="Timestep distribution per code",
                ylabel="timestep",
                box_fill_color="#4c78a8",
                box_fill_alpha=0.5,
            )
            code_time_pane.object = boxwhisker
        except Exception:
            logger.warning("Code-timestep boxplot error: %s", traceback.format_exc())

        # Store sub-arrays for click inspect
        app_state["latent_sub"] = {
            "z_np": z_np,
            "K_np": K_np,
            "K_code_np": K_code_np,
            "task_sub": task_sub,
            "ep_sub": ep_sub,
            "split_sub": split_sub,
            "idx": idx,
            "x_sub": x_sub,
            "x_recon": _to_numpy(x_recon),
            "radii": radii,
        }

    def _on_tap(x, y):
        """Handle click on 2D scatter to inspect nearest point."""
        sub = app_state.get("latent_sub")
        if sub is None:
            return
        z_np = sub["z_np"]
        if z_np.shape[1] < 2:
            return

        # Find nearest point in z0-z1 display space
        dists = (z_np[:, 0] - x) ** 2 + (z_np[:, 1] - y) ** 2
        nearest = int(np.argmin(dists))

        z_vec = z_np[nearest]
        chart = int(sub["K_np"][nearest])
        task = int(sub["task_sub"][nearest])
        ep = int(sub["ep_sub"][nearest])
        split = str(sub["split_sub"][nearest])
        global_idx = int(sub["idx"][nearest])
        cache = app_state["cache"]
        ts = int(cache["timesteps"][global_idx]) if cache is not None else 0
        radius = float(sub["radii"][nearest])

        # Camera image
        provider = app_state.get("image_provider")
        inspect_image.clear()
        if provider is not None:
            img = provider.get_image(ep, ts, camera=camera_selector.value)
            if img is not None:
                inspect_image.append(_numpy_to_png_pane(img, width=180))
            else:
                inspect_image.append(pn.pane.Markdown("*Image unavailable*"))
        else:
            inspect_image.append(pn.pane.Markdown("*No image provider*"))

        # Latent bar chart
        try:
            inspect_latent_bar.object = _latent_bar_chart(z_vec, width=300, height=130)
        except Exception:
            pass

        # Metadata
        inspect_meta.object = (
            f"**Task:** {task}  \n"
            f"**Split:** {split}  \n"
            f"**Episode:** {ep}  \n"
            f"**Timestep:** {ts}  \n"
            f"**Chart:** {chart}  \n"
            f"**||z||:** {radius:.4f}  \n"
            f"**Index:** {global_idx}"
        )

    tap_stream.param.watch(lambda event: _on_tap(event.new, tap_stream.y), "x")

    def _refresh_recon():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or cache is None:
            recon_pane.clear()
            recon_pane.append(pn.pane.Markdown("*No data loaded.*"))
            return

        features = cache["features"]
        task_labels = cache["task_labels"]
        episode_ids = cache["episode_ids"]
        timesteps = cache["timesteps"]
        split_labels = cache["split_labels"]

        N = features.shape[0]
        n_rec = min(n_samples.value, N)
        rng = np.random.RandomState(seed_input.value + 1)
        idx = rng.choice(N, size=n_rec, replace=False) if n_rec < N else np.arange(N)

        x_sub = features[idx]
        with torch.no_grad():
            enc_out = loaded.encoder.encoder(x_sub)
            K_code = enc_out[1]  # index 1 is K_code from PrimitiveAttentiveAtlasEncoder
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
            ) = loaded.encoder(x_sub)

        _to_numpy(K_code).astype(int)
        x_np = _to_numpy(x_sub)
        xr_np = _to_numpy(x_recon)
        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        radii = np.linalg.norm(z_np, axis=1)

        provider = app_state.get("image_provider")

        rows = []
        mse_list = []
        for i in range(n_rec):
            gi = int(idx[i])
            ep = int(episode_ids[gi])
            ts = int(timesteps[gi])
            task = int(task_labels[gi])
            split = str(split_labels[gi])
            chart = int(K_np[i])
            r = float(radii[i])
            mse = float(((x_np[i] - xr_np[i]) ** 2).mean())
            mse_list.append(mse)

            # Image column
            img_pane = pn.pane.Markdown("*N/A*")
            if provider is not None:
                img = provider.get_image(ep, ts, camera=camera_selector.value)
                if img is not None:
                    img_pane = _numpy_to_png_pane(img, width=120)

            # Latent bar
            latent_bar = pn.pane.HoloViews(
                _latent_bar_chart(z_np[i], width=250, height=80),
                width=270,
                height=100,
            )

            # Recon error bar
            recon_bar = pn.pane.HoloViews(
                _feature_recon_bar(x_np[i], xr_np[i], top_k=15, width=250, height=80),
                width=270,
                height=100,
            )

            # Metadata
            meta = pn.pane.Markdown(
                f"Task: {task} | Split: {split} | Ep: {ep} | t: {ts}  \n"
                f"Chart: {chart} | ||z||: {r:.3f} | MSE: {mse:.4f}"
            )

            row = pn.Row(img_pane, latent_bar, recon_bar, meta)
            rows.append(row)

        # Summary
        mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
        ss_res = ((x_np - xr_np) ** 2).sum()
        ss_tot = ((x_np - x_np.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        n_active = len(np.unique(K_np))
        mean_r = float(radii.mean())

        recon_pane.clear()
        recon_pane.extend(rows)
        recon_summary.object = (
            f"**Summary:** Mean MSE = {mean_mse:.5f} | R² = {r2:.4f} | "
            f"Active charts = {n_active} | Mean ||z|| = {mean_r:.4f}"
        )

    def _refresh_dynamics():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        if loaded is None or loaded.world_model is None:
            dyn_status.object = "*No world model in this checkpoint.*"
            dyn_transition_pane.object = hv.Div("")
            dyn_alignment_pane.object = hv.Div("")
            dyn_trajectory_pane.object = hv.Div("")
            return

        from fragile.vla.visualize import (
            hv_chart_alignment,
            hv_chart_transitions,
            hv_dynamics_trajectory,
        )

        features = cache["features"]
        episode_ids = cache["episode_ids"]
        timesteps_arr = cache["timesteps"]

        N = features.shape[0]
        n_sub = min(5000, N)
        rng = np.random.RandomState(seed_input.value + 2)
        idx = rng.choice(N, size=n_sub, replace=False) if n_sub < N else np.arange(N)

        with torch.no_grad():
            enc_out = loaded.encoder.encoder(features[idx])
            K_code = enc_out[1]
            (
                _x_recon,
                _vq_loss,
                _enc_rw,
                _dec_rw,
                K_chart,
                z_geo,
                _z_n,
                _c_bar,
                _aux,
            ) = loaded.encoder(features[idx])

        z_np = _to_numpy(z_geo)
        K_np = _to_numpy(K_chart).astype(int)
        K_code_np = _to_numpy(K_code).astype(int)
        ep_sub = episode_ids[idx]
        ts_sub = timesteps_arr[idx]

        # Build labels based on granularity
        if dynamics_granularity.value == "symbol":
            dyn_labels = np.array([f"c{k}:s{c}" for k, c in zip(K_np, K_code_np)])
            label_name = "Symbol"
        else:
            dyn_labels = K_np
            label_name = "Chart"

        # Chart/symbol transitions
        try:
            dyn_transition_pane.object = hv_chart_transitions(
                dyn_labels,
                ep_sub,
                title=f"{label_name} Transition Matrix",
                label_name=label_name,
            )
        except Exception:
            logger.warning("Chart transitions error: %s", traceback.format_exc())

        # Chart/symbol alignment (episode or timestep)
        try:
            mode = alignment_mode.value
            if mode == "episode":
                group_labels = ep_sub
                group_name = "Episode"
            else:  # timestep
                n_bins = min(20, len(np.unique(ts_sub)))
                bins = np.linspace(ts_sub.min(), ts_sub.max() + 1, n_bins + 1)
                group_labels = np.digitize(ts_sub, bins) - 1
                group_name = "Timestep bin"
            dyn_alignment_pane.object = hv_chart_alignment(
                dyn_labels,
                group_labels,
                title=f"{label_name}-{group_name} Alignment",
                label_name=label_name,
                group_name=group_name,
            )
        except Exception:
            logger.warning("Chart alignment error: %s", traceback.format_exc())

        # Dynamics trajectory
        try:
            num_charts = loaded.args.get("num_charts", 16)
            action_dim = loaded.args.get("action_dim", 6)
            unique_eps = np.unique(ep_sub)
            for ep_id in unique_eps[:3]:
                mask = ep_sub == ep_id
                if mask.sum() < 4:
                    continue
                z_ep = z_np[mask]
                K_ep = K_np[mask]
                # World model expects: z_0 [B,D], actions [B,H,A], router_weights [B,K]
                z_0 = torch.from_numpy(z_ep[:-1]).float()
                B = z_0.shape[0]
                actions = torch.zeros(B, 1, action_dim)  # H=1 one-step prediction
                rw_0 = torch.zeros(B, num_charts)
                for i in range(B):
                    rw_0[i, K_ep[i]] = 1.0  # one-hot from chart assignment
                with torch.no_grad():
                    wm_out = loaded.world_model(z_0, actions, rw_0)
                z_pred = _to_numpy(wm_out["z_trajectory"][:, 0, :])  # [B, D]
                z_target = z_ep[1:]

                dyn_trajectory_pane.object = hv_dynamics_trajectory(
                    z_pred,
                    z_target,
                    title=f"Dynamics Trajectory: Episode {ep_id}",
                )
                break
            else:
                dyn_trajectory_pane.object = hv.Div("*No episode with enough steps.*")
        except Exception:
            logger.warning("Dynamics trajectory error: %s", traceback.format_exc())

        dyn_status.object = (
            f"**Dynamics** — {n_sub} samples, "
            f"granularity={dynamics_granularity.value}, "
            f"alignment by {alignment_mode.value}"
        )

    def _refresh_symbol_examples():
        loaded = app_state.get("loaded")
        cache = app_state.get("cache")
        provider = app_state.get("image_provider")
        if loaded is None or cache is None:
            symbol_examples_summary.object = "*Load a checkpoint to browse symbol examples.*"
            symbol_examples_pane.clear()
            symbol_examples_pane.append(pn.pane.Markdown("*No symbol examples yet.*"))
            return

        symbol_index = _ensure_symbol_index()
        if symbol_index is None or len(symbol_index["labels"]) == 0:
            symbol_examples_summary.object = "*No symbol assignments available.*"
            symbol_examples_pane.clear()
            symbol_examples_pane.append(pn.pane.Markdown("*No symbol assignments found.*"))
            return

        selected_symbols = list(symbol_selector.value or [])
        if not selected_symbols and symbol_selector.options:
            selected_symbols = [next(iter(symbol_selector.options.values()))]
            symbol_selector.value = selected_symbols

        if not selected_symbols:
            symbol_examples_summary.object = "*Select one or more symbols to inspect examples.*"
            symbol_examples_pane.clear()
            symbol_examples_pane.append(
                pn.pane.Markdown("*Select one or more symbols to inspect examples.*"),
            )
            return

        split_filter = symbol_split_filter.value
        split_labels = cache["split_labels"]
        episode_ids = cache["episode_ids"]
        timesteps = cache["timesteps"]
        task_labels = cache["task_labels"]
        limit = symbol_examples_limit.value

        symbol_examples_summary.object = (
            f"**Selected symbols:** {len(selected_symbols)} | "
            f"**Examples / symbol:** {limit} | "
            f"**Split filter:** {split_filter}"
        )

        blocks: list[object] = []
        for symbol_label in selected_symbols:
            total_mask = symbol_index["labels"] == symbol_label
            filtered_mask = total_mask.copy()
            if split_filter != "all":
                filtered_mask &= split_labels == split_filter
            match_idx = np.flatnonzero(filtered_mask)

            if match_idx.size == 0:
                blocks.append(
                    pn.Column(
                        pn.pane.Markdown(f"### `{symbol_label}`"),
                        pn.pane.Markdown("*No matching examples for this split.*"),
                    ),
                )
                continue

            total_train = int(np.sum(total_mask & (split_labels == "train")))
            total_test = int(np.sum(total_mask & (split_labels == "test")))
            rng = np.random.RandomState(seed_input.value + _symbol_seed(symbol_label))
            if match_idx.size > limit:
                chosen = np.sort(rng.choice(match_idx, size=limit, replace=False))
            else:
                chosen = match_idx

            cards = []
            for global_idx in chosen:
                cards.append(
                    _symbol_example_card(
                        provider,
                        ep_id=int(episode_ids[global_idx]),
                        timestep=int(timesteps[global_idx]),
                        task=int(task_labels[global_idx]),
                        split=str(split_labels[global_idx]),
                        symbol_label=symbol_label,
                    ),
                )

            blocks.append(
                pn.Column(
                    pn.pane.Markdown(
                        f"### `{symbol_label}`\n"
                        f"{match_idx.size} filtered matches | "
                        f"train={total_train}, test={total_test}",
                    ),
                    pn.GridBox(*cards, ncols=3, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width",
                ),
            )

        symbol_examples_pane.clear()
        symbol_examples_pane.extend(blocks)

    # ---- Wire callbacks ----
    scan_btn.on_click(_on_scan)
    load_btn.on_click(_on_load)

    # Refresh on widget changes
    for w in (
        color_by,
        split_overlay_mode,
        point_size,
        latent_samples,
        seed_input,
        show_latents,
        show_chart_centers,
        show_code_centers,
        show_tree_lines,
        tree_line_color,
        tree_line_width,
        trajectory_episode,
        trajectory_color,
    ):
        w.param.watch(lambda _: _refresh_latent(), "value")

    n_samples.param.watch(lambda _: _refresh_recon(), "value")
    alignment_mode.param.watch(lambda _: _refresh_dynamics(), "value")
    dynamics_granularity.param.watch(lambda _: _refresh_dynamics(), "value")
    for w in (symbol_selector, symbol_examples_limit, symbol_split_filter, seed_input):
        w.param.watch(lambda _: _refresh_symbol_examples(), "value")

    # ---- Template ----
    return pn.template.FastListTemplate(
        title="VLA World Model Dashboard",
        sidebar=[sidebar],
        main=[tabs],
    )
