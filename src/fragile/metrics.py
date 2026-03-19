"""Reusable utilities for aggregating, formatting, and displaying metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from fragile.rl.macro_data import ActionPrototypeTable


def average_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of scalar metric dictionaries."""
    if not metric_list:
        return {}
    averaged: dict[str, float] = {}
    keys = set().union(*(metrics.keys() for metrics in metric_list))
    for key in keys:
        values = [metrics[key] for metrics in metric_list if key in metrics]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged


def format_metric_value(value: float) -> str:
    """Format metric values compactly for CLI logging."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def print_metric_groups(title: str, metrics: dict[str, float]) -> None:
    """Print every metric grouped by its prefix."""
    print(f"{title}:")
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, value in sorted(metrics.items()):
        prefix, sep, rest = key.partition("/")
        label = rest if sep else key
        grouped[prefix].append((label, value))
    for prefix in sorted(grouped):
        parts = " ".join(
            f"{name}={format_metric_value(value)}" for name, value in grouped[prefix]
        )
        print(f"  {prefix}: {parts}")


def summarize_collection(
    infos: list[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Aggregate episode-collection metrics for one epoch."""
    if not infos:
        return {
            f"{prefix}/return_mean": 0.0,
            f"{prefix}/return_std": 0.0,
            f"{prefix}/length_mean": 0.0,
            f"{prefix}/random_action_frac": 0.0,
            f"{prefix}/action_usage_active": 0.0,
            f"{prefix}/action_usage_perplexity": 0.0,
        }
    returns = [float(info["return"]) for info in infos]
    lengths = [float(info["length"]) for info in infos]
    random_fracs = [float(info["random_action_frac"]) for info in infos]
    action_active = [float(info.get("action_usage_active", 0.0)) for info in infos]
    action_perplexity = [float(info.get("action_usage_perplexity", 0.0)) for info in infos]
    return {
        f"{prefix}/return_mean": float(sum(returns) / len(returns)),
        f"{prefix}/return_std": float(torch.tensor(returns).std(unbiased=False).item()),
        f"{prefix}/length_mean": float(sum(lengths) / len(lengths)),
        f"{prefix}/random_action_frac": float(sum(random_fracs) / len(random_fracs)),
        f"{prefix}/action_usage_active": float(sum(action_active) / len(action_active)),
        f"{prefix}/action_usage_perplexity": float(sum(action_perplexity) / len(action_perplexity)),
    }


def prototype_metrics(
    prototypes: ActionPrototypeTable | None,
) -> dict[str, float]:
    """Summarize how many macro action symbols currently have valid prototypes."""
    if prototypes is None:
        return {
            "proto/valid_symbols": 0.0,
            "proto/coverage": 0.0,
            "proto/mean_count": 0.0,
        }
    valid = prototypes.valid.float()
    counts = prototypes.counts.float()
    num_symbols = float(prototypes.valid.numel())
    valid_count = float(valid.sum().item())
    mean_count = float(counts[prototypes.valid].mean().item()) if bool(prototypes.valid.any()) else 0.0
    return {
        "proto/valid_symbols": valid_count,
        "proto/coverage": (valid_count / num_symbols) if num_symbols > 0 else 0.0,
        "proto/mean_count": mean_count,
    }


def init_symbol_usage_accumulator(
    *,
    obs_num_charts: int,
    obs_codes_per_chart: int,
    act_num_charts: int,
    act_codes_per_chart: int,
) -> dict[str, torch.Tensor]:
    """Allocate per-chart code-count tensors for obs and actions."""
    return {
        "obs": torch.zeros(obs_num_charts, obs_codes_per_chart, dtype=torch.long),
        "act": torch.zeros(act_num_charts, act_codes_per_chart, dtype=torch.long),
    }


def update_symbol_usage_from_forward(
    usage: dict[str, torch.Tensor],
    forward: dict[str, Any],
) -> None:
    """Accumulate hard chart/code assignments produced during replay training."""
    for key in ("obs", "act"):
        chart_idx = forward[key]["chart_idx_valid"].reshape(-1).detach().cpu()
        code_idx = forward[key]["code_idx_valid"].reshape(-1).detach().cpu()
        for chart, code in zip(chart_idx.tolist(), code_idx.tolist(), strict=False):
            usage[key][int(chart), int(code)] += 1


def update_symbol_usage_from_episode_info(
    usage: dict[str, torch.Tensor],
    info: dict[str, Any],
    *,
    obs_codes_per_chart: int,
    act_codes_per_chart: int,
) -> None:
    """Accumulate symbol usage from collected rollout traces."""
    for state_idx in info.get("obs_state_indices", []):
        chart = int(state_idx) // int(obs_codes_per_chart)
        code = int(state_idx) % int(obs_codes_per_chart)
        if 0 <= chart < usage["obs"].shape[0]:
            usage["obs"][chart, code] += 1
    for action_idx in info.get("action_indices", []):
        if int(action_idx) < 0:
            continue
        chart = int(action_idx) // int(act_codes_per_chart)
        code = int(action_idx) % int(act_codes_per_chart)
        if 0 <= chart < usage["act"].shape[0]:
            usage["act"][chart, code] += 1


def format_symbol_distribution(
    counts: torch.Tensor,
    *,
    total_symbols: int,
) -> str:
    """Render one chart's usage as ``num_active/num_total [p0, p1, ...]``."""
    counts = counts.detach().cpu().to(dtype=torch.float32)
    total = counts.sum()
    distribution = counts / total.clamp_min(1.0)
    active = int((counts > 0).sum().item())
    probs = ", ".join(f"{int(round(100.0 * float(value))):02d}" for value in distribution.tolist())
    return f"{active}/{int(total_symbols)} [{probs}]"


def print_symbol_usage(
    prefix: str,
    usage: dict[str, torch.Tensor],
) -> None:
    """Print per-chart symbol distributions for observations and actions."""
    print(f"  {prefix} obs symbol dist/chart:")
    for chart_idx, counts in enumerate(usage["obs"]):
        print(
            f"    c{chart_idx:02d} "
            f"{format_symbol_distribution(counts, total_symbols=counts.numel())}",
        )
    print(f"  {prefix} act symbol dist/chart:")
    for chart_idx, counts in enumerate(usage["act"]):
        print(
            f"    c{chart_idx:02d} "
            f"{format_symbol_distribution(counts, total_symbols=counts.numel())}",
        )


def log_epoch(
    *,
    header: str,
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    train_symbol_usage: dict[str, torch.Tensor],
    eval_symbol_usage: dict[str, torch.Tensor] | None,
    env_steps: int,
    update_steps: int,
    should_eval: bool,
    eval_every: int,
    collect_return_key: str = "collect/return_mean",
    q_loss_key: str = "q/loss",
    eval_return_key: str = "eval/return_mean",
) -> None:
    """Print a full epoch log block with metrics and symbol usage."""
    eval_display = (
        format_metric_value(eval_metrics.get(eval_return_key, 0.0))
        if should_eval
        else "skipped"
    )
    print(
        f"{header} E{epoch:05d} | "
        f"collect={format_metric_value(train_metrics.get(collect_return_key, 0.0))} | "
        f"q={format_metric_value(train_metrics.get(q_loss_key, 0.0))} | "
        f"eval={eval_display} | "
        f"env_steps={env_steps} | "
        f"updates={update_steps}",
    )
    print_metric_groups("Train metrics", train_metrics)
    print_symbol_usage("train", train_symbol_usage)
    if should_eval:
        print_metric_groups("Eval metrics", eval_metrics)
        if eval_symbol_usage is not None:
            print_symbol_usage("eval", eval_symbol_usage)
    else:
        print(f"Eval metrics: skipped (runs every {eval_every} epochs)")
    print("-" * 80)
