"""Reusable utilities for aggregating, formatting, and displaying metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from fragile.rl.macro_data import ActionPrototypeTable


# Known metric name suffixes that should always display as integers.
_INTEGER_SUFFIXES = frozenset({
    "episodes",
    "steps",
    "env_steps",
    "update_steps",
    "num_valid",
    "valid_symbols",
})

# Logical section ordering for macro RL metrics.  Sections not listed here
# are appended alphabetically after the last group.
_MACRO_RL_SECTION_ORDER: list[str] = [
    "collect",
    "eval",
    "loss",
    "q",
    "obs",
    "act",
    "markov",
    "model",
    "enclosure",
    "grad",
    "param",
    "replay",
    "routing",
    "transitions",
    "proto",
]

# Group headers inserted *before* the first section in each group.
_MACRO_RL_SECTION_LABELS: dict[str, str] = {
    "collect": "Rollout",
    "loss": "Losses",
    "obs": "Encoders",
    "markov": "Model",
    "grad": "Optimization",
    "replay": "Replay & routing",
}


def _is_integer_metric(name: str, value: float) -> bool:
    """Heuristic: a metric should display as an integer."""
    if value != value:  # NaN
        return False
    for suffix in _INTEGER_SUFFIXES:
        if name == suffix or name.endswith(f"/{suffix}"):
            return True
    # Values that are exact integers and large enough to not be fractions.
    if value == int(value) and abs(value) >= 1.0:
        return abs(value) < 1e15
    return False


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


def format_metric_value(value: float, name: str = "") -> str:
    """Format metric values compactly for CLI logging.

    Integer-valued metrics (counts, steps) are rendered without decimals.
    """
    value = float(value)
    if _is_integer_metric(name, value):
        return f"{int(value):,}"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def print_metric_groups(
    title: str,
    metrics: dict[str, float],
    *,
    section_order: list[str] | None = None,
    section_labels: dict[str, str] | None = None,
) -> None:
    """Print every metric grouped by its prefix.

    Parameters
    ----------
    section_order:
        If given, sections are printed in this order (unlisted sections
        follow alphabetically).  If ``None``, all sections are alphabetical.
    section_labels:
        Mapping from section name to a human-readable group header.  A
        separator line is printed before each group's first section.
    """
    print(f"{title}:")
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for key, value in sorted(metrics.items()):
        prefix, sep, rest = key.partition("/")
        label = rest if sep else key
        grouped[prefix].append((label, key, value))

    if section_order is not None:
        ordered = []
        seen: set[str] = set()
        for section in section_order:
            if section in grouped:
                ordered.append(section)
                seen.add(section)
        for section in sorted(grouped):
            if section not in seen:
                ordered.append(section)
    else:
        ordered = sorted(grouped)

    labels = section_labels or {}
    emitted_labels: set[str] = set()
    for prefix in ordered:
        if prefix in labels and labels[prefix] not in emitted_labels:
            lbl = labels[prefix]
            emitted_labels.add(lbl)
            print(f"  ── {lbl} {'─' * max(1, 40 - len(lbl))}")
        items = grouped[prefix]
        parts = " ".join(
            f"{name}={format_metric_value(value, full_key)}" for name, full_key, value in items
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
        f"{prefix}/action_usage_perplexity": float(
            sum(action_perplexity) / len(action_perplexity)
        ),
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
    mean_count = (
        float(counts[prototypes.valid].mean().item()) if bool(prototypes.valid.any()) else 0.0
    )
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
    probs = ", ".join(f"{round(100.0 * float(value)):02d}" for value in distribution.tolist())
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
        format_metric_value(eval_metrics.get(eval_return_key, 0.0)) if should_eval else "skipped"
    )
    print(
        f"{header} E{epoch:05d} | "
        f"collect={format_metric_value(train_metrics.get(collect_return_key, 0.0))} | "
        f"q={format_metric_value(train_metrics.get(q_loss_key, 0.0))} | "
        f"eval={eval_display} | "
        f"env_steps={env_steps:,} | "
        f"updates={update_steps:,}",
    )

    # Collection return summary before the detailed metrics.
    ret_mean = train_metrics.get("collect/return_mean", 0.0)
    ret_std = train_metrics.get("collect/return_std", 0.0)
    ep_len = train_metrics.get("collect/length_mean", 0.0)
    rand_frac = train_metrics.get("collect/random_action_frac", 0.0)
    print(
        f"  Collect: return={format_metric_value(ret_mean)}"
        f" +/- {format_metric_value(ret_std)}"
        f"  ep_len={format_metric_value(ep_len)}"
        f"  random={format_metric_value(rand_frac)}",
    )

    print_metric_groups(
        "Train metrics",
        train_metrics,
        section_order=_MACRO_RL_SECTION_ORDER,
        section_labels=_MACRO_RL_SECTION_LABELS,
    )

    print(f"  ── Symbol usage {'─' * 23}")
    print_symbol_usage("train", train_symbol_usage)

    if should_eval:
        print_metric_groups(
            "Eval metrics",
            eval_metrics,
            section_order=_MACRO_RL_SECTION_ORDER,
            section_labels=_MACRO_RL_SECTION_LABELS,
        )
        if eval_symbol_usage is not None:
            print_symbol_usage("eval", eval_symbol_usage)
    else:
        print(f"Eval metrics: skipped (runs every {eval_every} epochs)")
    print("─" * 80)
