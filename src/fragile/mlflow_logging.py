"""Optional MLflow integration for training."""

import math


try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def log_mlflow_metrics(
    metrics: dict[str, float],
    step: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    safe_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            safe_metrics[key] = val
    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=step)


def end_mlflow_run(enabled: bool) -> None:
    if enabled and MLFLOW_AVAILABLE:
        mlflow.end_run()


def log_mlflow_params(params: dict[str, object], enabled: bool) -> None:
    """Log additional params to an active MLflow run."""
    if not enabled or not MLFLOW_AVAILABLE:
        return
    mlflow.log_params(params)
