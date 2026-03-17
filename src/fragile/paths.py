from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_EXPERIMENT_DIR = OUTPUTS_DIR / "3d_topoencoder_mnist_cpu_adapt_lr7_bnch"

__all__ = ["DEFAULT_EXPERIMENT_DIR", "OUTPUTS_DIR", "REPO_ROOT"]
