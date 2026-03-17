"""Extract and cache VLA latents (SmolVLA hidden states) for world model training.

Runs the frozen SmolVLA backbone over every frame in a LeRobot dataset,
mean-pools the final-layer hidden states, and saves per-episode ``.pt``
files with features, actions, states, and observation-to-frame indices so
the encoder + world model can train without touching the VLA again.

Usage::

    uv run fragile dataset
    uv run fragile dataset --dataset lerobot/svla_so100_pickplace
    uv run fragile dataset --output-dir outputs/vla/features --max-episodes 10
"""

from __future__ import annotations

import argparse

from fragile.vla.config import VLAConfig
from fragile.vla.extract_features import extract_smolvla_features


def main() -> None:
    """CLI entry point for VLA latent dataset creation."""
    p = argparse.ArgumentParser(
        description=(
            "Extract SmolVLA latents (frozen backbone hidden states) from a "
            "LeRobot dataset and cache them as per-episode .pt files."
        ),
    )
    p.add_argument(
        "--model-id",
        default=VLAConfig.smolvla_model_id,
        help=f"HuggingFace model ID for SmolVLA (default: {VLAConfig.smolvla_model_id}).",
    )
    p.add_argument(
        "--dataset",
        default=VLAConfig.dataset_name,
        help=f"LeRobot dataset name (default: {VLAConfig.dataset_name}).",
    )
    p.add_argument(
        "--output-dir",
        default=VLAConfig.feature_cache_dir,
        help=f"Where to save the cached features (default: {VLAConfig.feature_cache_dir}).",
    )
    p.add_argument(
        "--pooling",
        default=VLAConfig.pooling,
        choices=["mean", "modality_aware"],
        help=f"Token pooling strategy (default: {VLAConfig.pooling}).",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Max episodes to extract (0 = all).",
    )
    p.add_argument(
        "--test-episodes",
        type=int,
        default=VLAConfig.held_out_test_episodes,
        help=(
            "How many cached episodes to reserve as held-out test data "
            f"(default: {VLAConfig.held_out_test_episodes})."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Device: "auto", "cuda", or "cpu".',
    )

    args = p.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = VLAConfig(
        smolvla_model_id=args.model_id,
        dataset_name=args.dataset,
        feature_cache_dir=args.output_dir,
        pooling=args.pooling,
        max_episodes=args.max_episodes,
        held_out_test_episodes=args.test_episodes,
        device=device,
    )

    extract_smolvla_features(config)


if __name__ == "__main__":
    main()
