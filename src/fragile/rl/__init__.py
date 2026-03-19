"""RL utilities for Fragile.

The package keeps imports lazy so lightweight helpers and the standalone macro
RL runner can be used without eagerly importing the older Dreamer trainer.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "DreamerConfig",
    "GeometricActionBoundaryDecoder",
    "GeometricActionEncoder",
    "GeometricActor",
    "GeometricCritic",
    "RewardHead",
    "SequenceReplayBuffer",
    "compute_lambda_returns",
    "train",
]


def __getattr__(name: str):
    if name == "DreamerConfig":
        return import_module("fragile.rl.config").DreamerConfig
    if name in {"GeometricActionBoundaryDecoder", "GeometricActionEncoder"}:
        module = import_module("fragile.rl.boundary")
        return getattr(module, name)
    if name == "GeometricActor":
        return import_module("fragile.rl.actor").GeometricActor
    if name == "GeometricCritic":
        return import_module("fragile.rl.critic").GeometricCritic
    if name == "RewardHead":
        return import_module("fragile.rl.reward_head").RewardHead
    if name == "SequenceReplayBuffer":
        return import_module("fragile.rl.replay_buffer").SequenceReplayBuffer
    if name == "compute_lambda_returns":
        return import_module("fragile.rl.returns").compute_lambda_returns
    if name == "train":
        return import_module("fragile.rl.train_dreamer").train
    msg = f"module 'fragile.rl' has no attribute {name!r}"
    raise AttributeError(msg)
