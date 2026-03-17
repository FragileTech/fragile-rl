"""Entry point: ``python -m fragile.rl``."""

from .train_dreamer import _parse_args, train


if __name__ == "__main__":
    train(_parse_args())
