"""CLI entry point for fragile: ``uv run fragile <command>``."""

import click


@click.group()
def run():
    """Fragile – geometric RL toolkit."""


@run.command("vla-dashboard")
@click.option("--port", default=5009, type=int, help="Port to run server on.")
@click.option("--open", "open_browser", is_flag=True, help="Open browser on launch.")
@click.option("--outputs", default="outputs/vla", help="VLA outputs directory.")
def vla_dashboard(port, open_browser, outputs):
    """Launch the VLA World Model dashboard."""
    import panel as pn

    from fragile.vla.dashboard import create_app

    click.echo("Starting VLA World Model Dashboard...")
    click.echo(f"Open http://localhost:{port} in your browser (Ctrl+C to stop)")
    pn.serve(
        lambda: create_app(outputs_dir=outputs),
        port=port,
        show=open_browser,
        title="VLA World Model Dashboard",
        websocket_origin="*",
    )


@run.command(
    "vla-train",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_train(args):
    """Train the TopoEncoder VLA experiment (3-phase pipeline).

    All arguments after 'vla-train' are forwarded to the training script.
    Run `uv run fragile vla-train -- --help` for all config options.
    """
    import sys

    from fragile.vla.train import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-train", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-unsup",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_unsup(args):
    """Unsupervised TopoEncoder training on VLA features.

    All arguments after 'vla-unsup' are forwarded to the training script.
    Run `uv run fragile vla-unsup -- --help` for all training options.
    """
    import sys

    from fragile.vla.train_unsupervised import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-unsup", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-phase1",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_phase1(args):
    """Standalone Phase 1 encoder training.

    All arguments after 'vla-phase1' are forwarded to the training script.
    Run `uv run fragile vla-phase1 -- --help` for all training options.
    """
    import sys

    from fragile.vla.train_phase_1 import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-phase1", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-geometry",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_geometry(args):
    """Train the new sequence-based geometry stack on cached VLA windows.

    Uses Hydra-style key=value overrides, e.g.:
        uv run fragile vla-geometry epochs=200 agent.obs_encoder.num_charts=16
    """
    import sys

    from fragile.commands.train_geometry import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-geometry", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "macro-rl",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def macro_rl(args):
    """Train the standalone off-policy macro RL agent.

    Uses Hydra-style key=value overrides, e.g.:
        uv run fragile macro-rl epochs=200 domain=cartpole task=balance
    """
    import sys

    from fragile.commands.train_macro_rl import main

    original_argv = sys.argv
    sys.argv = ["fragile-macro-rl", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "dataset",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def dataset(args):
    """Extract SmolVLA latents and cache them for world model training.

    Runs the frozen SmolVLA backbone over every frame in a LeRobot dataset
    and saves per-episode features, actions, and states as .pt files.

    Run `uv run fragile dataset -- --help` for all options.
    """
    import sys

    from fragile.vla.create_latent_dataset import main

    original_argv = sys.argv
    sys.argv = ["fragile-dataset", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


@run.command(
    "vla-joint",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vla_joint(args):
    """Joint encoder + world model training (3-phase pipeline).

    All arguments after 'vla-joint' are forwarded to the training script.
    Run `uv run fragile vla-joint -- --help` for all training options.
    """
    import sys

    from fragile.vla.train_joint import main

    original_argv = sys.argv
    sys.argv = ["fragile-vla-joint", *args]
    try:
        main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    run()
