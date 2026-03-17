#!/usr/bin/env python3
"""
Generate prompt download files (with and without proofs) for the docs site.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_VOLUMES = ("1_agent", "2_hypostructure", "3_fractal_gas")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "source"
    default_out_dir = script_dir / "_static" / "prompts"

    parser = argparse.ArgumentParser(
        description="Build prompt downloads into docs/_static/prompts.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Docs source directory (default: docs/source).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for download files (default: docs/_static/prompts).",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all top-level subdirectories under source.",
    )
    parser.add_argument(
        "--no-file-headings",
        action="store_true",
        help="Disable file headings in the output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source
    out_dir = args.out_dir

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    try:
        from collect_prf_directives import build_volume_output, volume_dirs
    except ImportError as exc:
        raise SystemExit(f"Failed to import collect_prf_directives: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("*-with-proofs.*", "*-no-proofs.*"):
        for path in out_dir.glob(pattern):
            path.unlink()

    include_file_headings = not args.no_file_headings

    volumes = volume_dirs(source_dir, include_all=args.include_all)
    if not args.include_all:
        volumes = [volume for volume in volumes if volume.name in DEFAULT_VOLUMES]

    for volume_dir in volumes:
        for include_proofs, proof_slug in ((True, "with-proofs"), (False, "no-proofs")):
            output, block_count, file_count = build_volume_output(
                volume_dir,
                include_proofs=include_proofs,
                include_file_headings=include_file_headings,
            )
            if not output.strip():
                continue
            base_name = f"{volume_dir.name}-{proof_slug}"
            for ext in ("md", "txt"):
                out_path = out_dir / f"{base_name}.{ext}"
                out_path.write_text(output, encoding="utf-8")
            print(
                f"{volume_dir.name} ({proof_slug}): {block_count} blocks from"
                f" {file_count} files -> {base_name}.md/.txt"
            )


if __name__ == "__main__":
    main()
