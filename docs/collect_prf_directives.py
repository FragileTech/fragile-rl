#!/usr/bin/env python3
"""
Collect MyST prf directives into per-volume markdown files.

By default, prf:proof blocks are excluded. Use --include-proofs to include them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


DIRECTIVE_OPEN_RE = re.compile(r"^\s*:{2,}\s*\{(?P<name>[^}]+)\}.*$")
DIRECTIVE_CLOSE_RE = re.compile(r"^\s*:{2,}\s*$")
VOLUME_DIR_RE = re.compile(r"^\d+_")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_source = script_dir / "source"
    default_out_dir = repo_root / "outputs" / "prf"

    parser = argparse.ArgumentParser(
        description="Extract prf directives from docs/source into per-volume markdown files.",
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
        help="Output directory for condensed markdown files (default: outputs/prf).",
    )
    parser.add_argument(
        "--include-proofs",
        action="store_true",
        help="Include prf:proof directives (default: excluded).",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all top-level subdirectories under source, not just numeric volumes.",
    )
    parser.add_argument(
        "--include-file-headings",
        action="store_true",
        help="Add a '## <file>' heading before blocks from each file.",
    )
    return parser.parse_args()


def volume_dirs(source_dir: Path, include_all: bool) -> list[Path]:
    volumes = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        if include_all or VOLUME_DIR_RE.match(child.name):
            volumes.append(child)
    return volumes


def extract_prf_blocks(text: str, include_proofs: bool) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    stack: list[tuple[int, bool, str]] = []

    for idx, line in enumerate(lines):
        open_match = DIRECTIVE_OPEN_RE.match(line)
        if open_match:
            name = open_match.group("name").strip()
            is_prf = False
            kind = ""
            if name.startswith("prf:"):
                is_prf = True
                kind = name.split(":", 1)[1].strip()
            stack.append((idx, is_prf, kind))
            continue

        if DIRECTIVE_CLOSE_RE.match(line):
            if not stack:
                continue
            start_idx, is_prf, kind = stack.pop()
            if not is_prf:
                continue
            if kind.lower() == "proof" and not include_proofs:
                continue
            blocks.append("\n".join(lines[start_idx : idx + 1]))

    return blocks


def build_volume_output(
    volume_dir: Path,
    include_proofs: bool,
    include_file_headings: bool,
) -> tuple[str, int, int]:
    parts: list[str] = []
    block_count = 0
    file_count = 0

    for md_file in sorted(volume_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        blocks = extract_prf_blocks(content, include_proofs=include_proofs)
        if not blocks:
            continue
        file_count += 1
        if include_file_headings:
            parts.append(f"## {md_file.relative_to(volume_dir)}")
        parts.extend(blocks)
        block_count += len(blocks)

    output = "\n\n".join(parts).rstrip()
    if output:
        output += "\n"
    return output, block_count, file_count


def main() -> None:
    args = parse_args()
    source_dir = args.source
    out_dir = args.out_dir

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for volume_dir in volume_dirs(source_dir, include_all=args.include_all):
        output, block_count, file_count = build_volume_output(
            volume_dir,
            include_proofs=args.include_proofs,
            include_file_headings=args.include_file_headings,
        )
        out_path = out_dir / f"{volume_dir.name}.md"
        out_path.write_text(output, encoding="utf-8")
        print(f"{volume_dir.name}: {block_count} blocks from {file_count} files -> {out_path}")


if __name__ == "__main__":
    main()
