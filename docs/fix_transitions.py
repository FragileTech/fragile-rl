#!/usr/bin/env python3
"""Remove horizontal rules for JupyterBook compatibility.

JupyterBook/Sphinx docutils fails with AssertionError when --- (transitions)
appear in certain contexts. This script removes all --- except YAML frontmatter.
"""

import sys


def fix_transitions(content: str) -> tuple[str, int]:
    """Remove all --- except YAML frontmatter. Returns (fixed_content, count_removed)."""
    lines = content.split("\n")
    result = []
    removed = 0
    hr_count = 0  # Track which --- we're at
    in_frontmatter = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "---":
            hr_count += 1
            # First --- starts frontmatter (if at line 0 or 1)
            if hr_count == 1 and i <= 1:
                in_frontmatter = True
                result.append(line)
                continue
            # Second --- closes frontmatter
            if hr_count == 2 and in_frontmatter:
                in_frontmatter = False
                result.append(line)
                continue
            # All other --- are removed
            removed += 1
            result.append("")  # Keep empty line for spacing
            continue

        result.append(line)

    return "\n".join(result), removed


if __name__ == "__main__":
    total = 0
    for filepath in sys.argv[1:]:
        with open(filepath) as f:
            content = f.read()
        fixed, count = fix_transitions(content)
        with open(filepath, "w") as f:
            f.write(fixed)
        print(f"{filepath}: removed {count} transitions")
        total += count
    print(f"Total: {total}")
