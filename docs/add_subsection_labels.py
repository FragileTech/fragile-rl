#!/usr/bin/env python3
"""
Add MyST labels to subsections that are referenced but don't have labels.
"""

from pathlib import Path
import re


DOCS_DIR = Path("/home/guillem/fragile/docs")

# Subsection numbers that need labels
NEEDED_SECTIONS = [
    "1.1",
    "1.1.1",
    "1.1.2",
    "1.1.4",
    "2.2a",
    "2.3",
    "2.6",
    "2.11.3",
    "2.11.4",
    "3.1",
    "3.4",
    "4",
    "4.1.2",
    "7.7.2",
    "7.7.4",
    "7.7.5",
    "7.8.1",
    "7.10.1",
    "7.11.3",
    "7.12",
    "7.12.3",
    "7.13",
    "8.2",
    "8.2.5",
    "9.10",
    "10",
    "10.1.3",
    "11.5",
    "11.5.4",
    "14.2",
    "18.3",
    "21.5",
    "22.1",
    "22.3",
    "22.5",
    "22.7",
    "23.2",
    "23.3",
    "23.4",
    "23.5",
    "23.7",
    "23.8",
    "24.1",
    "24.2.1",
    "24.3",
    "24.5",
    "24.7",
    "25.1",
    "25.2",
    "25.3",
    "25.4",
    "27.1",
    "27.4",
    "29.2",
    "29.4",
    "29.5",
    "29.6",
    "29.7",
    "29.9",
    "29.13",
    "29.14",
    "29.17",
    "29.21",
    "30.4",
    "30.7",
    "30.12",
    "30.14",
    "33.2",
    "34.3",
    "34.6",
]


def slugify(title):
    """Convert title to a slug for label."""
    # Remove special characters, convert to lowercase
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:50]  # Limit length


def find_and_label_sections():
    """Find sections and add labels where missing."""

    # Pattern to match section headers with numbers
    # Matches: ## 2.6 Title, ### 7.10.1 Title, ## 2.2a Title, etc.
    header_pattern = re.compile(
        r"^(#{2,4})\s*(\d+(?:\.\d+)*[A-Za-z]?)[\.:·]?\s+(.+)$", re.MULTILINE
    )

    # Pattern to check if line before is already a label
    label_pattern = re.compile(r"^\(sec-[^)]+\)=$")

    added_labels = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        if "_build" in str(md_file) or ".venv" in str(md_file):
            continue
        if md_file.name in {"index.md.bak", "add_subsection_labels.py", "convert_section_refs.py"}:
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
            continue

        lines = content.split("\n")
        new_lines = []
        modified = False
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a section header
            match = header_pattern.match(line)
            if match:
                match.group(1)
                section_num = match.group(2)
                title = match.group(3).strip()

                # Check if this section needs a label
                if section_num in NEEDED_SECTIONS:
                    # Check if previous line is already a label
                    has_label = False
                    if new_lines and label_pattern.match(new_lines[-1]):
                        has_label = True

                    if not has_label:
                        # Generate label
                        slug = slugify(title)
                        label = f"sec-{section_num.replace('.', '-')}-{slug}"
                        label_line = f"({label})="

                        # Insert label before header
                        new_lines.append(label_line)
                        added_labels.append((section_num, label, md_file.name))
                        modified = True
                        print(
                            f"  Added: ({label})= before '{section_num}. {title[:40]}...' in {md_file.name}"
                        )

            new_lines.append(line)
            i += 1

        if modified:
            md_file.write_text("\n".join(new_lines), encoding="utf-8")

    return added_labels


def main():
    print("Adding labels to subsections...")
    print(f"Looking for {len(NEEDED_SECTIONS)} section numbers\n")

    added = find_and_label_sections()

    print("\n=== SUMMARY ===")
    print(f"Added {len(added)} new labels")

    if added:
        print("\nNew labels added:")
        for section_num, label, filename in sorted(added):
            print(f"  {section_num}: {label}")


if __name__ == "__main__":
    main()
