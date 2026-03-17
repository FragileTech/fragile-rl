#!/usr/bin/env python3
"""
Convert hardcoded section references to MyST {ref} links.

This script:
1. Builds a registry of all MyST labels and their associated section numbers
2. Finds all hardcoded section references not in {ref} syntax
3. Converts them to proper {ref} links where labels exist
4. Reports unmatched references for manual review
"""

from collections import defaultdict
import os
from pathlib import Path
import re


DOCS_DIR = Path("/home/guillem/fragile/docs")

# Patterns
LABEL_PATTERN = re.compile(r"^\(([^)]+)\)=$", re.MULTILINE)
# Match headers with 2-4 hashes: ## 2.6 Title, ### 23.2 Title, #### 7.10.1 Title
SECTION_HEADER_PATTERN = re.compile(
    r"^#{2,4}\s*(\d+(?:\.\d+)*[A-Za-z]?)[\.:·]?\s+(.+)$", re.MULTILINE
)
APPENDIX_HEADER_PATTERN = re.compile(r"^#{2,4}\s*(Appendix\s+[A-E])[\.:·]?\s*(.+)$", re.MULTILINE)

# Pattern to find hardcoded refs NOT already in {ref}
# Simpler patterns - we'll handle context checking in the replacement logic
HARDCODED_SECTION = re.compile(r"\b(Section\s+(\d+(?:\.\d+)*[A-Za-z]?))\b")

HARDCODED_APPENDIX = re.compile(r"\b(Appendix\s+([A-E])(?:\.\d+)?)\b")

HARDCODED_FAQ = re.compile(r"\b(FAQ\s+([A-E])\.(\d+(?:\.\d+)?))\b")


def build_label_registry():
    """Build a mapping of section numbers to MyST labels."""
    registry = {
        "sections": {},  # "2.8" -> "sec-conditional-independence..."
        "appendices": {},  # "A" -> "sec-appendix-a-full-derivations"
        "faq": {},  # "D.1.1" -> "sec-appendix-d-..."
        "all_labels": set(),  # All labels for validation
    }

    # First pass: collect all labels
    for md_file in DOCS_DIR.rglob("*.md"):
        if "_build" in str(md_file) or ".venv" in str(md_file):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Find all labels
        for match in LABEL_PATTERN.finditer(content):
            label = match.group(1)
            registry["all_labels"].add(label)

    # Second pass: map section numbers to labels by looking at headers
    for md_file in DOCS_DIR.rglob("*.md"):
        if "_build" in str(md_file) or ".venv" in str(md_file):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.split("\n")
        pending_label = None

        for line in lines:
            # Check for label definition
            label_match = LABEL_PATTERN.match(line)
            if label_match:
                pending_label = label_match.group(1)
                continue

            # Check for section header
            section_match = SECTION_HEADER_PATTERN.match(line)
            if section_match and pending_label:
                section_num = section_match.group(1)
                registry["sections"][section_num] = pending_label
                pending_label = None
                continue

            # Check for appendix header
            appendix_match = APPENDIX_HEADER_PATTERN.match(line)
            if appendix_match and pending_label:
                appendix_letter = appendix_match.group(1).split()[-1]  # Get "A" from "Appendix A"
                registry["appendices"][appendix_letter] = pending_label
                pending_label = None
                continue

            # Reset pending label if we hit a non-empty, non-label line
            if line.strip() and not line.startswith("("):
                pending_label = None

    # Add known mappings that might be missed
    known_mappings = {
        # Main sections
        "0": "sec-positioning-connections-to-prior-work-differences-and-advantages",
        "0.6": "sec-standard-rl-as-the-degenerate-limit",
        "1": "sec-introduction-the-agent-as-a-bounded-rationality-controller",
        "1.2": "sec-units-and-dimensional-conventions",
        "1.3": "sec-the-chronology-temporal-distinctions",
        "2": "sec-the-control-loop-representation-and-control",
        "2.2b": "sec-the-shutter-as-a-vq-vae",
        "2.5": "sec-second-order-sensitivity-value-defines-a-local-metric",
        "2.5.1": "sec-levi-civita-connection-and-parallel-transport",
        "2.7": "sec-the-hjb-correspondence",
        "2.8": "sec-conditional-independence-and-sufficiency",
        "2.11": "sec-variance-value-duality-and-information-conservation",
        "3": "sec-diagnostics-stability-checks",
        "3.1": "sec-the-60-diagnostic-monitoring-nodes",
        "3.2": "sec-scaling-exponents-characterizing-the-agent",
        "3.3": "sec-defect-functionals-implementing-regulation",
        "3.5": "sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration",
        "4": "sec-limits-barriers-the-limits-of-control",
        "5": "sec-failure-modes",
        "6": "sec-interventions-mitigations",
        "7": "sec-computational-considerations",
        "7.7": "sec-tier-atlas-based-fragile-agent",
        "7.8": "sec-tier-the-attentive-atlas",
        "7.10": "sec-decoder-architecture-overview-topological-decoder",
        "7.11": "sec-the-geometry-of-the-latent-space-a-hyperbolic-hierarchy",
        "7.12": "sec-jump-operators-skill-transitions-and-discrete-control",
        "7.13": "sec-stacked-topoencoders-multi-scale-representation",
        "8": "sec-infeasible-implementation-replacements",
        "9": "sec-the-disentangled-variational-architecture-hierarchical-latent-separation",
        "11": "sec-intrinsic-motivation-maximum-entropy-exploration",
        "12": "sec-belief-dynamics-prediction-update-projection",
        "12.3": "sec-sieve-events-as-projections-reweightings",
        "12.5": "sec-optional-operator-valued-belief-updates",
        "13": "sec-correspondence-table-filtering-control-template",
        "14": "sec-duality-of-exploration-and-soft-optimality",
        "15": "sec-implementation-note-entropy-regularized-optimal-transport-bridge",
        "16": "sec-theorem-the-information-stability-threshold-coupling-window",
        "17": "sec-summary-unified-information-theoretic-control-view",
        "18": "sec-capacity-constrained-metric-law-geometry-from-interface-limits",
        "18.2": "sec-main-result",
        "19": "sec-conclusion",
        "20": "sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces",
        "20.2": "sec-the-wfr-metric",
        "20.5": "sec-connection-to-gksl-master-equation",
        "20.6": "sec-the-unified-world-model",
        "21": "sec-radial-generation-entropic-drift-and-policy-control",
        "21.1": "sec-hyperbolic-volume-and-entropic-drift",
        "21.2": "sec-policy-control-field",
        "21.3": "sec-the-retrieval-texture-firewall",
        "21.4": "sec-summary-and-diagnostic-node",
        "22": "sec-the-equations-of-motion-geodesic-jump-diffusion",
        "22.2": "sec-the-coupled-jump-diffusion-sde",
        "22.4": "sec-the-geodesic-baoab-integrator",
        "22.5": "sec-full-integration-step-summary",
        "23": "sec-the-boundary-interface-symplectic-structure",
        "23.1": "sec-the-symplectic-interface-position-momentum-duality",
        "23.6": "sec-relationship-to-the-context-conditioned-framework",
        "24": "sec-the-reward-field-value-forms-and-hodge-geometry",
        "24.2": "sec-the-bulk-potential-screened-poisson-equation",
        "24.4": "sec-geometric-back-reaction-the-conformal-coupling",
        "25": "sec-supervised-topology-semantic-potentials-and-metric-segmentation",
        "26": "sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller",
        "27": "sec-section-non-local-memory-as-self-interaction-functional",
        "28": "sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back",
        "29": "sec-symplectic-multi-agent-field-theory",
        "30": "sec-ontological-expansion-topological-fission-and-the-semantic-vacuum",
        "30.3": "sec-the-fission-criterion",
        "30.7": "sec-ontological-fission-creating-new-symbols",
        "30.8": "sec-ontological-fusion-concept-consolidation",
        "31": "sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics",
        "32": "sec-causal-discovery-interventional-geometry-and-the-singularity-of-action",
        "33": "sec-causal-information-bound",
        "34": "sec-standard-model-cognition",
    }

    # Add known appendix mappings
    appendix_mappings = {
        "A": "sec-appendix-a-full-derivations",
        "B": "sec-appendix-b-units-parameters-and-coefficients",
        "C": "sec-appendix-c-wfr-stress-energy-tensor",
        "D": "sec-appendix-d-frequently-asked-questions",
        "E": "sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws",
    }

    # FAQ section mappings
    faq_mappings = {
        "D.1": "sec-appendix-d-computational-complexity-scalability",
        "D.1.1": "sec-appendix-d-the-metric-inversion-problem",
        "D.4": "sec-appendix-d-physics-geometry-isomorphisms",
        "D.4.1": "sec-appendix-d-the-validity-of-the-hjb-helmholtz-map",
        "D.4.2": "sec-appendix-d-thermodynamic-metaphors-vs-reality",
        "D.5": "sec-appendix-d-control-theory-system-safety",
        "D.5.4": "sec-appendix-d-adversarial-robustness-of-the-sieve",
        "D.6": "sec-appendix-d-falsifiability",
        "D.6.1": "sec-appendix-d-the-fragile-branding",
        "D.6.2": "sec-appendix-d-the-degenerate-case-claim",
        "D.8": "sec-appendix-d-information-theory-ontology",
        "D.8.4": "sec-appendix-d-singularity-causal-stasis",
    }

    # Merge known mappings (don't override discovered ones)
    for k, v in known_mappings.items():
        if k not in registry["sections"]:
            registry["sections"][k] = v

    for k, v in appendix_mappings.items():
        if k not in registry["appendices"]:
            registry["appendices"][k] = v

    for k, v in faq_mappings.items():
        registry["faq"][k] = v

    return registry


def convert_file(filepath: Path, registry: dict, dry_run: bool = False):
    """Convert hardcoded references in a single file."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    conversions = []
    unmatched = []

    # Convert Section references
    def replace_section(match):
        full_match = match.group(0)
        section_text = match.group(1)
        section_num = match.group(2)

        # Skip if already in a ref or inside code
        if "{ref}" in full_match or "<sec-" in full_match:
            return full_match

        # Look up label
        label = registry["sections"].get(section_num)
        if label and label in registry["all_labels"]:
            replacement = "{ref}`" + section_text + " <" + label + ">`"
            conversions.append((section_text, label))
            return replacement
        unmatched.append(("Section", section_num, str(filepath)))
        return full_match

    # Convert Appendix references
    def replace_appendix(match):
        full_match = match.group(0)
        appendix_text = match.group(1)
        appendix_letter = match.group(2)

        if "{ref}" in full_match or "<sec-" in full_match:
            return full_match

        label = registry["appendices"].get(appendix_letter)
        if label and label in registry["all_labels"]:
            replacement = "{ref}`" + appendix_text + " <" + label + ">`"
            conversions.append((appendix_text, label))
            return replacement
        unmatched.append(("Appendix", appendix_letter, str(filepath)))
        return full_match

    # Convert FAQ references
    def replace_faq(match):
        full_match = match.group(0)
        faq_text = match.group(1)
        faq_letter = match.group(2)
        faq_num = match.group(3)
        faq_key = f"{faq_letter}.{faq_num}"

        if "{ref}" in full_match or "<sec-" in full_match:
            return full_match

        # Try exact match first, then parent section
        label = registry["faq"].get(faq_key)
        if not label:
            # Try parent (e.g., D.1 for D.1.1)
            parent_key = f"{faq_letter}.{faq_num.split('.')[0]}"
            label = registry["faq"].get(parent_key)

        if label and label in registry["all_labels"]:
            replacement = "{ref}`" + faq_text + " <" + label + ">`"
            conversions.append((faq_text, label))
            return replacement
        unmatched.append(("FAQ", faq_key, str(filepath)))
        return full_match

    # Apply conversions (be careful with order to avoid double-converting)
    # Process line by line to avoid converting inside existing {ref} blocks
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Skip lines that are already {ref} formatted
        if "{ref}`" not in line or "Section" in line.split("{ref}")[0]:
            # Only convert if not already in a {ref} on this line
            # Split by {ref} and only process parts before {ref}
            parts = re.split(r"(\{ref\}`[^`]+`)", line)
            new_parts = []
            for part in parts:
                if part.startswith("{ref}`"):
                    new_parts.append(part)
                else:
                    # Apply conversions to non-ref parts
                    part = HARDCODED_SECTION.sub(replace_section, part)
                    part = HARDCODED_APPENDIX.sub(replace_appendix, part)
                    part = HARDCODED_FAQ.sub(replace_faq, part)
                    new_parts.append(part)
            line = "".join(new_parts)
        new_lines.append(line)

    content = "\n".join(new_lines)

    if not dry_run and content != original:
        filepath.write_text(content, encoding="utf-8")

    return conversions, unmatched, content != original


def main():
    print("Building label registry...")
    registry = build_label_registry()
    print(f"  Found {len(registry['all_labels'])} labels")
    print(f"  Mapped {len(registry['sections'])} section numbers")
    print(f"  Mapped {len(registry['appendices'])} appendix letters")
    print(f"  Mapped {len(registry['faq'])} FAQ sections")

    print("\nProcessing files...")

    all_conversions = []
    all_unmatched = []
    modified_files = []

    # Process all markdown files
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        if "_build" in str(md_file) or ".venv" in str(md_file) or "node_modules" in str(md_file):
            continue
        if md_file.name in {"index.md.bak", "extract_sections.py"}:
            continue

        conversions, unmatched, was_modified = convert_file(md_file, registry, dry_run=False)

        if conversions:
            all_conversions.extend(conversions)
            if was_modified:
                modified_files.append(md_file)
                print(f"  {md_file.relative_to(DOCS_DIR)}: {len(conversions)} conversions")

        all_unmatched.extend(unmatched)

    print("\n=== SUMMARY ===")
    print(f"Modified {len(modified_files)} files")
    print(f"Total conversions: {len(all_conversions)}")
    print(f"Unmatched references: {len(all_unmatched)}")

    if all_unmatched:
        print("\n=== UNMATCHED REFERENCES (need manual review) ===")
        # Group by type
        by_type = defaultdict(list)
        for ref_type, ref_num, filepath in all_unmatched:
            by_type[ref_type].append((ref_num, filepath))

        for ref_type, refs in sorted(by_type.items()):
            print(f"\n{ref_type}:")
            # Deduplicate
            seen = set()
            for ref_num, filepath in refs:
                key = (ref_num, filepath)
                if key not in seen:
                    seen.add(key)
                    print(f"  {ref_num} in {Path(filepath).relative_to(DOCS_DIR)}")


if __name__ == "__main__":
    main()
