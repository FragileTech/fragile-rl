---
name: analyze-training-run
description: Analyze model-training CLI logs and convergence traces, especially epoch-by-epoch outputs with train/eval losses, component metrics, router statistics, checkpoint saves, and code-activity tables. Use when Codex needs to read logs like debug.md, explain what worked and what failed, diagnose collapse/overfitting/ungrounded symbols, compare the best and final phases of a run, and produce a standardized report with concrete fixes and keep-as-is guidance.
---

# Analyze Training Run

## Overview

Read training logs, extract the run structure, identify the best and worst phases, and write a consistent convergence report that separates real learning from misleading loss drops.

Prefer evidence over intuition. Every major conclusion should point to specific metrics, epochs, or checkpoint events.

## Workflow

1. Locate the main log artifact.
   Common inputs are `.md`, `.txt`, copied terminal output, or notebook cell output.

2. Detect whether the log matches the structured epoch format:
   `Name E00000 | train=... | eval=... | step=...`

3. If the format matches, run `scripts/summarize_training_log.py <log-path> --format markdown`.
   Also run `--format json` when you need to inspect the parsed fields more closely.

4. Read `references/report-template.md` and use that section order in the final report.

5. Read `references/diagnostic-patterns.md` when you need help interpreting:
   - convergence vs false progress
   - collapse in routers or codebooks
   - chance-level enclosure or transition behavior
   - train/eval divergence
   - checkpoint-selection issues

6. Cross-check the code only when metric meaning is unclear.
   Typical files to inspect are the training command, trainer, and loss definitions.

7. Write the final report using the template.
   Keep it concrete: start, best, degradation point, final state, fixes, and keep-as-is items.

## Required Checks

Always cover these points, even if the answer is "not present in the log":

- Requested epochs vs completed epochs
- Best eval phase and best logged epoch
- Whether the best phase was actually saved as a checkpoint
- Whether checkpoint metadata may be stale because save cadence and eval cadence differ
- Startup instability vs persistent instability
- Train/eval gap around the best phase and near the end
- Which objective terms truly improved and which stayed flat or worsened
- Whether symbolic metrics are grounded or effectively random
- Whether any collapse happened in routers, charts, codes, or probes
- Whether later training improved quality or only reduced a narrow loss
- What should be kept as-is vs changed next

## Interpreting The Run

Do not treat lower `loss/main` as sufficient evidence of success.

Check component metrics separately:

- `loss/act`, `loss/obs`: say whether each encoder stack actually learned
- `enclosure/*`: say whether structured state predicts the next symbolic target
- `markov/*`: say whether transition modeling is useful or merely imitating diffuse targets
- `I_XK`, `H_K`, `top1_prob_mean`, `active_code_charts`, code-activity arrays:
  use these to detect grounding vs collapse
- `alpha`, warmup, temperature, and checkpoint events:
  use these to explain phase transitions

When a metric can look "good" for the wrong reason, say so explicitly.
Examples:

- low CE because the target space collapsed
- high confidence because the router collapsed to one chart
- low transition CE while hard chart accuracy stays near chance
- better train loss with a worse eval phase

## Script Usage

Use `scripts/summarize_training_log.py` first when the log is structured.

The script is meant to:

- extract epochs, eval checkpoints, and save events
- summarize key component trends
- flag likely failure patterns
- produce a compact table you can cite while writing the report

If the parser misses fields, continue manually. Do not force the script output to be the final answer.

## Output Rules

- Follow `references/report-template.md`
- Use exact numbers for the start, best, degradation, and final phases
- Separate `What Worked`, `What Failed`, `Keep As Is`, and `Change Next`
- Rank fixes by importance
- Explain why each fix addresses the observed evidence
- Call out uncertainty where the log is incomplete
- Mention when the run ended early or lacks a final checkpoint

## References

- Read `references/report-template.md` every time before writing the final report
- Read `references/diagnostic-patterns.md` when interpreting convergence behavior or proposing fixes
