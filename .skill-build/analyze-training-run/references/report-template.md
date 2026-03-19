# Training Run Report Template

Use this exact section order. Replace bracketed guidance with concrete findings.

## 1. Executive Verdict

- Overall result: [`successful`, `partially successful`, `unsuccessful`]
- One-sentence verdict: [State what truly learned, what did not, and whether the run is worth keeping]
- Best phase: [epoch/checkpoint and why]
- Immediate recommendation: [`keep`, `keep with caveats`, `do not use`, `rerun after fixes`]

## 2. Run Snapshot

- Command or run identifier: [if present]
- Device / scale summary: [device, dataset size, model size, sequence length if present]
- Requested duration vs completed duration: [e.g. 1000 requested, 134 completed]
- Eval cadence and save cadence: [state both if known]
- Available checkpoints: [list saved checkpoints if present]

## 3. Best Window

- Best logged eval epoch: [epoch, metric value]
- Best saved checkpoint: [checkpoint or “none exactly matches best eval”]
- Why this is the best window:
  - [best train/eval balance]
  - [which components were strongest here]
  - [what had not broken yet]

## 4. What Worked

For each point:

- Signal: [metric and epoch range]
- Why it is real: [why this is not just a logging artifact]
- Keep as is? [`yes` or `mostly yes`]

Typical topics:

- basic optimization stability
- action encoder learning
- observation encoder learning
- generalization around the best phase
- checkpointing / operational hygiene

## 5. What Failed

For each issue:

- Issue: [short name]
- Evidence: [metrics, epochs, checkpoint events]
- Impact: [why it matters]
- Severity: [`critical`, `high`, `medium`, `low`]

Typical topics:

- startup instability that persisted
- ungrounded symbols
- chart or code collapse
- enclosure staying at chance
- markov transition model staying weak
- widening train/eval gap
- checkpoint metadata lagging actual model state
- run ending before completion

## 6. Root Cause Hypotheses

List 2-5 likely causes, each with:

- Hypothesis
- Evidence that supports it
- Evidence that weakens it
- Confidence: [`high`, `medium`, `low`]

## 7. Keep As Is

Only list items with evidence behind them.

- [hyperparameter, schedule, component, or practice]
- Why to keep it

## 8. Change Next

Rank fixes in priority order.

For each fix include:

- Change
- Why it addresses the observed failure
- Expected upside
- Risk / tradeoff
- How to verify on the next run

## 9. Checkpoint Recommendation

- Recommended checkpoint to keep: [path or “none”]
- Reason
- Caveat: [for example best eval not saved, or checkpoint metadata is stale]

## 10. Validation Gaps

- What cannot be concluded from the log alone
- What extra artifact would reduce uncertainty next time

## Writing Rules

- Use concrete numbers for start, best, degradation, and final phases
- Distinguish clearly between optimization success and representation success
- Compare `loss/main` against component losses; do not rely on `loss/main` alone
- Mention whether the run is incomplete
- Mention whether the best phase was saved
- Keep recommendations action-oriented
