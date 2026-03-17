# Mathematical Review: docs/source/1_agent/02_sieve/01_diagnostics.md

## Metadata
- Reviewed file: docs/source/1_agent/02_sieve/01_diagnostics.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: terminology and notation defined within this file

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 1
- Notes: 0
- Primary themes: External dependency

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Roadmap (line 33) | Minor | External dependency | External dependency in narrative claim |

## Detailed findings

### [E-001] External dependency in narrative claim
- Location: Roadmap (line 33)
- Severity: Minor
- Type: External dependency (secondary: None)
- Claim (paraphrase): A narrative statement cites external sources without an internal permit or scoped assumption for reuse in formal results.
- Why this is an error in the framework: Citations in narrative sections are acceptable for context, but if the statement is used downstream as a formal step, the framework requires an internal permit or an explicit assumption block.
- Impact on downstream results: Downstream derivations may inherit unvetted external assumptions if the claim is treated as formal.
- Fix guidance (step-by-step):
  1. Clarify whether the cited statement is contextual or a formal dependency.
  2. If formal, add an internal permit/lemma encapsulating the cited result and its hypotheses.
  3. If contextual, mark the statement as heuristic and avoid referencing it in proofs.
- Required new assumptions/permits (if any): None if treated as context; otherwise add a permit with explicit hypotheses.
- Framework-first proof sketch for the fix: Not applicable unless the claim is promoted to a formal dependency.
- Validation plan: Check references to this statement in later chapters and ensure only formal permits are cited in proofs.


## Scope restrictions and clarifications
- Replace external citations in formal statements with internal permits or explicit assumptions.

## Proposed edits (optional)
- Add a short scope paragraph tying summary claims to specific theorems/permits.

## Open questions
- Which permits in Volume 1 should be treated as foundational for this chapter?
- Are any of the referenced external results intended to be axioms rather than derived permits?
