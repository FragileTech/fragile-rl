# Mathematical Review: docs/source/1_agent/04_control/02_belief_dynamics.md

## Metadata
- Reviewed file: docs/source/1_agent/04_control/02_belief_dynamics.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: Belief operator; GKSL generator

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: External dependency

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Optional: Operator-Valued Belief Updates (GKSL / "Lindblad" Form) / GKSL generator (line 268) | Moderate | External dependency | External dependency in GKSL generator |

## Detailed findings

### [E-001] External dependency in GKSL generator
- Location: Optional: Operator-Valued Belief Updates (GKSL / "Lindblad" Form) / GKSL generator (line 268)
- Severity: Moderate
- Type: External dependency (secondary: None)
- Claim (paraphrase): The block "GKSL generator" cites external sources within the formal statement, but does not provide an internal permit or derivation.
- Why this is an error in the framework: The framework-first rule requires that any classical results used in formal statements be reproduced as internal permits or explicitly adopted assumptions. Inline citations are not sufficient to justify the step within the Volume 1 framework.
- Impact on downstream results: Any result that depends on this cited step inherits unproven external assumptions, weakening the internal proof chain.
- Fix guidance (step-by-step):
  1. Replace the external citation with an internal permit/lemma statement (or add one in the appendices).
  2. List the exact hypotheses required by the external result (e.g., smoothness, compactness, boundary conditions).
  3. If the result is intended as heuristic, mark it as such and quarantine downstream use to heuristic sections.
- Required new assumptions/permits (if any): A formal permit encapsulating the cited result, with explicit hypotheses aligned to the framework.
- Framework-first proof sketch for the fix: Either derive the result from existing Volume 1 axioms or add a permit that encodes the external theorem as an admissible step, then re-run the local proof chain using that permit.
- Validation plan: Check that any downstream theorem referencing this block is updated to cite the new permit and its hypotheses.


## Scope restrictions and clarifications
- Replace external citations in formal statements with internal permits or explicit assumptions.

## Proposed edits (optional)
- Add hypothesis lines to theorem blocks that currently state results without domains/regularity.
- Insert permit statements for any external results referenced in formal claims.

## Open questions
- Which permits in Volume 1 should be treated as foundational for this chapter?
- Are any of the referenced external results intended to be axioms rather than derived permits?
