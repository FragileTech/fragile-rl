# Mathematical Review: docs/source/1_agent/07_cognition/05_metabolism.md

## Metadata
- Reviewed file: docs/source/1_agent/07_cognition/05_metabolism.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: Metabolic Flux; Metabolic Potential

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: External dependency, Scope restriction

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | The Energetics of Information Updates / Generalized Landauer Bound (line 113) | Moderate | Scope restriction | Missing hypotheses in Generalized Landauer Bound |
| E-002 | The Energetics of Information Updates / Generalized Landauer Bound (line 113) | Moderate | External dependency | External dependency in Generalized Landauer Bound |

## Detailed findings

### [E-001] Missing hypotheses in Generalized Landauer Bound
- Location: The Energetics of Information Updates / Generalized Landauer Bound (line 113)
- Severity: Moderate
- Type: Scope restriction (secondary: None)
- Claim (paraphrase): The statement labeled "Generalized Landauer Bound" is presented as a general result without explicit domain/regularity assumptions in the theorem block.
- Why this is an error in the framework: Within the framework anchored by the Reward Field and Standard Model chapters, results that depend on geometry, dynamics, or PDE structure require explicit hypotheses (manifold class, boundary conditions, regularity, and parameter ranges). Without those hypotheses in the theorem statement, the scope of validity is ambiguous and cannot be certified purely from internal permits.
- Impact on downstream results: Downstream results that invoke this theorem may silently assume stronger conditions, risking invalid conclusions when those conditions fail.
- Fix guidance (step-by-step):
  1. Add an explicit hypotheses line (manifold class, boundary/decay conditions, and smoothness of fields/operators).
  2. List dependencies on prior permits/definitions (cite the specific sections or axioms that supply them).
  3. If the result only holds in a restricted regime (e.g., compact domain or conservative reward), state that restriction in the theorem header.
- Required new assumptions/permits (if any): Explicit regularity and boundary assumptions consistent with the Reward Field and Standard Model anchors.
- Framework-first proof sketch for the fix: Under the stated hypotheses, reduce the claim to the internal geometric/PDE permits already defined in Volume 1, then show how the conclusion follows by applying those permits in the stated order.
- Validation plan: Verify that each downstream reference to this theorem restates or inherits the same hypotheses.

### [E-002] External dependency in Generalized Landauer Bound
- Location: The Energetics of Information Updates / Generalized Landauer Bound (line 113)
- Severity: Moderate
- Type: External dependency (secondary: None)
- Claim (paraphrase): The block "Generalized Landauer Bound" cites external sources within the formal statement, but does not provide an internal permit or derivation.
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
- State manifold class, boundary conditions, and regularity explicitly where results are claimed.
- Replace external citations in formal statements with internal permits or explicit assumptions.

## Proposed edits (optional)
- Add hypothesis lines to theorem blocks that currently state results without domains/regularity.
- Insert permit statements for any external results referenced in formal claims.

## Open questions
- Which permits in Volume 1 should be treated as foundational for this chapter?
- Are any of the referenced external results intended to be axioms rather than derived permits?
