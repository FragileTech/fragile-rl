# Mathematical Review: docs/source/1_agent/01_foundations/01_definitions.md

## Metadata
- Reviewed file: docs/source/1_agent/01_foundations/01_definitions.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: Bounded-Rationality Controller; Boundary / Markov Blanket

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: Conceptual, Notation conflict

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Re-typing RL Primitives / Reward r_t (line 244) and Units / Discrete vs continuous reward (line 376) | Moderate | Notation conflict | r_t and \mathcal{R} overloaded with incompatible units |
| E-002 | Symmetries and Gauge Freedoms / Agent symmetry group; operational (line 295) | Moderate | Conceptual | Objective gauge includes scaling without compensating coefficients |

## Detailed findings

### [E-001] r_t and \mathcal{R} overloaded with incompatible units
- Location: Re-typing RL Primitives / Reward r_t (line 244) and Units / Discrete vs continuous reward (line 376)
- Severity: Moderate
- Type: Notation conflict (secondary: None)
- Claim (paraphrase): The text uses r_t both as an instantaneous rate (⟨\mathcal{R}, \dot z⟩) and as a per-step scalar reward, while \mathcal{R} denotes both a 1-form and a scalar rate.
- Why this is an error in the framework: In the Reward Field anchor, \mathcal{R} is a 1-form with units nat/length; r_t as its evaluation has units nat/time. Later, r_t is defined as a per-step scalar (nat), and \mathcal{R} is re-used as a scalar rate (nat/s). This creates a dimensional/notation conflict inside the foundational definitions.
- Impact on downstream results: Downstream equations can silently mix rates and per-step rewards, leading to incorrect scaling in PDEs and discretizations.
- Fix guidance (step-by-step):
  1. Introduce distinct symbols for the 1-form (\mathcal{R}), the instantaneous rate (e.g., \dot r or \rho_r), and the discrete reward sample (r_t).
  2. State the discretization explicitly: r_t = ∫_{t}^{t+Δt} ⟨\mathcal{R}, \dot z⟩ dt.
  3. Update unit annotations in both sections to reflect the revised notation.
- Required new assumptions/permits (if any): None; this is a notational/units clarification.
- Framework-first proof sketch for the fix: Not applicable; the fix is definitional.
- Validation plan: Check that all later uses of r_t and \mathcal{R} consistently refer to the corrected meanings.

### [E-002] Objective gauge includes scaling without compensating coefficients
- Location: Symmetries and Gauge Freedoms / Agent symmetry group; operational (line 295)
- Severity: Moderate
- Type: Conceptual (secondary: None)
- Claim (paraphrase): The objective gauge group includes positive affine transformations r → a r + b as symmetries that should not change policy updates.
- Why this is an error in the framework: In the Standard Model anchor, the utility gauge freedom corresponds to additive phase shifts (U(1)). Positive scaling generally changes the relative weight of entropy and constraint penalties unless temperature and multipliers are rescaled simultaneously. As written, the symmetry claim is too strong and conflicts with the framework’s entropy-regularized objectives.
- Impact on downstream results: Treating scaling as a gauge invariance can mask genuine changes in policy behavior under reward rescaling.
- Fix guidance (step-by-step):
  1. Restrict the objective gauge to additive shifts only, or explicitly state the compensating rescaling of T_c and penalty multipliers required for invariance.
  2. Reference the utility-gauge axiom in the Standard Model chapter to align with the U(1) phase symmetry.
  3. Add a short note distinguishing invariance of action ranking from invariance of entropy-regularized objectives.
- Required new assumptions/permits (if any): If scaling is retained, assume simultaneous rescaling of all temperature and penalty coefficients.
- Framework-first proof sketch for the fix: Show that additive shifts leave policy gradients invariant, while positive scaling requires corresponding coefficient rescaling to preserve the optimization objective.
- Validation plan: Verify that the policy update equations are invariant under the stated gauge group after the corrections.


## Scope restrictions and clarifications
- Disambiguate symbols with conflicting meanings and units.

## Proposed edits (optional)
- Add explicit hypothesis or assumption blocks near each affected statement.
- Introduce distinct symbols for rates vs. per-step quantities and clarify units.

## Open questions
- Which permits in Volume 1 are intended to justify the summary claims in this file?
- Should any of the highlighted claims be reclassified as heuristic rather than formal?
