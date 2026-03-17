# Mathematical Review: docs/source/1_agent/intro_agent.md

## Metadata
- Reviewed file: docs/source/1_agent/intro_agent.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: Fragile Agent Stack summary; Gauge-Theoretic Unification bullets

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 3
- Minor: 0
- Notes: 0
- Primary themes: Miswording, Proof gap / omission, Scope restriction

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | TL;DR / Critic (Field Solver) (line 21) | Moderate | Scope restriction | Screened Poisson claim lacks hypotheses |
| E-002 | TL;DR / Standard Model of Cognition bullet (line 42) | Moderate | Miswording | “Emerges from three invariance principles” stated without axioms |
| E-003 | PoUW Security Properties table / Fake gradients (line 120) | Moderate | Proof gap / omission | Detection probability bound missing sampling assumptions |

## Detailed findings

### [E-001] Screened Poisson claim lacks hypotheses
- Location: TL;DR / Critic (Field Solver) (line 21)
- Severity: Moderate
- Type: Scope restriction (secondary: None)
- Claim (paraphrase): The TL;DR states that the critic solves the screened Poisson equation for the conservative component of reward, as a general fact.
- Why this is an error in the framework: Within the framework, the screened Poisson/Helmholtz correspondence holds only under explicit conditions (existence of a conservative component, boundary conditions, and regularity/compactness or confining envelope). Stating it without hypotheses makes it appear unconditional and hides the non-conservative case where no scalar potential exists.
- Impact on downstream results: Readers may treat the critic PDE form as universally valid and apply PDE-based diagnostics in regimes where the assumptions fail.
- Fix guidance (step-by-step):
  1. Add the exact hypotheses from the HJB–Helmholtz correspondence (conservative component only, boundary/decay conditions, regularity).
  2. Clarify that the PDE applies to the exact component dV and that non-conservative components remain as a connection term.
  3. Point to the formal theorem and state the conditions inline in the TL;DR bullet.
- Required new assumptions/permits (if any): Conservative reward component, well-posed boundary conditions (or confining envelope), and required smoothness of V and G.
- Framework-first proof sketch for the fix: Invoke the HJB/Helmholtz correspondence with the stated hypotheses to justify the screened Poisson equation for V.
- Validation plan: Check all summary references to the critic PDE and ensure they restate the same hypotheses.

### [E-002] “Emerges from three invariance principles” stated without axioms
- Location: TL;DR / Standard Model of Cognition bullet (line 42)
- Severity: Moderate
- Type: Miswording (secondary: None)
- Claim (paraphrase): The TL;DR asserts that the symmetry group G_Fragile emerges from three invariance principles, without listing the axioms needed for the derivation.
- Why this is an error in the framework: The Standard Model chapter derives the group only after adopting specific axioms (local utility invariance, cybernetic parity, feature confinement, etc.). Omitting these prerequisites in the summary turns a conditional theorem into an unconditional claim.
- Impact on downstream results: Downstream readers may treat the group structure as model-independent rather than contingent on the stated axioms.
- Fix guidance (step-by-step):
  1. Add an explicit qualifier that the derivation is conditional on the stated axioms/permits.
  2. Reference the specific axioms (e.g., ax-local-utility-invariance, ax-cybernetic-parity-violation, ax-feature-confinement).
  3. If intended as heuristic, mark it as a proposal rather than a derivation.
- Required new assumptions/permits (if any): Explicit adoption of the Standard Model axioms in Section 34.
- Framework-first proof sketch for the fix: State that the group follows from gauging the listed redundancies under locality and the axioms named in Section 34.
- Validation plan: Verify that every later use of G_Fragile in Volume 1 cites the axioms or the Standard Model chapter explicitly.

### [E-003] Detection probability bound missing sampling assumptions
- Location: PoUW Security Properties table / Fake gradients (line 120)
- Severity: Moderate
- Type: Proof gap / omission (secondary: None)
- Claim (paraphrase): The table states detection probability ≥ 1 − (1 − ε)^k without specifying the sampling model or independence assumptions.
- Why this is an error in the framework: The bound is valid only under explicit assumptions (independent spot-checks with detection probability ε). Without these, the inequality is not justified within the framework.
- Impact on downstream results: Security guarantees may be overstated if sampling is correlated, adversarial, or adaptive.
- Fix guidance (step-by-step):
  1. State the sampling model (independent Bernoulli detection with rate ε).
  2. Define k and ε precisely in terms of protocol parameters.
  3. If sampling is not independent, replace the bound with the correct dependence-aware estimate.
- Required new assumptions/permits (if any): Independence (or a stated dependence model) for spot-check samples.
- Framework-first proof sketch for the fix: Under independent trials, apply the complement probability of k consecutive misses to derive 1 − (1 − ε)^k.
- Validation plan: Confirm the protocol implementation matches the stated sampling assumptions.


## Scope restrictions and clarifications
- State explicit hypotheses (domain, boundary, regularity) wherever results are claimed.

## Proposed edits (optional)
- Add explicit hypothesis or assumption blocks near each affected statement.

## Open questions
- Which permits in Volume 1 are intended to justify the summary claims in this file?
- Should any of the highlighted claims be reclassified as heuristic rather than formal?
