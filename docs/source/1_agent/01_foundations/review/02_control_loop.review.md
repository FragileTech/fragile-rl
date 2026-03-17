# Mathematical Review: docs/source/1_agent/01_foundations/02_control_loop.md

## Metadata
- Reviewed file: docs/source/1_agent/01_foundations/02_control_loop.md
- Review date: January 27, 2026
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Reward field anchors: Reward 1-form, boundary reward flux, Hodge decomposition (docs/source/1_agent/06_fields/02_reward_field.md)
  - Standard model anchors: utility gauge freedom, local utility invariance, covariant derivatives (docs/source/1_agent/08_multiagent/02_standard_model.md)
  - Local anchors: State-Space Sensitivity Metric; Complete Latent Space Metric

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 3
- Minor: 0
- Notes: 0
- Primary themes: Definition mismatch, Dimensional mismatch, Invalid inference, Proof gap / omission

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | State-Space Sensitivity Metric (line 700) | Major | Definition mismatch | Hessian-based metric not guaranteed positive definite |
| E-002 | A Practical Diagonal Sensitivity Metric (line 807) | Moderate | Dimensional mismatch | Diagonal metric defined with four scalars lacks dimension mapping |
| E-003 | Closure Defect / Computational Meaning (line 1128) | Moderate | Invalid inference | Equivalence between δ_CE and conditional mutual information is not justified |
| E-004 | Variance–Curvature Correspondence (line 1290) | Moderate | Proof gap / omission | Policy covariance scaling lacks required assumptions |

## Detailed findings

### [E-001] Hessian-based metric not guaranteed positive definite
- Location: State-Space Sensitivity Metric (line 700)
- Severity: Major
- Type: Definition mismatch (secondary: None)
- Claim (paraphrase): The value-curvature metric component is defined as the Hessian of V, then used as part of a Riemannian metric.
- Why this is an error in the framework: A Riemannian metric must be positive definite, but the Hessian of V is generally indefinite unless V is convex. Later assumptions require G ≻ 0, which does not follow from the definition as stated. The definition therefore fails without additional hypotheses or a PSD proxy. **Implementation note:** the current `fragile/core/layers` architecture uses an SPD proxy metric by construction (conformal metric λ(z)^2 I with clamping/ε), so the deployed geometry is well‑behaved; the doc should make that implementation scoping explicit to avoid a theory/implementation mismatch.
- Impact on downstream results: The theoretical derivations that rely on Hess(V) as a metric are not licensed unless convexity/PSD conditions are stated. In practice, code paths using the conformal SPD proxy remain stable, but the document does not currently distinguish those regimes.
- Fix guidance (step-by-step):
  1. Either assume V is (locally) convex on the operating domain, or redefine G_V using a PSD proxy (Gauss–Newton / outer product), and explicitly state that the implementation uses an SPD proxy metric by construction.
  2. If using the Hessian directly, state a restriction to regions where Hess(V) ≽ 0.
  3. Add a regularization term (e.g., εI) to guarantee positive definiteness when needed.
- Required new assumptions/permits (if any): Convexity or local PSD of Hess(V), or explicit adoption of a PSD proxy metric.
- Framework-first proof sketch for the fix: Under convexity (or PSD proxy), G_V is PSD; combined with λ_G G_π and/or εI yields a positive-definite metric.
- Validation plan: Check that all uses of G assume SPD and that proxy/regularization is implemented in the diagnostics.

### [E-002] Diagonal metric defined with four scalars lacks dimension mapping
- Location: A Practical Diagonal Sensitivity Metric (line 807)
- Severity: Moderate
- Type: Dimensional mismatch (secondary: None)
- Claim (paraphrase): The text defines G = diag(α, β_π, γ_wm, δ) as a state-space metric.
- Why this is an error in the framework: The latent space typically has dimension d ≫ 4, so a 4×4 diagonal matrix cannot represent a metric on \mathcal{Z} without an explicit block structure or basis mapping to (K, z_n, z_tex, …). The statement is therefore dimensionally ambiguous.
- Impact on downstream results: Implementations could misinterpret the metric as a four-dimensional object, breaking the intended preconditioning and unit checks.
- Fix guidance (step-by-step):
  1. Define a block-diagonal structure, e.g., G = diag(α I_{d_K}, β_π I_{d_n}, γ_wm I_{d_tex}, δ I_{d_rest}).
  2. Specify the mapping from latent coordinates to each block.
  3. State the conditions under which scalar summaries are valid diagnostics rather than full metrics.
- Required new assumptions/permits (if any): Explicit block decomposition of \mathcal{Z} and fixed dimensions for each subspace.
- Framework-first proof sketch for the fix: Given a block decomposition, scalar coefficients scale each subspace metric consistently, yielding a valid diagonal approximation.
- Validation plan: Verify that the implementation applies each coefficient to the correct coordinate block.

### [E-003] Equivalence between δ_CE and conditional mutual information is not justified
- Location: Closure Defect / Computational Meaning (line 1128)
- Severity: Moderate
- Type: Invalid inference (secondary: None)
- Claim (paraphrase): The text states δ_CE > 0 is equivalent to I(K_{t+1}; Z_t | K_t, K_act_t) > 0.
- Why this is an error in the framework: The conditional mutual information equals an expected KL only when the macro kernel in the KL is the true conditional distribution. Here δ_CE is defined using a learned \bar P, so the equivalence fails unless \bar P matches the true conditional law.
- Impact on downstream results: The closure defect may be misinterpreted as a strict information-theoretic sufficiency test when it is actually a model-mismatch measure.
- Fix guidance (step-by-step):
  1. Replace “equivalently” with a conditional statement: δ_CE equals the conditional mutual information if \bar P = P(K_{t+1}|K_t,K_act_t).
  2. If \bar P is approximate, describe δ_CE as an upper bound or surrogate.
  3. Add a short note about estimation error and the role of \bar P in the diagnostic.
- Required new assumptions/permits (if any): Exact macro kernel matching the true conditional law, if equivalence is desired.
- Framework-first proof sketch for the fix: Under \bar P = P(K_{t+1}|K_t,K_act_t), δ_CE reduces to the expected KL defining conditional mutual information.
- Validation plan: Check the diagnostic code to ensure it is labeled as a surrogate unless \bar P is exact.

### [E-004] Policy covariance scaling lacks required assumptions
- Location: Variance–Curvature Correspondence (line 1290)
- Severity: Moderate
- Type: Proof gap / omission (secondary: None)
- Claim (paraphrase): The lemma asserts Σ_π(z) ∝ β_cpl^{-1} G^{-1}(z) based on a stationary distribution argument.
- Why this is an error in the framework: The proof sketch links a state distribution p(z) to policy covariance without specifying a policy family (e.g., Gaussian) or a quadratic approximation connecting action variance to state-space curvature. The step requires additional assumptions that are not stated.
- Impact on downstream results: Using this relation for diagnostics or controller tuning may be invalid when the policy or dynamics deviate from the assumed regime.
- Fix guidance (step-by-step):
  1. State the policy family (e.g., Gaussian with covariance Σ_π) and the local quadratic approximation of V.
  2. Clarify the dynamical assumptions that link action variance to state-space sensitivity.
  3. If the relation is heuristic, label it as such and provide a bounded empirical diagnostic instead.
- Required new assumptions/permits (if any): Gaussian policy (or exponential family), local quadratic V, and a link between action noise and state curvature.
- Framework-first proof sketch for the fix: Under a quadratic V and Gaussian policy, match the entropy-regularized objective to obtain Σ_π ∝ β_cpl^{-1} G^{-1}.
- Validation plan: Check empirical covariance against G^{-1} in regimes where the assumptions hold.


## Scope restrictions and clarifications
- Align definitions with required positivity/units and document any proxy choices.

## Proposed edits (optional)
- Add explicit hypothesis or assumption blocks near each affected statement.
- Introduce distinct symbols for rates vs. per-step quantities and clarify units.

## Open questions
- Which permits in Volume 1 are intended to justify the summary claims in this file?
- Should any of the highlighted claims be reclassified as heuristic rather than formal?
