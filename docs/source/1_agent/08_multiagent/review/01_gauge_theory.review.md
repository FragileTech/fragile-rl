# Mathematical Review: docs/source/1_agent/08_multiagent/01_gauge_theory.md

## Metadata
- Reviewed file: docs/source/1_agent/08_multiagent/01_gauge_theory.md
- Review date: 2026-01-28
- Reviewer: Codex
- Scope: Mass-gap section (sec-mass-gap) and Yang-Mills mass-gap claims/corollaries in this file.
- Foundational anchors (definitions/axioms in the text):
  - Theorem thm-causal-information-bound + Definition def-levin-length (capacity/UV cutoff).
  - Theorem thm-causal-stasis (stasis at capacity saturation).
  - Theorem thm-the-hjb-helmholtz-correspondence (screening mass via Helmholtz form).
  - Definition def-mass-gap (Hamiltonian gap Δ_H) + KG gap definition in sec-mass-gap.
  - Constructive axioms: ax-constructive-finite-resolution, ax-constructive-positivity, ax-constructive-locality (docs/source/1_agent/08_multiagent/02_standard_model.md).
  - Yang-Mills anchors: thm-yang-mills-equations + def-yang-mills-action (docs/source/1_agent/08_multiagent/01_gauge_theory.md).

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: Missing internal lemmas for correlation→information bounds; scope/definition mismatch between KG screening gaps and YM Hamiltonian gaps; conditional computability now anchored by constructive axioms but not cited; CFT scaling results used without stated hypotheses.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Theorem "Computational Necessity of Mass Gap" (sec-mass-gap) | Major | Proof gap / external dependency | Correlation decay ⇒ mutual-information divergence is asserted without an internal lemma or stated hypotheses. |
| E-002 | Theorem "Mass Gap from Screening" + Schrödinger reduction (sec-mass-gap) | Major | Definition mismatch / scope restriction | KG screening gap is treated as YM Hamiltonian mass gap without a proved bridge for the gauge sector. |
| E-003 | Theorem "Mass Gap Dichotomy for Yang-Mills" + Clay remark | Moderate | Scope restriction | Conditional computability assumption now exists as a constructive axiom but is not cited or made explicit in the theorem statement. |
| E-004 | Theorem "CFT Swampland Classification" + Corollary "Finite-Volume Mass Gap" | Moderate | External dependency | CFT scaling/finite-size gap claims use standard results without internal hypotheses. |

## Detailed findings

### [E-001] Correlation → mutual-information divergence is unproven
- Location: Theorem "Computational Necessity of Mass Gap" (sec-mass-gap), Step 3.
- Severity: Major
- Type: Proof gap / external dependency (secondary: None)
- Claim (paraphrase): Algebraic decay of correlations in a gapless theory implies divergent bulk mutual information, violating the Causal Information Bound.
- Why this is an error within the stated axioms: No internal lemma bounds mutual information by integrated correlations for the class of states under consideration. Without an explicit inequality (e.g., for Gaussian/quasi-free states or OS-reconstructable states), Step 3 relies on external results.
- Impact on downstream results: The key implication “Δ_KG=0 ⇒ Causal Stasis ⇒ Δ_KG>0” becomes conditional on an unstated external theorem. This weakens the mass-gap chain used later for YM claims.
- Fix guidance (step-by-step):
  1. Add a formal “Correlation–Information Inequality” lemma with explicit hypotheses (state class, regularity, decay rates).
  2. Restrict the theorem to that class of states and cite the lemma in Step 3.
  3. Re-run the mass-gap argument with the stated hypotheses.
- Required new assumptions/lemmas (if any): A lemma linking correlation decay to mutual-information growth (with explicit hypotheses).
- Proof sketch within stated axioms: For Gaussian/quasi-free states, bound mutual information by integrals of two-point functions; then show divergence for algebraic decay.
- Validation plan: Update all downstream mass-gap claims to cite the new lemma and its hypotheses.

### [E-002] KG screening gap is not yet the YM Hamiltonian mass gap
- Location: Theorem "Mass Gap from Screening" + Schrödinger reduction remark (sec-mass-gap).
- Severity: Major
- Type: Definition mismatch / scope restriction (secondary: Proof gap)
- Claim (paraphrase): The Helmholtz screening mass κ yields Δ_KG and then a Hamiltonian gap Δ_H used in YM claims.
- Why this is an error within the stated axioms: The screening gap is derived for a scalar Helmholtz/KG equation, while YM mass gap concerns the non-Abelian gauge sector Hamiltonian. A formal bridge lemma is missing that identifies or bounds the YM Hamiltonian gap using the KG screening gap.
- Impact on downstream results: YM mass-gap statements may overreach the established KG gap.
- Fix guidance (step-by-step):
  1. Add a bridge lemma tying the YM Hamiltonian spectrum to the screening gap (or restrict claims to the scalar sector only).
  2. Clarify which Hamiltonian is meant in each theorem.
  3. If no bridge is available, rename Δ_KG results as “screening gap” and avoid identifying them with YM Δ_H.
- Required new assumptions/lemmas (if any): A “KG→YM Hamiltonian Gap” lemma or explicit reduction theorem.
- Proof sketch within stated axioms: Use the gauge-invariant algebra and OS reconstruction to identify the YM Hamiltonian; then relate its lowest excitation to screened correlators under stated spectral hypotheses.
- Validation plan: Check all references to Δ_H in the YM section and ensure consistent definitions.

### [E-003] YM mass-gap theorem is conditional but not explicitly tied to constructive axioms
- Location: Theorem "Mass Gap Dichotomy for Yang-Mills" and Remark "Relation to the Clay Millennium Problem".
- Severity: Moderate
- Type: Scope restriction (secondary: Miswording)
- Claim (paraphrase): If YM describes physics, then Δ_KG>0, because physical theories are computable (ℓ_L>0).
- Why this is an error within the stated axioms: The computability premise is now a formal axiom (Axiom {prf:ref}`ax-constructive-finite-resolution`), but the YM theorem does not cite it or make the conditional dependence explicit in its statement.
- Impact on downstream results: The statement can be misread as unconditional; it should be explicitly conditional on the constructive axioms.
- Fix guidance (step-by-step):
  1. Add an explicit hypothesis in the YM theorem: “assume ax-constructive-finite-resolution (ℓ_L>0) and ax-constructive-positivity.”
  2. Move the computability premise into the theorem statement (not only in a remark).
  3. Retain the Clay disclaimer but anchor it to the explicit hypotheses.
- Required new assumptions/lemmas (if any): None new; just cite the constructive axioms already introduced in the standard model document.
- Proof sketch within stated axioms: Replace “physical ⇒ ℓ_L>0” with “ℓ_L>0 by ax-constructive-finite-resolution,” then apply thm-computational-necessity-mass-gap.
- Validation plan: Verify all YM mass-gap claims cite the constructive axiom set.

### [E-004] CFT scaling/finite-size gap rely on external results without hypotheses
- Location: Theorem "CFT Swampland Classification" and Corollary "Finite-Volume Mass Gap".
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): CFT correlator scaling implies mutual-information growth; finite-volume spectra yield ΔE ~ 1/L.
- Why this is an error within the stated axioms: These are standard CFT results but not encoded as internal hypotheses or lemmas; required conditions (operator dimensions, boundary conditions, unitarity) are not stated.
- Impact on downstream results: The swampland and finite-volume gap statements are heuristic unless the missing hypotheses are added.
- Fix guidance (step-by-step):
  1. Add lemmas for CFT scaling and finite-size spectral gaps, with explicit conditions.
  2. Restrict the statements to those conditions, or label them as heuristic remarks.
- Required new assumptions/lemmas (if any): CFT scaling lemma + finite-volume spectral-gap lemma.
- Proof sketch within stated axioms: Use OS/Wightman assumptions in the flat sector as explicit hypotheses, then derive the scaling and spectral gap under stated boundary conditions.
- Validation plan: Ensure the CFT claims cite the new lemmas or are moved to remarks.

## Scope restrictions and clarifications
- The mass-gap chain is rigorous only after adding a correlation→information lemma and a KG→YM gap bridge (or restricting claims to the scalar sector).
- YM mass-gap claims are conditional on the constructive axioms (finite resolution, positivity) and must be stated as such.

## Proposed edits (optional)
- Add explicit citations to ax-constructive-finite-resolution and ax-constructive-positivity in the YM mass-gap theorem.
- Rename Δ_KG results as “screening gap” unless a formal YM Hamiltonian bridge is added.

## Open questions
- Which class of states should the correlation→information lemma assume (Gaussian/quasi-free, KMS, OS-reconstructable)?
- Do you want YM mass-gap claims to remain conditional or to be strengthened by adding the KG→YM bridge lemma?
