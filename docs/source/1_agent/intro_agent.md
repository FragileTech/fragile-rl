---
title: "Fragile Mechanics"
subtitle: "On Geometry, Thermodynamics, and Bounded Intelligence"
author: "Guillem Duran-Ballester"
---
(sec-the-fragile-agent-bounded-rationality-control-and-information-geometry)=

# Fragile Mechanics
**On Geometry, Thermodynamics, and Bounded Intelligence**

by *Guillem Duran-Ballester and Sergio Hernández Cerezo*

:::{admonition} TL;DR — One-Page Summary
:class: tip dropdown

**What is this?** An engineering specification for building AI agents that remain stable, interpretable, and safe under partial observability and finite capacity. The framework unifies reinforcement learning, information geometry, and gauge theory into a coherent architecture with explicit runtime safety contracts.

**Core Loop (Fragile Agent Stack):**
- **State = $(K, z_n, z_{\mathrm{tex}})$**: Discrete macro-state $K$ (control-relevant symbols) + continuous nuisance $z_n$ (pose/basis) + texture $z_{\mathrm{tex}}$ (reconstruction residue). See {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`.
- **World Model / Belief Dynamics**: Prediction–update–projection on the latent bundle to evolve belief under partial observability. See {ref}`Section 12 <sec-belief-dynamics-prediction-update-projection>`.
- **Critic (Field Solver)**: Solves the screened Poisson equation $(-\Delta_G + \kappa^2)V = \rho_r$ for the **conservative component** of the reward 1-form; the non-conservative (curl) component persists as a connection/field in the dynamics. See {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`.
- **Policy**: Entropy-regularized control on the learned metric $G$ with geometric trust-region behavior. See {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`.
- **Universal Governor + Sieve**: Runtime monitors and recovery logic that enforce stability, capacity, and grounding constraints. See {ref}`Sections 3–6 <sec-diagnostics-stability-checks>` and {ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`.

**The Sieve (60+ Runtime Checks):**
A catalog of online diagnostics organized by failure mode ({ref}`Sections 3–6 <sec-diagnostics-stability-checks>`):
- **Stability**: Lyapunov descent, Lipschitz bounds, bifurcation detection
- **Capacity**: Codebook entropy, rate constraints, information closure
- **Grounding**: Input/output coupling, mixing time, saturation limits
- **Multi-Agent**: Game tensor bounds, Nash residual, symplectic bridge conservation, mean-field scalability, geometric locking
- **Ontology**: Texture predictability, fission readiness, thermodynamic hysteresis, hyperbolic coalescence

**Geometry & Field-Theoretic Layer:**
1. **Capacity-Constrained Metric Law** ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`): Interface limits induce curvature and consistency defects
2. **WFR Metric** ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`): Unifies continuous transport and discrete jumps in a single variational principle
3. **Holographic Interface** ({ref}`Sections 23–24 <sec-the-boundary-interface-symplectic-structure>`): Sensors = Dirichlet BC, Motors = Neumann BC, Reward = Source BC (boundary flux; scalar charge in the conservative case)
4. **Conformal Coupling** ({ref}`Section 24.4 <sec-geometric-back-reaction-the-conformal-coupling>`): High-curvature value regions acquire inertia via $\Omega = 1 + \alpha\|\nabla^2 V\|$
5. **Symmetry-Breaking Generation** ({ref}`Section 21.2 <sec-policy-control-field>`): Policy as a symmetry-breaking kick at the origin
6. **Geodesic Jump Dynamics** ({ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>`): BAOAB-integrated motion on the latent manifold

**Gauge-Theoretic Unification (Part VIII):**
- **Standard Model of Cognition**: The symmetry group $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$) emerges from three invariance principles: utility phase invariance, sensor-motor chirality, and feature basis freedom ({ref}`Section 34 <sec-standard-model-cognition>`)
- **Parameter Space Sieve**: Fundamental constants $(c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ are derived from constraint satisfaction, not free parameters ({ref}`Section 35 <sec-parameter-space-sieve>`)

**Gauge-Covariant Implementation (Part VIII, Chapters 04–06):**
- **Geometric Micro-Architecture**: Standard DL primitives (Linear+bias, ReLU) violate gauge invariance. Covariant replacements: `SpectralLinear` (bias-free, Lipschitz-constrained), `NormGatedGELU` (acts on bundle norms). See {ref}`Chapter 04 <sec-geometric-micro-architecture>`.
- **Covariant Cross-Attention**: World model as geodesic integrator. Wilson-line parallel transport, position-dependent temperature (= inverse conformal factor), Christoffel symbols from quadratic Query terms. Four attention heads implement Boris-BAOAB steps. See {ref}`Chapter 05 <sec-covariant-cross-attention-architecture>`.
- **Universal Geometric Network**: End-to-end synthesis achieving universal approximation (unconstrained encoder/decoder) while maintaining geometric consistency (soft equivariance via L1 regularization). Discovers texture zeros automatically. See {ref}`Chapter 06 <sec-end-to-end-architecture>`.

**Lorentzian Memory (Part VII, Chapter 09):**
- **Covariant Memory Attention**: Standard self-attention ignores causality. Lorentzian metric signature $(-,+,\ldots,+)$ for memory manifold. Light cone structure determines causal masks. Retarded potentials ensure finite information speed $c_{\text{info}}$. See {ref}`Chapter 09 <sec-covariant-memory-attention-architecture>`.

**Why "Fragile"?**
The agent is designed to *degrade gracefully* and *fail loudly*. When constraints are violated, the system halts or degrades predictably rather than silently misbehaving. Fragility is a feature: it makes failure modes observable and debuggable. See {ref}`FAQ D.6.1 <sec-appendix-d-the-fragile-branding>`.

**What's Novel (Categorized):**

*Architectural:*
- Discrete macro-register $K$ as the auditable control state (not just compression)
- The Sieve: 60+ explicit monitors connecting theory to implementation
- Holographic Interface: boundary-condition architecture unifying perception, action, reward
- Single notation tying representation, filtering, and control

*Geometric:*
- Capacity-constrained metric law linking interface limits to geometry
- WFR geometry for hybrid discrete/continuous belief states
- Critic as PDE solver with geometric back-reaction
- Policy as symmetry-breaking kick (pitchfork bifurcation)

*Field-Theoretic:*
- Causal Information Bound (Area Law) derived from first principles
- Multi-agent field theory with Game Tensor and mean-field scalability
- Non-local memory as self-interaction functional

*Gauge-Theoretic:*
- Standard Model of Cognition: $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$)
- Three gauge fields from three invariance principles
- Parameter Space Sieve: fundamental constants from constraint satisfaction

*Ontological:*
- Ontological expansion via topological fission when texture becomes predictable
- Thermodynamic hysteresis from Landauer bound
- Computational metabolism as operational limits

*Implementation (Theory → Neural Networks):*
- Gauge-covariant neural primitives: `SpectralLinear`, `NormGatedGELU`, `IsotropicBlock` replace standard DL ops
- Covariant Cross-Attention implementing Boris-BAOAB geodesic integration via attention heads
- Universal Geometric Network: unconstrained boundaries + geometrically structured interior
- Lorentzian memory attention with causal light-cone masks and retarded potentials

**What's Repackaged (by Domain):**

*Reinforcement Learning:* POMDP/belief control, entropy-regularized RL, world models, safe RL constraints

*Representation Learning:* VQ-VAE, InfoNCE/CPC, VICReg/Barlow collapse prevention

*Mathematics:* Optimal transport (WFR metric), symplectic geometry, Helmholtz/Poisson equations, bifurcation theory, stochastic differential geometry, BAOAB integrators

**Fragile Generalizes RL:**
Standard RL appears as a degenerate limit of the Fragile Agent when geometry is flattened ($G \to I$), capacity is unbounded ($|\mathcal{K}| \to \infty$), and the Sieve is disabled ($\Xi_{\text{crit}} \to \infty$). In that limit, the extra structure vanishes and the familiar RL equations are recovered ({ref}`Section 0.6 <sec-standard-rl-as-the-degenerate-limit>`).

**Extensions & System Laws:**
- **Supervised topology** and metric shaping for classification ({ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`)
- **Non-local memory** and **retrieval-augmented geometry** ({ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>`, {ref}`Section 28 <sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back>`)
- **Multi-agent field theory** and Nash stasis ({ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`)
- **Ontological expansion** via chart fission and semantic vacuum dynamics ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`)
- **Computational metabolism** and Landauer-bound deliberation ({ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`)
- **Causal discovery** and interventional geometry ({ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`)
- **Causal Information Bound** (area law for representational capacity) ({ref}`Section 33 <sec-causal-information-bound>`)
- **Standard Model of Cognition** and gauge-theoretic unification ({ref}`Section 34 <sec-standard-model-cognition>`)
- **Parameter Space Sieve** deriving fundamental constants ({ref}`Section 35 <sec-parameter-space-sieve>`)
- **Economic applications** via Partially Observable Markov Wealth ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`)
- **Proof of Useful Work** consensus replacing hash mining with gradient computation ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`)

**Proof of Useful Work (PoUW) — Key Security Properties:**
The economic layer introduces a novel consensus mechanism where the cryptographic puzzle is replaced by neural network training. Security against bad actors is guaranteed by multiple geometric and thermodynamic mechanisms:

| Attack Vector | Defense Mechanism | Guarantee |
|:--------------|:------------------|:----------|
| **Fake gradients** | Sieve constraints + spot-check verification | Detection probability $\geq 1 - (1-\epsilon)^k$ with $k$ samples |
| **51% attack** | Spontaneous Fission (Theorem {prf:ref}`thm-51-attack-rejection`) | Attackers isolated on high-friction shards that die of neglect |
| **Byzantine validators** | Minimum Friction BFT (Theorem {prf:ref}`thm-minimum-friction-bft`) | Tolerates $< 1/3$ adversarial validators |
| **Gradient poisoning** | Adversarial Geometric Damping (Theorem {prf:ref}`thm-adversarial-geometric-damping`) | Adversaries geometrically isolated, not just outvoted |
| **Flash loans / front-running** | Causal Theft Prevention (Theorem {prf:ref}`thm-causal-theft-prevention`) | CausalityViolationCheck rejects acausal transactions |
| **Coordinated deception** | Babel Limit Attack (Theorem {prf:ref}`thm-corruption-babel-detection`) | Information-theoretic bound on collusion effectiveness |

*Core insight:* Adversaries are not defeated by voting—they are **geometrically damped**. Their updates carry infinite inertia (Causal Stasis) and cannot influence the consensus trajectory. Security comes from coherence, not coercion.

**Quick Navigation:**
- *Want the math?* → {ref}`Sections 20–24 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`
- *Want implementation?* → {ref}`Sections 3–8 <sec-diagnostics-stability-checks>`
- *Want the architecture overview?* → {ref}`Architecture at a glance <sec-architecture-at-a-glance>`
- *Want multi-agent?* → {ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`
- *Want ontology expansion?* → {ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`
- *Want causality?* → {ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`
- *Want limits?* → {ref}`Section 33 <sec-causal-information-bound>`
- *Want gauge theory?* → {ref}`Section 34 <sec-standard-model-cognition>`
- *Want fundamental constants?* → {ref}`Section 35 <sec-parameter-space-sieve>`
- *Want blockchain/economics?* → {ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`
- *Want objections answered?* → {ref}`Appendix D <sec-appendix-d-frequently-asked-questions>`
- *Want proofs?* → {ref}`Appendix A <sec-appendix-a-full-derivations>`
- *Want loss functions?* → {ref}`Appendix F <sec-appendix-f-loss-terms-reference>`
- *Want gauge-covariant neural ops?* → {ref}`Part VIII Ch. 04 <sec-geometric-micro-architecture>`
- *Want the world model architecture?* → {ref}`Part VIII Ch. 05 <sec-covariant-cross-attention-architecture>`
- *Want end-to-end network design?* → {ref}`Part VIII Ch. 06 <sec-end-to-end-architecture>`
- *Want causal memory?* → {ref}`Part VII Ch. 09 <sec-covariant-memory-attention-architecture>`

**Researcher Bridge Index (Quick Links):**
| Researcher Bridge | Location |
| :--- | :--- |
| Bounded Rationality as a POMDP with Costs | {ref}`Link <rb-bounded-rationality>` |
| Markov Blanket = Observation/Action Interface | {ref}`Link <rb-markov-blanket>` |
| Actor-Critic + World Model, Typed | {ref}`Link <rb-actor-critic>` |
| Adversarial Immunity via Firewalling | {ref}`Link <rb-adversarial-immunity>` |
| Beyond Parameter-Space Adam | {ref}`Link <rb-beyond-adam>` |
| Safety as a Unit Test | {ref}`Link <rb-safety-unit-test>` |
| Barriers vs. Trust Regions | {ref}`Link <rb-barriers-trust-regions>` |
| RL Pathologies, Named and Localized | {ref}`Link <rb-rl-pathologies>` |
| Heuristic Fixes as Typed Surgeries | {ref}`Link <rb-heuristic-fixes>` |
| Engineering Tradeoffs, Made Explicit | {ref}`Link <rb-engineering-tradeoffs>` |
| Hyperbolic Hierarchy = Tree-Like Abstraction | {ref}`Link <rb-hyperbolic-hierarchy>` |
| Renormalization Group vs. ResNets | {ref}`Link <rb-renormalization-resnets>` |
| Jump Operators as Skill Switches | {ref}`Link <rb-jump-operators>` |
| Practical Substitutions for Idealized Laws | {ref}`Link <rb-practical-substitutions>` |
| World Models with Typed Latents | {ref}`Link <rb-world-models>` |
| Max-Entropy Exploration in Macro Space | {ref}`Link <rb-maxent-exploration>` |
| Bayes Filter with Safety Projection | {ref}`Link <rb-bayes-filter>` |
| Soft RL Equals Exploration Duality | {ref}`Link <rb-soft-rl-duality>` |
| KL Control as a Schrödinger Bridge | {ref}`Link <rb-kl-control-bridge>` |
| The Stable Learning Window | {ref}`Link <rb-stable-learning-window>` |
| Information Bottleneck Becomes Geometry | {ref}`Link <rb-info-bottleneck-geometry>` |
| Handling Distribution Shift | {ref}`Link <rb-distribution-shift>` |
| Diffusion-Style Generation with Policy Drift | {ref}`Link <rb-diffusion-generation>` |
| Continuous-Time Actor-Critic | {ref}`Link <rb-continuous-actor-critic>` |
| Observations and Actions as Boundary Conditions | {ref}`Link <rb-boundary-conditions>` |
| Value as a Smooth Field (PINN) | {ref}`Link <rb-non-conservative-value>` |
| Metric Learning for Classification | {ref}`Link <rb-metric-learning>` |
| Automated Homeostasis vs. Hyperparameter Tuning | {ref}`Link <rb-homeostasis>` |
| Experience Replay as a Potential Field | {ref}`Link <rb-experience-replay>` |
| Retrieval-Augmented Control | {ref}`Link <rb-retrieval-augmented>` |
| Opponents as Geometric Inertia | {ref}`Link <rb-opponents-inertia>` |
| Dynamic Architecture vs. Fixed Capacity | {ref}`Link <rb-dynamic-architecture>` |
| Pruning via Metabolic Efficiency | {ref}`Link <rb-pruning-efficiency>` |
| Principled "Thinking Fast and Slow" | {ref}`Link <rb-thinking-fast-slow>` |
| Curiosity as a Vector Field (Not a Scalar) | {ref}`Link <rb-curiosity-vector>` |
| The Sensor Bandwidth Ceiling | {ref}`Link <rb-sensor-bandwidth>` |
| The Fragile Agent Lexicon | {ref}`Link <rb-fragile-lexicon>` |
:::

(sec-how-to-read)=
## How to Read This Book

### Reading Modes

Use the toggle button at the top of the page to switch between **Full Mode** and **Expert Mode**:

**Full Mode** (First-time readers, researchers seeking complete understanding):
- Sequential reading from Part I through Part VIII
- Follow all cross-references and researcher bridges
- Engage with proofs in appendices

**Expert Mode** (Practitioners, implementers, those familiar with RL/geometry):
- Start with TL;DR and Book Map above
- Jump directly to relevant parts via Quick Navigation
- Use the Sieve (Part II) as implementation reference
- Skip intuitive explanation, focus on definitions and algorithms

### Modularity: Take Only What You Need

This framework is designed to be **modular**. Each part is written to be as self-contained as possible while extensively cross-referencing related material:

| If you want...              | Read...                  | Dependencies                        |
|-----------------------------|--------------------------|-------------------------------------|
| Safety contracts only       | Part II (The Sieve)      | Minimal (definitions in Part I)     |
| The geometry                | Parts V–VI               | Part I foundations                  |
| Multi-agent theory          | Part VIII Ch. 01–03      | Part VI boundary interface          |
| Implementation guidance     | Parts II–III             | Can standalone                      |
| Gauge-theoretic unification | Part VIII Ch. 01–03      | Parts V–VI helpful but not required |
| **Neural network implementation** | **Part VIII Ch. 04–06** | **Part VIII Ch. 01–03 (gauge theory)** |
| **Causal memory architecture** | **Part VII Ch. 09** | **Part VII Ch. 03 (memory retrieval)** |
| Economic applications       | Part IX                  | Parts I–II foundations              |

Cross-references are provided throughout so you can dive deeper when needed, but **you are not required to read linearly**. Each theorem and definition is self-contained with explicit dependencies stated.

### LLM-Assisted Exploration

A recommended approach for understanding this framework:

1. **Provide the markdown files** to an LLM (Claude, GPT-5.2, Gemini, etc.)
2. **Ask targeted questions** about specific concepts, theorems, or connections
3. **Request explanations** of how different parts connect
4. **Use the LLM to trace cross-references** and build intuition
5. **Generate examples** by asking the LLM to instantiate abstract concepts

The extensive cross-referencing, formal definitions, and explicit theorem statements make this document particularly amenable to LLM-assisted exploration. The structured format (definitions → theorems → proofs → connections) allows LLMs to provide accurate, grounded responses.

**Example queries:**
- "Explain how the Sieve Node 15 relates to the Coupling Window Theorem"
- "What is the relationship between the WFR metric and belief dynamics?"
- "How does the Parameter Space Sieve derive the information speed bound?"
- "Summarize the novel contributions in the gauge-theoretic formulation"

(sec-document-map)=
## Book Map

Table of contents:

**Part I: Foundations**
- {doc}`01_foundations/01_definitions`
- {doc}`01_foundations/02_control_loop`

**Part II: The Sieve (Runtime Safety)**
- {doc}`02_sieve/01_diagnostics`
- {doc}`02_sieve/02_limits_barriers`
- {doc}`02_sieve/03_failures_interventions`
- {doc}`02_sieve/04_approximations`

**Part III: Implementation Architecture**
- {doc}`03_architecture/00_architecture_at_a_glance`
- {doc}`03_architecture/01_compute_tiers`
- {doc}`03_architecture/02_disentangled_vae`
- {doc}`03_architecture/03_optimization`

**Part IV: Control and Belief**
- {doc}`04_control/01_exploration`
- {doc}`04_control/02_belief_dynamics`
- {doc}`04_control/03_coupling_window`

**Part V: Geometric Dynamics**
- {doc}`05_geometry/01_metric_law`
- {doc}`05_geometry/02_wfr_geometry`
- {doc}`05_geometry/03_holographic_gen`
- {doc}`05_geometry/04_equations_motion`

**Part VI: Holography and Field Theory**
- {doc}`06_fields/01_boundary_interface`
- {doc}`06_fields/02_reward_field`
- {doc}`06_fields/03_info_bound`

**Part VII: Cognitive Extensions**
- {doc}`07_cognition/01_supervised_topo`
- {doc}`07_cognition/02_governor`
- {doc}`07_cognition/03_memory_retrieval`
- {doc}`07_cognition/04_ontology`
- {doc}`07_cognition/05_metabolism`
- {doc}`07_cognition/06_causality`
- {doc}`07_cognition/07_metabolic_transducer`
- {doc}`07_cognition/08_intersubjective_metric`
- {doc}`07_cognition/09_retrieval_attention`

**Part VIII: Multi-Agent Gauge Theory & Implementation**
- {doc}`08_multiagent/01_gauge_theory`
- {doc}`08_multiagent/02_standard_model`
- {doc}`08_multiagent/03_parameter_sieve`
- {doc}`08_multiagent/04_dnn_blocks`
- {doc}`08_multiagent/05_architecture`
- {doc}`08_multiagent/06_full_net`

**Conclusion**
- {doc}`/conclusions`

**Appendices**
- {doc}`10_appendices/01_derivations`
- {doc}`10_appendices/02_parameters`
- {doc}`10_appendices/03_wfr_tensor`
- {doc}`10_appendices/04_faq`
- {doc}`10_appendices/05_proofs`
- {doc}`10_appendices/06_losses`
- {doc}`10_appendices/07_architecture`

(sec-positioning-connections-to-prior-work-differences-and-advantages)=
## Positioning: Connections to Prior Work, Differences, and Advantages

This document is a **synthesis and engineering specification** for building agents that remain stable, grounded, and debuggable under partial observability and finite capacity. Most mathematical ingredients are standard in **safe RL**, **robust control**, **information geometry**, **representation learning**, and **Bayesian filtering**. The contribution is to make the dependencies *explicit* and to provide a set of **online-auditable contracts** (Gate Nodes + Barriers) that connect representation, dynamics, value, and control.

(sec-main-advantages)=
### Main Advantages (Why This Framing Is Useful)

This framework introduces a unified nomenclature. While these terms may seem novel, they are strictly isomorphic to specific constructs in Differential Geometry and Information Theory. We use them because standard RL terminology is insufficient to describe the topological phase transitions of an agent under finite capacity.

1. **Online auditability.** Constraints are stated in quantities you can compute during training/inference (entropies, KLs, value gradients, stability inequalities), not only as "eventual performance". The agent operates as a Bounded-Rationality Controller ({prf:ref}`def-bounded-rationality-controller`) with explicit capacity limits.
2. **Explicit macro-state abstraction.** The discrete macro register $K_t$ makes sufficiency, capacity, and closure conditions well-typed and testable ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`, {ref}`Section 3 <sec-diagnostics-stability-checks>`, {ref}`Section 15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`).
3. **Predictive vs structured residual separation.** The "micro" channel is structured: we explicitly separate **structured nuisance** (pose/basis/disturbance coordinates that can be modeled and monitored) from **texture** (high-rate reconstruction detail). This prevents the world model and policy from silently depending on texture while still allowing nuisance to be represented and audited ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`, Axiom {prf:ref}`ax-bulk-boundary-decoupling`).
4. **Geometry-aware regulation.** A state-space sensitivity metric $G$ is used as a runtime trust-region / conditioning signal ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`, {ref}`Section 18.2 <sec-main-result>`), complementing standard natural-gradient methods {cite}`amari1998natural,schulman2015trpo,martens2015kfac`.
5. **Safety as a first-class interface contract.** "Safety" is not a single scalar constraint: it decomposes into 60 explicit checks (switching limits, capacity limits, saturation, grounding, mixing, multi-agent coupling, ontological stress, capacity horizon) with known compute cost ({ref}`Sections 3–6 <sec-diagnostics-stability-checks>`).
6. **Unified treatment of discrete and continuous dynamics.** The Wasserstein-Fisher-Rao (WFR) metric ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`) provides a single variational principle for belief evolution that seamlessly handles both continuous flow within charts and discrete jumps between charts ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`).
7. **Geometric field-theoretic formulation.** The critic is a PDE solver propagating the **conservative component** of the
   reward 1-form via the screened Poisson (Helmholtz) equation; the non-conservative component persists as a connection/curl term.
   The discount factor $\gamma$ determines the screening
   length $\ell = 1/\kappa$ where $\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln\gamma / \Delta t$
   (natural units: $\kappa = -\ln\gamma$) ({ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`).
8. **Holographic interface symmetry.** Sensors and motors are dual boundary conditions on the same symplectic manifold—
   perception imposes Dirichlet (position) BCs, action imposes Neumann (flux) BCs, and reward injects boundary flux
   (scalar charge in the conservative case) ({ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`,
   {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`).
9. **Multi-agent geometric coupling.** Strategic interaction is encoded in the Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`), which modulates the effective metric; adversarial coupling increases the effective metric tensor eigenvalues ({ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`).
10. **Principled ontology expansion.** When texture becomes predictable (violating Axiom {prf:ref}`ax-bulk-boundary-decoupling`), the framework prescribes chart fission via pitchfork bifurcation ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).
11. **Mean-field scalability.** Multi-agent interactions scale to $N \to \infty$ via the Mean-Field Metric Law ({prf:ref}`thm-mean-field-metric-law`), which proves that the effective metric converges to a deterministic Vlasov-geometry equation. Cooperation emerges metabolically via the Geometric Locking Principle ({prf:ref}`thm-geometric-locking-principle`).
12. **Thermodynamic grounding.** Constants like the hysteresis threshold $\epsilon_{\text{hysteresis}}$ are not free parameters but are derived from Landauer thermodynamics ({prf:ref}`thm-thermodynamic-hysteresis-bound`), ensuring ontological operations respect computational metabolism.
13. **Gauge-theoretic unification.** The three forces governing agent dynamics—value gradient transport, prediction-error correction, and feature binding—are derived as gauge fields from local invariance principles. The symmetry group $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$) emerges from cybernetic first principles ({ref}`Section 34 <sec-standard-model-cognition>`).
14. **Fundamental constants from constraint satisfaction.** The Agent Parameter Vector $\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ solves a constrained optimization problem. Sieve constraints (causal, holographic, metabolic, hierarchical, stiffness, temporal) define a feasible region; viable agents operate on its Pareto boundary ({ref}`Section 35 <sec-parameter-space-sieve>`).
15. **Metabolic transducer architecture.** Energy-information coupling is made explicit through the metabolic transducer, which converts computational resources into information updates while respecting Landauer bounds. This provides principled "thinking fast vs slow" phase transitions ({ref}`Part VII <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`).
16. **Intersubjective metric for shared meaning.** Multi-agent communication is grounded in a shared metric space where meaning emerges from geometric alignment between agents' latent representations, enabling principled analysis of language grounding and semantic drift ({ref}`Part VII <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`).
17. **Economic unification via POMW.** Partially Observable Markov Wealth (POMW) extends the framework to economic agents, treating wealth as a conserved quantity under metabolic constraints and unifying game-theoretic equilibria with the geometric field theory ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`).
18. **Proof of Useful Work consensus.** The framework enables a novel blockchain consensus mechanism where hash mining is replaced by gradient computation on a shared neural network. Security is guaranteed by the Landauer bound (thermodynamic hardness), the Sieve constraints (fake gradient detection), and geometric coherence (adversaries are damped, not outvoted). Key theorems: Cognitive Equivalency ({prf:ref}`thm-cognitive-equivalency`), Holographic Verification ({prf:ref}`thm-holographic-verification`), Verifier's Nash Equilibrium ({prf:ref}`thm-verifier-nash-equilibrium`), 51% Attack Rejection ({prf:ref}`thm-51-attack-rejection`), and Adversarial Geometric Damping ({prf:ref}`thm-adversarial-geometric-damping`). The result: energy expenditure produces intelligence instead of heat, and adversaries cannot buy consensus because they cannot buy geometric alignment ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`).

(sec-what-is-novel-here-vs-what-is-repackaging)=
### What Is Novel Here vs What Is Repackaging

**Novel Contributions (organized by category):**

*Architectural Contributions (Framework Design):*

1. **A discrete macro register used as the control-relevant state.** VQ-style discretization is treated as an enabler for audit-friendly information constraints (closure, capacity, window conditions) rather than merely a compression mechanism ({ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`Section 2.8 <sec-conditional-independence-and-sufficiency>`, {ref}`Section 15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`).
2. **The Sieve as an explicit catalog of monitors and limits.** Gate Nodes + Barriers are presented as a concrete interface between theory and implementation: "what to measure", "what to penalize", and "what to halt on" ({ref}`Sections 3–6 <sec-diagnostics-stability-checks>`).
3. **Coupling-window operationalization.** The grounding/mixing window is stated directly in measurable information rates (Theorem {prf:ref}`thm-information-stability-window-operational`), turning "stability/grounding" into an online diagnostic rather than a post-hoc story.
4. **A single notation tying representation, filtering, and control.** The same objects ($K_t$, $\bar P$, $V$, $G$, KL-control) appear consistently across the loop, reducing category errors between "learning" and "control" ({ref}`Section 2 <sec-the-control-loop-representation-and-control>`, {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>`).
5. **The Holographic Interface as boundary-condition architecture.** Perception (Dirichlet BC), action (Neumann BC), and
   reward (Source BC / boundary flux; scalar charge in the conservative case) are unified as boundary conditions on the
   latent manifold ({ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`, {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`).

*Geometric Contributions (Mathematical Framework):*

6. **WFR geometry for hybrid state spaces.** The Wasserstein-Fisher-Rao metric is the canonical geometry for agent belief states, seamlessly interpolating between continuous Wasserstein transport and discrete Fisher-Rao jumps via the teleportation length $\lambda$ ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`).
7. **Critic as Helmholtz solver with screening.** The value function is recast as a solution to the screened Poisson
   equation $-\Delta_G V + \kappa^2 V = \rho_r$ with the conservative component of the reward 1-form as sources and discount as screening mass
   $\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln\gamma / \Delta t$ (natural units: $\kappa = -\ln\gamma$)
   ({ref}`Section 24.2 <sec-the-bulk-potential-screened-poisson-equation>`).
8. **Conformal back-reaction of value on metric.** High-curvature value regions modulate the metric via $\Omega = 1 + \alpha\|\nabla^2 V\|$, creating a feedback loop where high-curvature regions have increased metric coefficients ({ref}`Section 24.4 <sec-geometric-back-reaction-the-conformal-coupling>`).
9. **Policy as symmetry-breaking kick.** Generation and control are unified as perturbations breaking $SO(D)$ angular symmetry at the origin; the policy-to-noise ratio determines whether direction selection is deterministic or stochastic ({ref}`Section 21.2 <sec-policy-control-field>`, Theorem {prf:ref}`thm-angular-symmetry-breaking`).
10. **Non-local memory as self-interaction.** Trajectory history induces a memory potential $\Psi_{\text{mem}}$ via heat-kernel convolution, creating conservative forces that stabilize learned attractors ({ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>`).

*Field-Theoretic Contributions (Multi-Agent & Bounds):*

11. **Multi-agent field theory.** Strategic interaction derives from coupled Helmholtz equations; the Game Tensor $\mathcal{G}_{ij}$ increases the effective metric eigenvalues under adversarial coupling, making Nash equilibrium a geometric fixed point ({ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`). The Mean-Field Metric Law ({prf:ref}`thm-mean-field-metric-law`) proves scalability to $N \to \infty$ agents. The Geometric Locking Principle ({prf:ref}`thm-geometric-locking-principle`) establishes that cooperative equilibria emerge from metabolic constraints.
12. **Causal Information Bound (Area Law).** The maximum representable information is bounded by interface area: $I_{\max} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ where the Holographic Coefficient $\nu_D = (D-1)\Omega_{D-1}/(8\pi)$ is dimension-dependent (recovering $\nu_2 = 1/4$ for $D=2$). Derived rigorously from the Capacity-Constrained Metric Law via generalized Gauss-Bonnet identity. Structural parallel to Bekenstein-Hawking with information-theoretic content ({ref}`Section 33 <sec-causal-information-bound>`, {ref}`Appendix A.6 <sec-appendix-a-area-law>`).
13. **Causal Isometry and Safe Retrieval.** The Causal Isometry Theorem ({prf:ref}`thm-causal-isometry`) proves that Interventionally Closed representations in different modalities induce isometric metrics, enabling principled cross-modal transfer. The Safe Retrieval Bandwidth Theorem ({prf:ref}`thm-safe-retrieval-bandwidth`) bounds retrieval injection to prevent saturation of the holographic interface.

*Ontological Contributions (Dynamic Architecture):*

14. **Ontological expansion via fission.** When texture becomes predictable (ontological stress $\Xi > \Xi_{\text{crit}}$), the framework prescribes chart bifurcation, expanding the agent's categorical structure ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`). The hysteresis constant is derived from Landauer thermodynamics ({prf:ref}`thm-thermodynamic-hysteresis-bound`). Chart coalescence uses the Fréchet mean on hyperbolic space ({prf:ref}`def-hyperbolic-frechet-coalescence`), and the Fission Inhibition Corollary ({prf:ref}`thm-fission-inhibition`) guarantees hierarchical stability.

*Gauge-Theoretic Contributions (Unification):*

15. **Standard Model of Cognition.** The gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$) is derived from three invariance principles: utility phase invariance ($U(1)_Y$), sensor-motor chirality ($SU(r)_L$), and feature basis freedom ($SU(N_f)_C$). The belief state is a chiral spinor; ontological symmetry breaking via Higgs mechanism ({ref}`Section 34 <sec-standard-model-cognition>`).
16. **Parameter Space Sieve.** Operational constants are derived from constraint intersection: causal buffer (Theorem {prf:ref}`thm-speed-window`), holographic bound (Theorem {prf:ref}`thm-holographic-bound`), Landauer constraint (Theorem {prf:ref}`thm-landauer-constraint`), asymptotic freedom with IR confinement (Corollary {prf:ref}`cor-coupling-window`), stiffness bounds (Theorem {prf:ref}`thm-stiffness-bounds`), and temporal screening (Theorem {prf:ref}`thm-discount-window`) ({ref}`Section 35 <sec-parameter-space-sieve>`).
17. **Isomorphism Dictionary.** Complete correspondence: $c_{\text{info}} \leftrightarrow c$, $\sigma \leftrightarrow \hbar$, $\ell_L \leftrightarrow \ell_P$, $T_c \leftrightarrow k_B T$, $g_s \leftrightarrow \alpha_s$, $\gamma \leftrightarrow$ cosmological screening. The mapping is structural ({ref}`Section 34.6 <sec-isomorphism-dictionary>`).

*Extended Cognitive Architecture:*

18. **Metabolic transducer.** Energy-information coupling is formalized through the metabolic transducer architecture, which converts computational resources into information updates while respecting Landauer bounds. Provides principled "thinking fast vs slow" phase transitions with explicit switching criteria ({ref}`Part VII <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`).
19. **Intersubjective metric.** Multi-agent semantic alignment is grounded in a shared metric space where meaning emerges from geometric alignment between agents' latent representations. Enables principled analysis of language grounding, semantic drift, and communication bandwidth ({ref}`Part VII <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`).

*Economic Extensions:*

20. **Partially Observable Markov Wealth (POMW).** Economic agents are unified with the geometric field theory by treating wealth as a conserved quantity under metabolic constraints. Game-theoretic equilibria emerge as geometric fixed points; resource allocation follows from holographic interface capacity ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`).
21. **Proof of Useful Work (PoUW) consensus.** A novel blockchain consensus mechanism where the cryptographic puzzle is replaced by gradient computation on a shared neural network. Novel contributions include: (a) Cognitive Equivalency Theorem proving gradients have the same thermodynamic hardness as hashes; (b) Holographic Verification reducing verification cost from $O(N)$ to $O(\sqrt{N})$; (c) Verifier's Nash Equilibrium proving honest computation is strictly dominant; (d) Minimum Friction BFT achieving Byzantine tolerance via geometric coherence; (e) 51% Attack Rejection via Spontaneous Fission; (f) Adversarial Geometric Damping isolating malicious actors through metric friction rather than voting ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`).

**Repackaging (directly inherited ingredients, organized by domain):**

*Reinforcement Learning:*
- **POMDP/belief-control viewpoint:** partial observability, belief updates, and control on internal state {cite}`kaelbling1998planning,rabiner1989tutorial`.
- **Entropy-regularized / KL-regularized control:** soft Bellman objectives, KL-control, and exponential-family optimal policies {cite}`todorov2009efficient,kappen2005path,haarnoja2018soft`.
- **World-model based RL:** learning latent dynamics for planning/control (e.g., Dreamer-like latent rollouts) {cite}`hafner2019dreamer,ha2018worldmodels`.
- **Safe RL and constrained optimization:** Lyapunov-style constraints and constrained policy updates {cite}`chow2018lyapunov,berkenkamp2017safe,altman1999constrained,achiam2017constrained`.

*Representation Learning:*
- **Representation learning primitives:** VQ-VAE, InfoNCE/CPC, VICReg/Barlow-type collapse prevention {cite}`oord2017vqvae,oord2018cpc,bardes2022vicreg,zbontar2021barlow`.

*Mathematics (Geometry, PDEs, Dynamics):*
- **Optimal transport / WFR metric:** The Wasserstein-Fisher-Rao metric and unbalanced optimal transport machinery {cite}`chizat2018unbalanced,liero2018optimal`.
- **Symplectic geometry and Legendre transforms:** Classical mechanics textbook material applied to the boundary interface.
- **Helmholtz / screened Poisson equation:** Standard PDE theory (electrostatics, Yukawa potential); the mathematical form is textbook.
- **Bifurcation theory:** Pitchfork bifurcations and symmetry breaking are standard dynamical systems.
- **Stochastic differential geometry:** Geodesic SDEs, Onsager-Machlup functionals, and Langevin dynamics on manifolds {cite}`onsager1953fluctuations`.
- **Molecular dynamics integrators:** The BAOAB splitting scheme is from computational chemistry {cite}`leimkuhler2016computation`.

*Physics Inspiration (Mathematical Structure Only):*
- **Holographic principle (AdS/CFT structural correspondence):** The bulk/boundary structural parallel shares mathematical structure with physics holography, but the actual derivation is original: the Area Law is proven from first principles in {ref}`Appendix A.6 <sec-appendix-a-area-law>` using the Capacity-Constrained Metric Law and generalized Gauss-Bonnet identity—no physics is imported, only the mathematical structure.

(sec-comparison-snapshot)=
### Comparison Snapshot (Where This Differs in Practice)

| Area                                | Typical baseline                             | Fragile Agent difference                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model-free RL**                   | optimize return; debugging via reward curves | adds explicit monitors and stop/penalty mechanisms tied to identifiable failure modes ({ref}`Sections 3–6 <sec-diagnostics-stability-checks>`)                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **World models**                    | continuous latent rollouts; implicit "state" | enforces a discrete macro state with closure and capacity checks; the "micro" channel is split into **structured nuisance** $z_n$ (auditable and optionally control-relevant) and **texture** $z_{\mathrm{tex}}$ (reconstruction-only, excluded from closure/control) ({ref}`Sections 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`2.8 <sec-conditional-independence-and-sufficiency>`, {ref}`9 <sec-the-disentangled-variational-architecture-hierarchical-latent-separation>`, {ref}`15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`) |
| **Safe RL / CMDPs**                 | few scalar constraints (expected cost)       | uses a *vector* of auditable constraints (grounding, mixing, saturation, switching, stiffness) with compute-cost accounting ({ref}`Sections 3–8 <sec-diagnostics-stability-checks>`)                                                                                                                                                                                                                                                                                                                                                                             |
| **Info bottleneck RL**              | compression via an information penalty       | makes the bottleneck operational via $\log\lvert\mathcal{K}\rvert$, $H(K)$, $I(X;K)$ and closure, not only via a single Lagrange term ({ref}`Sections 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`3 <sec-diagnostics-stability-checks>`, {ref}`15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`)                                                                                                                                                                                                                                        |
| **Natural gradient / trust region** | parameter-space Fisher metric                | emphasizes state-space sensitivity $G$ as a runtime regulator of updates and checks ({ref}`Sections 2.5–2.6 <sec-second-order-sensitivity-value-defines-a-local-metric>`, {ref}`9.10 <sec-differential-geometry-view-curvature-as-conditioning>`)                                                                                                                                                                                                                                                                                                                |
| **Diffusion models**                | reverse SDE from noise to data               | forward SDE from origin to boundary via holographic generation; policy steers the entropy-driven expansion ({ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`)                                                                                                                                                                                                                                                                                                                                                                         |
| **AdS/CFT-inspired architectures**  | bulk/boundary duality as loose metaphor      | explicit boundary-condition architecture: Dirichlet (sensors), Neumann (motors), Source (reward flux; scalar in conservative case) mapped to neural components ({ref}`Sections 23–24 <sec-the-boundary-interface-symplectic-structure>`)                                                                                                                                                                                                                                                                                                                         |
| **Critic / value function**         | MLP fitting $V(z)$ via TD error              | PDE-solver propagating the conservative component of the reward 1-form via screened Poisson equation; Helmholtz regularization and conformal coupling to metric ({ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`)                                                                                                                                                                                                                                                                                                                        |
| **Multi-agent RL**                  | independent or centralized learners          | coupled Helmholtz equations with Game Tensor $\mathcal{G}_{ij}$ modulating effective metric; Nash equilibrium as geometric stasis; mean-field scalability to $N \to \infty$ ({ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`)                                                                                                                                                                                                                                                                                                                        |
| **Ontology learning**               | implicit via representation                  | explicit fission criterion: when texture becomes predictable ($\Xi > \Xi_{\text{crit}}$), chart bifurcation expands categories; hysteresis thermodynamically calibrated ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`)                                                                                                                                                                                                                                                                                              |
| **Gauge structure**                 | implicit or absent                           | explicit gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$) with three derived gauge fields; covariant derivative ensures coordinate-invariant dynamics ({ref}`Section 34 <sec-standard-model-cognition>`)                                                                                                                                                                                                                                                                                                                |
| **Hyperparameter tuning**           | grid search, Bayesian optimization           | Parameter Space Sieve derives operational constants from constraint satisfaction; feasible region defined by causal, holographic, metabolic, and stiffness bounds ({ref}`Section 35 <sec-parameter-space-sieve>`)                                                                                                                                                                                                                                                                                                                                                |
| **Economic agents**                 | separate game-theoretic models               | Partially Observable Markov Wealth (POMW) unifies economic dynamics with agent geometry; wealth as conserved quantity under metabolic constraints ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`)                                                                                                                                                                                                                                                                                                                                  |
| **Blockchain consensus**            | Proof of Work (useless hash mining)          | Proof of Useful Work: gradients replace hashes; energy produces intelligence not heat; adversaries geometrically damped via metric friction; 51% attacks trigger Spontaneous Fission ({ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`)                                                                                                                                                                                                                                                                                               |

**Reading guide (connections by section).**
- Representation + abstraction: {ref}`Sections 2.2b <sec-the-shutter-as-a-vq-vae>`, {ref}`2.8 <sec-conditional-independence-and-sufficiency>`, {ref}`9.7–9.9 <sec-literature-connections>`
- Safety monitors and limits: {ref}`Sections 3–6 <sec-diagnostics-stability-checks>`
- Filtering + projection (belief evolution): {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>`
- Entropy-regularized control + exploration: {ref}`Sections 11–14 <sec-intrinsic-motivation-maximum-entropy-exploration>`
- Coupling window and capacity constraints: {ref}`Sections 15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`, {ref}`17 <sec-summary-unified-information-theoretic-control-view>`, and {ref}`18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`
- Hybrid state-space geometry (WFR): {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`
- Holographic generation and symmetry breaking: {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`
- Equations of motion and integrators: {ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>`
- Holographic interface and boundary conditions: {ref}`Sections 23–24 <sec-the-boundary-interface-symplectic-structure>`
- Supervised topology and classification: {ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`
- Meta-stability and the Universal Governor: {ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`
- Non-local memory and self-interaction: {ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>`
- Retrieval-augmented geometry: {ref}`Section 28 <sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back>`
- Multi-agent field theory: {ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`
- Ontological expansion: {ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`
- Computational metabolism and Landauer bound: {ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`
- Causal discovery and interventional geometry: {ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`
- Causal information bound and representational limits: {ref}`Section 33 <sec-causal-information-bound>`
- Gauge-theoretic unification (Standard Model of Cognition): {ref}`Section 34 <sec-standard-model-cognition>`
- Fundamental constants from constraints (Parameter Space Sieve): {ref}`Section 35 <sec-parameter-space-sieve>`
- Metabolic transducer and energy-information coupling: {ref}`Part VII <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`
- Intersubjective metric and shared meaning: {ref}`Part VII <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`
- Economic applications (Partially Observable Markov Wealth): {ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`
- Frequently asked questions (rigorous objections and responses): {ref}`Appendix D <sec-appendix-d-frequently-asked-questions>`

(sec-for-skeptical-readers)=
### For Skeptical Readers

This framework makes strong claims about structure, geometry, and safety. A rigorous reader should ask: *Is this over-engineered? Does the math actually buy anything? What breaks?*

**{ref}`Appendix D <sec-appendix-d-frequently-asked-questions>`** addresses forty such objections head-on, organized by theme:
- **Computational complexity:** Can you actually invert those matrices? Run those PDEs? {ref}`Appendix D.1 <sec-appendix-d-computational-complexity-scalability>`
- **Optimization dynamics:** Do all these loss terms fight each other into deadlock? {ref}`Appendix D.2 <sec-appendix-d-optimization-dynamics-convergence>`
- **Information theory:** Is "texture" just a way to hide inconvenient signals? {ref}`Appendix D.3 <sec-appendix-d-information-theory-representation>`
- **Physics correspondences:** Are the thermodynamic isomorphisms rigorous or merely suggestive? {ref}`Appendix D.4 <sec-appendix-d-physics-geometry-isomorphisms>`
- **Control & safety:** What stops the agent from gaming the Sieve by doing nothing? {ref}`Appendix D.5 <sec-appendix-d-control-theory-system-safety>`
- **Gauge theory:** Is the Standard Model analogy more than metaphor? Does the symmetry group actually constrain dynamics? {ref}`Appendix D.4 <sec-appendix-d-physics-geometry-isomorphisms>`
- **Parameter derivation:** Are the "fundamental constants" genuinely derived or just relabeled hyperparameters? {ref}`Appendix D.1 <sec-appendix-d-computational-complexity-scalability>`

Each question is stated in its strongest form, then answered with specific mechanisms and section references. If the answers are unconvincing, the framework deserves skepticism.

(sec-document-layers)=
### Document Map

The document is organized into nine parts plus appendices:

| Layer              | Parts    | Purpose                                                                                                                        |
|--------------------|----------|--------------------------------------------------------------------------------------------------------------------------------|
| **Foundations**    | I–II     | Positioning, definitions, control loop architecture, the Sieve (60+ runtime diagnostics)                                       |
| **Implementation** | III      | Computational tiers, hyperbolic geometry, TopoEncoder architecture                                                             |
| **Control Theory** | IV       | Exploration, belief dynamics, capacity constraints                                                                             |
| **Geometry**       | V–VI     | WFR metric, holographic generation, boundary interface, reward field, information bounds                                       |
| **Cognition**      | VII      | Supervised topology, meta-stability, memory, retrieval, ontology, metabolism, causality                                        |
| **Gauge Theory**   | VIII     | Multi-agent field theory, Standard Model of Cognition, Parameter Space Sieve                                                   |
| **Applications**   | IX       | Economic applications (Partially Observable Markov Wealth)                                                                     |
| **Appendices**     | A–F      | Derivations, units, WFR tensor, FAQ (40 objections), proofs, loss terms reference                                              |

**Detailed Section Guide:**

**Part I: Foundations (Sections 0–2)**
- **{ref}`Section 0 <sec-positioning-connections-to-prior-work-differences-and-advantages>`**: How this framework relates to prior work; what's novel vs repackaged
- **{ref}`Section 1 <sec-introduction-the-agent-as-a-bounded-rationality-controller>`**: Core definitions—the agent as a bounded-rationality controller operating on a Markov blanket ({prf:ref}`def-boundary-markov-blanket`) interface
- **{ref}`Section 2 <sec-the-control-loop-representation-and-control>`**: The control loop—objective, architecture, state manifolds, metric hierarchy

**Part II: The Sieve ({ref}`Sections 3–6 <sec-diagnostics-stability-checks>`)**
- **{ref}`Section 3 <sec-diagnostics-stability-checks>`**: The 60 diagnostic nodes—what to measure, when to warn, when to halt
- **{ref}`Section 4 <sec-limits-barriers>`**: Barriers—hard limits that cannot be crossed (BarrierLock, BarrierGap, BarrierSat)
- **{ref}`Section 5 <sec-failure-modes>`**: Observed failure modes with symptoms and root causes
- **{ref}`Section 6 <sec-interventions>`**: Interventions—what the Governor does when checks fail

**Part III: Implementation ({ref}`Sections 7–9 <sec-computational-considerations>`)**
- **{ref}`Section 7 <sec-computational-considerations>`**: Computational tiers from fast inference to slow audit; hyperbolic geometry; stacked TopoEncoders
- **{ref}`Section 8 <sec-infeasible-implementation-replacements>`**: When exact solutions are infeasible—practical replacements
- **{ref}`Section 9 <sec-the-disentangled-variational-architecture-hierarchical-latent-separation>`**: The TopoEncoder architecture—typed latent separation

**Part IV: Control Theory ({ref}`Sections 11–18 <sec-intrinsic-motivation-maximum-entropy-exploration>`)**
- **{ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>`**: Maximum-entropy exploration and intrinsic motivation
- **{ref}`Section 12 <sec-belief-dynamics-prediction-update-projection>`**: Belief dynamics—prediction, update, projection
- **{ref}`Sections 13–14 <sec-correspondence-table-filtering-control-template>`**: Filtering/control correspondence; duality of exploration and soft optimality
- **{ref}`Sections 15–16 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`**: Coupling window theorem—the information-stability threshold
- **{ref}`Section 17 <sec-summary-unified-information-theoretic-control-view>`**: Summary of the unified information-theoretic view
- **{ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`**: Capacity-constrained metric law—how interface limits determine geometry

**Part V: Geometric Dynamics ({ref}`Sections 19–22 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`)**
- **{ref}`Section 19 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`**: Capacity-constrained metric law—how interface limits determine geometry
- **{ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`**: Wasserstein-Fisher-Rao geometry—transport + reaction on hybrid state spaces
- **{ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`**: Radial generation—entropic drift from origin to boundary; policy as symmetry-breaking kick
- **{ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>`**: Equations of motion—geodesic jump-diffusion with BAOAB integrator

**Part VI: Holography and Field Theory ({ref}`Sections 23–24 <sec-the-boundary-interface-symplectic-structure>`)**
- **{ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`**: The boundary interface—symplectic structure; sensors/motors as dual boundary conditions
- **{ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`**: The reward field—reward 1-form with scalar potential in the conservative case; critic as Helmholtz solver; conformal coupling
- **{ref}`Section 24.5 <sec-causal-information-bound>`**: The Causal Information Bound—area law for representational capacity

**Part VII: Cognitive Extensions ({ref}`Sections 25–33 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`)**
- **{ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`**: Supervised topology—using class labels to shape the metric and attractor basins
- **{ref}`Section 26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`**: Meta-stability—the Universal Governor as a homeostatic controller over the Sieve
- **{ref}`Section 27 <sec-section-non-local-memory-as-self-interaction-functional>`**: Non-local memory—self-interaction functional from trajectory history
- **{ref}`Section 28 <sec-section-hyperbolic-active-retrieval-geodesic-search-and-semantic-pull-back>`**: Retrieval-augmented geometry—external knowledge as spatial mass injection
- **{ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`**: Multi-agent field theory—coupled WFR dynamics, Game Tensor, Nash equilibrium
- **{ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`**: Ontological expansion—topological fission, semantic vacuum, chart bifurcation
- **{ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>`**: Computational metabolism—Landauer bound, deliberation dynamics, fast/slow phase transition
- **{ref}`Section 32 <sec-causal-discovery-interventional-geometry-and-the-singularity-of-action>`**: Causal discovery—interventional geometry, curiosity force, causal enclosure
- **{ref}`Section 33 <sec-causal-information-bound>`**: The Causal Information Bound—area law for representational capacity; Causal Stasis
- **{ref}`Section 33.5 <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`**: Metabolic transducer—energy-information coupling, Landauer-bounded updates, fast/slow phase transitions
- **{ref}`Section 33.6 <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`**: Intersubjective metric—shared semantic space, language grounding, multi-agent meaning alignment

**Part VIII: Multi-Agent Gauge Theory ({ref}`Sections 34–35 <sec-standard-model-cognition>`)**
- **{ref}`Section 34 <sec-standard-model-cognition>`**: The Standard Model of Cognition—gauge-theoretic formulation; $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$ (minimal $r=2$); belief spinors; ontological symmetry breaking
- **{ref}`Section 35 <sec-parameter-space-sieve>`**: The Parameter Space Sieve—deriving fundamental constants from constraint satisfaction; causal, holographic, metabolic, coupling, stiffness, and screening bounds

**Part IX: Economics**
- **{ref}`Section 36 <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`**: Partially Observable Markov Wealth (POMW)—economic applications, resource allocation, game-theoretic wealth dynamics

**Appendices**
- **{ref}`Appendix A <sec-appendix-a-full-derivations>`**: Full derivations of the capacity-constrained curvature functional and the Area Law coefficient (A.6)
- **{ref}`Appendix B <sec-appendix-b-units-parameters-and-coefficients>`**: Units, parameters, and coefficient table (audit reference)
- **{ref}`Appendix C <sec-appendix-c-wfr-stress-energy-tensor>`**: WFR stress-energy tensor derivation
- **{ref}`Appendix D <sec-appendix-d-frequently-asked-questions>`**: FAQ—40 rigorous objections and responses
- **{ref}`Appendix E <sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws>`**: Rigorous proof sketches for ontological and metabolic laws
- **{ref}`Appendix F <sec-appendix-f-loss-terms-reference>`**: Loss Terms Reference—37 loss functions with curved formulations and flat limits

(sec-standard-rl-as-the-degenerate-limit)=
### Standard RL as the Degenerate Limit

:::{important}
This framework is not an alternative to Reinforcement Learning. It is the **General Theory** of which standard RL is a **degenerate special case**—standard RL is recovered when geometric and capacity constraints are removed.
:::

Standard RL emerges from the Fragile Agent when three degeneracy conditions are imposed:

:::{prf:theorem} The RL Degeneracy Theorem
:label: thm-rl-degeneracy

Standard Reinforcement Learning is recovered from the Fragile Agent framework under the joint limit:

$$
\text{Standard RL} = \lim_{\substack{G \to I \\ |\mathcal{K}| \to \infty \\ \Xi_{\text{crit}} \to \infty}} \text{Fragile Agent}
$$
where:
1. **Flat Geometry** ($G \to I$): The state-space metric becomes Euclidean, eliminating coordinate-invariant updates
2. **Infinite Capacity** ($|\mathcal{K}| \to \infty$): No information bottleneck, continuous state space without quantization
3. **No Safety Constraints** ($\Xi_{\text{crit}} \to \infty$): The Sieve is disabled, all actions permitted

*Proof.* Each of the 37 Connection boxes below demonstrates a specific reduction. The composite limit follows from the independence of the five degeneracy conditions. $\square$
:::

**Table 0.6.1 (The 37 Reductions).** Each row shows a standard algorithm (degenerate case), the corresponding Fragile Agent construct (general law), and the mathematical limit that recovers the standard algorithm. Includes RL algorithms, economic models, and blockchain consensus.

| #  | Standard RL (Degenerate)                  | Fragile Agent (General Law)                                               | Limit                                                                   | Section                                                                                                |
|----|-------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1  | REINFORCE / Vanilla PG                    | Natural Gradient $\delta z = G^{-1}\nabla_z \mathcal{L}$                  | $G \to I$                                                               | {ref}`2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`                                 |
| 2  | Euclidean SGD                             | Geodesic Flow on $(\mathcal{Z}, G)$                                       | Flat metric                                                             | {ref}`2.4 <sec-a-geometry-regularized-objective>`                                                      |
| 3  | TRPO/PPO Trust Region                     | State-space metric $G(z)$ vs parameter-space $\mathcal{F}(\theta)$        | Conflate manifolds                                                      | {ref}`2.6 <sec-the-metric-hierarchy-fixing-the-category-error>`                                        |
| 4  | Bellman Equation                          | Screened Poisson PDE $(-\Delta_G + \kappa^2)V = \rho_r$ (exact component) | Discretize lattice                                                      | {ref}`24.2 <sec-the-bulk-potential-screened-poisson-equation>`                                         |
| 5  | Tabular Q-Learning                        | VQ-VAE macro-register $K$                                                 | $\lvert\mathcal{K}\rvert = \lvert\mathcal{S}\rvert$, encoder = identity | {ref}`2.2b <sec-the-shutter-as-a-vq-vae>`                                                              |
| 6  | Options Framework                         | Split state $(K, z_n, z_{\text{tex}})$                                    | Codebook read-only                                                      | {ref}`2.2b <sec-the-shutter-as-a-vq-vae>`                                                              |
| 7  | Dreamer/World Models                      | Symplectic integrators on $(\mathcal{Z}, \omega)$                         | Generic RNN (non-conservative)                                          | {ref}`23.7 <sec-implementation-the-holographicinterface-module>`                                       |
| 8  | Constrained MDPs                          | Topological Sieve (hard firewall)                                         | Soft $\lambda$-penalty                                                  | {ref}`3 <sec-diagnostics-stability-checks>`                                                            |
| 9  | CQL (Offline RL)                          | Coupling Window (Node 13)                                                 | Soft Q-value penalty                                                    | {ref}`15 <sec-implementation-note-entropy-regularized-optimal-transport-bridge>`                       |
| 10 | Soft Actor-Critic                         | MaxEnt control on $(\mathcal{Z}, G)$                                      | Entropy in action space only                                            | {ref}`14 <sec-duality-of-exploration-and-soft-optimality>`                                             |
| 11 | RND (Curiosity)                           | Ontological Stress $\Xi$                                                  | Feed $\Xi$ to reward, never fission                                     | {ref}`30.2 <sec-ontological-stress>`                                                                   |
| 12 | Fixed Network Arch                        | Pitchfork Fission when $\Xi > \Xi_{\text{crit}}$                          | $\Xi_{\text{crit}} \to \infty$                                          | {ref}`30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`                      |
| 13 | EWC (Continual)                           | Atlas of Charts (topological isolation)                                   | Single chart + quadratic penalty                                        | {ref}`30.7 <sec-summary-the-lifecycle-of-an-ontology>`                                                 |
| 14 | $\max \mathbb{E}[R]$                      | Free Energy $\mathcal{F} = E - T_c S$ with Landauer                       | $T_c \to 0$ (ignore compute)                                            | {ref}`31.1 <sec-the-energetics-of-information-updates>`                                                |
| 15 | UCB1 (Bandits)                            | Thermodynamic Value of Information                                        | Single-state manifold                                                   | {ref}`31.3 <sec-optimal-deliberation-the-fast-slow-law>`                                               |
| 16 | Entropy Maximization                      | Causal Information Potential $\Psi_{\text{causal}}$                       | Remove causal graph                                                     | {ref}`32.2 <sec-the-causal-information-potential>`                                                     |
| 17 | Independent PPO (IPPO)                    | Sheaf sections with shared topology                                       | Disconnect sheaf                                                        | {ref}`29 <sec-symplectic-multi-agent-field-theory>`                                                    |
| 18 | Lyapunov (implicit)                       | Neural Lyapunov Constraint $\dot{V} \le -\lambda V$                       | Remove stability check                                                  | {ref}`2.3 <sec-the-bridge-rl-as-lyapunov-constrained-control>`                                         |
| 19 | POMDP Belief Update                       | Filtering + Sieve Projection                                              | Remove projections                                                      | {ref}`12 <sec-belief-dynamics-prediction-update-projection>`                                           |
| 20 | Experience Replay                         | Memory Potential via Heat Kernel                                          | Uniform sampling                                                        | {ref}`27 <sec-section-non-local-memory-as-self-interaction-functional>`                                |
| 21 | Imitation Learning                        | Supervised Topology + Class Potentials                                    | $V_{\text{base}} \to 0$                                                 | {ref}`25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`                        |
| 22 | KL-Regularized Policies                   | Path-Space Exponential Tilt                                               | Single-step KL                                                          | {ref}`14 <sec-duality-of-exploration-and-soft-optimality>`                                             |
| 23 | MAML / Meta-RL                            | Universal Governor + Training Lyapunov                                    | Ignore Sieve                                                            | {ref}`26 <sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller>`              |
| 24 | Diffusion Policies                        | Radial Generation + Symmetry Breaking                                     | Reverse SDE, $G \to I$                                                  | {ref}`21 <sec-radial-generation-entropic-drift-and-policy-control>`                                    |
| 25 | Information Bottleneck                    | Capacity-Constrained Metric Law                                           | Scalar rate                                                             | {ref}`18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`                         |
| 26 | Distributional RL (C51)                   | WFR Geometry on $\mathcal{M}^+(\mathcal{Z})$                              | Value dist. only                                                        | {ref}`20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`               |
| 27 | Auxiliary Tasks                           | Conformal Back-Reaction                                                   | $\alpha_{\text{conf}} \to 0$                                            | {ref}`24.4 <sec-geometric-back-reaction-the-conformal-coupling>`                                       |
| 28 | CURL/DrQ/SPR                              | VICReg per Chart                                                          | Contrastive only                                                        | {ref}`7.7 <sec-vicreg-geometric-collapse-prevention>`                                                  |
| 29 | Contrastive RL (CPC)                      | InfoNCE Anchoring                                                         | No macro-micro                                                          | {ref}`8 <sec-geomcheck-efficient-infonce>`                                                             |
| 30 | Temporal Discount $\gamma$                | Screening Length $\ell = 1/\kappa$                                        | Temporal only                                                           | {ref}`24.2 <sec-the-bulk-potential-screened-poisson-equation>`                                         |
| 31 | Mean-Field Games (MFG)                    | Mean-Field Metric Law + Geometric Locking                                 | Finite $N$                                                              | {ref}`29.8 <sec-mean-field-metric-law>`                                                                |
| 32 | Scalar reward shaping (conservative case) | Gauge-covariant value transport                                           | Abelian limit ($SU(2), SU(N_f) \to 1$)                                  | {ref}`34.1 <sec-gauge-principle-derivation>`                                                           |
| 33 | Hand-tuned hyperparameters                | Parameter Space Sieve (Constrained Optimization)                          | Remove constraints ($\mathcal{S} \to 0$)                                | {ref}`35 <sec-parameter-space-sieve>`                                                                  |
| 34 | System 1/2 heuristics                     | Metabolic Transducer (Landauer-bounded switching)                         | Remove energy accounting                                                | {ref}`Part VII <sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine>`                      |
| 35 | Multi-agent language alignment            | Intersubjective Metric (geometric semantic grounding)                     | Independent representations                                             | {ref}`Part VII <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>` |
| 36 | Economic game theory                      | POMW (wealth as conserved geometric quantity)                             | Decouple economics from geometry                                        | {ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`                            |
| 37 | Bitcoin/Proof of Work                     | Proof of Useful Work (gradient mining + geometric consensus)              | Remove Sieve, use useless hashes, vote-based BFT                        | {ref}`Part IX <sec-proof-of-useful-work-cognitive-metabolism-as-consensus>`                            |

**The Five Degeneracy Classes:**

1. **Geometric Degeneracy** (Rows 1–4, 18, 20, 24, 27, 30–31, 35–36): Setting $G \to I$ flattens the manifold. Natural gradients become vanilla gradients; the Helmholtz PDE becomes the Bellman recursion; diffusion policies lose hyperbolic structure; screening length reduces to a temporal discount parameter; mean-field limits reduce to finite-agent games; intersubjective metric loses geometric grounding; economic dynamics decouple from geometry.

2. **Capacity Degeneracy** (Rows 5–6, 11–13, 25, 28–29): Setting $|\mathcal{K}| \to \infty$ removes the information bottleneck. VQ-VAE becomes continuous representations; information bottleneck becomes scalar rate; VICReg loses atlas structure; InfoNCE loses macro-micro separation.

3. **Safety Degeneracy** (Rows 8–9, 12, 19, 23): Setting $\Xi_{\text{crit}} \to \infty$ disables the Sieve. Hard topological constraints become soft penalties; POMDP belief updates lose Sieve projections; MAML ignores constraint structure.

4. **Metabolic Degeneracy** (Rows 14, 34): Setting $T_c \to 0$ removes energy accounting. Pure return maximization ignores computational cost; System 1/2 switching becomes heuristic rather than thermodynamically derived.

5. **Consensus Degeneracy** (Row 37): Replacing gradient computation with hash computation, disabling the Sieve, and using vote-based rather than geometric BFT yields standard Proof of Work. Security shifts from geometric coherence (adversaries damped by metric friction) to purely economic cost (energy expenditure on useless computation). The key insight: PoUW uses the *same* thermodynamic cost but produces *useful* information (model improvement) rather than waste heat.

:::{admonition} Reading the Connection Boxes
:class: note
:name: reading-connection-boxes
Throughout this document, `:::{admonition} Connection to RL #N` admonition boxes (with `:class: note`) mark each reduction. Each box contains:
- **The General Law**: The Fragile Agent formulation
- **The Degenerate Limit**: The mathematical limit operation
- **The Special Case**: The resulting standard RL algorithm
- **What the generalization offers**: Why the general form is preferable
:::

**Conclusion.** Standard RL and traditional blockchain consensus are recovered from the Fragile Agent under five degeneracy conditions: flat geometry, infinite capacity, disabled Sieve, zero metabolic cost, and useless consensus computation. The 37 reductions in Table 0.6.1 demonstrate that each standard RL algorithm—and even Bitcoin's Proof of Work—corresponds to a specific limit of the unified framework. The generalizations are not optional decorations; they restore coordinate invariance, impose hard safety guarantees, provide principled answers to questions (like "when should I stop thinking?" and "how do we achieve Byzantine fault tolerance?") that standard approaches leave to heuristics, and extend naturally to economic, multi-agent semantic, and distributed consensus settings where security comes from geometric coherence rather than wasted computation.



(sec-introduction-the-agent-as-a-bounded-rationality-controller)=
